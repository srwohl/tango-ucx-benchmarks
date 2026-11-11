#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run tango_ucx_perftest over a list of frame sizes, with both roles on this host (local), or
the publisher on another host and the subscriber here (remote).

With --reference, or a --tier, ucx_perftest runs at each size too. A point whose rings do not
fit this host's memory is left out, and said to be.

With --hdf5 the frames of an image stack are published, read from the file before each publish,
beside generated frames of the same size: the difference is what reading the file costs.

With --placement each point runs with its threads placed in each of these ways, and the
ucx_perftest references with theirs placed alike:
    unpinned    where the scheduler puts them: what a job gets by default
    one-l3      four cores that share one L3 cache
    spread-l3   the publisher's two threads and the subscriber's two on different L3 caches
On a CPU of several L3 caches, as an EPYC of several chiplets, these can differ several times.

remote runs the publisher on --peer and the subscriber here, or on --subscriber, so the sweep
can be run from a workstation. It reaches the hosts through remote.py: one password, one login
each. They need a deployment (deploy.sh puts it in /tmp/USER/tango-ucx), and each
--transports variant runs with the same UCX settings on both:
    default         UCX's own choice: every port it finds
    rdma:DEVICES    UCX_NET_DEVICES=DEVICES, as rdma:mlx5_2:1; join several with +
    tcp:INTERFACE   UCX_TLS=tcp on that interface, as tcp:ens4f0np0
"""
import argparse
import json
import os
from pathlib import Path
import random
import resource
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import remote as remote_login  # noqa: E402
UNITS = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}


def size(text):
    text = text.strip()
    scale = UNITS.get(text[-1:].upper(), 1)
    return int(text[:-1] if scale > 1 else text) * scale


def sizes(text):
    """4K,1M or 1K..64M: a range is every power of two from its first size to its last."""
    found = []
    for part in text.split(","):
        first, dots, last = part.partition("..")
        at, end = size(first), size(last if dots else first)
        while at <= end:
            found.append(at)
            at *= 2
    return found


# What a sweep covers when nothing else is said; an option given replaces the tier's.
PLACEMENTS = ("unpinned", "one-l3", "spread-l3")
TIERS = {
    "quick": dict(sizes="4K,64K,1M,16M", depth=[32], runs=1, bytes="256M",
                  placement=list(PLACEMENTS)),
    "baseline": dict(sizes="1K,4K,16K,64K,256K,1M,4M,16M", depth=[2, 32], runs=3, bytes="2G",
                     placement=list(PLACEMENTS)),
    "standard": dict(sizes="1K..64M", depth=[2, 32], runs=3, bytes="2G",
                     placement=list(PLACEMENTS)),
    "full": dict(sizes="1K..2G", depth=[2, 8, 32, 128], runs=5, bytes="4G",
                 placement=list(PLACEMENTS)),
}


def placements(text):
    found = [name.strip() for name in text.split(",") if name.strip()]
    unknown = set(found) - set(PLACEMENTS)
    if unknown or not found:
        raise argparse.ArgumentTypeError(f"placements are {', '.join(PLACEMENTS)}")
    return list(dict.fromkeys(found))


def l3_caches():
    """The allowed physical cores, one CPU of each, grouped by the L3 cache they share, in
    CPU order. A CPU whose cache is not described is grouped by its package."""
    groups, seen = {}, set()
    for cpu in sorted(os.sched_getaffinity(0)):
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        try:
            core = ((base / "topology/physical_package_id").read_text().strip(),
                    (base / "topology/core_id").read_text().strip())
        except OSError:
            core = ("?", str(cpu))
        if core in seen:
            continue  # an SMT sibling of a core already chosen
        seen.add(core)
        cache = f"package {core[0]}"
        for index in sorted(base.glob("cache/index*")):
            try:
                if (index / "level").read_text().strip() == "3":
                    cache = (index / "shared_cpu_list").read_text().strip()
                    break
            except OSError:
                continue
        groups.setdefault(cache, []).append(cpu)
    return list(groups.values())


def placement_cpus(name, caches):
    """The CPUs of the publisher loop, producer, subscriber loop and reader for a placement,
    and why there are none when the host cannot have it."""
    if name == "unpinned":
        return None, None
    if name == "one-l3":
        group = next((group for group in caches if len(group) >= 4), None)
        return (group[:4], None) if group else (None, "no L3 cache has four allowed cores")
    # spread-l3: the caches halved between the publisher's side and the subscriber's, and the
    # two threads of a side on two caches where the side has two.
    half = len(caches) // 2
    if not half:
        return None, "one L3 cache: nothing to spread across"
    cpus = []
    for side in (caches[:half], caches[half:]):
        second = (side[1][0] if len(side) > 1 else side[0][1] if len(side[0]) > 1 else None)
        if second is None:
            return None, "too few allowed cores for two on each side"
        cpus += [side[0][0], second]
    return cpus, None


def pinned_to(cpu):
    """A preexec_fn that keeps a child to one CPU, or None for none."""
    return None if cpu is None else (lambda: os.sched_setaffinity(0, {cpu}))


def available_bytes():
    with open("/proc/meminfo") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return None


def left_out(frame_bytes, depth, args, read_ahead=0):
    """Why a point is not run, or nothing. Both roles' memory is this host's."""
    if frame_bytes > 2 << 30:
        return "a frame is at most 2 GiB"
    # A publisher has at least three slots, and one more for each frame read ahead; ucx_perftest
    # has a buffer on each side.
    need = frame_bytes * (2 if depth is None else max(depth, 3) + read_ahead + depth)
    free = getattr(args, "memory_available", None) or available_bytes()
    if free is not None and need > free * args.memory_share:
        return (f"needs {need / (1 << 30):.1f} GiB, and {args.memory_share:.0%} of the "
                f"{free / (1 << 30):.1f} GiB available is the most a point takes")
    return None


def revision(checkout):
    try:
        return subprocess.run(["git", "-C", str(checkout), "describe", "--always", "--dirty"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def describe_host(args):
    """What a reader of the rates needs to know of the host and of what was measured."""
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return "unknown"

    with open("/proc/cpuinfo") as cpuinfo:
        cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo
                    if line.startswith("model name")), "unknown")
    locked = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
    return {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname().split(".")[0],
        "kernel": os.uname().release,
        "cpu": cpu,
        "cpus": os.cpu_count(),
        "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "load": round(os.getloadavg()[0], 2),
        "memory_available": available_bytes(),
        "memory_locked_limit": None if locked == resource.RLIM_INFINITY else locked,
        "tango_ucx": revision(os.environ.get("TANGO_UCX_SOURCE", ROOT.parent / "tango-ucx")),
        "benchmarks": revision(ROOT),
        "tier": args.tier,
        "options": " ".join(sys.argv[1:]),
        # What both roles inherit and UCX or the library reads.
        "settings": " ".join(f"{name}={value}" for name, value in sorted(os.environ.items())
                             if name.startswith(("UCX_", "TANGO_UCX_"))
                             and name != "TANGO_UCX_SOURCE") or "none",
    }


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Publisher:
    """tango_ucx_perf_publisher as a -nodb device server, stopped when the block ends. Other
    hosts reach it at `address`, which is this host's name or address on their network."""

    def __init__(self, build, env=None, address="127.0.0.1"):
        self.address = address
        # A port found free can be taken before the server binds it: then another is tried.
        for _ in range(5):
            self.port = free_port()
            self.device = f"{address}:{self.port}/bench/ucx/1#dbase=no"
            self.log = tempfile.TemporaryFile(mode="w+")
            self.process = subprocess.Popen(
                [str(build / "tango_ucx_perf_publisher"), "bench", "-nodb", "-dlist",
                 "bench/ucx/1", "-ORBendPoint", f"giop:tcp:{address}:{self.port}"],
                stdout=self.log, stderr=subprocess.STDOUT, env=env)
            if self.listening():
                return
            self.log.close()
        raise RuntimeError("the publisher did not start")

    def listening(self):
        for _ in range(300):
            if self.process.poll() is not None:
                return False
            with socket.socket() as probe:
                if probe.connect_ex((self.address, self.port)) == 0:
                    return True
            time.sleep(0.1)
        self.process.kill()
        return False

    def said(self):
        self.log.seek(0)
        return self.log.read()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.process.send_signal(signal.SIGTERM)  # lets the device close its publisher
        try:
            self.process.wait(10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.log.close()


def frames_for(frame_bytes, args):
    """Each point moves the same number of bytes, within a floor and a ceiling of frames."""
    # A quarter of the floor above 16 MiB: a run of 1 GiB frames would otherwise take minutes.
    floor = args.min_frames if frame_bytes <= 16 << 20 else max(32, args.min_frames // 4)
    return max(floor, min(args.max_frames, args.bytes // frame_bytes))


def hdf5_options(args):
    return ["--hdf5", args.hdf5] + (["--hdf5-dataset", args.hdf5_dataset]
                                    if args.hdf5_dataset else [])


def describe_file(build, args):
    """What the publisher finds in the --hdf5 file: its frame size is the sweep's one size."""
    with Publisher(build) as publisher:
        done = subprocess.run([str(build / "tango_ucx_perftest"), publisher.device,
                               *hdf5_options(args), "--describe", "--json"],
                              capture_output=True, text=True)
    if done.returncode:
        said = (done.stderr.strip() or done.stdout.strip()).splitlines()
        raise SystemExit(f"bench: the publisher cannot replay {args.hdf5}: "
                         + (said[-1] if said else "no answer"))
    return json.loads(done.stdout.strip().splitlines()[-1])


def read_ahead_of(program, args):
    return args.read_ahead if program == HDF5 else 0


def run_point(build, device, frame_bytes, depth, args, hdf5=False, cpus=None, env=None,
              runner=None):
    """One run of tango_ucx_perftest: here, or through runner(command), which runs it
    elsewhere and returns what it said."""
    size = (hdf5_options(args) + ["--read-ahead", str(args.read_ahead)] if hdf5
            else ["-s", str(frame_bytes)])
    command = [str(build / "tango_ucx_perftest"), device, "-t", "bw", *size,
               "-n", str(frames_for(frame_bytes, args)), "-d", str(depth),
               "-b", str(args.batch), "--json"]
    if args.verify or args.corrupt_every:
        command.append("--verify")
    if args.corrupt_every:
        command += ["--corrupt-every", str(args.corrupt_every)]
    if getattr(args, 'rate', 0):
        command += ['--rate', str(args.rate)]
    if cpus:
        command += ['--thread-cpus', ','.join(map(str, cpus))]
    done = (runner(command) if runner else
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                           env=env))
    lines = done.stdout.strip().splitlines()
    if not lines or not lines[-1].startswith("{"):
        said = done.stderr.strip().splitlines()
        return {"ok": False, "frame_bytes": frame_bytes,
                "failure": said[-1] if said else "no result"}
    return json.loads(lines[-1])


# ucx_perftest at the same frame size, as the measure of what UCX can do on this link. It sends
# from one buffer, so its working set is that of a ring of the least depth.
#   as UCX chooses:     an active message a frame, and nothing asked of the endpoint.
#   as tango-ucx asks:  the frame header of an array stream, registered memory, peer error
#                       handling and the wakeup feature, with the two settings the library makes.
REFERENCES = {
    "ucx_perftest": ([], {}),
    "ucx_perftest -e": (["-H", "40", "-z", "-e", "-I"],
                        {"UCX_MM_ERROR_HANDLING": "y", "UCX_KEEPALIVE_INTERVAL": "2s"}),
}


def protocols(report, frame_bytes, memory_type=None):
    """What UCX_PROTO_INFO names for this size, optionally filtering source memory type."""
    def edge(text):
        if text == "inf":
            return float("inf")
        scale = UNITS.get(text[-1], 1)
        return int(text[:-1] if scale > 1 else text) * scale

    # Different worker threads/processes can print tables concurrently. Keep each log
    # channel's headers and ranges together instead of combining unrelated table rows.
    channels = {}
    for line in report.splitlines():
        prefix = re.match(r'\[[^\]]+\]\s+\[([^\]]+)\]', line)
        channels.setdefault(prefix[1] if prefix else '', []).append(line)
    found = []
    for lines in channels.values():
        titles, in_rows = [], False
        for line in lines:
            if "|" not in line:
                continue
            cells = [cell.strip() for cell in line[line.index("|"):].strip().strip("|").split("|")]
            if len(cells) == 1:  # a table's two title lines: its configuration, then the operation
                titles = [cells[0]] if in_rows or len(titles) >= 2 else titles + [cells[0]]
                in_rows = False
                continue
            if len(cells) != 3 or cells[0] == "Range" or len(titles) < 2:
                continue
            in_rows = True
            to_a_peer = "intra-node" in titles[0] or "inter-node" in titles[0]
            a_frame = titles[1].startswith("active message by ucp_am_send*") and "reply" not in titles[1]
            if memory_type is not None:
                a_frame = a_frame and f"from {memory_type}" in titles[1]
            low, _, high = cells[0].partition("..")
            if to_a_peer and a_frame and high and edge(low) <= frame_bytes <= edge(high):
                name = f"{cells[1].removeprefix('(?)').strip()} ({cells[2]})"
                if name not in found:
                    found.append(name)
    return "; or ".join(found)


def run_reference(name, frame_bytes, args, cpus=None):
    # Placed as tango-ucx is: the sender on the publisher loop's CPU, the receiver on the
    # subscriber loop's.
    flags, settings = REFERENCES[name]
    frames = frames_for(frame_bytes, args)
    port = free_port()
    command = ["ucx_perftest", "-t", "ucp_am_bw", "-s", str(frame_bytes), "-n", str(frames),
               "-w", str(max(10, frames // 10)), "-p", str(port)] + flags
    env = {**os.environ, **settings}
    point = {"program": name, "ok": False, "frame_bytes": frame_bytes, "frames": frames,
             "command": " ".join(command), "settings": settings}
    try:
        server = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL,
                                  preexec_fn=pinned_to(cpus[2] if cpus else None))
    except FileNotFoundError:
        return {**point, "failure": "ucx_perftest is not on the path"}
    try:
        # The server listens a moment after it starts: the subscriber's side is tried again.
        for _ in range(50):
            time.sleep(0.2)
            done = subprocess.run(command[:1] + ["127.0.0.1"] + command[1:],
                                  env={**env, "UCX_PROTO_INFO": "y"}, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300,
                                  preexec_fn=pinned_to(cpus[0] if cpus else None))
            final = [line.split() for line in done.stdout.splitlines()
                     if line.lstrip().startswith("Final:")]
            if final or server.poll() is not None:
                break
        if not final:
            return {**point, "failure": done.stdout.strip().splitlines()[-1:] or "no result"}
        # Final: iterations, three overheads, then bandwidth and message rate, average and
        # overall. Its MB is 2^20 bytes.
        rate = float(final[-1][6])
        return {**point, "ok": True, "frames_per_second": float(final[-1][8]),
                "MiB_per_second": rate, "Gb_per_second": rate * (1 << 20) * 8 / 1e9,
                "protocol": protocols(done.stdout, frame_bytes)}
    finally:
        if server.poll() is None:
            try:
                server.wait(10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


HDF5 = "tango-ucx hdf5"  # frames read from the --hdf5 file; "tango-ucx" are generated


def rows_of(args):
    """The rows of one frame size: the references, then tango-ucx at each depth, then the
    file's frames at each depth."""
    rows = [(name, None) for name in (REFERENCES if args.reference else ())]
    rows += [("tango-ucx", depth) for depth in args.depth]
    return rows + [(HDF5, depth) for depth in args.depth if args.hdf5]




def placement_line(name, cpus, unavailable, peer=None):
    if unavailable:
        return f"# placement {name}: left out: {unavailable}"
    if cpus is None:
        return f"# placement {name}: threads where the scheduler puts them"
    if peer:
        return (f"# placement {name}: CPUs {cpus[0]},{cpus[1]} on {peer} for the publisher loop "
                f"and producer, {cpus[2]},{cpus[3]} here for the subscriber loop and reader")
    return (f"# placement {name}: CPUs {','.join(map(str, cpus))} for the publisher loop, "
            "producer, subscriber loop and reader")


def print_host(host):
    print(f"# {host['date']}  {host['host']}  Linux {host['kernel']}")
    print(f"# {host['cpu']}, {host['cpus']} CPUs, governor {host['governor']}, "
          f"load {host['load']:.2f}")
    print(f"# tango-ucx {host['tango_ucx']}  tango-ucx-benchmarks {host['benchmarks']}")
    if host["governor"] not in ("performance", "unknown"):
        print(f"bench: the CPU governor is {host['governor']}, not performance: rates will be "
              "low and uneven", file=sys.stderr)
    if host["load"] > 1.0:
        print(f"bench: the host is busy, load {host['load']:.2f}: rates will vary from run to "
              "run", file=sys.stderr)


def dry_run(args, plan):
    points = runs = 0
    for name, cpus, unavailable, line in plan:
        print(line)
        if unavailable:
            continue
        for frame_bytes in args.sizes:
            for program, depth in rows_of(args):
                why = left_out(frame_bytes, depth, args, read_ahead_of(program, args))
                print(f"{frame_bytes:>10}  {program:<16} {'' if depth is None else depth:>5} "
                      f"{frames_for(frame_bytes, args):>8} frames  "
                      + (f"left out: {why}" if why else f"{args.runs} runs"))
                points += not why
                runs += 0 if why else args.runs
    return points, runs


TABLE_HEAD = (f"{'#bytes':>10}  {'program':<16} {'depth':>5} {'#frames':>8} {'frames/s':>12} "
              f"{'MiB/s':>10} {'best':>9} {'worst':>9} {'of raw':>7}  path")


def measure(args, out, publisher, plan, point_of, tag=None):
    """Each placement of the plan, each frame size, each row: its runs, their median and the
    table line. point_of(program, frame_bytes, depth, cpus) measures one run. Returns the points
    that failed, or None when the publisher ended. tag goes on every line written."""
    # One run says little on a machine that is doing anything else: the median of the runs
    # is the figure, and the best and the worst are beside it. "of raw" is tango-ucx against
    # ucx_perftest as tango-ucx asks, placed alike.
    tag = tag or {}
    failed = 0
    for name, cpus, unavailable, line in plan:
        print(line)
        if unavailable:
            if out:
                out.write(json.dumps({**tag, "placement": name, "skipped": unavailable}) + "\n")
            continue
        for frame_bytes in args.sizes:
            raw = None
            for program, depth in rows_of(args):
                why = left_out(frame_bytes, depth, args, read_ahead_of(program, args))
                if why:
                    print(f"{frame_bytes:>10}  {program:<16} "
                          f"{'' if depth is None else depth:>5} left out: {why}")
                    if out:
                        out.write(json.dumps({**tag, "program": program, "frame_bytes": frame_bytes,
                                              "depth_asked": depth, "placement": name,
                                              "skipped": why}) + "\n")
                    continue
                runs = []
                for run in range(args.runs):
                    if publisher.process.poll() is not None:
                        print("the publisher ended:\n" + publisher.said(), file=sys.stderr)
                        return None
                    point = point_of(program, frame_bytes, depth, cpus)
                    point.update(tag)
                    point["depth_asked"] = depth
                    point["run"] = run
                    point["placement"], point["cpus"] = name, cpus
                    if out:
                        out.write(json.dumps(point) + "\n")
                        out.flush()
                    runs.append(point)
                bad = [p for p in runs if not p.get("ok")]
                if bad:
                    p = bad[0]
                    why = (f"{p['outcome']}, {p['mismatched']} mismatched of "
                           f"{p['mismatched_expected']} expected" if "outcome" in p
                           else p.get("failure", ""))
                    print(f"{frame_bytes:>10}  {program:<16} FAILED in {len(bad)} of {len(runs)} "
                          f"runs: {why}")
                    failed += 1
                    continue
                rates = sorted(p["MiB_per_second"] for p in runs)
                median = statistics.median(rates)
                middle = min(runs, key=lambda p: abs(p["MiB_per_second"] - median))
                if program == "ucx_perftest -e":
                    raw = median
                share = f"{100 * median / raw:.0f}%" if raw and depth is not None else ""
                print(f"{frame_bytes:>10}  {program:<16} "
                      f"{'' if depth is None else middle['depth']['subscription']:>5} "
                      f"{middle['frames']:>8} {middle['frames_per_second']:>12.1f} "
                      f"{median:>10.1f} {rates[-1]:>9.1f} {rates[0]:>9.1f} {share:>7}  "
                      f"{middle['protocol'] if depth is None else middle['lanes']}")
    return failed


def open_out(args, host):
    if not args.out:
        return None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = open(args.out, "a")
    out.write(json.dumps({"meta": host}) + "\n")
    return out


def local(args):
    build = Path(args.build).resolve()
    host = describe_host(args)
    print_host(host)
    if args.hdf5:
        source = host["source"] = describe_file(build, args)
        args.sizes = [source["frame_bytes"]]
        filters = f", filters {', '.join(map(str, source['filters']))}" if source["filters"] else ""
        rate = f", acquired at {source['nominal_rate']:g} frames/s" if source["nominal_rate"] else ""
        print(f"# {source['file']}:{source['dataset']}  {source['element']} "
              f"{'x'.join(map(str, source['shape']))}, {source['frames']} frames, "
              f"{source['layout']}{filters}{rate}")
    host["l3_caches"] = l3_caches()
    plan = []
    for name in args.placement:
        cpus, unavailable = placement_cpus(name, host["l3_caches"])
        plan.append((name, cpus, unavailable, placement_line(name, cpus, unavailable)))
    if args.dry_run:
        points, runs = dry_run(args, plan)
        print(f"# {points} points, {runs} runs")
        return 0
    out = open_out(args, host)
    with Publisher(build) as publisher:
        print(TABLE_HEAD)

        def point_of(program, frame_bytes, depth, cpus):
            if depth is None:
                return run_reference(program, frame_bytes, args, cpus)
            point = run_point(build, publisher.device, frame_bytes, depth, args,
                              hdf5=program == HDF5, cpus=cpus)
            point["program"] = program
            return point

        failed = measure(args, out, publisher, plan, point_of)
    return 1 if failed is None or failed else 0


# --- remote: the publisher on another host -------------------------------------------------

def transports(text):
    """default, rdma:DEVICES and tcp:INTERFACE, with commas between; + joins devices."""
    found = []
    for part in (part.strip() for part in text.split(",")):
        kind, _, where = part.partition(":")
        where = where.replace("+", ",")
        if kind == "default" and not where:
            found.append((part, {}))
        elif kind == "rdma" and where:
            found.append((part, {"UCX_NET_DEVICES": where}))
        elif kind == "tcp" and where:
            found.append((part, {"UCX_TLS": "tcp", "UCX_NET_DEVICES": where}))
        else:
            raise argparse.ArgumentTypeError(
                f"{part}: a transport is default, rdma:DEVICES or tcp:INTERFACE")
    return found


def side_cpus(name, caches):
    """The CPUs of one host's two threads for a placement, and why there are none."""
    if name == "unpinned":
        return None, None
    if name == "one-l3":
        group = next((group for group in caches if len(group) >= 2), None)
        return (group[:2], None) if group else (None, "no L3 cache has two allowed cores")
    if len(caches) < 2:
        return None, "one L3 cache: nothing to spread across"
    return [caches[0][0], caches[1][0]], None


def forwarded():
    """The settings of this environment that both roles must share."""
    return {name: value for name, value in os.environ.items()
            if name.startswith(("UCX_", "TANGO_UCX_")) and name != "TANGO_UCX_SOURCE"}


def words(settings):
    return [f"{name}={value}" for name, value in sorted(settings.items())]


# Run by sh, a command in the background that is killed when sh's input closes: when the sweep
# ends or the login is lost. Killing the local ssh alone would leave it running there. A
# background job's input is /dev/null unless redirected, so the watcher reads it through
# descriptor 3, and the command reads none of it.
WATCH = ('exec 3<&0; "$@" 3<&- & p=$!; '
         '{ cat <&3 >/dev/null; kill -TERM $p 2>/dev/null; } >/dev/null 2>&1 & '
         'exec 3<&-; wait $p')

# Waits there until a server answers: every address of its name, as one can listen on IPv6
# alone. From the host that connects, which this one need not be able to reach.
ANSWERS = ("import socket, sys, time\n"
           "for _ in range(int(sys.argv[3])):\n"
           "    try:\n"
           "        socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1).close()\n"
           "        sys.exit(0)\n"
           "    except OSError:\n"
           "        time.sleep(0.1)\n"
           "sys.exit(1)\n")


class Side:
    """Where a role runs: this host, or another through the run's login, and the deployment
    there. A command gets its settings with env and its CPU with taskset."""

    def __init__(self, session, host, root):
        self.session, self.host = session, host
        if host is None:  # this host: this checkout and this environment
            self.name = socket.gethostname().split(".")[0]
            self.root, self.python = ROOT, Path(sys.executable)
            self.perftest = Path(shutil.which("ucx_perftest") or "ucx_perftest")
        else:
            self.name = host.rsplit("@", 1)[-1]
            self.root = Path(root)
            self.python = self.root / ".pixi/envs/default/bin/python"
            self.perftest = self.root / ".pixi/envs/default/bin/ucx_perftest"

    def argv(self, command, settings=None, slot=0, cpu=None, watched=False):
        line = ["env", *words(settings or {})]
        if watched:
            line += ["sh", "-c", WATCH, "sh"]
        if cpu is not None:
            line += ["taskset", "-c", str(cpu)]
        line += [str(word) for word in command]
        if self.host is None:
            return line
        return self.session.argv(self.host, f"cd {shlex.quote(str(self.root))} && "
                                 + shlex.join(line), slot)

    def run(self, command, settings=None, slot=0, cpu=None, timeout=600):
        return subprocess.run(self.argv(command, settings, slot, cpu), text=True,
                              stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout,
                              cwd=self.root if self.host is None else None)

    def start(self, command, settings=None, slot=0, cpu=None, log=None):
        """A watched command: stop() closes its input, and so ends it there."""
        return subprocess.Popen(self.argv(command, settings, slot, cpu, watched=True),
                                stdin=subprocess.PIPE, stdout=log or subprocess.DEVNULL,
                                stderr=subprocess.STDOUT)

    def free_port(self):
        """A port free on this side's host now, from the system's own choice."""
        if self.host is None:
            return free_port()
        done = self.run([self.python, "-c", "import socket; s = socket.socket(); "
                         "s.bind(('', 0)); print(s.getsockname()[1])"], slot=1, timeout=60)
        found = [line for line in meaningful(done.stdout) if line.strip().isdigit()]
        return int(found[-1]) if found else random.randint(20000, 59999)

    def answers(self, address, port, seconds=30):
        done = self.run([self.python, "-c", ANSWERS, address, port, int(seconds * 10)],
                        slot=1, timeout=seconds + 60)
        return done.returncode == 0

    def describe(self):
        if self.host is None:
            host = describe_host(argparse.Namespace(tier=None))
            host["l3_caches"] = l3_caches()
            return host
        done = self.run([self.python, "bench.py", "describe"], timeout=120)
        lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
        if done.returncode or not lines:
            said = meaningful(done.stderr) or meaningful(done.stdout)
            raise SystemExit(f"bench: {self.host} has no deployment at {self.root} that "
                             "answers: " + (said[-1] if said else "no answer"))
        return json.loads(lines[-1])


def stop(process, seconds=15):
    try:
        process.stdin.close()
        process.wait(seconds)
    except (OSError, subprocess.TimeoutExpired):
        process.kill()
        process.wait()


def deployment(session, host, given):
    """Where the benchmarks are deployed on host: given, or deploy.env's place for the user
    the login is, /tmp/USER/tango-ucx."""
    if given:
        return given
    done = subprocess.run(session.argv(host, 'cd "/tmp/$(id -un)/tango-ucx/tango-ucx-benchmarks"'
                                             ' && pwd'),
                          text=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
    if done.returncode or not done.stdout.strip():
        raise SystemExit(f"bench: no deployment in /tmp/USER/tango-ucx on {host}: run deploy.sh "
                         "there, or give --remote-root")
    return done.stdout.strip().splitlines()[-1]


class RemotePublisher:
    """tango_ucx_perf_publisher on the peer, with the settings given, stopped when the block
    ends. Whether it listens is asked from the subscriber's host, which connects to it."""

    def __init__(self, peer, subscriber, settings):
        self.address = peer.name
        for _ in range(5):
            self.port = random.randint(20000, 59999)
            self.device = f"{self.address}:{self.port}/bench/ucx/1#dbase=no"
            self.log = tempfile.TemporaryFile(mode="w+")
            self.process = peer.start(
                [peer.root / "build/tango_ucx_perf_publisher", "bench", "-nodb", "-dlist",
                 "bench/ucx/1", "-ORBendPoint", f"giop:tcp:{self.address}:{self.port}"],
                settings, slot=0, log=self.log)
            if subscriber.answers(self.address, self.port) and self.process.poll() is None:
                return
            said = self.said()
            stop(self.process)
            self.log.close()
        raise RuntimeError(f"the publisher did not start on {self.address}: "
                           + (meaningful(said) or ["no output"])[-1])

    def said(self):
        self.log.seek(0)
        return self.log.read()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        stop(self.process)
        self.log.close()


meaningful = remote_login.meaningful


def run_remote_reference(name, frame_bytes, args, cpus, peer, subscriber, settings, tries=2):
    """ucx_perftest from the peer to the subscriber's host: the client there sends, as
    tango-ucx's publisher does; the server receives, on the subscriber loop's CPU. A run that
    fails is tried once more, on another port, and the attempts are recorded."""
    for attempt in range(1, tries + 1):
        point = reference_once(name, frame_bytes, args, cpus, peer, subscriber, settings)
        point["attempts"] = attempt
        if point["ok"]:
            break
    return point


def reference_once(name, frame_bytes, args, cpus, peer, subscriber, settings):
    flags, extra = REFERENCES[name]
    frames = frames_for(frame_bytes, args)
    port = subscriber.free_port()
    common = ["-t", "ucp_am_bw", "-s", str(frame_bytes), "-n", str(frames),
              "-w", str(max(10, frames // 10)), "-p", str(port)] + flags
    point = {"program": name, "ok": False, "frame_bytes": frame_bytes, "frames": frames,
             "command": " ".join(["ucx_perftest", subscriber.name, *common]), "settings": extra}
    log = tempfile.TemporaryFile(mode="w+")
    server = subscriber.start([subscriber.perftest, *common], {**settings, **extra}, slot=1,
                              cpu=cpus[2] if cpus else None, log=log)
    try:
        # The server listens a moment after it starts: the client's side is tried again.
        for _ in range(50):
            time.sleep(0.2)
            done = peer.run([peer.perftest, subscriber.name, *common],
                            {**settings, **extra, "UCX_PROTO_INFO": "y"}, slot=1,
                            cpu=cpus[0] if cpus else None, timeout=300)
            said = done.stdout + done.stderr
            final = [line.split() for line in said.splitlines()
                     if line.lstrip().startswith("Final:")]
            if final or server.poll() is not None:
                break
        if not final:
            stop(server, 10)
            log.seek(0)
            client = meaningful(done.stdout) or meaningful(done.stderr)
            there = meaningful(log.read())
            why = client[-1] if client else f"the client said nothing, exit {done.returncode}"
            if there:
                why += f"; the server said: {there[-1]}"
            return {**point, "failure": why}
        rate = float(final[-1][6])
        return {**point, "ok": True, "frames_per_second": float(final[-1][8]),
                "MiB_per_second": rate, "Gb_per_second": rate * (1 << 20) * 8 / 1e9,
                "protocol": protocols(said, frame_bytes)}
    finally:
        stop(server, 10)
        log.close()


def update_hosts(args):
    """bench.py update: each host's deployed benchmarks brought to this checkout's commit."""
    hosts = [host.strip() for host in args.hosts.split(",") if host.strip()]
    try:
        with remote_login.Session.from_args(args, {host: 1 for host in hosts}) as session:
            for host in hosts:
                update(Side(session, host, deployment(session, host, args.remote_root)))
    except RuntimeError as error:  # a login that failed
        print(f"bench: {error}", file=sys.stderr)
        return 1
    return 0


def describe():
    """This host, for a remote sweep run from another: bench.py describe."""
    host = describe_host(argparse.Namespace(tier=None))
    host["l3_caches"] = l3_caches()
    print(json.dumps(host))
    return 0


def update(side):
    """Bring the benchmarks deployed on side to this checkout's commit, through the login: a git
    bundle of HEAD, so what is not committed here stays behind. They are built again there only
    when the commit changes what is built."""
    here = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    changed = [line[3:] for line in subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True).stdout.splitlines()]
    if changed:
        print(f"bench: not committed here, so {side.name} gets {here} without the changes to: "
              + ", ".join(changed), file=sys.stderr)
    root = shlex.quote(str(side.root))
    line = (f"cd {root} && old=$(git rev-parse HEAD) && cat > .bench-update.bundle && "
            "git fetch -q .bench-update.bundle HEAD && git merge -q --ff-only FETCH_HEAD; "
            "s=$?; rm -f .bench-update.bundle; [ $s -eq 0 ] || exit $s; "
            'if git diff --quiet "$old" HEAD -- src CMakeLists.txt pixi.toml pixi.lock; '
            "then echo 'nothing to build'; "
            "else . ../deploy.env && pixi run -q build >/dev/null && echo 'built again'; fi")
    with tempfile.TemporaryDirectory() as scratch:
        bundle = Path(scratch) / "update.bundle"
        subprocess.run(["git", "-C", str(ROOT), "bundle", "create", "-q", str(bundle), "HEAD"],
                       check=True)
        with open(bundle, "rb") as source:
            done = subprocess.run(side.session.argv(side.host, line), stdin=source, text=True,
                                  capture_output=True, timeout=1800)
    said = meaningful(done.stdout + done.stderr)
    if done.returncode:
        raise SystemExit(f"bench: {side.name} could not be brought to {here}: "
                         + (said[-1] if said else f"exit {done.returncode}"))
    print(f"# {side.name}: tango-ucx-benchmarks at {here}, {said[-1] if said else 'updated'}")


def allocate(side, args, nodes):
    """An exclusive allocation of the nodes, asked for on side; its job id. It is given up if
    the nodes are not free within two minutes."""
    nodes = list(dict.fromkeys(nodes))  # one host as both roles is one node
    line = shlex.join(["salloc", "--no-shell", "--exclusive", "--immediate=120",
                       "-J", "tango-ucx-bench", "-p", args.allocate, "-t", args.allocate_time,
                       "-w", ",".join(nodes)])
    done = subprocess.run(side.session.argv(side.host, line), text=True,
                          stdin=subprocess.DEVNULL, capture_output=True, timeout=300)
    said = done.stdout + done.stderr
    granted = re.search(r"Granted job allocation (\d+)", said)
    if not granted:
        lines = meaningful(said)
        raise SystemExit("bench: no allocation of " + ",".join(nodes) + ": "
                         + (lines[-1] if lines else f"exit {done.returncode}"))
    print(f"# job {granted[1]}: {','.join(nodes)} in {args.allocate}, exclusive, "
          f"for at most {args.allocate_time}")
    return granted[1]


def release(side, job):
    subprocess.run(side.session.argv(side.host, shlex.join(["scancel", job])),
                   stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
    print(f"# job {job} released")


def remote(args):
    # One sweep a file: its report reads every line of it. Said before any login or allocation.
    if args.out and not args.dry_run and Path(args.out).exists():
        raise SystemExit(f"bench: {args.out} exists: a remote sweep's results go to a new file")
    hosts = {args.peer: 2}
    if args.subscriber:
        hosts[args.subscriber] = 2
    with remote_login.Session.from_args(args, hosts) as session:
        peer = Side(session, args.peer, deployment(session, args.peer, args.remote_root))
        subscriber = (Side(session, args.subscriber,
                           deployment(session, args.subscriber, args.remote_root))
                      if args.subscriber else Side(session, None, None))
        if args.address:
            subscriber.name = args.address
        job = None
        try:
            if not args.dry_run:
                if args.update:
                    for side in (peer, subscriber):
                        if side.host:
                            update(side)
                if args.allocate:
                    job = allocate(peer, args, [peer.name, subscriber.name])
            status = measure_remote(args, peer, subscriber)
        finally:
            if job:
                release(peer, job)
    if args.out and not args.dry_run:
        md = str(Path(args.out).with_suffix(".md"))
        subprocess.run([sys.executable, str(ROOT / "report.py"), args.out, "--out", md])
    return status


def one_host(args):
    """Whether the peer is the subscriber's host too: a one-host sweep run from elsewhere."""
    name = lambda host: host.rsplit("@", 1)[-1] if host else None
    return bool(args.subscriber) and name(args.peer) == name(args.subscriber)


def measure_remote(args, peer, subscriber):
    host = subscriber.describe()
    other = peer.describe()
    # The subscriber's host is the host of the sweep; what ran it and how is recorded too.
    host.update(tier=args.tier, options=" ".join(sys.argv[1:]),
                settings=" ".join(words(forwarded())) or "none", peer=other,
                controller=socket.gethostname().split(".")[0],
                controller_benchmarks=revision(ROOT))
    print_host(host)
    print(f"# publisher on {other['host']}: {other['cpu']}, tango-ucx {other['tango_ucx']}  "
          f"tango-ucx-benchmarks {other['benchmarks']}")
    if args.subscriber:
        print(f"# run from {host['controller']}, tango-ucx-benchmarks "
              f"{host['controller_benchmarks']}")
    for key in ("tango_ucx", "benchmarks"):
        if other[key] != host[key]:
            print(f"bench: {key} differs: {other[key]} on {other['host']}, {host[key]} on "
                  f"{host['host']}", file=sys.stderr)
    args.memory_available = min(filter(None, (host.get("memory_available"),
                                              other.get("memory_available"))), default=None)
    plan = []
    for name in args.placement:
        if one_host(args):
            # Both roles on one host, as bench.py local places them, run from elsewhere.
            cpus, why = placement_cpus(name, host["l3_caches"])
            line = placement_line(name, cpus, why) + ("" if why else f", on {host['host']}")
            plan.append((name, cpus, why, line))
            continue
        there, why_there = side_cpus(name, other["l3_caches"])
        here, why_here = side_cpus(name, host["l3_caches"])
        why = (f"on {other['host']}: {why_there}" if why_there else
               f"on {host['host']}: {why_here}" if why_here else None)
        cpus = there + here if there and here else None
        line = placement_line(name, cpus, why, other["host"])
        if args.subscriber:
            line = line.replace(" here ", f" on {host['host']} ")
        plan.append((name, cpus, why, line))
    if args.dry_run:
        points = runs = 0
        for transport, settings in args.transports:
            print(f"# transport {transport}: {' '.join(words(settings)) or 'UCX defaults'}")
            more = dry_run(args, plan)
            points, runs = points + more[0], runs + more[1]
        print(f"# {points} points, {runs} runs")
        return 0
    out = open_out(args, host)
    failed = 0
    print(TABLE_HEAD)
    for transport, settings in args.transports:
        both = {**forwarded(), **settings}
        print(f"# transport {transport}: {' '.join(words(settings)) or 'UCX defaults'}")
        try:
            publisher = RemotePublisher(peer, subscriber, both)
        except RuntimeError as error:
            print(f"# transport {transport}: FAILED: {error}")
            if out:
                out.write(json.dumps({"transport": transport, "skipped": str(error)}) + "\n")
            failed += 1
            continue
        with publisher:
            def point_of(program, frame_bytes, depth, cpus):
                if depth is None:
                    return run_remote_reference(program, frame_bytes, args, cpus, peer,
                                                subscriber, both)
                build = (Path(args.build).resolve() if subscriber.host is None
                         else subscriber.root / "build")
                point = run_point(build, publisher.device, frame_bytes,
                                  depth, args, cpus=cpus,
                                  runner=lambda command: subscriber.run(command, both))
                point["program"] = program
                return point

            done = measure(args, out, publisher, plan, point_of,
                           tag={"transport": transport, "transport_settings": settings,
                                "publisher_host": other["host"],
                                "subscriber_host": host["host"]})
        if done is None:
            return 1
        failed += done
    return 1 if failed else 0


def sweep_options(parser):
    """The options of a sweep, local or remote."""
    parser.add_argument("--tier", choices=TIERS,
                        help="a sweep with ucx_perftest beside it: " + "; ".join(
                            f"{name}: {tier['sizes']} at depth "
                            f"{','.join(map(str, tier['depth']))}, {tier['runs']} runs"
                            for name, tier in TIERS.items()))
    parser.add_argument("--sizes", type=sizes,
                        help="frame sizes with K, M or G; 1K..64M is every power of two between "
                             "(4K,64K,1M,8M)")
    parser.add_argument("--bytes", type=size, help="what each point moves, measured (2G)")
    parser.add_argument("--min-frames", type=int, default=200)
    parser.add_argument("--max-frames", type=int, default=200_000)
    parser.add_argument("--depth", type=lambda s: [int(x) for x in s.split(",")],
                        help="ring depths on both sides, each a row (32)")
    parser.add_argument("--reference", action="store_true", default=None,
                        help="run ucx_perftest at each size too, as UCX chooses and as "
                             "tango-ucx asks")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--runs", type=int, help="runs of each point (3)")
    parser.add_argument("--memory-share", type=float, default=0.5,
                        help="the most of the available memory a point's rings may take (0.5)")
    parser.add_argument("--verify", action="store_true", help="check every byte; costs bandwidth")
    parser.add_argument("--corrupt-every", type=int, default=0,
                        help="damage every nth frame; the run passes when exactly those are found")
    parser.add_argument("--dry-run", action="store_true", help="list the points and run none")
    parser.add_argument("--placement", type=placements,
                        help="thread placements, each a block of rows: " + ", ".join(PLACEMENTS)
                             + " (local: unpinned, and all three with a tier; remote: "
                               "unpinned,one-l3)")
    parser.add_argument("--build", default=os.environ.get("BENCH_BUILD", ROOT / "build"))
    parser.add_argument("--out", help="append each point's JSON line to this file")


def main():
    sys.stdout.reconfigure(line_buffering=True)  # in order with what subprocesses print
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    roles = parser.add_subparsers(dest="where", required=True)
    one = roles.add_parser("local", help="both roles on this host")
    sweep_options(one)
    one.add_argument("--rate", type=int, default=0, help="paced frames/s; latency needs one host")
    one.add_argument("--hdf5", help="also publish this image stack's frames, beside generated "
                                    "ones of its frame size")
    one.add_argument("--hdf5-dataset", help="the dataset, when the file has no NeXus "
                                            "default/signal chain")
    one.add_argument("--read-ahead", type=int, default=8,
                     help="with --hdf5: frames read before they are published, in source slots "
                          "beyond the depth; 0 reads each as it is sent (8)")
    one.set_defaults(run=local)
    two = roles.add_parser("remote", help="the publisher on --peer, the subscriber here or on "
                                          "--subscriber")
    sweep_options(two)
    two.add_argument("--peer", required=True, help="the publisher's host, as ssh reaches it")
    two.add_argument("--subscriber", help="the subscriber's host, as ssh reaches it; this host "
                                          "then only runs the sweep (without it, the "
                                          "subscriber runs here). The peer's name again: "
                                          "both roles on that host, as local runs them")
    two.add_argument("--remote-root",
                     help="the deployed benchmarks on the other hosts "
                          "(/tmp/USER/tango-ucx/tango-ucx-benchmarks, as deploy.sh puts them)")
    two.add_argument("--address", help="the subscriber's name as the peer reaches it "
                                       "(its short name)")
    two.add_argument("--update", action="store_true",
                     help="first bring the other hosts' deployed benchmarks to this checkout's "
                          "commit, and build them again if it changes what is built")
    two.add_argument("--allocate", metavar="PARTITION",
                     help="first allocate both hosts in this SLURM partition, exclusively, and "
                          "release them at the end")
    two.add_argument("--allocate-time", default="1:00:00",
                     help="the allocation's time limit (1:00:00)")
    two.add_argument("--transports", type=transports, default=transports("default"),
                     help="default, rdma:DEVICES, tcp:INTERFACE, with commas between (default)")
    remote_login.add_arguments(two, batch_flag='--ssh-batch')
    two.set_defaults(run=remote, rate=0, hdf5=None, hdf5_dataset=None, read_ahead=0)
    roles.add_parser("describe", help="this host as JSON, for a remote sweep run elsewhere")
    three = roles.add_parser("update", help="bring other hosts' deployed benchmarks to this "
                                            "checkout's commit, and build them again if needed")
    three.add_argument("--hosts", required=True, help="hosts, with commas between")
    three.add_argument("--remote-root", help="the deployed benchmarks there "
                                             "(/tmp/USER/tango-ucx/tango-ucx-benchmarks)")
    remote_login.add_arguments(three, batch_flag="--ssh-batch")
    args = parser.parse_args()
    if args.where == "describe":
        return describe()
    if args.where == "update":
        return update_hosts(args)
    placement_given = args.placement is not None
    usual = dict(sizes="4K,64K,1M,8M", depth=[32], runs=3, bytes="2G", reference=False,
                 placement=["unpinned"])
    if args.tier:
        usual = {**TIERS[args.tier], "reference": True}
    for name, value in usual.items():
        if getattr(args, name) is None:
            setattr(args, name, {"sizes": sizes, "bytes": size}.get(name, lambda v: v)(value))
    if args.where == "remote" and not placement_given:
        # Between hosts, spread across one host's caches says little of the network: pinned and
        # not is the point. On one host, all three, as local.
        args.placement = list(PLACEMENTS) if one_host(args) else ["unpinned", "one-l3"]
    if not 0 <= args.rate <= 1_000_000_000:
        parser.error('--rate must be between zero and one billion frames/s')
    if args.rate and args.reference:
        parser.error('--rate cannot be compared with the unpaced ucx_perftest reference')
    if args.read_ahead < 0:
        parser.error("--read-ahead is zero or more frames")
    if args.hdf5_dataset and not args.hdf5:
        parser.error("--hdf5-dataset needs --hdf5")
    if args.hdf5:
        if any(a.startswith("--sizes") for a in sys.argv[1:]):
            parser.error("with --hdf5 the frame size is the dataset's: --sizes does not apply")
        if not Path(args.hdf5).is_file():
            parser.error(f"no file {args.hdf5}")
        args.hdf5 = str(Path(args.hdf5).resolve())
    started = time.monotonic()
    try:
        status = args.run(args)
    except RuntimeError as error:  # a login that failed
        print(f"bench: {error}", file=sys.stderr)
        status = 1
    print(f"# {time.monotonic() - started:.1f} s", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
