#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run tango_ucx_perftest over a list of frame sizes, with both roles on this host.

With --reference, or a --tier, ucx_perftest runs at each size too. A point whose rings do not
fit this host's memory is left out, and said to be.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import re
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
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
TIERS = {
    "quick": dict(sizes="4K,64K,1M,16M", depth=[32], runs=1, bytes="256M"),
    "baseline": dict(sizes="1K,4K,16K,64K,256K,1M,4M,16M", depth=[2, 32], runs=3, bytes="2G"),
    "standard": dict(sizes="1K..64M", depth=[2, 32], runs=3, bytes="2G"),
    "full": dict(sizes="1K..2G", depth=[2, 8, 32, 128], runs=5, bytes="4G"),
}


def available_bytes():
    with open("/proc/meminfo") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return None


def left_out(frame_bytes, depth, args):
    """Why a point is not run, or nothing. Both roles' memory is this host's."""
    if frame_bytes > 2 << 30:
        return "a frame is at most 2 GiB"
    # A publisher has at least three slots; ucx_perftest has a buffer on each side.
    need = frame_bytes * (2 if depth is None else max(depth, 3) + depth)
    free = available_bytes()
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
    """tango_ucx_perf_publisher as a -nodb device server, stopped when the block ends."""

    def __init__(self, build, env=None):
        # A port found free can be taken before the server binds it: then another is tried.
        for _ in range(5):
            self.port = free_port()
            self.device = f"127.0.0.1:{self.port}/bench/ucx/1#dbase=no"
            self.log = tempfile.TemporaryFile(mode="w+")
            self.process = subprocess.Popen(
                [str(build / "tango_ucx_perf_publisher"), "bench", "-nodb", "-dlist",
                 "bench/ucx/1", "-ORBendPoint", f"giop:tcp:127.0.0.1:{self.port}"],
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
                if probe.connect_ex(("127.0.0.1", self.port)) == 0:
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


def run_point(build, device, frame_bytes, depth, args):
    command = [str(build / "tango_ucx_perftest"), device, "-t", "bw", "-s", str(frame_bytes),
               "-n", str(frames_for(frame_bytes, args)), "-d", str(depth),
               "-b", str(args.batch), "--json"]
    if args.verify or args.corrupt_every:
        command.append("--verify")
    if args.corrupt_every:
        command += ["--corrupt-every", str(args.corrupt_every)]
    if getattr(args, 'rate', 0):
        command += ['--rate', str(args.rate)]
    done = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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


def run_reference(name, frame_bytes, args):
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
                                  stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return {**point, "failure": "ucx_perftest is not on the path"}
    try:
        # The server listens a moment after it starts: the subscriber's side is tried again.
        for _ in range(50):
            time.sleep(0.2)
            done = subprocess.run(command[:1] + ["127.0.0.1"] + command[1:],
                                  env={**env, "UCX_PROTO_INFO": "y"}, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
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


def rows_of(args):
    """The rows of one frame size: the references, then tango-ucx at each depth."""
    rows = [(name, None) for name in (REFERENCES if args.reference else ())]
    return rows + [("tango-ucx", depth) for depth in args.depth]


def local(args):
    build = Path(args.build).resolve()
    host = describe_host(args)
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
    if args.dry_run:
        points = runs = 0
        for frame_bytes in args.sizes:
            for program, depth in rows_of(args):
                why = left_out(frame_bytes, depth, args)
                print(f"{frame_bytes:>10}  {program:<16} {'' if depth is None else depth:>5} "
                      f"{frames_for(frame_bytes, args):>8} frames  "
                      + (f"left out: {why}" if why else f"{args.runs} runs"))
                points += not why
                runs += 0 if why else args.runs
        print(f"# {points} points, {runs} runs")
        return 0
    out = None
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out = open(args.out, "a")
        out.write(json.dumps({"meta": host}) + "\n")
    failed = 0
    with Publisher(build) as publisher:
        # One run says little on a machine that is doing anything else: the median of the runs
        # is the figure, and the best and the worst are beside it. "of raw" is tango-ucx against
        # ucx_perftest as tango-ucx asks.
        print(f"{'#bytes':>10}  {'program':<16} {'depth':>5} {'#frames':>8} {'frames/s':>12} "
              f"{'MiB/s':>10} {'best':>9} {'worst':>9} {'of raw':>7}  path")
        for frame_bytes in args.sizes:
            raw = None
            for program, depth in rows_of(args):
                why = left_out(frame_bytes, depth, args)
                if why:
                    print(f"{frame_bytes:>10}  {program:<16} "
                          f"{'' if depth is None else depth:>5} left out: {why}")
                    if out:
                        out.write(json.dumps({"program": program, "frame_bytes": frame_bytes,
                                              "depth_asked": depth, "skipped": why}) + "\n")
                    continue
                runs = []
                for run in range(args.runs):
                    if publisher.process.poll() is not None:
                        print("the publisher ended:\n" + publisher.said(), file=sys.stderr)
                        return 1
                    if depth is None:
                        point = run_reference(program, frame_bytes, args)
                    else:
                        point = run_point(build, publisher.device, frame_bytes, depth, args)
                        point["program"] = program
                    point["depth_asked"] = depth
                    point["run"] = run
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
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    roles = parser.add_subparsers(dest="where", required=True)
    one = roles.add_parser("local", help="both roles on this host")
    one.add_argument("--tier", choices=TIERS,
                     help="a sweep with ucx_perftest beside it: " + "; ".join(
                         f"{name}: {tier['sizes']} at depth "
                         f"{','.join(map(str, tier['depth']))}, {tier['runs']} runs"
                         for name, tier in TIERS.items()))
    one.add_argument("--sizes", type=sizes,
                     help="frame sizes with K, M or G; 1K..64M is every power of two between "
                          "(4K,64K,1M,8M)")
    one.add_argument("--bytes", type=size, help="what each point moves, measured (2G)")
    one.add_argument("--min-frames", type=int, default=200)
    one.add_argument("--max-frames", type=int, default=200_000)
    one.add_argument("--depth", type=lambda s: [int(x) for x in s.split(",")],
                     help="ring depths on both sides, each a row (32)")
    one.add_argument("--reference", action="store_true", default=None,
                     help="run ucx_perftest at each size too, as UCX chooses and as tango-ucx asks")
    one.add_argument("--batch", type=int, default=1)
    one.add_argument("--rate", type=int, default=0, help="paced frames/s; latency needs one host")
    one.add_argument("--runs", type=int, help="runs of each point (3)")
    one.add_argument("--memory-share", type=float, default=0.5,
                     help="the most of the available memory a point's rings may take (0.5)")
    one.add_argument("--verify", action="store_true", help="check every byte; costs bandwidth")
    one.add_argument("--corrupt-every", type=int, default=0,
                     help="damage every nth frame; the run passes when exactly those are found")
    one.add_argument("--dry-run", action="store_true", help="list the points and run none")
    one.add_argument("--build", default=os.environ.get("BENCH_BUILD", ROOT / "build"))
    one.add_argument("--out", help="append each point's JSON line to this file")
    one.set_defaults(run=local)
    args = parser.parse_args()
    usual = dict(sizes="4K,64K,1M,8M", depth=[32], runs=3, bytes="2G", reference=False)
    if args.tier:
        usual = {**TIERS[args.tier], "reference": True}
    for name, value in usual.items():
        if getattr(args, name) is None:
            setattr(args, name, {"sizes": sizes, "bytes": size}.get(name, lambda v: v)(value))
    if not 0 <= args.rate <= 1_000_000_000:
        parser.error('--rate must be between zero and one billion frames/s')
    if args.rate and args.reference:
        parser.error('--rate cannot be compared with the unpaced ucx_perftest reference')
    started = time.monotonic()
    status = args.run(args)
    print(f"# {time.monotonic() - started:.1f} s", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
