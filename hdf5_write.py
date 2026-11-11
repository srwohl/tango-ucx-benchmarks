#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Write a stream to HDF5 shards with writers laid out as processes x threads, on one host or
several, and make the master file of each run.

    hdf5_write.py --output-dir /gpfs/scratch/me --frame-bytes 4M --layouts 1x1,1x4,4x1
    hdf5_write.py --output-dir /gpfs/scratch/me --hdf5 scan.h5 --layouts 2x1 --hosts n1,n2 \\
        --publisher-address n0 --out results/writes.jsonl

A layout PxT runs P writer processes of T threads on each host; each thread is a puller of the
publisher's pull set and writes its own shard. Threads of a process share the HDF5 library and
its lock, processes do not, so 1xT against Tx1 shows what the lock costs. The publisher runs on
this host. Each run's master is a virtual dataset over the shards, made from the frame indices
they record, and every frame of the run must be in it once. Logs and results stay out of the
output directory, which holds only the files measured.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from bench import Publisher, describe_host, size  # noqa: E402
import remote as remote_login  # noqa: E402


def layouts(text):
    found = []
    for part in text.split(","):
        processes, _, threads = part.strip().partition("x")
        if not processes.isdigit() or not threads.isdigit() or not int(processes) or not int(threads):
            raise argparse.ArgumentTypeError(f"a layout is processes x threads, as 2x4: {part}")
        found.append((int(processes), int(threads)))
    return found


def control(args, *words):
    """A command of tango_ucx_hdf5_writer that answers with one JSON line."""
    done = subprocess.run([str(args.build / "tango_ucx_hdf5_writer"), *words], text=True,
                          capture_output=True, timeout=300)
    lines = done.stdout.strip().splitlines()
    if done.returncode or not lines:
        said = (done.stderr.strip() or done.stdout.strip()).splitlines()
        raise RuntimeError(f"{words[0]}: " + (said[-1] if said else "no answer"))
    return json.loads(lines[-1])


def is_local(host):
    return host in ("localhost", socket.gethostname(), socket.gethostname().split(".")[0])


class Process:
    """One writer process, here or on another host by ssh, with its output read as it comes.
    Another host is reached through the session's login to it; slot is the command's number
    among those that host runs at once."""

    def __init__(self, host, command, log, session=None, slot=0):
        settings = [f"{name}={value}" for name, value in sorted(os.environ.items())
                    if name.startswith(("UCX_", "TANGO_UCX_")) and name != "TANGO_UCX_SOURCE"]
        if is_local(host):
            argv = command
        else:
            remote = shlex.join(["env", *settings, *command])
            argv = session.argv(host, f"cd {shlex.quote(str(ROOT))} && {remote}", slot)
        self.host, self.lines, self.log = host, queue.Queue(), log
        self.process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=log, text=True)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.said = []

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def ready(self, deadline):
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=0.2)
            except queue.Empty:
                continue
            if line is None:
                return False
            self.said.append(line)
            if line.startswith('{"ready"'):
                return True
        return False

    def finish(self, timeout):
        try:
            self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.reader.join(5)
        while not self.lines.empty():
            line = self.lines.get()
            if line is not None:
                self.said.append(line)
        last = next((line for line in reversed(self.said) if line.startswith('{"ok"')), None)
        return json.loads(last) if last else None


def make_master(master, shards, frames, shape, dtype):
    """A virtual /images over the shards: each run of consecutive frame indices in a shard is one
    mapping. Returns the mappings, and the frames missing and written twice."""
    layout = h5py.VirtualLayout(shape=(frames, *shape), dtype=dtype)
    seen = np.zeros(frames, dtype=np.uint32)
    mappings = 0
    for shard in shards:
        with h5py.File(shard, "r") as f:
            index = f["/frame_index"][...]
            count = f["/images"].shape[0]
        if len(index) != count:
            raise RuntimeError(f"{shard}: {count} frames and {len(index)} indices")
        if not count:
            continue
        if index.max() >= frames:
            raise RuntimeError(f"{shard}: a frame index past the run's {frames} frames")
        np.add.at(seen, index, 1)
        source = h5py.VirtualSource(Path(shard).name, "/images", shape=(count, *shape))
        breaks = np.flatnonzero(np.diff(index.astype(np.int64)) != 1) + 1
        for begin, end in zip(np.r_[0, breaks], np.r_[breaks, count]):
            first = int(index[begin])
            layout[first:first + (end - begin)] = source[int(begin):int(end)]
            mappings += 1
    with h5py.File(master, "w", libver="latest") as f:
        f.create_virtual_dataset("/images", layout, fillvalue=0)
        f.attrs["shards"] = len(shards)
    return mappings, int(np.count_nonzero(seen == 0)), int(np.count_nonzero(seen > 1))


def expected_generated(index, frame_bytes):
    """A verified generated frame's bytes, as bench::fill makes them."""
    words = (frame_bytes + 7) // 8
    seed = np.uint64((index + 1) * 0x9e3779b97f4a7c15 % (1 << 64))
    with np.errstate(over="ignore"):
        values = np.arange(words, dtype=np.uint64) + seed
    return values.astype("<u8").view(np.uint8)[:frame_bytes]


def check_master(master, args, source):
    """Every frame of the master against what the publisher sent: the file's frame, or the
    pattern of a verified generated frame. Returns the frames that differ."""
    wrong = 0
    reference = h5py.File(source["file"], "r")[source["dataset"]] if source["kind"] == "hdf5" else None
    with h5py.File(master, "r") as f:
        images = f["/images"]
        for k in range(images.shape[0]):
            got = images[k]
            if reference is not None:
                want = reference[k % source["frames"]] if reference.ndim > 2 else reference[...]
                wrong += not np.array_equal(got, want)
            else:
                wrong += not np.array_equal(got.view(np.uint8).ravel(),
                                            expected_generated(k, got.nbytes))
    return wrong


def sustained(writers, start_ns, end_ns):
    """Rates of whole seconds from the first frame received, from the writers' 100 ms totals."""
    by_second = {}
    for writer in writers:
        for tenth, count in writer["bytes_by_tenth_second"].items():
            second = (int(tenth) * 100_000_000 - start_ns) // 1_000_000_000
            by_second[second] = by_second.get(second, 0) + count
    whole = int((end_ns - start_ns) // 1_000_000_000)
    rates = [by_second.get(s, 0) / 1e9 for s in range(whole)]
    if not rates:
        return {"samples": 0, "GB_per_second_mean": None, "GB_per_second_min": None,
                "GB_per_second_max": None}
    return {"samples": len(rates), "GB_per_second_mean": statistics.mean(rates),
            "GB_per_second_min": min(rates), "GB_per_second_max": max(rates)}


def run_once(args, publisher, session, processes, threads, run_name, repeat, logs):
    hosts = args.hosts
    writers_wanted = processes * threads * len(hosts)
    record = {"test": "hdf5_write", "ok": False, "run": run_name, "repeat": repeat,
              "hosts": hosts, "processes_per_host": processes, "threads": threads,
              "writers": writers_wanted, "frames": args.frames,
              "frames_per_block": args.frames_per_block,
              "blocks_per_stripe": args.blocks_per_stripe, "queue": args.queue,
              "chunk_frames": args.chunk_frames or args.frames_per_block,
              "read_ahead": args.read_ahead if args.hdf5 else 0, "sync": not args.no_sync}
    if writers_wanted > 60:
        return {**record, "failure": "more writers than a publisher's 64 sessions allow"}
    wanted = {"frames": args.frames, "sessions": min(64, writers_wanted + processes * len(hosts)),
              "rate": args.rate, "verify": args.check}
    if args.hdf5:
        wanted["hdf5"] = {"file": args.hdf5, **({"dataset": args.hdf5_dataset}
                                                 if args.hdf5_dataset else {})}
        wanted["read_ahead"] = args.read_ahead
    else:
        wanted["frame_bytes"] = args.frame_bytes
    range_frames = args.frames_per_block * args.blocks_per_stripe
    depth = args.publisher_depth or max(8, 2 * range_frames)
    depth += record["read_ahead"]
    configured = control(args, "configure", publisher.device, "--depth", str(depth),
                         json.dumps(wanted))
    frame_bytes = configured["frame_bytes"]
    record.update(frame_bytes=frame_bytes, source=configured["source"],
                  publisher_depth={"asked": depth, "slots": configured["slots"]})

    run_dir = Path(args.output_dir) / f"tango-ucx-hdf5-{args.sweep}-{run_name}"
    run_dir.mkdir(parents=True)
    prefix = run_dir / "capture"
    started = None
    try:
        launched = []
        shard = 0
        for host in hosts:
            for slot in range(processes):
                build = args.build if is_local(host) else (args.remote_build or args.build)
                command = [f"{build}/tango_ucx_hdf5_writer",
                           "write", publisher.device, str(prefix), "--frame-bytes", str(frame_bytes),
                           "--threads", str(threads), "--first-shard", str(shard),
                           "--frames-per-block", str(args.frames_per_block),
                           "--blocks-per-stripe", str(args.blocks_per_stripe),
                           "--queue", str(args.queue)]
                if args.chunk_frames:
                    command += ["--chunk-frames", str(args.chunk_frames)]
                if args.no_sync:
                    command.append("--no-sync")
                launched.append(Process(host, command, logs, session, slot))
                shard += threads
        deadline = time.monotonic() + 120
        for process in launched:
            if not process.ready(deadline):
                for other in launched:
                    other.process.kill()
                    other.finish(5)
                return {**record, "failure": f"a writer on {process.host} did not join: see the log"}
        started = time.monotonic()
        control(args, "start", publisher.device)
        results = [process.finish(args.timeout) for process in launched]
        wall = time.monotonic() - started
        publisher_result = control(args, "result", publisher.device)
    except Exception as error:  # a run that fails is recorded and the sweep goes on
        return {**record, "failure": str(error)}

    record["publisher"] = publisher_result
    if any(result is None for result in results):
        return {**record, "failure": "a writer process gave no result: see the log"}
    writers = [writer for result in results for writer in result["writers"]]
    record["processes"] = [{k: v for k, v in result.items() if k != "writers"}
                           for result in results]
    shards = [writer["shard"] for writer in writers]
    failures = [writer["failure"] for writer in writers if writer["failure"]]
    starts = [w["first_receive_unix_ns"] for w in writers if w["first_receive_unix_ns"]]
    ends = [w["last_write_unix_ns"] for w in writers if w["last_write_unix_ns"]]
    total = sum(writer["bytes"] for writer in writers)
    seconds = (max(ends) - min(starts)) / 1e9 if starts and ends else 0
    # To the last shard closed and fsynced: the time zarr_write.py's GB/s runs to.
    closed = [w["closed_unix_ns"] for w in writers if w.get("closed_unix_ns")]
    durable = (max(closed) - min(starts)) / 1e9 if starts and closed else 0
    write_seconds = sum(writer["write_seconds"] for writer in writers)
    waits = sum(writer["hdf5_wait_seconds"] for writer in writers)
    per_writer = [w["bytes"] / ((w["last_write_unix_ns"] - w["first_receive_unix_ns"]) / 1e9)
                  / (1 << 20) for w in writers
                  if w["last_write_unix_ns"] > w["first_receive_unix_ns"]]
    record.update(
        bytes=total, seconds=seconds, wall_seconds=wall,
        GB_per_second=total / seconds / 1e9 if seconds else 0,
        MiB_per_second=total / seconds / (1 << 20) if seconds else 0,
        durable_GB_per_second=total / durable / 1e9 if durable else 0,
        write_concurrency=write_seconds / seconds if seconds else 0,
        hdf5_wait_share=waits / (waits + write_seconds) if waits + write_seconds else 0,
        per_writer_MiB_per_second=per_writer,
        frames_per_writer=[writer["frames"] for writer in writers],
        sustained=sustained(writers, min(starts), max(ends)) if starts else None,
        close_seconds_max=max(writer["close_seconds"] for writer in writers),
        sync_seconds_max=max(writer["sync_seconds"] for writer in writers),
        hdf5=results[0]["hdf5"], writer_failures=failures,
        writers_detail=[{k: v for k, v in writer.items() if k != "bytes_by_tenth_second"}
                        for writer in writers])
    try:
        made = time.monotonic()
        with h5py.File(shards[0], "r") as f:
            images = f["/images"]
            shape, dtype = images.shape[1:], images.dtype
        mappings, missing, twice = make_master(run_dir / "capture.h5", shards, args.frames,
                                               shape, dtype)
        record["master"] = {"mappings": mappings, "seconds": time.monotonic() - made,
                            "missing": missing, "duplicates": twice}
        if args.check:
            record["master"]["wrong"] = check_master(run_dir / "capture.h5", args,
                                                     record["source"])
    except Exception as error:
        return {**record, "failure": f"master: {error}"}
    master = record["master"]
    record["ok"] = (not failures and all(result["ok"] for result in results)
                    and not master["missing"] and not master["duplicates"]
                    and not master.get("wrong") and total == args.frames * frame_bytes)
    if not record["ok"] and not failures:
        record["failure"] = (f"{master['missing']} frames missing, {master['duplicates']} twice, "
                             f"{master.get('wrong', 0)} wrong")
    elif failures:
        record["failure"] = failures[0]
    if not args.keep and record["ok"]:
        shutil.rmtree(run_dir)
    return record


def sweep(args):
    host = describe_host(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        filesystem = subprocess.run(["stat", "-f", "-c", "%T", str(output)], capture_output=True,
                                    text=True).stdout.strip() or "unknown"
    except OSError:
        filesystem = "unknown"
    host.update(test="hdf5_write", output_dir=str(output.resolve()), filesystem=filesystem,
                hosts=args.hosts, notes=dict(note.split("=", 1) for note in args.note))
    print(f"# {host['date']}  {host['host']}  writing to {host['output_dir']} ({filesystem})")
    plan = [(p, t, r) for r in range(args.repeats) for p, t in args.layouts]
    if args.dry_run:
        for p, t, r in plan:
            print(f"repeat {r}: {p} processes x {t} threads on {', '.join(args.hosts)}: "
                  f"{p * t * len(args.hosts)} writers")
        return 0
    out = None
    if args.out:
        if Path(args.out).exists():
            raise SystemExit(f"hdf5_write: {args.out} exists: results go to a new file")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out = open(args.out, "w")
        out.write(json.dumps({"meta": host}) + "\n")
    # The writers' own output: beside the results, never in the directory measured.
    logs = (open(Path(args.out).with_suffix(".log"), "w") if args.out else
            tempfile.NamedTemporaryFile("w+", prefix="hdf5-write-", suffix=".log", delete=False))
    remote = not all(is_local(h) for h in args.hosts)
    address = args.publisher_address or (socket.getfqdn() if remote else "127.0.0.1")
    failed = 0
    print(f"{'layout':>8} {'hosts':>5} {'writers':>7} {'GB/s':>7} {'durable':>7} {'per writer':>10} "
          f"{'H5Dwrite':>8} {'waiting':>7} {'1 s min':>7} {'1 s max':>7} {'sync s':>7}  result")
    # One login to each other host for the sweep, enough for its largest layout.
    logins = {h: max(p for p, _ in args.layouts) for h in args.hosts if not is_local(h)}
    with remote_login.Session.from_args(args, logins) as session, \
            Publisher(args.build, address=address) as publisher:
        for number, (processes, threads, repeat) in enumerate(plan):
            if publisher.process.poll() is not None:
                print("the publisher ended:\n" + publisher.said(), file=sys.stderr)
                return 1
            record = run_once(args, publisher, session, processes, threads, f"run-{number:04d}",
                              repeat, logs)
            if out:
                out.write(json.dumps(record) + "\n")
                out.flush()
            failed += not record["ok"]
            mean = (statistics.mean(record["per_writer_MiB_per_second"]) / 1024
                    if record.get("per_writer_MiB_per_second") else 0)
            steady = record.get("sustained") or {}
            def gb(value):
                return f"{value:7.2f}" if isinstance(value, (int, float)) else f"{'':>7}"
            print(f"{f'{processes}x{threads}':>8} {len(args.hosts):>5} {record['writers']:>7} "
                  f"{gb(record.get('GB_per_second'))} {gb(record.get('durable_GB_per_second'))} "
                  f"{mean * 1.073741824:>10.2f} "
                  f"{gb(record.get('write_concurrency'))} "
                  f"{record.get('hdf5_wait_share', 0):>7.0%} "
                  f"{gb(steady.get('GB_per_second_min'))} {gb(steady.get('GB_per_second_max'))} "
                  f"{gb(record.get('sync_seconds_max'))}  "
                  + ("ok" if record["ok"] else f"FAILED: {record.get('failure', '')}"))
    print(f"# writer logs: {logs.name}", file=sys.stderr)
    return 1 if failed else 0


def self_test(args):
    """Small runs of each layout, from generated frames and from a file, every frame checked."""
    sys.path.insert(0, str(ROOT))
    import hdf5_sample
    status = 0
    with tempfile.TemporaryDirectory() as scratch:
        stack = Path(scratch) / "stack.h5"
        hdf5_sample.write(stack, 40, (48, 64), "<u2")
        cases = [
            ("generated", dict(frame_bytes=64 << 10, frames=96), "1x1,1x2,2x1,2x2"),
            ("generated, stripes of 2, queue 1",
             dict(frame_bytes=64 << 10, frames=96, blocks_per_stripe=2, queue=1), "1x2"),
            ("file", dict(hdf5=str(stack), frames=96), "1x1,1x3,3x1"),
        ]
        for name, settings, layout in cases:
            options = [f"--output-dir={scratch}/out", f"--layouts={layout}", "--check",
                       f"--build={args.build}", "--frames-per-block=4"]
            for key, value in settings.items():
                options.append(f"--{key.replace('_', '-')}={value}")
            done = subprocess.run([sys.executable, __file__, *options], text=True,
                                  capture_output=True, timeout=600)
            print(f"{'ok' if done.returncode == 0 else 'FAILED'}    {name}")
            if done.returncode:
                status = 1
                print(done.stdout + done.stderr)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", help="the filesystem under test; a directory a run")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--frame-bytes", type=size, help="generated frames of this size (4M)")
    source.add_argument("--hdf5", help="the frames of this image stack, read by the publisher")
    parser.add_argument("--hdf5-dataset")
    parser.add_argument("--read-ahead", type=int, default=8)
    parser.add_argument("--frames", type=int, default=1024, help="frames a run (1024)")
    parser.add_argument("--layouts", type=layouts, default=layouts("1x1"),
                        help="processes x threads on each host, with commas between (1x1)")
    parser.add_argument("--hosts", default="localhost",
                        help="writer hosts, with commas between; ssh reaches the others (localhost)")
    parser.add_argument("--publisher-address",
                        help="this host as the writer hosts reach it (its name, with --hosts)")
    parser.add_argument("--remote-build", help="the build directory on the other hosts (--build)")
    parser.add_argument("--frames-per-block", type=int, default=8,
                        help="frames an H5Dwrite, the subscription's batch (8)")
    parser.add_argument("--blocks-per-stripe", type=int, default=1,
                        help="blocks a writer takes at a turn (1)")
    parser.add_argument("--queue", type=int, default=2, help="ranges a receive ring holds (2)")
    parser.add_argument("--chunk-frames", type=int, help="frames an HDF5 chunk (the block)")
    parser.add_argument("--publisher-depth", type=int,
                        help="source slots for sending, beyond read-ahead (two ranges, 8 or more)")
    parser.add_argument("--rate", type=int, default=0, help="paced frames/s (0: unpaced)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--no-sync", action="store_true",
                        help="leave out each shard's fsync, and so durability, from a run")
    parser.add_argument("--check", action="store_true",
                        help="read every frame of the master back and compare it with what was sent")
    parser.add_argument("--keep", action="store_true", help="keep the files of runs that pass")
    parser.add_argument("--note", action="append", default=[],
                        help="key=value kept with the results, such as gpfs_block_size=8MiB")
    parser.add_argument("--timeout", type=float, default=1800, help="the longest run, in seconds")
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("--out", help="a new file for the run records, one JSON line each")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    remote_login.add_arguments(parser)
    args = parser.parse_args()
    args.build = args.build.resolve()
    if args.self_test:
        return self_test(args)
    if not args.output_dir:
        parser.error("--output-dir names the filesystem under test")
    if args.hdf5:
        args.hdf5 = str(Path(args.hdf5).resolve())
    elif not args.frame_bytes:
        args.frame_bytes = 4 << 20
    if args.hdf5_dataset and not args.hdf5:
        parser.error("--hdf5-dataset needs --hdf5")
    for name in ("frames", "frames_per_block", "blocks_per_stripe", "queue", "repeats"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} is positive")
    if args.read_ahead < 0:
        parser.error("--read-ahead is zero or more")
    if any("=" not in note for note in args.note):
        parser.error("a note is key=value")
    args.hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    args.tier = None  # what describe_host records of bench.py's options
    args.sweep = time.strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
    started = time.monotonic()
    status = sweep(args)
    print(f"# {time.monotonic() - started:.1f} s", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
