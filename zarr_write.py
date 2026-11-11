#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Write a stream to Zarr v3 parts with acquire-zarr, writers laid out as processes x threads on
one host or several, and stitch each run's parts into one Zarr store. Runs in `pixi run -e zarr`.

    zarr_write.py --build build-zarr --output-dir /gpfs/scratch/me --frame-bytes 4M \\
        --layouts 1x1,1x4,4x1 --direct-io
    zarr_write.py --build build-zarr --output-dir /gpfs/scratch/me --hdf5 scan.h5 --layouts 2x1 \\
        --hosts n1,n2 --publisher-address n0 --compression blosc-lz4 --out results/zarr.jsonl

The counterpart of hdf5_write.py: a layout PxT runs P writer processes of T threads on each
host, each thread a puller of the publisher's pull set writing its own part. acquire-zarr's
streams share no lock, so 1xT against Tx1 shows what threads cost without one.

A part's shard files hold its frames in the order it was given them, as an HDF5 shard does.
--shard-frames part makes one shard a part, so a writer makes and fsyncs one file, as an HDF5
writer does. Each part's /images is an OME-Zarr 0.5 image whose one level, /images/0, holds the
frames as [t, ...image]. When a shard divides the pull range, the run's capture.zarr is the
whole stream as such an image, which any OME-Zarr or Zarr v3 reader opens: its level's
zarr.json is a part's with the run's frame count, and each of its shard files is a relative
symlink to a part's. That works
because ranges start at multiples of the range and each is one puller's, unless a puller leaves.
A larger shard spans the ranges of several pullers, so there is no capture.zarr and each part is
read on its own instead. Either way the frames' runs in each part's attributes say which frames
it holds; a frame missing or written twice fails the run. Every image is validated by
ome-zarr-models and opened by ome-zarr, and --check reads every frame back with zarr-python and
with tensorstore.

GPFS: --direct-io writes with O_DIRECT. GPFS takes appends that allocate blocks, and writes not
aligned to its sectors, off its fast direct path, so sweep it against buffered writes, with
chunks of the file system's block size or more (mmlsfs -B) and without compression, which makes
writes unaligned. acquire-zarr fsyncs each shard file either way.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from bench import Publisher, describe_host, size  # noqa: E402
from hdf5_write import Process, expected_generated, is_local, layouts, sustained  # noqa: E402
import remote as remote_login  # noqa: E402

WRITER = "tango_ucx_zarr_writer"
COMPRESSIONS = ("none", "blosc-lz4", "blosc-zstd", "zstd")


def control(args, *words):
    """A command of tango_ucx_zarr_writer that answers with one JSON line."""
    done = subprocess.run([str(args.build / WRITER), *words], text=True, capture_output=True,
                          timeout=300)
    lines = done.stdout.strip().splitlines()
    if done.returncode or not lines:
        said = (done.stderr.strip() or done.stdout.strip()).splitlines()
        raise RuntimeError(f"{words[0]}: " + (said[-1] if said else "no answer"))
    return json.loads(lines[-1])


# A store's /images is an OME-Zarr image: a group, with the frames' runs among its attributes,
# whose one level, LEVEL, is the array of frames.
LEVEL = "0"


def image_meta(store):
    return json.loads((Path(store) / "images" / "zarr.json").read_text())


def array_meta(store):
    return json.loads((Path(store) / "images" / LEVEL / "zarr.json").read_text())


def frame_runs(store):
    return image_meta(store)["attributes"]["tango_ucx"]["frame_runs"]


def check_ngff(store):
    """The store's /images as the OME-Zarr tools see it: ome-zarr-models validates its metadata
    against the specification, and ome-zarr, napari's reader, finds its levels."""
    from ome_zarr_models import open_ome_zarr
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader
    image = Path(store) / "images"
    model = open_ome_zarr(zarr.open_group(str(image), mode="r"))
    nodes = list(Reader(parse_url(str(image)))())
    if not nodes or not nodes[0].data:
        raise RuntimeError(f"{image}: ome-zarr finds no image")
    return {"ome_version": model.attributes.ome.version, "ngff_levels": len(nodes[0].data),
            "ngff_shape": list(nodes[0].data[0].shape),
            "ngff_axes": [axis["name"] for axis in nodes[0].metadata.get("axes", [])]}


def stitch(capture, parts, frames):
    """capture.zarr over the parts, an OME-Zarr image like theirs: one symlink a shard. Returns
    the links, and the frames missing and written twice."""
    capture = Path(capture)
    metas = [array_meta(part) for part in parts]
    first = metas[0]
    same = ("chunk_grid", "codecs", "data_type", "fill_value", "chunk_key_encoding",
            "dimension_names")
    for part, meta in zip(parts, metas):
        if any(meta.get(k) != first.get(k) for k in same) or meta["shape"][1:] != first["shape"][1:]:
            raise RuntimeError(f"{part}: not the array of {parts[0]}")
    if first["chunk_key_encoding"] != {"name": "default", "configuration": {"separator": "/"}}:
        raise RuntimeError(f"chunk keys {first['chunk_key_encoding']}: only '/' keys are linked")
    shard = first["chunk_grid"]["configuration"]["chunk_shape"][0]
    image_key = Path(*["0"] * (len(first["shape"]) - 1))  # a frame is one shard across
    seen = np.zeros(frames, dtype=np.uint32)
    links = 0
    (capture / "images" / LEVEL / "c").mkdir(parents=True)
    for part, meta in zip(parts, metas):
        offset = 0  # the part's frame of the run's first
        for start, count in frame_runs(part):
            if start + count > frames:
                raise RuntimeError(f"{part}: frames {start}+{count} past the run's {frames}")
            if start % shard or offset % shard or (count % shard and start + count != frames):
                raise RuntimeError(f"{part}: frames {start}+{count}, its frame {offset} on, are "
                                   f"not whole shards of {shard}: a range was split")
            seen[start:start + count] += 1
            for k in range(math.ceil(count / shard)):
                source = Path(part) / "images" / LEVEL / "c" / str(offset // shard + k) / image_key
                if not source.exists():
                    continue  # a shard of fill values is not written
                target = capture / "images" / LEVEL / "c" / str(start // shard + k) / image_key
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink():
                    continue  # counted as written twice
                target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()))
                links += 1
            offset += count
        if offset != meta["shape"][0]:
            raise RuntimeError(f"{part}: {meta['shape'][0]} frames, and runs of {offset}")
    (capture / "zarr.json").write_text((Path(parts[0]) / "zarr.json").read_text())
    (capture / "images" / LEVEL / "zarr.json").write_text(
        json.dumps(dict(first, shape=[frames, *first["shape"][1:]]), indent=4))
    # The parts' OME metadata names the level and its axes, not its length, so it is the run's.
    image = image_meta(parts[0])
    image["attributes"]["tango_ucx"] = {
        "frames": frames, "shard_frames": shard,
        "parts": [os.path.relpath(Path(p).resolve(), capture.resolve()) for p in parts],
        "frame_shape": image["attributes"]["tango_ucx"]["frame_shape"]}
    (capture / "images" / "zarr.json").write_text(json.dumps(image, indent=4))
    return links, int(np.count_nonzero(seen == 0)), int(np.count_nonzero(seen > 1))


def readers(array):
    """The array opened by zarr-python, and by tensorstore when it is installed."""
    images = zarr.open_array(str(array), mode="r")
    try:
        import tensorstore
    except ImportError:
        return images, None
    spec = {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(Path(array).resolve())}}
    other = tensorstore.open(spec, read=True).result()
    if list(other.shape) != list(images.shape):
        raise RuntimeError(f"{array}: tensorstore sees {list(other.shape)}, zarr {images.shape}")
    return images, other


def compare(images, other, frame_of, source):
    """Every frame of the array, frame i being the stream's frame_of[i], against what the
    publisher sent: the file's frame, or a verified generated frame's pattern. Returns the frames
    wrong, and the blocks where tensorstore reads something else."""
    reference = None
    if source["kind"] == "hdf5":
        import h5py
        reference = h5py.File(source["file"], "r")[source["dataset"]]
    step = images.chunks[0]
    wrong = disagree = 0
    for begin in range(0, images.shape[0], step):
        end = min(begin + step, images.shape[0])  # tensorstore refuses a slice past the end
        block = images[begin:end]
        if other is not None:
            disagree += int(not np.array_equal(block, other[begin:end].read().result()))
        for i, got in enumerate(block):
            k = int(frame_of[begin + i])
            if reference is not None:
                want = reference[k % source["frames"]] if reference.ndim > 2 else reference[...]
                wrong += not np.array_equal(got.reshape(want.shape), want)
            else:
                wrong += not np.array_equal(got.view(np.uint8).ravel(),
                                            expected_generated(k, got.nbytes))
    return wrong, disagree


def check_capture(capture, source, check_frames):
    """capture.zarr's image opened by the OME-Zarr tools and its level by both Zarr readers; with
    check_frames, every frame compared."""
    images, other = readers(Path(capture) / "images" / LEVEL)
    seen = {"shape": list(images.shape), "dtype": str(images.dtype), **check_ngff(capture)}
    if seen["ngff_shape"] != seen["shape"]:
        raise RuntimeError(f"ome-zarr sees {seen['ngff_shape']}, zarr {seen['shape']}")
    if other is not None:
        seen["tensorstore_shape"] = list(other.shape)
    if check_frames:
        wrong, disagree = compare(images, other, np.arange(images.shape[0]), source)
        seen.update(wrong=wrong, tensorstore_disagrees=disagree if other is not None else None)
    return seen


def check_parts(parts, frames, source, check_frames):
    """Each part's image opened by the OME-Zarr tools and its level by both Zarr readers, its
    length that of its frames' runs; with
    check_frames, every frame compared with the frame of the stream its runs say it is."""
    seen = np.zeros(frames, dtype=np.uint32)
    result = {"parts": len(parts), "wrong": 0, "tensorstore_disagrees": 0}
    tensorstore = False
    for part in parts:
        runs = frame_runs(part)
        frame_of = np.concatenate([np.arange(start, start + count) for start, count in runs]
                                  or [np.zeros(0, dtype=np.int64)])
        if frame_of.size and frame_of.max() >= frames:
            raise RuntimeError(f"{part}: frames past the run's {frames}")
        np.add.at(seen, frame_of, 1)
        images, other = readers(Path(part) / "images" / LEVEL)
        tensorstore |= other is not None
        ngff = check_ngff(part)
        if ngff["ngff_shape"] != list(images.shape):
            raise RuntimeError(f"{part}: ome-zarr sees {ngff['ngff_shape']}, zarr {images.shape}")
        result["ome_version"] = ngff["ome_version"]
        if images.shape[0] != frame_of.size:
            raise RuntimeError(f"{part}: {images.shape[0]} frames, and runs of {frame_of.size}")
        if check_frames:
            wrong, disagree = compare(images, other, frame_of, source)
            result["wrong"] += wrong
            result["tensorstore_disagrees"] += disagree
    if not check_frames:
        del result["wrong"], result["tensorstore_disagrees"]
    elif not tensorstore:
        result["tensorstore_disagrees"] = None
    result.update(missing=int(np.count_nonzero(seen == 0)),
                  duplicates=int(np.count_nonzero(seen > 1)))
    return result


def shard_chunks(args, writers):
    """Chunks a shard, or None for the writer's own choice, a shard a range."""
    chunk = args.chunk_frames or args.frames_per_block
    if args.shard_frames == "part":
        # The most frames a part can be given: the ranges dealt out in turn.
        range_frames = args.frames_per_block * args.blocks_per_stripe
        most = min(args.frames,
                   math.ceil(math.ceil(args.frames / range_frames) / writers) * range_frames)
        return max(1, math.ceil(most / chunk))
    if args.shard_frames:
        if int(args.shard_frames) % chunk:
            raise ValueError(f"--shard-frames {args.shard_frames} is not whole chunks of {chunk}")
        return int(args.shard_frames) // chunk
    return args.shard_chunks


def writer_options(args, shards):
    options = ["--frames-per-block", str(args.frames_per_block),
               "--blocks-per-stripe", str(args.blocks_per_stripe), "--queue", str(args.queue),
               "--compression", args.compression, "--level", str(args.level),
               "--shuffle", str(args.shuffle), "--zarr-threads", str(args.zarr_threads)]
    if args.chunk_frames:
        options += ["--chunk-frames", str(args.chunk_frames)]
    if shards:
        options += ["--shard-chunks", str(shards)]
    if args.direct_io:
        options.append("--direct-io")
    return options


def run_once(args, publisher, session, processes, threads, run_name, repeat, logs):
    hosts = args.hosts
    writers_wanted = processes * threads * len(hosts)
    record = {"test": "zarr_write", "ok": False, "run": run_name, "repeat": repeat,
              "hosts": hosts, "processes_per_host": processes, "threads": threads,
              "writers": writers_wanted, "frames": args.frames,
              "frames_per_block": args.frames_per_block,
              "blocks_per_stripe": args.blocks_per_stripe, "queue": args.queue,
              "compression": args.compression, "direct_io": args.direct_io,
              "zarr_threads": args.zarr_threads,
              "read_ahead": args.read_ahead if args.hdf5 else 0}
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
    chunk = args.chunk_frames or args.frames_per_block
    try:
        shards = shard_chunks(args, writers_wanted)
    except ValueError as error:
        return {**record, "failure": str(error)}
    shard_frames = chunk * (shards or max(1, range_frames // chunk))
    # A part's shards are the stream's only when a shard divides the range.
    stitched = range_frames % shard_frames == 0
    record.update(shard_frames=shard_frames, stitched=stitched)
    depth = (args.publisher_depth or max(8, 2 * range_frames)) + record["read_ahead"]
    try:
        configured = control(args, "configure", publisher.device, "--depth", str(depth),
                             json.dumps(wanted))
    except Exception as error:
        return {**record, "failure": str(error)}
    frame_bytes = configured["frame_bytes"]
    record.update(frame_bytes=frame_bytes, source=configured["source"],
                  publisher_depth={"asked": depth, "slots": configured["slots"]})

    run_dir = Path(args.output_dir) / f"tango-ucx-zarr-{args.sweep}-{run_name}"
    run_dir.mkdir(parents=True)
    prefix = run_dir / "capture"
    try:
        launched = []
        part = 0
        for host in hosts:
            for slot in range(processes):
                build = args.build if is_local(host) else (args.remote_build or args.build)
                command = [f"{build}/{WRITER}", "write", publisher.device, str(prefix),
                           "--frame-bytes", str(frame_bytes), "--threads", str(threads),
                           "--first-part", str(part), *writer_options(args, shards)]
                launched.append(Process(host, command, logs, session, slot))
                part += threads
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
    parts = [writer["part"] for writer in writers]
    failures = [writer["failure"] for writer in writers if writer["failure"]]
    starts = [w["first_receive_unix_ns"] for w in writers if w["first_receive_unix_ns"]]
    appended = [w["last_append_unix_ns"] for w in writers if w["last_append_unix_ns"]]
    closed = [w["closed_unix_ns"] for w in writers if w["closed_unix_ns"]]
    total = sum(writer["bytes"] for writer in writers)
    # Frames are on disk, and durable, once their part is closed; appended, once copied.
    seconds = (max(closed) - min(starts)) / 1e9 if starts and closed else 0
    accept_seconds = (max(appended) - min(starts)) / 1e9 if starts and appended else 0
    append_seconds = sum(writer["append_seconds"] for writer in writers)
    per_writer = [w["bytes"] / ((w["closed_unix_ns"] - w["first_receive_unix_ns"]) / 1e9)
                  / (1 << 20) for w in writers
                  if w["closed_unix_ns"] > w["first_receive_unix_ns"] > 0]
    record.update(
        bytes=total, seconds=seconds, wall_seconds=wall,
        GB_per_second=total / seconds / 1e9 if seconds else 0,
        MiB_per_second=total / seconds / (1 << 20) if seconds else 0,
        accepted_GB_per_second=total / accept_seconds / 1e9 if accept_seconds else 0,
        append_concurrency=append_seconds / accept_seconds if accept_seconds else 0,
        per_writer_MiB_per_second=per_writer,
        frames_per_writer=[writer["frames"] for writer in writers],
        sustained_accepted=sustained(writers, min(starts), max(appended)) if starts else None,
        close_seconds_max=max(writer["close_seconds"] for writer in writers),
        memory_peak_bytes=sum(writer["memory_peak_bytes"] for writer in writers),
        memory_estimate_bytes=sum(writer["memory_estimate_bytes"] for writer in writers),
        zarr=results[0]["zarr"], writer_failures=failures,
        writers_detail=[{k: v for k, v in writer.items() if k != "bytes_by_tenth_second"}
                        for writer in writers])
    try:
        made = time.monotonic()
        if stitched:
            links, missing, twice = stitch(run_dir / "capture.zarr", parts, args.frames)
            record["capture"] = {"kind": "stitched", "links": links,
                                 "seconds": time.monotonic() - made,
                                 "missing": missing, "duplicates": twice}
            record["capture"].update(check_capture(run_dir / "capture.zarr", record["source"],
                                                   args.check))
        else:
            record["capture"] = {"kind": "parts", **check_parts(parts, args.frames,
                                                                record["source"], args.check)}
    except Exception as error:
        return {**record, "failure": f"capture: {error}"}
    capture = record["capture"]
    record["ok"] = (not failures and all(result["ok"] for result in results)
                    and not capture["missing"] and not capture["duplicates"]
                    and not capture.get("wrong") and not capture.get("tensorstore_disagrees")
                    and total == args.frames * frame_bytes)
    if failures:
        record["failure"] = failures[0]
    elif not record["ok"]:
        record["failure"] = (f"{capture['missing']} frames missing, {capture['duplicates']} "
                             f"twice, {capture.get('wrong', 0)} wrong, tensorstore disagrees on "
                             f"{capture.get('tensorstore_disagrees') or 0} blocks")
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
    host.update(test="zarr_write", output_dir=str(output.resolve()), filesystem=filesystem,
                hosts=args.hosts, notes=dict(note.split("=", 1) for note in args.note))
    print(f"# {host['date']}  {host['host']}  writing to {host['output_dir']} ({filesystem}), "
          f"{args.compression}{', direct I/O' if args.direct_io else ''}")
    plan = [(p, t, r) for r in range(args.repeats) for p, t in args.layouts]
    if args.dry_run:
        for p, t, r in plan:
            print(f"repeat {r}: {p} processes x {t} threads on {', '.join(args.hosts)}: "
                  f"{p * t * len(args.hosts)} writers")
        return 0
    out = None
    if args.out:
        if Path(args.out).exists():
            raise SystemExit(f"zarr_write: {args.out} exists: results go to a new file")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out = open(args.out, "w")
        out.write(json.dumps({"meta": host}) + "\n")
    # The writers' own output: beside the results, never in the directory measured.
    logs = (open(Path(args.out).with_suffix(".log"), "w") if args.out else
            tempfile.NamedTemporaryFile("w+", prefix="zarr-write-", suffix=".log", delete=False))
    remote = not all(is_local(h) for h in args.hosts)
    address = args.publisher_address or (socket.getfqdn() if remote else "127.0.0.1")
    failed = 0
    print(f"{'layout':>8} {'hosts':>5} {'writers':>7} {'shard':>6} {'GB/s':>7} {'accepted':>8} "
          f"{'per writer':>10} {'appends':>7} {'1 s min':>7} {'1 s max':>7} {'close s':>7} "
          f"{'MiB':>7}  result")
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
            steady = record.get("sustained_accepted") or {}

            def gb(value, width=7):
                return f"{value:{width}.2f}" if isinstance(value, (int, float)) else " " * width
            print(f"{f'{processes}x{threads}':>8} {len(args.hosts):>5} {record['writers']:>7} "
                  f"{record.get('shard_frames', ''):>6} "
                  f"{gb(record.get('GB_per_second'))} {gb(record.get('accepted_GB_per_second'), 8)} "
                  f"{mean * 1.073741824:>10.2f} {gb(record.get('append_concurrency'))} "
                  f"{gb(steady.get('GB_per_second_min'))} {gb(steady.get('GB_per_second_max'))} "
                  f"{gb(record.get('close_seconds_max'))} "
                  f"{record.get('memory_peak_bytes', 0) / (1 << 20):>7.0f}  "
                  + ("ok" if record["ok"] else f"FAILED: {record.get('failure', '')}"))
    print(f"# writer logs: {logs.name}", file=sys.stderr)
    return 1 if failed else 0


def self_test(args):
    """Small runs of each layout, from generated frames and from a file, every frame read back
    by both readers, from capture.zarr or, for shards wider than a range, from each part; and
    direct I/O, where the build directory's file system takes it."""
    import hdf5_sample
    status = 0
    filesystem = subprocess.run(["stat", "-f", "-c", "%T", str(args.build)], capture_output=True,
                                text=True).stdout.strip()
    with tempfile.TemporaryDirectory() as scratch, \
            tempfile.TemporaryDirectory(dir=args.build, prefix="zarr-direct-") as beside:
        stack = Path(scratch) / "stack.h5"
        hdf5_sample.write(stack, 40, (48, 64), "<u2")
        generated = dict(frame_bytes=64 << 10, frames=96)
        cases = [
            ("generated", generated, "1x1,1x2,2x1,2x2", scratch),
            ("generated, stripes of 2, queue 1",
             dict(generated, blocks_per_stripe=2, queue=1), "1x2", scratch),
            ("generated, chunks of 2 in shards of 2", dict(generated, chunk_frames=2), "2x2",
             scratch),
            ("generated, a part ends in a partial shard", dict(generated, frames=90), "1x3",
             scratch),
            ("generated, blosc-zstd", dict(generated, compression="blosc-zstd"), "1x2", scratch),
            ("file", dict(hdf5=str(stack), frames=96), "1x1,1x3,3x1", scratch),
            ("file, zstd", dict(hdf5=str(stack), frames=96, compression="zstd"), "2x1", scratch),
            ("generated, one shard a part", dict(generated, shard_frames="part"), "1x1,1x2,3x1",
             scratch),
            ("generated, shards of 12 over ranges of 4, partial", dict(generated, frames=90,
                                                                       shard_frames=12), "2x1",
             scratch),
            ("file, one shard a part", dict(hdf5=str(stack), frames=96, shard_frames="part"),
             "1x3", scratch),
        ]
        if filesystem in ("nfs", "gpfs"):
            cases.append((f"generated, direct I/O on {filesystem}", dict(generated, direct_io=True),
                          "1x2", beside))
        else:
            print(f"skip  direct I/O: {args.build} is on {filesystem or 'an unknown file system'}, "
                  "not NFS or GPFS")
        for name, settings, layout, where in cases:
            options = [f"--output-dir={where}/out", f"--layouts={layout}", "--check",
                       f"--build={args.build}", "--frames-per-block=4"]
            for key, value in settings.items():
                flag = f"--{key.replace('_', '-')}"
                options.append(flag if value is True else f"{flag}={value}")
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
                        help="frames an append, the subscription's batch (8)")
    parser.add_argument("--blocks-per-stripe", type=int, default=1,
                        help="blocks a writer takes at a turn (1)")
    parser.add_argument("--queue", type=int, default=2, help="ranges a receive ring holds (2)")
    parser.add_argument("--chunk-frames", type=int, help="frames a Zarr chunk (the block)")
    shard = parser.add_mutually_exclusive_group()
    shard.add_argument("--shard-chunks", type=int, help="chunks a shard file (a range's)")
    shard.add_argument("--shard-frames",
                       help="frames a shard file, whole chunks; 'part' for one shard a writer, "
                            "as an HDF5 writer makes one file. capture.zarr is made only when a "
                            "shard divides the range; otherwise the parts are read one by one")
    parser.add_argument("--compression", choices=COMPRESSIONS, default="none")
    parser.add_argument("--level", type=int, default=1, help="the compression level (1)")
    parser.add_argument("--shuffle", type=int, default=1, choices=(0, 1, 2),
                        help="blosc's shuffle: 0 none, 1 bytes, 2 bits (1; 0 without blosc)")
    parser.add_argument("--zarr-threads", type=int, default=0,
                        help="acquire-zarr's threads a stream (0: its own choice, which is every "
                             "core, for each stream)")
    parser.add_argument("--direct-io", action="store_true",
                        help="write with O_DIRECT; the file system must take unaligned writes")
    parser.add_argument("--publisher-depth", type=int,
                        help="source slots for sending, beyond read-ahead (two ranges, 8 or more)")
    parser.add_argument("--rate", type=int, default=0, help="paced frames/s (0: unpaced)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--check", action="store_true",
                        help="read every frame back, with both readers, against what was sent")
    parser.add_argument("--keep", action="store_true", help="keep the files of runs that pass")
    parser.add_argument("--note", action="append", default=[],
                        help="key=value kept with the results, such as gpfs_block_size=8MiB")
    parser.add_argument("--timeout", type=float, default=1800, help="the longest run, in seconds")
    parser.add_argument("--build", type=Path, default=ROOT / "build-zarr")
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
    if args.read_ahead < 0 or args.zarr_threads < 0:
        parser.error("--read-ahead and --zarr-threads are zero or more")
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
