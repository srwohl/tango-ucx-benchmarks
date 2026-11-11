#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Write a NeXus image stack for the publisher to replay, or test the replay with several.

    hdf5_sample.py stack.h5 --frames 200 --shape 2208,3216 --dtype uint16
    hdf5_sample.py stack.h5 --frames 64 --shape 1024,1024 --layout chunked --compression gzip
    hdf5_sample.py --self-test --build build

A file has /entry/data/data as [frame, ...image], reached through the NeXus default/signal chain,
and /entry/instrument/detector/acquisition/{exposure_time, latency_time} when --rate is given.
Each frame differs from the others, so a frame from the wrong position is detected.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent


def frames_of(count, shape, dtype):
    """Frame k holds a ramp offset by k, wrapped into the type's range."""
    dtype = np.dtype(dtype)
    pixels = int(np.prod(shape))
    ramp = np.arange(pixels, dtype=np.int64).reshape(shape)
    for k in range(count):
        values = ramp * 7 + k * 131
        if dtype.kind in "ui":
            yield (values % (1 << min(dtype.itemsize * 8 - (dtype.kind == "i"), 62))).astype(dtype)
        else:
            yield (values / 3.0).astype(dtype)


def write(path, frames, shape, dtype, layout="contiguous", compression=None, rate=None,
          nexus=True, dataset="/entry/data/data"):
    """A stack of `frames` frames; a single frame is written as a rank-two dataset."""
    shape = tuple(shape)
    full = shape if frames == 1 and len(shape) == 2 and not nexus else (frames, *shape)
    options = {}
    if layout == "chunked" or compression:
        options["chunks"] = (1, *shape) if len(full) > len(shape) else shape
    if compression:
        options["compression"] = compression
    # With a rate the stack is laid out as a detector's often is: the data and its acquisition
    # times under the instrument, and the NeXus signal a soft link to that data.
    stored = "/entry/instrument/detector/data" if rate else dataset
    with h5py.File(path, "w") as f:
        data = f.create_dataset(stored, full, dtype=dtype, **options)
        for k, frame in enumerate(frames_of(frames, shape, dtype)):
            if len(full) > len(shape):
                data[k] = frame
            else:
                data[...] = frame
        if stored != dataset:
            f[dataset] = h5py.SoftLink(stored)
        if nexus:
            entry, group = dataset.rsplit("/", 2)[0], dataset.rsplit("/", 1)[0]
            f.attrs["default"] = entry.lstrip("/")
            f[entry].attrs["NX_class"] = "NXentry"
            f[entry].attrs["default"] = group.rsplit("/", 1)[1]
            f[group].attrs["NX_class"] = "NXdata"
            f[group].attrs["signal"] = dataset.rsplit("/", 1)[1]
        if rate:
            acquisition = f.create_group("/entry/instrument/detector/acquisition")
            acquisition["exposure_time"] = 0.8 / rate
            acquisition["latency_time"] = 0.2 / rate


def perftest(build, device, *options):
    done = subprocess.run([str(build / "tango_ucx_perftest"), device, *options, "--json"],
                          capture_output=True, text=True, timeout=120)
    lines = done.stdout.strip().splitlines()
    if not lines or not lines[-1].startswith("{"):
        raise AssertionError(f"perftest {' '.join(options)}: {done.stderr.strip()}")
    return json.loads(lines[-1])


def self_test(build):
    """Replays files of each kind the reader handles, and fails on the first that is wrong."""
    sys.path.insert(0, str(ROOT))
    from bench import Publisher

    cases = [
        # name, file options, what Describe must find, replay options
        ("nexus-u16-contiguous", dict(frames=40, shape=(48, 64), dtype="<u2", rate=500),
         dict(element="u16", shape=[48, 64], frames=40, layout="contiguous", nominal_rate=500),
         []),
        ("big-endian-i32-gzip", dict(frames=12, shape=(32, 40), dtype=">i4", layout="chunked",
                                     compression="gzip"),
         dict(element="i32", shape=[32, 40], frames=12, layout="chunked", chunk=[1, 32, 40],
              filters=["deflate"]), []),
        ("named-f32-single-frame", dict(frames=1, shape=(256, 300), dtype="<f4", nexus=False,
                                        dataset="/images"),
         dict(element="f32", shape=[256, 300], frames=1, nominal_rate=None),
         ["--hdf5-dataset", "/images"]),
        # A frame over the 8 KiB eager limit: each one in fragments or by rendezvous.
        ("nexus-u8-large", dict(frames=8, shape=(512, 1024), dtype="u1"),
         dict(element="u8", shape=[512, 1024], frame_bytes=512 * 1024), []),
    ]
    failures = 0
    with tempfile.TemporaryDirectory() as scratch, Publisher(build) as publisher:
        for name, file_options, expected, replay in cases:
            path = Path(scratch) / f"{name}.h5"
            write(path, **file_options)
            source = ["--hdf5", str(path), *replay]
            try:
                found = perftest(build, publisher.device, *source, "--describe")
                for key, value in expected.items():
                    if key == "nominal_rate" and value is not None:
                        assert abs(found[key] - value) < 1e-6 * value, (key, found[key], value)
                    else:
                        assert found[key] == value, (key, found[key], value)
                # More frames than the file has: the reader starts it again, in order.
                run = perftest(build, publisher.device, *source, "-n", "100", "-d", "4",
                               "--verify")
                assert run["ok"] and run["source"]["kind"] == "hdf5", run
                assert run["frame_bytes"] == found["frame_bytes"], run["frame_bytes"]
                assert run["publisher"]["seconds_reading"] > 0, run["publisher"]
                damaged = perftest(build, publisher.device, *source, "-n", "100", "-d", "4",
                                   "--verify", "--corrupt-every", "10")
                assert damaged["ok"] and damaged["mismatched"] == damaged["mismatched_expected"] > 0, \
                    damaged
                assert run["read_ahead"] == 8 and run["publisher"]["reader"]["frames_read"] == 110, \
                    run["publisher"]
                # Read inline, one frame ahead, more ahead than the ring is deep, and paced.
                for variant in (["--read-ahead", "0"], ["--read-ahead", "1"],
                                ["--read-ahead", "40", "-d", "2"], ["--rate", "2000"]):
                    other = perftest(build, publisher.device, *source, "-n", "60", "-d", "4",
                                     "--verify", *variant)
                    assert other["ok"], (variant, other)
                    assert (other["publisher"]["reader"] is None) == (variant[-1] == "0"), variant
                print(f"ok    {name}: {found['element']} {found['shape']}, "
                      f"{run['frames']} frames checked, {damaged['mismatched']} damaged found")
            except AssertionError as error:
                failures += 1
                print(f"FAILED {name}: {error}")
        # A generated run after a file's: the publisher no longer reads it.
        run = perftest(build, publisher.device, "-s", "64K", "-n", "100", "-d", "4", "--verify")
        if run["ok"] and run["source"]["kind"] == "generated":
            print("ok    generated frames after HDF5 ones")
        else:
            failures += 1
            print(f"FAILED generated after HDF5: {run}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", nargs="?", type=Path, help="the file to write; it must be new")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--shape", default="512,512", help="image axes, slowest first (512,512)")
    parser.add_argument("--dtype", default="uint16", help="a NumPy type, such as uint16 or >i4")
    parser.add_argument("--layout", choices=["contiguous", "chunked"], default="contiguous")
    parser.add_argument("--compression", choices=["gzip", "lzf"],
                        help="filters need chunks: a chunk is one frame")
    parser.add_argument("--rate", type=float, help="record exposure and latency times for this "
                                                   "frame rate")
    parser.add_argument("--self-test", action="store_true",
                        help="replay sample files through the publisher and check every frame")
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    if args.self_test:
        return self_test(args.build.resolve())
    if not args.output:
        parser.error("name the file to write, or use --self-test")
    if args.output.exists():
        parser.error(f"{args.output} exists")
    shape = [int(n) for n in args.shape.split(",")]
    if args.frames < 1 or not shape or min(shape) < 1:
        parser.error("frames and image axes are positive")
    write(args.output, args.frames, shape, args.dtype, args.layout, args.compression, args.rate)
    bytes_per_frame = int(np.prod(shape)) * np.dtype(args.dtype).itemsize
    print(f"{args.output}: {args.frames} frames of {args.shape} {np.dtype(args.dtype).name}, "
          f"{bytes_per_frame} bytes each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
