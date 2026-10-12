#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Measure inverse throughput across frame sizes, ring depths and batches.

Build variants run in shuffled, interleaved order so a later build does not systematically
benefit from a different CPU temperature or host load. This measures sustained ns/frame,
not arrival latency. Each JSON line retains the full benchmark result.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import bench


def integers(text):
    values = [int(value) for value in text.split(',')]
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError('counts must be positive')
    return values


def pinned_point(build, publisher, size, depth, args):
    """Place the four hot threads of existing benchmark executables, without changing them.

    Setup creates/replaces the publisher loop, so discover its thread by the library's name.
    Keep checking during warmup; only set affinity once for each thread. Monitoring continues
    during the run and is included in both variants' measurements.
    """
    pub_cpu, producer_cpu, sub_cpu, reader_cpu = args.thread_cpus
    command = [str(build / 'tango_ucx_perftest'), publisher.device, '-t', 'bw',
               '-s', str(size), '-n', str(args.frames), '-d', str(depth),
               '-b', str(args.batch), '--json']
    if getattr(args, 'memory', 'host') == 'cuda':
        command += ['--memory', 'cuda', '--thread-cpus', ','.join(str(cpu) for cpu in args.thread_cpus)]
    if getattr(args, 'warmup', None) is not None:
        command += ['-w', str(args.warmup)]
    if getattr(args, 'rate', 0):
        command += ['--rate', str(args.rate)]
    if getattr(args, 'verify', False):
        command.append('--verify')
    placed = {}
    publisher.log.seek(0, os.SEEK_END)
    log_start = publisher.log.tell()

    def pin(tid, cpu):
        if tid not in placed:
            try:
                os.sched_setaffinity(tid, {cpu})
            except ProcessLookupError:
                return  # a setup trial may already have replaced this loop
            placed[tid] = cpu

    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, env=getattr(args, 'process_env', None)) as process:
        # CUDA executables set the application affinity after creating the explicitly
        # placed progress workers. Pinning the main thread before setup would shrink
        # the affinity those workers inherit and invalidate their CPU selections.
        if getattr(args, 'memory', 'host') != 'cuda':
            pin(process.pid, reader_cpu)
        while True:
            for pid, names in ((publisher.process.pid,
                                {'tango-ucx-pub': pub_cpu, 'perf-frames': producer_cpu}),
                               (process.pid, {'tango-ucx-sub': sub_cpu})):
                try:
                    tasks = list(Path(f'/proc/{pid}/task').iterdir())
                except (FileNotFoundError, ProcessLookupError):
                    continue
                for task in tasks:
                    try:
                        name = (task / 'comm').read_text().strip()
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                    if name in names:
                        pin(int(task.name), names[name])
            try:
                stdout, stderr = process.communicate(timeout=0.01)
                break
            except subprocess.TimeoutExpired:
                pass
    lines = stdout.strip().splitlines()
    if not lines or not lines[-1].startswith('{'):
        return dict(ok=False, frame_bytes=size, failure=stderr.strip() or 'no result')
    result = json.loads(lines[-1])
    result['thread_affinity'] = {str(tid): cpu for tid, cpu in placed.items()}
    result['command'] = command
    publisher.log.seek(log_start)
    result['raw_output'] = stderr + '\n'.join(lines[:-1]) + publisher.log.read()
    if process.returncode:
        result['ok'] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='append', required=True, metavar='LABEL=PATH',
                        help='saved executables for each variant; repeat to compare builds')
    parser.add_argument('--sizes', type=bench.sizes, default=bench.sizes('64,1K,4K,16K'))
    parser.add_argument('--depth', type=integers, default=[32, 128])
    parser.add_argument('--batch', type=integers, default=[1, 8, 32])
    parser.add_argument('--frames', type=int, default=1_000_000)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--thread-cpus', type=lambda s: [int(x) for x in s.split(',')],
                        help='four distinct CPUs: publisher loop, producer, subscriber loop, reader')
    parser.add_argument('--reference', action='store_true',
                        help='UCX with peer handling, registered memory, wakeup and eager copy')
    parser.add_argument('--out', required=True, help='new JSONL file; refuses to overwrite')
    parser.add_argument('--resume', action='store_true', help='continue an interrupted output file')
    args = parser.parse_args()
    if args.frames <= 0 or args.runs <= 0 or not args.sizes or min(args.sizes) <= 0:
        parser.error('sizes, frames and runs must be positive')
    if max(args.batch) > min(args.depth):
        parser.error('every batch must fit in every ring depth')
    if args.thread_cpus and (len(args.thread_cpus) != 4 or len(set(args.thread_cpus)) != 4
                            or not set(args.thread_cpus) <= os.sched_getaffinity(0)):
        parser.error('--thread-cpus needs four distinct CPUs in the inherited affinity')
    builds = {}
    for item in args.build:
        label, separator, path = item.partition('=')
        if not separator or not label or label in builds:
            parser.error('--build needs a unique LABEL=PATH')
        builds[label] = Path(path).resolve()
    fingerprints = {}
    for label, path in builds.items():
        fingerprints[label] = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                               for name in ('tango_ucx_perf_publisher', 'tango_ucx_perftest')}
    # bench.run_point takes the same arguments as the ordinary sweep. A fixed frame count
    # makes small-frame measurements long enough, independent of payload size.
    args.min_frames = args.max_frames = args.frames
    args.bytes = args.frames * max(args.sizes)
    args.verify = False
    args.corrupt_every = 0
    args.tier = None
    args.memory_share = 0.5
    # The eager AM callback in tango-ucx copies the payload into its receive ring. Ask the
    # reference to copy too; the bandwidth sweep's reference normally omits that copy.
    reference_name = 'ucx_perftest -e -y'
    flags, settings = bench.REFERENCES['ucx_perftest -e']
    bench.REFERENCES[reference_name] = (flags + ['-y'], settings)
    cases = [[(label, size, depth, batch) for label in builds]
             for size in args.sizes for depth in args.depth for batch in args.batch]
    if args.reference:
        cases += [[(reference_name, size, None, None)] for size in args.sizes]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = {}
    completed = set()
    if args.resume:
        records = [json.loads(line) for line in output.read_text().splitlines()]
        prior = records[0]['meta']
        for name, expected in (('binary_sha256', fingerprints), ('seed', args.seed),
                               ('thread_cpus', args.thread_cpus),
                               ('affinity', sorted(os.sched_getaffinity(0)))):
            if prior.get(name) != expected:
                parser.error(f'cannot resume with different {name}')
        if 'configuration' in prior and prior['configuration'] != dict(
                frames=args.frames, sizes=args.sizes, depths=args.depth, batches=args.batch,
                reference=args.reference):
            parser.error('cannot resume with different measurement options')
        valid = {case for group in cases for case in group}
        for point in records:
            if not point.get('ok'):
                continue
            key = (point['variant'], point['frame_bytes'], point['depth_asked'],
                   point['batch_asked'])
            # A batch can cross the warmup boundary, shortening the measured frame count.
            # New files record the requested count in their configuration; legacy files in
            # this experiment used warmup counts divisible by every batch size.
            legacy_frames_differ = 'configuration' not in prior and point['frames'] != args.frames
            if key not in valid or point['run'] >= args.runs or legacy_frames_differ:
                parser.error('saved measurements do not match the requested cases or frames')
            completed.add((*key, point['run']))
            samples.setdefault(key, []).append(point['ns_per_frame'])
    failed = 0
    start = time.monotonic()
    with output.open('a' if args.resume else 'x') as out, ExitStack() as stack:
        meta = bench.describe_host(args)
        meta.update(metric='inverse throughput, not per-frame latency', seed=args.seed,
                    affinity=sorted(os.sched_getaffinity(0)),
                    thread_cpus=args.thread_cpus,
                    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    configuration=dict(frames=args.frames, sizes=args.sizes, depths=args.depth,
                                       batches=args.batch, reference=args.reference),
                    resumed=args.resume,
                    builds={label: str(path) for label, path in builds.items()},
                    binary_sha256=fingerprints)
        out.write(json.dumps({'meta': meta}) + '\n')
        out.flush()
        publishers = {label: stack.enter_context(bench.Publisher(path))
                      for label, path in builds.items()}
        rng = random.Random(args.seed)
        for run in range(args.runs):
            order = cases.copy()
            rng.shuffle(order)
            for group in order:
                group = group.copy()
                rng.shuffle(group)
                for label, size, depth, batch in group:
                    if (label, size, depth, batch, run) in completed:
                        continue
                    failed += measure(label, size, depth, batch, run, args, builds,
                                      publishers, out, samples)
    print('\nvariant        bytes depth batch   median ns/frame       min       max')
    for (label, size, depth, batch), values in sorted(samples.items(), key=str):
        print(f'{label:14} {size:6} {str(depth):>5} {str(batch):>5} '
              f'{statistics.median(values):17.1f} {min(values):9.1f} {max(values):9.1f}')
    print(f'{time.monotonic() - start:.1f} s; {failed} unsuccessful points; saved {output}')
    return int(failed > 0)


def measure(label, size, depth, batch, run, args, builds, publishers, out, samples):
    args.batch = batch
    why = bench.left_out(size, depth, args)
    if why:
        point = dict(skipped=why, frame_bytes=size)
    elif depth is None:
        point = bench.run_reference(label, size, args)
    elif args.thread_cpus:
        point = pinned_point(builds[label], publishers[label], size, depth, args)
    else:
        point = bench.run_point(builds[label], publishers[label].device,
                               size, depth, args)
    point.update(variant=label, run=run, depth_asked=depth, batch_asked=batch,
                 frames_requested=args.frames)
    if point.get('ok'):
        ns = 1e9 / point['frames_per_second']
        point['ns_per_frame'] = ns
        samples.setdefault((label, size, depth, batch), []).append(ns)
        print(f'{run + 1}/{args.runs} {label:12} {size:6} bytes '
              f'depth={str(depth):3} batch={str(batch):4} {ns:9.1f} ns/frame',
              flush=True)
    else:
        print(f'{run + 1}/{args.runs} {label}: '
              f"{point.get('skipped') or point.get('failure') or point.get('outcome')}",
              file=sys.stderr)
    out.write(json.dumps(point) + '\n')
    out.flush()
    return int(not point.get('ok'))


if __name__ == '__main__':
    sys.exit(main())
