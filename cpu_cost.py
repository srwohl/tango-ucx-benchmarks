#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare CPU per frame and delivery latency at matched, paced rates on one host."""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import statistics

import bench
import frame_cost


@contextmanager
def idle_mode(mode):
    previous = os.environ.pop('TANGO_UCX_IDLE', None)
    if mode != 'default':
        os.environ['TANGO_UCX_IDLE'] = mode
    try:
        yield
    finally:
        os.environ.pop('TANGO_UCX_IDLE', None)
        if previous is not None:
            os.environ['TANGO_UCX_IDLE'] = previous


def read_int(path):
    try:
        return int(Path(path).read_text())
    except (OSError, ValueError):
        return None


def thermal_snapshot():
    result = {}
    for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:
            name = (zone / 'type').read_text().strip()
        except OSError:
            continue
        result[name] = read_int(zone / 'temp')
    counters = Path('/sys/devices/system/cpu/cpu0/thermal_throttle')
    for name in ('package_throttle_count', 'package_throttle_total_time_ms'):
        result[name] = read_int(counters / name)
    return result


def cpu_cost(point):
    sub = point['subscription']['cpu_seconds']
    pub = point['publisher']['cpu_seconds']
    frames = point['frames']
    published = point['publisher']['published']
    stages = {name: sub.get(name, 0) * 1e6 / frames
              for name in ('tango-ucx-sub', 'tango_ucx_perft')}
    stages.update({name: pub.get(name, 0) * 1e6 / published
                   for name in ('tango-ucx-pub', 'perf-frames')})
    point['cpu_us_per_frame_by_thread'] = stages
    point['cpu_us_per_frame'] = sum(stages.values())
    point['cpu_cores'] = sum(stages.values()) * point['frames_per_second'] / 1e6
    point['progress_cpu_cores'] = (stages['tango-ucx-pub'] + stages['tango-ucx-sub']) * point[
        'frames_per_second'] / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', action='append', required=True, metavar='LABEL=PATH,MODE',
                        help='MODE is default, wait or spin; repeat to compare builds/settings')
    parser.add_argument('--rates', type=lambda s: [int(x) for x in s.split(',')],
                        default=[0, 1000, 10000, 50000], help='frames/s; 0 is unpaced')
    parser.add_argument('--seconds', type=float, default=2, help='measured seconds per paced point')
    parser.add_argument('--unpaced-frames', type=int, default=1_000_000)
    parser.add_argument('--size', type=bench.size, default=1024)
    parser.add_argument('--depth', type=int, default=128)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--thread-cpus', type=lambda s: [int(x) for x in s.split(',')],
                        required=True, help='publisher loop, producer, subscriber loop, reader')
    parser.add_argument('--out', required=True, help='new JSONL output')
    args = parser.parse_args()
    if (args.seconds <= 0 or args.unpaced_frames <= 0 or args.size <= 0 or args.runs <= 0
            or not args.rates or min(args.rates) < 0 or max(args.rates) > 1_000_000_000
            or not 0 < args.batch <= args.depth):
        parser.error('sizes and counts must be positive; rates may also be zero')
    if (len(args.thread_cpus) != 4 or len(set(args.thread_cpus)) != 4
            or not set(args.thread_cpus) <= os.sched_getaffinity(0)):
        parser.error('--thread-cpus needs four distinct CPUs in the inherited affinity')
    variants = {}
    for item in args.variant:
        label, equals, rest = item.partition('=')
        path, comma, mode = rest.rpartition(',')
        if not equals or not comma or not label or label in variants or mode not in (
                'default', 'wait', 'spin'):
            parser.error('--variant needs a unique LABEL=PATH,MODE')
        variants[label] = (Path(path).resolve(), mode)
    args.tier = None
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = {}
    failed = 0
    with output.open('x') as out, ExitStack() as stack:
        meta = bench.describe_host(args)
        meta.update(metric='CPU at matched rates; same-host publish-to-read latency',
                    thread_cpus=args.thread_cpus, seed=args.seed,
                    configuration=dict(size=args.size, depth=args.depth, batch=args.batch,
                                       rates=args.rates, seconds=args.seconds,
                                       unpaced_frames=args.unpaced_frames, runs=args.runs),
                    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    variants={label: dict(build=str(path), mode=mode, binary_sha256={
                        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                        for name in ('tango_ucx_perf_publisher', 'tango_ucx_perftest')})
                        for label, (path, mode) in variants.items()})
        out.write(json.dumps({'meta': meta}) + '\n')
        publishers = {}
        for label, (path, mode) in variants.items():
            with idle_mode(mode):
                publishers[label] = stack.enter_context(bench.Publisher(path))
        rng = random.Random(args.seed)
        for run in range(args.runs):
            rates = args.rates.copy()
            rng.shuffle(rates)
            for rate in rates:
                args.rate = rate
                args.frames = (max(2 * args.depth, int(rate * args.seconds))
                               if rate else args.unpaced_frames)
                labels = list(variants)
                rng.shuffle(labels)
                for label in labels:
                    path, mode = variants[label]
                    before = thermal_snapshot()
                    with idle_mode(mode):
                        point = frame_cost.pinned_point(path, publishers[label], args.size,
                                                        args.depth, args)
                    point.update(variant=label, idle_mode=mode, run=run,
                                 thermal_before=before, thermal_after=thermal_snapshot())
                    if point.get('ok'):
                        cpu_cost(point)
                        point['rate_met'] = not rate or abs(point['frames_per_second'] / rate - 1) < 0.05
                        samples.setdefault((rate, label), []).append(point)
                        latency = point['publish_to_read_ns']
                        p99 = f"{latency['p99'] / 1000:.1f} us" if latency else 'unpaced'
                        print(f'{run + 1}/{args.runs} {label:10} rate={rate:6} '
                              f"actual={point['frames_per_second']:10.1f} "
                              f"CPU={point['cpu_cores']:5.2f} cores "
                              f"CPU/frame={point['cpu_us_per_frame']:8.2f} us p99={p99}", flush=True)
                        failed += not point['rate_met']
                    else:
                        failed += 1
                        print(f"FAILED {label} rate={rate}: {point.get('failure') or point.get('outcome')}",
                              flush=True)
                    out.write(json.dumps(point) + '\n')
                    out.flush()
    print('\nrate   variant       CPU cores   CPU us/frame  p50 us  p99 us')
    for (rate, label), points in sorted(samples.items()):
        median = lambda name: statistics.median(p[name] for p in points)
        latencies = [p['publish_to_read_ns'] for p in points if p['publish_to_read_ns']]
        p50 = statistics.median(p['p50'] for p in latencies) / 1000 if latencies else 0
        p99 = statistics.median(p['p99'] for p in latencies) / 1000 if latencies else 0
        print(f'{rate:6} {label:12} {median("cpu_cores"):9.2f} '
              f'{median("cpu_us_per_frame"):14.2f} {p50:7.1f} {p99:7.1f}')
    print(f'{failed} failed or rate-missed points; saved {output}')
    return int(failed > 0)


if __name__ == '__main__':
    raise SystemExit(main())
