#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare stock ucx_perftest, our rotating UCP baseline, and full application delivery."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import statistics
import subprocess
import tempfile
import time

import bench
import cpu_cost
import cuda_ipc
import frame_cost


def stock_point(size, frames, cpus, window, memory='host', sender_gpu='0', receiver_gpu='0', warmup=None):
    """Stock binaries; CPU includes startup/warmup and is deliberately a separate metric."""
    if memory == 'cuda':
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        options = argparse.Namespace(cpus=[cpus[0], cpus[2]], sender_gpu=sender_gpu,
                                     receiver_gpu=receiver_gpu, warmup=warmup,
                                     window=window, timeout=120)
        point = cuda_ipc.run_point('am', size, frames, options)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        point.update(program='stock ucx_perftest, matched flags, hot buffers',
                     cpu_invocation_seconds=(after.ru_utime + after.ru_stime -
                                             before.ru_utime - before.ru_stime),
                     cpu_measurement_scope='whole invocation, including setup and warmup',
                     payload_working_set_bytes=2 * size, publish_to_read_ns=None)
        return point
    port = bench.free_port()
    args = ['ucx_perftest', '-t', 'ucp_am_bw', '-s', str(size), '-n', str(frames),
            '-w', str(max(10, frames // 10)), '-p', str(port), '-O', str(window),
            '-H', '40', '-z', '-e', '-I', '-y']
    env = {**os.environ, 'UCX_MM_ERROR_HANDLING': 'y', 'UCX_KEEPALIVE_INTERVAL': '2s',
           'UCX_PROTO_INFO': 'y'}
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with tempfile.TemporaryFile(mode='w+') as log:
        server = subprocess.Popen(['taskset', '-c', str(cpus[2]), *args], env=env,
                                  stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(50):
                time.sleep(0.1)
                client_args = ['taskset', '-c', str(cpus[0]), args[0], '127.0.0.1', *args[1:]]
                done = subprocess.run(client_args, env=env, text=True, capture_output=True,
                                      timeout=120)
                final = [s.split() for s in done.stdout.splitlines()
                         if s.lstrip().startswith('Final:')]
                if final or server.poll() is not None:
                    break
            server.wait(timeout=10)
            log.seek(0)
            server_log = log.read()
            if not final or done.returncode or server.returncode:
                return dict(ok=False, failure=done.stdout + done.stderr + server_log)
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            return dict(ok=True, program='stock ucx_perftest, matched flags, hot buffers',
                        frame_bytes=size, frames=frames, warmup=max(10, frames // 10),
                        frames_per_second=float(final[-1][8]),
                        cpu_invocation_seconds=(after.ru_utime + after.ru_stime -
                                                before.ru_utime - before.ru_stime),
                        cpu_measurement_scope='whole invocation, including setup and warmup',
                        command=client_args, server_command=['taskset', '-c', str(cpus[2]), *args],
                        payload_working_set_bytes=2 * size, outstanding_sends=window,
                        protocol=bench.protocols(done.stdout + done.stderr + server_log, size),
                        publish_to_read_ns=None,
                        raw_output=done.stdout + done.stderr + server_log)
        finally:
            if server.poll() is None:
                server.kill()
                server.wait()


def raw_point(build, size, slots, frames, rate, mode, args):
    command = [str(build / 'ucp_ring'), '--bytes', str(size), '--slots', str(slots),
               '--window', str(min(slots, args.window)), '--frames', str(frames),
               '--rate', str(rate), '--sender-cpu', str(args.thread_cpus[0]),
               '--receiver-cpu', str(args.thread_cpus[2])]
    if args.memory == 'cuda':
        command += ['--cuda', '--sender-gpu', args.sender_gpu, '--receiver-gpu', args.receiver_gpu,
                    '--warmup', str(args.warmup)]
    if mode == 'wait':
        command.append('--wait')
    if args.verify:
        command.append('--verify')
    done = subprocess.run(command, text=True, capture_output=True, timeout=120,
                          env={**os.environ, **(cuda_ipc.SETTINGS if args.memory == 'cuda' else {}),
                               'UCX_PROTO_INFO': 'y'})
    lines = done.stdout.strip().splitlines()
    if done.returncode or not lines or not lines[-1].startswith('{'):
        return dict(ok=False, failure=done.stdout + done.stderr, command=command)
    result = json.loads(lines[-1])
    raw_output = done.stderr + '\n'.join(lines[:-1])
    result.update(command=command, protocol=bench.protocols(raw_output, size,
                  memory_type='cuda' if args.memory == 'cuda' else None),
                  cpu_measurement_scope='receiver measured frames; sender includes warmup',
                  raw_output=raw_output)
    return result


def physical_cpus():
    """Choose four distinct allowed physical cores, avoiding SMT siblings."""
    found, seen = [], set()
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        core = (topology.joinpath('physical_package_id').read_text().strip(),
                topology.joinpath('core_id').read_text().strip())
        if core not in seen:
            seen.add(core)
            found.append(cpu)
    return found[:4]


def confirmed_ipc(point):
    protocol = point.get('protocol', '')
    valid = bool(protocol) and all('cuda_ipc/cuda' in p for p in protocol.split('; or '))
    point['cuda_ipc_payload'] = valid
    if not valid:
        point.update(ok=False, failure='CUDA IPC payload path not confirmed; see raw_output')
    return point


def cuda_summary(samples, args, failures, skipped):
    lines = ['# CUDA IPC comparison', '',
             'Two local processes per transport path; payloads are confirmed to use CUDA IPC.',
             'Stock UCX reuses buffers. Raw UCP and tango-ucx rotate matched payload rings.',
             'Times exclude setup and warmup. GPU verification adds generation/upload and download/check work.', '',
             '| Bytes | Slots per ring | Rate requested | Variant | Median GiB/s | Worst–best GiB/s | Of stock UCX | Of rotating UCP | CPU us/frame | Runs |',
             '|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    def bandwidth(points):
        return statistics.median(p['frames_per_second'] * p['frame_bytes'] / (1 << 30) for p in points)
    for (size, slots, rate, label), points in sorted(samples.items()):
        values = [p['frames_per_second'] * size / (1 << 30) for p in points]
        median = statistics.median(values)
        stock = samples.get((size, slots, rate, 'stock:poll:batch1'))
        program, mode, _ = label.split(':')
        raw = samples.get((size, slots, rate, f'ucp_ring:{mode}:batch1'))
        of_stock = f'{100 * median / bandwidth(stock):.1f}%' if stock else '—'
        of_raw = f'{100 * median / bandwidth(raw):.1f}%' if raw else '—'
        cpu = [p['cpu_us_per_frame'] for p in points if 'cpu_us_per_frame' in p]
        cpu_text = f'{statistics.median(cpu):.2f}' if cpu else '—'
        lines.append(f'| {size} | {slots} | {rate} | {label} | {median:.2f} | '
                     f'{min(values):.2f}–{max(values):.2f} | {of_stock} | {of_raw} | '
                     f'{cpu_text} | {len(points)}/{args.runs} |')
    lines += ['', f'{failures} failed or rate-missed points; {skipped} skipped groups.', '',
              'CPU scopes differ: stock covers its full invocation and is excluded from this CPU column. '
              'Raw UCP covers both role processes; tango-ucx covers four application/progress threads.',
              'The raw baseline consumes immediately after transfer completion. The library also records '
              'consumer CUDA release events and returns credit after they complete.',
              'One-GPU results measure copies between process-owned allocations on that GPU. '
              'A GPU pair measures its supported PCIe/NVLink path.']
    return '\n'.join(lines) + '\n'


def self_test(build, memory='host', sender_gpu='0', receiver_gpu='0'):
    allowed = sorted(os.sched_getaffinity(0))
    env = {**os.environ, **(cuda_ipc.SETTINGS if memory == 'cuda' else {})}
    cuda_flags = (['--cuda', '--sender-gpu', sender_gpu, '--receiver-gpu', receiver_gpu]
                  if memory == 'cuda' else [])
    for size, slots, window in [(1024, 1, 1), (1024, 128, 128),
                                (100 * 1024, 128, 32), (1024 * 1024, 8, 8)]:
        for wait in [False, True]:
            command = [str(build / 'ucp_ring'), '--bytes', str(size), '--slots', str(slots),
                       '--window', str(window), '--frames', '1000', '--verify',
                       '--sender-cpu', str(allowed[0]), '--receiver-cpu', str(allowed[-1]), *cuda_flags]
            if wait:
                command.append('--wait')
            result = subprocess.run(command, capture_output=True, text=True, timeout=60, env=env)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            point = json.loads(result.stdout.splitlines()[-1])
            assert point['ok'] and point['mismatched'] == 0 and point['frames'] == 1000
            if memory == 'cuda':
                report = result.stdout + result.stderr
                protocol = bench.protocols(report, size, memory_type='cuda')
                assert confirmed_ipc(dict(ok=True, protocol=protocol))['ok']
                assert not confirmed_ipc(dict(ok=True, protocol=bench.protocols(
                    report.replace('from cuda', 'from other'), size, memory_type='cuda')))['ok']
                assert point['eager_frames'] == 0
                assert point['rendezvous_frames'] == point['frames'] + point['warmup']
            print(f'verified {size} bytes, {slots} slots, window {window}, wait={wait}')
    damage = subprocess.run([str(build / 'ucp_ring'), '--bytes', '129', '--slots', '8',
                             '--window', '8', '--frames', '200', '--verify', '--corrupt-every', '10',
                             '--sender-cpu', str(allowed[0]), '--receiver-cpu', str(allowed[-1]),
                             *cuda_flags], capture_output=True, text=True, timeout=60, env=env, check=True)
    point = json.loads(damage.stdout.splitlines()[-1])
    assert point['ok'] and point['mismatched'] == point['mismatched_expected'] > 0
    print('verified deliberate corruption including partial-word payloads')
    if memory == 'cuda':
        cpus = physical_cpus()
        assert len(cpus) == 4
        receiver_env = {**env, 'CUDA_VISIBLE_DEVICES': receiver_gpu}
        with bench.Publisher(build, env={**env, 'CUDA_VISIBLE_DEVICES': sender_gpu}) as publisher:
            for size, batch, corrupt in [(1, 1, 0), (129, 8, 0), (100 * 1024, 1, 0),
                                          (1024 * 1024, 8, 0), (129, 1, 10)]:
                command = [str(build / 'tango_ucx_perftest'), publisher.device, '--memory', 'cuda',
                           '-s', str(size), '-d', '32', '-b', str(batch), '-n', '200',
                           '--thread-cpus', ','.join(str(cpu) for cpu in cpus),
                           '--verify', '--json']
                if corrupt:
                    command += ['--corrupt-every', str(corrupt)]
                done = subprocess.run(command, env=receiver_env, capture_output=True, text=True,
                                      check=True, timeout=60)
                point = json.loads(done.stdout.splitlines()[-1])
                assert point['ok'] and point['memory'] == 'cuda to cuda'
                assert point['mismatched'] == point['mismatched_expected']
                assert bool(point['mismatched']) == bool(corrupt)
                assert point['depth'] == {'publisher': 32, 'subscription': 32}
                assert point['reader_cpu'] == cpus[3]
                assert point['publisher']['producer_cpu'] == cpus[1]
                assert point['subscription']['health']['topology']['cpu'] == cpus[2]
                assert point['publisher']['health_at_finish']['topology']['cpu'] == cpus[0]
                protocol = bench.protocols(done.stdout + done.stderr + publisher.said(), size,
                                           memory_type='cuda')
                assert confirmed_ipc(dict(ok=True, protocol=protocol))['ok']
                print(f'verified library CUDA {size} bytes, batch={batch}, corrupt={corrupt}')
        return 0
    paced = subprocess.run([str(build / 'ucp_ring'), '--rate', '1000', '--frames', '1000',
                            '--verify', '--wait', '--sender-cpu', str(allowed[0]),
                            '--receiver-cpu', str(allowed[-1])], capture_output=True,
                           text=True, timeout=60, check=True)
    point = json.loads(paced.stdout.splitlines()[-1])
    assert point['ok'] and point['publish_to_read_ns']['samples'] == point['frames']
    assert abs(point['frames_per_second'] / 1000 - 1) < 0.05
    print('verified pacing and latency samples')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=Path('build'))
    parser.add_argument('--memory', choices=['host', 'cuda'], default='host')
    parser.add_argument('--sender-gpu', default='0', help='physical GPU index or UUID')
    parser.add_argument('--receiver-gpu', help='physical GPU index or UUID (same as sender)')
    parser.add_argument('--case', action='append', metavar='SIZE:SLOTS',
                        help='repeat; defaults: 1K:128,4K:128,256K:128,1M:128')
    parser.add_argument('--rates', type=lambda s: [int(x) for x in s.split(',')], default=[0])
    parser.add_argument('--modes', type=lambda s: s.split(','), default=['default'])
    parser.add_argument('--batch', type=frame_cost.integers, default=[1, 8])
    parser.add_argument('--window', type=int, default=128,
                        help='raw UCP credit window and stock send window; fixed across working sets')
    parser.add_argument('--frames', type=int, help='override unpaced frame count')
    parser.add_argument('--transfer-bytes', type=bench.size, default=2 << 30)
    parser.add_argument('--seconds', type=float, default=2)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--thread-cpus', type=lambda s: [int(x) for x in s.split(',')])
    parser.add_argument('--verify', action='store_true',
                        help='fresh frame generation and full checking on raw UCP and library; omit stock')
    parser.add_argument('--raw-only', action='store_true',
                        help='isolate working-set changes in raw UCP, without other benchmarks')
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--out', help='new JSONL output')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    args.receiver_gpu = args.receiver_gpu or args.sender_gpu
    if args.memory == 'cuda' and not args.thread_cpus:
        args.thread_cpus = physical_cpus()
    build = args.build.resolve()
    if args.self_test:
        return self_test(build, args.memory, args.sender_gpu, args.receiver_gpu)
    if (not args.out or args.runs <= 0 or args.seconds <= 0 or args.window <= 0
            or not args.rates or min(args.rates) < 0 or max(args.rates) > 1_000_000_000
            or not set(args.modes) <= {'default', 'wait'} or not args.modes
            or args.transfer_bytes <= 0 or args.frames is not None and args.frames <= 0):
        parser.error('output and positive sizes/counts required; modes: default,wait; rates: 0..1e9')
    if (not args.thread_cpus or len(args.thread_cpus) != 4
            or len(set(args.thread_cpus)) != 4
            or not set(args.thread_cpus) <= os.sched_getaffinity(0)):
        parser.error('--thread-cpus needs four distinct allowed physical CPUs')
    cases = []
    for case in args.case or ['1K:128', '4K:128', '256K:128', '1M:128']:
        size, separator, slots = case.partition(':')
        if not separator or int(slots) <= 0 or bench.size(size) <= 0:
            parser.error('--case needs positive SIZE:SLOTS')
        cases.append((bench.size(size), int(slots)))
    if any(batch > slots for _, slots in cases for batch in args.batch):
        parser.error('batch must not exceed any ring depth')
    if args.memory == 'cuda' and not args.raw_only:
        if any(slots < max(3, 2 * max(args.batch)) or slots > min(1024, args.window) or
               any(slots % batch for batch in args.batch) for _, slots in cases):
            parser.error('CUDA comparisons need two full batches per ring, slots <= window and 1024, '
                         'and depths divisible by each batch size')
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_suffix('.md')
    if args.memory == 'cuda' and (output == report_path or report_path.exists()):
        parser.error('CUDA output needs a separate, new Markdown report path (derived from --out)')
    args.tier = None
    samples = {}
    failed = 0
    skipped_count = 0
    with output.open('x') as out, ExitStack() as stack:
        meta = bench.describe_host(args)
        files = [build / name for name in ['ucp_ring', 'tango_ucx_perf_publisher', 'tango_ucx_perftest']]
        files.append(Path(shutil.which('ucx_perftest')))
        meta.update(metric='inverse throughput and CPU/frame; distinct benchmark semantics',
                    memory=args.memory,
                    cases=cases, rates=args.rates, modes=args.modes, batch=args.batch,
                    thread_cpus=args.thread_cpus, raw_window=args.window, seed=args.seed,
                    binary_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                   for name in ['compare_ucx.py', 'cpu_cost.py', 'frame_cost.py',
                                                'bench.py', 'cuda_ipc.py', 'src/ucp_ring.cpp',
                                                'src/publisher.cpp', 'src/perftest.cpp', 'src/common.h', 'src/memory.h']},
                    differences=['stock reuses one buffer and has no application release credits',
                                 'raw UCP combines producer/progress and progress/consumer: two hot threads',
                                 'library has separate application/progress threads: four hot threads',
                                 'raw flow window fixed; library credit capacity follows ring depth',
                                 'stock CPU spans whole invocation; not compared with measured-window CPU'])
        if args.memory == 'cuda':
            meta.update(sender_gpu=cuda_ipc.gpu_info(args.sender_gpu),
                        receiver_gpu=cuda_ipc.gpu_info(args.receiver_gpu),
                        settings=cuda_ipc.SETTINGS,
                        cuda_verification='fresh host generation/upload and full device download/check; stock omitted')
            args.sender_gpu, args.receiver_gpu = meta['sender_gpu']['uuid'], meta['receiver_gpu']['uuid']
            meta['ucx'] = subprocess.run(['ucx_info', '-v'], text=True, capture_output=True,
                                         check=True, timeout=10).stdout
            cache = build.joinpath('CMakeCache.txt').read_text()
            package_dir = next(line.split('=', 1)[1] for line in cache.splitlines()
                               if line.startswith('tango-ucx_DIR:'))
            marker = Path(package_dir).parents[2] / '.checkout'
            meta['library_source'] = marker.read_text().strip() if marker.exists() else 'unknown'
            if marker.exists():
                library_root = Path(meta['library_source'])
                meta['library_source_sha256'] = {str(p.relative_to(library_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for directory in ('src', 'include') for p in sorted(library_root.joinpath(directory).rglob('*'))
                    if p.is_file() and p.suffix in {'.h', '.cpp'}}
        out.write(json.dumps({'meta': meta}) + '\n')
        publishers = {}
        for mode in args.modes if not args.raw_only else []:
            with cpu_cost.idle_mode(mode):
                publisher_env = ({**os.environ, **cuda_ipc.SETTINGS, 'CUDA_VISIBLE_DEVICES': args.sender_gpu}
                                 if args.memory == 'cuda' else None)
                publishers[mode] = stack.enter_context(bench.Publisher(build, env=publisher_env))
        rng = random.Random(args.seed)
        for run in range(args.runs):
            groups = [(size, slots, rate) for size, slots in cases for rate in args.rates]
            rng.shuffle(groups)
            for size, slots, rate in groups:
                if args.memory == 'cuda':
                    sender, receiver = meta['sender_gpu'], meta['receiver_gpu']
                    buffers = 2 if sender['uuid'] == receiver['uuid'] else 1
                    if buffers * size * slots > min(sender['free_bytes'], receiver['free_bytes']) // 2:
                        skipped_count += 1
                        skipped = dict(skipped=True, frame_bytes=size, case_slots=slots, run=run,
                                       failure='payload rings exceed half the free GPU memory')
                        out.write(json.dumps(skipped) + '\n')
                        out.flush()
                        print(f'SKIPPED {size} bytes x {slots} slots: {skipped["failure"]}', flush=True)
                        continue
                frames = (max(2 * slots, int(rate * args.seconds)) if rate else
                          args.frames or max(1000, min(2_000_000, args.transfer_bytes // size)))
                if args.memory == 'cuda':
                    batch_multiple = math.lcm(*args.batch)
                    args.warmup = ((max(frames // 10, 2 * slots) + batch_multiple - 1) //
                                   batch_multiple * batch_multiple)
                variants = [('ucp_ring', mode, 1) for mode in args.modes]
                if not args.raw_only:
                    variants += [('library', mode, batch) for mode in args.modes for batch in args.batch]
                if not rate and not args.verify and not args.raw_only:
                    variants.append(('stock', 'poll', 1))
                rng.shuffle(variants)
                for program, mode, batch in variants:
                    before = cpu_cost.thermal_snapshot()
                    if program == 'stock':
                        point = stock_point(size, frames, args.thread_cpus,
                                            min(slots, args.window) if args.memory == 'cuda' else args.window,
                                            args.memory, args.sender_gpu, args.receiver_gpu,
                                            getattr(args, 'warmup', None))
                    elif program == 'ucp_ring':
                        point = raw_point(build, size, slots, frames, rate, mode, args)
                    else:
                        library_args = argparse.Namespace(**vars(args))
                        library_args.frames, library_args.rate, library_args.batch = frames, rate, batch
                        if args.memory == 'cuda':
                            library_args.process_env = {**os.environ, **cuda_ipc.SETTINGS,
                                                        'CUDA_VISIBLE_DEVICES': args.receiver_gpu}
                        with cpu_cost.idle_mode(mode):
                            point = frame_cost.pinned_point(build, publishers[mode], size, slots, library_args)
                        if point.get('ok'):
                            cpu_cost.cpu_cost(point)
                            point['payload_working_set_bytes'] = size * (
                                point['depth']['publisher'] + point['depth']['subscription'])
                            point['cpu_measurement_scope'] = 'subscriber measured frames; publisher includes warmup'
                            if args.memory == 'cuda':
                                point['protocol'] = bench.protocols(point['raw_output'], size, memory_type='cuda')
                                if point['depth']['publisher'] != slots or point['depth']['subscription'] != slots:
                                    point.update(ok=False, failure='library rings do not match the requested raw working set')
                    if args.memory == 'cuda' and point.get('ok'):
                        confirmed_ipc(point)
                    label = f'{program}:{mode}:batch{batch}'
                    point.update(variant=label, run=run, case_slots=slots, frame_bytes=size,
                                 rate_requested=rate, thermal_before=before,
                                 thermal_after=cpu_cost.thermal_snapshot())
                    if point.get('ok'):
                        point['ns_per_frame'] = 1e9 / point['frames_per_second']
                        point['rate_met'] = not rate or abs(point['frames_per_second'] / rate - 1) < 0.05
                        failed += not point['rate_met']
                        samples.setdefault((size, slots, rate, label), []).append(point)
                        cpu = f"{point['cpu_us_per_frame']:.2f}" if 'cpu_us_per_frame' in point else 'n/a'
                        print(f'{run + 1}/{args.runs} {size:8} slots={slots:5} rate={rate:6} '
                              f'{label:26} {point["ns_per_frame"]:9.1f} ns/frame CPU={cpu} us/frame', flush=True)
                    else:
                        failed += 1
                        print(f'FAILED {label}: {point.get("failure", point.get("outcome"))}', flush=True)
                    out.write(json.dumps(point) + '\n')
                    out.flush()
    print('\nbytes slots rate variant                       ns/frame   CPU us/frame p50 us p99 us')
    for (size, slots, rate, label), points in sorted(samples.items()):
        ns = statistics.median(p['ns_per_frame'] for p in points)
        cpus = [p['cpu_us_per_frame'] for p in points if 'cpu_us_per_frame' in p]
        cpu = f'{statistics.median(cpus):.2f}' if cpus else 'n/a'
        latencies = [p['publish_to_read_ns'] for p in points if p.get('publish_to_read_ns')]
        p50 = statistics.median(p['p50'] for p in latencies) / 1000 if latencies else 0
        p99 = statistics.median(p['p99'] for p in latencies) / 1000 if latencies else 0
        print(f'{size:7} {slots:5} {rate:5} {label:27} {ns:9.1f} {cpu:>12} {p50:6.1f} {p99:6.1f}')
    print(f'{failed} failed or rate-missed points; saved {output}')
    if args.memory == 'cuda':
        report = cuda_summary(samples, args, failed, skipped_count)
        with report_path.open('x') as report_file:
            report_file.write(report)
        print('\n' + report)
        print(f'Report saved {report_path}')
    return int(failed > 0)


if __name__ == '__main__':
    raise SystemExit(main())
