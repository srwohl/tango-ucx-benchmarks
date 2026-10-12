#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Local, two-process CUDA IPC references: direct UCT puts and UCP active messages.

These reuse buffers and do not measure tango-ucx's GPU rings or application delivery.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import socket
import statistics
import subprocess
import tempfile
import time

import bench


TESTS = {
    'uct': ['-t', 'put_bw', '-x', 'cuda_ipc', '-d', 'cuda', '-D', 'zcopy'],
    'am': ['-t', 'ucp_am_bw', '-H', '40', '-z', '-e', '-I'],
}
SETTINGS = {'UCX_TLS': 'sm,cuda_copy,cuda_ipc', 'UCX_PROTO_INFO': 'y', 'UCX_LOG_FILE': 'stderr',
            'UCX_MM_ERROR_HANDLING': 'y', 'UCX_KEEPALIVE_INTERVAL': '2s'}


def positive_sizes(text):
    # bench.sizes doubles its starting value, so reject zero before expanding a range.
    for part in text.split(','):
        first, dots, last = part.partition('..')
        if bench.size(first) <= 0 or (dots and bench.size(last) <= 0):
            raise argparse.ArgumentTypeError('frame sizes must be positive')
    return bench.sizes(text)


def commands(test, size, frames, args):
    common = ['ucx_perftest', *TESTS[test], '-m', 'cuda', '-s', str(size),
              '-n', str(frames), '-w', str(args.warmup), '-O', str(args.window),
              '-p', str(bench.free_port())]
    return (common + ['-c', str(args.cpus[1])],
            [common[0], '127.0.0.1', *common[1:], '-c', str(args.cpus[0])])


def run_point(test, size, frames, args):
    server_command, client_command = commands(test, size, frames, args)
    env = {**os.environ, **SETTINGS}
    sender_env = {**env, 'CUDA_VISIBLE_DEVICES': args.sender_gpu}
    receiver_env = {**env, 'CUDA_VISIBLE_DEVICES': args.receiver_gpu}
    point = dict(program=f'ucx_perftest cuda {test}', test=test, ok=False,
                 memory='cuda to cuda', frame_bytes=size, frames=frames,
                 warmup=args.warmup, outstanding_sends=args.window,
                 sender_gpu=args.sender_gpu, receiver_gpu=args.receiver_gpu,
                 command=client_command, server_command=server_command,
                 settings=SETTINGS, payload_verified=False)
    with tempfile.TemporaryFile(mode='w+') as log:
        server = subprocess.Popen(server_command, env=receiver_env, stdout=log,
                                  stderr=subprocess.STDOUT)
        output = ''
        try:
            for _ in range(30):
                time.sleep(0.1)
                done = subprocess.run(client_command, env=sender_env, text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      timeout=args.timeout)
                output = done.stdout
                if done.returncode == 0 or server.poll() is not None or 'Connection refused' not in output:
                    break
            if done.returncode == 0:
                server.wait(timeout=10)
            point.update(client_returncode=done.returncode, server_returncode=server.poll())
        except subprocess.TimeoutExpired:
            point['failure'] = 'ucx_perftest timed out'
        finally:
            if server.poll() is None:
                server.kill()
                server.wait()
            log.seek(0)
            point['raw_output'] = output + log.read()

    final = [line.split() for line in output.splitlines() if line.lstrip().startswith('Final:')]
    if (point.get('failure') or point.get('client_returncode') != 0 or
            point.get('server_returncode') != 0 or not final):
        point.setdefault('failure', 'ucx_perftest failed; see raw_output')
        return point
    rate = float(final[-1][6])  # UCX prints MiB/s as MB/s.
    protocol = ('put_zcopy (cuda_ipc/cuda)' if test == 'uct' else
                bench.protocols(point['raw_output'], size, memory_type='cuda'))
    # A host-control table or an available IPC lane is not payload-path evidence.
    ipc = bool(protocol) and all('cuda_ipc/cuda' in p for p in protocol.split('; or '))
    point.update(ok=ipc, protocol=protocol, cuda_ipc_payload=ipc,
                 MiB_per_second=rate, GiB_per_second=rate / 1024,
                 frames_per_second=float(final[-1][8]))
    if not ipc:
        point['failure'] = 'CUDA IPC payload path not confirmed; rate is not an IPC result'
    return point


def gpu_info(selector):
    done = subprocess.run(['nvidia-smi', '-i', selector,
                           '--query-gpu=uuid,name,pci.bus_id,driver_version,memory.free',
                           '--format=csv,noheader,nounits'], text=True, capture_output=True,
                          check=True, timeout=10)
    uuid, name, bus, driver, free = [part.strip() for part in done.stdout.strip().split(',')]
    return dict(uuid=uuid, name=name, pci_bus_id=bus, driver=driver,
                free_bytes=int(free) * (1 << 20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=positive_sizes, default=bench.sizes('4K,64K,1M,16M'))
    parser.add_argument('--tests', type=lambda s: s.split(','), default=['uct', 'am'])
    parser.add_argument('--sender-gpu', default='0', help='physical GPU index or UUID')
    parser.add_argument('--receiver-gpu', help='physical GPU index or UUID (same as sender)')
    parser.add_argument('--cpus', type=lambda s: [int(x) for x in s.split(',')],
                        help='sender,receiver CPU IDs; defaults to first and last allowed')
    parser.add_argument('--bytes', type=bench.size, default=256 << 20,
                        help='target bytes moved per point (256M)')
    parser.add_argument('--frames', type=int, help='fixed frame count instead of --bytes')
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--window', type=int, default=32)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--seed', type=int, default=20261011)
    parser.add_argument('--out', type=Path, help='new JSONL output, including full UCX logs')
    parser.add_argument('--dry-run', action='store_true', help='print commands without needing a GPU')
    args = parser.parse_args()
    args.receiver_gpu = args.receiver_gpu or args.sender_gpu
    allowed = sorted(os.sched_getaffinity(0))
    args.cpus = args.cpus or [allowed[0], allowed[-1]]
    if len(args.cpus) != 2 or any(cpu not in allowed for cpu in args.cpus):
        parser.error('--cpus needs two CPUs from the driver\'s allowed affinity')
    if not args.sizes or any(s <= 0 for s in args.sizes) or not args.tests or any(t not in TESTS for t in args.tests):
        parser.error('positive sizes and --tests uct,am are required')
    frames = args.frames if args.frames is not None else args.bytes
    if min(args.bytes, frames, args.window, args.runs, args.timeout) <= 0 or args.warmup < 0:
        parser.error('counts and timeout must be positive; warmup must be nonnegative')
    jobs = [(test, size, run) for test in args.tests for size in args.sizes
            for run in range(args.runs)]
    random.Random(args.seed).shuffle(jobs)
    if args.dry_run:
        for test, size, run in jobs:
            frames = args.frames if args.frames is not None else max(100, min(100_000, args.bytes // size))
            server, client = commands(test, size, frames, args)
            print(json.dumps(dict(test=test, run=run, settings=SETTINGS,
                                  sender_gpu=args.sender_gpu, receiver_gpu=args.receiver_gpu,
                                  server_command=server, command=client)))
        return 0
    with ExitStack() as stack:
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            out = stack.enter_context(args.out.open('x'))
        else:
            out = None
        sender, receiver = gpu_info(args.sender_gpu), gpu_info(args.receiver_gpu)
        args.sender_gpu, args.receiver_gpu = sender['uuid'], receiver['uuid']
        executable = Path(shutil.which('ucx_perftest') or '')
        metadata = dict(kind='metadata', date=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                        host=socket.gethostname(), sender=sender, receiver=receiver,
                        cpus=args.cpus, seed=args.seed, benchmarks=bench.revision(bench.ROOT),
                        settings=SETTINGS,
                        inherited_ucx_settings={k: v for k, v in os.environ.items() if k.startswith('UCX_')},
                        ucx=subprocess.run(['ucx_info', '-v'], text=True, capture_output=True,
                                           check=True, timeout=10).stdout,
                        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if out:
            out.write(json.dumps(metadata) + '\n')
            out.flush()
        print(f"# {sender['name']} -> {receiver['name']}; two processes, reusable CUDA buffers")
        results = []
        for test, size, run in jobs:
            frames = args.frames if args.frames is not None else max(100, min(100_000, args.bytes // size))
            # Each role allocates send and receive buffers. Leave half the free memory
            # for CUDA contexts and UCX internals, even on a same-GPU run.
            buffers = 4 if sender['uuid'] == receiver['uuid'] else 2
            if size * buffers > min(sender['free_bytes'], receiver['free_bytes']) // 2:
                point = dict(ok=False, skipped=True, test=test, frame_bytes=size,
                             failure='CUDA buffers exceed half the free GPU memory')
            else:
                point = run_point(test, size, frames, args)
            point['run'] = run
            results.append(point)
            if out:
                out.write(json.dumps(point) + '\n')
                out.flush()
            rate = f"{point['GiB_per_second']:.2f} GiB/s" if point['ok'] else point['failure']
            print(f"{test:3} {size:10} bytes run {run + 1}: {rate}", flush=True)
        print('\n# Median [worst, best], confirmed CUDA IPC runs only')
        for test in args.tests:
            for size in args.sizes:
                good = [p['GiB_per_second'] for p in results
                        if p['test'] == test and p['frame_bytes'] == size and p['ok']]
                if good:
                    print(f"{test:3} {size:10} bytes: {statistics.median(good):.2f} "
                          f"[{min(good):.2f}, {max(good):.2f}] GiB/s ({len(good)}/{args.runs} runs)")
        return int(any(not p['ok'] and not p.get('skipped') for p in results))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f'cuda-ipc: {error}; run in the GPU environment with a CUDA-enabled UCX')
