# tango-ucx-benchmarks

Measure [tango-ucx](https://github.com/srwohl/tango-ucx) bandwidth, CPU cost and delivery
latency against UCX references, and how fast its frames can be replayed from and written to HDF5. The benchmark programs link the installed library through
its public C++ API, so they exercise the same publishing and subscription path as an application.

Use the tools to compare library builds, size transport rings, choose batch sizes and idle
modes, and distinguish transport costs from application delivery. Transport comparisons run
locally with host memory or CUDA IPC; the Tango publisher and subscriber can also be launched
manually on separate hosts. HDF5 writes run on this host or on several, reached by ssh.
Automated tango-bulk comparisons are not provided.

## Run a bandwidth sweep

Use Linux and Pixi >=0.55 with a tango-ucx checkout that has the `build-profile` task.
The Pixi environment provides a C++20 compiler, CMake, UCX >=1.21, cppTango >=10, HDF5,
h5py and Python >=3.11.
From this directory:

```sh
TANGO_UCX_SOURCE=/path/to/tango-ucx ./sweep.sh quick
```

The script builds the library optimized, installs it into `stage/`, builds the benchmarks,
and runs tango-ucx beside `ucx_perftest`. The default library path is `../tango-ucx`.
Results go into `results/`: a text table, JSONL for every run and a Markdown report.

| Tier | Frame sizes | Ring depths | Runs per point |
|---|---|---|---|
| `./sweep.sh quick` | 4 KiB, 64 KiB, 1 MiB, 16 MiB | 32 | 1 |
| `./baseline.sh` or `./sweep.sh baseline` | Eight sizes from 1 KiB to 16 MiB | 2, 32 | 3 |
| `./sweep.sh` or `./sweep.sh standard` | Powers of two from 1 KiB to 64 MiB | 2, 32 | 3 |
| `./sweep.sh full` | Powers of two from 1 KiB to 2 GiB | 2, 8, 32, 128 | 5 |

Points requiring more than half the available host memory are recorded as skipped.
`--dry-run` lists points without measuring them; the wrapper still builds first.
Options override the tier defaults:

```sh
./sweep.sh full --dry-run
./sweep.sh standard --runs 5 --depth 2,32,256
```

The driver warns about a busy host or a CPU governor other than `performance`, since both
can affect rates. Use repeated runs for comparisons; the quick tier is a smoke check.

## Build and test separately

```sh
export TANGO_UCX_SOURCE=/path/to/tango-ucx
(cd "$TANGO_UCX_SOURCE" && pixi run build-profile)
pixi run test
```

`test` stages the optimized library, configures, builds and runs verification checks.
When the selected checkout changes, `stage.sh` recreates `stage/` and `build/` to avoid
reusing artifacts from the previous checkout. Configure rejects an unoptimized library.
Keep UCX and cppTango versions aligned between the library and benchmark environments.
`build/tango_ucx_perftest --version` reports build identities and checks library startup.

## Choose a measurement

| Tool | Question it answers |
|---|---|
| [bench.py](bench.py) | What bandwidth does tango-ucx sustain at each frame size and ring depth? |
| [bench.py](bench.py) `--hdf5` | How fast can a detector's image stack be replayed from its HDF5 file? |
| [hdf5_write.py](hdf5_write.py) | How fast can a stream be written to HDF5 shards by threads, processes and hosts? |
| [zarr_write.py](zarr_write.py) | The same, written to Zarr v3 by acquire-zarr, with buffered or direct I/O |
| [frame_cost.py](frame_cost.py) | What is sustained time per frame as size, depth and batch size change? |
| [cpu_cost.py](cpu_cost.py) | How much CPU and publish-to-read latency does an application rate require? |
| [compare_ucx.py](compare_ucx.py) | How do reused UCX buffers, rotating raw UCP rings and library delivery compare? |
| [cuda_ipc.py](cuda_ipc.py) | What bandwidth do direct CUDA IPC puts and UCX CUDA active messages achieve? |
| [report.py](report.py) | How do saved sweeps compare across builds, hosts or settings? |

### Bandwidth and payload verification

`bench.py local` starts `tango_ucx_perf_publisher`, configures it through Tango, receives
through an every subscription, and prints medians with best/worst runs:

```sh
pixi run python bench.py local
pixi run python bench.py local --sizes 1M,16M --depth 2 --runs 5 --out results/depth2.jsonl
pixi run python bench.py local --sizes 64K --verify
```

The driver finds the byte budget that produces each requested ring depth; the library
derives depth from a budget. Publishers have at least three source slots. Timing starts
after both rings have been used twice and ends when the last batch is released. Normal
runs fill source slots once. `--verify` fills and checks every frame, adding work to the
timed path; `--corrupt-every N` passes only if the expected intentional damage is found.

`--reference` adds two stock `ucx_perftest -t ucp_am_bw` rows:

| Reference | Endpoint configuration |
|---|---|
| `ucx_perftest` | UCX defaults |
| `ucx_perftest -e` | `-H 40 -z -e -I`, `UCX_MM_ERROR_HANDLING=y`, `UCX_KEEPALIVE_INTERVAL=2s`: 40-byte header, registered memory, peer handling and wakeup |

The table's "of raw" value compares tango-ucx with the second reference. Stock UCX reuses
one payload buffer, so a depth-2 library ring is the nearer working-set comparison.
These bandwidth runs do not pin either role by default. Protocol columns come from
`UCX_PROTO_INFO` for the measured frame size; differing endpoint tables are reported.

For manual runs on one or two hosts, replace the host and port:

```sh
build/tango_ucx_perf_publisher bench -nodb -dlist bench/ucx/1 -ORBendPoint giop:tcp:publisher-node:22080
build/tango_ucx_perftest 'publisher-node:22080/bench/ucx/1#dbase=no' -s 1M -n 2000
```

### Frames from an HDF5 file

The publisher can replay an image stack instead of generated bytes. A rank-three dataset is
`[frame, height, width]`; a rank-two dataset is one frame. When no dataset is named, the NeXus
`default`/`signal` attributes lead to it. Integers of 8 to 64 bits and 32 or 64-bit floats are
published as their element type, converted to the host's byte order. Frames are read with
`H5Dread` straight into source slots, in file order, wrapping when a run needs more frames
than the file has.

A reader thread reads up to `--read-ahead` frames (8) before they are published, so a slow or
uneven read does not delay a publish; the publishing thread only paces and publishes. Frames
read ahead wait in source slots beyond `--depth`, so the publisher ring is that much deeper.
`--read-ahead 0` reads each frame on the publishing thread, for comparison.

```sh
pixi run python hdf5_sample.py results/stack.h5 --frames 200 --shape 2208,3216 --dtype uint16 --rate 100
pixi run python bench.py local --hdf5 results/stack.h5 --depth 2,32 --reference --out results/hdf5.jsonl
pixi run python bench.py local --hdf5 /data/scan.h5 --hdf5-dataset /entry/data/data --verify
```

The frame size is the dataset's, so `--sizes` does not apply. Each depth runs twice: generated
frames of that size, then `tango-ucx hdf5` with the file's frames. The difference is the cost
of reading. The report adds a section with the dataset's layout and filters, the read-ahead,
the share of the run spent in `H5Dread`, the share the publishing thread waited for the
reader, and the replay rate as a share of the acquisition rate when the file records
`acquisition/exposure_time` and `latency_time`. High reading and waiting shares mean the file,
not the transport, sets the rate. To compare read-ahead settings, give `report.py` sweeps
made with different `--read-ahead` values. A file that fits in the
page cache is read from memory after its first pass; drop caches or use a larger file to
measure storage. Compressed datasets need their HDF5 filter plugin on `HDF5_PLUGIN_PATH`.

Each record carries the dataset frame it came from. `--verify` adds a checksum computed after
the read, then checks the frame order and every byte. The fields are named
`source_frame` and `checksum` for Python subscribers. The file path is the publisher host's:

```sh
build/tango_ucx_perftest 'publisher-node:22080/bench/ucx/1#dbase=no' --hdf5 /data/scan.h5 --describe
build/tango_ucx_perftest 'publisher-node:22080/bench/ucx/1#dbase=no' --hdf5 /data/scan.h5 -n 2000
```

`hdf5_sample.py` writes stacks of a chosen shape, type, layout and compression. Its
`--self-test` is the `hdf5_replay` check run by `pixi run test`. Configure with
`-DBENCHMARKS_HDF5=OFF` to build without HDF5. CUDA source slots are not supported for files.

### Writing HDF5 shards

`hdf5_write.py` writes a stream to HDF5 with writers laid out as processes x threads.
Each thread of `tango_ucx_hdf5_writer` joins the publisher's pull set and writes its own shard,
`capture.part_NNN.h5`. A range of `--frames-per-block` x `--blocks-per-stripe` frames goes to
one writer at a time. A writer writes each block with one `H5Dwrite`, directly from its receive
ring, and then releases it; the publisher's credit returns only after the write. After each
run the driver creates `capture.h5`, whose virtual `/images` dataset restores stream order
from each shard's `/frame_index`. Every frame must appear exactly once.

```sh
pixi run python hdf5_write.py --output-dir /gpfs/scratch/me --frame-bytes 4M --frames 4096 \
  --layouts 1x1,1x4,4x1,8x1 --repeats 3 --note gpfs_block_size=8MiB --out results/writes.jsonl
pixi run python report.py results/writes.jsonl
```

The layouts separate HDF5's limits:

| Layout | What it shows |
|---|---|
| `1x1` | The most one writer process achieves |
| `1xT` | Threads in one process: the thread-safe library runs one call at a time, so "waiting" grows and "overlap" stays near 1 |
| `Px1` | Processes on one host: separate libraries, so HDF5 calls overlap |
| `--hosts n1,n2 --layouts Px1` | P processes on each host |

For other hosts, the driver starts writers by ssh from this checkout's path, or from
`--remote-build`; `deploy.sh` creates matching paths. `--publisher-address` is this host's
name as the writer hosts reach it. The output directory must be shared, because the driver
creates the master and removes the files. The publisher runs here and replays `--hdf5` files
or generated frames of `--frame-bytes`. A publisher has at most 64 sessions, so a run has at
most 60 writers.

Shards are chunked, with one block per chunk unless `--chunk-frames` is given. They grow as
frames arrive, because the frames each writer receives are not known in advance. Each writer
calls `fsync` after closing its shard; `--no-sync` omits that, and so durability, from the
measurement. Throughput runs from the first frame received to the last write. The report
lists each layout's median, best and worst GB/s. It also gives the per-writer rate, overlap
(time in HDF5 calls over run time), the share of HDF5 time spent waiting for another thread in
the same process, the minimum and maximum one-second rates, and sync time. `--check` reads the
master back and compares each frame with what was sent. Files of passing runs are deleted
unless `--keep` is given. Writer logs are kept beside `--out`. The `hdf5_write` test runs
`--self-test`.

### Writing Zarr with acquire-zarr

`zarr_write.py` runs the same layouts as `hdf5_write.py`, but the writer is
`tango_ucx_zarr_writer`, built on [acquire-zarr](https://github.com/acquire-project/acquire-zarr).
It lives in its own pixi environment, because acquire-zarr is not on conda-forge and its
release libraries need glibc 2.38. `stage_zarr.sh` builds v0.10.0 from source into
`stage-zarr/`:

```sh
pixi run -e zarr test-zarr       # stage tango-ucx and acquire-zarr, build build-zarr, self-tests
pixi run -e zarr python zarr_write.py --output-dir /gpfs/scratch/me --frame-bytes 4M \
  --frames 4096 --layouts 1x1,1x4,4x1,8x1 --repeats 3 --out results/zarr-buffered.jsonl
pixi run -e zarr python zarr_write.py --output-dir /gpfs/scratch/me --frame-bytes 4M \
  --frames 4096 --layouts 1x1,1x4,4x1,8x1 --repeats 3 --direct-io --out results/zarr-direct.jsonl
```

Each writer thread owns a part, `capture.part_NNN.zarr`: a Zarr v3 store whose `/images` is
`[frame, ...image]`, sharded along frames. A one-axis frame is stored as `[1, n]`, because
acquire-zarr needs two image axes. Differences from the HDF5 writer:

- **An append copies and returns.** `ZarrStream_append` copies a block into the stream's queue
  (about 256 MiB) and returns; acquire-zarr's own threads write it. So the credit returns
  before the frames are on disk. "GB/s" runs from the first frame received to the last part
  closed, which is when every shard is written and fsynced; "accepted" runs to the last append.
- **No lock between threads.** Each stream is independent, so `1xT` is not serialized the way
  HDF5 is. Each stream starts its own pool, by default one thread per core: set
  `--zarr-threads` when T is large.
- **No `--no-sync`.** acquire-zarr fsyncs each shard file, and `ZarrStream_close` reports a
  failed write or flush.
- **Compression** is `--compression none|blosc-lz4|blosc-zstd|zstd` with `--level` and
  `--shuffle`.

After each run the driver makes `capture.zarr`, a native Zarr v3 store of the whole stream that
zarr-python, tensorstore or any other Zarr v3 reader opens:

```python
import zarr
images = zarr.open_array("/gpfs/scratch/me/tango-ucx-zarr-.../capture.zarr/images", mode="r")
```

Its `images/zarr.json` is the parts' array with the run's frame count. Each shard file is a
relative symlink to a part's shard file, so nothing is copied. This works because the publisher
gives each puller whole ranges that start at multiples of the range, and the writer refuses to
start unless a shard (`--chunk-frames` x `--shard-chunks`, by default the range) divides the
range. Each part records its frames' runs in `images/zarr.json` under
`attributes.tango_ucx.frame_runs`. A frame missing or written twice fails the run, and so does a
range split between pullers, which happens only if a puller leaves mid-run. `--check` reads
every frame back with zarr-python and with tensorstore and compares it with what was sent. To
copy a capture elsewhere, follow the links (`cp -rL`, `rsync -L`). Moving the run directory
as a whole keeps them working.

`--direct-io` sets `ZARR_DIRECT_IO=1`, which opens shard files with `O_DIRECT`. acquire-zarr
writes one `pwrite` per chunk, then a small index at the end of each shard file, so only the
filesystem decides whether unaligned direct writes work. NFS takes them; ext4 and XFS fail
every write with `EINVAL`. On GPFS, a direct write that allocates a new block, or is not
aligned, leaves the fast direct path and is synced before it returns. Streaming writes allocate
on nearly every block, so measure direct I/O against buffered writes, not instead of them. Use
uncompressed chunks of at least the GPFS block size (`mmlsfs <fs> -B`); compressed chunks have
odd lengths, so every write after a shard's first is unaligned. The self-test runs a direct
I/O case only when the build directory is on NFS or GPFS.

### Sustained time per frame

`frame_cost.py` measures fixed frame counts and reports nanoseconds per frame, the inverse
of frames/s. This is a throughput measure. To compare builds, preserve their executables
before rebuilding; runs are interleaved in shuffled order:

```sh
mkdir -p build-before
cp build/tango_ucx_perf_publisher build/tango_ucx_perftest build-before/
# Rebuild the library and benchmarks before running the comparison.
pixi run python frame_cost.py --build before=build-before --build after=build \
  --sizes 64,1K,4K,16K --depth 32,128 --batch 1,8,32 --frames 1000000 --runs 5 \
  --reference --out results/frame-cost.jsonl
```

`--thread-cpus 0,2,4,6` assigns publisher progress, producer, subscriber progress and reader
threads. Replace these IDs with four distinct allowed physical cores. Discovery runs every
10 ms during setup and warmup; allow enough warmup frames for all threads to appear.
Without the option, threads inherit the driver's affinity.

Results retain Health, thread CPU times and software versions; metadata records executable
hashes, shuffle seed and inherited affinity. Output must be new, or use `--resume` with the
same measurement options and hashes. `--reference` adds `ucx_perftest -e -y`, including the
eager receive copy. Its reused buffer and fewer application threads make the difference
in ns/frame an estimate of the throughput gap, rather than a particular function's cost.

### CPU and latency at an application rate

`--rate` paces publication in frames/s; zero is unpaced. Both roles must run on one host
for `publish_to_read_ns`, which uses a producer steady-clock timestamp. Latency includes
library delivery, scheduling and the application's read, and excludes waiting for the
scheduled publication time. Sample collection is included in reader CPU time; sorting is excluded.

```sh
pixi run python bench.py local --sizes 1K --bytes 20M --rate 10000 --out results/paced.jsonl
```

`cpu_cost.py` interleaves builds and idle modes at matched rates:

```sh
pixi run taskset -c 0,2,4,6 python cpu_cost.py \
  --variant default=build,default --variant wait=build,wait \
  --rates 0,1000,10000,50000 --seconds 2 --runs 3 \
  --size 1K --depth 128 --batch 1 --thread-cpus 0,2,4,6 --out results/cpu-cost.jsonl
```

Choose four allowed physical cores. Add `--variant before=build-before,default` for another
build. `default` clears `TANGO_UCX_IDLE`; `wait` skips the 200 microsecond polling window;
`spin` polls continuously. The library must support the selected mode. Output must be new.

CPU microseconds per frame are summed across the four hot threads. Subscriber times exclude
warmup; publisher times include it and are normalized over all published frames. CPU time
per frame multiplied by the achieved rate estimates occupied cores. A rate more than 5%
from the requested value fails the point. Executable hashes, settings, affinity and seed
are retained with the results.

The paced producer requests 1 ns Linux timer slack and records the actual setting plus
`pacing_late_frames`; a correct average rate can still have uneven arrivals. Temperature
and package-throttle snapshots cover the whole invocation, including setup and warmup.
Throttle deltas include other workloads on the host.

### Working-set and application-delivery comparisons

`compare_ucx.py` compares stock UCX, `ucp_ring` and tango-ucx. Stock UCX reuses a single
payload buffer per role. `ucp_ring` is a separate raw UCP benchmark with rotating registered
rings and credit returned after consumption. The library adds application/progress threads,
slot and batch ownership, release queues and Health observations.

```sh
pixi run taskset -c 0,2,4,6 python compare_ucx.py --build build --runs 5 \
  --thread-cpus 0,2,4,6 --out results/ucx-comparison.jsonl
```

Default cases are 1 KiB, 4 KiB, 256 KiB and 1 MiB at 128 slots per ring: total active payload
working sets of 256 KiB, 1 MiB, 64 MiB and 256 MiB. UCX internal buffers and metadata are
additional. Stock UCX keeps two payload buffers regardless of its outstanding-send limit.

All host paths use a 40-byte header, registered memory, peer handling and wakeup. Stock
uses `-H 40 -z -e -I -y -O 128`; raw and library paths include eager copies and rendezvous
receives into their rings. Both consume in order and return credit after consumption; raw
source slots also wait for send completion. Raw combines its roles into two hot threads,
while the library uses four. Control messages, bookkeeping and clock checks differ.

To vary the working set at fixed message size and raw credit window:

```sh
pixi run taskset -c 0,2,4,6 python compare_ucx.py --build build --raw-only \
  --case 256K:8 --case 256K:128 --window 8 --frames 16384 --batch 1 --runs 5 \
  --thread-cpus 0,2,4,6 --out results/ucx-cache.jsonl
```

Publisher source rings are limited to 1024 slots. Inspect reported depths and working sets
for larger library cases; `--raw-only` can explore larger rings. Paced comparisons use
`--case 1K:128 --rates 1000,10000 --modes default,wait --batch 1` and omit stock UCX, which
has no paced-source CLI. Stock CPU covers the full invocation and is retained separately.
Raw CPU covers both role processes; library CPU sums its four hot threads. Receiver times
are normalized over measured frames, sender times over all published frames including warmup.
Raw rings are prefilled before timing; library source fills occur during warmup.

`--verify` generates and checks every raw/library payload and omits stock UCX. `--self-test`
checks ring reuse, eager/fragmented/rendezvous payloads, idle modes and paced latency.
Raw UCP comparisons use two processes on one host. Tango setup is outside the timed frame loop.

### CUDA IPC references and delivery

GPU measurements need NVIDIA hardware and a CUDA 12.9 compatible driver. The GPU
environment supplies the CUDA runtime required by UCX's benchmark allocator.
Run stock direct-transport puts and UCP active messages in CUDA memory:

```sh
pixi run -e gpu cuda-ipc --out results/cuda-ipc.jsonl
pixi run -e gpu cuda-ipc --sizes 1K..64M --runs 5 --out results/cuda-ipc-sweep.jsonl
pixi run -e gpu cuda-ipc --sender-gpu 0 --receiver-gpu 1 --out results/cuda-ipc-pair.jsonl
pixi run -e gpu cuda-ipc --dry-run
```

The UCT test uses `put_bw -x cuda_ipc -d cuda -D zcopy -m cuda`; UCP uses
`ucp_am_bw -m cuda -H 40 -z -e -I` with `UCX_TLS=sm,cuda_copy,cuda_ipc`.
Both processes use the same GPU by default. `--sender-gpu` and `--receiver-gpu` accept
physical indices or UUIDs, mapped to visible GPU 0 in each process. `--cpus` selects two
allowed CPUs; defaults are the first and last allowed IDs. Frame count, warmup, window
and byte count are configurable. These references reuse buffers, exclude setup handle
exchange/mapping from timing and do not verify every payload.

For rotating CUDA rings and full library delivery:

```sh
TANGO_UCX_SOURCE=/path/to/tango-ucx pixi run -e gpu cuda-compare --out results/cuda-comparison.jsonl
pixi run -e gpu cuda-compare --case 1M:128 --case 16M:32 --frames 16384 --runs 3 \
  --out results/cuda-comparison-large.jsonl
pixi run -e gpu cuda-compare --sender-gpu 0 --receiver-gpu 1 --out results/cuda-comparison-pair.jsonl
pixi run -e gpu test-gpu
```

`cuda-compare` builds and installs an optimized CUDA library into `stage-gpu/`, builds
benchmarks in `build-gpu/`, and runs `compare_ucx.py --memory cuda`. It compares stock
active messages, rotating raw CUDA rings and GPU slots/views, with batches 1 and 8 by
default. Header size, peer handling, wakeup, warmup and CUDA IPC settings are matched;
CUDA uses rendezvous, and the host eager-copy flag is omitted. Stock/raw send windows
match library depth. Depth must be divisible by every batch size, hold at least two
batches, and be no larger than 1024 or `--window`.

Four allowed physical cores are chosen automatically; `--thread-cpus p,g,s,r` overrides
them. Stock and raw use the progress cores. Rings needing more than half the free GPU
memory are skipped. Accepted UCP CUDA IPC points require a CUDA-source protocol table
confirming `cuda_ipc/cuda`; staged or unconfirmed paths are retained as failures and
excluded from medians. Reports retain traces, GPU identities, actual depths, placement,
hashes, temperatures and CPU accounting scope. JSONL and Markdown output paths must be new.

Normal runs reuse prefilled slots. Library timing includes the final consumer stream event;
raw consumes immediately after transfer completion and excludes that application event
work. `--verify` adds host generation, upload, download and full checks to both and omits
stock UCX. `test-gpu` covers ring reuse, idle modes, rendezvous, placement and corruption.

## Read and compare results

```sh
pixi run python report.py results/before.jsonl results/after.jsonl --label before,after --out compared.md
```

The first file is the reference. Reports compare matching points, changes, library rates
as a share of raw UCX, sustained ns/frame and selected protocols. Failed, missing or
high-variance points remain visible. Changes within the run spread and single-run
comparisons are marked inconclusive. For methodology and local measurements, see
[the library benchmark documentation](https://github.com/srwohl/tango-ucx/blob/master/docs/benchmarks.md).

## Deploy to other hosts

Edit `deploy.env` for host names, repositories, branches and deployment directory, then run:

```sh
./deploy.sh node1 node2
./deploy.sh --here
```

Deployment uses one directory under `/tmp` on each host for Pixi, package/CUDA caches,
checkouts and builds. Local repository paths are transferred as git bundles containing
committed work. GPU programs build on hosts with GPUs. In the deployed environment:

```sh
source /tmp/$USER/tango-ucx/deploy.env
cd "$DEPLOY_ROOT/tango-ucx-benchmarks"
./sweep.sh quick
```

Deployment supports running local comparisons on each host. `hdf5_write.py --hosts` is the
only automated benchmark that spans hosts.

LGPL-3.0-or-later; see [LICENSE](LICENSE).
