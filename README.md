# tango-ucx-benchmarks

Measures [tango-ucx](https://github.com/srwohl/tango-ucx) against raw UCX and against
tango-bulk, at the loads its devices have, and reports where the time goes. It is built against
the installed library, as an application is.

Status: **the project builds and links the library. No test is built yet.** The plan is
`docs/benchmarks.md` in tango-ucx: the loads, the sweep, what a host needs, and the order of
work.

## What it will have

Two programs, in the way `ib_send_bw` has two roles:

- `tango_ucx_perf_publisher`: a Tango device server that publishes what it is told to.
- `tango_ucx_perftest <device> -t <test>`: the subscriber. It configures the publisher,
  measures, and prints a table and one JSON line for each point.

A driver plans a sweep from the memory and the transports a host has, runs `ucx_perftest` and
`tango-bulk-bench` at the same points, and writes a report.

## Build

The library is built in its own environment and installed into `stage/` here:

```sh
cd ../tango-ucx && pixi run build-profile   # optimized, with frame pointers
cd ../tango-ucx-benchmarks
pixi run test                               # stage, configure, build, test
```

`TANGO_UCX_SOURCE=/path/to/tango-ucx pixi run test` uses another checkout.

Configure refuses a library that was built without optimization: `pixi run build` in tango-ucx
gives no build type where `CMAKE_ARGS` is empty. `tango_ucx_perftest --version` prints what
both were built from, and starts the library once to show that it runs with this environment's
UCX. The two environments resolve their packages separately: keep cppTango and UCX at the same
version in both.

## Related work

The roles and the tables follow
[`ucx_perftest`](https://github.com/openucx/ucx/tree/master/src/tools/perf) and
[perftest](https://github.com/linux-rdma/perftest) (`ib_send_bw`, `ib_write_bw`), which are
also the references each result is compared with. The rules for a fair comparison are those of
`docs/THROUGHPUT.md` in tango-bulk.

ucTrace attributes UCX traffic to the devices that carried it and the calls that caused it, by
interposing on UCT. It needs its own fork of UCX 1.16 and tango-ucx needs 1.21, so it is cited
here and not used:

```bibtex
@misc{gencer2026uctrace,
      title={ucTrace: A Multi-Layer Profiling Tool for UCX-driven Communication},
      author={Emir Gencer and Mohammad Kefah Taha Issa and Ilyas Turimbetov and James D. Trotter and Didem Unat},
      year={2026},
      eprint={2602.19084},
      archivePrefix={arXiv},
      doi={10.48550/arXiv.2602.19084}
}
```

Accepted at IPDPS 2026. Source: <https://github.com/ParCoreLab/ucTrace>.

## Licence

LGPL-3.0-or-later, as tango-ucx.
