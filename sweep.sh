#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# A sweep in one command: builds tango-ucx optimized, builds this project against it, runs
# tango-ucx beside ucx_perftest on this host, and writes a report. A point whose rings do not fit
# this host's memory is left out, and the report says so.
#
#   ./sweep.sh                    the standard tier: 1 KiB to 64 MiB, about five minutes
#   ./sweep.sh quick              four sizes, one run each: seconds
#   ./sweep.sh full               1 KiB to 2 GiB at four depths, five runs: much longer
#   ./sweep.sh full --dry-run     list the points and run none
#   ./sweep.sh standard --runs 5  an option of `bench.py local` replaces the tier's
#   TANGO_UCX_SOURCE=/a/checkout ./sweep.sh       another tango-ucx than ../tango-ucx
#
# Two sweeps are compared with: pixi run python report.py results/a.jsonl results/b.jsonl
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
tier=standard
case ${1:-} in
    quick | baseline | standard | full)
        tier=$1
        shift
        ;;
esac
wanted=${TANGO_UCX_SOURCE:-$here/../tango-ucx}
if ! library=$(cd -- "$wanted" 2>/dev/null && pwd) || [[ ! -f $library/pixi.toml ]]; then
    echo "sweep: no tango-ucx checkout at $wanted; set TANGO_UCX_SOURCE" >&2
    exit 1
fi
export TANGO_UCX_SOURCE=$library
if ! command -v pixi >/dev/null; then
    echo "sweep: pixi is needed, see https://pixi.sh" >&2
    exit 1
fi

# A build says nothing unless it fails.
quietly() {
    local said
    if ! said=$("$@" 2>&1); then
        printf '%s\n' "$said" >&2
        return 1
    fi
}
echo "building tango-ucx, optimized: $library" >&2
if ! (cd -- "$library" && quietly pixi run build-profile); then
    echo "sweep: tango-ucx did not build; it needs the build-profile task of its pixi.toml" >&2
    exit 1
fi
echo "building tango-ucx-benchmarks" >&2
(cd -- "$here" && quietly pixi run build)

cd -- "$here"
for option in "$@"; do
    if [[ $option == --dry-run ]]; then
        exec pixi run -q python bench.py local --tier "$tier" "$@"
    fi
done
mkdir -p results
name=$(hostname -s)-$(date +%Y-%m-%d-%H%M%S)-$tier
pixi run -q python -u bench.py local --tier "$tier" --out "results/$name.jsonl" "$@" 2>&1 |
    tee -- "results/$name.txt"
pixi run -q python report.py "results/$name.jsonl" --out "results/$name.md"
echo "saved results/$name.txt, results/$name.jsonl and the report results/$name.md" >&2
