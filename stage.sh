#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# Installs the optimized tango-ucx of TANGO_UCX_SOURCE, or of ../tango-ucx, into stage/. Run by
# `pixi run stage`. The library is first brought up to date with its source: the headers come
# from the source, and a library older than them crashes at the first call. When the checkout is
# another than last time, stage/ and build/ are removed: an install keeps each file's time, so a
# build would not see that an older header or library had replaced a newer one.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
tree=${TANGO_UCX_SOURCE:-../tango-ucx}/build-profile
if [[ ! -f $tree/build.ninja ]]; then
    echo "stage: $tree is not there: run \`pixi run build-profile\` in tango-ucx" >&2
    exit 1
fi
checkout=$(cd -- "$tree/.." && pwd)
if [[ $(cat stage/.checkout 2>/dev/null) != "$checkout" ]]; then
    rm -rf stage build
fi
cmake --build "$tree"
cmake --install "$tree" --prefix stage
echo "$checkout" > stage/.checkout
