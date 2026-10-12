#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
# SPDX-License-Identifier: LGPL-3.0-or-later
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
checkout=$(cd -- "${TANGO_UCX_SOURCE:-../tango-ucx}" && pwd)
if [[ -f stage-gpu/.checkout && $(cat stage-gpu/.checkout) != "$checkout" ]]; then
    # Installing older objects from another checkout preserves their times; start fresh
    # so Ninja cannot mistake the previously linked library for the newly installed one.
    rm -rf stage-gpu build-gpu
fi
# Separate optimized GPU build and install trees keep the host benchmark available.
pixi run --manifest-path "$checkout/pixi.toml" -e gpu cmake -GNinja \
    -S "$checkout" -B "$checkout/build-gpu-profile" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DTANGO_UCX_CUDA=ON -DTANGO_UCX_PROFILING=ON -DBUILD_TESTING=ON
pixi run --manifest-path "$checkout/pixi.toml" -e gpu cmake --build "$checkout/build-gpu-profile"
pixi run --manifest-path "$checkout/pixi.toml" -e gpu cmake \
    --install "$checkout/build-gpu-profile" --prefix "$PWD/stage-gpu"
printf '%s\n' "$checkout" > stage-gpu/.checkout
