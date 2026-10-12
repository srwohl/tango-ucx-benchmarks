#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# Puts the benchmarks on a host, with everything under one directory in /tmp: pixi, its cache,
# the CUDA caches and a clone of each project. Nothing is written to the home directory. The
# settings are in deploy.env: the directory, the hosts, and each project's repository and branch.
#
#   ./deploy.sh node1 node2       those hosts over ssh, all at once, each line with its host
#   ./deploy.sh                   the hosts of DEPLOY_HOSTS, or this host when it names none
#   ./deploy.sh --here            this host
#   ./deploy.sh --no-build node1  pixi and the clones, and nothing built
#   ./deploy.sh -e lab.env node1  other settings than deploy.env
#
# On a host it installs pixi where the directory has none, clones each project or brings its
# clone up to date, checks out the branch, builds tango-ucx optimized and the benchmarks against
# it, and where the host has a GPU the GPU programs and the examples. Run again, it does what is
# left to do.
#
# A repository that is a directory here is carried to the hosts as a git bundle, so a project
# that is on no server can be deployed. Only what is committed goes.
#
# Then, on the host:
#
#   source /tmp/$USER/tango-ucx/deploy.env
#   cd "$DEPLOY_ROOT/tango-ucx-benchmarks" && ./sweep.sh quick
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
settings=$here/deploy.env
only_here=no
build=yes
hosts=()

die() {
    echo "deploy: $*" >&2
    exit 1
}

while (($#)); do
    case $1 in
        -e)
            (($# > 1)) || die "-e takes a file"
            settings=$2
            shift
            ;;
        --here) only_here=yes ;;
        --no-build) build=no ;;
        -*) die "unknown option $1; the head of $0 says what can be given" ;;
        *) hosts+=("$1") ;;
    esac
    shift
done
[[ -f $settings ]] || die "no settings at $settings"
[[ $settings == */* ]] || settings=./$settings
# shellcheck source=deploy.env
source "$settings"

# Each project is cloned into a directory of its name under DEPLOY_ROOT, side by side: the
# benchmarks look for the library in ../tango-ucx.
projects=(tango-ucx tango-ucx-benchmarks)
declare -A repo=([tango-ucx]=$TANGO_UCX_REPO [tango-ucx-benchmarks]=${BENCHMARKS_REPO:-$here})
declare -A ref=([tango-ucx]=$TANGO_UCX_REF [tango-ucx-benchmarks]=$BENCHMARKS_REF)
if [[ -n ${GPU_EXAMPLES_REPO:-} ]]; then
    projects+=(tango-ucx-gpu-examples)
    repo[tango-ucx-gpu-examples]=$GPU_EXAMPLES_REPO
    ref[tango-ucx-gpu-examples]=$GPU_EXAMPLES_REF
fi

# A directory is cloned or bundled, and either takes what is committed. The mistake is found
# here, before any host is reached.
check_directories() {
    local name
    for name in "${projects[@]}"; do
        [[ -d ${repo[$name]} ]] || continue
        git -C "${repo[$name]}" rev-parse -q --verify "${ref[$name]}^{commit}" >/dev/null ||
            die "${repo[$name]} is not a repository with a ${ref[$name]}"
        if [[ -n $(git -C "${repo[$name]}" status --porcelain) ]]; then
            echo "deploy: ${repo[$name]} has changes that are not committed; they stay behind" >&2
        fi
    done
}

if [[ $only_here == no && ${#hosts[@]} -eq 0 ]]; then
    read -r -a hosts <<<"${DEPLOY_HOSTS:-}"
fi

if ((${#hosts[@]})); then
    [[ $only_here == no ]] || die "--here and a host were both given"
    check_directories
    stage=$(mktemp -d)
    trap 'rm -rf -- "$stage"' EXIT
    cp -- "$here/deploy.sh" "$stage/deploy.sh"
    cp -- "$settings" "$stage/deploy.env"
    for name in "${projects[@]}"; do
        if [[ -d ${repo[$name]} ]]; then
            git -C "${repo[$name]}" bundle create -q "$stage/$name.bundle" --all
        fi
    done
    options=--here
    [[ $build == yes ]] || options+=" --no-build"
    # The host's login shell may be another than bash, so bash is given the whole of it.
    there="set -e; d=\$(mktemp -d); trap \"rm -rf \$d\" EXIT; tar -C \$d -xf -; bash \$d/deploy.sh $options"
    pids=()
    for host in "${hosts[@]}"; do
        (tar -C "$stage" -cf - . | ssh -o BatchMode=yes "$host" "bash -c '$there'" 2>&1 |
            sed -u "s/^/$host: /") &
        pids+=($!)
    done
    failed=()
    for i in "${!hosts[@]}"; do
        wait "${pids[i]}" || failed+=("${hosts[i]}")
    done
    ((${#failed[@]} == 0)) || die "failed on: ${failed[*]}"
    exit 0
fi

# This host.
for tool in git tar; do
    command -v "$tool" >/dev/null || die "$tool is needed"
done
mkdir -p -- "$DEPLOY_ROOT" "$PIXI_CACHE_DIR" "$CUDA_CACHE_PATH" "$CUPY_CACHE_DIR"
printf '#!/bin/sh\n' >"$DEPLOY_ROOT/.runs"
chmod +x "$DEPLOY_ROOT/.runs"
"$DEPLOY_ROOT/.runs" 2>/dev/null ||
    die "nothing in $DEPLOY_ROOT can be run: it is mounted noexec. Set DEPLOY_ROOT elsewhere"
rm -f -- "$DEPLOY_ROOT/.runs"
if [[ $(stat -f -c %T -- "$DEPLOY_ROOT") == tmpfs ]]; then
    echo "deploy: $DEPLOY_ROOT is held in memory (tmpfs): what is put there is taken from" \
        "what the benchmarks can use" >&2
fi

# The script, the settings and the bundles are kept, for the next run and for the environment.
if [[ $here != "$DEPLOY_ROOT" ]]; then
    rm -f -- "$DEPLOY_ROOT"/*.bundle
    cp -- "$here/deploy.sh" "$DEPLOY_ROOT/deploy.sh"
    cp -- "$settings" "$DEPLOY_ROOT/deploy.env"
    for bundle in "$here"/*.bundle; do
        [[ ! -f $bundle ]] || cp -- "$bundle" "$DEPLOY_ROOT/"
    done
fi
for name in "${projects[@]}"; do
    [[ ! -f $DEPLOY_ROOT/$name.bundle ]] || repo[$name]=$DEPLOY_ROOT/$name.bundle
done
check_directories

# A command says nothing unless it fails.
quietly() {
    local said
    if ! said=$("$@" 2>&1); then
        printf '%s\n' "$said" >&2
        return 1
    fi
}

pixi=$PIXI_HOME/bin/pixi
if [[ ! -x $pixi ]] || [[ $DEPLOY_PIXI_VERSION != latest &&
    $("$pixi" --version) != "pixi ${DEPLOY_PIXI_VERSION#v}" ]]; then
    echo "installing pixi $DEPLOY_PIXI_VERSION in $PIXI_HOME" >&2
    if command -v curl >/dev/null; then
        installer=$(curl -fsSL https://pixi.sh/install.sh)
    elif command -v wget >/dev/null; then
        installer=$(wget -qO- https://pixi.sh/install.sh)
    else
        die "curl or wget is needed to install pixi"
    fi
    # The installer is kept from the shell's startup files, and downloads to DEPLOY_ROOT.
    PIXI_VERSION=$DEPLOY_PIXI_VERSION PIXI_NO_PATH_UPDATE=1 TMPDIR=$DEPLOY_ROOT \
        quietly bash -c "$installer"
fi
echo "$("$pixi" --version) in $PIXI_HOME" >&2

for name in "${projects[@]}"; do
    clone=$DEPLOY_ROOT/$name
    if [[ -d $clone/.git ]]; then
        git -C "$clone" remote set-url origin "${repo[$name]}"
        git -C "$clone" fetch -q --tags origin
    else
        GIT_TERMINAL_PROMPT=0 git clone -q -- "${repo[$name]}" "$clone" ||
            die "$name could not be cloned from ${repo[$name]}"
    fi
    # A branch follows its origin and loses nothing made here; a tag or a commit is checked out.
    git -C "$clone" checkout -q "${ref[$name]}"
    if git -C "$clone" show-ref -q --verify "refs/remotes/origin/${ref[$name]}"; then
        git -C "$clone" merge -q --ff-only "origin/${ref[$name]}"
    fi
    echo "$name: ${ref[$name]} at $(git -C "$clone" rev-parse --short HEAD)" >&2
done

if [[ $build == yes ]]; then
    gpu=$DEPLOY_GPU
    if [[ $gpu == auto ]]; then
        gpu=no
        if nvidia-smi -L >/dev/null 2>&1; then
            gpu=yes
        fi
    fi
    echo "building tango-ucx, optimized" >&2
    (cd -- "$DEPLOY_ROOT/tango-ucx" && quietly pixi run build-profile)
    echo "building tango-ucx-benchmarks" >&2
    (cd -- "$DEPLOY_ROOT/tango-ucx-benchmarks" && quietly pixi run build)
    if [[ $gpu == yes ]]; then
        echo "building tango-ucx and tango-ucx-benchmarks for the GPU" >&2
        (cd -- "$DEPLOY_ROOT/tango-ucx-benchmarks" && quietly pixi run -e gpu build-gpu)
        if [[ -d $DEPLOY_ROOT/tango-ucx-gpu-examples ]]; then
            echo "building and testing tango-ucx-gpu-examples" >&2
            (cd -- "$DEPLOY_ROOT/tango-ucx-gpu-examples" && quietly ./build_local.sh)
        fi
    else
        echo "no GPU: the GPU programs and the examples are not built" >&2
    fi
fi
echo "ready in $DEPLOY_ROOT; source $DEPLOY_ROOT/deploy.env before running anything there" >&2
