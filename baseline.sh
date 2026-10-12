#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# The baseline in one command: eight frame sizes from 1 KiB to 16 MiB at ring depths 2 and 32,
# three runs each, tango-ucx beside ucx_perftest, in about two minutes. It is the baseline tier
# of sweep.sh, which says what else can be given.
exec "$(dirname -- "${BASH_SOURCE[0]}")/sweep.sh" baseline "$@"
