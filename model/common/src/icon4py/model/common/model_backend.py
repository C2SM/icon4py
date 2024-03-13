# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
from gt4py.next.program_processors.runners.gtfn import (
    run_gtfn,
    run_gtfn_cached,
    run_gtfn_imperative,
    run_gtfn_gpu,
)


cached_backend = run_gtfn_cached
compiled_backend = run_gtfn
imperative_backend = run_gtfn_imperative
gpu_backend = run_gtfn_gpu
backend = cached_backend
