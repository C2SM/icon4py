# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: PGH003 [blanket-type-ignore] # just for the `type: ignore` in next line
# type: ignore

"""
Collection of all Fortran bindings for ICON4Py.

TODO(havogt): refactor py2fgen to generate a single module from functions in different files.
"""

from icon4py.tools.py2fgen.wrappers.diffusion_wrapper import diffusion_init, diffusion_run
from icon4py.tools.py2fgen.wrappers.dycore_wrapper import solve_nh_init, solve_nh_run
from icon4py.tools.py2fgen.wrappers.grid_wrapper import grid_init


__all__ = [
    "diffusion_init",
    "diffusion_run",
    "grid_init",
    "solve_nh_init",
    "solve_nh_run",
]
