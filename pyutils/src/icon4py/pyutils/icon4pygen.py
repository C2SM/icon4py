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

"""Utilities for generating icon stencils."""

from __future__ import annotations

import pathlib

import click


class ModuleType(click.ParamType):
    names = [
        "icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_",
        "icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_",
        "icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_",
    ]

    def shell_complete(self, ctx, param, incomplete):
        if len(incomplete) > 0 and incomplete.endswith(":"):
            completions = [incomplete + incomplete[:-1].split(".")[-1]]
        else:
            completions = self.names
        return [
            click.shell_completion.CompletionItem(name)
            for name in completions
            if name.startswith(incomplete)
        ]


@click.command(
    "icon4pygen",
)
@click.argument("fencil", type=ModuleType())
@click.argument("block_size", type=int, default=128)
@click.argument("levels_per_thread", type=int, default=4)
@click.argument("is_global", type=bool, default=False)
@click.argument(
    "outpath",
    type=click.Path(dir_okay=True, resolve_path=True, path_type=pathlib.Path),
    default=".",
)
def main(
    fencil: str,
    block_size: int,
    levels_per_thread: int,
    is_global: bool,
    outpath: pathlib.Path,
) -> None:
    """
    Generate Gridtools C++ code for an icon4py fencil as well as all the associated C++ and Fortran bindings.

    Args:
        fencil: may be specified as <module>:<member>, where <module> is the dotted name of the containing module
            and <member> is the name of the fencil.

        block_size: refers to the number of threads per block to use in a cuda kernel.

        levels_per_thread: how many k-levels to process per thread.

        outpath: represents a path to the folder in which to write all generated code.
    """
    from icon4py.bindings.workflow import PyBindGen
    from icon4py.pyutils.backend import GTHeader
    from icon4py.pyutils.metadata import get_stencil_info, import_definition

    fencil_def = import_definition(fencil)
    stencil_info = get_stencil_info(fencil_def, is_global)
    GTHeader(stencil_info)(outpath)
    PyBindGen(stencil_info, levels_per_thread, block_size)(outpath)
