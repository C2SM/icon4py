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

import os
import pathlib
from typing import ClassVar

import click


class ModuleType(click.ParamType):
    dycore_import_path = "icon4py.model.atmosphere.dycore"
    names: ClassVar[list[str]] = [
        f"{dycore_import_path}.mo_nh_diffusion_stencil_",
        f"{dycore_import_path}.mo_solve_nonhydro_stencil_",
        f"{dycore_import_path}.mo_velocity_advection_stencil_",
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
@click.option("--is_global", is_flag=True, type=bool, help="Whether this is a global run.")
@click.option(
    "--enable-mixed-precision", is_flag=True, type=bool, help="Enable mixed precision dycore"
)
@click.argument(
    "outpath",
    type=click.Path(dir_okay=True, resolve_path=True, path_type=pathlib.Path),
    default=".",
)
@click.option(
    "--imperative",
    is_flag=True,
    type=bool,
    help="Whether to use the imperative mode in generated gridtools code.",
)
@click.option(
    "--temporaries",
    is_flag=True,
    type=bool,
    help="Whether to use the temporaries in generated gridtools code.",
)
def main(
    fencil: str,
    block_size: int,
    levels_per_thread: int,
    is_global: bool,
    enable_mixed_precision: bool,
    outpath: pathlib.Path,
    imperative: bool,
    temporaries: bool,
) -> None:
    """
    Generate Gridtools C++ code for an icon4py fencil as well as all the associated C++ and Fortran bindings.

    Arguments:
        FENCIL: may be specified as <module>:<member>, where <module> is the dotted name of the containing module and <member> is the name of the fencil.
        BLOCK_SIZE: refers to the number of threads per block to use in a cuda kernel.
        LEVELS_PER_THREAD: how many k-levels to process per thread.
        OUTPATH: represents a path to the folder in which to write all generated code.
    """
    from icon4pytools.icon4pygen.backend import GTHeader
    from icon4pytools.icon4pygen.bindings.workflow import PyBindGen
    from icon4pytools.icon4pygen.metadata import get_stencil_info, import_definition

    os.environ["FLOAT_PRECISION"] = "mixed" if enable_mixed_precision else "double"
    fencil_def = import_definition(fencil)
    stencil_info = get_stencil_info(fencil_def, is_global)
    GTHeader(stencil_info)(outpath, imperative, temporaries)
    PyBindGen(stencil_info, levels_per_thread, block_size)(outpath)


if __name__ == "__main__":
    main()
