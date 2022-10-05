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

from icon4py.bindings.workflow import PyBindGen
from icon4py.pyutils.backend import GTHeader
from icon4py.pyutils.metadata import get_stencil_info


@click.command(
    "icon4pygen",
)
@click.argument("fencil", type=str)
@click.argument("block_size", type=int)
@click.argument("levels_per_thread", type=int)
@click.argument(
    "outpath",
    type=click.Path(dir_okay=True, resolve_path=True, path_type=pathlib.Path),
)
def main(
    fencil: str,
    block_size: int,
    levels_per_thread: int,
    outpath: pathlib.Path,
) -> None:
    """
    Generate Gridtools C++ code for an icon4py fencil as well as all the associated C++ and Fortran bindings.

    A fencil may be specified as <module>:<member>, where <module> is the
    dotted name of the containing module and <member> is the name of the fencil.

    The outpath represents a path to the folder in which to write all generated code.
    """
    stencil_info = get_stencil_info(fencil)
    GTHeader(stencil_info)(outpath)
    PyBindGen(stencil_info, levels_per_thread, block_size)(outpath)
