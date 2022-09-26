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

from icon4py.bindings.workflow import CppBindGen, PyBindGen
from icon4py.common.dimension import Koff
from icon4py.pyutils.backend import GTHeader
from icon4py.pyutils.metadata import (
    StencilInfo,
    get_fvprog,
    import_definition,
    provide_offset,
    scan_for_offsets,
)

import sys


def get_stencil_info(fencil) -> StencilInfo:
    fencil_def = import_definition(fencil)
    fvprog = get_fvprog(fencil_def)
    offsets = scan_for_offsets(fvprog)
    offset_provider = {}
    for offset in offsets:
        offset_provider[offset] = provide_offset(offset)
    connectivity_chains = [offset for offset in offsets if offset != Koff.value]
    return StencilInfo(fvprog, connectivity_chains, offset_provider)


@click.command(
    "icon4pygen",
)
@click.option(
    "--cppbindgen-path",
    type=click.Path(
        exists=True, dir_okay=True, resolve_path=True, path_type=pathlib.Path
    ),
    help="Path to cppbindgen source folder. Specifying this option will compile and execute the c++ bindings generator and store all generated files in the build folder under build/generated.",
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
    cppbindgen_path: pathlib.Path = None,
) -> None:
    """
    Generate C++ code for an icon4py fencil as well as all the associated C++ and Fortran bindings.

    A fencil may be specified as <module>:<member>, where <module> is the
    dotted name of the containing module and <member> is the name of the fencil.

    The outpath represents a path to the folder in which to write all generated code.
    """
    stencil_info = get_stencil_info(fencil)

    # todo: this is temporary and should be removed once PyBindGen is complete
    if cppbindgen_path:
        CppBindGen(stencil_info)(cppbindgen_path)

    GTHeader(stencil_info)(outpath)
    PyBindGen(stencil_info, levels_per_thread, block_size)(outpath)