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

import importlib
import pathlib
import types

import click
from functional.common import Dimension, DimensionKind
from functional.ffront import common_types as ct
from functional.ffront import program_ast as past
from functional.ffront.decorator import FieldOperator, Program, program
from functional.iterator import ir as itir

from icon4py.bindings.workflow import CppBindGen, PyBindGen
from icon4py.common.dimension import CellDim, EdgeDim, Koff, VertexDim
from icon4py.pyutils.exceptions import (
    InvalidConnectivityException,
    MultipleFieldOperatorException,
)
from icon4py.pyutils.icochainsize import IcoChainSize
from icon4py.pyutils.stencil_info import StencilInfo


def scan_for_offsets(fvprog: Program) -> set[str]:
    """Scan PAST node for offsets and return a set of all offsets."""
    all_types = (
        fvprog.past_node.pre_walk_values().if_isinstance(past.Symbol).getattr("type")
    )
    all_field_types = [
        symbol_type
        for symbol_type in all_types
        if isinstance(symbol_type, ct.FieldType)
    ]
    all_dims = set(i for j in all_field_types for i in j.dims)
    all_offset_labels = (
        fvprog.itir.pre_walk_values()
        .if_isinstance(itir.OffsetLiteral)
        .getattr("value")
        .to_list()
    )
    all_dim_labels = [dim.value for dim in all_dims if dim.kind == DimensionKind.LOCAL]
    return set(all_offset_labels + all_dim_labels)


def provide_neighbor_table(chain: str) -> types.SimpleNamespace:
    """Build an offset provider based on connectivity chain string.

    Connectivity strings must contain one of the following connectivity type identifiers:
    C (cell), E (Edge), V (Vertex) and be separated by a '2' e.g. 'E2V'. If the origin is to
    be included, the string should terminate with O (uppercase o), e.g. 'C2E2CO`.

    Handling of "new" sparse dimensions

    A new sparse dimension may look like C2CE or V2CVEC. In this case, we need to strip the 2
    and pass the tokens after to the algorithm below
    """
    # note: this seems really brittle. maybe agree on a keyword to indicate new sparse fields?
    new_sparse_field = any(
        len(token) > 1 for token in chain.split("2")
    ) and not chain.endswith("O")
    if new_sparse_field:
        chain = chain.split("2")[1]

    location_chain = []
    include_center = False
    for letter in chain:
        if letter == "C":
            location_chain.append(CellDim)
        elif letter == "E":
            location_chain.append(EdgeDim)
        elif letter == "V":
            location_chain.append(VertexDim)
        elif letter == "O":
            include_center = True
        elif letter == "2":
            pass
        else:
            raise InvalidConnectivityException(location_chain)
    return types.SimpleNamespace(
        max_neighbors=IcoChainSize.get(location_chain) + include_center,
        has_skip_values=False,
    )


def provide_offset(offset: str) -> type.SimpleNamespace | Dimension:
    if offset == Koff.value:
        assert len(Koff.target) == 1
        assert Koff.source == Koff.target[0]
        return Koff.source
    else:
        return provide_neighbor_table(offset)


def import_definition(name: str) -> Program | FieldOperator | types.FunctionType:
    module_name, member_name = name.split(":")
    fencil = getattr(importlib.import_module(module_name), member_name)
    return fencil


def get_fvprog(fencil_def: Program | FieldOperator | types.FunctionType) -> Program:
    match fencil_def:
        case Program():
            fvprog = fencil_def
        case FieldOperator():
            fvprog = fencil_def.as_program()
        case _:
            fvprog = program(fencil_def)

    if len(fvprog.past_node.body) > 1:
        raise MultipleFieldOperatorException()

    return fvprog


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
@click.argument(
    "outpath",
    type=click.Path(dir_okay=True, resolve_path=True, path_type=pathlib.Path),
)
def main(
    outpath: pathlib.Path, fencil: str, cppbindgen_path: pathlib.Path = None
) -> None:
    """
    Generate C++ code for an icon4py fencil as well as all the associated C++ and Fortran bindings.

    A fencil may be specified as <module>:<member>, where <module> is the
    dotted name of the containing module and <member> is the name of the fencil.

    The outpath represents a path to the folder in which to write all generated code.
    """
    stencil_info = get_stencil_info(fencil)

    if cppbindgen_path:
        CppBindGen(stencil_info)(cppbindgen_path)

    PyBindGen(stencil_info)(outpath)


if __name__ == "__main__":
    main()
