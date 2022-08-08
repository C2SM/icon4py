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

import dataclasses
import importlib
import pathlib
import types
from collections.abc import Iterable
from typing import Any, TypeGuard

import click
import tabulate
from functional.common import Connectivity, Dimension, DimensionKind
from functional.fencil_processors.codegens.gtfn.gtfn_backend import generate
from functional.ffront import common_types as ct
from functional.ffront import program_ast as past
from functional.ffront.decorator import FieldOperator, Program, program
from functional.iterator import ir as itir

from icon4py.common.dimension import CellDim, EdgeDim, Koff, VertexDim
from icon4py.pyutils.exceptions import (
    InvalidConnectivityException,
    MultipleFieldOperatorException,
)
from icon4py.pyutils.icochainsize import IcoChainSize


@dataclasses.dataclass(frozen=True)
class _FieldInfo:
    field: past.DataSymbol
    inp: bool
    out: bool


def is_list_of_names(obj: Any) -> TypeGuard[list[past.Name]]:
    return isinstance(obj, list) and all(isinstance(i, past.Name) for i in obj)


def get_field_infos(fvprog: Program) -> dict[str, _FieldInfo]:
    """Extract and format the in/out fields from a Program."""
    assert is_list_of_names(
        fvprog.past_node.body[0].args
    ), "Found unsupported expression in input arguments."
    input_arg_ids = set(arg.id for arg in fvprog.past_node.body[0].args)

    out_arg = fvprog.past_node.body[0].kwargs["out"]
    out_arg = out_arg.value if isinstance(out_arg, past.Subscript) else out_arg
    assert isinstance(out_arg, (past.Name, past.TupleExpr))
    output_fields = out_arg.elts if isinstance(out_arg, past.TupleExpr) else [out_arg]
    output_arg_ids = set(arg.id for arg in output_fields)

    fields: dict[str, _FieldInfo] = {
        field_node.id: _FieldInfo(
            field=field_node,
            inp=(field_node.id in input_arg_ids),
            out=(field_node.id in output_arg_ids),
        )
        for field_node in fvprog.past_node.params
    }

    return fields


def format_io_string(fieldinfo: _FieldInfo) -> str:
    """Format the output for the "io" column: in/inout/out."""
    return f"{'in' if fieldinfo.inp else ''}{'out' if fieldinfo.out else ''}"


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
    """
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


def format_metadata(fvprog: Program, chains: Iterable[str], **kwargs: Any) -> str:
    """Format in/out field and connectivity information from a program as a string table."""
    field_infos = get_field_infos(fvprog)
    table = []
    for name, info in field_infos.items():
        display_type = info.field.type

        if isinstance(info.field.type, ct.ScalarType):
            display_type = ct.FieldType(dims=[], dtype=info.field.type)
        elif not isinstance(info.field.type, ct.FieldType):
            raise NotImplementedError("Found unsupported argument type.")

        table.append({"name": name, "type": display_type, "io": format_io_string(info)})
    kwargs.setdefault("tablefmt", "plain")
    return (
        ", ".join([chain for chain in chains])
        + "\n"
        + tabulate.tabulate(table, **kwargs)
    )


# TODO: provide a better typing for offset_provider
def generate_cpp_code(
    fencil: itir.FencilDefinition,
    offset_provider: dict[str, Dimension | Connectivity],
    **kwargs: Any,
) -> str:
    """Generate C++ code using the GTFN backend."""
    return generate(
        fencil,
        offset_provider=offset_provider,
        **kwargs,
    )


def import_definition(name: str) -> Program | FieldOperator | types.FunctionType:
    module_name, member_name = name.split(":")
    fencil = getattr(importlib.import_module(module_name), member_name)
    return fencil


def _is_size_param(param: itir.Sym) -> bool:
    """Check if parameter is a size parameter introduced by field view frontend."""
    return param.id.startswith("__") and "_size_" in param.id


def adapt_domain(fencil: itir.FencilDefinition) -> itir.FencilDefinition:
    """Replace field view size parameters by horizontal and vertical range paramters."""
    if len(fencil.closures) > 1:
        raise MultipleFieldOperatorException()

    fencil.closures[0].domain = itir.FunCall(
        fun=itir.SymRef(id="unstructured_domain"),
        args=[
            itir.FunCall(
                fun=itir.SymRef(id="named_range"),
                args=[
                    itir.AxisLiteral(value="horizontal"),
                    itir.SymRef(id="horizontal_start"),
                    itir.SymRef(id="horizontal_end"),
                ],
            ),
            itir.FunCall(
                fun=itir.SymRef(id="named_range"),
                args=[
                    itir.AxisLiteral(value=Koff.source.value),
                    itir.SymRef(id="vertical_start"),
                    itir.SymRef(id="vertical_end"),
                ],
            ),
        ],
    )
    return itir.FencilDefinition(
        id=fencil.id,
        function_definitions=fencil.function_definitions,
        params=[
            *(p for p in fencil.params if not _is_size_param(p)),
            itir.Sym(id="horizontal_start"),
            itir.Sym(id="horizontal_end"),
            itir.Sym(id="vertical_start"),
            itir.Sym(id="vertical_end"),
        ],
        closures=fencil.closures,
    )


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


@click.command(
    "icon4pygen",
)
@click.option(
    "--output-metadata",
    type=click.Path(
        exists=False, dir_okay=False, resolve_path=True, path_type=pathlib.Path
    ),
    help="file path for optional metadata output",
)
@click.argument("fencil", type=str)
def main(output_metadata: pathlib.Path, fencil: str) -> None:
    """
    Generate metadata and C++ code for an icon4py fencil.

    A fencil may be specified as <module>:<member>, where <module> is the
    dotted name of the containing module and <member> is the name of the fencil.
    """
    fencil_def = import_definition(fencil)
    fvprog = get_fvprog(fencil_def)
    offsets = scan_for_offsets(fvprog)
    offset_provider = {}
    for offset in offsets:
        offset_provider[offset] = provide_offset(offset)
    if output_metadata:
        connectivity_chains = [c for c in offsets if offset != Koff.value]
        output_metadata.write_text(format_metadata(fvprog, connectivity_chains))
    click.echo(generate_cpp_code(adapt_domain(fvprog.itir), offset_provider))
