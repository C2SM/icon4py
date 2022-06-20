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
import importlib
import pathlib
from types import SimpleNamespace
from collections import namedtuple
from typing import Union

import click
import tabulate
import sys
from functional.ffront import common_types as ct
from functional.ffront import itir_makers as im
from functional.ffront import program_ast as past
from functional.ffront.decorator import FieldOperator, Program, program
from functional.iterator import ir as itir
from functional.iterator.backends.gtfn.gtfn_backend import generate
from icon4py.common.dimension import CellDim, EdgeDim, VertexDim
from icon4py.pyutils.icochainsize import ico_chain_size
from icon4py.pyutils.exceptions import MultipleFieldOperatorException

_FIELDINFO = namedtuple("_FIELDINFO", ["field", "inp", "out"])


def get_fieldinfo(fvprog: Program) -> dict[str, _FIELDINFO]:
    """Extract and format the in/out fields from a Program."""
    fields = {
        field.id: _FIELDINFO(field, True, False) for field in fvprog.past_node.params
    }

    for out_field in [fvprog.past_node.body[0].kwargs["out"]]:
        if out_field.id in [arg.id for arg in fvprog.past_node.body[0].args]:
            fields[out_field.id] = _FIELDINFO(fields[out_field.id].field, True, True)
        else:
            fields[out_field.id] = _FIELDINFO(out_field, False, True)

    return fields


def format_io_string(fieldinfo: _FIELDINFO) -> str:
    """Format the output for the "io" column: in/inout/out."""
    iostring = ""
    if fieldinfo.inp:
        iostring += "in"
    if fieldinfo.out:
        iostring += "out"
    return iostring


def scan_for_chains(fvprog: Program) -> list[str]:
    all_types = fvprog.past_node.pre_walk_values().if_isinstance(past.Symbol).getattr("type")
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
    all_dim_labels = [dim.value for dim in all_dims if dim.local]
    return set(all_offset_labels + all_dim_labels)

def provide_offset(chain) -> SimpleNamespace:
    location_chain = []
    for letter in chain:
        if letter == "C":
            location_chain.append(CellDim)
        elif letter == "E":
            location_chain.append(EdgeDim)
        elif letter == "V":
            location_chain.append(VertexDim)
    return SimpleNamespace(max_neighbors=ico_chain_size(location_chain), has_skip_values=True)


def tabulate_fields(fvprog: Program, chains, **kwargs) -> str:
    """Format in/out field information from a program as a string table."""
    fieldinfos = get_fieldinfo(fvprog)
    table = []
    for name, info in fieldinfos.items():
        display_type = info.field.type
        if not isinstance(info.field.type, ct.FieldType):
            display_type = ct.FieldType(dims=[], dtype=info.field.type)
        table.append({"name": name, "type": display_type, "io": format_io_string(info)})
    kwargs.setdefault("tablefmt", "plain")
    return (
        ", ".join([chain for chain in chains])
        + "\n"
        + tabulate.tabulate(table, **kwargs)
    )


def gtfn_program(fencil_function) -> Program:
    fvprog = program(fencil_function, backend="gtfn")
    adapt_program_gtfn(fvprog)
    return fvprog


def adapt_program_gtfn(fvprog):
    fvprog.itir.params.append(im.sym("domain_"))
    fvprog.itir.closures[0].domain = im.ref("domain_")
    return fvprog

def generate_cpp_code(fvprog, offset_provider, **kwargs) -> str:
    """Generate C++ code using the GTFN backend."""
    return generate(fvprog.itir, grid_type="unstructured", offset_provider=offset_provider, **kwargs)


def import_fencil(fencil: str) -> Union[Program, FieldOperator]:
    module_name, member_name = fencil.split(":")
    fencil = getattr(importlib.import_module(module_name), member_name)
    return fencil


def generate_cli(fencil_function):
    @click.command(
        fencil_function.__name__,
        help=f"Generate metadata and C++ code for {fencil_function.__name__}",
    )
    @click.option(
        "--output-metadata",
        type=click.Path(
            exists=False, dir_okay=False, resolve_path=True, path_type=pathlib.Path
        ),
    )
    def cli(output_metadata):
        fvprog = gtfn_program(fencil_function)
        chains = scan_for_chains(fvprog)
        offsets = {}
        for chain in chains:
            print(chain, file=sys.stderr)
            print(provide_offset(chain), file=sys.stderr)
            offsets[chain] = provide_offset(chain)
        if output_metadata:
            output_metadata.write_text(tabulate_fields(fvprog, chains))
        click.echo(generate_cpp_code(fvprog, offsets))

    return cli


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
def main(output_metadata, fencil):
    """
    Generate metadata and C++ code for an icon4py fencil.

    A fencil may be specified as <module>:<member>, where <module> is the
    dotted name of the containing module and <member> is the name of the fencil.
    """
    fencil = import_fencil(fencil)

    fvprog = None
    match fencil:
        case Program():
            fvprog = fencil.with_backend("gtfn")
        case FieldOperator():
            fvprog = fencil.with_backend("gtfn").as_program()
        case _:
            fvprog = program(fencil, backend="gtfn")

    if len(fvprog.past_node.body) > 1:
        raise MultipleFieldOperatorException()

    fvprog = adapt_program_gtfn(fvprog)
    chains = scan_for_chains(fvprog)
    offsets = {}
    for chain in chains:
        print(chain, file=sys.stderr)
        print(provide_offset(chain), file=sys.stderr)
        offsets[chain] = provide_offset(chain)
    if output_metadata:
        output_metadata.write_text(tabulate_fields(fvprog, chains))
    click.echo(generate_cpp_code(fvprog, offsets))
