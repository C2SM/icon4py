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
from collections import namedtuple

import click
import tabulate
from functional.ffront import common_types as ct
from functional.ffront import itir_makers as im
from functional.ffront.decorator import FieldOperator, Program, program
from functional.iterator.backends.gtfn.gtfn_backend import generate


_FIELDINFO = namedtuple("_FIELDINFO", ["field", "inp", "out"])


def get_fieldinfo(fvprog: Program) -> dict[str, _FIELDINFO]:
    """Extract and format the in/out fields from a Program."""
    assert len(fvprog.past_node.body) == 1
    fields = {
        field.id: _FIELDINFO(field, True, False)
        for field in fvprog.past_node.body[0].args
    }
    for field in [fvprog.past_node.body[0].kwargs["out"]]:
        if field.id in fields:
            fields[field.id] = _FIELDINFO(fields[field.id].field, True, True)
        else:
            fields[field.id] = _FIELDINFO(field, False, True)
    return fields


def format_io_string(fieldinfo: _FIELDINFO) -> str:
    """Format the output for the "io" column: in/inout/out."""
    iostring = ""
    if fieldinfo.inp:
        iostring += "in"
    if fieldinfo.out:
        iostring += "out"
    return iostring


def tabulate_fields(fvprog: Program, **kwargs) -> str:
    """Format in/out field information from a program as a string table."""
    fieldinfos = get_fieldinfo(fvprog)
    table = []
    for name, info in fieldinfos.items():
        display_type = info.field.type
        if not isinstance(info.field.type, ct.FieldType):
            display_type = ct.FieldType(dims=[], dtype=info.field.type)
        table.append({"name": name, "type": display_type, "io": format_io_string(info)})
    kwargs.setdefault("tablefmt", "plain")
    return tabulate.tabulate(table, **kwargs)


def gtfn_program(fencil_function) -> Program:
    fvprog = program(fencil_function, backend="gtfn")
    adapt_program_gtfn(fvprog)
    return fvprog


def adapt_program_gtfn(fvprog):
    fvprog.itir.params.append(im.sym("domain_"))
    fvprog.itir.closures[0].domain = im.ref("domain_")
    return fvprog


def generate_cpp_code(fvprog, **kwargs) -> str:
    """Generate C++ code using the GTFN backend."""
    return generate(fvprog.itir, grid_type="unstructured", **kwargs)


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
        if output_metadata:
            output_metadata.write_text(tabulate_fields(fvprog))
        click.echo(generate_cpp_code(fvprog))

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
    module_name, member_name = fencil.split(":")
    fencil = getattr(importlib.import_module(module_name), member_name)

    fvprog = None
    match fencil:
        case Program():
            fvprog = fencil.with_backend("gtfn")
        case FieldOperator():
            fvprog = fencil.with_backend("gtfn").as_program()
        case _:
            fvprog = program(fencil, backend="gtfn")

    fvprog = adapt_program_gtfn(fvprog)
    if output_metadata:
        output_metadata.write_text(tabulate_fields(fvprog))
    click.echo(generate_cpp_code(fvprog))
