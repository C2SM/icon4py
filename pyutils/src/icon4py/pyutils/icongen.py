# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Utilities for generating icon stencils."""
import pathlib
from collections import namedtuple

import click
import tabulate

from functional.ffront.decorator import Program, program
from functional.ffront import itir_makers as im
from functional.iterator.backends.gtfn.gtfn_backend import generate


_FIELDINFO = namedtuple("_FIELDINFO", ["field", "inp", "out"])


def get_fieldinfo(fvprog: Program) -> dict[str, _FIELDINFO]:
    """Extract and format the in/out fields from a Program."""
    fields = {field.id: _FIELDINFO(field, True, False) for field in fvprog.past_node.body[0].args}
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
    table = [
        {"name": name, "type": info.field.type, "io": format_io_string(info)}
        for name, info in fieldinfos.items()
    ]
    kwargs.setdefault("tablefmt", "plain")
    return tabulate.tabulate(table, **kwargs)


def gtfn_program(fencil_function) -> Program:
    fvprog = program(fencil_function, backend="gtfn")
    fvprog.itir.params.append(im.sym("domain_"))
    fvprog.itir.closures[0].domain = im.ref("domain_")
    return fvprog


def generate_cpp_code(fvprog) -> str:
    """Generate C++ code using the GTFN backend."""
    return generate(fvprog.itir, grid_type="unstructured")


def generate_cli(fencil_function):
    @click.command(
        fencil_function.__name__,
        help=f"Generate metadata and C++ code for {fencil_function.__name__}",
    )
    @click.option(
        "--output-metadata",
        type=click.Path(exists=False, dir_okay=False, resolve_path=True, path_type=pathlib.Path),
    )
    def cli(output_metadata):
        fvprog = gtfn_program(fencil_function)
        output_metadata.write_text(tabulate_fields(fvprog))
        click.echo(generate_cpp_code(fvprog))

    return cli
