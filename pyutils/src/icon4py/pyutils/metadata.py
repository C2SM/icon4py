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

from collections.abc import Iterable
from typing import Any, TypeGuard

import tabulate
from functional.ffront import common_types as ct
from functional.ffront import program_ast as past
from functional.ffront.decorator import Program

from icon4py.pyutils.stencil_info import FieldInfo


def format_io_string(fieldinfo: FieldInfo) -> str:
    """Format the output for the "io" column: in/inout/out."""
    return f"{'in' if fieldinfo.inp else ''}{'out' if fieldinfo.out else ''}"


def is_list_of_names(obj: Any) -> TypeGuard[list[past.Name]]:
    return isinstance(obj, list) and all(isinstance(i, past.Name) for i in obj)


def get_field_infos(fvprog: Program) -> dict[str, FieldInfo]:
    """Extract and format the in/out fields from a Program."""
    assert is_list_of_names(
        fvprog.past_node.body[0].args
    ), "Found unsupported expression in input arguments."
    input_arg_ids = set(arg.id for arg in fvprog.past_node.body[0].args)

    out_arg = fvprog.past_node.body[0].kwargs["out"]
    assert isinstance(out_arg, (past.Name, past.TupleExpr))
    output_fields = out_arg.elts if isinstance(out_arg, past.TupleExpr) else [out_arg]
    output_arg_ids = set(arg.id for arg in output_fields)

    fields: dict[str, FieldInfo] = {
        field_node.id: FieldInfo(
            field=field_node,
            inp=(field_node.id in input_arg_ids),
            out=(field_node.id in output_arg_ids),
        )
        for field_node in fvprog.past_node.params
    }

    return fields


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
