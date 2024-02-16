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

from enum import Enum

from gt4py.next.common import Dimension
from gt4py.next.type_system.type_specifications import FieldType, ScalarKind, ScalarType, TypeSpec


def build_array_size_args() -> dict[str, str]:
    array_size_args = {}
    from icon4py.model.common import dimension

    for var_name, var in vars(dimension).items():
        if isinstance(var, Dimension):
            dim_name = var_name.replace(
                "Dim", ""
            )  # Assumes we keep suffixing each Dimension with Dim
            size_name = f"n_{dim_name}"
            array_size_args[dim_name] = size_name
    return array_size_args


class Backend(Enum):
    CPU = "run_gtfn"
    GPU = "run_gtfn_gpu"
    ROUNDTRIP = "run_roundtrip"


def parse_type_spec(type_spec: TypeSpec) -> tuple[list[Dimension], ScalarKind]:
    if isinstance(type_spec, ScalarType):
        return [], type_spec.kind
    elif isinstance(type_spec, FieldType):
        return type_spec.dims, type_spec.dtype.kind
    else:
        raise ValueError(f"Unsupported type specification: {type_spec}")


def flatten_and_get_unique_elts(list_of_lists: list[list[str]]) -> list[str]:
    return sorted(set(item for sublist in list_of_lists for item in sublist))
