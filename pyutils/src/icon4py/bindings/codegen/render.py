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

from typing import Union

from functional.ffront.common_types import ScalarKind

from icon4py.bindings.locations import (
    BasicLocation,
    ChainedLocation,
    CompoundLocation,
)


BUILTIN_TO_ISO_C_TYPE = {
    ScalarKind.FLOAT64: "real(c_double)",
    ScalarKind.FLOAT32: "real(c_float)",
    ScalarKind.BOOL: "logical(c_int)",
    ScalarKind.INT32: "c_int",
    ScalarKind.INT64: "c_long",
}

BUILTIN_TO_CPP_TYPE = {
    ScalarKind.FLOAT64: "double",
    ScalarKind.FLOAT32: "float",
    ScalarKind.BOOL: "int",
    ScalarKind.INT32: "int",
    ScalarKind.INT64: "long",
}


class FieldRenderer:
    @staticmethod
    def pointer(rank: int) -> str:
        return "" if rank == 0 else "*"

    @staticmethod
    def dim_tags(
        is_sparse: bool, is_dense: bool, is_compound: bool, has_vertical_dimension: bool
    ) -> str:
        if is_sparse:
            raise Exception("can not render dimension tags for sparse field")
        tags = []
        if is_dense or is_compound:
            tags.append("unstructured::dim::horizontal")
        if has_vertical_dimension:
            tags.append("unstructured::dim::vertical")
        return ",".join(tags)

    @staticmethod
    def sid(
        is_sparse: bool,
        is_dense: bool,
        is_compound: bool,
        has_vertical_dimension: bool,
        name: str,
        rank: int,
        location: Union[BasicLocation, CompoundLocation, ChainedLocation],
    ) -> str:
        if is_sparse:
            raise Exception("can not render sid of sparse field")

        if rank == 0:
            raise Exception("can not render sid of a scalar")

        values_str = (
            "1"
            if rank == 1 or is_compound
            else f"1, mesh_.{FieldRenderer.stride_type(is_dense, is_sparse, location)}"
        )
        return f"get_sid({name}_, gridtools::hymap::keys<{FieldRenderer.dim_tags(is_sparse, is_dense, is_compound, has_vertical_dimension)}>::make_values({values_str}))"

    @staticmethod
    def ranked_dim_string(rank: int) -> str:
        return "dimension(" + ",".join([":"] * rank) + ")" if rank != 0 else "value"

    @staticmethod
    def serialise_func(location: str) -> str:
        _serializers = {
            "E": "serialize_dense_edges",
            "C": "serialize_dense_cells",
            "V": "serialize_dense_verts",
        }
        return _serializers[location]

    @staticmethod
    def dim_string(rank: int):
        return "dimension(*)" if rank != 0 else "value"

    @staticmethod
    def stride_type(
        is_dense: bool,
        is_sparse: bool,
        location: Union[BasicLocation, CompoundLocation, ChainedLocation],
    ):
        _strides = {"E": "EdgeStride", "C": "CellStride", "V": "VertexStride"}
        if is_dense:
            return _strides[str(location)]
        elif is_sparse:
            return _strides[str(location[0])]
        else:
            raise Exception("stride type called on compound location or scalar")

    @staticmethod
    def ctype(binding_type: str, field_type: str):
        match binding_type:
            case "f90":
                return BUILTIN_TO_ISO_C_TYPE[field_type]
            case "c++":
                return BUILTIN_TO_CPP_TYPE[field_type]


class OffsetRenderer:
    # todo: these shorthands can potentially be improved after regression testing

    @staticmethod
    def lowercase_shorthand(
        is_compound_location: bool,
        includes_center: bool,
        target: tuple[BasicLocation, ChainedLocation],
    ) -> str:
        if is_compound_location:
            lhs = str(target[0]).lower()
            rhs = "".join([char for char in str(target[1]) if char != "2"]).lower()
            return f"{lhs}2{rhs}" + ("o" if includes_center else "")
        else:
            return "".join([char for char in str(target[1]) if char != "2"]).lower() + (
                "o" if includes_center else ""
            )

    @staticmethod
    def uppercase_shorthand(
        is_compound_location: bool,
        includes_center: bool,
        target: tuple[BasicLocation, ChainedLocation],
    ) -> str:
        if is_compound_location:
            return OffsetRenderer.lowercase_shorthand(
                is_compound_location, includes_center, target
            ).upper()
        else:
            return str(target[1]) + ("O" if includes_center else "")
