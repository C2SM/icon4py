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

from icon4py.bindings.codegen.type_conversion import (
    BUILTIN_TO_CPP_TYPE,
    BUILTIN_TO_ISO_C_TYPE,
)
from icon4py.bindings.exceptions import BindingsRenderingException
from icon4py.bindings.locations import (
    BasicLocation,
    ChainedLocation,
    CompoundLocation,
)


class FieldRenderer:
    """A class to render Field attributes for their the respective c++ or f90 bindings."""

    @staticmethod
    def pointer(rank: int) -> str:
        return "" if rank == 0 else "*"

    @staticmethod
    def dim_tags(
        is_sparse: bool, is_dense: bool, is_compound: bool, has_vertical_dimension: bool
    ) -> str:
        if is_sparse:
            raise BindingsRenderingException(
                "can not render dimension tags for sparse field"
            )
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
            raise BindingsRenderingException("can not render sid of sparse field")

        if rank == 0:
            raise BindingsRenderingException("can not render sid of a scalar")

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
        if location not in _serializers:
            raise BindingsRenderingException(f"location {location} is not E,C or V")
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
            raise BindingsRenderingException(
                "stride type called on compound location or scalar"
            )

    @staticmethod
    def ctype(binding_type: str, field_type: str):
        match binding_type:
            case "f90":
                return BUILTIN_TO_ISO_C_TYPE[field_type]
            case "c++":
                return BUILTIN_TO_CPP_TYPE[field_type]
            case _:
                raise BindingsRenderingException(
                    f"binding_type {binding_type} needs to be either c++ or f90"
                )
