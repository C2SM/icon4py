# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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

from typing import Optional, Type

from gt4py.next import Field
from gt4py.next.ffront.fbuiltins import int32, int64

from icon4py.model.common import type_alias


try:
    import dace
except ImportError:
    from types import ModuleType

    dace: Optional[ModuleType] = None  # type: ignore[no-redef]


if dace:
    CellDim_sym = dace.symbol("CellDim_sym")
    EdgeDim_sym = dace.symbol("EdgeDim_sym")
    VertexDim_sym = dace.symbol("VertexDim_sym")
    KDim_sym = dace.symbol("KDim_sym")

    icon4py_primitive_dtypes = (type_alias.wpfloat, type_alias.vpfloat, float, bool, int32, int64)
    dace_primitive_dtypes = (
        dace.float64,
        dace.float64 if type_alias.precision == "double" else dace.float32,
        dace.float64,
        dace.bool,
        dace.int32,
        dace.int64,
    )

    def stride_symbol_name_from_field(cls: Type, field_name: str, stride: int) -> str:
        return f"{cls.__name__}_{field_name}_s{stride}_sym"

    def dace_structure_dict(cls):
        """
        Function that returns a dictionary to be used to define DaCe Structures based on the provided data class.
        The function extracts the GT4Py field members of the data class and builds the dictionary accordingly.
        """
        if not hasattr(cls, "__dataclass_fields__"):
            raise ValueError("The provided class is not a data class.")

        dace_structure_dict = {}

        for member_name, dataclass_field in cls.__dataclass_fields__.items():
            if not hasattr(dataclass_field, "type"):
                continue
            type_ = dataclass_field.type
            if not hasattr(type_, "__origin__"):
                continue
            # TODO(kotsaloscv): DaCe Structure with GT4Py Fields. Disregard the rest of the fields.
            if type_.__origin__ is not Field:
                continue

            dims_ = type_.__args__[0].__args__  # dimensions of the field
            dtype_ = type_.__args__[1]  # data type of the field

            dace_dims = []
            for dim_ in dims_:
                if "cell" in dim_.value.lower():
                    dace_dims.append(CellDim_sym)
                elif "edge" in dim_.value.lower():
                    dace_dims.append(EdgeDim_sym)
                elif "vertex" in dim_.value.lower():
                    dace_dims.append(VertexDim_sym)
                elif "k" == dim_.value.lower():
                    dace_dims.append(KDim_sym)
                else:
                    raise ValueError(f"The dimension [{dim_}] is not supported.")

            # Define DaCe Symbols: Field Sizes and Strides
            dace_symbols = {
                stride_symbol_name_from_field(cls, member_name, stride): dace.symbol(
                    stride_symbol_name_from_field(cls, member_name, stride)
                )
                for stride in range(len(dims_))
            }

            # TODO(kotsaloscv): how about StorageType (?)
            dace_structure_dict[member_name] = dace.data.Array(
                dtype=dace_primitive_dtypes[icon4py_primitive_dtypes.index(dtype_)],
                shape=dace_dims,
                strides=[
                    dace_symbols[f"{cls.__name__}_{member_name}_s{0}_sym"],
                    dace_symbols[f"{cls.__name__}_{member_name}_s{1}_sym"],
                ],
            )

        return dace_structure_dict
