# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Final, Optional, Type

from gt4py.next import Field, common
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

    ICON4PY_PRIMITIVE_DTYPES: Final = (
        type_alias.wpfloat,
        type_alias.vpfloat,
        float,
        bool,
        int32,
        int64,
    )
    DACE_PRIMITIVE_DTYPES: Final = (
        dace.float64,
        dace.float64 if type_alias.precision == "double" else dace.float32,
        dace.float64,
        dace.bool,
        dace.int32,
        dace.int64,
    )

    def stride_symbol_name_from_field(cls: Type, field_name: str, stride: int) -> str:
        return f"{cls.__name__}_{field_name}_s{stride}_sym"

    def gt4py_dim_to_dace_symbol(dim: common.Dimension) -> dace.symbol:
        # See dims.global_dimensions.values()
        # TODO(kotsaloscv): generalize this
        if "cell" in dim.value.lower():
            return CellDim_sym
        elif "edge" in dim.value.lower():
            return EdgeDim_sym
        elif "vertex" in dim.value.lower():
            return VertexDim_sym
        elif "k" == dim.value.lower():
            return KDim_sym
        else:
            raise ValueError(f"The dimension [{dim}] is not supported.")

    def dace_structure_dict(cls: Type) -> dict[str, dace.data.Array]:
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

            dace_dims = [gt4py_dim_to_dace_symbol(dim_) for dim_ in dims_]

            # Define DaCe Symbols: Field Sizes and Strides
            dace_symbols = {
                stride_symbol_name_from_field(cls, member_name, stride): dace.symbol(
                    stride_symbol_name_from_field(cls, member_name, stride)
                )
                for stride in range(len(dims_))
            }

            # TODO(kotsaloscv): how about StorageType (?)
            dace_structure_dict[member_name] = dace.data.Array(
                dtype=DACE_PRIMITIVE_DTYPES[ICON4PY_PRIMITIVE_DTYPES.index(dtype_)],
                shape=dace_dims,
                strides=[
                    dace_symbols[f"{cls.__name__}_{member_name}_s{0}_sym"],
                    dace_symbols[f"{cls.__name__}_{member_name}_s{1}_sym"],
                ],
            )

        return dace_structure_dict
