# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import sys
from typing import Final, Optional, Type, Union, get_args, get_origin, get_type_hints

from gt4py.next import Field, common
from gt4py.next.ffront.fbuiltins import int32, int64

from icon4py.model.common import type_alias, utils as common_utils


try:
    import dace
except ImportError:
    from types import ModuleType

    dace: Optional[ModuleType] = None  # type: ignore[no-redef]


if dace:
    ICON4PY_PRIMITIVE_DTYPES: Final = (
        type_alias.wpfloat,
        type_alias.vpfloat,
        float,
        bool,
        int32,
        int64,
        int,
    )
    DACE_PRIMITIVE_DTYPES: Final = (
        dace.float64,
        dace.float64 if type_alias.precision == "double" else dace.float32,
        dace.float64,
        dace.bool,
        dace.int32,
        dace.int64,
        dace.int64 if sys.maxsize > 2**32 else dace.int32,
    )

    CellDim_sym = dace.symbol("CellDim_sym")
    EdgeDim_sym = dace.symbol("EdgeDim_sym")
    VertexDim_sym = dace.symbol("VertexDim_sym")
    KDim_sym = dace.symbol("KDim_sym")
    KHalfDim_sym = dace.symbol("KHalfDim_sym")

    def gt4py_dim_to_dace_symbol(dim: common.Dimension) -> dace.symbol:
        if "cell" == dim.value.lower():
            return CellDim_sym
        elif "edge" == dim.value.lower():
            return EdgeDim_sym
        elif "vertex" == dim.value.lower():
            return VertexDim_sym
        elif "k" == dim.value.lower():
            return KDim_sym
        elif "khalf" == dim.value.lower():
            return KHalfDim_sym
        else:
            raise ValueError(f"The dimension [{dim}] is not supported.")

    def symbol_name_for_field(
        cls: Type | str,
        field_name: str,
        field_atrr: str,
        axis: int,
        pair_member: Optional[str] = None,
    ) -> str:
        if not pair_member:
            return f"{cls if isinstance(cls, str) else cls.__name__}_{field_name}_{field_atrr}_{axis}_sym"
        else:
            return f"{cls if isinstance(cls, str) else cls.__name__}_{field_name}_{pair_member}_{field_atrr}_{axis}_sym"

    def is_optional_type(tp) -> bool:
        return get_origin(tp) is Union and type(None) in get_args(tp)

    def dace_structure_dict(cls: Type) -> dict[str, dace.data.Array]:
        """
        Function that returns a dictionary to be used to define DaCe Structures based on the provided data class.
        The function extracts the GT4Py field members of the data class and builds the dictionary accordingly.
        """
        if not hasattr(cls, "__dataclass_fields__"):
            raise ValueError("The provided class is not a data class.")

        dace_structure_dict_ = {}

        type_hints = get_type_hints(cls)
        for member_name, dataclass_field in cls.__dataclass_fields__.items():
            if not hasattr(dataclass_field, "type"):
                continue
            type_ = type_hints[member_name]
            if not hasattr(type_, "__origin__"):
                continue

            if is_optional_type(type_):
                type_ = get_args(type_)[0]

            if type_.__origin__ is Field:
                dims_ = type_.__args__[0].__args__  # dimensions of the field
                dtype_ = type_.__args__[1]  # data type of the field

                # Define DaCe Symbols: Field Sizes and Strides
                dace_symbols = {
                    symbol_name_for_field(cls, member_name, "size", axis): gt4py_dim_to_dace_symbol(
                        dims_[axis]
                    )
                    for axis in range(len(dims_))
                }
                dace_symbols |= {
                    symbol_name_for_field(cls, member_name, "stride", axis): dace.symbol(
                        symbol_name_for_field(cls, member_name, "stride", axis)
                    )
                    for axis in range(len(dims_))
                }

                # TODO(kotsaloscv): how about StorageType (?)
                dace_structure_dict_[member_name] = dace.data.Array(
                    dtype=DACE_PRIMITIVE_DTYPES[ICON4PY_PRIMITIVE_DTYPES.index(dtype_)],
                    # By not providing the shape as Edge, Cell, K, etc., we avoid the KHalfDim issue.
                    shape=[
                        dace_symbols[f"{cls.__name__}_{member_name}_size_0_sym"],
                        dace_symbols[f"{cls.__name__}_{member_name}_size_1_sym"],
                    ],
                    strides=[
                        dace_symbols[f"{cls.__name__}_{member_name}_stride_0_sym"],
                        dace_symbols[f"{cls.__name__}_{member_name}_stride_1_sym"],
                    ],
                )
            elif type_.__origin__ is common_utils.PredictorCorrectorPair:
                dims_ = type_.__args__[0].__args__[0].__args__  # dimensions of the field
                dtype_ = type_.__args__[0].__args__[1]  # data type of the field

                # Define DaCe Symbols: Field Sizes and Strides
                dace_symbols = {
                    symbol_name_for_field(
                        cls, member_name, "size", axis, pair_member
                    ): gt4py_dim_to_dace_symbol(dims_[axis])
                    for axis in range(len(dims_))
                    for pair_member in ("predictor", "corrector")
                }
                dace_symbols |= {
                    symbol_name_for_field(
                        cls, member_name, "stride", axis, pair_member
                    ): dace.symbol(
                        symbol_name_for_field(cls, member_name, "stride", axis, pair_member)
                    )
                    for axis in range(len(dims_))
                    for pair_member in ("predictor", "corrector")
                }

                predictor = dace.data.Array(
                    dtype=DACE_PRIMITIVE_DTYPES[ICON4PY_PRIMITIVE_DTYPES.index(dtype_)],
                    shape=[
                        dace_symbols[f"{cls.__name__}_{member_name}_predictor_size_0_sym"],
                        dace_symbols[f"{cls.__name__}_{member_name}_predictor_size_1_sym"],
                    ],
                    strides=[
                        dace_symbols[f"{cls.__name__}_{member_name}_predictor_stride_0_sym"],
                        dace_symbols[f"{cls.__name__}_{member_name}_predictor_stride_1_sym"],
                    ],
                )

                corrector = dace.data.Array(
                    dtype=DACE_PRIMITIVE_DTYPES[ICON4PY_PRIMITIVE_DTYPES.index(dtype_)],
                    shape=[
                        dace_symbols[f"{cls.__name__}_{member_name}_corrector_size_0_sym"],
                        dace_symbols[f"{cls.__name__}_{member_name}_corrector_size_1_sym"],
                    ],
                    strides=[
                        dace_symbols[f"{cls.__name__}_{member_name}_corrector_stride_0_sym"],
                        dace_symbols[f"{cls.__name__}_{member_name}_corrector_stride_1_sym"],
                    ],
                )

                dace_structure_dict_[member_name] = dace.data.Structure(
                    dict(
                        predictor=predictor,
                        corrector=corrector,
                    ),
                    name=f"{cls.__name__}_{member_name}_Struct",
                )
                pass
            else:
                raise ValueError(f"The type {type_} is not supported.")

        return dace_structure_dict_
