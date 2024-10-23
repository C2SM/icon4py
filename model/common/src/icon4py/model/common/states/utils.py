# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import enum
from typing import Protocol, Sequence, TypeAlias, TypeVar, Union

import gt4py.next as gtx
import xarray as xa

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.settings import xp
from icon4py.model.common.states import model


T = TypeVar("T", ta.wpfloat, ta.vpfloat, float, bool, gtx.int32, gtx.int64)
DimT = TypeVar("DimT", dims.KDim, dims.KHalfDim, dims.CellDim, dims.EdgeDim, dims.VertexDim)
FloatType: TypeAlias = Union[ta.wpfloat, ta.vpfloat, float]
IntegerType: TypeAlias = Union[gtx.int32, gtx.int64, int]
ScalarType: TypeAlias = Union[FloatType, bool, IntegerType]


FieldType: TypeAlias = Union[gtx.Field[Sequence[gtx.Dims[DimT]], T], xp.ndarray]


def to_data_array(field: FieldType, attrs: dict):
    data = field if isinstance(field, xp.ndarray) else field.ndarray
    return xa.DataArray(data, attrs=attrs)


class RetrievalType(enum.Enum):
    FIELD = 0
    DATA_ARRAY = 1
    METADATA = 2


class FieldSource(Protocol):
    """Protocol for object that can be queried for fields."""

    def get(
        self, field_name: str, type_: RetrievalType = RetrievalType.FIELD
    ) -> Union[FieldType, xa.DataArray, model.FieldMetaData]:
        ...
