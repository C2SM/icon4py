# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Sequence, TypeAlias, TypeVar, Union

import gt4py.next as gtx
import xarray as xa

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.utils import data_allocation as data_alloc


DimT = TypeVar("DimT", dims.KDim, dims.KHalfDim, dims.CellDim, dims.EdgeDim, dims.VertexDim)

FloatType: TypeAlias = Union[ta.wpfloat, ta.vpfloat, float]
IntegerType: TypeAlias = Union[gtx.int32, gtx.int64, int]
ScalarType: TypeAlias = Union[FloatType, bool, IntegerType]

T = TypeVar("T", ta.wpfloat, ta.vpfloat, float, bool, gtx.int32, gtx.int64)

GTXFieldType: TypeAlias = gtx.Field[Sequence[gtx.Dims[DimT]], T]
FieldType: TypeAlias = Union[gtx.Field[Sequence[gtx.Dims[DimT]], T], data_alloc.NDArray]


def to_data_array(field: FieldType, attrs: dict):
    data = data_alloc.as_numpy(field)
    return xa.DataArray(data, attrs=attrs)
