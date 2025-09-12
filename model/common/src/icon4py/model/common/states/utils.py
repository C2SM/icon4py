# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import TypeAlias, TypeVar

import gt4py.next as gtx
import xarray as xa
from gt4py.next.common import DimsT

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.utils import data_allocation as data_alloc


DimT = TypeVar("DimT", dims.KDim, dims.KHalfDim, dims.CellDim, dims.EdgeDim, dims.VertexDim)

FloatType: TypeAlias = ta.wpfloat | ta.vpfloat | float
IntegerType: TypeAlias = gtx.int32 | gtx.int64 | int
ScalarType: TypeAlias = FloatType | bool | IntegerType

T = TypeVar("T", ta.wpfloat, ta.vpfloat, float, bool, gtx.int32, gtx.int64)

GTXFieldType: TypeAlias = gtx.Field[DimsT, T]
FieldType: TypeAlias = gtx.Field[DimsT, T] | data_alloc.NDArray


def to_data_array(field: FieldType, attrs: dict):
    data = data_alloc.as_numpy(field)
    return xa.DataArray(data, attrs=attrs)
