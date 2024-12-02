# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Sequence, TypeAlias, TypeVar, Union

import gt4py.next as gtx
import numpy as np
import xarray as xa

from icon4py.model.common import dimension as dims, type_alias as ta


DimT = TypeVar("DimT", dims.KDim, dims.KHalfDim, dims.CellDim, dims.EdgeDim, dims.VertexDim)

FloatType: TypeAlias = Union[ta.wpfloat, ta.vpfloat, float]
IntegerType: TypeAlias = Union[gtx.int32, gtx.int64, int]
ScalarType: TypeAlias = Union[FloatType, bool, IntegerType]

T = TypeVar("T", ta.wpfloat, ta.vpfloat, float, bool, gtx.int32, gtx.int64)

FieldType: TypeAlias = Union[gtx.Field[Sequence[gtx.Dims[DimT]], T], alloc.NDArray]


def to_data_array(field: FieldType, attrs: dict):
    data = field if isinstance(field, np.ndarray) else field.ndarray
    return xa.DataArray(data, attrs=attrs)
