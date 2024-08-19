# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import TypeAlias, TypeVar

import gt4py.next as gtx

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, KHalfDim, VertexDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


T = TypeVar("T", wpfloat, vpfloat, float, bool, gtx.int32, gtx.int64)

CellField: TypeAlias = gtx.Field[gtx.Dims[CellDim], T]
EdgeField: TypeAlias = gtx.Field[gtx.Dims[EdgeDim], T]
VertexField: TypeAlias = gtx.Field[gtx.Dims[VertexDim], T]
KField: TypeAlias = gtx.Field[gtx.Dims[KDim], T]

CellKField: TypeAlias = gtx.Field[gtx.Dims[CellDim, KDim], T]
EdgeKField: TypeAlias = gtx.Field[gtx.Dims[EdgeDim, KDim], T]
VertexKField: TypeAlias = gtx.Field[gtx.Dims[VertexDim, KDim], T]

CellKHalfField: TypeAlias = gtx.Field[gtx.Dims[CellDim, KHalfDim], T]
