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
