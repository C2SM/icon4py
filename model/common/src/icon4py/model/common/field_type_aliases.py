# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import TypeAlias, TypeVar

from gt4py.next import Dims, Field
from gt4py.next.ffront.fbuiltins import int32, int64

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, VertexDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


T = TypeVar("T", wpfloat, vpfloat, float, bool, int32, int64)

CellField: TypeAlias = Field[Dims[CellDim], T]
EdgeField: TypeAlias = Field[Dims[EdgeDim], T]
VertexField: TypeAlias = Field[Dims[VertexDim], T]
KField: TypeAlias = Field[Dims[KDim], T]

CellKField: TypeAlias = Field[Dims[CellDim, KDim], T]
EdgeKField: TypeAlias = Field[Dims[EdgeDim, KDim], T]
VertexKField: TypeAlias = Field[Dims[VertexDim, KDim], T]
