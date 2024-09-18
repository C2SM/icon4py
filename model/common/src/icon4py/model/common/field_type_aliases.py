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

from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat, wpfloat


T = TypeVar("T", wpfloat, vpfloat, float, bool, int32, int64)

CellField: TypeAlias = Field[Dims[dims.CellDim], T]
EdgeField: TypeAlias = Field[Dims[dims.EdgeDim], T]
VertexField: TypeAlias = Field[Dims[dims.VertexDim], T]
KField: TypeAlias = Field[Dims[dims.KDim], T]
KHalfField: TypeAlias = Field[Dims[dims.KHalfDim], T]

CellKField: TypeAlias = Field[Dims[dims.CellDim, dims.KDim], T]
EdgeKField: TypeAlias = Field[Dims[dims.EdgeDim, dims.KDim], T]
VertexKField: TypeAlias = Field[Dims[dims.VertexDim, dims.KDim], T]
