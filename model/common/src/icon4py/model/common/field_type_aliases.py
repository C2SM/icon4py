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

from gt4py.next import Dims, Field
from gt4py.next.ffront.fbuiltins import int32, int64

from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat, wpfloat


T = TypeVar("T", wpfloat, vpfloat, float, bool, int32, int64)

CellField: TypeAlias = Field[Dims[dims.CellDim], T]
EdgeField: TypeAlias = Field[Dims[dims.EdgeDim], T]
VertexField: TypeAlias = Field[Dims[dims.VertexDim], T]
KField: TypeAlias = Field[Dims[dims.KDim], T]

CellKField: TypeAlias = Field[Dims[dims.CellDim, dims.KDim], T]
EdgeKField: TypeAlias = Field[Dims[dims.EdgeDim, dims.KDim], T]
VertexKField: TypeAlias = Field[Dims[dims.VertexDim, dims.KDim], T]
