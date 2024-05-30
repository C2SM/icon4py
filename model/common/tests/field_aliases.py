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

from typing import TypeAlias

from gt4py.next import Dims, Field
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, VertexDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


CwpField: TypeAlias = Field[Dims[CellDim], wpfloat]
EwpField: TypeAlias = Field[Dims[EdgeDim], wpfloat]
VwpField: TypeAlias = Field[Dims[VertexDim], wpfloat]
KwpField: TypeAlias = Field[Dims[KDim], wpfloat]
CKwpField: TypeAlias = Field[Dims[CellDim, KDim], wpfloat]
EKwpField: TypeAlias = Field[Dims[EdgeDim, KDim], wpfloat]

CKvpField: TypeAlias = Field[Dims[CellDim, KDim], vpfloat]

CintField: TypeAlias = Field[Dims[CellDim], int32]
EintField: TypeAlias = Field[Dims[EdgeDim], int32]
KintField: TypeAlias = Field[Dims[KDim], int32]
EKintField: TypeAlias = Field[Dims[EdgeDim, KDim], int32]

CboolField: TypeAlias = Field[Dims[CellDim], bool]
EboolField: TypeAlias = Field[Dims[EdgeDim], bool]
EKboolField: TypeAlias = Field[Dims[EdgeDim, KDim], bool]
