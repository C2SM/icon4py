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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32
from model.common.tests import field_type_aliases as fa

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import backend


@field_operator
def _diagnose_temperature(
    theta_v: fa.CKvpField,
    exner: fa.CKvpField,
) -> fa.CKvpField:
    temperature = theta_v * exner
    return temperature


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def diagnose_temperature(
    theta_v: fa.CKvpField,
    exner: fa.CKvpField,
    temperature: fa.CKvpField,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _diagnose_temperature(
        theta_v,
        exner,
        out=(temperature),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
