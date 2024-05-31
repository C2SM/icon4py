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
from gt4py.next.ffront.fbuiltins import Field, int32
from model.common.tests import field_type_aliases as fa

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_z import (
    _calculate_nabla2_for_z,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_of_theta import (
    _calculate_nabla2_of_theta,
)
from icon4py.model.common.dimension import CEDim, CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _calculate_nabla2_for_theta(
    kh_smag_e: fa.EKvpField,
    inv_dual_edge_length: fa.EwpField,
    theta_v: fa.CKwpField,
    geofac_div: Field[[CEDim], wpfloat],
) -> fa.CKvpField:
    z_nabla2_e = _calculate_nabla2_for_z(kh_smag_e, inv_dual_edge_length, theta_v)
    z_temp = _calculate_nabla2_of_theta(z_nabla2_e, geofac_div)
    return z_temp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def calculate_nabla2_for_theta(
    kh_smag_e: fa.EKfloatField,
    inv_dual_edge_length: fa.EfloatField,
    theta_v: fa.CKfloatField,
    geofac_div: Field[[CEDim], float],
    z_temp: fa.CKfloatField,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _calculate_nabla2_for_theta(
        kh_smag_e,
        inv_dual_edge_length,
        theta_v,
        geofac_div,
        out=z_temp,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
