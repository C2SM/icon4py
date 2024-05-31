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
from icon4py.model.atmosphere.diffusion.stencils.truly_horizontal_diffusion_nabla_of_theta_over_steep_points import (
    _truly_horizontal_diffusion_nabla_of_theta_over_steep_points,
)
from icon4py.model.atmosphere.diffusion.stencils.update_theta_and_exner import (
    _update_theta_and_exner,
)
from icon4py.model.common.dimension import CECDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _apply_diffusion_to_theta_and_exner(
    kh_smag_e: Field[[EdgeDim, KDim], vpfloat],
    inv_dual_edge_length: fa.EwpField,
    theta_v_in: fa.CKwpField,
    geofac_div: Field[[CEDim], wpfloat],
    mask: Field[[CellDim, KDim], bool],
    zd_vertoffset: Field[[CECDim, KDim], int32],
    zd_diffcoef: fa.CKwpField,
    geofac_n2s_c: fa.CwpField,
    geofac_n2s_nbh: Field[[CECDim], wpfloat],
    vcoef: Field[[CECDim, KDim], wpfloat],
    area: fa.CwpField,
    exner: fa.CKwpField,
    rd_o_cvd: vpfloat,
) -> tuple[fa.CKwpField, fa.CKwpField]:
    z_nabla2_e = _calculate_nabla2_for_z(kh_smag_e, inv_dual_edge_length, theta_v_in)
    z_temp = _calculate_nabla2_of_theta(z_nabla2_e, geofac_div)
    z_temp = _truly_horizontal_diffusion_nabla_of_theta_over_steep_points(
        mask,
        zd_vertoffset,
        zd_diffcoef,
        geofac_n2s_c,
        geofac_n2s_nbh,
        vcoef,
        theta_v_in,
        z_temp,
    )
    theta_v, exner = _update_theta_and_exner(z_temp, area, theta_v_in, exner, rd_o_cvd)
    return theta_v, exner


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def apply_diffusion_to_theta_and_exner(
    kh_smag_e: Field[[EdgeDim, KDim], vpfloat],
    inv_dual_edge_length: fa.EwpField,
    theta_v_in: fa.CKwpField,
    geofac_div: Field[[CEDim], wpfloat],
    mask: Field[[CellDim, KDim], bool],
    zd_vertoffset: Field[[CECDim, KDim], int32],
    zd_diffcoef: fa.CKwpField,
    geofac_n2s_c: fa.CwpField,
    geofac_n2s_nbh: Field[[CECDim], wpfloat],
    vcoef: Field[[CECDim, KDim], wpfloat],
    area: fa.CwpField,
    theta_v: fa.CKwpField,
    exner: fa.CKwpField,
    rd_o_cvd: vpfloat,
):
    _apply_diffusion_to_theta_and_exner(
        kh_smag_e,
        inv_dual_edge_length,
        theta_v_in,
        geofac_div,
        mask,
        zd_vertoffset,
        zd_diffcoef,
        geofac_n2s_c,
        geofac_n2s_nbh,
        vcoef,
        area,
        exner,
        rd_o_cvd,
        out=(theta_v, exner),
    )
