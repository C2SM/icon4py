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

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_theta_and_exner import (
    apply_diffusion_to_theta_and_exner,
)
from icon4py.model.common.dimension import C2E2CDim, CECDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    flatten_first_two_dims,
    random_field,
    random_mask,
    unflatten_first_two_dims,
    zero_field,
    uses_icon_grid_with_otf
)

from .test_calculate_nabla2_for_z import calculate_nabla2_for_z_numpy
from .test_calculate_nabla2_of_theta import calculate_nabla2_of_theta_numpy
from .test_truly_horizontal_diffusion_nabla_of_theta_over_steep_points import (
    truly_horizontal_diffusion_nabla_of_theta_over_steep_points_numpy,
)
from .test_update_theta_and_exner import update_theta_and_exner_numpy


class TestApplyDiffusionToThetaAndExner(StencilTest):
    PROGRAM = apply_diffusion_to_theta_and_exner
    OUTPUTS = ("theta_v", "exner")

    @staticmethod
    def reference(
        grid,
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
        **kwargs,
    ):
        z_nabla2_e = calculate_nabla2_for_z_numpy(grid, kh_smag_e, inv_dual_edge_length, theta_v_in)
        z_temp = calculate_nabla2_of_theta_numpy(grid, z_nabla2_e, geofac_div)

        geofac_n2s_nbh = unflatten_first_two_dims(geofac_n2s_nbh)
        zd_vertoffset = unflatten_first_two_dims(zd_vertoffset)

        z_temp = truly_horizontal_diffusion_nabla_of_theta_over_steep_points_numpy(
            grid,
            mask,
            zd_vertoffset,
            zd_diffcoef,
            geofac_n2s_c,
            geofac_n2s_nbh,
            vcoef,
            theta_v_in,
            z_temp,
        )
        theta_v, exner = update_theta_and_exner_numpy(
            grid, z_temp, area, theta_v_in, exner, rd_o_cvd
        )
        return dict(theta_v=theta_v, exner=exner)

    @pytest.fixture
    def input_data(self, grid, uses_icon_grid_with_otf):
        if uses_icon_grid_with_otf:
            pytest.skip("Execution domain needs to be restricted or boundary taken into account in stencil.")

        kh_smag_e = random_field(grid, EdgeDim, KDim)
        inv_dual_edge_length = random_field(grid, EdgeDim)
        theta_v_in = random_field(grid, CellDim, KDim)
        geofac_div = random_field(grid, CEDim)
        mask = random_mask(grid, CellDim, KDim)
        zd_vertoffset = zero_field(grid, CellDim, C2E2CDim, KDim, dtype=int32)
        rng = np.random.default_rng()
        for k in range(grid.num_levels):
            # construct offsets that reach all k-levels except the last (because we are using the entries of this field with `+1`)
            zd_vertoffset[:, :, k] = rng.integers(
                low=0 - k,
                high=grid.num_levels - k - 1,
                size=(zd_vertoffset.shape[0], zd_vertoffset.shape[1]),
            )
        zd_diffcoef = random_field(grid, CellDim, KDim)
        geofac_n2s_c = random_field(grid, CellDim)
        geofac_n2s_nbh = random_field(grid, CellDim, C2E2CDim)
        vcoef = random_field(grid, CellDim, C2E2CDim, KDim)
        area = random_field(grid, CellDim)
        theta_v = random_field(grid, CellDim, KDim)
        exner = random_field(grid, CellDim, KDim)
        rd_o_cvd = 5.0

        vcoef_new = flatten_first_two_dims(CECDim, KDim, field=vcoef)
        zd_vertoffset_new = flatten_first_two_dims(CECDim, KDim, field=zd_vertoffset)
        geofac_n2s_nbh_new = flatten_first_two_dims(CECDim, field=geofac_n2s_nbh)

        return dict(
            kh_smag_e=kh_smag_e,
            inv_dual_edge_length=inv_dual_edge_length,
            theta_v_in=theta_v_in,
            geofac_div=geofac_div,
            mask=mask,
            zd_vertoffset=zd_vertoffset_new,
            zd_diffcoef=zd_diffcoef,
            geofac_n2s_c=geofac_n2s_c,
            geofac_n2s_nbh=geofac_n2s_nbh_new,
            vcoef=vcoef_new,
            area=area,
            theta_v=theta_v,
            exner=exner,
            rd_o_cvd=rd_o_cvd,
        )
