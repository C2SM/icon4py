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

from icon4py.model.atmosphere.diffusion.stencils.truly_horizontal_diffusion_nabla_of_theta_over_steep_points import (
    truly_horizontal_diffusion_nabla_of_theta_over_steep_points,
)
from icon4py.model.common.dimension import C2E2CDim, CECDim, CellDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    flatten_first_two_dims,
    random_field,
    random_mask,
    zero_field,
)


def truly_horizontal_diffusion_nabla_of_theta_over_steep_points_numpy(
    grid,
    mask: np.array,
    zd_vertoffset: np.array,
    zd_diffcoef: np.array,
    geofac_n2s_c: np.array,
    geofac_n2s_nbh: np.array,
    vcoef: np.array,
    theta_v: np.array,
    z_temp: np.array,
    **kwargs,
) -> np.array:
    c2e2c = grid.connectivities[C2E2CDim]
    shape = c2e2c.shape + vcoef.shape[1:]
    vcoef = vcoef.reshape(shape)
    zd_vertoffset = zd_vertoffset.reshape(shape)
    geofac_n2s_nbh = geofac_n2s_nbh.reshape(c2e2c.shape)
    full_shape = vcoef.shape

    geofac_n2s_nbh = np.expand_dims(geofac_n2s_nbh, axis=2)

    theta_v_at_zd_vertidx = np.zeros_like(vcoef)
    theta_v_at_zd_vertidx_p1 = np.zeros_like(vcoef)
    for ic in range(full_shape[0]):
        for isparse in range(full_shape[1]):
            for ik in range(full_shape[2]):
                theta_v_at_zd_vertidx[ic, isparse, ik] = theta_v[
                    c2e2c[ic, isparse], ik + zd_vertoffset[ic, isparse, ik]
                ]
                theta_v_at_zd_vertidx_p1[ic, isparse, ik] = theta_v[
                    c2e2c[ic, isparse], ik + zd_vertoffset[ic, isparse, ik] + 1
                ]

    sum_over = np.sum(
        geofac_n2s_nbh * (vcoef * theta_v_at_zd_vertidx + (1.0 - vcoef) * theta_v_at_zd_vertidx_p1),
        axis=1,
    )

    geofac_n2s_c = np.expand_dims(geofac_n2s_c, axis=1)  # add KDim
    z_temp = np.where(mask, z_temp + zd_diffcoef * (theta_v * geofac_n2s_c + sum_over), z_temp)
    return z_temp


class TestTrulyHorizontalDiffusionNablaOfThetaOverSteepPoints(StencilTest):
    PROGRAM = truly_horizontal_diffusion_nabla_of_theta_over_steep_points
    OUTPUTS = ("z_temp",)

    @staticmethod
    def reference(
        grid,
        mask: np.array,
        zd_vertoffset: np.array,
        zd_diffcoef: np.array,
        geofac_n2s_c: np.array,
        geofac_n2s_nbh: np.array,
        vcoef: np.array,
        theta_v: np.array,
        z_temp: np.array,
        **kwargs,
    ) -> np.array:

        z_temp = truly_horizontal_diffusion_nabla_of_theta_over_steep_points_numpy(
            grid,
            mask,
            zd_vertoffset,
            zd_diffcoef,
            geofac_n2s_c,
            geofac_n2s_nbh,
            vcoef,
            theta_v,
            z_temp,
        )
        return dict(z_temp=z_temp)

    @pytest.fixture
    def input_data(self, grid):

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
        theta_v = random_field(grid, CellDim, KDim)
        z_temp = random_field(grid, CellDim, KDim)

        vcoef_new = flatten_first_two_dims(CECDim, KDim, field=vcoef)
        zd_vertoffset_new = flatten_first_two_dims(CECDim, KDim, field=zd_vertoffset)
        geofac_n2s_nbh_new = flatten_first_two_dims(CECDim, field=geofac_n2s_nbh)

        return dict(
            mask=mask,
            zd_vertoffset=zd_vertoffset_new,
            zd_diffcoef=zd_diffcoef,
            geofac_n2s_c=geofac_n2s_c,
            geofac_n2s_nbh=geofac_n2s_nbh_new,
            theta_v=theta_v,
            z_temp=z_temp,
            vcoef=vcoef_new,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
