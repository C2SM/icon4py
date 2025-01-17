# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.atmosphere.dycore.stencils.compute_divergence_of_fluxes_of_rho_and_theta import (
    compute_divergence_of_fluxes_of_rho_and_theta,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.testing.helpers import StencilTest


class TestComputeDivergenceConnectivityOfFluxesOfRhoAndTheta(StencilTest):
    PROGRAM = compute_divergence_of_fluxes_of_rho_and_theta
    OUTPUTS = ("z_flxdiv_mass", "z_flxdiv_theta")

    @staticmethod
    def reference(
        grid,
        geofac_div: np.array,
        mass_fl_e: np.array,
        z_theta_v_fl_e: np.array,
        **kwargs,
    ) -> tuple[np.array]:
        c2e = grid.connectivities[dims.C2EDim]
        geofac_div = np.expand_dims(geofac_div, axis=-1)
        c2ce = grid.get_offset_provider("C2CE").table

        z_flxdiv_mass = np.sum(
            geofac_div[c2ce] * mass_fl_e[c2e],
            axis=1,
        )
        z_flxdiv_theta = np.sum(
            geofac_div[c2ce] * z_theta_v_fl_e[c2e],
            axis=1,
        )
        return dict(z_flxdiv_mass=z_flxdiv_mass, z_flxdiv_theta=z_flxdiv_theta)

    @pytest.fixture
    def input_data(self, grid):
        geofac_div = data_alloc.as_1D_sparse_field(
            data_alloc.random_field(grid, dims.CellDim, dims.C2EDim, dtype=wpfloat), dims.CEDim
        )
        z_theta_v_fl_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        z_flxdiv_theta = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        mass_fl_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        z_flxdiv_mass = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            geofac_div=geofac_div,
            mass_fl_e=mass_fl_e,
            z_theta_v_fl_e=z_theta_v_fl_e,
            z_flxdiv_mass=z_flxdiv_mass,
            z_flxdiv_theta=z_flxdiv_theta,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
