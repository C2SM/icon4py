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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_41 import (
    mo_solve_nonhydro_stencil_41,
)
from icon4py.model.common.dimension import C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    random_field,
    zero_field,
)


class TestMoSolveNonhydroStencil41(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_41
    OUTPUTS = ("z_flxdiv_mass", "z_flxdiv_theta")

    @staticmethod
    def reference(
        grid,
        geofac_div: np.array,
        mass_fl_e: np.array,
        z_theta_v_fl_e: np.array,
        **kwargs,
    ) -> tuple[np.array]:
        c2e = grid.connectivities[C2EDim]
        geofac_div = np.expand_dims(geofac_div, axis=-1)
        z_flxdiv_mass = np.sum(
            geofac_div[grid.get_offset_provider("C2CE").table] * mass_fl_e[c2e],
            axis=1,
        )
        z_flxdiv_theta = np.sum(
            geofac_div[grid.get_offset_provider("C2CE").table] * z_theta_v_fl_e[c2e],
            axis=1,
        )
        return dict(z_flxdiv_mass=z_flxdiv_mass, z_flxdiv_theta=z_flxdiv_theta)

    @pytest.fixture
    def input_data(self, grid):
        geofac_div = as_1D_sparse_field(random_field(grid, CellDim, C2EDim), CEDim)
        z_theta_v_fl_e = random_field(grid, EdgeDim, KDim)
        z_flxdiv_theta = zero_field(grid, CellDim, KDim)
        mass_fl_e = random_field(grid, EdgeDim, KDim)
        z_flxdiv_mass = zero_field(grid, CellDim, KDim)

        return dict(
            geofac_div=geofac_div,
            mass_fl_e=mass_fl_e,
            z_theta_v_fl_e=z_theta_v_fl_e,
            z_flxdiv_mass=z_flxdiv_mass,
            z_flxdiv_theta=z_flxdiv_theta,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
