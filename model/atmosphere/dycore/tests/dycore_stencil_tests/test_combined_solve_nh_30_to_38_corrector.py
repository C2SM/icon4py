# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.atmosphere.dycore.stencils.combined_solve_nh_30_to_38 import (
    combined_solve_nh_30_to_38_corrector,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.testing import helpers as test_helpers

from .test_compute_avg_vn import (
    compute_avg_vn_numpy,
)
from .test_compute_mass_flux import (
    compute_mass_flux_numpy,
)
from .test_accumulate_prep_adv_fields import (
    accumulate_prep_adv_fields_numpy,
)

class TestCombinedSolveNh30To38Corrector(test_helpers.StencilTest):
    PROGRAM = combined_solve_nh_30_to_38_corrector
    OUTPUTS = ("z_vn_avg", "mass_fl_e", "z_theta_v_fl_e", "vn_traj", "mass_flx_me",)
    MARKERS = (pytest.mark.embedded_remap_error,)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        vn_traj: np.ndarray,
        mass_flx_me: np.ndarray,
        e_flx_avg: np.ndarray,
        vn: np.ndarray,
        z_rho_e: np.ndarray,
        ddzq_z_full_e: np.ndarray,
        z_theta_v_e: np.ndarray,
        at_initial_timestep: bool,
        r_nsubsteps: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:

        z_vn_avg = compute_avg_vn_numpy(connectivities, e_flx_avg, vn)

        mass_fl_e, z_theta_v_fl_e = compute_mass_flux_numpy(
            z_rho_e,
            z_vn_avg,
            ddzq_z_full_e,
            z_theta_v_e,
        )

        if at_initial_timestep:
            vn_traj = 0

        if at_initial_timestep:
            mass_flx_me = 0

        vn_traj, mass_flx_me = accumulate_prep_adv_fields_numpy(
            z_vn_avg,
            mass_fl_e,
            vn_traj,
            mass_flx_me,
            r_nsubsteps,
        )

        return dict(z_vn_avg=z_vn_avg, mass_fl_e=mass_fl_e, z_theta_v_fl_e=z_theta_v_fl_e, vn_traj=vn_traj, mass_flx_me=mass_flx_me,)

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict[str, gtx.Field | state_utils.ScalarType]:

        z_vn_avg = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        mass_fl_e = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        z_theta_v_fl_e = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)

        vn_traj = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        mass_flx_me = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        e_flx_avg = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EODim)
        vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_rho_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ddzq_z_full_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_theta_v_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        at_initial_timestep = True
        r_nsubsteps = 0.2

        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.LOCAL))

        return dict(
            z_vn_avg=z_vn_avg,
            mass_fl_e=mass_fl_e,
            z_theta_v_fl_e=z_theta_v_fl_e,
            vn_traj=vn_traj,
            mass_flx_me=mass_flx_me,
            e_flx_avg=e_flx_avg,
            vn=vn,
            z_rho_e=z_rho_e,
            ddzq_z_full_e=ddzq_z_full_e,
            z_theta_v_e=z_theta_v_e,
            at_initial_timestep=at_initial_timestep,
            r_nsubsteps=r_nsubsteps,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
