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
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_velocity_quantities import (
    compute_averaged_vn_and_fluxes_and_prepare_tracer_advection,
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

class TestComputeAveragedVnAndFluxesAndPrepareTracerAdvection(test_helpers.StencilTest):
    PROGRAM = compute_averaged_vn_and_fluxes_and_prepare_tracer_advection
    OUTPUTS = ("spatially_averaged_vn", "mass_flux_at_edges_on_model_levels", "theta_v_flux_at_edges_on_model_levels", "substep_and_spatially_averaged_vn", "substep_averaged_mass_flux",)
    MARKERS = (pytest.mark.embedded_remap_error,)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        substep_and_spatially_averaged_vn: np.ndarray,
        substep_averaged_mass_flux: np.ndarray,
        e_flx_avg: np.ndarray,
        vn: np.ndarray,
        rho_at_edges_on_model_levels: np.ndarray,
        ddqz_z_full_e: np.ndarray,
        theta_v_at_edges_on_model_levels: np.ndarray,
        prepare_advection: bool,
        at_first_substep: bool,
        r_nsubsteps: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:

        spatially_averaged_vn = compute_avg_vn_numpy(connectivities, e_flx_avg, vn)

        mass_fl_e, z_theta_v_fl_e = compute_mass_flux_numpy(
            rho_at_edges_on_model_levels,
            spatially_averaged_vn,
            ddqz_z_full_e,
            theta_v_at_edges_on_model_levels,
        )

        substep_and_spatially_averaged_vn, substep_averaged_mass_flux = (
            (
                (r_nsubsteps * spatially_averaged_vn, r_nsubsteps * mass_fl_e)
                if at_first_substep
                else accumulate_prep_adv_fields_numpy(
                    spatially_averaged_vn,
                    mass_fl_e,
                    substep_and_spatially_averaged_vn,
                    substep_averaged_mass_flux,
                    r_nsubsteps,
                )
            )
            if prepare_advection
            else (substep_and_spatially_averaged_vn, substep_averaged_mass_flux)
        )

        return dict(spatially_averaged_vn=spatially_averaged_vn, mass_flux_at_edges_on_model_levels=mass_fl_e, theta_v_flux_at_edges_on_model_levels=z_theta_v_fl_e, substep_and_spatially_averaged_vn=substep_and_spatially_averaged_vn, substep_averaged_mass_flux=substep_averaged_mass_flux,)

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict[str, gtx.Field | state_utils.ScalarType]:

        spatially_averaged_vn = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        mass_fl_e = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        z_theta_v_fl_e = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)

        substep_and_spatially_averaged_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        substep_averaged_mass_flux = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        e_flx_avg = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EODim)
        vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_rho_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ddqz_z_full_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_theta_v_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        prepare_advection = True
        at_first_substep = True
        r_nsubsteps = 0.2

        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.LOCAL))

        return dict(
            spatially_averaged_vn=spatially_averaged_vn,
            mass_flux_at_edges_on_model_levels=mass_fl_e,
            theta_v_flux_at_edges_on_model_levels=z_theta_v_fl_e,
            substep_and_spatially_averaged_vn=substep_and_spatially_averaged_vn,
            substep_averaged_mass_flux=substep_averaged_mass_flux,
            e_flx_avg=e_flx_avg,
            vn=vn,
            rho_at_edges_on_model_levels=z_rho_e,
            ddqz_z_full_e=ddqz_z_full_e,
            theta_v_at_edges_on_model_levels=z_theta_v_e,
            prepare_advection=prepare_advection,
            at_first_substep=at_first_substep,
            r_nsubsteps=r_nsubsteps,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
