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
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import precision, wpfloat
from icon4py.model.testing import stencil_tests

from .test_accumulate_prep_adv_fields import accumulate_prep_adv_fields_numpy
from .test_compute_mass_flux import compute_mass_flux_numpy
from .test_spatially_average_flux_or_velocity import spatially_average_flux_or_velocity_numpy


@pytest.mark.single_precision_ready
@pytest.mark.embedded_remap_error
@pytest.mark.continuous_benchmarking
class TestComputeAveragedVnAndFluxesAndPrepareTracerAdvection(stencil_tests.StencilTest):
    PROGRAM = compute_averaged_vn_and_fluxes_and_prepare_tracer_advection
    OUTPUTS = (
        "spatially_averaged_vn",
        "mass_flux_at_edges_on_model_levels",
        "theta_v_flux_at_edges_on_model_levels",
        "substep_and_spatially_averaged_vn",
        "substep_averaged_mass_flux",
    )
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "vertical_start",
            "vertical_end",
            "prepare_advection",
            "at_first_substep",
            "r_nsubsteps",
        ),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
            "prepare_advection",
            "at_first_substep",
            "r_nsubsteps",
        ),
    }
    if precision == "single":
        RTOL = 1e-1
        ATOL = 1e-2

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        spatially_averaged_vn: np.ndarray,
        mass_flux_at_edges_on_model_levels: np.ndarray,
        theta_v_flux_at_edges_on_model_levels: np.ndarray,
        substep_and_spatially_averaged_vn: np.ndarray,
        substep_averaged_mass_flux: np.ndarray,
        e_flx_avg: np.ndarray,
        vn: np.ndarray,
        rho_at_edges_on_model_levels: np.ndarray,
        ddqz_z_full_e: np.ndarray,
        theta_v_at_edges_on_model_levels: np.ndarray,
        prepare_advection: bool,
        at_first_substep: bool,
        r_nsubsteps: wpfloat,
        horizontal_start: int,
        horizontal_end: int,
        **kwargs: Any,
    ) -> dict:
        initial_spatially_averaged_vn = spatially_averaged_vn.copy()
        initial_mass_flux_at_edges_on_model_levels = mass_flux_at_edges_on_model_levels.copy()
        initial_theta_v_flux_at_edges_on_model_levels = theta_v_flux_at_edges_on_model_levels.copy()
        initial_substep_and_spatially_averaged_vn = substep_and_spatially_averaged_vn.copy()
        initial_substep_averaged_mass_flux = substep_averaged_mass_flux.copy()

        spatially_averaged_vn = spatially_average_flux_or_velocity_numpy(
            connectivities, e_flx_avg, vn
        )

        mass_flux_at_edges_on_model_levels, theta_v_flux_at_edges_on_model_levels = (
            compute_mass_flux_numpy(
                rho_at_edges_on_model_levels,
                spatially_averaged_vn,
                ddqz_z_full_e,
                theta_v_at_edges_on_model_levels,
            )
        )

        if prepare_advection:
            substep_and_spatially_averaged_vn, substep_averaged_mass_flux = (
                (
                    r_nsubsteps * spatially_averaged_vn,
                    r_nsubsteps * mass_flux_at_edges_on_model_levels,
                )
                if at_first_substep
                else accumulate_prep_adv_fields_numpy(
                    spatially_averaged_vn,
                    mass_flux_at_edges_on_model_levels,
                    substep_and_spatially_averaged_vn,
                    substep_averaged_mass_flux,
                    r_nsubsteps,
                )
            )

        spatially_averaged_vn[:horizontal_start, :] = initial_spatially_averaged_vn[
            :horizontal_start, :
        ]
        spatially_averaged_vn[horizontal_end:, :] = initial_spatially_averaged_vn[
            horizontal_end:, :
        ]

        mass_flux_at_edges_on_model_levels[:horizontal_start, :] = (
            initial_mass_flux_at_edges_on_model_levels[:horizontal_start, :]
        )
        mass_flux_at_edges_on_model_levels[horizontal_end:, :] = (
            initial_mass_flux_at_edges_on_model_levels[horizontal_end:, :]
        )

        theta_v_flux_at_edges_on_model_levels[:horizontal_start, :] = (
            initial_theta_v_flux_at_edges_on_model_levels[:horizontal_start, :]
        )
        theta_v_flux_at_edges_on_model_levels[horizontal_end:, :] = (
            initial_theta_v_flux_at_edges_on_model_levels[horizontal_end:, :]
        )

        substep_and_spatially_averaged_vn[:horizontal_start, :] = (
            initial_substep_and_spatially_averaged_vn[:horizontal_start, :]
        )
        substep_and_spatially_averaged_vn[horizontal_end:, :] = (
            initial_substep_and_spatially_averaged_vn[horizontal_end:, :]
        )

        substep_averaged_mass_flux[:horizontal_start, :] = initial_substep_averaged_mass_flux[
            :horizontal_start, :
        ]
        substep_averaged_mass_flux[horizontal_end:, :] = initial_substep_averaged_mass_flux[
            horizontal_end:, :
        ]

        return dict(
            spatially_averaged_vn=spatially_averaged_vn,
            mass_flux_at_edges_on_model_levels=mass_flux_at_edges_on_model_levels,
            theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges_on_model_levels,
            substep_and_spatially_averaged_vn=substep_and_spatially_averaged_vn,
            substep_averaged_mass_flux=substep_averaged_mass_flux,
        )

    @pytest.fixture(
        params=[
            {"prepare_advection": pa, "at_first_substep": afs}
            for pa, afs in [
                (True, True),
                (True, False),
            ]
        ],
        ids=lambda p: f"prepare_advection[{p['prepare_advection']}]__at_first_substep[{p['at_first_substep']}]",
    )
    def input_data(
        self, request: pytest.FixtureRequest, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        random_fields = data_alloc.get_random_fields(
            grid,
            [
                "substep_and_spatially_averaged_vn",
                "substep_averaged_mass_flux",
                "e_flx_avg",
                "vn",
                "rho_at_edges_on_model_levels",
                "ddqz_z_full_e",
                "theta_v_at_edges_on_model_levels",
            ],
        )

        zero_fields = data_alloc.get_zero_fields(
            grid,
            [
                "spatially_averaged_vn",
                "mass_flux_at_edges_on_model_levels",
                "theta_v_flux_at_edges_on_model_levels",
            ],
        )

        edge_domain = h_grid.domain(dims.EdgeDim)
        return (
            random_fields
            | zero_fields
            | dict(
                prepare_advection=request.param["prepare_advection"],
                at_first_substep=request.param["at_first_substep"],
                r_nsubsteps=wpfloat(0.5),
                horizontal_start=grid.start_index(
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
                ),
                horizontal_end=grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2)),
                vertical_start=0,
                vertical_end=grid.num_levels,
            )
        )
