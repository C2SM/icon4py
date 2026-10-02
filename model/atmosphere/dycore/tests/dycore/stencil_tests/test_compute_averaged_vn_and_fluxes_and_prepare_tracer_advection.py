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

from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_velocity_quantities import (
    compute_averaged_vn_and_fluxes,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.testing import stencil_tests

from .test_accumulate_prep_adv_fields import accumulate_prep_adv_fields_numpy
from .test_compute_contravariant_correction import compute_contravariant_correction_numpy
from .test_compute_mass_flux import compute_mass_flux_numpy
from .test_spatially_average_flux_or_velocity import spatially_average_flux_or_velocity_numpy


@pytest.mark.embedded_remap_error
@pytest.mark.continuous_benchmarking
class TestComputeAveragedVnAndFluxesAndPrepareTracerAdvection(stencil_tests.StencilTest):
    PROGRAM = compute_averaged_vn_and_fluxes
    OUTPUTS = (
        "spatially_averaged_vn",
        "mass_flux_at_edges_on_model_levels",
        "theta_v_flux_at_edges_on_model_levels",
        "substep_and_spatially_averaged_vn",
        "substep_averaged_mass_flux",
        "tangential_wind",
        "contravariant_correction_at_edges_on_model_levels",
    )
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "vertical_start",
            "vertical_end",
            "nflatlev",
            "prepare_fluxes_for_advection",
            "at_first_substep",
            "recompute_contravariant_correction",
            "r_nsubsteps",
        ),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
            "nflatlev",
            "prepare_fluxes_for_advection",
            "at_first_substep",
            "recompute_contravariant_correction",
            "r_nsubsteps",
        ),
    }

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        spatially_averaged_vn: np.ndarray,
        mass_flux_at_edges_on_model_levels: np.ndarray,
        theta_v_flux_at_edges_on_model_levels: np.ndarray,
        substep_and_spatially_averaged_vn: np.ndarray,
        substep_averaged_mass_flux: np.ndarray,
        tangential_wind: np.ndarray,
        contravariant_correction_at_edges_on_model_levels: np.ndarray,
        e_flx_avg: np.ndarray,
        rbf_vec_coeff_e: np.ndarray,
        vn: np.ndarray,
        rho_at_edges_on_model_levels: np.ndarray,
        ddqz_z_full_e: np.ndarray,
        ddxn_z_full: np.ndarray,
        ddxt_z_full: np.ndarray,
        theta_v_at_edges_on_model_levels: np.ndarray,
        prepare_fluxes_for_advection: bool,
        at_first_substep: bool,
        recompute_contravariant_correction: bool,
        r_nsubsteps: ta.wpfloat,
        nflatlev: int,
        horizontal_start: int,
        horizontal_end: int,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        initial_spatially_averaged_vn = spatially_averaged_vn.copy()
        initial_mass_flux_at_edges_on_model_levels = mass_flux_at_edges_on_model_levels.copy()
        initial_theta_v_flux_at_edges_on_model_levels = theta_v_flux_at_edges_on_model_levels.copy()
        initial_substep_and_spatially_averaged_vn = substep_and_spatially_averaged_vn.copy()
        initial_substep_averaged_mass_flux = substep_averaged_mass_flux.copy()
        initial_tangential_wind = tangential_wind.copy()
        initial_contravariant_correction_at_edges_on_model_levels = (
            contravariant_correction_at_edges_on_model_levels.copy()
        )

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

        if prepare_fluxes_for_advection:
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

        if recompute_contravariant_correction:
            e2c2e = connectivities[dims.E2C2E]
            tangential_wind = np.sum(
                np.where(
                    (e2c2e != -1)[:, :, np.newaxis],
                    vn[e2c2e] * rbf_vec_coeff_e[:, :, np.newaxis],
                    0.0,
                ),
                axis=1,
            )
            k = np.arange(vn.shape[1])[np.newaxis, :]
            contravariant_correction_at_edges_on_model_levels = np.where(
                k >= nflatlev,
                compute_contravariant_correction_numpy(
                    vn, ddxn_z_full, ddxt_z_full, tangential_wind
                ),
                contravariant_correction_at_edges_on_model_levels,
            )
        for field, initial in (
            (tangential_wind, initial_tangential_wind),
            (
                contravariant_correction_at_edges_on_model_levels,
                initial_contravariant_correction_at_edges_on_model_levels,
            ),
        ):
            field[:horizontal_start, :] = initial[:horizontal_start, :]
            field[horizontal_end:, :] = initial[horizontal_end:, :]

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
            tangential_wind=tangential_wind,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
        )

    @stencil_tests.input_data_fixture(
        params=[
            {
                "prepare_fluxes_for_advection": pa,
                "at_first_substep": afs,
                "recompute_contravariant_correction": rcc,
            }
            for pa, afs, rcc in [
                (True, True, False),
                (True, False, False),
                (True, False, True),
            ]
        ],
        ids=lambda p: (
            f"prepare_fluxes_for_advection[{p['prepare_fluxes_for_advection']}]__at_first_substep[{p['at_first_substep']}]"
            f"__recompute_contravariant_correction[{p['recompute_contravariant_correction']}]"
        ),
    )
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper,
        grid: base.Grid,
        request: pytest.FixtureRequest,
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        spatially_averaged_vn = data_alloc.zero_field(dims.EdgeDim, dims.KDim)
        mass_fl_e = data_alloc.zero_field(dims.EdgeDim, dims.KDim)
        z_theta_v_fl_e = data_alloc.zero_field(dims.EdgeDim, dims.KDim)

        substep_and_spatially_averaged_vn = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        substep_averaged_mass_flux = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        e_flx_avg = data_alloc.random_field(dims.EdgeDim, dims.E2C2EODim)
        vn = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        z_rho_e = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        ddqz_z_full_e = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        z_theta_v_e = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        tangential_wind = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        contravariant_correction_at_edges_on_model_levels = data_alloc.random_field(
            dims.EdgeDim, dims.KDim
        )
        rbf_vec_coeff_e = data_alloc.random_field(dims.EdgeDim, dims.E2C2EDim)
        ddxn_z_full = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        ddxt_z_full = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        nflatlev = 5  # value is set to reflect the MCH ch1 experiment
        prepare_fluxes_for_advection = request.param["prepare_fluxes_for_advection"]
        at_first_substep = request.param["at_first_substep"]
        recompute_contravariant_correction = request.param["recompute_contravariant_correction"]
        r_nsubsteps = 0.5

        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))

        return dict(
            spatially_averaged_vn=spatially_averaged_vn,
            mass_flux_at_edges_on_model_levels=mass_fl_e,
            theta_v_flux_at_edges_on_model_levels=z_theta_v_fl_e,
            substep_and_spatially_averaged_vn=substep_and_spatially_averaged_vn,
            substep_averaged_mass_flux=substep_averaged_mass_flux,
            tangential_wind=tangential_wind,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            e_flx_avg=e_flx_avg,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            vn=vn,
            rho_at_edges_on_model_levels=z_rho_e,
            ddqz_z_full_e=ddqz_z_full_e,
            ddxn_z_full=ddxn_z_full,
            ddxt_z_full=ddxt_z_full,
            theta_v_at_edges_on_model_levels=z_theta_v_e,
            prepare_fluxes_for_advection=prepare_fluxes_for_advection,
            at_first_substep=at_first_substep,
            recompute_contravariant_correction=recompute_contravariant_correction,
            r_nsubsteps=r_nsubsteps,
            nflatlev=nflatlev,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
