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
    combined_solve_nh_30_to_38_predictor,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.testing import helpers as test_helpers

from .test_compute_contravariant_correction import (
    compute_contravariant_correction_numpy,
)
from .test_compute_derived_horizontal_winds_and_ke_and_horizontal_advection_of_w_and_contravariant_correction import \
    extrapolate_to_surface_numpy
from .test_compute_horizontal_kinetic_energy import compute_horizontal_kinetic_energy_numpy
from .test_compute_mass_flux import (
    compute_mass_flux_numpy,
)
from .test_compute_avg_vn_and_graddiv_vn_and_vt import (
    compute_avg_vn_and_graddiv_vn_and_vt_numpy,
)
from .test_interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges import \
    interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_numpy
from .test_interpolate_vt_to_interface_edges import interpolate_vt_to_interface_edges_numpy


@staticmethod
def compute_vt_vn_on_half_levels_and_kinetic_energy_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    vn: np.ndarray,
    tangential_wind: np.ndarray,
    vn_on_half_levels: np.ndarray,
    tangential_wind_on_half_levels: np.ndarray,
    horizontal_kinetic_energy_at_edges_on_model_levels: np.ndarray,
    wgtfac_e: np.ndarray,
    wgtfacq_e: np.ndarray,
    nlevp1: int,
) -> tuple[np.ndarray, ...]:
    k = np.arange(nlevp1)
    k = k[np.newaxis, :]
    k_nlev = k[:, :-1]

    condition1 = 1 <= k_nlev
    vn_on_half_levels[:, :-1], horizontal_kinetic_energy_at_edges_on_model_levels = np.where(
        condition1,
        interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_numpy(
            wgtfac_e, vn, tangential_wind
        ),
        (vn_on_half_levels[:, :-1], horizontal_kinetic_energy_at_edges_on_model_levels),
    )

    tangential_wind_on_half_levels = np.where(
        condition1,
        interpolate_vt_to_interface_edges_numpy(wgtfac_e, tangential_wind),
        tangential_wind_on_half_levels,
    )

    condition2 = k_nlev == 0
    (
        vn_on_half_levels[:, :-1],
        tangential_wind_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
    ) = np.where(
        condition2,
        compute_horizontal_kinetic_energy_numpy(vn, tangential_wind),
        (
            vn_on_half_levels[:, :-1],
            tangential_wind_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels,
        ),
    )

    vn_on_half_levels[:, -1] = extrapolate_to_surface_numpy(wgtfacq_e, vn)

    return (
        vn_on_half_levels,
        tangential_wind_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
    )


class TestCombinedSolveNh30To38Predictor(test_helpers.StencilTest):
    PROGRAM = combined_solve_nh_30_to_38_predictor
    OUTPUTS = (
        "z_vn_avg",
        "horizontal_gradient_of_normal_wind_divergence",
        "tangential_wind",
        "mass_flux_at_edges_on_model_levels",
        "theta_v_flux_at_edges_on_model_levels",
        "vn_on_half_levels",
        "tangential_wind_on_half_levels",
        "horizontal_kinetic_energy_at_edges_on_model_levels",
        "contravariant_correction_at_edges_on_model_levels",
    )
    MARKERS = (pytest.mark.embedded_remap_error,)


    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        tangential_wind_on_half_levels: np.ndarray,
        vn_on_half_levels: np.ndarray,
        horizontal_kinetic_energy_at_edges_on_model_levels: np.ndarray,
        contravariant_correction_at_edges_on_model_levels: np.ndarray,
        vn: np.ndarray,
        e_flx_avg: np.ndarray,
        geofac_grdiv: np.ndarray,
        rbf_vec_coeff_e: np.ndarray,
        rho_at_edges_on_model_levels: np.ndarray,
        theta_v_at_edges_on_model_levels: np.ndarray,
        ddqz_z_full_e: np.ndarray,
        ddxn_z_full: np.ndarray,
        ddxt_z_full: np.ndarray,
        wgtfac_e: np.ndarray,
        wgtfacq_e: np.ndarray,
        nflatlev: gtx.int32,
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
        vertical_start: gtx.int32,
        vertical_end: gtx.int32,
        **kwargs: Any,
    ) -> dict:
        k = np.arange(vertical_end)
        k = k[np.newaxis, :]
        k_nlev = k[:, :-1]

        z_vn_avg, horizontal_gradient_of_normal_wind_divergence, tangential_wind = compute_avg_vn_and_graddiv_vn_and_vt_numpy(
            connectivities,
            e_flx_avg,
            vn,
            geofac_grdiv,
            rbf_vec_coeff_e,
        )

        mass_flux_at_edges_on_model_levels, theta_v_flux_at_edges_on_model_levels = compute_mass_flux_numpy(
            rho_at_edges_on_model_levels,
            z_vn_avg,
            ddqz_z_full_e,
            theta_v_at_edges_on_model_levels,
        )

        contravariant_correction_at_edges_on_model_levels = np.where(
            k_nlev >= nflatlev,
            compute_contravariant_correction_numpy(vn, ddxn_z_full, ddxt_z_full, tangential_wind),
            contravariant_correction_at_edges_on_model_levels,
        )

        (
            vn_on_half_levels,
            tangential_wind_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels,
        ) = compute_vt_vn_on_half_levels_and_kinetic_energy_numpy(
            connectivities,
            vn,
            tangential_wind,
            vn_on_half_levels,
            tangential_wind_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels,
            wgtfac_e,
            wgtfacq_e,
            vertical_end,
        )

        return dict(
            z_vn_avg=z_vn_avg,
            horizontal_gradient_of_normal_wind_divergence=horizontal_gradient_of_normal_wind_divergence,
            tangential_wind=tangential_wind,
            mass_flux_at_edges_on_model_levels=mass_flux_at_edges_on_model_levels,
            theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges_on_model_levels,
            vn_on_half_levels=vn_on_half_levels,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels= horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels= contravariant_correction_at_edges_on_model_levels,
        )

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict[str, gtx.Field | state_utils.ScalarType]:

        z_vn_avg = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        horizontal_gradient_of_normal_wind_divergence = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        tangential_wind = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        mass_flux_at_edges_on_model_levels = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        theta_v_flux_at_edges_on_model_levels = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        tangential_wind_on_half_levels = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        vn_on_half_levels = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1})
        horizontal_kinetic_energy_at_edges_on_model_levels = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        contravariant_correction_at_edges_on_model_levels = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)

        vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        wgtfac_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        wgtfacq_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        e_flx_avg = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EODim)
        geofac_grdiv = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EODim)
        rbf_vec_coeff_e = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EDim)
        rho_at_edges_on_model_levels = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        theta_v_at_edges_on_model_levels = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ddqz_z_full_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ddxn_z_full = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ddxt_z_full = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)

        nflatlev = 5


        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.LOCAL))

        return dict(
            z_vn_avg=z_vn_avg,
            horizontal_gradient_of_normal_wind_divergence=horizontal_gradient_of_normal_wind_divergence,
            tangential_wind=tangential_wind,
            mass_flux_at_edges_on_model_levels=mass_flux_at_edges_on_model_levels,
            theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges_on_model_levels,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn_on_half_levels=vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            vn=vn,
            e_flx_avg=e_flx_avg,
            geofac_grdiv=geofac_grdiv,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            rho_at_edges_on_model_levels=rho_at_edges_on_model_levels,
            theta_v_at_edges_on_model_levels=theta_v_at_edges_on_model_levels,
            ddqz_z_full_e=ddqz_z_full_e,
            ddxn_z_full=ddxn_z_full,
            ddxt_z_full=ddxt_z_full,
            wgtfac_e=wgtfac_e,
            wgtfacq_e=wgtfacq_e,
            nflatlev=nflatlev,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels + 1),
        )
