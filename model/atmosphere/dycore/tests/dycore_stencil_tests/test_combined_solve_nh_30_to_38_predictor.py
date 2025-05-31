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
    skip_compute_predictor_vertical_advection: bool,
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

    if not skip_compute_predictor_vertical_advection:
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
        "z_graddiv_vn",
        "vt",
        "mass_fl_e",
        "z_theta_v_fl_e",
        "vn_ie",
        "vt_ie",
        "z_kin_hor_e",
        "z_w_concorr_me",
    )
    MARKERS = (pytest.mark.embedded_remap_error,)


    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        vt_ie: np.ndarray,
        vn_ie: np.ndarray,
        z_kin_hor_e: np.ndarray,
        z_w_concorr_me: np.ndarray,
        vn: np.ndarray,
        e_flx_avg: np.ndarray,
        geofac_grdiv: np.ndarray,
        rbf_vec_coeff_e: np.ndarray,
        z_rho_e: np.ndarray,
        z_theta_v_e: np.ndarray,
        ddzq_z_full_e: np.ndarray,
        ddxn_z_full: np.ndarray,
        ddxt_z_full: np.ndarray,
        wgtfac_e: np.ndarray,
        wgtfacq_e: np.ndarray,
        nflatlev: gtx.int32,
        skip_compute_predictor_vertical_advection: bool,
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
        vertical_start: gtx.int32,
        vertical_end: gtx.int32,
        **kwargs: Any,
    ) -> dict:
        k = np.arange(vertical_end)
        k = k[np.newaxis, :]
        k_nlev = k[:, :-1]

        z_vn_avg, z_graddiv_vn, vt = compute_avg_vn_and_graddiv_vn_and_vt_numpy(
            connectivities,
            e_flx_avg,
            vn,
            geofac_grdiv,
            rbf_vec_coeff_e,
        )

        mass_fl_e, z_theta_v_fl_e = compute_mass_flux_numpy(
            z_rho_e,
            z_vn_avg,
            ddzq_z_full_e,
            z_theta_v_e,
        )

        z_w_concorr_me = np.where(
            k_nlev >= nflatlev,
            compute_contravariant_correction_numpy(vn, ddxn_z_full, ddxt_z_full, vt),
            z_w_concorr_me,
        )

        (
            vn_ie,
            vt_ie,
            z_kin_hor_e,
        ) = compute_vt_vn_on_half_levels_and_kinetic_energy_numpy(
            connectivities,
            vn,
            vt,
            vn_ie,
            vt_ie,
            z_kin_hor_e,
            wgtfac_e,
            wgtfacq_e,
            skip_compute_predictor_vertical_advection,
            vertical_end,
        )

        return dict(
            z_vn_avg=z_vn_avg,
            z_graddiv_vn=z_graddiv_vn,
            vt=vt,
            mass_fl_e=mass_fl_e,
            z_theta_v_fl_e=z_theta_v_fl_e,
            vn_ie=vn_ie,
            vt_ie=vt_ie,
            z_kin_hor_e= z_kin_hor_e,
            z_w_concorr_me= z_w_concorr_me,
        )

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict[str, gtx.Field | state_utils.ScalarType]:

        z_vn_avg = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        z_graddiv_vn = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        vt = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        mass_fl_e = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        z_theta_v_fl_e = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        vt_ie = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        vn_ie = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1})
        z_kin_hor_e = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        z_w_concorr_me = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)

        vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        wgtfac_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        wgtfacq_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        e_flx_avg = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EODim)
        geofac_grdiv = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EODim)
        rbf_vec_coeff_e = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EDim)
        z_rho_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_theta_v_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ddzq_z_full_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ddxn_z_full = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ddxt_z_full = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)

        nflatlev = 5

        skip_compute_predictor_vertical_advection = True

        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.LOCAL))

        return dict(
            z_vn_avg=z_vn_avg,
            z_graddiv_vn=z_graddiv_vn,
            vt=vt,
            mass_fl_e=mass_fl_e,
            z_theta_v_fl_e=z_theta_v_fl_e,
            vt_ie=vt_ie,
            vn_ie=vn_ie,
            z_kin_hor_e=z_kin_hor_e,
            z_w_concorr_me=z_w_concorr_me,
            vn=vn,
            e_flx_avg=e_flx_avg,
            geofac_grdiv=geofac_grdiv,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            z_rho_e=z_rho_e,
            z_theta_v_e=z_theta_v_e,
            ddzq_z_full_e=ddzq_z_full_e,
            ddxn_z_full=ddxn_z_full,
            ddxt_z_full=ddxt_z_full,
            wgtfac_e=wgtfac_e,
            wgtfacq_e=wgtfacq_e,
            nflatlev=nflatlev,
            skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels + 1),
        )
