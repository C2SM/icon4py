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

from icon4py.model.atmosphere.dycore.stencils.compute_edge_diagnostics_for_velocity_advection import (
    compute_derived_horizontal_winds_and_ke_and_horizontal_advection_of_w_and_contravariant_correction,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import helpers as test_helpers

from .test_compute_contravariant_correction import compute_contravariant_correction_numpy
from .test_compute_horizontal_advection_term_for_vertical_velocity import (
    compute_horizontal_advection_term_for_vertical_velocity_numpy,
)
from .test_compute_horizontal_kinetic_energy import compute_horizontal_kinetic_energy_numpy
from .test_compute_tangential_wind import compute_tangential_wind_numpy
from .test_interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges import (
    interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_numpy,
)
from .test_interpolate_vt_to_interface_edges import interpolate_vt_to_interface_edges_numpy
from .test_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy,
)


def extrapolate_to_surface_numpy(wgtfacq_e: np.ndarray, vn: np.ndarray) -> np.ndarray:
    vn_k_minus_1 = vn[:, -1]
    vn_k_minus_2 = vn[:, -2]
    vn_k_minus_3 = vn[:, -3]
    wgtfacq_e_k_minus_1 = wgtfacq_e[:, -1]
    wgtfacq_e_k_minus_2 = wgtfacq_e[:, -2]
    wgtfacq_e_k_minus_3 = wgtfacq_e[:, -3]
    vn_at_surface = (
        wgtfacq_e_k_minus_1 * vn_k_minus_1
        + wgtfacq_e_k_minus_2 * vn_k_minus_2
        + wgtfacq_e_k_minus_3 * vn_k_minus_3
    )
    return vn_at_surface


class TestComputeDerivedHorizontalWindsAndKEAndHorizontalAdvectionofWAndContravariantCorrection(
    test_helpers.StencilTest
):
    PROGRAM = compute_derived_horizontal_winds_and_ke_and_horizontal_advection_of_w_and_contravariant_correction
    OUTPUTS = (
        "tangential_wind",
        "tangential_wind_on_half_levels",
        "vn_on_half_levels",
        "horizontal_kinetic_energy_at_edges_on_model_levels",
        "contravariant_correction_at_edges_on_model_levels",
        "horizontal_advection_of_w_at_edges_on_half_levels",
    )
    MARKERS = (pytest.mark.embedded_remap_error,)

    @staticmethod
    def _fused_velocity_advection_stencil_1_to_6_numpy(
        connectivities: dict[gtx.Dimension, np.ndarray],
        tangential_wind_on_half_levels: np.ndarray,
        vn_on_half_levels: np.ndarray,
        horizontal_kinetic_energy_at_edges_on_model_levels: np.ndarray,
        contravariant_correction_at_edges_on_model_levels: np.ndarray,
        vn: np.ndarray,
        rbf_vec_coeff_e: np.ndarray,
        wgtfac_e: np.ndarray,
        ddxn_z_full: np.ndarray,
        ddxt_z_full: np.ndarray,
        wgtfacq_e: np.ndarray,
        skip_compute_predictor_vertical_advection: bool,
        nflatlev: int,
        nlevp1: int,
    ) -> tuple[np.ndarray, ...]:
        k: np.ndarray = np.arange(nlevp1)
        k = k[np.newaxis, :]
        k_nlev = k[:, :-1]

        tangential_wind = compute_tangential_wind_numpy(connectivities, vn, rbf_vec_coeff_e)

        condition2 = 1 <= k_nlev
        vn_on_half_levels[:, :-1], horizontal_kinetic_energy_at_edges_on_model_levels = np.where(
            condition2,
            interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_numpy(
                wgtfac_e, vn, tangential_wind
            ),
            (vn_on_half_levels[:, :-1], horizontal_kinetic_energy_at_edges_on_model_levels),
        )

        if not skip_compute_predictor_vertical_advection:
            tangential_wind_on_half_levels = np.where(
                condition2,
                interpolate_vt_to_interface_edges_numpy(wgtfac_e, tangential_wind),
                tangential_wind_on_half_levels,
            )

        condition3 = k_nlev == 0
        (
            vn_on_half_levels[:, :-1],
            tangential_wind_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels,
        ) = np.where(
            condition3,
            compute_horizontal_kinetic_energy_numpy(vn, tangential_wind),
            (
                vn_on_half_levels[:, :-1],
                tangential_wind_on_half_levels,
                horizontal_kinetic_energy_at_edges_on_model_levels,
            ),
        )

        vn_on_half_levels[:, -1] = extrapolate_to_surface_numpy(wgtfacq_e, vn)

        condition5 = nflatlev <= k_nlev
        contravariant_correction_at_edges_on_model_levels = np.where(
            condition5,
            compute_contravariant_correction_numpy(vn, ddxn_z_full, ddxt_z_full, tangential_wind),
            contravariant_correction_at_edges_on_model_levels,
        )

        return (
            tangential_wind,
            tangential_wind_on_half_levels,
            vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels,
        )

    @classmethod
    def reference(
        cls,
        connectivities: dict[gtx.Dimension, np.ndarray],
        tangential_wind: np.ndarray,
        tangential_wind_on_half_levels: np.ndarray,
        vn_on_half_levels: np.ndarray,
        horizontal_kinetic_energy_at_edges_on_model_levels: np.ndarray,
        contravariant_correction_at_edges_on_model_levels: np.ndarray,
        horizontal_advection_of_w_at_edges_on_half_levels: np.ndarray,
        vn: np.ndarray,
        w: np.ndarray,
        rbf_vec_coeff_e: np.ndarray,
        wgtfac_e: np.ndarray,
        ddxn_z_full: np.ndarray,
        ddxt_z_full: np.ndarray,
        wgtfacq_e: np.ndarray,
        c_intp: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        tangent_orientation: np.ndarray,
        skip_compute_predictor_vertical_advection: bool,
        nflatlev: int,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
    ) -> dict:
        initial_tangential_wind = tangential_wind.copy()
        initial_tangential_wind_on_half_levels = tangential_wind_on_half_levels.copy()
        initial_vn_on_half_levels = vn_on_half_levels.copy()
        initial_horizontal_kinetic_energy_at_edges_on_model_levels = (
            horizontal_kinetic_energy_at_edges_on_model_levels.copy()
        )
        initial_contravariant_correction_at_edges_on_model_levels = (
            contravariant_correction_at_edges_on_model_levels.copy()
        )
        initial_horizontal_advection_of_w_at_edges_on_half_levels = (
            horizontal_advection_of_w_at_edges_on_half_levels.copy()
        )

        (
            tangential_wind,
            tangential_wind_on_half_levels,
            vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels,
        ) = cls._fused_velocity_advection_stencil_1_to_6_numpy(
            connectivities,
            tangential_wind_on_half_levels,
            vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels,
            vn,
            rbf_vec_coeff_e,
            wgtfac_e,
            ddxn_z_full,
            ddxt_z_full,
            wgtfacq_e,
            skip_compute_predictor_vertical_advection,
            nflatlev,
            vertical_end,
        )

        khalf_w_at_edge = mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy(
            connectivities, w, c_intp
        )

        if not skip_compute_predictor_vertical_advection:
            horizontal_advection_of_w_at_edges_on_half_levels = (
                compute_horizontal_advection_term_for_vertical_velocity_numpy(
                    connectivities,
                    vn_on_half_levels[:, :-1],
                    inv_dual_edge_length,
                    w,
                    tangential_wind_on_half_levels,
                    inv_primal_edge_length,
                    tangent_orientation,
                    khalf_w_at_edge,
                )
            )

        tangential_wind[:horizontal_start, :] = initial_tangential_wind[:horizontal_start, :]
        tangential_wind[horizontal_end:, :] = initial_tangential_wind[horizontal_end:, :]

        tangential_wind_on_half_levels[
            :horizontal_start, :
        ] = initial_tangential_wind_on_half_levels[:horizontal_start, :]
        tangential_wind_on_half_levels[horizontal_end:, :] = initial_tangential_wind_on_half_levels[
            horizontal_end:, :
        ]

        vn_on_half_levels[:horizontal_start, :] = initial_vn_on_half_levels[:horizontal_start, :]
        vn_on_half_levels[horizontal_end:, :] = initial_vn_on_half_levels[horizontal_end:, :]

        horizontal_kinetic_energy_at_edges_on_model_levels[
            :horizontal_start, :
        ] = initial_horizontal_kinetic_energy_at_edges_on_model_levels[:horizontal_start, :]
        horizontal_kinetic_energy_at_edges_on_model_levels[
            horizontal_end:, :
        ] = initial_horizontal_kinetic_energy_at_edges_on_model_levels[horizontal_end:, :]

        contravariant_correction_at_edges_on_model_levels[
            :horizontal_start, :
        ] = initial_contravariant_correction_at_edges_on_model_levels[:horizontal_start, :]
        contravariant_correction_at_edges_on_model_levels[
            horizontal_end:, :
        ] = initial_contravariant_correction_at_edges_on_model_levels[horizontal_end:, :]

        horizontal_advection_of_w_at_edges_on_half_levels[
            :horizontal_start, :
        ] = initial_horizontal_advection_of_w_at_edges_on_half_levels[:horizontal_start, :]
        horizontal_advection_of_w_at_edges_on_half_levels[
            horizontal_end:, :
        ] = initial_horizontal_advection_of_w_at_edges_on_half_levels[horizontal_end:, :]

        return dict(
            tangential_wind=tangential_wind,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn_on_half_levels=vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        horizontal_advection_of_w_at_edges_on_half_levels = data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim
        )
        tangential_wind = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        tangential_wind_on_half_levels = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        vn_on_half_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}
        )
        horizontal_kinetic_energy_at_edges_on_model_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        contravariant_correction_at_edges_on_model_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        w = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        rbf_vec_coeff_e = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EDim)
        wgtfac_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ddxn_z_full = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ddxt_z_full = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        inv_dual_edge_length = data_alloc.random_field(grid, dims.EdgeDim)
        inv_primal_edge_length = data_alloc.random_field(grid, dims.EdgeDim)
        tangent_orientation = data_alloc.random_field(grid, dims.EdgeDim)
        wgtfacq_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        c_intp = data_alloc.random_field(grid, dims.VertexDim, dims.V2CDim)

        nlev = grid.num_levels
        nflatlev = 13

        skip_compute_predictor_vertical_advection = False

        edge_domain = h_grid.domain(dims.EdgeDim)
        # For the ICON grid we use the proper domain bounds (otherwise we will run into non-protected skip values)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))
        vertical_start = 0
        vertical_end = nlev + 1

        return dict(
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            tangential_wind=tangential_wind,
            vn_on_half_levels=vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
            vn=vn,
            w=w,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            wgtfac_e=wgtfac_e,
            ddxn_z_full=ddxn_z_full,
            ddxt_z_full=ddxt_z_full,
            wgtfacq_e=wgtfacq_e,
            c_intp=c_intp,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
            skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
            nflatlev=nflatlev,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
        )
