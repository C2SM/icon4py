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

import icon4py.model.common.type_alias as ta
import icon4py.model.testing.helpers as test_helpers
from icon4py.model.atmosphere.dycore.dycore_states import (
    HorizontalPressureDiscretizationType,
    RhoThetaAdvectionType,
)
from icon4py.model.atmosphere.dycore.stencils.compute_edge_diagnostics_for_dycore_and_update_vn import (
    compute_theta_rho_face_values_and_pressure_gradient_and_update_vn,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc


rhotheta_avd_type = RhoThetaAdvectionType()
horzpres_discr_type = HorizontalPressureDiscretizationType()


def compute_theta_rho_face_value_by_miura_scheme_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    vn: np.ndarray,
    tangential_wind: np.ndarray,
    pos_on_tplane_e_x: np.ndarray,
    pos_on_tplane_e_y: np.ndarray,
    primal_normal_cell_x: np.ndarray,
    dual_normal_cell_x: np.ndarray,
    primal_normal_cell_y: np.ndarray,
    dual_normal_cell_y: np.ndarray,
    p_dthalf: float,
    reference_rho_at_edges_on_model_levels: np.ndarray,
    reference_theta_at_edges_on_model_levels: np.ndarray,
    ddx_perturbed_rho: np.ndarray,
    ddy_perturbed_rho: np.ndarray,
    ddx_perturbed_theta_v: np.ndarray,
    ddy_perturbed_theta_v: np.ndarray,
    perturbed_rho_at_cells_on_model_levels: np.ndarray,
    perturbed_theta_v_at_cells_on_model_levels: np.ndarray,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    e2c = connectivities[dims.E2CDim]
    pos_on_tplane_e_x = pos_on_tplane_e_x.reshape(e2c.shape)
    pos_on_tplane_e_y = pos_on_tplane_e_y.reshape(e2c.shape)
    primal_normal_cell_x = primal_normal_cell_x.reshape(e2c.shape)
    dual_normal_cell_x = dual_normal_cell_x.reshape(e2c.shape)
    primal_normal_cell_y = primal_normal_cell_y.reshape(e2c.shape)
    dual_normal_cell_y = dual_normal_cell_y.reshape(e2c.shape)

    lvn_pos = np.where(vn > 0.0, True, False)
    pos_on_tplane_e_x = np.expand_dims(pos_on_tplane_e_x, axis=-1)
    pos_on_tplane_e_y = np.expand_dims(pos_on_tplane_e_y, axis=-1)
    primal_normal_cell_x = np.expand_dims(primal_normal_cell_x, axis=-1)
    dual_normal_cell_x = np.expand_dims(dual_normal_cell_x, axis=-1)
    primal_normal_cell_y = np.expand_dims(primal_normal_cell_y, axis=-1)
    dual_normal_cell_y = np.expand_dims(dual_normal_cell_y, axis=-1)

    z_ntdistv_bary_1 = -(
        vn * p_dthalf + np.where(lvn_pos, pos_on_tplane_e_x[:, 0], pos_on_tplane_e_x[:, 1])
    )
    z_ntdistv_bary_2 = -(
        tangential_wind * p_dthalf
        + np.where(lvn_pos, pos_on_tplane_e_y[:, 0], pos_on_tplane_e_y[:, 1])
    )

    p_distv_bary_1 = np.where(
        lvn_pos,
        z_ntdistv_bary_1 * primal_normal_cell_x[:, 0] + z_ntdistv_bary_2 * dual_normal_cell_x[:, 0],
        z_ntdistv_bary_1 * primal_normal_cell_x[:, 1] + z_ntdistv_bary_2 * dual_normal_cell_x[:, 1],
    )

    p_distv_bary_2 = np.where(
        lvn_pos,
        z_ntdistv_bary_1 * primal_normal_cell_y[:, 0] + z_ntdistv_bary_2 * dual_normal_cell_y[:, 0],
        z_ntdistv_bary_1 * primal_normal_cell_y[:, 1] + z_ntdistv_bary_2 * dual_normal_cell_y[:, 1],
    )

    perturbed_rho_e2c = perturbed_rho_at_cells_on_model_levels[e2c]
    perturbed_theta_v_e2c = perturbed_theta_v_at_cells_on_model_levels[e2c]
    ddx_perturbed_rho_e2c = ddx_perturbed_rho[e2c]
    ddy_perturbed_rho_e2c = ddy_perturbed_rho[e2c]
    ddx_perturbed_theta_v_e2c = ddx_perturbed_theta_v[e2c]
    ddy_perturbed_theta_v_e2c = ddy_perturbed_theta_v[e2c]

    rho_at_edges_on_model_levels = np.where(
        vn > 0,
        reference_rho_at_edges_on_model_levels
        + perturbed_rho_e2c[:, 0]
        + p_distv_bary_1 * ddx_perturbed_rho_e2c[:, 0]
        + p_distv_bary_2 * ddy_perturbed_rho_e2c[:, 0],
        reference_rho_at_edges_on_model_levels
        + perturbed_rho_e2c[:, 1]
        + p_distv_bary_1 * ddx_perturbed_rho_e2c[:, 1]
        + p_distv_bary_2 * ddy_perturbed_rho_e2c[:, 1],
    )

    theta_v_at_edges_on_model_levels = np.where(
        vn > 0,
        reference_theta_at_edges_on_model_levels
        + perturbed_theta_v_e2c[:, 0]
        + p_distv_bary_1 * ddx_perturbed_theta_v_e2c[:, 0]
        + p_distv_bary_2 * ddy_perturbed_theta_v_e2c[:, 0],
        reference_theta_at_edges_on_model_levels
        + perturbed_theta_v_e2c[:, 1]
        + p_distv_bary_1 * ddx_perturbed_theta_v_e2c[:, 1]
        + p_distv_bary_2 * ddy_perturbed_theta_v_e2c[:, 1],
    )

    return rho_at_edges_on_model_levels, theta_v_at_edges_on_model_levels


class TestComputeThetaRhoPressureGradientAndUpdateVn(test_helpers.StencilTest):
    PROGRAM = compute_theta_rho_face_values_and_pressure_gradient_and_update_vn
    OUTPUTS = (
        "rho_at_edges_on_model_levels",
        "theta_v_at_edges_on_model_levels",
        "horizontal_pressure_gradient",
        "next_vn",
    )

    MARKERS = (
        pytest.mark.uses_as_offset,
        pytest.mark.skip_value_error,
        pytest.mark.embedded_remap_error,
    )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        rho_at_edges_on_model_levels: np.ndarray,
        theta_v_at_edges_on_model_levels: np.ndarray,
        horizontal_pressure_gradient: np.ndarray,
        next_vn: np.ndarray,
        current_vn: np.ndarray,
        tangential_wind: np.ndarray,
        reference_rho_at_edges_on_model_levels: np.ndarray,
        reference_theta_at_edges_on_model_levels: np.ndarray,
        perturbed_rho_at_cells_on_model_levels: np.ndarray,
        perturbed_theta_v_at_cells_on_model_levels: np.ndarray,
        temporal_extrapolation_of_perturbed_exner: np.ndarray,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: np.ndarray,
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: np.ndarray,
        hydrostatic_correction_on_lowest_level: np.ndarray,
        predictor_normal_wind_advective_tendency: np.ndarray,
        normal_wind_tendency_due_to_slow_physics_process: np.ndarray,
        normal_wind_iau_increments: np.ndarray,
        geofac_grg_x: np.ndarray,
        geofac_grg_y: np.ndarray,
        pos_on_tplane_e_x: np.ndarray,
        pos_on_tplane_e_y: np.ndarray,
        primal_normal_cell_x: np.ndarray,
        dual_normal_cell_x: np.ndarray,
        primal_normal_cell_y: np.ndarray,
        dual_normal_cell_y: np.ndarray,
        ddxn_z_full: np.ndarray,
        c_lin_e: np.ndarray,
        ikoffset: np.ndarray,
        zdiff_gradp: np.ndarray,
        ipeidx_dsl: np.ndarray,
        pg_exdist: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        dtime: ta.wpfloat,
        cpd: ta.wpfloat,
        iau_wgt_dyn: ta.wpfloat,
        is_iau_active: gtx.int32,
        limited_area: gtx.int32,
        iadv_rhotheta: gtx.int32,
        igradp_method: gtx.int32,
        start_edge_halo_level_2: gtx.int32,
        end_edge_halo_level_2: gtx.int32,
        start_edge_lateral_boundary: gtx.int32,
        end_edge_halo: gtx.int32,
        start_edge_lateral_boundary_level_7: gtx.int32,
        start_edge_nudging_level_2: gtx.int32,
        end_edge_local: gtx.int32,
        end_edge_end: gtx.int32,
        nflatlev: gtx.int32,
        nflat_gradp: gtx.int32,
        horizontal_end: gtx.int32,
        vertical_end: gtx.int32,
        **kwargs,
    ) -> dict:
        vert_idx = np.arange(vertical_end)
        horz_idx = np.arange(horizontal_end)[:, np.newaxis]
        default_shape = perturbed_rho_at_cells_on_model_levels.shape

        ddx_perturbed_rho = np.zeros(default_shape)
        ddy_perturbed_rho = np.zeros(default_shape)
        ddx_perturbed_theta_v = np.zeros(default_shape)
        ddy_perturbed_theta_v = np.zeros(default_shape)

        if iadv_rhotheta == rhotheta_avd_type.MIURA:
            # Compute Green-Gauss gradients for rho and theta
            c2e2cO = connectivities[dims.C2E2CODim]

            geofac_grg_x = np.expand_dims(geofac_grg_x, axis=-1)
            ddx_perturbed_rho = np.sum(
                np.where(
                    (c2e2cO != -1)[:, :, np.newaxis],
                    geofac_grg_x * perturbed_rho_at_cells_on_model_levels[c2e2cO],
                    0,
                ),
                axis=1,
            )
            ddx_perturbed_theta_v = np.sum(
                np.where(
                    (c2e2cO != -1)[:, :, np.newaxis],
                    geofac_grg_x * perturbed_theta_v_at_cells_on_model_levels[c2e2cO],
                    0,
                ),
                axis=1,
            )

            geofac_grg_y = np.expand_dims(geofac_grg_y, axis=-1)
            ddy_perturbed_rho = np.sum(
                np.where(
                    (c2e2cO != -1)[:, :, np.newaxis],
                    geofac_grg_y * perturbed_rho_at_cells_on_model_levels[c2e2cO],
                    0,
                ),
                axis=1,
            )
            ddy_perturbed_theta_v = np.sum(
                np.where(
                    (c2e2cO != -1)[:, :, np.newaxis],
                    geofac_grg_y * perturbed_theta_v_at_cells_on_model_levels[c2e2cO],
                    0,
                ),
                axis=1,
            )

        if iadv_rhotheta <= 2:
            # if idiv_method == 1:
            (rho_at_edges_on_model_levels, theta_v_at_edges_on_model_levels) = np.where(
                (horz_idx >= start_edge_halo_level_2) & (horz_idx < end_edge_halo_level_2),
                (
                    np.zeros_like(rho_at_edges_on_model_levels),
                    np.zeros_like(theta_v_at_edges_on_model_levels),
                ),
                (rho_at_edges_on_model_levels, theta_v_at_edges_on_model_levels),
            )

            # initialize also nest boundary points with zero
            if limited_area:
                (rho_at_edges_on_model_levels, theta_v_at_edges_on_model_levels) = np.where(
                    (horz_idx >= start_edge_lateral_boundary) & (horz_idx < end_edge_halo),
                    (
                        np.zeros_like(rho_at_edges_on_model_levels),
                        np.zeros_like(theta_v_at_edges_on_model_levels),
                    ),
                    (rho_at_edges_on_model_levels, theta_v_at_edges_on_model_levels),
                )

            if iadv_rhotheta == rhotheta_avd_type.MIURA:
                # Compute upwind-biased values for rho and theta starting from centered differences
                # Note: the length of the backward trajectory should be 0.5*dtime*(vn,tangential_wind) in order to arrive
                # at a second-order accurate FV discretization, but twice the length is needed for numerical stability
                (rho_at_edges_on_model_levels, theta_v_at_edges_on_model_levels) = np.where(
                    (start_edge_lateral_boundary_level_7 <= horz_idx) & (horz_idx < end_edge_halo),
                    compute_theta_rho_face_value_by_miura_scheme_numpy(
                        connectivities=connectivities,
                        vn=current_vn,
                        tangential_wind=tangential_wind,
                        pos_on_tplane_e_x=pos_on_tplane_e_x,
                        pos_on_tplane_e_y=pos_on_tplane_e_y,
                        primal_normal_cell_x=primal_normal_cell_x,
                        dual_normal_cell_x=dual_normal_cell_x,
                        primal_normal_cell_y=primal_normal_cell_y,
                        dual_normal_cell_y=dual_normal_cell_y,
                        p_dthalf=float(0.5 * dtime),
                        reference_rho_at_edges_on_model_levels=reference_rho_at_edges_on_model_levels,
                        reference_theta_at_edges_on_model_levels=reference_theta_at_edges_on_model_levels,
                        ddx_perturbed_rho=ddx_perturbed_rho,
                        ddy_perturbed_rho=ddy_perturbed_rho,
                        ddx_perturbed_theta_v=ddx_perturbed_theta_v,
                        ddy_perturbed_theta_v=ddy_perturbed_theta_v,
                        perturbed_rho_at_cells_on_model_levels=perturbed_rho_at_cells_on_model_levels,
                        perturbed_theta_v_at_cells_on_model_levels=perturbed_theta_v_at_cells_on_model_levels,
                    ),
                    (rho_at_edges_on_model_levels, theta_v_at_edges_on_model_levels),
                )

        # Remaining computations at edge points
        e2c = connectivities[dims.E2CDim]
        temporal_extrapolation_of_perturbed_exner_at_edges = (
            temporal_extrapolation_of_perturbed_exner[e2c]
        )
        weighted_temporal_extrapolation_of_perturbed_exner_at_edges = (
            temporal_extrapolation_of_perturbed_exner_at_edges[:, 1]
            - temporal_extrapolation_of_perturbed_exner_at_edges[:, 0]
        )
        inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)

        horizontal_pressure_gradient = np.where(
            (start_edge_nudging_level_2 <= horz_idx)
            & (horz_idx < end_edge_local)
            & (vert_idx < nflatlev),
            inv_dual_edge_length * weighted_temporal_extrapolation_of_perturbed_exner_at_edges,
            horizontal_pressure_gradient,
        )

        if igradp_method == horzpres_discr_type.TAYLOR_HYDRO:

            def _apply_index_field_for_multi_level_pressure_gradient(
                shape: tuple,
                to_index: np.ndarray,
                neighbor_table: np.ndarray,
                offset_field: np.ndarray,
            ) -> np.ndarray:
                indexed = np.zeros(shape)
                for iprimary in range(shape[0]):
                    for isparse in range(shape[1]):
                        for ik in range(shape[2]):
                            indexed[iprimary, isparse, ik] = to_index[
                                neighbor_table[iprimary, isparse],
                                ik + offset_field[iprimary, isparse, ik],
                            ]
                return indexed

            def at_neighbor(i: int) -> np.ndarray:
                return temporal_extrapolation_of_perturbed_exner_at_kidx[:, i, :] + zdiff_gradp[
                    :, i, :
                ] * (
                    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_at_kidx[
                        :, i, :
                    ]
                    + zdiff_gradp[:, i, :]
                    * d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_at_kidx[
                        :, i, :
                    ]
                )

            c_lin_e = np.expand_dims(c_lin_e, axis=-1)

            # horizontal gradient of Exner pressure, including metric correction
            # horizontal gradient of Exner pressure, Taylor-expansion-based reconstruction
            horizontal_pressure_gradient = np.where(
                (start_edge_nudging_level_2 <= horz_idx)
                & (horz_idx < end_edge_local)
                & (vert_idx >= nflatlev)
                & (vert_idx < (nflat_gradp + gtx.int32(1))),
                inv_dual_edge_length * weighted_temporal_extrapolation_of_perturbed_exner_at_edges
                - ddxn_z_full
                * np.sum(
                    c_lin_e * ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels[e2c],
                    axis=1,
                ),
                horizontal_pressure_gradient,
            )

            full_shape = e2c.shape + zdiff_gradp.shape[1:]
            zdiff_gradp = zdiff_gradp.reshape(full_shape)
            ikoffset = ikoffset.reshape(full_shape)

            temporal_extrapolation_of_perturbed_exner_at_kidx = (
                _apply_index_field_for_multi_level_pressure_gradient(
                    full_shape, temporal_extrapolation_of_perturbed_exner, e2c, ikoffset
                )
            )
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_at_kidx = (
                _apply_index_field_for_multi_level_pressure_gradient(
                    full_shape,
                    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
                    e2c,
                    ikoffset,
                )
            )
            d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_at_kidx = (
                _apply_index_field_for_multi_level_pressure_gradient(
                    full_shape,
                    d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
                    e2c,
                    ikoffset,
                )
            )
            sum_expr = at_neighbor(1) - at_neighbor(0)
            horizontal_pressure_gradient = np.where(
                (start_edge_nudging_level_2 <= horz_idx)
                & (horz_idx < end_edge_local)
                & (vert_idx >= (nflat_gradp + gtx.int32(1))),
                inv_dual_edge_length * sum_expr,
                horizontal_pressure_gradient,
            )

            hydrostatic_correction = np.repeat(
                np.expand_dims(hydrostatic_correction_on_lowest_level, axis=-1),
                horizontal_pressure_gradient.shape[1],
                axis=1,
            )
            horizontal_pressure_gradient = np.where(
                (start_edge_nudging_level_2 <= horz_idx) & (horz_idx < end_edge_end),
                np.where(
                    ipeidx_dsl,
                    horizontal_pressure_gradient + hydrostatic_correction * pg_exdist,
                    horizontal_pressure_gradient,
                ),
                horizontal_pressure_gradient,
            )

        next_vn = np.where(
            (start_edge_nudging_level_2 <= horz_idx) & (horz_idx < end_edge_local),
            current_vn
            + dtime
            * (
                predictor_normal_wind_advective_tendency
                + normal_wind_tendency_due_to_slow_physics_process
                - cpd * theta_v_at_edges_on_model_levels * horizontal_pressure_gradient
            ),
            next_vn,
        )

        if is_iau_active:
            next_vn = np.where(
                (start_edge_nudging_level_2 <= horz_idx) & (horz_idx < end_edge_local),
                next_vn + (iau_wgt_dyn * normal_wind_iau_increments),
                next_vn,
            )

        return dict(
            rho_at_edges_on_model_levels=rho_at_edges_on_model_levels,
            theta_v_at_edges_on_model_levels=theta_v_at_edges_on_model_levels,
            horizontal_pressure_gradient=horizontal_pressure_gradient,
            next_vn=next_vn,
        )

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict:
        geofac_grg_x = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CODim)
        geofac_grg_y = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CODim)
        current_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        tangential_wind = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        pos_on_tplane_e_x = data_alloc.random_field(grid, dims.ECDim)
        pos_on_tplane_e_y = data_alloc.random_field(grid, dims.ECDim)
        primal_normal_cell_x = data_alloc.random_field(grid, dims.ECDim)
        dual_normal_cell_x = data_alloc.random_field(grid, dims.ECDim)
        primal_normal_cell_y = data_alloc.random_field(grid, dims.ECDim)
        dual_normal_cell_y = data_alloc.random_field(grid, dims.ECDim)
        reference_rho_at_edges_on_model_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        reference_theta_at_edges_on_model_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        perturbed_rho_at_cells_on_model_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        perturbed_theta_v_at_cells_on_model_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        ddxn_z_full = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        c_lin_e = data_alloc.random_field(grid, dims.EdgeDim, dims.E2CDim)
        temporal_extrapolation_of_perturbed_exner = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = (
            data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        )
        hydrostatic_correction_on_lowest_level = data_alloc.random_field(grid, dims.EdgeDim)
        zdiff_gradp = data_alloc.random_field(grid, dims.ECDim, dims.KDim)
        ipeidx_dsl = data_alloc.random_mask(grid, dims.EdgeDim, dims.KDim)
        pg_exdist = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        inv_dual_edge_length = data_alloc.random_field(grid, dims.EdgeDim)
        predictor_normal_wind_advective_tendency = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        normal_wind_tendency_due_to_slow_physics_process = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        normal_wind_iau_increments = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        next_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        theta_v_at_edges_on_model_levels = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        horizontal_pressure_gradient = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        rho_at_edges_on_model_levels = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)

        ikoffset = data_alloc.zero_field(
            grid, dims.EdgeDim, dims.E2CDim, dims.KDim, dtype=gtx.int32
        )
        rng = np.random.default_rng()
        k_levels = grid.num_levels

        ikoffset_np = np.zeros_like(ikoffset.asnumpy())
        for k in range(k_levels):
            # construct offsets that reach all k-levels except the last (because we are using the entries of this field with `+1`)
            ikoffset_np[:, :, k] = rng.integers(
                low=0 - k,
                high=k_levels - k - 1,
                size=(ikoffset.asnumpy().shape[0], ikoffset.asnumpy().shape[1]),
            )
        ikoffset = data_alloc.flatten_first_two_dims(dims.ECDim, dims.KDim, field=ikoffset_np)

        dtime = 0.9
        cpd = 1004.64
        iau_wgt_dyn = 1.0
        is_iau_active = True
        limited_area = True
        iadv_rhotheta = 2
        igradp_method = 3
        edge_domain = h_grid.domain(dims.EdgeDim)

        start_edge_halo_level_2 = grid.start_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))
        end_edge_halo_level_2 = grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))
        start_edge_lateral_boundary = grid.end_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY))
        end_edge_halo = grid.end_index(edge_domain(h_grid.Zone.HALO))
        start_edge_lateral_boundary_level_7 = grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7)
        )
        start_edge_nudging_level_2 = grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
        end_edge_local = grid.end_index(edge_domain(h_grid.Zone.LOCAL))
        end_edge_end = grid.num_edges
        nflatlev = 4
        nflat_gradp = 27

        return dict(
            rho_at_edges_on_model_levels=rho_at_edges_on_model_levels,
            theta_v_at_edges_on_model_levels=theta_v_at_edges_on_model_levels,
            horizontal_pressure_gradient=horizontal_pressure_gradient,
            next_vn=next_vn,
            current_vn=current_vn,
            tangential_wind=tangential_wind,
            reference_rho_at_edges_on_model_levels=reference_rho_at_edges_on_model_levels,
            reference_theta_at_edges_on_model_levels=reference_theta_at_edges_on_model_levels,
            perturbed_rho_at_cells_on_model_levels=perturbed_rho_at_cells_on_model_levels,
            perturbed_theta_v_at_cells_on_model_levels=perturbed_theta_v_at_cells_on_model_levels,
            temporal_extrapolation_of_perturbed_exner=temporal_extrapolation_of_perturbed_exner,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            hydrostatic_correction_on_lowest_level=hydrostatic_correction_on_lowest_level,
            predictor_normal_wind_advective_tendency=predictor_normal_wind_advective_tendency,
            normal_wind_tendency_due_to_slow_physics_process=normal_wind_tendency_due_to_slow_physics_process,
            normal_wind_iau_increments=normal_wind_iau_increments,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            pos_on_tplane_e_x=pos_on_tplane_e_x,
            pos_on_tplane_e_y=pos_on_tplane_e_y,
            primal_normal_cell_x=primal_normal_cell_x,
            dual_normal_cell_x=dual_normal_cell_x,
            primal_normal_cell_y=primal_normal_cell_y,
            dual_normal_cell_y=dual_normal_cell_y,
            ddxn_z_full=ddxn_z_full,
            c_lin_e=c_lin_e,
            ikoffset=ikoffset,
            zdiff_gradp=zdiff_gradp,
            ipeidx_dsl=ipeidx_dsl,
            pg_exdist=pg_exdist,
            inv_dual_edge_length=inv_dual_edge_length,
            dtime=dtime,
            cpd=cpd,
            iau_wgt_dyn=iau_wgt_dyn,
            is_iau_active=is_iau_active,
            limited_area=limited_area,
            iadv_rhotheta=iadv_rhotheta,
            igradp_method=igradp_method,
            nflatlev=nflatlev,
            nflat_gradp=nflat_gradp,
            start_edge_halo_level_2=start_edge_halo_level_2,
            end_edge_halo_level_2=end_edge_halo_level_2,
            start_edge_lateral_boundary=start_edge_lateral_boundary,
            end_edge_halo=end_edge_halo,
            start_edge_lateral_boundary_level_7=start_edge_lateral_boundary_level_7,
            start_edge_nudging_level_2=start_edge_nudging_level_2,
            end_edge_local=end_edge_local,
            end_edge_end=end_edge_end,
            horizontal_start=0,
            horizontal_end=grid.num_edges,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
