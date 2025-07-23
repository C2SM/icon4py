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

import icon4py.model.common.type_alias as ta
import icon4py.model.testing.helpers as test_helpers
from icon4py.model.atmosphere.dycore.dycore_states import DivergenceDampingOrder
from icon4py.model.atmosphere.dycore.stencils.compute_edge_diagnostics_for_dycore_and_update_vn import (
    apply_divergence_damping_and_update_vn,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc


divergence_damp_order = DivergenceDampingOrder()


class TestApplyDivergenceDampingAndUpdateVn(test_helpers.StencilTest):
    PROGRAM = apply_divergence_damping_and_update_vn
    OUTPUTS = ("next_vn",)

    MARKERS = (pytest.mark.embedded_remap_error,)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        horizontal_gradient_of_normal_wind_divergence: np.ndarray,
        next_vn: np.ndarray,
        current_vn: np.ndarray,
        dwdz_at_cells_on_model_levels: np.ndarray,
        predictor_normal_wind_advective_tendency: np.ndarray,
        corrector_normal_wind_advective_tendency: np.ndarray,
        normal_wind_tendency_due_to_slow_physics_process: np.ndarray,
        normal_wind_iau_increment: np.ndarray,
        theta_v_at_edges_on_model_levels: np.ndarray,
        horizontal_pressure_gradient: np.ndarray,
        reduced_fourth_order_divdamp_coeff_at_nest_boundary: np.ndarray,
        fourth_order_divdamp_scaling_coeff: np.ndarray,
        second_order_divdamp_scaling_coeff: ta.wpfloat,
        horizontal_mask_for_3d_divdamp: np.ndarray,
        scaling_factor_for_3d_divdamp: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        nudgecoeff_e: np.ndarray,
        geofac_grdiv: np.ndarray,
        advection_explicit_weight_parameter: ta.wpfloat,
        advection_implicit_weight_parameter: ta.wpfloat,
        dtime: ta.wpfloat,
        iau_wgt_dyn: ta.wpfloat,
        is_iau_active: gtx.int32,
        limited_area: gtx.int32,
        apply_2nd_order_divergence_damping: bool,
        apply_4th_order_divergence_damping: bool,
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
        vertical_start: gtx.int32,
        vertical_end: gtx.int32,
    ) -> dict:
        horz_idx = np.arange(horizontal_end)[:, np.newaxis]

        scaling_factor_for_3d_divdamp = np.expand_dims(scaling_factor_for_3d_divdamp, axis=0)
        horizontal_mask_for_3d_divdamp = np.expand_dims(horizontal_mask_for_3d_divdamp, axis=-1)
        inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)

        e2c = connectivities[dims.E2CDim]
        dwdz_at_edges_on_model_levels = dwdz_at_cells_on_model_levels[e2c]
        weighted_dwdz_at_edges_on_model_levels = (
            dwdz_at_edges_on_model_levels[:, 1] - dwdz_at_edges_on_model_levels[:, 0]
        )

        horizontal_gradient_of_total_divergence = horizontal_gradient_of_normal_wind_divergence + (
            horizontal_mask_for_3d_divdamp
            * scaling_factor_for_3d_divdamp
            * inv_dual_edge_length
            * weighted_dwdz_at_edges_on_model_levels
        )

        next_vn = np.where(
            (horizontal_start <= horz_idx) & (horz_idx < horizontal_end),
            current_vn
            + dtime
            * (
                advection_explicit_weight_parameter * predictor_normal_wind_advective_tendency
                + advection_implicit_weight_parameter * corrector_normal_wind_advective_tendency
                + normal_wind_tendency_due_to_slow_physics_process
                - constants.CPD * theta_v_at_edges_on_model_levels * horizontal_pressure_gradient
            ),
            next_vn,
        )

        if apply_4th_order_divergence_damping:
            e2c2eO = connectivities[dims.E2C2EODim]
            # verified for e-10
            squared_horizontal_gradient_of_total_divergence = np.where(
                (horizontal_start <= horz_idx) & (horz_idx < horizontal_end),
                np.sum(
                    np.where(
                        (e2c2eO != -1)[:, :, np.newaxis],
                        horizontal_gradient_of_total_divergence[e2c2eO]
                        * np.expand_dims(geofac_grdiv, axis=-1),
                        0,
                    ),
                    axis=1,
                ),
                np.zeros_like(horizontal_gradient_of_total_divergence),
            )

        if apply_2nd_order_divergence_damping:
            next_vn = np.where(
                (horizontal_start <= horz_idx) & (horz_idx < horizontal_end),
                next_vn
                + (second_order_divdamp_scaling_coeff * horizontal_gradient_of_total_divergence),
                next_vn,
            )

        if apply_4th_order_divergence_damping:
            if limited_area:
                next_vn = np.where(
                    (horizontal_start <= horz_idx) & (horz_idx < horizontal_end),
                    next_vn
                    + (
                        fourth_order_divdamp_scaling_coeff
                        + reduced_fourth_order_divdamp_coeff_at_nest_boundary
                        * np.expand_dims(nudgecoeff_e, axis=-1)
                    )
                    * squared_horizontal_gradient_of_total_divergence,
                    next_vn,
                )
            else:
                next_vn = np.where(
                    (horizontal_start <= horz_idx) & (horz_idx < horizontal_end),
                    next_vn
                    + (
                        np.expand_dims(fourth_order_divdamp_scaling_coeff, axis=0)
                        * squared_horizontal_gradient_of_total_divergence
                    ),
                    next_vn,
                )

        if is_iau_active:
            next_vn = np.where(
                (horizontal_start <= horz_idx) & (horz_idx < horizontal_end),
                next_vn + (iau_wgt_dyn * normal_wind_iau_increment),
                next_vn,
            )

        return dict(next_vn=next_vn)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        current_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        horizontal_mask_for_3d_divdamp = data_alloc.random_field(grid, dims.EdgeDim)
        scaling_factor_for_3d_divdamp = data_alloc.random_field(grid, dims.KDim)
        dwdz_at_cells_on_model_levels = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        inv_dual_edge_length = data_alloc.random_field(grid, dims.EdgeDim)
        corrector_normal_wind_advective_tendency = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        predictor_normal_wind_advective_tendency = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        normal_wind_tendency_due_to_slow_physics_process = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        horizontal_gradient_of_normal_wind_divergence = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        normal_wind_iau_increment = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        next_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        theta_v_at_edges_on_model_levels = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        horizontal_pressure_gradient = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        geofac_grdiv = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EODim)
        fourth_order_divdamp_scaling_coeff = data_alloc.random_field(grid, dims.KDim)
        reduced_fourth_order_divdamp_coeff_at_nest_boundary = data_alloc.random_field(
            grid, dims.KDim
        )
        nudgecoeff_e = data_alloc.random_field(grid, dims.EdgeDim)

        dtime = 0.9
        advection_implicit_weight_parameter = 0.75
        advection_explicit_weight_parameter = 0.25
        iau_wgt_dyn = 1.0
        is_iau_active = True
        fourth_order_divdamp_factor = 0.004
        second_order_divdamp_factor = 0.012
        divdamp_order = 24
        second_order_divdamp_scaling_coeff = 194588.14247428576
        apply_2nd_order_divergence_damping = (divdamp_order == divergence_damp_order.COMBINED) and (
            second_order_divdamp_scaling_coeff > 1.0e-6
        )
        apply_4th_order_divergence_damping = (
            divdamp_order == divergence_damp_order.FOURTH_ORDER
        ) or (
            (divdamp_order == divergence_damp_order.COMBINED)
            and (second_order_divdamp_factor <= (4.0 * fourth_order_divdamp_factor))
        )

        limited_area = True
        edge_domain = h_grid.domain(dims.EdgeDim)

        start_edge_nudging_level_2 = grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
        end_edge_local = grid.end_index(edge_domain(h_grid.Zone.LOCAL))

        return dict(
            horizontal_gradient_of_normal_wind_divergence=horizontal_gradient_of_normal_wind_divergence,
            next_vn=next_vn,
            current_vn=current_vn,
            dwdz_at_cells_on_model_levels=dwdz_at_cells_on_model_levels,
            predictor_normal_wind_advective_tendency=predictor_normal_wind_advective_tendency,
            corrector_normal_wind_advective_tendency=corrector_normal_wind_advective_tendency,
            normal_wind_tendency_due_to_slow_physics_process=normal_wind_tendency_due_to_slow_physics_process,
            normal_wind_iau_increment=normal_wind_iau_increment,
            theta_v_at_edges_on_model_levels=theta_v_at_edges_on_model_levels,
            horizontal_pressure_gradient=horizontal_pressure_gradient,
            reduced_fourth_order_divdamp_coeff_at_nest_boundary=reduced_fourth_order_divdamp_coeff_at_nest_boundary,
            fourth_order_divdamp_scaling_coeff=fourth_order_divdamp_scaling_coeff,
            second_order_divdamp_scaling_coeff=second_order_divdamp_scaling_coeff,
            horizontal_mask_for_3d_divdamp=horizontal_mask_for_3d_divdamp,
            scaling_factor_for_3d_divdamp=scaling_factor_for_3d_divdamp,
            inv_dual_edge_length=inv_dual_edge_length,
            nudgecoeff_e=nudgecoeff_e,
            geofac_grdiv=geofac_grdiv,
            advection_explicit_weight_parameter=advection_explicit_weight_parameter,
            advection_implicit_weight_parameter=advection_implicit_weight_parameter,
            dtime=dtime,
            iau_wgt_dyn=iau_wgt_dyn,
            is_iau_active=is_iau_active,
            limited_area=limited_area,
            apply_2nd_order_divergence_damping=apply_2nd_order_divergence_damping,
            apply_4th_order_divergence_damping=apply_4th_order_divergence_damping,
            horizontal_start=start_edge_nudging_level_2,
            horizontal_end=end_edge_local,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
