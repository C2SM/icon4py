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
from icon4py.model.atmosphere.dycore.stencils.compute_advection_in_horizontal_momentum_equation import (
    compute_advection_in_horizontal_momentum_equation,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.testing import helpers as test_helpers

from .test_add_extra_diffusion_for_normal_wind_tendency_approaching_cfl import (
    add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_numpy,
)
from .test_compute_advective_normal_wind_tendency import (
    compute_advective_normal_wind_tendency_numpy,
)
from .test_mo_math_divrot_rot_vertex_ri_dsl import mo_math_divrot_rot_vertex_ri_dsl_numpy


class TestFusedVelocityAdvectionStencilsHMomentum(test_helpers.StencilTest):
    PROGRAM = compute_advection_in_horizontal_momentum_equation
    OUTPUTS = ("normal_wind_advective_tendency",)
    MARKERS = (pytest.mark.embedded_remap_error,)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        normal_wind_advective_tendency: np.ndarray,
        vn: np.ndarray,
        horizontal_kinetic_energy_at_edges_on_model_levels: np.ndarray,
        horizontal_kinetic_energy_at_cells_on_model_levels: np.ndarray,
        tangential_wind: np.ndarray,
        coriolis_frequency: np.ndarray,
        contravariant_corrected_w_at_cells_on_model_levels: np.ndarray,
        vn_on_half_levels: np.ndarray,
        geofac_rot: np.ndarray,
        coeff_gradekin: np.ndarray,
        c_lin_e: np.ndarray,
        ddqz_z_full_e: np.ndarray,
        area_edge: np.ndarray,
        tangent_orientation: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        geofac_grdiv: np.ndarray,
        cfl_w_limit: ta.wpfloat,
        scalfac_exdiff: ta.wpfloat,
        d_time: ta.wpfloat,
        levelmask: np.ndarray,
        nlev: int,
        end_index_of_damping_layer: int,
        **kwargs: Any,
    ) -> dict:
        normal_wind_advective_tendency_cp = normal_wind_advective_tendency.copy()
        k = np.arange(nlev)

        upward_vorticity_at_vertices = mo_math_divrot_rot_vertex_ri_dsl_numpy(
            connectivities, vn, geofac_rot
        )

        normal_wind_advective_tendency = compute_advective_normal_wind_tendency_numpy(
            connectivities,
            horizontal_kinetic_energy_at_edges_on_model_levels,
            coeff_gradekin,
            horizontal_kinetic_energy_at_cells_on_model_levels,
            upward_vorticity_at_vertices,
            tangential_wind,
            coriolis_frequency,
            c_lin_e,
            contravariant_corrected_w_at_cells_on_model_levels,
            vn_on_half_levels,
            ddqz_z_full_e,
        )

        condition = (np.maximum(3, end_index_of_damping_layer - 2) - 1 <= k) & (k < nlev - 4)

        normal_wind_advective_tendency_extra_diffu = (
            add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_numpy(
                connectivities,
                levelmask,
                c_lin_e,
                contravariant_corrected_w_at_cells_on_model_levels,
                ddqz_z_full_e,
                area_edge,
                tangent_orientation,
                inv_primal_edge_length,
                upward_vorticity_at_vertices,
                geofac_grdiv,
                vn,
                normal_wind_advective_tendency,
                cfl_w_limit,
                scalfac_exdiff,
                d_time,
            )
        )

        normal_wind_advective_tendency = np.where(
            condition, normal_wind_advective_tendency_extra_diffu, normal_wind_advective_tendency
        )
        # restriction of execution domain
        normal_wind_advective_tendency[
            0 : kwargs["horizontal_start"], :
        ] = normal_wind_advective_tendency_cp[0 : kwargs["horizontal_start"], :]
        normal_wind_advective_tendency[
            kwargs["horizontal_end"] :, :
        ] = normal_wind_advective_tendency_cp[kwargs["horizontal_end"] :, :]

        return dict(normal_wind_advective_tendency=normal_wind_advective_tendency)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        normal_wind_advective_tendency = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        horizontal_kinetic_energy_at_edges_on_model_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        horizontal_kinetic_energy_at_cells_on_model_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        tangential_wind = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        coriolis_frequency = data_alloc.random_field(grid, dims.EdgeDim)
        contravariant_corrected_w_at_cells_on_model_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        vn_on_half_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}
        )
        coeff_gradekin = data_alloc.random_field(grid, dims.ECDim)
        c_lin_e = data_alloc.random_field(grid, dims.EdgeDim, dims.E2CDim)
        ddqz_z_full_e = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, low=0.0
        )  # this makes sure that the simplified stencil produces the same result as the numpy version
        levelmask = data_alloc.random_mask(grid, dims.KDim, extend={dims.KDim: 1})
        area_edge = data_alloc.random_field(grid, dims.EdgeDim)
        tangent_orientation = data_alloc.random_field(grid, dims.EdgeDim)
        inv_primal_edge_length = data_alloc.random_field(grid, dims.EdgeDim)
        geofac_grdiv = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EODim)

        geofac_rot = data_alloc.random_field(grid, dims.VertexDim, dims.V2EDim)
        cfl_w_limit = 4.0
        scalfac_exdiff = 6.0
        d_time = 2.0

        nlev = grid.num_levels

        end_index_of_damping_layer = 5
        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.LOCAL))

        return dict(
            normal_wind_advective_tendency=normal_wind_advective_tendency,
            vn=vn,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            horizontal_kinetic_energy_at_cells_on_model_levels=horizontal_kinetic_energy_at_cells_on_model_levels,
            tangential_wind=tangential_wind,
            coriolis_frequency=coriolis_frequency,
            contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
            vn_on_half_levels=vn_on_half_levels,
            geofac_rot=geofac_rot,
            coeff_gradekin=coeff_gradekin,
            c_lin_e=c_lin_e,
            ddqz_z_full_e=ddqz_z_full_e,
            area_edge=area_edge,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
            geofac_grdiv=geofac_grdiv,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            d_time=d_time,
            levelmask=levelmask,
            nlev=nlev,
            end_index_of_damping_layer=end_index_of_damping_layer,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
