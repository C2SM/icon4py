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

from icon4py.model.atmosphere.dycore.stencils.compute_advection_in_vertical_momentum_equation import (
    compute_contravariant_correction_and_advection_in_vertical_momentum_equation,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import helpers as test_helpers

from .test_compute_advection_in_vertical_momentum_equation import (
    compute_advective_vertical_wind_tendency_and_apply_diffusion_numpy,
    compute_maximum_cfl_and_clip_contravariant_vertical_velocity_numpy,
    interpolate_contravariant_vertical_velocity_to_full_levels_numpy,
)
from .test_interpolate_cell_field_to_half_levels_vp import (
    interpolate_cell_field_to_half_levels_vp_numpy,
)
from .test_interpolate_to_cell_center import (
    interpolate_to_cell_center_numpy,
)


def interpolate_contravariant_correction_to_cells_on_half_levels_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    contravariant_correction_at_cells_on_half_levels: np.ndarray,
    contravariant_correction_at_edges_on_model_levels: np.ndarray,
    e_bln_c_s: np.ndarray,
    wgtfac_c: np.ndarray,
    nflatlev: int,
    nlev: int,
) -> np.ndarray:
    k = np.arange(nlev)

    condition1 = k >= nflatlev
    contravariant_correction_at_cells_model_levels = np.where(
        condition1,
        interpolate_to_cell_center_numpy(
            connectivities, contravariant_correction_at_edges_on_model_levels, e_bln_c_s
        ),
        np.zeros_like(contravariant_correction_at_cells_on_half_levels),
    )

    condition2 = k >= nflatlev + 1
    contravariant_correction_at_cells_on_half_levels = np.where(
        condition2,
        interpolate_cell_field_to_half_levels_vp_numpy(
            wgtfac_c=wgtfac_c, interpolant=contravariant_correction_at_cells_model_levels
        ),
        np.zeros_like(contravariant_correction_at_cells_on_half_levels),
    )

    return contravariant_correction_at_cells_on_half_levels


class TestFusedVelocityAdvectionStencilVMomentum(test_helpers.StencilTest):
    PROGRAM = compute_contravariant_correction_and_advection_in_vertical_momentum_equation
    OUTPUTS = (
        "contravariant_correction_at_cells_on_half_levels",
        "vertical_wind_advective_tendency",
        "contravariant_corrected_w_at_cells_on_model_levels",
        "vertical_cfl",
    )
    MARKERS = (pytest.mark.embedded_remap_error,)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        contravariant_correction_at_cells_on_half_levels: np.ndarray,
        vertical_wind_advective_tendency: np.ndarray,
        contravariant_corrected_w_at_cells_on_model_levels: np.ndarray,
        vertical_cfl: np.ndarray,
        w: np.ndarray,
        tangential_wind_on_half_levels: np.ndarray,
        vn_on_half_levels: np.ndarray,
        contravariant_correction_at_edges_on_model_levels: np.ndarray,
        coeff1_dwdz: np.ndarray,
        coeff2_dwdz: np.ndarray,
        c_intp: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        tangent_orientation: np.ndarray,
        e_bln_c_s: np.ndarray,
        wgtfac_c: np.ndarray,
        ddqz_z_half: np.ndarray,
        area: np.ndarray,
        geofac_n2s: np.ndarray,
        scalfac_exdiff: ta.wpfloat,
        cfl_w_limit: ta.wpfloat,
        dtime: ta.wpfloat,
        owner_mask: np.ndarray,
        nflatlev: int,
        end_index_of_damping_layer: int,
        **kwargs: Any,
    ) -> dict:
        nlev = kwargs["vertical_end"]

        # We need to store the initial return field, because we only compute on a subdomain.
        contravariant_correction_at_cells_on_half_levels_ret = (
            contravariant_correction_at_cells_on_half_levels.copy()
        )
        contravariant_corrected_w_at_cells_on_model_levels_ret = (
            contravariant_corrected_w_at_cells_on_model_levels.copy()
        )
        vertical_wind_advective_tendency_ret = vertical_wind_advective_tendency.copy()
        vertical_cfl_ret = vertical_cfl.copy()

        contravariant_correction_at_cells_on_half_levels_nlev = (
            interpolate_contravariant_correction_to_cells_on_half_levels_numpy(
                connectivities,
                contravariant_correction_at_cells_on_half_levels[:, :-1],
                contravariant_correction_at_edges_on_model_levels,
                e_bln_c_s,
                wgtfac_c,
                nflatlev,
                nlev,
            )
        )

        (
            contravariant_corrected_w_at_cells_on_half_levels,
            cfl_clipping,
            vertical_cfl,
        ) = compute_maximum_cfl_and_clip_contravariant_vertical_velocity_numpy(
            w[:, :-1],
            contravariant_correction_at_cells_on_half_levels_nlev,
            ddqz_z_half,
            cfl_w_limit,
            dtime,
            nflatlev,
            nlev,
            end_index_of_damping_layer,
        )

        vertical_wind_advective_tendency = (
            compute_advective_vertical_wind_tendency_and_apply_diffusion_numpy(
                connectivities,
                vertical_wind_advective_tendency,
                w,
                tangential_wind_on_half_levels,
                vn_on_half_levels,
                contravariant_corrected_w_at_cells_on_half_levels,
                cfl_clipping,
                coeff1_dwdz,
                coeff2_dwdz,
                c_intp,
                inv_dual_edge_length,
                inv_primal_edge_length,
                tangent_orientation,
                e_bln_c_s,
                ddqz_z_half,
                area,
                geofac_n2s,
                owner_mask,
                scalfac_exdiff,
                cfl_w_limit,
                dtime,
                nlev,
                end_index_of_damping_layer,
            )
        )

        contravariant_corrected_w_at_cells_on_model_levels = (
            interpolate_contravariant_vertical_velocity_to_full_levels_numpy(
                contravariant_corrected_w_at_cells_on_half_levels
            )
        )

        # Apply the slicing.
        horizontal_start = kwargs["horizontal_start"]
        horizontal_end = kwargs["horizontal_end"]
        vertical_start = kwargs["vertical_start"]
        vertical_end = kwargs["vertical_end"]

        contravariant_correction_at_cells_on_half_levels_ret[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ] = contravariant_correction_at_cells_on_half_levels_nlev[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        contravariant_corrected_w_at_cells_on_model_levels_ret[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ] = contravariant_corrected_w_at_cells_on_model_levels[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        vertical_wind_advective_tendency_ret[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ] = vertical_wind_advective_tendency[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        vertical_cfl_ret[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ] = vertical_cfl[horizontal_start:horizontal_end, vertical_start:vertical_end]

        return dict(
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels_ret,
            vertical_wind_advective_tendency=vertical_wind_advective_tendency_ret,
            contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels_ret,
            vertical_cfl=vertical_cfl_ret,
        )

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        contravariant_corrected_w_at_cells_on_model_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim
        )
        vertical_wind_advective_tendency = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        w = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        tangential_wind_on_half_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}
        )
        vn_on_half_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}
        )
        contravariant_correction_at_edges_on_model_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        contravariant_correction_at_cells_on_half_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )

        coeff1_dwdz = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        coeff2_dwdz = data_alloc.random_field(grid, dims.CellDim, dims.KDim)

        c_intp = data_alloc.random_field(grid, dims.VertexDim, dims.V2CDim)
        inv_dual_edge_length = data_alloc.random_field(grid, dims.EdgeDim, low=1.0e-5)
        inv_primal_edge_length = data_alloc.random_field(grid, dims.EdgeDim, low=1.0e-5)
        tangent_orientation = data_alloc.random_field(grid, dims.EdgeDim, low=1.0e-5)
        e_bln_c_s = data_alloc.random_field(grid, dims.CEDim)
        wgtfac_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim)

        vertical_cfl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        owner_mask = data_alloc.random_mask(grid, dims.CellDim)
        ddqz_z_half = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        area = data_alloc.random_field(grid, dims.CellDim)
        geofac_n2s = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CODim)

        scalfac_exdiff = 10.0
        dtime = 2.0
        cfl_w_limit = 0.65 / dtime

        skip_compute_predictor_vertical_advection = False

        nflatlev = 3
        end_index_of_damping_layer = 5

        cell_domain = h_grid.domain(dims.CellDim)
        horizontal_start = grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
        horizontal_end = grid.end_index(cell_domain(h_grid.Zone.HALO))
        vertical_start = 0
        vertical_end = grid.num_levels

        return dict(
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            vertical_wind_advective_tendency=vertical_wind_advective_tendency,
            contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
            vertical_cfl=vertical_cfl,
            w=w,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn_on_half_levels=vn_on_half_levels,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            coeff1_dwdz=coeff1_dwdz,
            coeff2_dwdz=coeff2_dwdz,
            c_intp=c_intp,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
            e_bln_c_s=e_bln_c_s,
            wgtfac_c=wgtfac_c,
            ddqz_z_half=ddqz_z_half,
            area=area,
            geofac_n2s=geofac_n2s,
            owner_mask=owner_mask,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
            nflatlev=nflatlev,
            end_index_of_damping_layer=end_index_of_damping_layer,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
        )
