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
    compute_advection_in_vertical_momentum_equation,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import helpers as test_helpers

from .test_add_interpolated_horizontal_advection_of_w import (
    add_interpolated_horizontal_advection_of_w_numpy,
)
from .test_compute_advective_vertical_wind_tendency import (
    compute_advective_vertical_wind_tendency_numpy,
)
from .test_interpolate_contravariant_vertical_velocity_to_full_levels import (
    interpolate_contravariant_vertical_velocity_to_full_levels_numpy,
)


def _add_extra_diffusion_for_w_con_approaching_cfl_wihtout_levmask_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    cfl_clipping: np.ndarray,
    owner_mask: np.ndarray,
    z_w_con_c: np.ndarray,
    ddqz_z_half: np.ndarray,
    area: np.ndarray,
    geofac_n2s: np.ndarray,
    w: np.ndarray,
    ddt_w_adv: np.ndarray,
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.wpfloat,
    dtime: ta.wpfloat,
) -> np.ndarray:
    owner_mask = np.expand_dims(owner_mask, axis=-1)
    area = np.expand_dims(area, axis=-1)
    geofac_n2s = np.expand_dims(geofac_n2s, axis=-1)

    difcoef = np.where(
        (cfl_clipping == 1) & (owner_mask == 1),
        scalfac_exdiff
        * np.minimum(
            0.85 - cfl_w_limit * dtime,
            np.abs(z_w_con_c) * dtime / ddqz_z_half - cfl_w_limit * dtime,
        ),
        0,
    )

    c2e2cO = connectivities[dims.C2E2CODim]
    ddt_w_adv = np.where(
        (cfl_clipping == 1) & (owner_mask == 1),
        ddt_w_adv
        + difcoef
        * area
        * np.sum(
            np.where(
                (c2e2cO != -1)[:, :, np.newaxis],
                w[c2e2cO] * geofac_n2s,
                0,
            ),
            axis=1,
        ),
        ddt_w_adv,
    )
    return ddt_w_adv


def _compute_advective_vertical_wind_tendency_and_apply_diffusion(
    connectivities: dict[gtx.Dimension, np.ndarray],
    vertical_wind_advective_tendency: np.ndarray,
    w: np.ndarray,
    contravariant_corrected_w_at_cells_on_half_levels: np.ndarray,
    horizontal_advection_of_w_at_edges_on_half_levels: np.ndarray,
    coeff1_dwdz: np.ndarray,
    coeff2_dwdz: np.ndarray,
    e_bln_c_s: np.ndarray,
    ddqz_z_half: np.ndarray,
    area: np.ndarray,
    geofac_n2s: np.ndarray,
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.wpfloat,
    dtime: ta.wpfloat,
    cfl_clipping: np.ndarray,
    owner_mask: np.ndarray,
    nlev: int,
    end_index_of_damping_layer: int,
) -> np.ndarray:
    k = np.arange(nlev)

    condition1 = k >= 1
    vertical_wind_advective_tendency = np.where(
        condition1,
        compute_advective_vertical_wind_tendency_numpy(
            contravariant_corrected_w_at_cells_on_half_levels[:, :-1], w, coeff1_dwdz, coeff2_dwdz
        ),
        vertical_wind_advective_tendency,
    )

    vertical_wind_advective_tendency = np.where(
        condition1,
        add_interpolated_horizontal_advection_of_w_numpy(
            connectivities,
            e_bln_c_s,
            horizontal_advection_of_w_at_edges_on_half_levels,
            vertical_wind_advective_tendency,
        ),
        vertical_wind_advective_tendency,
    )

    condition2 = (np.maximum(3, end_index_of_damping_layer - 2) - 1 <= k) & (k < nlev - 3)

    vertical_wind_advective_tendency = np.where(
        condition2,
        _add_extra_diffusion_for_w_con_approaching_cfl_wihtout_levmask_numpy(
            connectivities,
            cfl_clipping,
            owner_mask,
            contravariant_corrected_w_at_cells_on_half_levels[:, :-1],
            ddqz_z_half,
            area,
            geofac_n2s,
            w[:, :-1],
            vertical_wind_advective_tendency,
            scalfac_exdiff,
            cfl_w_limit,
            dtime,
        ),
        vertical_wind_advective_tendency,
    )

    return vertical_wind_advective_tendency


class TestFusedVelocityAdvectionStencilVMomentum(test_helpers.StencilTest):
    PROGRAM = compute_advection_in_vertical_momentum_equation
    OUTPUTS = (
        "contravariant_corrected_w_at_cells_on_model_levels",
        "vertical_wind_advective_tendency",
    )
    MARKERS = (pytest.mark.embedded_remap_error, pytest.mark.skip_value_error)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        contravariant_corrected_w_at_cells_on_model_levels: np.ndarray,
        vertical_wind_advective_tendency: np.ndarray,
        w: np.ndarray,
        contravariant_corrected_w_at_cells_on_half_levels: np.ndarray,
        horizontal_advection_of_w_at_edges_on_half_levels: np.ndarray,
        coeff1_dwdz: np.ndarray,
        coeff2_dwdz: np.ndarray,
        e_bln_c_s: np.ndarray,
        ddqz_z_half: np.ndarray,
        area: np.ndarray,
        geofac_n2s: np.ndarray,
        scalfac_exdiff: ta.wpfloat,
        cfl_w_limit: ta.wpfloat,
        dtime: ta.wpfloat,
        skip_compute_predictor_vertical_advection: np.ndarray,
        cfl_clipping: np.ndarray,
        owner_mask: np.ndarray,
        nlev: int,
        end_index_of_damping_layer: int,
        **kwargs: Any,
    ) -> dict:
        # We need to store the initial return field, because we only compute on a subdomain.
        contravariant_corrected_w_at_cells_on_model_levels_ret = (
            contravariant_corrected_w_at_cells_on_model_levels.copy()
        )
        vertical_wind_advective_tendency_ret = vertical_wind_advective_tendency.copy()

        contravariant_corrected_w_at_cells_on_model_levels = (
            interpolate_contravariant_vertical_velocity_to_full_levels_numpy(
                contravariant_corrected_w_at_cells_on_half_levels
            )
        )

        if not skip_compute_predictor_vertical_advection:
            vertical_wind_advective_tendency = (
                _compute_advective_vertical_wind_tendency_and_apply_diffusion(
                    connectivities,
                    vertical_wind_advective_tendency,
                    w,
                    contravariant_corrected_w_at_cells_on_half_levels,
                    horizontal_advection_of_w_at_edges_on_half_levels,
                    coeff1_dwdz,
                    coeff2_dwdz,
                    e_bln_c_s,
                    ddqz_z_half,
                    area,
                    geofac_n2s,
                    scalfac_exdiff,
                    cfl_w_limit,
                    dtime,
                    cfl_clipping,
                    owner_mask,
                    nlev,
                    end_index_of_damping_layer,
                )
            )

        # Apply the slicing.
        horizontal_start = kwargs["horizontal_start"]
        horizontal_end = kwargs["horizontal_end"]
        vertical_start = kwargs["vertical_start"]
        vertical_end = kwargs["vertical_end"]

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

        return dict(
            contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels_ret,
            vertical_wind_advective_tendency=vertical_wind_advective_tendency,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        contravariant_corrected_w_at_cells_on_model_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim
        )
        vertical_wind_advective_tendency = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        w = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        contravariant_corrected_w_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        horizontal_advection_of_w_at_edges_on_half_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        coeff1_dwdz = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        coeff2_dwdz = data_alloc.random_field(grid, dims.CellDim, dims.KDim)

        e_bln_c_s = data_alloc.random_field(grid, dims.CEDim)

        cfl_clipping = data_alloc.random_mask(grid, dims.CellDim, dims.KDim)
        owner_mask = data_alloc.random_mask(grid, dims.CellDim)
        ddqz_z_half = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        area = data_alloc.random_field(grid, dims.CellDim)
        geofac_n2s = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CODim)

        scalfac_exdiff = 10.0
        cfl_w_limit = 3.0
        dtime = 2.0

        nlev = grid.num_levels
        end_index_of_damping_layer = 5
        skip_compute_predictor_vertical_advection = False

        cell_domain = h_grid.domain(dims.CellDim)
        horizontal_start = grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
        horizontal_end = grid.end_index(cell_domain(h_grid.Zone.HALO))
        vertical_start = 0
        vertical_end = nlev

        return dict(
            contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
            vertical_wind_advective_tendency=vertical_wind_advective_tendency,
            w=w,
            contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
            horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
            coeff1_dwdz=coeff1_dwdz,
            coeff2_dwdz=coeff2_dwdz,
            e_bln_c_s=e_bln_c_s,
            ddqz_z_half=ddqz_z_half,
            area=area,
            geofac_n2s=geofac_n2s,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
            cfl_clipping=cfl_clipping,
            owner_mask=owner_mask,
            nlev=nlev,
            end_index_of_damping_layer=end_index_of_damping_layer,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
        )
