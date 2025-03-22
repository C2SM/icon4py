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

import icon4py.model.testing.helpers as test_helpers
from icon4py.model.atmosphere.dycore.stencils.compute_cell_diagnostics_for_velocity_advection import (
    interpolate_horizontal_kinetic_energy_to_cells_and_compute_contravariant_terms,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc

from .test_copy_cell_kdim_field_to_vp import copy_cell_kdim_field_to_vp_numpy
from .test_correct_contravariant_vertical_velocity import (
    correct_contravariant_vertical_velocity_numpy,
)
from .test_init_cell_kdim_field_with_zero_vp import init_cell_kdim_field_with_zero_vp_numpy
from .test_interpolate_cell_field_to_half_levels_vp import (
    interpolate_cell_field_to_half_levels_vp_numpy,
)
from .test_interpolate_to_cell_center import interpolate_to_cell_center_numpy


class TestInterpolateHorizontalKineticWnergyToCellsAndComputeContravariantTerms(
    test_helpers.StencilTest
):
    PROGRAM = interpolate_horizontal_kinetic_energy_to_cells_and_compute_contravariant_terms
    OUTPUTS = (
        "horizontal_kinetic_energy_at_cells_on_model_levels",
        "contravariant_correction_at_cells_on_half_levels",
        "contravariant_corrected_w_at_cells_on_half_levels",
    )
    MARKERS = (pytest.mark.requires_concat_where,)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        horizontal_kinetic_energy_at_cells_on_model_levels: np.ndarray,
        contravariant_correction_at_cells_on_half_levels: np.ndarray,
        contravariant_corrected_w_at_cells_on_half_levels: np.ndarray,
        w: np.ndarray,
        horizontal_kinetic_energy_at_edges_on_model_levels: np.ndarray,
        contravariant_correction_at_edges_on_model_levels: np.ndarray,
        e_bln_c_s: np.ndarray,
        wgtfac_c: np.ndarray,
        nflatlev: ta.wpfloat,
        nlev: ta.wpfloat,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
    ) -> dict:
        k = np.arange(0, nlev + 1)
        k_nlev = k[:-1]

        horizontal_kinetic_energy_at_cells_on_model_levels_cp = (
            horizontal_kinetic_energy_at_cells_on_model_levels.copy()
        )
        contravariant_correction_at_cells_on_half_levels_cp = (
            contravariant_correction_at_cells_on_half_levels.copy()
        )
        contravariant_corrected_w_at_cells_on_half_levels_cp = (
            contravariant_corrected_w_at_cells_on_half_levels.copy()
        )

        horizontal_kinetic_energy_at_cells_on_model_levels = np.where(
            k_nlev < nlev,
            interpolate_to_cell_center_numpy(
                connectivities, horizontal_kinetic_energy_at_edges_on_model_levels, e_bln_c_s
            ),
            horizontal_kinetic_energy_at_cells_on_model_levels,
        )

        contravariant_correction_at_cell = np.where(
            k_nlev >= nflatlev,
            interpolate_to_cell_center_numpy(
                connectivities, contravariant_correction_at_edges_on_model_levels, e_bln_c_s
            ),
            0.0,
        )

        contravariant_correction_at_cells_on_half_levels = np.where(
            (nflatlev + 1 <= k_nlev) & (k_nlev < nlev),
            interpolate_cell_field_to_half_levels_vp_numpy(
                wgtfac_c=wgtfac_c, interpolant=contravariant_correction_at_cell
            ),
            contravariant_correction_at_cells_on_half_levels,
        )

        contravariant_corrected_w_at_cells_on_half_levels = np.where(
            k < nlev,
            copy_cell_kdim_field_to_vp_numpy(w),
            init_cell_kdim_field_with_zero_vp_numpy(
                contravariant_corrected_w_at_cells_on_half_levels
            ),
        )

        contravariant_corrected_w_at_cells_on_half_levels[:, :-1] = np.where(
            (nflatlev + 1 <= k_nlev) & (k_nlev < nlev),
            correct_contravariant_vertical_velocity_numpy(
                contravariant_corrected_w_at_cells_on_half_levels[:, :-1],
                contravariant_correction_at_cells_on_half_levels,
            ),
            contravariant_corrected_w_at_cells_on_half_levels[:, :-1],
        )
        horizontal_kinetic_energy_at_cells_on_model_levels_cp[
            horizontal_start:horizontal_end, :
        ] = horizontal_kinetic_energy_at_cells_on_model_levels[horizontal_start:horizontal_end, :]
        contravariant_correction_at_cells_on_half_levels_cp[
            horizontal_start:horizontal_end, :
        ] = contravariant_correction_at_cells_on_half_levels[horizontal_start:horizontal_end, :]
        contravariant_corrected_w_at_cells_on_half_levels_cp[
            horizontal_start:horizontal_end, :
        ] = contravariant_corrected_w_at_cells_on_half_levels[horizontal_start:horizontal_end, :]

        return dict(
            horizontal_kinetic_energy_at_cells_on_model_levels=horizontal_kinetic_energy_at_cells_on_model_levels_cp,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels_cp,
            contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels_cp,
        )

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        horizontal_kinetic_energy_at_cells_on_model_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim
        )
        contravariant_correction_at_cells_on_half_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim
        )
        contravariant_corrected_w_at_cells_on_half_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        w = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        horizontal_kinetic_energy_at_edges_on_model_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        contravariant_correction_at_edges_on_model_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        e_bln_c_s = data_alloc.random_field(grid, dims.CEDim)
        wgtfac_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        nlev = grid.num_levels
        nflatlev = 4

        cell_domain = h_grid.domain(dims.CellDim)
        horizontal_start = grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
        horizontal_end = grid.end_index(cell_domain(h_grid.Zone.HALO))
        vertical_start = 0
        vertical_end = nlev + 1

        return dict(
            horizontal_kinetic_energy_at_cells_on_model_levels=horizontal_kinetic_energy_at_cells_on_model_levels,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
            w=w,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            e_bln_c_s=e_bln_c_s,
            wgtfac_c=wgtfac_c,
            nflatlev=nflatlev,
            nlev=nlev,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
        )
