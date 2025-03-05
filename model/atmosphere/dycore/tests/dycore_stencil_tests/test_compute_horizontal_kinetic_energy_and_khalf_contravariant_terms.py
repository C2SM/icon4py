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

from icon4py.model.atmosphere.dycore.stencils.compute_cell_diagnostics_for_velocity_advection import (
    compute_horizontal_kinetic_energy_and_khalf_contravariant_terms,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.helpers import StencilTest

from .test_copy_cell_kdim_field_to_vp import copy_cell_kdim_field_to_vp_numpy
from .test_correct_contravariant_vertical_velocity import (
    correct_contravariant_vertical_velocity_numpy,
)
from .test_init_cell_kdim_field_with_zero_vp import init_cell_kdim_field_with_zero_vp_numpy
from .test_interpolate_to_cell_center import interpolate_to_cell_center_numpy
from .test_interpolate_to_half_levels_vp import interpolate_to_half_levels_vp_numpy


class TestComputeHorizontalKineticEnergyAndKhalfContravariantTerms(StencilTest):
    PROGRAM = compute_horizontal_kinetic_energy_and_khalf_contravariant_terms
    OUTPUTS = (
        "horizontal_kinetic_energy_at_cell",
        "khalf_contravariant_correction_at_cell",
        "khalf_contravariant_corrected_w_at_cell",
    )
    MARKERS = (pytest.mark.requires_concat_where,)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        horizontal_kinetic_energy_at_cell: np.ndarray,
        khalf_contravariant_correction_at_cell: np.ndarray,
        khalf_contravariant_corrected_w_at_cell: np.ndarray,
        w: np.ndarray,
        horizontal_kinetic_energy_at_edge: np.ndarray,
        contravariant_correction_at_edge: np.ndarray,
        e_bln_c_s: np.ndarray,
        wgtfac_c: np.ndarray,
        k: np.ndarray,
        nflatlev: ta.wpfloat,
        nlev: ta.wpfloat,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
    ) -> dict:
        k_nlev = k[:-1]

        horizontal_kinetic_energy_at_cell_cp = horizontal_kinetic_energy_at_cell.copy()
        khalf_contravariant_correction_at_cell_cp = khalf_contravariant_correction_at_cell.copy()
        khalf_contravariant_corrected_w_at_cell_cp = khalf_contravariant_corrected_w_at_cell.copy()

        horizontal_kinetic_energy_at_cell = np.where(
            k_nlev < nlev,
            interpolate_to_cell_center_numpy(
                connectivities, horizontal_kinetic_energy_at_edge, e_bln_c_s
            ),
            horizontal_kinetic_energy_at_cell,
        )

        contravariant_correction_at_cell = np.where(
            k_nlev >= nflatlev,
            interpolate_to_cell_center_numpy(
                connectivities, contravariant_correction_at_edge, e_bln_c_s
            ),
            0.0,
        )

        khalf_contravariant_correction_at_cell = np.where(
            (nflatlev + 1 <= k_nlev) & (k_nlev < nlev),
            interpolate_to_half_levels_vp_numpy(
                wgtfac_c=wgtfac_c, interpolant=contravariant_correction_at_cell
            ),
            khalf_contravariant_correction_at_cell,
        )

        khalf_contravariant_corrected_w_at_cell = np.where(
            k < nlev,
            copy_cell_kdim_field_to_vp_numpy(w),
            init_cell_kdim_field_with_zero_vp_numpy(khalf_contravariant_corrected_w_at_cell),
        )

        khalf_contravariant_corrected_w_at_cell[:, :-1] = np.where(
            (nflatlev + 1 <= k_nlev) & (k_nlev < nlev),
            correct_contravariant_vertical_velocity_numpy(
                khalf_contravariant_corrected_w_at_cell[:, :-1],
                khalf_contravariant_correction_at_cell,
            ),
            khalf_contravariant_corrected_w_at_cell[:, :-1],
        )
        horizontal_kinetic_energy_at_cell_cp[
            horizontal_start:horizontal_end, :
        ] = horizontal_kinetic_energy_at_cell[horizontal_start:horizontal_end, :]
        khalf_contravariant_correction_at_cell_cp[
            horizontal_start:horizontal_end, :
        ] = khalf_contravariant_correction_at_cell[horizontal_start:horizontal_end, :]
        khalf_contravariant_corrected_w_at_cell_cp[
            horizontal_start:horizontal_end, :
        ] = khalf_contravariant_corrected_w_at_cell[horizontal_start:horizontal_end, :]

        return dict(
            horizontal_kinetic_energy_at_cell=horizontal_kinetic_energy_at_cell_cp,
            khalf_contravariant_correction_at_cell=khalf_contravariant_correction_at_cell_cp,
            khalf_contravariant_corrected_w_at_cell=khalf_contravariant_corrected_w_at_cell_cp,
        )

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid):
        horizontal_kinetic_energy_at_cell = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        khalf_contravariant_correction_at_cell = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim
        )
        khalf_contravariant_corrected_w_at_cell = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        w = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        horizontal_kinetic_energy_at_edge = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        contravariant_correction_at_edge = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        e_bln_c_s = data_alloc.random_field(grid, dims.CEDim)
        wgtfac_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim)

        k = data_alloc.index_field(dim=dims.KDim, grid=grid, extend={dims.KDim: 1})

        nlev = grid.num_levels
        nflatlev = 4

        cell_domain = h_grid.domain(dims.CellDim)
        horizontal_start = grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
        horizontal_end = grid.end_index(cell_domain(h_grid.Zone.HALO))
        vertical_start = 0
        vertical_end = nlev + 1

        return dict(
            horizontal_kinetic_energy_at_cell=horizontal_kinetic_energy_at_cell,
            khalf_contravariant_correction_at_cell=khalf_contravariant_correction_at_cell,
            khalf_contravariant_corrected_w_at_cell=khalf_contravariant_corrected_w_at_cell,
            w=w,
            horizontal_kinetic_energy_at_edge=horizontal_kinetic_energy_at_edge,
            contravariant_correction_at_edge=contravariant_correction_at_edge,
            e_bln_c_s=e_bln_c_s,
            wgtfac_c=wgtfac_c,
            k=k,
            nflatlev=nflatlev,
            nlev=nlev,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
        )
