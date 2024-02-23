# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.fused_velocity_advection_stencil_8_to_13 import (
    fused_velocity_advection_stencil_8_to_13,
)
from icon4py.model.atmosphere.dycore.state_utils.utils import indices_field
from icon4py.model.common.dimension import C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    random_field,
    zero_field,
)

from .test_copy_cell_kdim_field_to_vp import copy_cell_kdim_field_to_vp_numpy
from .test_correct_contravariant_vertical_velocity import (
    correct_contravariant_vertical_velocity_numpy,
)
from .test_interpolate_to_cell_center import interpolate_to_cell_center_numpy
from .test_interpolate_to_half_levels_vp import interpolate_to_half_levels_vp_numpy
from .test_set_cell_kdim_field_to_zero_vp import set_cell_kdim_field_to_zero_vp_numpy


class TestFusedVelocityAdvectionStencil8To13(StencilTest):
    PROGRAM = fused_velocity_advection_stencil_8_to_13
    OUTPUTS = (
        "z_ekinh",
        "w_concorr_c",
        "z_w_con_c",
    )

    @staticmethod
    def reference(
        grid,
        z_kin_hor_e,
        e_bln_c_s,
        z_w_concorr_me,
        wgtfac_c,
        w,
        z_w_concorr_mc,
        w_concorr_c,
        z_ekinh,
        k,
        istep,
        nlev,
        nflatlev,
        z_w_con_c,
        **kwargs,
    ):
        k_nlev = k[:-1]

        z_ekinh = np.where(
            k_nlev < nlev,
            interpolate_to_cell_center_numpy(grid, z_kin_hor_e, e_bln_c_s),
            z_ekinh,
        )

        if istep == 1:
            z_w_concorr_mc = np.where(
                (nflatlev <= k_nlev) & (k_nlev < nlev),
                interpolate_to_cell_center_numpy(grid, z_w_concorr_me, e_bln_c_s),
                z_w_concorr_mc,
            )

            w_concorr_c = np.where(
                (nflatlev + 1 <= k_nlev) & (k_nlev < nlev),
                interpolate_to_half_levels_vp_numpy(grid, z_w_concorr_mc, wgtfac_c),
                w_concorr_c,
            )

        z_w_con_c = np.where(
            k < nlev,
            copy_cell_kdim_field_to_vp_numpy(w),
            set_cell_kdim_field_to_zero_vp_numpy(z_w_con_c),
        )

        z_w_con_c[:, :-1] = np.where(
            (nflatlev + 1 <= k_nlev) & (k_nlev < nlev),
            correct_contravariant_vertical_velocity_numpy(z_w_con_c[:, :-1], w_concorr_c),
            z_w_con_c[:, :-1],
        )

        return dict(
            z_ekinh=z_ekinh,
            w_concorr_c=w_concorr_c,
            z_w_con_c=z_w_con_c,
        )

    @pytest.fixture
    def input_data(self, grid):
        z_kin_hor_e = random_field(grid, EdgeDim, KDim)
        e_bln_c_s = random_field(grid, CellDim, C2EDim)
        z_ekinh = zero_field(grid, CellDim, KDim)
        z_w_concorr_me = random_field(grid, EdgeDim, KDim)
        z_w_concorr_mc = zero_field(grid, CellDim, KDim)
        wgtfac_c = random_field(grid, CellDim, KDim)
        w_concorr_c = zero_field(grid, CellDim, KDim)
        w = random_field(grid, CellDim, KDim, extend={KDim: 1})
        z_w_con_c = zero_field(grid, CellDim, KDim, extend={KDim: 1})

        k = indices_field(KDim, grid, is_halfdim=True, dtype=int32)

        nlev = grid.num_levels
        nflatlev = 4

        istep = 1

        horizontal_start = 0
        horizontal_end = grid.num_cells
        vertical_start = 0
        vertical_end = nlev + 1

        return dict(
            z_kin_hor_e=z_kin_hor_e,
            e_bln_c_s=as_1D_sparse_field(e_bln_c_s, CEDim),
            z_w_concorr_me=z_w_concorr_me,
            wgtfac_c=wgtfac_c,
            w=w,
            z_w_concorr_mc=z_w_concorr_mc,
            w_concorr_c=w_concorr_c,
            z_ekinh=z_ekinh,
            z_w_con_c=z_w_con_c,
            k=k,
            istep=istep,
            nlev=nlev,
            nflatlev=nflatlev,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
        )
