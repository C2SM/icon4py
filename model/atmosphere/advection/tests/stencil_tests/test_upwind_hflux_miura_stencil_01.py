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
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.program_processors.runners import roundtrip

from icon4py.model.atmosphere.advection.upwind_hflux_miura_stencil_01 import (
    upwind_hflux_miura_stencil_01,
)
from icon4py.model.common.dimension import CellDim, E2CDim, EdgeDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import constant_field, random_field, zero_field


def upwind_hflux_miura_stencil_01_numpy(
    e2c: np.array,
    z_lsq_coeff_1: np.array,
    z_lsq_coeff_2: np.array,
    z_lsq_coeff_3: np.array,
    distv_bary_1: np.array,
    distv_bary_2: np.array,
    p_mass_flx_e: np.array,
    cell_rel_idx_dsl: np.array,
) -> np.array:

    z_lsq_coeff_1_e2c = z_lsq_coeff_1[e2c]
    z_lsq_coeff_2_e2c = z_lsq_coeff_2[e2c]
    z_lsq_coeff_3_e2c = z_lsq_coeff_3[e2c]

    p_out_e = (
        np.where(
            cell_rel_idx_dsl == int32(1),
            z_lsq_coeff_1_e2c[:, 1],
            z_lsq_coeff_1_e2c[:, 0],
        )
        + distv_bary_1
        * np.where(
            cell_rel_idx_dsl == int32(1),
            z_lsq_coeff_2_e2c[:, 1],
            z_lsq_coeff_2_e2c[:, 0],
        )
        + distv_bary_2
        * np.where(
            cell_rel_idx_dsl == int32(1),
            z_lsq_coeff_3_e2c[:, 1],
            z_lsq_coeff_3_e2c[:, 0],
        )
    ) * p_mass_flx_e

    return p_out_e


def test_upwind_hflux_miura_stencil_01():
    grid = SimpleGrid()

    z_lsq_coeff_1 = random_field(grid, CellDim, KDim)
    z_lsq_coeff_2 = random_field(grid, CellDim, KDim)
    z_lsq_coeff_3 = random_field(grid, CellDim, KDim)
    distv_bary_1 = random_field(grid, EdgeDim, KDim)
    distv_bary_2 = random_field(grid, EdgeDim, KDim)
    p_mass_flx_e = random_field(grid, EdgeDim, KDim)
    cell_rel_idx_dsl = constant_field(grid, 0, EdgeDim, KDim, dtype=int32)
    p_out_e = zero_field(grid, EdgeDim, KDim)

    ref = upwind_hflux_miura_stencil_01_numpy(
        grid.connectivities[E2CDim],
        np.asarray(z_lsq_coeff_1),
        np.asarray(z_lsq_coeff_2),
        np.asarray(z_lsq_coeff_3),
        np.asarray(distv_bary_1),
        np.asarray(distv_bary_2),
        np.asarray(p_mass_flx_e),
        np.asarray(cell_rel_idx_dsl),
    )

    upwind_hflux_miura_stencil_01.with_backend(roundtrip.backend)(
        z_lsq_coeff_1,
        z_lsq_coeff_2,
        z_lsq_coeff_3,
        distv_bary_1,
        distv_bary_2,
        p_mass_flx_e,
        cell_rel_idx_dsl,
        p_out_e,
        offset_provider={
            "E2C": grid.get_offset_provider("E2C"),
        },
    )
    assert np.allclose(p_out_e, ref)
