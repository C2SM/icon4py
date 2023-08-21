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

from icon4py.model.atmosphere.advection.upwind_hflux_miura_cycl_stencil_01 import (
    upwind_hflux_miura_cycl_stencil_01,
)
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    _shape,
    random_field,
    random_mask,
    zero_field,
)
from icon4py.model.common.test_utils.simple_mesh import SimpleMesh


def upwind_hflux_miura_cycl_stencil_01_numpy(
    e2c: np.array,
    z_lsq_coeff_1_dsl: np.array,
    z_lsq_coeff_2_dsl: np.array,
    z_lsq_coeff_3_dsl: np.array,
    distv_bary_1: np.array,
    distv_bary_2: np.array,
    p_mass_flx_e: np.array,
    cell_rel_idx_dsl: int32,
):
    z_lsq_coeff_1_dsl_e2c = z_lsq_coeff_1_dsl[e2c]
    z_lsq_coeff_2_dsl_e2c = z_lsq_coeff_2_dsl[e2c]
    z_lsq_coeff_3_dsl_e2c = z_lsq_coeff_3_dsl[e2c]

    z_tracer_mflx_dsl = (
        np.where(
            cell_rel_idx_dsl == int32(1),
            z_lsq_coeff_1_dsl_e2c[:, 1],
            z_lsq_coeff_1_dsl_e2c[:, 0],
        )
        + distv_bary_1
        * np.where(
            cell_rel_idx_dsl == int32(1),
            z_lsq_coeff_2_dsl_e2c[:, 1],
            z_lsq_coeff_2_dsl_e2c[:, 0],
        )
        + distv_bary_2
        * np.where(
            cell_rel_idx_dsl == int32(1),
            z_lsq_coeff_3_dsl_e2c[:, 1],
            z_lsq_coeff_3_dsl_e2c[:, 0],
        )
    ) * p_mass_flx_e

    return z_tracer_mflx_dsl


def test_upwind_hflux_miura_cycl_stencil_01():
    mesh = SimpleMesh()

    z_lsq_coeff_1_dsl = random_field(mesh, CellDim, KDim)
    z_lsq_coeff_2_dsl = random_field(mesh, CellDim, KDim)
    z_lsq_coeff_3_dsl = random_field(mesh, CellDim, KDim)
    distv_bary_1 = random_field(mesh, EdgeDim, KDim)
    distv_bary_2 = random_field(mesh, EdgeDim, KDim)
    p_mass_flx_e = random_field(mesh, EdgeDim, KDim)
    cell_rel_idx_dsl = random_mask(mesh, EdgeDim, KDim, dtype=int32)
    z_tracer_mflx_dsl = zero_field(mesh, EdgeDim, KDim)

    ref = upwind_hflux_miura_cycl_stencil_01_numpy(
        mesh.e2c,
        np.asarray(z_lsq_coeff_1_dsl),
        np.asarray(z_lsq_coeff_2_dsl),
        np.asarray(z_lsq_coeff_3_dsl),
        np.asarray(distv_bary_1),
        np.asarray(distv_bary_2),
        np.asarray(p_mass_flx_e),
        np.asarray(cell_rel_idx_dsl),
    )

    upwind_hflux_miura_cycl_stencil_01(
        z_lsq_coeff_1_dsl,
        z_lsq_coeff_2_dsl,
        z_lsq_coeff_3_dsl,
        distv_bary_1,
        distv_bary_2,
        p_mass_flx_e,
        cell_rel_idx_dsl,
        z_tracer_mflx_dsl,
        offset_provider={
            "E2C": mesh.get_e2c_offset_provider(),
        },
    )
    assert np.allclose(ref, z_tracer_mflx_dsl)
