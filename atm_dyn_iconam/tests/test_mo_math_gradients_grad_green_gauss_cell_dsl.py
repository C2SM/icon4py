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

from icon4py.atm_dyn_iconam.mo_math_gradients_grad_green_gauss_cell_dsl import (
    mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_1_u,
    mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_1_v,
    mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_2_u,
    mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_2_v,
)
from icon4py.common.dimension import C2E2CODim, CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_1_u_numpy(
    c2ec2o: np.array, p_ccpr1: np.array, geofac_grg_x
):
    geofac_grg_x = np.expand_dims(geofac_grg_x, axis=-1)
    p_grad_1_u = np.sum(geofac_grg_x * p_ccpr1[c2ec2o], axis=1)
    return p_grad_1_u


def mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_1_v_numpy(
    c2ec2o: np.array, p_ccpr1: np.array, geofac_grg_y
):
    geofac_grg_y = np.expand_dims(geofac_grg_y, axis=-1)
    p_grad_1_v = np.sum(geofac_grg_y * p_ccpr1[c2ec2o], axis=1)
    return p_grad_1_v


def mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_2_u_numpy(
    c2ec2o: np.array, p_ccpr2: np.array, geofac_grg_x
):
    geofac_grg_x = np.expand_dims(geofac_grg_x, axis=-1)
    p_grad_2_u = np.sum(geofac_grg_x * p_ccpr2[c2ec2o], axis=1)
    return p_grad_2_u


def mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_2_v_numpy(
    c2ec2o: np.array, p_ccpr2: np.array, geofac_grg_y
):
    geofac_grg_y = np.expand_dims(geofac_grg_y, axis=-1)
    p_grad_2_v = np.sum(geofac_grg_y * p_ccpr2[c2ec2o], axis=1)
    return p_grad_2_v


def test_mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_2_v_numpy():
    mesh = SimpleMesh()

    p_ccpr = random_field(mesh, CellDim, KDim)
    geofac_grg = random_field(mesh, CellDim, C2E2CODim)
    out = zero_field(mesh, CellDim, KDim)

    stencil_funcs = {
        mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_1_u_numpy: mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_1_u,
        mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_1_v_numpy: mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_1_v,
        mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_2_u_numpy: mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_2_u,
        mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_2_v_numpy: mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_2_v,
    }

    for ref_func, func in stencil_funcs.items():
        ref = ref_func(mesh.c2e2cO, np.asarray(p_ccpr), np.asarray(geofac_grg))
        func(
            p_ccpr,
            geofac_grg,
            out,
            offset_provider={"C2E2CO": mesh.get_c2e2cO_offset_provider()},
        )
        assert np.allclose(out, ref)
