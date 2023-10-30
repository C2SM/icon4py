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
from gt4py.next.program_processors.runners.gtfn import run_gtfn

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_52 import (
    mo_solve_nonhydro_stencil_52,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field


def mo_solve_nonhydro_stencil_52_numpy(
    vwind_impl_wgt: np.array,
    theta_v_ic: np.array,
    ddqz_z_half: np.array,
    z_alpha: np.array,
    z_beta: np.array,
    z_exner_expl: np.array,
    z_w_expl: np.array,
    z_q_ref: np.array,
    w_ref: np.array,
    dtime,
    cpd,
) -> tuple[np.array]:
    z_q = np.copy(z_q_ref)
    w = np.copy(w_ref)
    vwind_impl_wgt = np.expand_dims(vwind_impl_wgt, axis=-1)

    z_gamma = dtime * cpd * vwind_impl_wgt * theta_v_ic / ddqz_z_half
    z_a = np.zeros_like(z_gamma)
    z_b = np.zeros_like(z_gamma)
    z_c = np.zeros_like(z_gamma)
    z_g = np.zeros_like(z_gamma)

    k_size = w.shape[1]
    for k in range(1, k_size):
        z_a[:, k] = -z_gamma[:, k] * z_beta[:, k - 1] * z_alpha[:, k - 1]
        z_c[:, k] = -z_gamma[:, k] * z_beta[:, k] * z_alpha[:, k + 1]
        z_b[:, k] = 1.0 + z_gamma[:, k] * z_alpha[:, k] * (z_beta[:, k - 1] + z_beta[:, k])
        z_g[:, k] = 1.0 / (z_b[:, k] + z_a[:, k] * z_q[:, k - 1])
        z_q[:, k] = -z_c[:, k] * z_g[:, k]

        w[:, k] = z_w_expl[:, k] - z_gamma[:, k] * (z_exner_expl[:, k - 1] - z_exner_expl[:, k])
        w[:, k] = (w[:, k] - z_a[:, k] * w[:, k - 1]) * z_g[:, k]
    return z_q, w


def test_mo_solve_nonhydro_stencil_52():
    grid = SimpleGrid()
    vwind_impl_wgt = random_field(grid, CellDim)
    theta_v_ic = random_field(grid, CellDim, KDim)
    ddqz_z_half = random_field(grid, CellDim, KDim)
    z_alpha = random_field(grid, CellDim, KDim, extend={KDim: 1})
    z_beta = random_field(grid, CellDim, KDim)
    z_exner_expl = random_field(grid, CellDim, KDim)
    z_w_expl = random_field(grid, CellDim, KDim, extend={KDim: 1})
    dtime = 8.0
    cpd = 7.0

    z_q = random_field(grid, CellDim, KDim)
    w = random_field(grid, CellDim, KDim)

    z_q_ref, w_ref = mo_solve_nonhydro_stencil_52_numpy(
        np.asarray(vwind_impl_wgt),
        np.asarray(theta_v_ic),
        np.asarray(ddqz_z_half),
        np.asarray(z_alpha),
        np.asarray(z_beta),
        np.asarray(z_exner_expl),
        np.asarray(z_w_expl),
        np.asarray(z_q),
        np.asarray(w),
        dtime,
        cpd,
    )
    h_start = int32(0)
    h_end = int32(grid.num_cells)
    v_start = int32(1)
    v_end = int32(grid.num_levels)
    # TODO we run this test with the C++ backend as the `embedded` backend doesn't handle this pattern
    mo_solve_nonhydro_stencil_52.with_backend(run_gtfn)(
        vwind_impl_wgt=vwind_impl_wgt,
        theta_v_ic=theta_v_ic,
        ddqz_z_half=ddqz_z_half,
        z_alpha=z_alpha,
        z_beta=z_beta,
        z_w_expl=z_w_expl,
        z_exner_expl=z_exner_expl,
        z_q=z_q,
        w=w,
        dtime=dtime,
        cpd=cpd,
        horizontal_start=h_start,
        horizontal_end=h_end,
        vertical_start=v_start,
        vertical_end=v_end,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(z_q_ref[h_start:h_end, v_start:v_end], z_q[h_start:h_end, v_start:v_end])
    assert np.allclose(w_ref[h_start:h_end, v_start:v_end], w[h_start:h_end, v_start:v_end])
