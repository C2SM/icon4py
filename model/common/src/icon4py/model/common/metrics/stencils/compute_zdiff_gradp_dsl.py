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
from icon4py.model.common.dimension import E2CDim
import numpy as np

def compute_zdiff_gradp_dsl(
    e2c,
    c_lin_e: np.array,
    z_mc: np.array,
    zdiff_gradp: np.array,
    z_ifc: np.array,
    flat_idx: np.array,
    nlev: int,
    nedges: int
) -> np.array:
    c_lin_e = np.expand_dims(c_lin_e, axis=-1)
    z_me = np.sum(z_mc[e2c] * c_lin_e, axis=1)
    zdiff_gradp[:, 0, :] = z_me - z_mc[e2c[:, 0]]
    zdiff_gradp[:, 1, :] = z_me - z_mc[e2c[:, 1]]
    extrapol_dist = 5.0
    nlevp1 = nlev + 1
    z_aux1 = np.maximum(z_ifc[e2c[:, 0], nlevp1], z_ifc[e2c[:, 1], nlevp1])
    z_aux2 = z_aux1 - extrapol_dist
    jk_start = flat_idx
    for je in range(nedges):
        for jk in range(flat_idx[je] + 1, nlev):
            if z_me[je, jk] < z_aux2[je]:
                for jk1 in (jk_start, nlev):
                    if jk1 == nlev or z_aux2 <= z_ifc[e2c[je, 1], jk1] and z_aux2 >= z_ifc[e2c[je, 1], jk1+1]:
                        zdiff_gradp[je, 0, jk1] = z_aux2 - z_mc[e2c[je, 0], jk1]
                        zdiff_gradp[je, 1, jk1] = z_aux2 - z_mc[e2c[je, 1], jk1]
                        jk_start = jk1
                        break

    return zdiff_gradp
