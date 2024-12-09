# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.common.settings import xp


def compute_flat_idx_max(
    e2c: xp.ndarray,
    z_mc: xp.ndarray,
    c_lin_e: xp.ndarray,
    z_ifc: xp.ndarray,
    k_lev: xp.ndarray,
    horizontal_lower: int,
    horizontal_upper: int,
) -> xp.ndarray:
    z_me = xp.sum(z_mc[e2c] * xp.expand_dims(c_lin_e, axis=-1), axis=1)
    z_ifc_e_0 = z_ifc[e2c[:, 0]]
    z_ifc_e_k_0 = xp.roll(z_ifc_e_0, -1, axis=1)
    z_ifc_e_1 = z_ifc[e2c[:, 1]]
    z_ifc_e_k_1 = xp.roll(z_ifc_e_1, -1, axis=1)
    flat_idx = xp.zeros_like(z_me)
    for je in range(horizontal_lower, horizontal_upper):
        for jk in range(k_lev.shape[0] - 1):
            if (
                (z_me[je, jk] <= z_ifc_e_0[je, jk])
                and (z_me[je, jk] >= z_ifc_e_k_0[je, jk])
                and (z_me[je, jk] <= z_ifc_e_1[je, jk])
                and (z_me[je, jk] >= z_ifc_e_k_1[je, jk])
            ):
                flat_idx[je, jk] = k_lev[jk]
    flat_idx_max = xp.amax(flat_idx, axis=1)
    return flat_idx_max.astype(xp.int32)
