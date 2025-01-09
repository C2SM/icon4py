# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from icon4py.model.common.utils import data_allocation as data_alloc


def compute_flat_idx_max(
    e2c: data_alloc.NDArray,
    z_mc: data_alloc.NDArray,
    c_lin_e: data_alloc.NDArray,
    z_ifc: data_alloc.NDArray,
    k_lev: data_alloc.NDArray,
    horizontal_lower: int,
    horizontal_upper: int,
) -> data_alloc.NDArray:
    z_me = np.sum(z_mc[e2c] * np.expand_dims(c_lin_e, axis=-1), axis=1)
    z_ifc_e_0 = z_ifc[e2c[:, 0]]
    z_ifc_e_k_0 = np.roll(z_ifc_e_0, -1, axis=1)
    z_ifc_e_1 = z_ifc[e2c[:, 1]]
    z_ifc_e_k_1 = np.roll(z_ifc_e_1, -1, axis=1)
    flat_idx = np.zeros_like(z_me)
    for je in range(horizontal_lower, horizontal_upper):
        for jk in range(k_lev.shape[0] - 1):
            if (
                (z_me[je, jk] <= z_ifc_e_0[je, jk])
                and (z_me[je, jk] >= z_ifc_e_k_0[je, jk])
                and (z_me[je, jk] <= z_ifc_e_1[je, jk])
                and (z_me[je, jk] >= z_ifc_e_k_1[je, jk])
            ):
                flat_idx[je, jk] = k_lev[jk]
    flat_idx_max = np.amax(flat_idx, axis=1)
    return flat_idx_max.astype(np.int32)
