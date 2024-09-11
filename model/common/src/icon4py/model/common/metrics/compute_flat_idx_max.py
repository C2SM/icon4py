# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def compute_flat_idx_max(
    e2c: np.array,
    z_me: np.array,
    z_ifc: np.array,
    k_lev: np.array,
    horizontal_lower: int,
    horizontal_upper: int,
) -> np.array:
    z_ifc_e_0 = z_ifc[e2c[horizontal_lower:horizontal_upper, 0]]
    z_ifc_e_k_0 = z_ifc_e_0[:, 1:]
    z_ifc_e_1 = z_ifc[e2c[horizontal_lower:horizontal_upper, 1]]
    z_ifc_e_k_1 = z_ifc_e_1[:, 1:]
    zero_f = np.zeros_like(z_ifc_e_k_0)
    k_lev_new = np.repeat(k_lev[:65], z_ifc_e_k_0.shape[0]).reshape(z_ifc_e_k_0.shape)
    flat_idx = np.where(
        (z_me[horizontal_lower:horizontal_upper, :65] <= z_ifc_e_0[:, :65])
        & (z_me[horizontal_lower:horizontal_upper, :65] >= z_ifc_e_k_0[:, :65])
        & (z_me[horizontal_lower:horizontal_upper, :65] <= z_ifc_e_1[:, :65])
        & (z_me[horizontal_lower:horizontal_upper, :65] >= z_ifc_e_k_1[:, :65]),
        k_lev_new,
        zero_f,
    )
    flat_idx_max = np.amax(flat_idx, axis=1)
    return np.astype(flat_idx_max, np.int32)
