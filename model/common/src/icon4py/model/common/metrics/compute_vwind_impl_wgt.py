# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from types import ModuleType

import numpy as np

from icon4py.model.common.utils import data_allocation as data_alloc


def compute_vwind_impl_wgt(
    c2e: data_alloc.NDArray,
    vct_a: data_alloc.NDArray,
    z_ifc: data_alloc.NDArray,
    z_ddxn_z_half_e: data_alloc.NDArray,
    z_ddxt_z_half_e: data_alloc.NDArray,
    dual_edge_length: data_alloc.NDArray,
    vwind_offctr: float,
    nlev: int,
    horizontal_start_cell: int,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    factor = max(vwind_offctr, 0.75)

    zn_off = array_ns.abs(z_ddxn_z_half_e[:, nlev][c2e])
    zt_off = array_ns.abs(z_ddxt_z_half_e[:, nlev][c2e])
    stacked = array_ns.concatenate((zn_off, zt_off), axis=1)
    maxslope = 0.425 * array_ns.amax(stacked, axis=1) ** (0.75)
    diff = array_ns.minimum(
        0.25, 0.00025 * (np.amax(np.abs(zn_off * dual_edge_length[c2e]), axis=1) - 250.0)
    )
    offctr = array_ns.minimum(
        factor, array_ns.maximum(vwind_offctr, array_ns.maximum(maxslope, diff))
    )
    vwind_impl_wgt = 0.5 + offctr

    k_start = max(0, nlev - 9)

    zdiff2 = (z_ifc[:, 0:nlev] - z_ifc[:, 1 : nlev + 1]) / (vct_a[0:nlev] - vct_a[1 : nlev + 1])

    for jk in range(k_start, nlev):
        zdiff2_sliced = zdiff2[horizontal_start_cell:, jk]
        index_for_k = np.where(zdiff2_sliced < 0.6)[0]
        max_value_k = np.maximum(1.2 - zdiff2_sliced, vwind_impl_wgt[horizontal_start_cell:])
        vwind_impl_wgt[index_for_k + horizontal_start_cell] = max_value_k[index_for_k]

    return vwind_impl_wgt
