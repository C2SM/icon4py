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
    n_cells: int,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    init_val = 0.5 + vwind_offctr
    vwind_impl_wgt = array_ns.full(z_ifc.shape[0], init_val)
    for je in range(horizontal_start_cell, n_cells):
        zn_off_0 = z_ddxn_z_half_e[c2e[je, 0], nlev]
        zn_off_1 = z_ddxn_z_half_e[c2e[je, 1], nlev]
        zn_off_2 = z_ddxn_z_half_e[c2e[je, 2], nlev]
        zt_off_0 = z_ddxt_z_half_e[c2e[je, 0], nlev]
        zt_off_1 = z_ddxt_z_half_e[c2e[je, 1], nlev]
        zt_off_2 = z_ddxt_z_half_e[c2e[je, 2], nlev]
        z_maxslope = max(
            abs(zn_off_0), abs(zt_off_0), abs(zn_off_1), abs(zt_off_1), abs(zn_off_2), abs(zt_off_2)
        )
        z_diff = max(
            abs(zn_off_0 * dual_edge_length[c2e[je, 0]]),
            abs(zn_off_1 * dual_edge_length[c2e[je, 1]]),
            abs(zn_off_2 * dual_edge_length[c2e[je, 2]]),
        )

        z_offctr = max(
            vwind_offctr, 0.425 * z_maxslope ** (0.75), min(0.25, 0.00025 * (z_diff - 250.0))
        )
        z_offctr = min(max(vwind_offctr, 0.75), z_offctr)
        vwind_impl_wgt[je] = 0.5 + z_offctr

    for jk in range(max(9, nlev - 9), nlev):
        for je in range(horizontal_start_cell, n_cells):
            z_diff_2 = (z_ifc[je, jk] - z_ifc[je, jk + 1]) / (vct_a[jk] - vct_a[jk + 1])
            if z_diff_2 < 0.6:
                vwind_impl_wgt[je] = max(vwind_impl_wgt[je], 1.2 - z_diff_2)
    return vwind_impl_wgt
