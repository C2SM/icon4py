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

from icon4py.model.common.metrics.metric_fields import compute_vwind_impl_wgt_partial


def compute_vwind_impl_wgt(
    backend,
    icon_grid,
    vct_a,
    z_ifc,
    z_ddxn_z_half_e,
    z_ddxt_z_half_e,
    dual_edge_length,
    vwind_impl_wgt_full,
    vwind_impl_wgt_k,
    global_exp: str,
    experiment: str,
    vwind_offctr: float,
    horizontal_start_cell: int,
):
    compute_vwind_impl_wgt_partial.with_backend(backend)(
        z_ddxn_z_half_e=z_ddxn_z_half_e,
        z_ddxt_z_half_e=z_ddxt_z_half_e,
        dual_edge_length=dual_edge_length,
        vct_a=vct_a,
        z_ifc=z_ifc,
        vwind_impl_wgt=vwind_impl_wgt_full,
        vwind_impl_wgt_k=vwind_impl_wgt_k,
        vwind_offctr=vwind_offctr,
        horizontal_start=horizontal_start_cell,
        horizontal_end=icon_grid.num_cells,
        vertical_start=max(10, icon_grid.num_levels - 8),
        vertical_end=icon_grid.num_levels,
        offset_provider={
            "C2E": icon_grid.get_offset_provider("C2E"),
            "Koff": icon_grid.get_offset_provider("Koff"),
        },
    )

    vwind_impl_wgt = (
        np.amin(vwind_impl_wgt_k.asnumpy(), axis=1)
        if experiment == global_exp
        else np.amax(vwind_impl_wgt_k.asnumpy(), axis=1)
    )
    return vwind_impl_wgt
