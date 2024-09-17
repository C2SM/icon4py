# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np

import icon4py.model.common.field_type_aliases as fa
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base as grid
from icon4py.model.common.metrics.metric_fields import compute_vwind_impl_wgt_partial
from icon4py.model.common.type_alias import wpfloat


def compute_vwind_impl_wgt(
    icon_grid: grid.BaseGrid,
    vct_a: fa.KField[wpfloat],
    z_ifc: fa.CellKField[wpfloat],
    z_ddxn_z_half_e: fa.EdgeKField[wpfloat],
    z_ddxt_z_half_e: fa.EdgeKField[wpfloat],
    dual_edge_length: fa.EdgeField[wpfloat],
    global_exp: str,
    experiment: str,
    vwind_offctr: float,
    horizontal_start_cell: int,
) -> np.ndarray:
    init_val = 0.65 if experiment == global_exp else 0.7
    vwind_impl_wgt_full = np.full(z_ifc.shape[0], 0.5 + vwind_offctr)
    vwind_impl_wgt_k = np.full(z_ifc.shape, init_val)

    z_ddxn_z_half_e = gtx.as_field(
        [dims.EdgeDim], z_ddxn_z_half_e[:, icon_grid.num_levels],
    )
    z_ddxt_z_half_e = gtx.as_field(
        [dims.EdgeDim], z_ddxt_z_half_e[:, icon_grid.num_levels],
    )
    compute_vwind_impl_wgt_partial(
        z_ddxn_z_half_e=z_ddxn_z_half_e,
        z_ddxt_z_half_e=z_ddxt_z_half_e,
        dual_edge_length=gtx.as_field([dims.EdgeDim], dual_edge_length),
        vct_a=gtx.as_field([dims.KDim], vct_a),
        z_ifc=gtx.as_field([dims.CellDim, dims.KDim], z_ifc),
        vwind_impl_wgt=gtx.as_field([dims.CellDim], vwind_impl_wgt_full),
        vwind_impl_wgt_k=gtx.as_field([dims.CellDim, dims.KDim], vwind_impl_wgt_k),
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
        np.amin(vwind_impl_wgt_k, axis=1)
        if experiment == global_exp
        else np.amax(vwind_impl_wgt_k, axis=1)
    )
    return vwind_impl_wgt
