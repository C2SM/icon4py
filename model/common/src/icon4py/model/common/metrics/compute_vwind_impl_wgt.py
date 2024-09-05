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
    backend,
    icon_grid: grid.BaseGrid,
    vct_a: fa.KField[wpfloat],
    z_ifc: fa.CellKField[wpfloat],
    z_ddxn_z_half_e: fa.EdgeKField[wpfloat],
    z_ddxt_z_half_e: fa.EdgeKField[wpfloat],
    dual_edge_length: fa.EdgeField[wpfloat],
    vwind_impl_wgt_full: fa.CellField[wpfloat],
    vwind_impl_wgt_k: fa.CellField[wpfloat],
    global_exp: str,
    experiment: str,
    vwind_offctr: float,
    horizontal_start_cell: int,
) -> np.ndarray:
    z_ddxn_z_half_e = gtx.as_field(
        [
            dims.EdgeDim,
        ],
        z_ddxn_z_half_e.asnumpy()[:, icon_grid.num_levels],
    )
    z_ddxt_z_half_e = gtx.as_field(
        [
            dims.EdgeDim,
        ],
        z_ddxt_z_half_e.asnumpy()[:, icon_grid.num_levels],
    )
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
