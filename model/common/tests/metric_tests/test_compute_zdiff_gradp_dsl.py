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
import pytest
from gt4py.next import as_field
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common.dimension import (
    CellDim,
    E2CDim,
    ECDim,
    EdgeDim,
    KDim,
)
from icon4py.model.common.grid.horizontal import (
    HorizontalMarkerIndex,
)
from icon4py.model.common.interpolation.stencils.cell_2_edge_interpolation import (
    _cell_2_edge_interpolation,
)
from icon4py.model.common.metrics.metric_fields import (
    _compute_flat_idx,
    _compute_z_aux2,
    compute_z_mc,
)
from icon4py.model.common.metrics.stencils.compute_zdiff_gradp_dsl import compute_zdiff_gradp_dsl
from icon4py.model.common.test_utils.helpers import (
    dallclose,
    flatten_first_two_dims,
    is_roundtrip,
    zero_field,
)


@pytest.mark.datatest
def test_compute_zdiff_gradp_dsl(icon_grid, metrics_savepoint, interpolation_savepoint, backend):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    zdiff_gradp_ref = metrics_savepoint.zdiff_gradp()
    z_mc = zero_field(icon_grid, CellDim, KDim)
    z_ifc = metrics_savepoint.z_ifc()
    k_lev = as_field((KDim,), np.arange(icon_grid.num_levels, dtype=int))
    z_me = zero_field(icon_grid, EdgeDim, KDim)
    horizontal_start_edge = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
    )
    end_edge_nudging = icon_grid.get_end_index(EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim))
    compute_z_mc.with_backend(backend)(
        z_ifc,
        z_mc,
        horizontal_start=int32(0),
        horizontal_end=icon_grid.num_cells,
        vertical_start=int32(0),
        vertical_end=int32(icon_grid.num_levels),
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )
    _cell_2_edge_interpolation(
        in_field=z_mc,
        coeff=interpolation_savepoint.c_lin_e(),
        out=z_me,
        offset_provider={"E2C": icon_grid.get_offset_provider("E2C")},
    )
    flat_idx = zero_field(icon_grid, EdgeDim, KDim)
    _compute_flat_idx(
        z_me=z_me,
        z_ifc=z_ifc,
        k_lev=k_lev,
        out=flat_idx,
        domain={
            EdgeDim: (horizontal_start_edge, icon_grid.num_edges),
            KDim: (int32(0), icon_grid.num_levels),
        },
        offset_provider={
            "E2C": icon_grid.get_offset_provider("E2C"),
            "Koff": icon_grid.get_offset_provider("Koff"),
        },
    )
    flat_idx_np = np.amax(flat_idx.asnumpy(), axis=1)
    z_ifc_sliced = as_field((CellDim,), z_ifc.asnumpy()[:, icon_grid.num_levels])
    z_aux2 = zero_field(icon_grid, EdgeDim)
    _compute_z_aux2(
        z_ifc=z_ifc_sliced,
        out=z_aux2,
        domain={EdgeDim: (end_edge_nudging, icon_grid.num_edges)},
        offset_provider={"E2C": icon_grid.get_offset_provider("E2C")},
    )

    zdiff_gradp_full_np = compute_zdiff_gradp_dsl(
        e2c=icon_grid.connectivities[E2CDim],
        z_me=z_me.asnumpy(),
        z_mc=z_mc.asnumpy(),
        z_ifc=metrics_savepoint.z_ifc().asnumpy(),
        flat_idx=flat_idx_np,
        z_aux2=z_aux2.asnumpy(),
        nlev=icon_grid.num_levels,
        horizontal_start=horizontal_start_edge,
        horizontal_start_1=end_edge_nudging,
        nedges=icon_grid.num_edges,
    )
    zdiff_gradp_full_field = flatten_first_two_dims(
        ECDim, KDim, field=as_field((EdgeDim, E2CDim, KDim), zdiff_gradp_full_np)
    )
    assert dallclose(zdiff_gradp_full_field.asnumpy(), zdiff_gradp_ref.asnumpy(), rtol=1.0e-5)
