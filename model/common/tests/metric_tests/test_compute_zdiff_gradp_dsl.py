# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.grid.horizontal as h_grid
from icon4py.model.common import dimension as dims
from icon4py.model.common.interpolation.stencils.cell_2_edge_interpolation import (
    _cell_2_edge_interpolation,
)
from icon4py.model.common.metrics.compute_zdiff_gradp_dsl import compute_zdiff_gradp_dsl
from icon4py.model.common.metrics.metric_fields import (
    _compute_flat_idx,
    compute_z_mc,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.helpers import (
    dallclose,
    is_roundtrip,
)


@pytest.mark.datatest
def test_compute_zdiff_gradp_dsl(icon_grid, metrics_savepoint, interpolation_savepoint, backend):
    if data_alloc.is_cupy_device(backend):
        pytest.skip(
            "skipping: gpu backend is unsupported because _compute_flat_idx and _compute_z_aux2 cannot be called with_backend"
        )
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    zdiff_gradp_ref = metrics_savepoint.zdiff_gradp()
    z_mc = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, backend=backend)
    z_ifc = metrics_savepoint.z_ifc()
    k_lev = gtx.as_field(
        (dims.KDim,), np.arange(icon_grid.num_levels, dtype=int), allocator=backend
    )
    z_me = data_alloc.zero_field(icon_grid, dims.EdgeDim, dims.KDim, backend=backend)
    edge_domain = h_grid.domain(dims.EdgeDim)
    horizontal_start_edge = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    start_nudging = icon_grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
    compute_z_mc.with_backend(backend)(
        z_ifc,
        z_mc,
        horizontal_start=gtx.int32(0),
        horizontal_end=icon_grid.num_cells,
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(icon_grid.num_levels),
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )
    _cell_2_edge_interpolation(
        in_field=z_mc,
        coeff=interpolation_savepoint.c_lin_e(),
        out=z_me,
        offset_provider={"E2C": icon_grid.get_offset_provider("E2C")},
    )
    flat_idx = data_alloc.zero_field(icon_grid, dims.EdgeDim, dims.KDim, backend=backend)
    _compute_flat_idx(
        z_mc=z_mc,
        c_lin_e=interpolation_savepoint.c_lin_e(),
        z_ifc=z_ifc,
        k_lev=k_lev,
        out=flat_idx,
        domain={
            dims.EdgeDim: (horizontal_start_edge, icon_grid.num_edges),
            dims.KDim: (0, icon_grid.num_levels),
        },
        offset_provider={
            "E2C": icon_grid.get_offset_provider("E2C"),
            "Koff": icon_grid.get_offset_provider("Koff"),
        },
    )
    xp = data_alloc.import_array_ns(backend)
    flat_idx_np = xp.amax(flat_idx.ndarray, axis=1)
    z_ifc_sliced = gtx.as_field(
        (dims.CellDim,), z_ifc.ndarray[:, icon_grid.num_levels], allocator=backend
    )

    zdiff_gradp_full_field = compute_zdiff_gradp_dsl(
        e2c=icon_grid.connectivities[dims.E2CDim],
        z_mc=z_mc.asnumpy(),
        c_lin_e=interpolation_savepoint.c_lin_e().asnumpy(),
        z_ifc=metrics_savepoint.z_ifc().asnumpy(),
        flat_idx=flat_idx_np,
        z_ifc_sliced=z_ifc_sliced.asnumpy(),
        nlev=icon_grid.num_levels,
        horizontal_start=horizontal_start_edge,
        horizontal_start_1=start_nudging,
    )

    assert dallclose(zdiff_gradp_full_field, zdiff_gradp_ref.asnumpy(), rtol=1.0e-5)
