# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import pytest

import icon4py.model.common.grid.horizontal as h_grid
import icon4py.model.testing.datatest_utils as dt_utils
from icon4py.model.common import dimension as dims
from icon4py.model.common.metrics.compute_zdiff_gradp_dsl import compute_zdiff_gradp_dsl
from icon4py.model.common.metrics.metric_fields import (
    compute_flat_idx,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.helpers import (
    dallclose,
)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_zdiff_gradp_dsl(
    icon_grid, metrics_savepoint, interpolation_savepoint, backend, experiment
):
    xp = data_alloc.import_array_ns(backend)
    zdiff_gradp_ref = metrics_savepoint.zdiff_gradp()

    c_lin_e = interpolation_savepoint.c_lin_e()
    z_ifc = metrics_savepoint.z_ifc()
    z_ifc_ground_level = z_ifc.ndarray[:, icon_grid.num_levels]
    z_mc = metrics_savepoint.z_mc()
    k_lev = data_alloc.index_field(icon_grid, dims.KDim, dtype=gtx.int32, backend=backend)
    flat_idx = data_alloc.zero_field(
        icon_grid, dims.EdgeDim, dims.KDim, dtype=gtx.int32, backend=backend
    )
    edge_domain = h_grid.domain(dims.EdgeDim)
    horizontal_start_edge = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    start_nudging = icon_grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))

    compute_flat_idx.with_backend(backend)(
        z_mc=z_mc,
        c_lin_e=c_lin_e,
        z_ifc=z_ifc,
        k_lev=k_lev,
        flat_idx=flat_idx,
        horizontal_start=horizontal_start_edge,
        horizontal_end=icon_grid.num_edges,
        vertical_start=0,
        vertical_end=icon_grid.num_levels - 1,
        offset_provider={
            "E2C": icon_grid.get_offset_provider("E2C"),
            "Koff": icon_grid.get_offset_provider("Koff"),
        },
    )

    flat_idx_np = xp.amax(flat_idx.ndarray, axis=1)

    zdiff_gradp_full_field = compute_zdiff_gradp_dsl(
        e2c=icon_grid.connectivities[dims.E2CDim],
        z_mc=z_mc.ndarray,
        c_lin_e=c_lin_e.ndarray,
        z_ifc=metrics_savepoint.z_ifc().ndarray,
        flat_idx=flat_idx_np,
        z_ifc_sliced=z_ifc_ground_level,
        nlev=icon_grid.num_levels,
        horizontal_start=horizontal_start_edge,
        horizontal_start_1=start_nudging,
        array_ns=xp,
    )

    assert dallclose(
        data_alloc.as_numpy(zdiff_gradp_full_field),
        zdiff_gradp_ref.asnumpy(),
        atol=1e-10,
        rtol=1.0e-9,
    )
