# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
import pytest

import icon4py.model.common.grid.horizontal as h_grid
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.metrics.compute_zdiff_gradp_dsl import compute_zdiff_gradp_dsl
from icon4py.model.common.metrics.metric_fields import compute_flat_max_idx
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    processor_props,
    ranked_data_path,
)
from icon4py.model.testing.test_utils import dallclose


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


@pytest.mark.level("unit")
@pytest.mark.datatest
def test_compute_zdiff_gradp_dsl(
    icon_grid: base_grid.Grid,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    backend: gtx_typing.Backend,
) -> None:
    xp = data_alloc.import_array_ns(backend)
    zdiff_gradp_ref = metrics_savepoint.zdiff_gradp()
    vertoffset_gradp_ref = metrics_savepoint.vertoffset_gradp()

    c_lin_e = interpolation_savepoint.c_lin_e()
    z_ifc = metrics_savepoint.z_ifc()
    z_ifc_ground_level = z_ifc.ndarray[:, icon_grid.num_levels]
    z_mc = metrics_savepoint.z_mc()
    k_lev = data_alloc.index_field(
        icon_grid, dims.KDim, extend={dims.KDim: 1}, dtype=gtx.int32, allocator=backend
    )
    edge_domain = h_grid.domain(dims.EdgeDim)
    horizontal_start_edge = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    start_nudging = icon_grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))

    flat_idx_np = compute_flat_max_idx(
        e2c=icon_grid.get_connectivity("E2C").ndarray,
        z_mc=z_mc.ndarray,
        c_lin_e=c_lin_e.ndarray,
        z_ifc=z_ifc.ndarray,
        k_lev=k_lev.ndarray,
        array_ns=xp,
    )

    zdiff_gradp_full_field, vertoffset_gradp_full_field = compute_zdiff_gradp_dsl(
        e2c=icon_grid.get_connectivity("E2C").ndarray,
        z_mc=z_mc.ndarray,
        c_lin_e=c_lin_e.ndarray,
        z_ifc=metrics_savepoint.z_ifc().ndarray,
        flat_idx=flat_idx_np,
        topography=z_ifc_ground_level,
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

    assert dallclose(
        data_alloc.as_numpy(vertoffset_gradp_full_field),
        vertoffset_gradp_ref.asnumpy(),
        atol=1e-10,
        rtol=1.0e-9,
    )
