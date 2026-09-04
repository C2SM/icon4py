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

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.interpolation.stencils import cell_2_edge_interpolation
from icon4py.model.common.metrics import compute_weight_factors as weight_factors
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import test_utils
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    process_props,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
def test_compute_wgtfac_c(
    icon_grid: base_grid.Grid,
    metrics_savepoint: sb.MetricSavepoint,
    backend: gtx_typing.Backend | None,
) -> None:
    wgtfac_c = data_alloc.zero_field(
        icon_grid,
        dims.CellDim,
        dims.KHalfDim,
        dtype=ta.wpfloat,
        allocator=backend,
    )
    wgtfac_c_ref = metrics_savepoint.wgtfac_c()
    z_ifc = metrics_savepoint.z_ifc()

    vertical_end = icon_grid.num_levels

    weight_factors.compute_wgtfac_c.with_backend(backend)(
        wgtfac_c,
        z_ifc,
        nlev=vertical_end,
        horizontal_start=0,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=vertical_end + 1,
        offset_provider={},
    )

    assert test_utils.dallclose(wgtfac_c.asnumpy(), wgtfac_c_ref.asnumpy())


@pytest.mark.level("unit")
@pytest.mark.datatest
def test_compute_wgtfacq_e_dsl(
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: base_grid.Grid,
    backend: gtx_typing.Backend | None,
) -> None:
    wgtfacq_e_ref = metrics_savepoint.wgtfacq_e()
    wgtfacq_c_ref = metrics_savepoint.wgtfacq_c()

    nlev = icon_grid.num_levels
    wgtfacq_e = gtx.constructors.zeros(
        gtx.domain({dims.EdgeDim: (0, icon_grid.num_edges), dims.KDim: (nlev - 3, nlev)}),
        allocator=backend,
    )
    cell_2_edge_interpolation.cell_2_edge_interpolation.with_backend(backend)(
        in_field=wgtfacq_c_ref,
        coeff=interpolation_savepoint.c_lin_e(),
        out_field=wgtfacq_e,
        horizontal_start=0,
        horizontal_end=icon_grid.num_edges,
        vertical_start=nlev - 3,
        vertical_end=nlev,
        offset_provider={"E2C": icon_grid.get_connectivity("E2C")},
    )

    assert test_utils.dallclose(wgtfacq_e.asnumpy(), wgtfacq_e_ref.asnumpy())


@pytest.mark.datatest
def test_compute_wgtfacq_c_dsl(
    icon_grid: base_grid.Grid,
    metrics_savepoint: sb.MetricSavepoint,
    backend: gtx_typing.Backend | None,
) -> None:
    wgtfacq_c_ref = metrics_savepoint.wgtfacq_c()

    nlev = icon_grid.num_levels
    wgtfacq_c = gtx.constructors.zeros(
        gtx.domain({dims.CellDim: (0, icon_grid.num_cells), dims.KDim: (nlev - 3, nlev)}),
        allocator=backend,
    )
    weight_factors.compute_wgtfacq_c_dsl.with_backend(backend)(
        z_ifc=metrics_savepoint.z_ifc(),
        wgtfacq_c=wgtfacq_c,
        nlev=nlev,
        horizontal_start=0,
        horizontal_end=icon_grid.num_cells,
        vertical_start=nlev - 3,
        vertical_end=nlev,
        offset_provider={},
    )
    assert test_utils.dallclose(wgtfacq_c.asnumpy(), wgtfacq_c_ref.asnumpy())
