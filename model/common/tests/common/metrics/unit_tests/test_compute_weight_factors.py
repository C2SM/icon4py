# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from model.common.tests.common.grid.utils import horizontal_dims

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.metrics import compute_weight_factors as weight_factors
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import test_utils
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
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, extend={dims.KDim: 1}, backend=backend
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
        offset_provider={"Koff": dims.KDim},
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
    wgtfacq_e_dsl_ref = metrics_savepoint.wgtfacq_e_dsl(icon_grid.num_levels + 1)
    wgtfacq_c_dsl = metrics_savepoint.wgtfacq_c_dsl()

    xp = data_alloc.import_array_ns(backend)
    wgtfacq_e_dsl_full = weight_factors.compute_wgtfacq_e_dsl(
        e2c=icon_grid.get_connectivity("E2C").ndarray,
        z_ifc=metrics_savepoint.z_ifc().ndarray,
        wgtfacq_c_dsl=wgtfacq_c_dsl.ndarray,
        c_lin_e=interpolation_savepoint.c_lin_e().ndarray,
        n_edges=icon_grid.num_edges,
        nlev=icon_grid.num_levels,
        array_ns=xp,
    )

    assert test_utils.dallclose(
        data_alloc.as_numpy(wgtfacq_e_dsl_full), wgtfacq_e_dsl_ref.asnumpy()
    )


@pytest.mark.datatest
def test_compute_wgtfacq_c_dsl(
    icon_grid: base_grid.Grid,
    metrics_savepoint: sb.MetricSavepoint,
    backend: gtx_typing.Backend | None,
) -> None:
    wgtfacq_c_dsl = metrics_savepoint.wgtfacq_c_dsl()

    xp = data_alloc.import_array_ns(backend)
    wgtfacq_c_dsl_ndarray = weight_factors.compute_wgtfacq_c_dsl(
        z_ifc=metrics_savepoint.z_ifc().ndarray,
        nlev=icon_grid.num_levels,
        array_ns=xp,
    )
    assert test_utils.dallclose(data_alloc.as_numpy(wgtfacq_c_dsl_ndarray), wgtfacq_c_dsl.asnumpy())
