# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.metrics import compute_weight_factors as weight_factors
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils, helpers


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_wgtfac_c(icon_grid, metrics_savepoint, backend):  # fixture
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
        offset_provider={"Koff": dims.KDim},
    )

    assert helpers.dallclose(wgtfac_c.asnumpy(), wgtfac_c_ref.asnumpy())


@pytest.mark.datatest
def test_compute_wgtfacq_e_dsl(metrics_savepoint, interpolation_savepoint, icon_grid, backend):
    wgtfacq_e_dsl_ref = metrics_savepoint.wgtfacq_e_dsl(icon_grid.num_levels + 1)
    wgtfacq_c_dsl = metrics_savepoint.wgtfacq_c_dsl()

    xp = data_alloc.import_array_ns(backend)
    wgtfacq_e_dsl_full = weight_factors.compute_wgtfacq_e_dsl(
        e2c=icon_grid.connectivities[dims.E2CDim],
        z_ifc=metrics_savepoint.z_ifc().ndarray,
        wgtfacq_c_dsl=wgtfacq_c_dsl.ndarray,
        c_lin_e=interpolation_savepoint.c_lin_e().ndarray,
        n_edges=icon_grid.num_edges,
        nlev=icon_grid.num_levels,
        array_ns=xp,
    )

    assert helpers.dallclose(data_alloc.as_numpy(wgtfacq_e_dsl_full), wgtfacq_e_dsl_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_wgtfacq_c_dsl(icon_grid, metrics_savepoint, backend):
    wgtfacq_c_dsl = metrics_savepoint.wgtfacq_c_dsl()

    xp = data_alloc.import_array_ns(backend)
    wgtfacq_c_dsl_ndarray = weight_factors.compute_wgtfacq_c_dsl(
        z_ifc=metrics_savepoint.z_ifc().ndarray,
        nlev=icon_grid.num_levels,
        array_ns=xp,
    )
    assert helpers.dallclose(data_alloc.as_numpy(wgtfacq_c_dsl_ndarray), wgtfacq_c_dsl.asnumpy())
