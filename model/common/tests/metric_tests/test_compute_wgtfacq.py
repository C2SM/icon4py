# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.metrics.compute_wgtfacq import (
    compute_wgtfacq_c_dsl,
    compute_wgtfacq_e_dsl,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils
from icon4py.model.testing.helpers import dallclose


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_wgtfacq_c_dsl(icon_grid, metrics_savepoint, backend):
    wgtfacq_c_dsl = metrics_savepoint.wgtfacq_c_dsl()

    xp = data_alloc.import_array_ns(backend)
    wgtfacq_c_dsl_ndarray = compute_wgtfacq_c_dsl(
        z_ifc=metrics_savepoint.z_ifc().ndarray,
        nlev=icon_grid.num_levels,
        array_ns=xp,
    )
    assert dallclose(data_alloc.as_numpy(wgtfacq_c_dsl_ndarray), wgtfacq_c_dsl.asnumpy())


@pytest.mark.datatest
def test_compute_wgtfacq_e_dsl(metrics_savepoint, interpolation_savepoint, icon_grid, backend):
    wgtfacq_e_dsl_ref = metrics_savepoint.wgtfacq_e_dsl(icon_grid.num_levels + 1)
    wgtfacq_c_dsl = metrics_savepoint.wgtfacq_c_dsl()

    xp = data_alloc.import_array_ns(backend)
    wgtfacq_e_dsl_full = compute_wgtfacq_e_dsl(
        e2c=icon_grid.connectivities[dims.E2CDim],
        z_ifc=metrics_savepoint.z_ifc().ndarray,
        wgtfacq_c_dsl=wgtfacq_c_dsl.ndarray,
        c_lin_e=interpolation_savepoint.c_lin_e().ndarray,
        n_edges=icon_grid.num_edges,
        nlev=icon_grid.num_levels,
        array_ns=xp,
    )

    assert dallclose(data_alloc.as_numpy(wgtfacq_e_dsl_full), wgtfacq_e_dsl_ref.asnumpy())
