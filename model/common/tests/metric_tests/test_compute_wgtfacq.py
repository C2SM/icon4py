# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common.dimension import E2CDim
from icon4py.model.common.metrics.compute_wgtfacq import (
    compute_wgtfacq_c_dsl,
    compute_wgtfacq_e_dsl,
)
from icon4py.model.common.test_utils import datatest_utils as dt_utils
from icon4py.model.common.test_utils.helpers import dallclose


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_wgtfacq_c_dsl(icon_grid, metrics_savepoint):
    wgtfacq_c_dsl = metrics_savepoint.wgtfacq_c_dsl()

    wgtfacq_c_dsl_np = compute_wgtfacq_c_dsl(
        z_ifc=metrics_savepoint.z_ifc().asnumpy(),
        nlev=icon_grid.num_levels,
    )
    assert dallclose(wgtfacq_c_dsl_np, wgtfacq_c_dsl.asnumpy())


@pytest.mark.datatest
def test_compute_wgtfacq_e_dsl(metrics_savepoint, interpolation_savepoint, icon_grid):
    wgtfacq_e_dsl_ref = metrics_savepoint.wgtfacq_e_dsl(icon_grid.num_levels + 1)

    wgtfacq_e_dsl_full = compute_wgtfacq_e_dsl(
        e2c=icon_grid.connectivities[E2CDim],
        z_ifc=metrics_savepoint.z_ifc().asnumpy(),
        z_aux_c=metrics_savepoint.wgtfac_c().asnumpy(),
        c_lin_e=interpolation_savepoint.c_lin_e().asnumpy(),
        n_edges=icon_grid.num_edges,
        nlev=icon_grid.num_levels,
    )

    assert dallclose(wgtfacq_e_dsl_full, wgtfacq_e_dsl_ref.asnumpy())
