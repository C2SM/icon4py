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

import pytest

from icon4py.model.common.dimension import E2CDim
from icon4py.model.common.metrics.stencils.compute_wgtfacq import (
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
