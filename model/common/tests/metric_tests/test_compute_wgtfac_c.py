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

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.metrics.stencils.compute_wgtfac_c import compute_wgtfac_c
from icon4py.model.common.test_utils import datatest_utils as dt_utils
from icon4py.model.common.test_utils.helpers import dallclose, zero_field
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_wgtfac_c(icon_grid, metrics_savepoint):  # fixture
    wgtfac_c = zero_field(icon_grid, CellDim, KDim, dtype=wpfloat, extend={KDim: 1})
    wgtfac_c_ref = metrics_savepoint.wgtfac_c()
    z_ifc = metrics_savepoint.z_ifc()
    k = field_alloc.allocate_indices(KDim, grid=icon_grid, is_halfdim=True)

    vertical_end = icon_grid.num_levels

    compute_wgtfac_c(
        wgtfac_c,
        z_ifc,
        k,
        nlev=vertical_end,
        offset_provider={"Koff": KDim},
    )

    assert dallclose(wgtfac_c.asnumpy(), wgtfac_c_ref.asnumpy())
