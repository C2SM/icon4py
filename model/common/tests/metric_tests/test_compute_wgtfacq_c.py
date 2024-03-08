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

from icon4py.model.common.metrics.stencils.compute_wgtfacq_c import compute_wgtfacq_c
from icon4py.model.common.test_utils.helpers import dallclose


@pytest.mark.datatest
def test_compute_wgtfacq_c(icon_grid, metrics_savepoint):  # fixture
    wgtfacq_c_dsl = metrics_savepoint.wgtfacq_c_dsl()
    z_ifc = metrics_savepoint.z_ifc()

    vertical_end = icon_grid.num_levels

    wgtfacq_c = compute_wgtfacq_c(
        z_ifc.asnumpy(),
        vertical_end,
    )
    assert dallclose(wgtfacq_c, wgtfacq_c_dsl.asnumpy())
