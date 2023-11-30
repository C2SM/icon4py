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

import numpy as np
import pytest

from icon4py.model.common.dimension import EdgeDim
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex
from icon4py.model.common.interpolation.stencils.calc_nudgecoeffs import calc_nudgecoeffs
from icon4py.model.common.test_utils.datatest_fixtures import (  # noqa: F401  # import fixtures from test_utils package
    data_provider,
    datapath,
    download_ser_data,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    processor_props,
    ranked_data_path,
)


@pytest.mark.datatest
def test_calc_nudgecoeffs_e(
    grid_savepoint, interpolation_savepoint, icon_grid  # noqa: F811  # fixture
):
    nudgecoeff_e_ref = interpolation_savepoint.nudgecoeff_e()
    refin_ctrl = grid_savepoint.refin_ctrl(EdgeDim)
    grf_nudge_start_e = HorizontalMarkerIndex.nudging(EdgeDim)
    nudge_max_coeff = 0.075

    nudgecoeff_e = calc_nudgecoeffs(
        refin_ctrl,
        grf_nudge_start_e,
        nudge_max_coeff,
        offset_provider={},
    )

    assert np.allclose(nudgecoeff_e, nudgecoeff_e_ref.asnumpy())
