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
from gt4py.next.ffront.fbuiltins import Field, int32
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.test_utils.helpers import zero_field
from gt4py.next import as_field
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
    nudgecoeff_e = zero_field(icon_grid,EdgeDim, dtype=wpfloat)
    nudgecoeff_e_ref = interpolation_savepoint.nudgecoeff_e()
    refin_ctrl = as_field((EdgeDim,),np.squeeze(grid_savepoint.refin_ctrl(EdgeDim)))
    grf_nudge_start_e = HorizontalMarkerIndex.nudging(EdgeDim)
    nudge_max_coeff = wpfloat(0.075)

    horizontal_start = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
    )
    horizontal_end = icon_grid.get_end_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
    )

    calc_nudgecoeffs(
        nudgecoeff_e,
        refin_ctrl,
        grf_nudge_start_e,
        nudge_max_coeff,
        horizontal_start,
        horizontal_end,
        offset_provider={},
    )

    assert np.allclose(nudgecoeff_e.asnumpy(), nudgecoeff_e_ref.asnumpy())
