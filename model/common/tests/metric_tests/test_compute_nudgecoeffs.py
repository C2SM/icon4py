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
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex, RefinCtrlLevel
from icon4py.model.common.metrics.compute_nudgecoeffs import compute_nudgecoeffs
from icon4py.model.common.test_utils import datatest_utils as dt_utils
from icon4py.model.common.test_utils.datatest_fixtures import (  # noqa: F401  # import fixtures from test_utils package
    data_provider,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    processor_props,
    ranked_data_path,
)
from icon4py.model.common.test_utils.helpers import zero_field
from icon4py.model.common.type_alias import wpfloat


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_nudgecoeffs_e(
    grid_savepoint,  # noqa: F811 # fixture
    interpolation_savepoint,  # noqa: F811 # fixture
    icon_grid,  # noqa: F811  # fixture
):
    nudgecoeff_e = zero_field(icon_grid, EdgeDim, dtype=wpfloat)
    nudgecoeff_e_ref = interpolation_savepoint.nudgecoeff_e()
    refin_ctrl = grid_savepoint.refin_ctrl(EdgeDim)
    grf_nudge_start_e = RefinCtrlLevel.boundary_nudging_start(EdgeDim)
    nudge_max_coeff = wpfloat(0.375)
    nudge_efold_width = wpfloat(2.0)
    nudge_zone_width = 10

    horizontal_start = icon_grid.get_start_index(
        EdgeDim, HorizontalMarkerIndex.nudging_2nd_level(EdgeDim)
    )
    horizontal_end = icon_grid.get_end_index(
        EdgeDim,
        HorizontalMarkerIndex.local(EdgeDim),
    )

    compute_nudgecoeffs(
        nudgecoeff_e,
        refin_ctrl,
        grf_nudge_start_e,
        nudge_max_coeff,
        nudge_efold_width,
        nudge_zone_width,
        horizontal_start,
        horizontal_end,
        offset_provider={},
    )

    assert np.allclose(nudgecoeff_e.asnumpy(), nudgecoeff_e_ref.asnumpy())
