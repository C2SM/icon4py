# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import icon4py.model.common.grid.horizontal as h_grid
from icon4py.model.common import dimension as dims
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
    nudgecoeff_e = zero_field(icon_grid, dims.EdgeDim, dtype=wpfloat)
    nudgecoeff_e_ref = interpolation_savepoint.nudgecoeff_e()
    refin_ctrl = grid_savepoint.refin_ctrl(dims.EdgeDim)
    grf_nudge_start_e = h_grid.RefinCtrlLevel.boundary_nudging_start(dims.EdgeDim)
    nudge_max_coeff = wpfloat(0.375)
    nudge_efold_width = wpfloat(2.0)
    nudge_zone_width = 10

    domain = h_grid.domain(dims.EdgeDim)
    horizontal_start = icon_grid.start_index(domain(h_grid.Zone.NUDGING_LEVEL_2))
    horizontal_end = icon_grid.end_index(domain(h_grid.Zone.LOCAL))

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
