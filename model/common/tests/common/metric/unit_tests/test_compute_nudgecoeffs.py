# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid, refinement
from icon4py.model.common.metrics.compute_nudgecoeffs import compute_nudgecoeffs
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    processor_props,
    ranked_data_path,
)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_nudgecoeffs_e(
    grid_savepoint,
    interpolation_savepoint,
    icon_grid,
    backend,
):
    nudgecoeff_e = data_alloc.zero_field(icon_grid, dims.EdgeDim, dtype=wpfloat, backend=backend)
    nudgecoeff_e_ref = interpolation_savepoint.nudgecoeff_e()
    refin_ctrl = grid_savepoint.refin_ctrl(dims.EdgeDim)
    grf_nudge_start_e = refinement.refine_control_value(dims.EdgeDim, h_grid.Zone.NUDGING).value
    max_nudging_coefficient = wpfloat(0.375)
    nudge_efold_width = wpfloat(2.0)
    nudge_zone_width = 10

    domain = h_grid.domain(dims.EdgeDim)
    horizontal_start = icon_grid.start_index(domain(h_grid.Zone.NUDGING_LEVEL_2))
    horizontal_end = icon_grid.end_index(domain(h_grid.Zone.LOCAL))

    compute_nudgecoeffs.with_backend(backend)(
        nudgecoeff_e,
        refin_ctrl,
        grf_nudge_start_e,
        max_nudging_coefficient,
        nudge_efold_width,
        nudge_zone_width,
        horizontal_start,
        horizontal_end,
        offset_provider={},
    )

    assert np.allclose(nudgecoeff_e.asnumpy(), nudgecoeff_e_ref.asnumpy())
