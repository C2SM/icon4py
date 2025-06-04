# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import logging

import numpy as np
import pytest
from gt4py import next as gtx

from icon4py.model.common.grid import base, gridfile
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils

from . import utils


@pytest.mark.parametrize("grid_file", (dt_utils.REGIONAL_EXPERIMENT, dt_utils.R02B04_GLOBAL))
def test_replace_skip_values(grid_file, caplog, backend):
    caplog.set_level(logging.DEBUG)
    xp = data_alloc.import_array_ns(backend)

    grid = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=None).grid
    clear_skip_values = functools.partial(
        base.replace_skip_values, limited_area=grid.limited_area, array_ns=xp
    )

    horizontal_connectivities = (
        c for c in grid.neighbor_tables.values() if isinstance(c, gtx.Connectivity)
    )
    for connectivity in horizontal_connectivities:
        clear_skip_values(connectivity)
        assert not np.any(connectivity.asnumpy() == gridfile.GridFile.INVALID_INDEX).item()
