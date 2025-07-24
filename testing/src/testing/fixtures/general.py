# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common.utils import data_allocation as data_alloc


@pytest.fixture(scope="session")
def connectivities_as_numpy(grid) -> dict[gtx.Dimension, np.ndarray]:
    return {dim: data_alloc.as_numpy(table) for dim, table in grid.neighbor_tables.items()}
