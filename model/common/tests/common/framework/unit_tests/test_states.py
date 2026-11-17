# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common import framework as fw
from icon4py.model.common.framework import states
from icon4py.model.common.grid import base as base_grid
from icon4py.model.testing.fixtures import backend_like, grid, grid_manager


def test_tracer_state_allocates_the_configured_tracers_only(grid: base_grid.Grid) -> None:
    config = states.TracerConfig.from_ntracer(2)
    tracers = fw.allocate(states.TracerState, grid, allocator=None, only=config.active_names)
    assert list(fw.fields(tracers)) == ["qv", "qc"]
    assert tracers.qi is None
