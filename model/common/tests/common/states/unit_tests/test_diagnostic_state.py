# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``states.diagnostic_state``."""

from typing import Any

import gt4py.next.typing as gtx_typing
import numpy as np

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import simple
from icon4py.model.common.states import diagnostic_state as diagnostics
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.fixtures import backend


def test_surface_pressure_is_a_view_of_the_bottom_half_level(
    backend: gtx_typing.Backend | None,
) -> None:
    """Reading the surface pressure neither allocates nor copies, on host and device alike."""
    grid = simple.simple_grid()
    state = diagnostics.initialize_diagnostic_state(grid, allocator=backend)

    surface_pressure = state.surface_pressure

    assert surface_pressure.domain.dims == (dims.CellDim,)
    assert surface_pressure.ndarray.shape == (grid.num_cells,)
    # a view: writes into the bottom half level show through, and the other way round
    # (Any: the array protocol of gt4py fields does not declare item assignment)
    pressure_ifc: Any = state.pressure_ifc.ndarray
    pressure_ifc[:, -1] = 101325.0
    np.testing.assert_array_equal(data_alloc.as_numpy(surface_pressure), 101325.0)
    surface: Any = surface_pressure.ndarray
    surface[0] = 0.0
    assert data_alloc.as_numpy(state.pressure_ifc)[0, -1] == 0.0
    # the other half levels are untouched
    assert not data_alloc.as_numpy(state.pressure_ifc)[:, :-1].any()
