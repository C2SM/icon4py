# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.common.components import framework as fw, states
from icon4py.model.common.grid import base as base_grid, simple


@pytest.fixture
def grid() -> base_grid.Grid:
    return simple.simple_grid()


def test_prognostic_state_lists_its_five_leaves() -> None:
    assert [d.name for d in states.PrognosticState.declarations()] == [
        "rho",
        "w",
        "vn",
        "exner",
        "theta_v",
    ]


def test_tracer_state_allocates_the_configured_tracers_only(grid: base_grid.Grid) -> None:
    config = states.TracerConfig.from_ntracer(2)
    tracers = fw.allocate(states.TracerState, grid, allocator=None, only=config.active_names)
    assert [d.name for d, _ in tracers.leaves()] == ["qv", "qc"]
    assert tracers.qi is None
    assert all(d.optional for d in states.TracerState.declarations())


def test_tracer_state_copy_allocates_new_buffers_for_the_active_tracers(
    grid: base_grid.Grid,
) -> None:
    tracers = fw.allocate(states.TracerState, grid, allocator=None, only=("qv",))
    np.asarray(tracers.qv.data.ndarray)[...] = 2.0  # type: ignore[union-attr]
    copy = fw.copy(tracers, None)
    assert copy.qc is None
    assert copy.qv is not None and copy.qv is not tracers.qv
    assert np.all(np.asarray(copy.qv.data.ndarray) == 2.0)


def test_prep_advection_merges_the_dycore_and_advection_views() -> None:
    assert [d.name for d in states.PrepAdvection.declarations()] == [
        "vn_traj",
        "mass_flx_me",
        "dynamical_vertical_mass_flux_at_cells_on_half_levels",
        "dynamical_vertical_volumetric_flux_at_cells_on_half_levels",
    ]
