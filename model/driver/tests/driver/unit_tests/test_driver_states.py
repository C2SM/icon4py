# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for driver-state assembly helpers (``driver_states``).

Data-free: they use the ``simple_grid`` and need no serialized test data.
"""

import pytest

from icon4py.model.common.grid import base, simple
from icon4py.model.common.states import adv_states
from icon4py.model.driver import driver_states


@pytest.fixture
def grid() -> base.Grid:
    return simple.simple_grid()


def test_prep_adv_shares_the_tracer_advection_buffers(grid: base.Grid) -> None:
    adv_prep_adv_state = adv_states.initialize_advection_prep_adv_state(grid, None)
    prep_adv = driver_states.link_prep_adv_to_dycore(
        grid, None, adv_prep_adv_state=adv_prep_adv_state, solve_nonhydro_enabled=True
    )
    assert prep_adv is not None
    # Advection must read the very buffers the dycore accumulates into: identity, not equality.
    assert prep_adv.vn_traj is adv_prep_adv_state.vn_traj
    assert prep_adv.mass_flx_me is adv_prep_adv_state.mass_flx_me
    assert (
        prep_adv.dynamical_vertical_mass_flux_at_cells_on_half_levels
        is adv_prep_adv_state.mass_flx_ic
    )


def test_prep_adv_is_none_when_the_dycore_is_disabled(grid: base.Grid) -> None:
    assert (
        driver_states.link_prep_adv_to_dycore(
            grid, None, adv_prep_adv_state=None, solve_nonhydro_enabled=False
        )
        is None
    )


def test_prep_adv_without_tracer_advection_falls_back_to_zero_fields(grid: base.Grid) -> None:
    prep_adv = driver_states.link_prep_adv_to_dycore(
        grid, None, adv_prep_adv_state=None, solve_nonhydro_enabled=True
    )
    assert prep_adv is not None
    assert prep_adv.vn_traj.asnumpy().sum() == 0.0
    assert prep_adv.mass_flx_me.asnumpy().sum() == 0.0
    assert prep_adv.dynamical_vertical_mass_flux_at_cells_on_half_levels.asnumpy().sum() == 0.0
