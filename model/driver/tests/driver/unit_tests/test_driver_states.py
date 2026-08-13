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

from typing import TYPE_CHECKING

import pytest

from icon4py.model.common.grid import simple as simple_grid
from icon4py.model.common.states import adv_states
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver import driver_states

from ..fixtures import *  # noqa: F403


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


def test_dycore_prep_adv_shares_the_advection_prep_adv_buffers(
    backend: gtx_typing.Backend,
) -> None:
    allocator = data_alloc.get_allocator(backend=backend)
    grid = simple_grid.simple_grid(allocator=allocator)
    adv_prep_adv_state = adv_states.initialize_advection_prep_adv_state(
        grid=grid,
        allocator=allocator,
    )
    dycore_prep_adv = driver_states.link_prep_adv_to_dycore(
        grid=grid,
        allocator=allocator,
        adv_prep_adv_state=adv_prep_adv_state,
        solve_nonhydro_enabled=True,
    )
    assert dycore_prep_adv is not None
    assert dycore_prep_adv.vn_traj is adv_prep_adv_state.vn_traj
    assert dycore_prep_adv.mass_flx_me is adv_prep_adv_state.mass_flx_me
    assert (
        dycore_prep_adv.dynamical_vertical_mass_flux_at_cells_on_half_levels
        is adv_prep_adv_state.mass_flx_ic
    )


@pytest.mark.parametrize(
    "solve_nonhydro_enabled, with_adv_prep_adv_state",
    [
        (False, False),
        (False, True),
    ],
)
def test_dycore_prep_adv_is_none_when_disabled(
    solve_nonhydro_enabled: bool,
    with_adv_prep_adv_state: bool,
    backend: gtx_typing.Backend,
) -> None:
    allocator = data_alloc.get_allocator(backend=backend)
    grid = simple_grid.simple_grid(allocator=allocator)
    adv_prep_adv_state = (
        adv_states.initialize_advection_prep_adv_state(
            grid=grid,
            allocator=allocator,
        )
        if with_adv_prep_adv_state
        else None
    )
    assert (
        driver_states.link_prep_adv_to_dycore(
            grid=grid,
            allocator=allocator,
            adv_prep_adv_state=adv_prep_adv_state,
            solve_nonhydro_enabled=solve_nonhydro_enabled,
        )
        is None
    )
