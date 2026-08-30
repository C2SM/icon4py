# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of what the granule refuses, as opposed to what it computes.

Everything the granule computes is checked against serialized ICON output by the integration
tests. What is checked here is the other half of the interface: the state containers carry
members the port does not implement, and a container that is handed one has to be refused
rather than accepted and ignored.

'tracers' and 'ddt_tracers' are the case. They are declared because 'vertdiff' has 'ptr(:)'
and 'ndtr' (turb_vertdiff.f90:135), and they are read nowhere in 'turbulence.py'. Two guards
already exist -- 'check_supported_configuration' in ICON's 'mo_icon4py_turbulence.f90' and the
'nturb_tracer_tot' argument of 'turbulence_init' in 'icon4py.bindings.turbulence_wrapper' --
and both sit on the path from ICON. Neither is on the path a green-line driver, a standalone
experiment or a second wrapper takes, which is to build the containers and call the granule.
That is the path these tests stand on.

THE OTHER HALF -- that an EMPTY tuple is accepted -- is pinned by the integration tests rather
than here: every call of 'run_vertdiff' and 'run' in 'integration_tests/' passes empty tuples
and is compared against the capture, so a refusal that fired unconditionally would fail there.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence import (
    turbulence,
    turbulence_states as states,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import simple
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.fixtures.datatest import backend


if TYPE_CHECKING:
    import gt4py.next as gtx
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


NUM_LEVELS = 4


@pytest.fixture
def grid(backend: gtx_typing.Backend | None) -> base_grid.Grid:
    return simple.simple_grid(allocator=backend, num_levels=NUM_LEVELS)


def _column(grid: base_grid.Grid, allocator: Any, dtype: Any = None) -> gtx.Field:
    return data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=dtype or ta.wpfloat, allocator=allocator
    )


def _state(state_class: type, grid: base_grid.Grid, allocator: Any, **overrides: Any) -> Any:
    """Build a state container of zero fields, shaped from the annotations it declares.

    The values are never read: the refusal under test runs before the first of them is
    touched. They exist because the containers are frozen dataclasses with no defaults, so
    there is no shorter way to hold a tracer tuple than to build the whole container around it.
    """
    values: dict[str, Any] = {}
    for field in dataclasses.fields(state_class):
        annotation = field.type
        dtype = bool if "[bool]" in annotation else ta.wpfloat
        if annotation.startswith("tuple["):
            values[field.name] = ()
        elif "CellKField" in annotation:
            values[field.name] = _column(grid, allocator, dtype)
        else:
            values[field.name] = data_alloc.zero_field(
                grid, dims.CellDim, dtype=dtype, allocator=allocator
            )
    return state_class(**{**values, **overrides})


def _uninitialised_granule() -> turbulence.Turbulence:
    """A granule that never ran '__init__', which is enough and is the point.

    The refusal precedes every use of 'self' -- it depends on neither the grid nor the
    configuration nor the compiled programs -- so exercising it must not require the several
    seconds of GT4Py compilation that 'Turbulence.__init__' performs. Constructing the granule
    properly is what the integration tests do.
    """
    return object.__new__(turbulence.Turbulence)


@pytest.mark.parametrize("count", [1, 2, 5])
def test_run_vertdiff_refuses_the_tracers_it_would_otherwise_ignore(
    grid: base_grid.Grid, backend: gtx_typing.Backend | None, count: int
) -> None:
    """A non-empty 'input_state.tracers' is refused, and the count is in the message.

    Without this the tracers ICON's 'ldiff_qi', 'ldiff_qs', two-moment or SBM microphysics, ART
    or ComIn produce would simply not be diffused: no exception, no NaN, a forecast missing a
    physical process.
    """
    input_state = _state(
        states.TurbulenceInputState,
        grid,
        backend,
        tracers=tuple(_column(grid, backend) for _ in range(count)),
    )
    tendency_state = _state(states.TurbulenceTendencyState, grid, backend)

    with pytest.raises(NotImplementedError, match=f"Got {count} tracers"):
        _uninitialised_granule().run_vertdiff(
            input_state=input_state,
            surface_state=_state(states.TurbulenceSurfaceState, grid, backend),
            diagnostic_state=_state(states.TurbulenceDiagnosticState, grid, backend),
            tendency_state=tendency_state,
            dt_var=1.0,
        )


def test_run_vertdiff_refuses_the_tracer_tendencies_on_their_own(
    grid: base_grid.Grid, backend: gtx_typing.Backend | None
) -> None:
    """Both tuples are checked, not just the input one.

    'ddt_tracers' alone is not a configuration ICON produces -- the two are filled from the
    same 'ptr(:)' -- but it is a caller error the granule can see, and a caller that filled
    only the tendencies would otherwise be told nothing.
    """
    tendency_state = _state(
        states.TurbulenceTendencyState,
        grid,
        backend,
        ddt_tracers=(_column(grid, backend),),
    )

    with pytest.raises(NotImplementedError, match="1 tracer tendencies"):
        _uninitialised_granule().run_vertdiff(
            input_state=_state(states.TurbulenceInputState, grid, backend),
            surface_state=_state(states.TurbulenceSurfaceState, grid, backend),
            diagnostic_state=_state(states.TurbulenceDiagnosticState, grid, backend),
            tendency_state=tendency_state,
            dt_var=1.0,
        )
