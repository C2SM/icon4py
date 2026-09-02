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

THE SECOND GROUP OF TESTS is the configuration the granule cannot represent, which is the same
defect in a different container: 'TurbulenceConfig' accepts what ICON's namelist can express and
'Turbulence' runs less than that, so the difference has to be refused rather than computed
differently. Four of those refusals date from the port; four were added on 2026-09-02 from the
configuration audit ('docs/superpowers/notes/2026-08-31-granule-accepts-what-it-ignores.md') and
each of the four had been accepted and silently ignored until then -- among them 'icldm_turb = 1',
which the configuration's own doc comment names as the DWD global operational setting, and
'a_stab > 0', which is what 'ensemble_pert_nml' produces for an EPS member. The same "other half"
applies: that a supported configuration is ACCEPTED is pinned by the integration tests, which
construct the granule for real.
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


#: The eight configurations 'Turbulence' refuses, each with a value that triggers it and the
#: Fortran citation its message has to carry. The first four are fused guards -- a stencil that
#: folds a guarded Fortran block into an unguarded expression -- and the last four are Fortran
#: blocks the port does not contain at all, which is the class the 2026-08-31 audit found.
REFUSED_CONFIGURATIONS = (
    ("itype_sher", 1, "turb_diffusion.f90:1330"),
    ("ltkeshs", False, "turb_diffusion.f90:1531-1536"),
    ("ltkesso", False, "turb_diffusion.f90:1572-1596"),
    ("c_diff", 0.0, "turb_diffusion.f90:2541"),
    ("icldm_turb", 1, "turb_utilities.f90:869-882"),
    ("rsur_sher", 0.5, "turb_diffusion.f90:968"),
    ("a_stab", 0.5, "turb_utilities.f90:1339-1353"),
    ("it_end", 2, "turb_utilities.f90:387"),
)

#: A configuration 'Turbulence' accepts, so that each test below changes exactly ONE thing and
#: the check that fires is the one it is named after. Only 'itype_sher' has to be said: the
#: compiled-in Fortran default is 0 and the granule runs 2, which is the widest of the eight gaps
#: and is why 'TurbulenceConfig()' on its own is not a configuration the granule will construct.
#: Every other default already satisfies its refusal.
ACCEPTED_BY_THE_GRANULE: dict[str, Any] = {"itype_sher": 2}

#: The settings each refused value needs alongside it to build a valid 'TurbulenceConfig'.
#: 'ltkeshs' is crosschecked against 'a_hshr' inside the configuration, exactly as ICON does at
#: 'mo_nml_crosscheck.f90:432', so switching it off without zeroing 'a_hshr' would fail in
#: 'TurbulenceConfig' and never reach the granule -- which is a different test.
COMPANION_SETTINGS: dict[str, dict[str, Any]] = {"ltkeshs": {"a_hshr": 0.0}}


def _configuration_refusing_only(name: str, value: Any) -> turbulence.TurbulenceConfig:
    """A configuration the granule accepts, with `name` set to a value it does not."""
    return turbulence.TurbulenceConfig(
        **{**ACCEPTED_BY_THE_GRANULE, **COMPANION_SETTINGS.get(name, {}), name: value}
    )


def _construct_the_granule(config: turbulence.TurbulenceConfig, grid: base_grid.Grid) -> None:
    """Construct 'Turbulence' far enough for it to validate `config`.

    'Turbulence.__init__' validates before it allocates a single field or compiles a single
    program, so a refusal costs nothing and neither the vertical grid nor the metric state is
    ever reached -- which is why 'None' can stand in for both. That ordering is part of what
    these tests pin: a refusal placed after the working set was built would still be correct,
    but every case below would then pay the several seconds of GT4Py compilation that the
    integration tests pay once, and a refusal is exactly the situation in which the caller
    should not be made to wait for a build it will not use.
    """
    turbulence.Turbulence(
        grid=grid,
        config=config,
        params=turbulence.TurbulenceParams(config),
        vertical_grid=None,  # type: ignore[arg-type]  # never read: validation comes first
        metric_state=None,  # type: ignore[arg-type]  # never read: validation comes first
        backend=None,
    )


@pytest.mark.parametrize("name, value, citation", REFUSED_CONFIGURATIONS)
def test_the_granule_refuses_the_formulations_it_does_not_carry(
    grid: base_grid.Grid, name: str, value: Any, citation: str
) -> None:
    """Every refusal names the parameter, the value given and the Fortran it stands in for.

    The Fortran citation is what makes the message actionable rather than merely obstructive:
    the reader wants to know which block would have to be ported, and a message that says only
    "not supported" sends them back to the scheme to find out.
    """
    config = _configuration_refusing_only(name, value)

    with pytest.raises(NotImplementedError) as excinfo:
        _construct_the_granule(config, grid)

    message = str(excinfo.value)
    assert name in message, "the message must name the parameter"
    assert citation in message, "the message must name the Fortran block that was not ported"


@pytest.mark.parametrize(
    "name, value",
    [(name, value) for name, value, _ in REFUSED_CONFIGURATIONS if not isinstance(value, bool)],
)
def test_the_refusal_echoes_the_value_it_was_given(
    grid: base_grid.Grid, name: str, value: Any
) -> None:
    """A refusal that does not repeat the offending value makes the caller guess which one it is.

    The boolean refusals are excluded because there is nothing to echo: 'ltkeshs = False' is
    named by the parameter alone.
    """
    config = _configuration_refusing_only(name, value)

    with pytest.raises(NotImplementedError, match=str(value)):
        _construct_the_granule(config, grid)


def test_the_configuration_accepts_what_the_granule_refuses(
    grid: base_grid.Grid,
) -> None:
    """The two layers are deliberately different widths, and this is where that is stated.

    'TurbulenceConfig' is the granule's INTERFACE and has to stay expressible from Fortran, C
    and Python alike (port spec D5/D6), so it accepts whatever ICON's namelist can express;
    'Turbulence' is the implementation and refuses what its stencils cannot represent. Reading
    the config's acceptance as the contract is the mistake this asserts against: 'icldm_turb =
    1' is the DWD global operational setting, it constructs a configuration without complaint,
    and the run still has to stop before it computes anything.
    """
    config = _configuration_refusing_only("icldm_turb", 1)
    assert int(config.icldm_turb) == 1

    with pytest.raises(NotImplementedError, match="icldm_turb"):
        _construct_the_granule(config, grid)


def test_the_ensemble_perturbation_of_the_length_scale_is_refused(
    grid: base_grid.Grid,
) -> None:
    """'a_stab' is not a hypothetical setting, and the message has to say where it came from.

    It is not reachable from 'turbdiff_nml' in practice: 'set_scalar_ens_pert' overwrites it
    with a positive-definite perturbation for every EPS member
    (mo_ensemble_pert_config.f90:818-820). A member configured that way and run through the
    granule would produce the UNPERTURBED member's stable-boundary-layer mixing, collapsing the
    spread the perturbation exists to create -- and the run would look entirely healthy. So the
    message names 'ensemble_pert_nml', which is where the reader has to go to change it.
    """
    with pytest.raises(NotImplementedError, match="ensemble_pert_nml"):
        _construct_the_granule(_configuration_refusing_only("a_stab", 0.5), grid)


def test_the_surface_shear_refusal_names_the_outputs_that_would_go_unwritten(
    grid: base_grid.Grid,
) -> None:
    """'rsur_sher > 0' is the one refusal whose absence would show up a timestep later.

    Nothing inside the call reads 'tfm', 'tfh' or 'tfv' after 'turbdiff' writes them, so a
    blue-line VERIFY run comparing only what the granule returns would report nothing at all.
    The error appears through the next 'turbtran', which is not ported and therefore not
    compared. A message that named only the switch would leave that invisible.
    """
    with pytest.raises(NotImplementedError) as excinfo:
        _construct_the_granule(_configuration_refusing_only("rsur_sher", 0.5), grid)

    message = str(excinfo.value)
    for output in ("tfm", "tfh", "tfv"):
        assert output in message, f"the message must name '{output}'"


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
