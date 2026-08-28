# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the state containers of the NWP 1D turbulence granule.

These pin the three properties the containers exist to guarantee: they are frozen (ADR-0001 --
a physics component never mutates its input state), every field carries the dimensions the
Fortran declaration gives it, and the four dycore-supplied shear fields are declared 'vpfloat'
while everything else is 'wpfloat'.

The precision assertion has to inspect the annotations rather than the allocated fields: the
default build is double precision, where 'vpfloat is wpfloat', so a field allocated as 'vpfloat'
is indistinguishable from a 'wpfloat' one at runtime. The declaration is the whole point --
'turb_diffusion.f90:610' is the only 'REAL(KIND=vp)' in the four scheme files, and typing those
four fields 'wpfloat' would be silently right today and wrong the day mixed precision is enabled.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import TYPE_CHECKING, Any

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence import turbulence_states as states
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.grid import simple
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.fixtures.datatest import backend_like


if TYPE_CHECKING:
    import gt4py.next as gtx
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


NUM_LEVELS = 8

#: The only fields of the whole granule interface that are declared 'REAL(KIND=vp)' in the
#: Fortran (turb_diffusion.f90:610). They are produced by the mixed-precision dycore diffusion.
VP_FIELDS = frozenset({"hdef2", "hdiv", "dwdx", "dwdy"})


#: Number of passive tracers the tracer tuples are exercised with. In ICON this is 'ndtr', a
#: runtime count that depends on the microphysics and ART configuration.
NUM_TRACERS = 3


#: What the `TileField` alias of 'turbulence_states' stands for. Spelled out rather than
#: imported so that a change to the alias has to be made twice; `test_tile_field_alias` pins
#: the two together.
TILE_FIELD_ANNOTATION = "tuple[fa.CellField[ta.wpfloat], ...]"


#: Fields whose Fortran declaration puts them on half levels (nlev + 1 entries).
_HALF_LEVEL_FIELDS = frozenset(
    {
        "hhl",
        "w",
        "tke",
        "tket_conv",
        "hdef2",
        "hdiv",
        "dwdx",
        "dwdy",
        "tkvm",
        "tkvh",
        "tprn",
        "rcld",
        "rhon",
        "edr",
        "tur_len_scale",
        "ddt_tke",
        "tket_hshr",
    }
)

STATE_CLASSES = (
    states.TurbulenceMetricState,
    states.TurbulenceInputState,
    states.TurbulenceSurfaceState,
    states.TurbulenceDiagnosticState,
    states.TurbulenceTendencyState,
    states.TurbulenceTileState,
)


@pytest.fixture
def grid(backend_like: gtx_typing.Backend | None) -> base_grid.Grid:
    return simple.simple_grid(allocator=backend_like, num_levels=NUM_LEVELS)


def _full(grid: base_grid.Grid, allocator: Any, dtype: Any = None) -> gtx.Field:
    return data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=dtype or ta.wpfloat, allocator=allocator
    )


def _half(grid: base_grid.Grid, allocator: Any, dtype: Any = None) -> gtx.Field:
    return data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        extend={dims.KDim: 1},
        dtype=dtype or ta.wpfloat,
        allocator=allocator,
    )


def _surface(grid: base_grid.Grid, allocator: Any, dtype: Any = None) -> gtx.Field:
    return data_alloc.zero_field(grid, dims.CellDim, dtype=dtype or ta.wpfloat, allocator=allocator)


def _annotation(field: dataclasses.Field) -> str:
    """The source annotation, with the `TileField` alias resolved to what it stands for."""
    return TILE_FIELD_ANNOTATION if field.type == "TileField" else field.type


def _allocate(state_class: type, grid: base_grid.Grid, allocator: Any) -> Any:
    """Allocate a zero field of the right shape for every annotated field of `state_class`."""
    values = {}
    for field in dataclasses.fields(state_class):
        annotation = _annotation(field)
        dtype = bool if "bool" in annotation else None
        if annotation.startswith("tuple"):
            count = states.NUM_TILES if "CellField" in annotation else NUM_TRACERS
            allocate = _surface if "CellField" in annotation else _full
            values[field.name] = tuple(allocate(grid, allocator) for _ in range(count))
        elif "CellKField" in annotation:
            values[field.name] = (
                _half(grid, allocator, dtype)
                if field.name in _HALF_LEVEL_FIELDS
                else _full(grid, allocator, dtype)
            )
        else:
            values[field.name] = _surface(grid, allocator, dtype)
    return state_class(**values)


@pytest.mark.parametrize("state_class", STATE_CLASSES)
def test_state_can_be_constructed_from_allocated_fields(
    state_class: type, grid: base_grid.Grid, backend_like: gtx_typing.Backend | None
) -> None:
    state = _allocate(state_class, grid, backend_like)
    assert dataclasses.is_dataclass(state)
    assert len(dataclasses.fields(state)) > 0


@pytest.mark.parametrize("state_class", STATE_CLASSES)
def test_state_is_frozen(
    state_class: type, grid: base_grid.Grid, backend_like: gtx_typing.Backend | None
) -> None:
    state = _allocate(state_class, grid, backend_like)
    first = dataclasses.fields(state)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(state, first, None)


@pytest.mark.parametrize("state_class", STATE_CLASSES)
def test_state_fields_have_the_declared_dimensions(
    state_class: type, grid: base_grid.Grid, backend_like: gtx_typing.Backend | None
) -> None:
    state = _allocate(state_class, grid, backend_like)
    for field in dataclasses.fields(state):
        value = getattr(state, field.name)
        entries = value if isinstance(value, tuple) else (value,)
        for entry in entries:
            if "CellKField" in _annotation(field):
                assert entry.domain.dims == (dims.CellDim, dims.KDim), field.name
                expected = NUM_LEVELS + 1 if field.name in _HALF_LEVEL_FIELDS else NUM_LEVELS
                assert entry.shape[1] == expected, field.name
            else:
                assert entry.domain.dims == (dims.CellDim,), field.name


def test_tile_field_alias_is_a_tuple_of_cell_fields() -> None:
    """The tile axis is a Python tuple of plain cell fields, not a field dimension (spec 7.3)."""
    element, ellipsis = typing.get_args(states.TileField)
    assert ellipsis is Ellipsis
    assert element == fa.CellField[ta.wpfloat]


def test_only_the_dycore_shear_fields_are_declared_vpfloat() -> None:
    declared_vp = {
        field.name
        for state_class in STATE_CLASSES
        for field in dataclasses.fields(state_class)
        if "vpfloat" in _annotation(field)
    }
    assert declared_vp == VP_FIELDS


@pytest.mark.parametrize("name", sorted(VP_FIELDS))
def test_shear_fields_are_vpfloat_cell_k_fields(name: str) -> None:
    annotations = {
        field.name: field.type for field in dataclasses.fields(states.TurbulenceInputState)
    }
    assert annotations[name] == "fa.CellKField[ta.vpfloat]"


def test_every_float_field_is_wpfloat_unless_it_is_one_of_the_four() -> None:
    for state_class in STATE_CLASSES:
        for field in dataclasses.fields(state_class):
            annotation = _annotation(field)
            if "bool" in annotation:
                continue
            expected = "ta.vpfloat" if field.name in VP_FIELDS else "ta.wpfloat"
            assert expected in annotation, f"{state_class.__name__}.{field.name}"


def test_input_state_carries_the_four_dycore_shear_fields() -> None:
    names = {field.name for field in dataclasses.fields(states.TurbulenceInputState)}
    assert names >= VP_FIELDS


@pytest.mark.parametrize(
    "dropped",
    ["c_big", "c_sml", "r_air", "tkhm", "tkhh", "tket_sso", "tket_nstc", "tketadv", "qv_conv"],
)
def test_never_passed_dummy_arguments_are_absent(dropped: str) -> None:
    """Arguments the ICON interfaces never pass must not appear in the granule interface."""
    names = {
        field.name for state_class in STATE_CLASSES for field in dataclasses.fields(state_class)
    }
    assert dropped not in names


def test_tile_state_holds_one_field_per_tile(
    grid: base_grid.Grid, backend_like: gtx_typing.Backend | None
) -> None:
    state = _allocate(states.TurbulenceTileState, grid, backend_like)
    assert states.NUM_TILES == 9
    for field in dataclasses.fields(state):
        assert len(getattr(state, field.name)) == states.NUM_TILES, field.name


def test_tile_state_accepts_the_degenerate_untiled_shape(
    grid: base_grid.Grid, backend_like: gtx_typing.Backend | None
) -> None:
    """'ntiles_lnd == 1' forces 'lsnowtile = .FALSE.' and 'ntiles_water = 0' (spec 7.2)."""
    single = {
        field.name: (_surface(grid, backend_like),)
        for field in dataclasses.fields(states.TurbulenceTileState)
    }
    state = states.TurbulenceTileState(**single)
    assert state.num_tiles == 1


def test_tile_state_rejects_a_ragged_tile_axis(
    grid: base_grid.Grid, backend_like: gtx_typing.Backend | None
) -> None:
    values = {
        field.name: tuple(_surface(grid, backend_like) for _ in range(states.NUM_TILES))
        for field in dataclasses.fields(states.TurbulenceTileState)
    }
    values["frac_t"] = values["frac_t"][:-1]
    with pytest.raises(ValueError, match="Ragged tile axis"):
        states.TurbulenceTileState(**values)
