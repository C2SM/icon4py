# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the state containers of the NWP 1D turbulence granule.

These pin the three properties the containers exist to guarantee: they are frozen (ADR-0001 --
a physics component never mutates its input state), every field carries the axes the Fortran
declaration gives it, and the four dycore-supplied shear fields are declared 'vpfloat' while
everything else is 'wpfloat'.

The axes are pinned against `DECLARED_SHAPES`, a table read off the dummy-argument declarations
of the three subroutines and written down here rather than derived from 'turbulence_states'. That
independence is the point: a table built from the annotations it then checks would agree with any
annotation, including one that has lost its vertical axis.

A type annotation cannot say whether a column carries 'nlev' or 'nlev + 1' entries -- both are
'fa.CellKField' -- so the vertical extent in the table is pinned against the field's own doc
comment instead, which is the module's separate written record of the same Fortran fact.

The precision assertion likewise has to inspect the annotations rather than the allocated fields:
the default build is double precision, where 'vpfloat is wpfloat', so a field allocated as
'vpfloat' is indistinguishable from a 'wpfloat' one at runtime. The declaration is the whole point
-- 'turb_diffusion.f90:612' is the only 'REAL(KIND=vp)' in the four scheme files, and typing those
four fields 'wpfloat' would be silently right today and wrong the day mixed precision is enabled.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import typing
from typing import TYPE_CHECKING, Any

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence import turbulence_states as states
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.grid import simple
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.fixtures.datatest import backend


if TYPE_CHECKING:
    import gt4py.next as gtx
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


NUM_LEVELS = 8

#: The only fields of the whole granule interface that are declared 'REAL(KIND=vp)' in the
#: Fortran (turb_diffusion.f90:612). They are produced by the mixed-precision dycore diffusion.
VP_FIELDS = frozenset({"hdef2", "hdiv", "dwdx", "dwdy"})


#: Number of passive tracers the tracer tuples are exercised with. In ICON this is 'ndtr', a
#: runtime count that depends on the microphysics and ART configuration.
NUM_TRACERS = 3


#: What the `TileField` alias of 'turbulence_states' stands for. Spelled out rather than
#: imported so that a change to the alias has to be made twice; `test_tile_field_alias` pins
#: the two together.
TILE_FIELD_ANNOTATION = "tuple[fa.CellField[ta.wpfloat], ...]"


class Shape(enum.Enum):
    """The shape a state field is declared with in the Fortran it ports."""

    #: A horizontal field, 'DIMENSION(:)' in the Fortran: one value per cell, no vertical axis.
    SURFACE = enum.auto()
    #: A column on the main ("full") levels: 'nlev' entries.
    FULL = enum.auto()
    #: A column on the boundary ("half") levels: 'nlev + 1' entries.
    HALF = enum.auto()
    #: A tuple of `SURFACE` fields, one per surface tile -- the third index of ICON's '_t' arrays.
    PER_TILE = enum.auto()
    #: A tuple of `FULL` columns, one per diffused passive tracer -- an entry of 'ptr(:)'.
    PER_TRACER = enum.auto()


#: The axis structure of each `Shape`, at the resolution a type annotation can express. 'FULL'
#: and 'HALF' are one annotation, which is why the vertical extent is checked separately.
_AXES = {
    Shape.SURFACE: "cell",
    Shape.FULL: "cell, k",
    Shape.HALF: "cell, k",
    Shape.PER_TILE: "tuple of cell",
    Shape.PER_TRACER: "tuple of cell, k",
}


#: What the Fortran declares for every field of every state container, read off the dummy-argument
#: lists of 'turbdiff' (turb_diffusion.f90:488-668), 'turbtran' (turb_transfer.f90:224ff) and
#: 'vertdiff' (turb_vertdiff.f90:114ff), and written down independently of 'turbulence_states'.
#: The tests below assert the module against this table.
DECLARED_SHAPES: dict[type, dict[str, Shape]] = {
    states.TurbulenceMetricState: {
        "hhl": Shape.HALF,  # 'height of model half(=boundary)-levels', :489
        "dp0": Shape.FULL,  # 'pressure thickness of layer', :490
        "l_hori": Shape.SURFACE,
        "trop_mask": Shape.SURFACE,
        "innertrop_mask": Shape.SURFACE,
    },
    states.TurbulenceInputState: {
        "u": Shape.FULL,
        "v": Shape.FULL,
        "w": Shape.HALF,
        "t": Shape.FULL,
        "qv": Shape.FULL,
        "qc": Shape.FULL,
        "prs": Shape.FULL,
        "rhoh": Shape.FULL,
        "epr": Shape.FULL,
        "tke": Shape.HALF,  # 'DIMENSION(nvec,ke1,ntim)', :576
        "tracers": Shape.PER_TRACER,
        "ut_sso": Shape.FULL,
        "vt_sso": Shape.FULL,
        "tket_conv": Shape.HALF,
        "hdef2": Shape.HALF,
        "hdiv": Shape.HALF,
        "dwdx": Shape.HALF,
        "dwdy": Shape.HALF,
    },
    states.TurbulenceSurfaceState: {
        "t_g": Shape.SURFACE,
        "qv_s": Shape.SURFACE,
        "ps": Shape.SURFACE,
        "fr_land": Shape.SURFACE,
        "l_lake": Shape.SURFACE,
        "l_sice": Shape.SURFACE,
        "l_pat": Shape.SURFACE,
        "urb_isa": Shape.SURFACE,
        "rlamh_fac": Shape.SURFACE,
        "z0_waves": Shape.SURFACE,
    },
    states.TurbulenceDiagnosticState: {
        "gz0": Shape.SURFACE,
        "tcm": Shape.SURFACE,
        "tch": Shape.SURFACE,
        "tvm": Shape.SURFACE,
        "tvh": Shape.SURFACE,
        "tfm": Shape.SURFACE,
        "tfh": Shape.SURFACE,
        "tfv": Shape.SURFACE,
        "tkred_sfc": Shape.SURFACE,
        "tkred_sfc_h": Shape.SURFACE,
        "shfl_s": Shape.SURFACE,
        "qvfl_s": Shape.SURFACE,
        "umfl_s": Shape.SURFACE,
        "vmfl_s": Shape.SURFACE,
        "t_2m": Shape.SURFACE,
        "qv_2m": Shape.SURFACE,
        "td_2m": Shape.SURFACE,
        "rh_2m": Shape.SURFACE,
        "u_10m": Shape.SURFACE,
        "v_10m": Shape.SURFACE,
        "tkvm": Shape.HALF,  # :586
        "tkvh": Shape.HALF,  # :587
        "tprn": Shape.HALF,  # 'turbulent Prandtl-number (at half-levels)', :591
        "rcld": Shape.HALF,  # main levels plus the lower boundary, so 'nlev + 1' entries, :603
        "rhon": Shape.HALF,  # 'total density of air (at half levels)', :538
        "edr": Shape.HALF,  # :652
        "tur_len_scale": Shape.HALF,  # :653
        # Not a dummy argument of its own: the Fortran writes the updated turbulent
        # velocity back into 'tke(:,:,ntur)', which is the same storage as the input
        # 'tke(:,:,nvor)' while 'ntim == 1'. ADR-0001 makes it a field of its own here.
        "updated_tke": Shape.HALF,  # 'tke', :576
    },
    states.TurbulenceTendencyState: {
        "ddt_u": Shape.FULL,
        "ddt_v": Shape.FULL,
        "ddt_t": Shape.FULL,
        "ddt_qv": Shape.FULL,
        "ddt_qc": Shape.FULL,
        "ddt_tke": Shape.HALF,  # 'diffusion tendency of q=SQRT(2*TKE)', :634
        "ddt_tracers": Shape.PER_TRACER,
        "tket_hshr": Shape.HALF,  # :659
    },
    states.TurbulenceTileState: {
        "gz0_t": Shape.PER_TILE,
        "sai_t": Shape.PER_TILE,
        "t_g_t": Shape.PER_TILE,
        "qv_s_t": Shape.PER_TILE,
        "frac_t": Shape.PER_TILE,
        "tvs_s_t": Shape.PER_TILE,
        "tkvm_s_t": Shape.PER_TILE,
        "tkvh_s_t": Shape.PER_TILE,
        "rcld_s_t": Shape.PER_TILE,
        "tkr_t": Shape.PER_TILE,
    },
}

STATE_CLASSES = tuple(DECLARED_SHAPES)


@pytest.fixture
def grid(backend: gtx_typing.Backend | None) -> base_grid.Grid:
    return simple.simple_grid(allocator=backend, num_levels=NUM_LEVELS)


def _column(grid: base_grid.Grid, allocator: Any, dtype: Any, levels: int) -> gtx.Field:
    return data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, extend={dims.KDim: levels}, dtype=dtype, allocator=allocator
    )


def _surface(grid: base_grid.Grid, allocator: Any, dtype: Any = None) -> gtx.Field:
    return data_alloc.zero_field(grid, dims.CellDim, dtype=dtype or ta.wpfloat, allocator=allocator)


def _annotation(field: dataclasses.Field) -> str:
    """The source annotation, with the `TileField` alias resolved to what it stands for."""
    return TILE_FIELD_ANNOTATION if field.type == "TileField" else field.type


def _annotated_axes(annotation: str) -> str:
    """The axis structure the module's annotation declares, in the vocabulary of `_AXES`."""
    inner = annotation[len("tuple[") :] if annotation.startswith("tuple[") else annotation
    axes = "cell, k" if "CellKField" in inner else "cell"
    return f"tuple of {axes}" if annotation.startswith("tuple[") else axes


def _doc_comment(state_class: type, name: str) -> str:
    """The '#:' block immediately above the declaration of a state field."""
    lines = inspect.getsource(state_class).splitlines()
    (index,) = [i for i, line in enumerate(lines) if line.startswith(f"    {name}:")]
    comment = []
    while index > 0 and lines[index - 1].lstrip().startswith("#:"):
        index -= 1
        comment.insert(0, lines[index].lstrip()[2:].strip())
    return " ".join(comment)


def _allocate(state_class: type, grid: base_grid.Grid, allocator: Any) -> Any:
    """Allocate a zero field per field of `state_class`, shaped as `DECLARED_SHAPES` says."""
    values = {}
    for field in dataclasses.fields(state_class):
        shape = DECLARED_SHAPES[state_class][field.name]
        dtype = bool if "bool" in _annotation(field) else ta.wpfloat
        if shape is Shape.PER_TILE:
            values[field.name] = tuple(
                _surface(grid, allocator, dtype) for _ in range(states.NUM_TILES)
            )
        elif shape is Shape.PER_TRACER:
            values[field.name] = tuple(
                _column(grid, allocator, dtype, 0) for _ in range(NUM_TRACERS)
            )
        elif shape is Shape.SURFACE:
            values[field.name] = _surface(grid, allocator, dtype)
        else:
            values[field.name] = _column(grid, allocator, dtype, 1 if shape is Shape.HALF else 0)
    return state_class(**values)


# --- the declared shapes ----------------------------------------------------------------------


@pytest.mark.parametrize("state_class", STATE_CLASSES)
def test_declared_shapes_cover_exactly_the_fields_of_the_container(state_class: type) -> None:
    """A field added to a container without a Fortran declaration behind it fails here."""
    declared = set(DECLARED_SHAPES[state_class])
    assert declared == {field.name for field in dataclasses.fields(state_class)}


@pytest.mark.parametrize(
    "state_class, name",
    [(cls, name) for cls, shapes in DECLARED_SHAPES.items() for name in shapes],
)
def test_field_annotation_carries_the_declared_axes(state_class: type, name: str) -> None:
    """The annotation must have the axes the Fortran declaration gives the field.

    This is the assertion that catches a 'fa.CellKField' silently demoted to 'fa.CellField': the
    expectation comes from `DECLARED_SHAPES`, not from the annotation being checked.
    """
    (field,) = [f for f in dataclasses.fields(state_class) if f.name == name]
    assert _annotated_axes(_annotation(field)) == _AXES[DECLARED_SHAPES[state_class][name]], name


@pytest.mark.parametrize(
    "state_class, name",
    [
        (cls, name)
        for cls, shapes in DECLARED_SHAPES.items()
        for name, shape in shapes.items()
        if shape in (Shape.FULL, Shape.HALF, Shape.PER_TRACER)
    ],
)
def test_vertical_extent_agrees_with_the_documented_one(state_class: type, name: str) -> None:
    """No annotation distinguishes 'nlev' from 'nlev + 1', so the doc comment is the second record.

    Both are written from the same Fortran declaration but by hand and separately, so a table
    entry that drifts from the module's own description of the field is caught here.
    """
    shape = DECLARED_SHAPES[state_class][name]
    comment = _doc_comment(state_class, name).lower()
    if shape is Shape.HALF:
        assert "half level" in comment or "nlev + 1" in comment, name
    else:
        assert "full level" in comment, name


@pytest.mark.parametrize("state_class", STATE_CLASSES)
def test_allocated_fields_have_the_declared_dimensions(
    state_class: type, grid: base_grid.Grid, backend: gtx_typing.Backend | None
) -> None:
    """The container accepts fields shaped as declared, and keeps them unchanged."""
    state = _allocate(state_class, grid, backend)
    for field in dataclasses.fields(state):
        shape = DECLARED_SHAPES[state_class][field.name]
        value = getattr(state, field.name)
        entries = value if isinstance(value, tuple) else (value,)
        for entry in entries:
            if shape is Shape.SURFACE or shape is Shape.PER_TILE:
                assert entry.domain.dims == (dims.CellDim,), field.name
            else:
                assert entry.domain.dims == (dims.CellDim, dims.KDim), field.name
                expected = NUM_LEVELS + 1 if shape is Shape.HALF else NUM_LEVELS
                assert entry.shape[1] == expected, field.name


# --- the dataclass contract -------------------------------------------------------------------


@pytest.mark.parametrize("state_class", STATE_CLASSES)
def test_state_is_frozen(
    state_class: type, grid: base_grid.Grid, backend: gtx_typing.Backend | None
) -> None:
    state = _allocate(state_class, grid, backend)
    first = dataclasses.fields(state)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(state, first, None)


def test_tile_field_alias_is_a_tuple_of_cell_fields() -> None:
    """The tile axis is a Python tuple of plain cell fields, not a field dimension (spec 7.3)."""
    element, ellipsis = typing.get_args(states.TileField)
    assert ellipsis is Ellipsis
    assert element == fa.CellField[ta.wpfloat]


# --- precision ----------------------------------------------------------------------------------


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


# --- what the interface deliberately leaves out -------------------------------------------------


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


# --- the tile axis ------------------------------------------------------------------------------


def test_tile_indices_address_the_water_tiles_of_a_full_tile_axis() -> None:
    """The constants a stencil selects the three special tiles with (mo_lnd_nwp_config.f90:236)."""
    assert states.NUM_TILES == states.NUM_LAND_TILES + states.NUM_WATER_TILES
    water = (states.OPEN_SEA_TILE, states.LAKE_TILE, states.SEA_ICE_TILE)
    assert water == (6, 7, 8)
    assert len(set(water)) == states.NUM_WATER_TILES
    assert all(states.NUM_LAND_TILES <= index < states.NUM_TILES for index in water)


def test_tile_state_reports_the_number_of_tiles_it_carries(
    grid: base_grid.Grid, backend: gtx_typing.Backend | None
) -> None:
    state = _allocate(states.TurbulenceTileState, grid, backend)
    assert state.num_tiles == states.NUM_TILES


def test_tile_state_accepts_the_degenerate_untiled_shape(
    grid: base_grid.Grid, backend: gtx_typing.Backend | None
) -> None:
    """'ntiles_lnd == 1' forces 'lsnowtile = .FALSE.' and 'ntiles_water = 0' (spec 7.2)."""
    single = {
        field.name: (_surface(grid, backend),)
        for field in dataclasses.fields(states.TurbulenceTileState)
    }
    state = states.TurbulenceTileState(**single)
    assert state.num_tiles == 1


@pytest.mark.parametrize("count", [2, 5, 8, 10])
def test_tile_state_rejects_a_tile_count_no_ICON_configuration_produces(
    count: int, grid: base_grid.Grid, backend: gtx_typing.Backend | None
) -> None:
    values = {
        field.name: tuple(_surface(grid, backend) for _ in range(count))
        for field in dataclasses.fields(states.TurbulenceTileState)
    }
    with pytest.raises(ValueError, match="Invalid number of tiles"):
        states.TurbulenceTileState(**values)


def test_tile_state_rejects_a_ragged_tile_axis(
    grid: base_grid.Grid, backend: gtx_typing.Backend | None
) -> None:
    values = {
        field.name: tuple(_surface(grid, backend) for _ in range(states.NUM_TILES))
        for field in dataclasses.fields(states.TurbulenceTileState)
    }
    values["frac_t"] = values["frac_t"][:-1]
    with pytest.raises(ValueError, match="Ragged tile axis"):
        states.TurbulenceTileState(**values)
