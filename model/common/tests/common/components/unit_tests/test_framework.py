# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.components import framework as fw
from icon4py.model.common.grid import base as base_grid, simple


CELL_K = (dims.CellDim, dims.KDim)


class Pressure(fw.Quantity, dims=CELL_K, standard_name="air_pressure", units="Pa"): ...


class Salt(fw.Quantity, dims=(dims.CellDim,), units="1"): ...


class Vt(fw.Quantity, dims=(dims.EdgeDim, dims.KDim), units="m s-1", precision="vp"): ...


class Column(fw.State):
    pressure: fw.Field[Pressure]
    salt: fw.Field[Salt]
    dtime: float


class Tracers(fw.State):
    pressure: fw.Field[Pressure]
    salt: fw.Field[Salt] | None = None
    vt: fw.Field[Vt] | None = None


class Halve(fw.Component):
    class Input(fw.State):
        pressure: fw.Field[Pressure]

    class Output(fw.State):
        pressure: fw.Field[Pressure]

    def run(self, inputs: Input, out: Output | None = None) -> Output:
        out = self.buffers(out)
        np.asarray(out.pressure.data.ndarray)[...] = 0.5 * np.asarray(inputs.pressure.data.ndarray)
        return out


@pytest.fixture
def grid() -> base_grid.Grid:
    return simple.simple_grid()


def test_quantity_is_a_tag_with_metadata() -> None:
    assert Pressure.dims == CELL_K
    assert Pressure.standard_name == "air_pressure"
    assert Pressure.units == "Pa"
    assert Pressure.precision == "wp"
    assert Salt.standard_name is None
    assert Vt.precision == "vp"
    with pytest.raises(TypeError):
        Pressure()


def test_field_keeps_its_quantity(grid: base_grid.Grid) -> None:
    pressure = fw.zeros(Pressure, grid, allocator=None)
    assert pressure.quantity is Pressure
    assert pressure.data.ndarray.shape == (grid.num_cells, grid.num_levels)
    assert pressure.data.domain.dims == CELL_K


def test_zeros_resolves_the_precision_at_allocation(
    grid: base_grid.Grid, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ta, "vpfloat", gtx.float32)
    assert fw.zeros(Vt, grid, allocator=None).data.ndarray.dtype == np.float32
    assert fw.zeros(Pressure, grid, allocator=None).data.ndarray.dtype == np.float64


def test_state_declarations_list_the_field_leaves_only(grid: base_grid.Grid) -> None:
    assert [(d.name, d.quantity, d.optional) for d in Column.declarations()] == [
        ("pressure", Pressure, False),
        ("salt", Salt, False),
    ]
    column = Column(
        pressure=fw.zeros(Pressure, grid, allocator=None),
        salt=fw.zeros(Salt, grid, allocator=None),
        dtime=1.0,
    )
    assert [d.name for d, _ in column.leaves()] == ["pressure", "salt"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        column.dtime = 2.0  # type: ignore[misc]


def test_optional_leaves_are_declared_and_skipped_when_absent(grid: base_grid.Grid) -> None:
    assert [(d.name, d.optional) for d in Tracers.declarations()] == [
        ("pressure", False),
        ("salt", True),
        ("vt", True),
    ]
    tracers = fw.allocate(Tracers, grid, allocator=None, only=("vt",))
    assert tracers.salt is None
    assert tracers.vt is not None
    assert [d.name for d, _ in tracers.leaves()] == ["pressure", "vt"]
    assert [d.name for d, _ in fw.allocate(Tracers, grid, allocator=None).leaves()] == [
        "pressure",
        "salt",
        "vt",
    ]


def test_allocate_follows_the_declared_dims(grid: base_grid.Grid) -> None:
    class Fields(fw.State):
        pressure: fw.Field[Pressure]
        salt: fw.Field[Salt]

    fields = fw.allocate(Fields, grid, allocator=None, fill=lambda name, shape: float(len(name)))
    assert fields.pressure.data.ndarray.shape == (grid.num_cells, grid.num_levels)
    assert fields.salt.data.ndarray.shape == (grid.num_cells,)
    assert np.all(np.asarray(fields.salt.data.ndarray) == 4.0)


def test_allocate_fill_is_written_on_the_leaf_without_a_host_conversion(
    grid: base_grid.Grid, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A device buffer refuses `np.asarray`; the fill value is assigned on the buffer itself."""

    class Fields(fw.State):
        salt: fw.Field[Salt]

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise TypeError("Implicit conversion to a NumPy array is not allowed")

    def fill(name: str, shape: tuple[int, ...]) -> float:
        monkeypatch.setattr(np, "asarray", refuse)
        return 4.0

    fields = fw.allocate(Fields, grid, allocator=None, fill=fill)
    monkeypatch.undo()
    assert np.all(np.asarray(fields.salt.data.ndarray) == 4.0)


def test_component_writes_its_own_buffers_or_the_callers(grid: base_grid.Grid) -> None:
    halve = Halve(grid, allocator=None)
    pressure = fw.zeros(Pressure, grid, allocator=None)
    np.asarray(pressure.data.ndarray)[...] = 8.0
    own = halve.run(Halve.Input(pressure=pressure))
    assert own is halve.output and own is halve.run(Halve.Input(pressure=pressure))
    assert np.all(np.asarray(own.pressure.data.ndarray) == 4.0)
    given = Halve.Output(pressure=pressure)
    assert halve.run(Halve.Input(pressure=pressure), out=given) is given
    assert np.all(np.asarray(pressure.data.ndarray) == 4.0)


def test_pairs_swap(grid: base_grid.Grid) -> None:
    a = fw.allocate(Halve.Output, grid, allocator=None)
    b = fw.allocate(Halve.Output, grid, allocator=None)
    pair = fw.TimeStepPair(a, b)
    pair.swap()
    assert pair.current is b and pair.next is a


# What mypy and pyright check: a quantity mismatch is a type error. Each ignore below is
# required (an unused one is reported), so type-checking this module is the test.
def static_checks(pressure: fw.Field[Pressure], salt: fw.Field[Salt]) -> None:
    wrong: fw.Field[Pressure] = salt  # type: ignore[assignment]
    Halve.Input(pressure=salt)  # type: ignore[arg-type]
    Halve.Input(pressure=pressure)
    del wrong
