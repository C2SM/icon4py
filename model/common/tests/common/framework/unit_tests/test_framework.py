# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import dataclasses
import pickle
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    framework as fw,
    type_alias as ta,
)
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

    def run(self, inputs: Input, outputs: Output) -> Output:
        np.asarray(outputs.pressure.ndarray)[...] = 0.5 * np.asarray(inputs.pressure.ndarray)
        return outputs


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
    assert pressure.ndarray.shape == (grid.num_cells, grid.num_levels)
    assert pressure.domain.dims == CELL_K


def test_field_is_the_gt4py_field_it_tags(grid: base_grid.Grid) -> None:
    data = gtx.zeros({dims.CellDim: grid.num_cells, dims.KDim: grid.num_levels}, dtype=ta.wpfloat)
    pressure = fw.Field(Pressure, data)
    assert isinstance(pressure, gtx.Field)
    assert pressure.ndarray is data.ndarray and pressure.domain == data.domain
    retagged = fw.Field(Pressure, pressure)
    assert retagged.quantity is Pressure and retagged.ndarray is data.ndarray


def test_fields_derived_from_a_field_are_untagged(grid: base_grid.Grid) -> None:
    pressure = fw.zeros(Pressure, grid, allocator=None)
    assert not isinstance(pressure + 1.0, fw.Field)
    assert not isinstance(pressure[dims.CellDim(0) : dims.CellDim(2)], fw.Field)


@gtx.field_operator
def _double(field: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    return 2.0 * field


def test_a_program_takes_a_field_as_is(grid: base_grid.Grid) -> None:
    pressure = fw.zeros(Pressure, grid, allocator=None)
    doubled = fw.zeros(Pressure, grid, allocator=None)
    np.asarray(pressure.ndarray)[...] = 3.0
    _double(pressure, out=doubled, offset_provider={})
    assert np.all(np.asarray(doubled.ndarray) == 6.0)


def test_field_copies_keep_the_quantity(grid: base_grid.Grid) -> None:
    pressure = fw.zeros(Pressure, grid, allocator=None)
    for copied in (
        copy.copy(pressure),
        copy.deepcopy(pressure),
        pickle.loads(pickle.dumps(pressure)),
    ):
        assert isinstance(copied, fw.Field) and copied.quantity is Pressure


def test_zeros_resolves_the_precision_at_allocation(
    grid: base_grid.Grid, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ta, "vpfloat", gtx.float32)
    assert fw.zeros(Vt, grid, allocator=None).ndarray.dtype == np.float32
    assert fw.zeros(Pressure, grid, allocator=None).ndarray.dtype == np.float64


def test_state_fields_list_the_field_leaves_only(grid: base_grid.Grid) -> None:
    column = Column(
        pressure=fw.zeros(Pressure, grid, allocator=None),
        salt=fw.zeros(Salt, grid, allocator=None),
        dtime=1.0,
    )
    assert list(fw.fields(column)) == ["pressure", "salt"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        column.dtime = 2.0  # type: ignore[misc]


def test_state_maps_present_leaf_names_to_fields(grid: base_grid.Grid) -> None:
    tracers = fw.allocate(Tracers, grid, allocator=None, only=("vt",))
    assert list(fw.fields(tracers)) == ["pressure", "vt"]
    assert fw.fields(tracers)["vt"] is tracers.vt
    # an optional leaf left None is absent from the mapping
    assert "salt" not in fw.fields(tracers)
    assert fw.fields(tracers).get("salt") is None
    # states compare by identity and stay hashable, the eq=False dataclass choice
    same, other = tracers, fw.allocate(Tracers, grid, allocator=None, only=("vt",))
    assert tracers == same
    assert tracers != other
    assert hash(tracers) == hash(tracers)
    assert {tracers: 1}[tracers] == 1


def test_optional_leaves_are_skipped_when_absent(grid: base_grid.Grid) -> None:
    tracers = fw.allocate(Tracers, grid, allocator=None, only=("vt",))
    assert tracers.salt is None
    assert tracers.vt is not None
    assert list(fw.fields(tracers)) == ["pressure", "vt"]
    assert list(fw.fields(fw.allocate(Tracers, grid, allocator=None))) == [
        "pressure",
        "salt",
        "vt",
    ]


def test_allocate_follows_the_declared_dims(grid: base_grid.Grid) -> None:
    class Fields(fw.State):
        pressure: fw.Field[Pressure]
        salt: fw.Field[Salt]

    fields = fw.allocate(Fields, grid, allocator=None, fill=lambda name, shape: float(len(name)))
    assert fields.pressure.ndarray.shape == (grid.num_cells, grid.num_levels)
    assert fields.salt.ndarray.shape == (grid.num_cells,)
    assert np.all(np.asarray(fields.salt.ndarray) == 4.0)


def test_allocate_fill_is_written_on_the_leaf_without_a_host_conversion(
    grid: base_grid.Grid, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A device buffer refuses `np.asarray`; the fill value is assigned on the buffer itself."""

    class Fields(fw.State):
        salt: fw.Field[Salt]

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise TypeError("Implicit conversion to a NumPy array is not allowed")

    def fill(name: str, field_dims: tuple[gtx.Dimension, ...]) -> float:
        monkeypatch.setattr(np, "asarray", refuse)
        return 4.0

    fields = fw.allocate(Fields, grid, allocator=None, fill=fill)
    monkeypatch.undo()
    assert np.all(np.asarray(fields.salt.ndarray) == 4.0)


def test_component_writes_the_callers_buffers(grid: base_grid.Grid) -> None:
    pressure = fw.zeros(Pressure, grid, allocator=None)
    np.asarray(pressure.ndarray)[...] = 8.0
    given = fw.allocate(Halve.Output, grid, allocator=None)
    assert Halve().run(Halve.Input(pressure=pressure), outputs=given) is given
    assert np.all(np.asarray(given.pressure.ndarray) == 4.0)
    # in place: the caller passes the input's buffers as the output
    Halve().run(Halve.Input(pressure=pressure), outputs=Halve.Output(pressure=pressure))
    assert np.all(np.asarray(pressure.ndarray) == 4.0)


def test_empty_is_the_output_of_a_component_that_writes_no_field(grid: base_grid.Grid) -> None:
    class Sink(fw.Component):
        Input = Halve.Input
        Output = fw.Empty

        def run(self, inputs: Halve.Input, outputs: fw.Empty) -> fw.Empty:
            return outputs

    assert list(fw.fields(fw.Empty())) == []
    empty = fw.Empty()
    assert (
        Sink().run(Halve.Input(pressure=fw.zeros(Pressure, grid, allocator=None)), empty) is empty
    )


def test_pairs_swap(grid: base_grid.Grid) -> None:
    a = fw.allocate(Halve.Output, grid, allocator=None)
    b = fw.allocate(Halve.Output, grid, allocator=None)
    pair = fw.TimeStepPair(a, b)
    pair.swap()
    assert pair.current is b and pair.next is a


# What mypy checks: a quantity mismatch is a type error. Each ignore below is
# required (an unused one is reported), so type-checking this module is the test.
def static_checks(pressure: fw.Field[Pressure], salt: fw.Field[Salt]) -> None:
    wrong: fw.Field[Pressure] = salt  # type: ignore[assignment]
    Halve.Input(pressure=salt)  # type: ignore[arg-type]
    Halve.Input(pressure=pressure)
    del wrong
