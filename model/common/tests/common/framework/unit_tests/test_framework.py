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

from icon4py.model.common import (
    dimension as dims,
    field,
    field_type_aliases as fa,
    framework as fw,
    quantity,
    type_alias as ta,
    utils as common_utils,
)
from icon4py.model.common.grid import base as base_grid

from tests.common.fixtures import grid
from tests.common.unit_tests.utils import Pressure, Salt, TendencyOfPressure, Vt


class Column(fw.State):
    pressure: field.Field[Pressure]
    salt: field.Field[Salt]
    dtime: float


class Tracers(fw.State):
    pressure: field.Field[Pressure]
    salt: field.Field[Salt] | None = None
    vt: field.Field[Vt] | None = None


class Budget(fw.State):
    pressure: field.Field[Pressure]
    tend_pressure: field.Field[TendencyOfPressure]
    salt: field.Field[Salt] | None = None


class Halve(fw.Component["Halve.Input", "Halve.Output"]):
    class Input(fw.State):
        pressure: field.Field[Pressure]

    class Output(fw.State):
        pressure: field.Field[Pressure]

    def run(self, input: Input, output: Output) -> Output:
        np.asarray(output.pressure.ndarray)[...] = 0.5 * np.asarray(input.pressure.ndarray)
        return output


class Double:
    """A component in structure only: it does not inherit from `fw.Component`."""

    Input = Halve.Input
    Output = Halve.Output

    def run(self, inputs: Halve.Input, outputs: Halve.Output) -> Halve.Output:
        np.asarray(outputs.pressure.ndarray)[...] = 2.0 * np.asarray(inputs.pressure.ndarray)
        return outputs


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


def test_allocate_takes_the_plain_leaves_by_keyword(grid: base_grid.Grid) -> None:
    column = fw.allocate(Column, grid, None, dtime=60.0)
    assert column.dtime == 60.0
    assert list(fw.fields(column)) == ["pressure", "salt"]
    # a plain leaf without a default must be given
    with pytest.raises(TypeError, match="dtime"):
        fw.allocate(Column, grid, None)


def test_fields_of_a_kind_are_the_leaves_whose_quantity_is_one(grid: base_grid.Grid) -> None:
    budget = fw.allocate(Budget, grid, None)
    assert list(fw.fields(budget, kind=quantity.Tendency)) == ["tend_pressure"]
    assert list(fw.fields(budget, kind=Pressure)) == ["pressure"]


def test_copy_of_a_kind_writes_and_compares_only_those_leaves(grid: base_grid.Grid) -> None:
    src = fw.allocate(Budget, grid, None, fill=lambda name, _: 1.0)
    # `salt` is present on one side only, and not a tendency: it is not compared
    dst = fw.allocate(Budget, grid, None, only=("pressure", "tend_pressure"))

    fw.copy(src, dst, kind=quantity.Tendency)

    np.testing.assert_array_equal(dst.tend_pressure.asnumpy(), 1.0)
    np.testing.assert_array_equal(dst.pressure.asnumpy(), 0.0)


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
        pressure: field.Field[Pressure]
        salt: field.Field[Salt]

    fields = fw.allocate(Fields, grid, allocator=None, fill=lambda name, shape: float(len(name)))
    assert fields.pressure.ndarray.shape == (grid.num_cells, grid.num_levels)
    assert fields.salt.ndarray.shape == (grid.num_cells,)
    assert np.all(np.asarray(fields.salt.ndarray) == 4.0)


def test_allocate_fill_is_written_on_the_leaf_without_a_host_conversion(
    grid: base_grid.Grid, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A device buffer refuses `np.asarray`; the fill value is assigned on the buffer itself."""

    class Fields(fw.State):
        salt: field.Field[Salt]

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
    assert Halve().run(Halve.Input(pressure=pressure), output=given) is given
    assert np.all(np.asarray(given.pressure.ndarray) == 4.0)
    # in place: the caller passes the input's buffers as the output
    Halve().run(Halve.Input(pressure=pressure), output=Halve.Output(pressure=pressure))
    assert np.all(np.asarray(pressure.ndarray) == 4.0)


def test_empty_is_the_output_of_a_component_that_writes_no_field(grid: base_grid.Grid) -> None:
    class Sink(fw.Component[Halve.Input, fw.Empty]):
        Input = Halve.Input
        Output = fw.Empty

        def run(self, input: Halve.Input, output: fw.Empty | None = None) -> fw.Empty:
            return output if output is not None else fw.Empty()

    assert list(fw.fields(fw.Empty())) == []
    input = Halve.Input(pressure=fw.zeros(Pressure, grid, allocator=None))
    empty = fw.Empty()
    assert Sink().run(input, empty) is empty
    assert isinstance(Sink().run(input), fw.Empty)


def test_a_component_that_does_not_inherit_satisfies_the_protocol(grid: base_grid.Grid) -> None:
    # What mypy checks: `Double` subclasses nothing; the annotated assignment is the
    # structural conformance check (Input, Output and run, by member name and type)
    pressure = fw.zeros(Pressure, grid, allocator=None)
    np.asarray(pressure.ndarray)[...] = 8.0
    component: fw.Component[Halve.Input, Halve.Output] = Double()
    given = fw.allocate(Halve.Output, grid, allocator=None)
    assert component.run(Halve.Input(pressure=pressure), outputs=given) is given
    assert np.all(np.asarray(given.pressure.ndarray) == 16.0)


def test_pairs_swap(grid: base_grid.Grid) -> None:
    a = fw.allocate(Halve.Output, grid, allocator=None)
    b = fw.allocate(Halve.Output, grid, allocator=None)
    pair = common_utils.TimeStepPair(a, b)
    pair.swap()
    assert pair.current is b and pair.next is a


# What mypy checks: a quantity mismatch is a type error. Each ignore below is
# required (an unused one is reported), so type-checking this module is the test.
def static_checks(pressure: field.Field[Pressure], salt: field.Field[Salt]) -> None:
    wrong: field.Field[Pressure] = salt  # type: ignore[assignment]
    Halve.Input(pressure=salt)  # type: ignore[arg-type]
    Halve.Input(pressure=pressure)
    del wrong
