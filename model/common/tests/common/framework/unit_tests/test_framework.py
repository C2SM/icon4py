# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import field, framework as fw, quantity, type_alias as ta
from icon4py.model.common.grid import base as base_grid
from icon4py.model.common.utils import PredictorCorrectorPair

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

    def run(self, input: Halve.Input, output: Halve.Output) -> Halve.Output:
        np.asarray(output.pressure.ndarray)[...] = 2.0 * np.asarray(input.pressure.ndarray)
        return output


def test_zeros_resolves_the_precision_at_allocation(
    grid: base_grid.Grid, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ta, "vpfloat", gtx.float32)
    assert fw.zeros(Vt, grid, allocator=None).ndarray.dtype == np.float32
    assert fw.zeros(Pressure, grid, allocator=None).ndarray.dtype == np.float64


def test_allocate_takes_the_plain_leaves_by_keyword(grid: base_grid.Grid) -> None:
    column = fw.allocate(Column, grid, None, dtime=60.0)
    assert column.dtime == 60.0
    assert list(fw.fields(column)) == ["pressure", "salt"]
    # a plain leaf without a default must be given
    with pytest.raises(TypeError, match="dtime"):
        fw.allocate(Column, grid, None)


def test_fields_of_a_kind_and_without_one(grid: base_grid.Grid) -> None:
    budget = fw.allocate(Budget, grid, None)
    assert list(fw.fields(budget, kind=quantity.Tendency)) == ["tend_pressure"]
    assert list(fw.fields(budget, kind=Pressure)) == ["pressure"]
    assert list(fw.fields(budget, exclude=quantity.Tendency)) == ["pressure", "salt"]


def test_copy_writes_the_destination_in_place(grid: base_grid.Grid) -> None:
    src = fw.allocate(Tracers, grid, None, only=("vt",), fill=lambda name, _: 2.0)
    dst = fw.allocate(Tracers, grid, None, only=("vt",))

    assert fw.copy(src, dst) is dst

    assert dst.salt is None and dst.vt is not src.vt
    np.testing.assert_array_equal(dst.vt.asnumpy(), 2.0)  # type: ignore[union-attr]
    # a leaf present on one side only is a mismatch
    with pytest.raises(ValueError, match="present leaves must match"):
        fw.copy(src, fw.allocate(Tracers, grid, None))


def test_copy_of_a_kind_writes_and_compares_only_those_leaves(grid: base_grid.Grid) -> None:
    src = fw.allocate(Budget, grid, None, fill=lambda name, _: 1.0)
    # `salt` is present on one side only, and not a tendency: it is not compared
    dst = fw.allocate(Budget, grid, None, only=("pressure", "tend_pressure"))

    fw.copy(src, dst, kind=quantity.Tendency)

    np.testing.assert_array_equal(dst.tend_pressure.asnumpy(), 1.0)
    np.testing.assert_array_equal(dst.pressure.asnumpy(), 0.0)


def test_optional_leaves_left_none_are_not_fields(grid: base_grid.Grid) -> None:
    tracers = fw.allocate(Tracers, grid, allocator=None, only=("vt",))
    assert tracers.salt is None
    assert list(fw.fields(tracers)) == ["pressure", "vt"]
    assert fw.fields(tracers)["vt"] is tracers.vt
    assert list(fw.fields(fw.allocate(Tracers, grid, allocator=None))) == [
        "pressure",
        "salt",
        "vt",
    ]


def test_allocate_follows_the_declared_dims(grid: base_grid.Grid) -> None:
    class Fields(fw.State):
        pressure: field.Field[Pressure]
        salt: field.Field[Salt]
        salt_pc: PredictorCorrectorPair[field.Field[Salt]]

    fields = fw.allocate(Fields, grid, allocator=None, fill=lambda name, shape: float(len(name)))
    assert fields.pressure.ndarray.shape == (grid.num_cells, grid.num_levels)
    assert fields.salt.ndarray.shape == (grid.num_cells,)
    assert np.all(np.asarray(fields.salt.ndarray) == 4.0)
    # a pair gets one field per member, which `fields` skips
    assert fields.salt_pc.predictor is not fields.salt_pc.corrector
    assert np.all(np.asarray(fields.salt_pc.corrector.ndarray) == 7.0)
    assert list(fw.fields(fields)) == ["pressure", "salt"]


# What mypy checks: a quantity mismatch is a type error. Each ignore below is
# required (an unused one is reported), so type-checking this module is the test.
def static_checks(pressure: field.Field[Pressure], salt: field.Field[Salt]) -> None:
    wrong: field.Field[Pressure] = salt  # type: ignore[assignment]
    Halve.Input(pressure=salt)  # type: ignore[arg-type]
    Halve.Input(pressure=pressure)
    # a component that does not inherit from `fw.Component` satisfies the protocol
    component: fw.Component[Halve.Input, Halve.Output] = Double()
    del wrong, component
