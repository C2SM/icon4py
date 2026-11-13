# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Base classes for the contract tests of typed components: the states they take and return,
not their numerics.

A suite subclasses `ComponentTest`, provides a `component` fixture and imports the `grid`
fixture of `icon4py.model.testing.fixtures`. It inherits the tests every `fw.Component` must
pass, and adds its own as ordinary methods. A suite of a component that has a `collect_input`
also subclasses `CollectInputTest`.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import framework as fw
from icon4py.model.common.grid import base as base_grid
from icon4py.model.common.utils import data_allocation as data_alloc


def random_dataclass[T](cls: type[T], grid: base_grid.Grid) -> T:
    """
    A dataclass of grid fields, each one random.
    Used, e.g., for the metric, interpolation and geometry states needed to
    build a component without savepoints in unit_tests.

    Fields, tuples of fields and optional fields are filled, every other member keeps its
    default. The values are positive, so that a scheme dividing by them stays finite.
    """

    def value(hint: Any) -> Any:
        origin = typing.get_origin(hint)
        if origin in (types.UnionType, typing.Union):
            (hint,) = (arg for arg in typing.get_args(hint) if arg is not type(None))
            return value(hint)
        if origin is tuple:
            return tuple(value(arg) for arg in typing.get_args(hint))
        if origin is not gtx.Field:
            return None
        dims = typing.get_args(hint)[0]
        # `Field[Dims[...], ...]` or `Field[[...], ...]`
        dims = dims if isinstance(dims, list) else typing.get_args(dims)
        return data_alloc.random_field(grid, *dims, low=0.5, high=1.5)

    hints = typing.get_type_hints(cls)
    values = {field.name: value(hints[field.name]) for field in dataclasses.fields(cls)}  # type: ignore[arg-type]  # checked by the caller
    return cls(**{name: v for name, v in values.items() if v is not None})


class ComponentTest:
    """
    The `run(input, output) -> output` contract of `fw.Component`.

    `component` is the subclass's fixture, and it can build the states needed
    by its constructor with `random_dataclass` defined above. `input` and `output`
    default to random states on `grid`, the testing fixture, and can be overridden.
    """

    @pytest.fixture
    def input(self, component: fw.Component, grid: base_grid.Grid) -> fw.State:
        # random, so that a misplaced read shows; positive, so that a scheme dividing by them stays finite
        return fw.allocate(
            component.Input,
            grid,
            None,
            fill=lambda _, dims: data_alloc.random_field(grid, *dims, low=0.5, high=1.5).ndarray,
        )

    @pytest.fixture
    def output(self, component: fw.Component, grid: base_grid.Grid) -> fw.State:
        return fw.allocate(component.Output, grid, None)

    def test_run_returns_the_output_it_is_given(
        self, component: fw.Component, input: fw.State, output: fw.State
    ) -> None:
        assert component.run(input, output) is output

    def test_run_leaves_the_input_untouched(
        self,
        component: fw.Component,
        grid: base_grid.Grid,
        input: fw.State,
        output: fw.State,
    ) -> None:
        before = fw.copy(input, fw.allocate(type(input), grid, None))

        component.run(input, output)

        for name, field in fw.fields(input).items():
            np.testing.assert_array_equal(
                field.asnumpy(), getattr(before, name).asnumpy(), err_msg=name
            )


# NOTE: temporary, replaced in part 02 (collect): goes with `collect_input`, once `fw.collect`
# picks the input by quantity.
class CollectInputTest:
    """
    `collect_input(state)` builds the component's input from views of `state`, no copies.

    The subclass sets `collect_input = staticmethod(...)` and provides a `state` fixture.
    """

    @staticmethod
    def collect_input(state: Any) -> fw.State:
        raise NotImplementedError

    def test_collect_input_views_the_state(self, state: fw.State) -> None:
        input = self.collect_input(state)

        leaves = fw.fields(state).values()
        for name, field in fw.fields(input).items():
            assert any(field is leaf for leaf in leaves), name
