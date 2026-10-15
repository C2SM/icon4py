# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import types
import typing
from collections.abc import Callable, Collection, Iterator
from typing import Any, ClassVar, Literal, dataclass_transform

import gt4py.next as gtx

from icon4py.model.common import type_alias as ta
from icon4py.model.common.utils import PredictorCorrectorPair, TimeStepPair, data_allocation


if typing.TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


__all__ = [
    "Component",
    "Decl",
    "Empty",
    "Field",
    "PredictorCorrectorPair",
    "Quantity",
    "State",
    "Tendency",
    "TimeStepPair",
    "allocate",
    "copy",
    "zeros",
]


# ------------------------------------------------------------------------------
# Quantity: a type-level tag, never instantiated
# ------------------------------------------------------------------------------
class Quantity:
    """
    A type-level tag, one subclass per quantity at one place on the grid, never instantiated.

    The metadata sits on the class; a keyword not given is inherited from the base (a marker
    base like `Tendency` gives none).
    """

    dims: ClassVar[tuple[gtx.Dimension, ...]]
    standard_name: ClassVar[str | None] = None
    units: ClassVar[str]
    long_name: ClassVar[str | None] = None
    precision: ClassVar[Literal["wp", "vp"]] = "wp"

    def __init_subclass__(
        cls,
        *,
        dims: tuple[gtx.Dimension, ...] | None = None,
        standard_name: str | None = None,
        units: str | None = None,
        long_name: str | None = None,
        precision: Literal["wp", "vp"] | None = None,
    ) -> None:
        super().__init_subclass__()
        given = {
            "dims": dims,
            "standard_name": standard_name,
            "units": units,
            "long_name": long_name,
            "precision": precision,
        }
        for name, value in given.items():
            if value is not None:
                setattr(cls, name, value)

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(f"'{cls.__name__}' is a type-level tag, not a value.")


class Tendency(Quantity):
    """Marker base of the tendencies."""


class Field[Q: Quantity]:
    """A gt4py field tagged by its quantity."""

    __slots__ = ("data", "quantity")

    def __init__(self, quantity: type[Q], data: gtx.Field[Any, Any]) -> None:
        self.quantity = quantity
        self.data = data

    def __repr__(self) -> str:
        return f"Field({self.quantity.__name__}, {self.data.ndarray.shape})"


def _dtype(quantity: type[Quantity]) -> Any:
    return ta.vpfloat if quantity.precision == "vp" else ta.wpfloat


def zeros[Q: Quantity](
    quantity: type[Q], grid: base_grid.Grid, allocator: gtx_typing.Allocator | None
) -> Field[Q]:
    """A zero field of the quantity's dimensions and precision."""
    return Field(
        quantity,
        data_allocation.zero_field(
            grid, *quantity.dims, dtype=_dtype(quantity), allocator=allocator
        ),
    )


# ------------------------------------------------------------------------------
# State: a frozen dataclass of typed leaves
# ------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Decl:
    """One `Field` leaf of a `State` class: its name, its quantity and whether it may be None."""

    name: str
    quantity: type[Quantity]
    optional: bool = False


@dataclass_transform(frozen_default=True, kw_only_default=True, eq_default=False)
class State:
    """
    A frozen, keyword-only dataclass of typed leaves.

    `Field` leaves are read back from the annotations by `declarations()`; plain leaves
    (floats, ints, datetimes) are ordinary fields and are not listed. A leaf annotated
    `Field[Q] | None` is optional: `allocate` may leave it None and `leaves()` skips it.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        dataclasses.dataclass(frozen=True, eq=False, kw_only=True)(cls)

    @classmethod
    def declarations(cls) -> tuple[Decl, ...]:
        """The `Field` leaves of the class, in declaration order."""
        return _declarations(cls)

    def leaves(self) -> Iterator[tuple[Decl, Field[Any]]]:
        """The declarations with their fields, skipping the optional leaves that are None."""
        for declaration in self.declarations():
            field = getattr(self, declaration.name)
            if field is not None:
                yield declaration, field


_DECLARATIONS: dict[type[State], tuple[Decl, ...]] = {}


def _declarations(cls: type[State]) -> tuple[Decl, ...]:
    """
    The `Field` leaves of a `State` class, read from its annotations once and cached.

    Behind `State.declarations()`. Code that handles any state walks it instead of a per-state
    list of fields: `allocate`, `leaves()` (and through it `copy`), the physics driver and the
    IO. Cached because `typing.get_type_hints` evaluates the annotations, which are strings here.
    """
    if cls not in _DECLARATIONS:
        hints = typing.get_type_hints(cls)
        found = []
        for field in dataclasses.fields(cls):  # type: ignore[arg-type]
            hint, optional = _without_none(hints[field.name])
            if typing.get_origin(hint) is Field:
                (quantity,) = typing.get_args(hint)
                found.append(Decl(field.name, quantity, optional))
        _DECLARATIONS[cls] = tuple(found)
    return _DECLARATIONS[cls]


def _without_none(hint: Any) -> tuple[Any, bool]:
    """
    `X | None` as `(X, True)`, any other hint as `(hint, False)`.

    `_declarations` uses it to tell an optional leaf (`Field[Q] | None`) from a required one.
    """
    if typing.get_origin(hint) in (types.UnionType, typing.Union):
        others = [arg for arg in typing.get_args(hint) if arg is not type(None)]
        if len(others) == 1:
            return others[0], True
    return hint, False


class Empty(State):
    pass


def allocate[S: State](
    cls: type[S],
    grid: base_grid.Grid,
    allocator: gtx_typing.Allocator | None,
    *,
    fill: Callable[[str, tuple[int, ...]], Any] | None = None,
    only: Collection[str] | None = None,
) -> S:
    """
    A state with a zero field per declaration.

    `fill(name, shape)` gives the value of a leaf instead of zeros. `only` names the optional
    leaves to allocate; the other optional leaves are None. Without it every leaf is allocated.
    """
    values: dict[str, Any] = {}
    for declaration in cls.declarations():
        if declaration.optional and only is not None and declaration.name not in only:
            values[declaration.name] = None
            continue
        field = zeros(declaration.quantity, grid, allocator)
        if fill is not None:
            buffer: Any = field.data.ndarray  # NDArrayObject declares no __setitem__
            buffer[...] = fill(declaration.name, buffer.shape)
        values[declaration.name] = field
    return cls(**values)


def copy[S: State](state: S, allocator: gtx_typing.Allocator | None) -> S:
    """A new state with a copy of each present leaf on the allocator; the plain leaves are shared."""
    copies: dict[str, Any] = {
        declaration.name: Field(
            declaration.quantity, data_allocation.reallocate(field.data, allocator=allocator)
        )
        for declaration, field in state.leaves()
    }
    copied: S = dataclasses.replace(typing.cast(Any, state), **copies)
    return copied


# ------------------------------------------------------------------------------
# Component: Input, Output, run
# ------------------------------------------------------------------------------
class Component:
    """
    `run(inputs, outputs=None) -> Output`, the only method a component implements.

    `output = component.run(inputs)` writes into the component's own buffers, allocated once
    on first use; `component.run(inputs, outputs=view)` writes where the caller says, numpy
    `out=` style. Either way `run` returns the output. Whether `outputs` may alias `inputs` is
    the component's business; nothing checks it here.

    The own buffers (`cached_output`) hold every leaf of `Output`, the optional ones included:
    a component with optional leaves (`Advection`, `PhysicsDriver`: the tracers) would allocate
    all of them, so its composer always passes `outputs=`, views over the active ones only.
    """

    Input: type[State]
    Output: type[State]

    def __init__(self, grid: base_grid.Grid, allocator: gtx_typing.Allocator | None) -> None:
        self.grid = grid
        self.allocator = allocator

    @functools.cached_property
    def cached_output(self) -> Any:
        """The buffers a component writes when no `outputs=` is passed."""
        return allocate(type(self).Output, self.grid, self.allocator)

    def buffers[S: State](self, outputs: S | None) -> S:
        """The caller's buffers when given, else this component's own."""
        return self.cached_output if outputs is None else outputs

    # `Any`: each component narrows `inputs` and `outputs` to its own `Input` and `Output`.
    # Typed `State` here, every override would be incompatible (parameters are contravariant);
    # the class attributes `Input` and `Output` are not valid in a type expression.
    def run(self, inputs: Any, outputs: Any = None) -> Any:
        raise NotImplementedError
