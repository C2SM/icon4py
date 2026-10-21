# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Callable, Collection
from typing import Any, ClassVar, Literal, Protocol, dataclass_transform

import gt4py.next as gtx

from icon4py.model.common import type_alias as ta
from icon4py.model.common.utils import PredictorCorrectorPair, TimeStepPair, data_allocation


if typing.TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


__all__ = [
    "Component",
    "Empty",
    "Field",
    "PredictorCorrectorPair",
    "Quantity",
    "State",
    "Tendency",
    "TimeStepPair",
    "allocate",
    "copy",
    "fields",
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

    `Field` leaves are read back from the annotations; plain leaves (floats, ints,
    datetimes) are ordinary fields and are not listed. A leaf annotated `Field[Q] | None`
    is optional: `allocate` may leave it None; `fields()` skips it.

    A state is not a `Mapping`: the present leaves of an instance are read back as a
    mapping by the `fields()` free function, next to `allocate` and `copy`. States are
    identity-equal and hashable, the `eq=False` dataclass choice.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        dataclasses.dataclass(frozen=True, eq=False, kw_only=True)(cls)


def fields(state: State) -> dict[str, Field[Any]]:
    """
    The present leaves of a state, by name, in declaration order.

    One level of `Field` leaves, the optional ones skipped while None. `copy`, the
    physics driver and the IO walk a state through it.
    """
    return {
        declaration.name: field
        for declaration in _declarations(type(state))
        if (field := getattr(state, declaration.name)) is not None
    }


_DECLARATIONS: dict[type[State], tuple[Decl, ...]] = {}


def _declarations(cls: type[State]) -> tuple[Decl, ...]:
    """
    The `Field` leaves of a `State` class, read from its annotations once and cached.

    Behind `allocate` and `fields`. Code that handles any state walks it instead of a per-state
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
    fill: Callable[[str, tuple[gtx.Dimension, ...]], Any] | None = None,
    only: Collection[str] | None = None,
) -> S:
    """
    A state with a zero field per declaration.

    `fill(name, dims)` gives the value of a leaf instead of zeros. `only` names the optional
    leaves to allocate; the other optional leaves are None. Without it every leaf is allocated.
    """
    values: dict[str, Any] = {}
    for declaration in _declarations(cls):
        if declaration.optional and only is not None and declaration.name not in only:
            values[declaration.name] = None
            continue
        field = zeros(declaration.quantity, grid, allocator)
        if fill is not None:
            buffer: Any = field.data.ndarray  # NDArrayObject declares no __setitem__
            buffer[...] = fill(declaration.name, declaration.quantity.dims)
        values[declaration.name] = field
    return cls(**values)


def copy[S: State](state: S, allocator: gtx_typing.Allocator | None) -> S:
    """A new state with a copy of each present leaf on the allocator; the plain leaves are shared."""
    copies: dict[str, Any] = {
        name: Field(field.quantity, data_allocation.reallocate(field.data, allocator=allocator))
        for name, field in fields(state).items()
    }
    copied: S = dataclasses.replace(typing.cast(Any, state), **copies)
    return copied


# ------------------------------------------------------------------------------
# Component: Input, Output, run
# ------------------------------------------------------------------------------
class Component(Protocol):
    """
    `run(inputs, outputs=None) -> Output`, the only method a component implements.

    A component declares its two states on the class (`class Input(fw.State)`); a component
    that writes nothing declares `Output = fw.Empty`. The protocol has no implementation: the
    constructor and the own buffers (`cached_output`) are each component's own.

    `output = component.run(inputs)` writes into the component's own buffers, allocated once
    on first use; `component.run(inputs, outputs=view)` writes where the caller says, numpy
    `out=` style. Either way `run` returns the output. Whether `outputs` may alias `inputs` is
    the component's business; nothing checks it here.

    The own buffers hold every leaf of `Output`, the optional ones included: a component with
    optional leaves (`Advection`, `PhysicsDriver`: the tracers) would allocate all of them, so
    its caller always passes `outputs=`, views over the active ones only.
    """

    Input: type[State]
    Output: type[State]

    # `Any`: each component narrows `inputs` and `outputs` to its own `Input` and `Output`.
    # Typed `State` here, every override would be incompatible (parameters are contravariant).
    def run(self, inputs: Any, outputs: Any = None) -> Any:
        raise NotImplementedError
