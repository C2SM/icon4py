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
from collections.abc import Callable, Collection, Sequence
from typing import Any, Protocol, dataclass_transform

import gt4py.next as gtx

from icon4py.model.common import type_alias as ta
from icon4py.model.common.field import Field
from icon4py.model.common.quantity import Quantity
from icon4py.model.common.utils import Pair, data_allocation


if typing.TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


# NOTE: temporary, replaced in part 03 (dims): the dims and the dtype come from the field type
# instead of the quantity.
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


def fields(
    state: State, kind: type[Quantity] = Quantity, exclude: type[Quantity] | None = None
) -> dict[str, Field]:
    """
    The present leaves of a state whose quantity is a `kind` and not an `exclude`, by name, in
    declaration order.

    One level of `Field` leaves, the optional ones skipped while None. `copy`, the
    physics driver and the IO walk a state through it.
    """
    return {
        name: field
        for name, quantity, _ in _declarations(type(state))  # type: ignore[arg-type]
        if issubclass(quantity, kind)
        and not (exclude is not None and issubclass(quantity, exclude))
        and (field := getattr(state, name)) is not None
    }


# NOTE: temporary, replaced in part 06 (io-config): goes with the optional leaves.
def require[Q: Quantity](field: Field[Q] | None, name: str) -> Field[Q]:
    """The optional leaf `name`, which the caller needs present."""
    if field is None:
        raise ValueError(f"'{name}' is required, but the state lacks it")
    return field


def _declaration(name: str, hint: Any) -> tuple[str, type[Quantity], bool] | None:
    """
    The `(name, quantity, optional)` triple for one state annotation, or None when the
    annotation is not a `Field` leaf. `X | None` makes the leaf optional.
    """
    if typing.get_origin(hint) in (types.UnionType, typing.Union):
        others = [arg for arg in typing.get_args(hint) if arg is not type(None)]
        if len(others) == 1:
            hint, optional = others[0], True
        else:
            optional = False
    else:
        optional = False
    if typing.get_origin(hint) is not Field:
        return None
    (quantity,) = typing.get_args(hint)
    return (name, quantity, optional)


@functools.cache
def _declarations(cls: type[State]) -> Sequence[tuple[str, type[Quantity], bool]]:
    """
    The `(name, quantity, optional)` triples of a `State` class's `Field` leaves, in
    declaration order. Cached because `typing.get_type_hints` evaluates the annotations,
    which are strings here, and `fields()` runs per step in the driver.

    Behind `allocate`, `fields` and `leaf_quantities`. Code that handles any state walks it instead of a
    per-state list of fields: `copy` (through `fields`), the physics driver and the IO.
    """
    hints = typing.get_type_hints(cls)
    return tuple(
        declaration
        for field in dataclasses.fields(cls)  # type: ignore[arg-type]
        if (declaration := _declaration(field.name, hints[field.name])) is not None
    )


@functools.cache
def _pairs(cls: type[State]) -> Sequence[tuple[str, type[Pair], type[Quantity]]]:
    """The `(name, pair type, quantity)` triples of a `State` class's `Pair[Field[Q]]` leaves."""
    hints = typing.get_type_hints(cls)
    pairs = []
    for field in dataclasses.fields(cls):  # type: ignore[arg-type]
        pair = typing.get_origin(hints[field.name])
        if isinstance(pair, type) and issubclass(pair, Pair):
            (member,) = typing.get_args(hints[field.name])
            (quantity,) = typing.get_args(member)
            pairs.append((field.name, pair, quantity))
    return tuple(pairs)


def leaf_quantities(cls: type[State]) -> dict[str, type[Quantity]]:
    """The quantity of each `Field` leaf of a `State` class, by name, in declaration order."""
    return {name: quantity for name, quantity, _ in _declarations(cls)}  # type: ignore[arg-type]


class Empty(State):
    pass


def allocate[S: State](
    cls: type[S],
    grid: base_grid.Grid,
    allocator: gtx_typing.Allocator | None,
    *,
    fill: Callable[[str, tuple[gtx.Dimension, ...]], Any] | None = None,
    only: Collection[str] | None = None,
    **plain: Any,
) -> S:
    """
    A state with a zero field per declaration.

    `fill(name, dims)` gives the value of a leaf instead of zeros. `only` names the optional
    leaves to allocate; the other optional leaves are None. Without it every leaf is allocated.
    A `Pair` of fields gets one field per member, which `fields` and `copy` skip.
    The plain leaves (times, flags) are given by keyword in `plain`.
    """

    def leaf(name: str, quantity: type[Quantity]) -> Field:
        field = zeros(quantity, grid, allocator)
        if fill is not None:
            field.ndarray[...] = fill(name, quantity.dims)  # type: ignore[index]  # NDArrayObject declares no __setitem__
        return field

    values: dict[str, Any] = {}
    # type objects are hashable, but typeshed's cache wrapper cannot say so
    for name, quantity, optional in _declarations(cls):  # type: ignore[arg-type]
        if optional and only is not None and name not in only:
            values[name] = None
            continue
        values[name] = leaf(name, quantity)
    for name, pair, quantity in _pairs(cls):  # type: ignore[arg-type]
        values[name] = pair(leaf(name, quantity), leaf(name, quantity))
    return cls(**values, **plain)


def copy[S1: State, S2: State](src: S1, dst: S2, kind: type[Quantity] = Quantity) -> S2:
    """
    The present leaves of `src` whose quantity is a `kind` written into `dst`, in place.

    NumPy `out=` style: `dst`'s buffers are written and returned; the caller owns them. The
    present `kind` leaves of the two states must match exactly, in both directions: a leaf
    present on either side without a counterpart on the other raises, naming the leaves. The
    other leaves are neither compared nor written.
    """
    src_fields, dst_fields = fields(src, kind), fields(dst, kind)
    missing = sorted(src_fields.keys() - dst_fields.keys())
    unexpected = sorted(dst_fields.keys() - src_fields.keys())
    if missing or unexpected:
        raise ValueError(
            "the states' present leaves must match on both sides; "
            f"missing in the destination: {missing}; unexpected in it: {unexpected}"
        )
    for name, field in src_fields.items():
        dst_fields[name].ndarray[...] = field.ndarray  # type: ignore[index]  # NDArrayObject declares no __setitem__
    return dst


# ------------------------------------------------------------------------------
# Component: Input, Output, run
# ------------------------------------------------------------------------------


class Component[I: State, O: State](Protocol):
    """
    `run(input, output) -> output`, the only method a component implements.

    A component declares its two states, as class attributes (`class Input(fw.State)`) or as
    instance attributes assigned at construction; a component that writes nothing declares
    `Output = fw.Empty`. The protocol has no implementation: the constructor is each
    component's own.

    `component.run(input, output)` writes where the caller says, numpy `out=` style, and
    returns `output`: the caller owns the buffers. Whether `output` may alias `input` is
    the component's business; nothing checks it here.
    """

    Input: type[I]
    Output: type[O]

    def run(self, input: I, output: O) -> O: ...


def copy_to_output(input: State, output: State) -> None:
    """
    Copy the leaves `input` and `output` share into `output`, unless they share a buffer.

    For components that update in place: when the caller passes distinct buffers, `run`
    starts from a copy of `input`.
    """
    sources = fields(input)
    for name, target in fields(output).items():
        source = sources.get(name)
        if source is not None and target.ndarray is not source.ndarray:
            target.ndarray[...] = source.ndarray  # type: ignore[index]  # NDArrayObject declares no __setitem__
