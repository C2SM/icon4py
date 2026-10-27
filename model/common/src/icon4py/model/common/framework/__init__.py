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
from typing import Any, ClassVar, Protocol, Self, dataclass_transform

import gt4py.next as gtx
from gt4py.next.embedded import nd_array_field

from icon4py.model.common import type_alias as ta
from icon4py.model.common.quantity import Quantity
from icon4py.model.common.utils import PredictorCorrectorPair, TimeStepPair, data_allocation


if typing.TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


__all__ = [
    "Component",
    "Empty",
    "Field",
    "PredictorCorrectorPair",
    "State",
    "TimeStepPair",
    "allocate",
    "copy",
    "fields",
    "zeros",
]


# NOTE: temporary, components MWE part 07 (tendencies): `TendencyOf[Q]`, derived from its parent
# quantity, replaces the marker base and the hand-declared tendency tags.


@dataclasses.dataclass(frozen=True, eq=False, init=False, repr=False)
class Field[Q: Quantity](nd_array_field.NdArrayField[Any, Any]):
    """
    A gt4py field tagged by its quantity: `Field(quantity, data)` shares `data`'s buffer.

    It is a gt4py field itself and goes straight into a program. A field gt4py derives from it
    (arithmetic, slicing, `astype`) is a plain gt4py field, untagged.
    """

    quantity: type[Q]

    # the plain gt4py field class of each array namespace -> its tagged subclass
    _TAGGED: ClassVar[dict[type, type[Field[Any]]]] = {}
    _plain: ClassVar[type[nd_array_field.NdArrayField[Any, Any]]]

    if typing.TYPE_CHECKING:
        # implemented by gt4py's FieldBuiltinFuncRegistry, which mypy does not see
        @classmethod
        def __gt_builtin_func__(cls, /, func: Any) -> Any: ...

    def __new__(cls, quantity: type[Q] | None = None, data: Any = None) -> Self:
        if data is None:  # the copy and pickle protocols pass no arguments
            return object.__new__(cls)
        plain = data._plain if isinstance(data, Field) else type(data)
        return typing.cast(Self, object.__new__(Field._TAGGED[plain]))

    def __init__(self, quantity: type[Q], data: gtx.Field[Any, Any]) -> None:
        object.__setattr__(self, "_domain", data.domain)
        object.__setattr__(self, "_ndarray", data.ndarray)
        object.__setattr__(self, "quantity", quantity)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "_plain" in vars(cls):
            Field._TAGGED[cls._plain] = cls

    @classmethod
    def from_array(cls, data: Any, /, *, domain: Any, dtype: Any = None) -> Any:
        # gt4py builds every derived field through `from_array`
        return cls._plain.from_array(data, domain=domain, dtype=dtype)

    def __repr__(self) -> str:
        return f"Field({self.quantity.__name__}, {self.ndarray.shape})"


class _NumPyField[Q: Quantity](Field[Q], nd_array_field.NumPyArrayField):
    _plain = nd_array_field.NumPyArrayField


if hasattr(nd_array_field, "CuPyArrayField"):

    class _CuPyField[Q: Quantity](Field[Q], nd_array_field.CuPyArrayField):
        _plain = nd_array_field.CuPyArrayField


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


def fields(state: State) -> dict[str, Field[Any]]:
    """
    The present leaves of a state, by name, in declaration order.

    One level of `Field` leaves, the optional ones skipped while None. `copy`, the
    physics driver and the IO walk a state through it.
    """
    return {
        name: field
        for name, _, _ in _declarations(type(state))  # type: ignore[arg-type]
        if (field := getattr(state, name)) is not None
    }


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

    Behind `allocate` and `fields`. Code that handles any state walks it instead of a
    per-state list of fields: `copy` (through `fields`), the physics driver and the IO.
    """
    hints = typing.get_type_hints(cls)
    return tuple(
        declaration
        for field in dataclasses.fields(cls)  # type: ignore[arg-type]
        if (declaration := _declaration(field.name, hints[field.name])) is not None
    )


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
    # type objects are hashable, but typeshed's cache wrapper cannot say so
    for name, quantity, optional in _declarations(cls):  # type: ignore[arg-type]
        if optional and only is not None and name not in only:
            values[name] = None
            continue
        field = zeros(quantity, grid, allocator)
        if fill is not None:
            buffer: Any = field.ndarray  # NDArrayObject declares no __setitem__
            buffer[...] = fill(name, quantity.dims)
        values[name] = field
    return cls(**values)


def copy[S: State](state: S, allocator: gtx_typing.Allocator | None) -> S:
    """A new state with a copy of each present leaf on the allocator; the plain leaves are shared."""
    copies: dict[str, Any] = {
        name: Field(field.quantity, data_allocation.reallocate(field, allocator=allocator))
        for name, field in fields(state).items()
    }
    copied: S = dataclasses.replace(typing.cast(Any, state), **copies)
    return copied


# ------------------------------------------------------------------------------
# Component: Input, Output, run
# ------------------------------------------------------------------------------


class Component[I: State, O: State](Protocol):
    """
    `run(inputs, outputs) -> outputs`, the only method a component implements.

    A component declares its two states on the class (`class Input(fw.State)`); a component
    that writes nothing declares `Output = fw.Empty`. The protocol has no implementation: the
    constructor is each component's own.

    `component.run(inputs, outputs)` writes where the caller says, numpy `out=` style, and
    returns `outputs`: the caller owns the buffers. Whether `outputs` may alias `inputs` is
    the component's business; nothing checks it here.
    """

    Input: ClassVar[type[I]]
    Output: ClassVar[type[O]]

    def run(self, inputs: I, outputs: O) -> O:
        raise NotImplementedError
