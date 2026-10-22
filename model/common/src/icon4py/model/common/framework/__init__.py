# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import dataclasses
import datetime
import functools
import types
import typing
from collections.abc import Callable, Collection, Sequence
from typing import Any, ClassVar, Literal, Protocol, Self, dataclass_transform

import gt4py.next as gtx
from gt4py.next.embedded import nd_array_field

from icon4py.model.common import type_alias as ta
from icon4py.model.common.utils import PredictorCorrectorPair, TimeStepPair, data_allocation


if typing.TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


__all__ = [
    "Component",
    "Empty",
    "Field",
    "PhysicsProcess",
    "PredictorCorrectorPair",
    "ProcessTimeControl",
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
class Component(Protocol):
    """
    `run(inputs, outputs) -> Output`, the only method a component implements.

    A component declares its two states on the class (`class Input(fw.State)`); a component
    that writes nothing declares `Output = fw.Empty`. The protocol has no implementation: the
    constructor is each component's own.

    `component.run(inputs, outputs)` writes where the caller says, numpy `out=` style, and
    returns `outputs`: the caller owns the buffers. Whether `outputs` may alias `inputs` is
    the component's business; nothing checks it here. A `PhysicsProcess` also keeps buffers
    of its own, since it reuses its last output between two computations.
    """

    Input: type[State]
    Output: type[State]

    # `Any`: each component narrows `inputs` and `outputs` to its own `Input` and `Output`.
    # Typed `State` here, every override would be incompatible (parameters are contravariant).
    def run(self, inputs: Any, outputs: Any) -> Any:
        raise NotImplementedError


# ------------------------------------------------------------------------------
# PhysicsProcess: a component under a time control
# ------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ProcessTimeControl:
    """icon4py analogue of the per-process time fields in AES `aes_phy_tc`.

    Mirrors `mo_aes_phy_main.f90` semantics, with one deviation: where AES
    disables a process via `dt_xxx == 0`, icon4py leaves a disabled process out
    of the driver's process list entirely, so it is never constructed and its
    stencils are never compiled.
      - `interval`   (`dt_xxx`): firing interval; must be > 0.
      - `start_date` (`sd_xxx`), `end_date` (`ed_xxx`): half-open
        `[start, end)` window during which the process exists at all.
    """

    interval: datetime.timedelta
    start_date: datetime.datetime
    end_date: datetime.datetime

    def is_in_window(self, simulation_current_datetime: datetime.datetime) -> bool:
        return self.start_date <= simulation_current_datetime < self.end_date

    def is_active(self, simulation_current_datetime: datetime.datetime) -> bool:
        """True if the process's mtime-event fires on the given step.

        Equivalent to AES `isCurrentEventActive(ev_xxx, datetime)`. Fires only
        when the elapsed time is an exact integer multiple of the interval.
        """
        if self.interval <= datetime.timedelta(0) or simulation_current_datetime < self.start_date:
            return False
        elapsed = simulation_current_datetime - self.start_date
        return elapsed % self.interval == datetime.timedelta(0)

    def validate_interval(self, dtime: datetime.timedelta) -> None:
        """Fail loud if the firing interval cannot align with the model timestep.

        ``is_active`` fires only when the elapsed time is an exact multiple of
        ``interval``. With a discrete timestep that is observed only when
        ``interval`` is a positive integer multiple of ``dtime`` -- otherwise the
        process fires only at common multiples of both (or never), silently.
        """
        if self.interval <= datetime.timedelta(0):
            raise ValueError(f"time-control interval must be positive, got {self.interval}")
        if self.interval % dtime != datetime.timedelta(0):
            raise ValueError(
                f"time-control interval {self.interval} is not an integer multiple of "
                f"the model timestep {dtime}: the process would fire only at common "
                "multiples of both (or never)"
            )


class PhysicsProcess(Component):
    """
    A physics component with a time control: computes when its time control fires, else
    returns its cached output; None outside its window.

    `step` is what the physics driver calls; `run` computes. Between two computations the
    driver accumulates the last output again, so `step` has `run` write it into the
    component's own buffers (`cached_output`).
    """

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        allocator: gtx_typing.Allocator | None,
        time_control: ProcessTimeControl,
    ) -> None:
        self.grid = grid
        self.allocator = allocator
        self.time_control = time_control
        self._last_output: State | None = None

    @abc.abstractmethod
    def collect_input(self, entry: Any) -> Any:
        """This component's `Input` from the physics driver's entry state; each subclass narrows both."""

    @functools.cached_property
    def cached_output(self) -> Any:
        """The buffers `run` writes when no `outputs=` is passed."""
        return allocate(type(self).Output, self.grid, self.allocator)

    def step(
        self, entry: Any, step_start: datetime.datetime, dtime: datetime.timedelta
    ) -> State | None:
        """This step's output: computed, reused from the cache, or None outside the window."""
        time_control = self.time_control
        time_control.validate_interval(dtime)
        # Outside its window the process contributes nothing, and must not reach the reuse
        # branch below, which would compute once on the first step.
        if not time_control.is_in_window(step_start):
            return None
        # Compute on a firing (active) step, and also on the first in-window step, when there
        # is nothing to reuse yet. Otherwise reuse the last computed output.
        if time_control.is_active(step_start) or self._last_output is None:
            self._last_output = self.run(self.collect_input(entry), self.cached_output)
        return self._last_output
