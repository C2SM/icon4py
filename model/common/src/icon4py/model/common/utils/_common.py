# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Concatenate, Generic, ParamSpec, TypeVar


__all__ = [
    "chainable",
    "Swapping",
]

P = ParamSpec("P")
T = TypeVar("T")


def chainable(method_fn: Callable[Concatenate[T, P], None]) -> Callable[Concatenate[T, P], T]:
    """
    Make an instance method return the actual instance so it can used in a chain of calls.

    Typically used for simple fluent interfaces.

    Examples:
        >>> class A:
        ...     @chainable
        ...     def set_value(self, value: int) -> None:
        ...         self.value = value
        ...
        ...     @chainable
        ...     def increment(self, value: int) -> None:
        ...         self.value += value
        ...
        ...
        ... a = A()
        ... a.set_value(1).increment(2)
        ... a.value
        3
    """

    @functools.wraps(method_fn)
    def wrapper(self: T, *args: P.args, **kwargs: P.kwargs) -> T:
        method_fn(self, *args, **kwargs)
        return self

    return wrapper


class Swapping(Generic[T]):
    """
    Generic double container for swapping between two values.

    This is useful for double buffering in numerical algorithms.

    Examples:
        >>> a = Swapping(current=1, other=2)
        Swapping(1, 2)

        >>> a.swap()
        Swapping(current=2, other=1)

        >>> a.current = 3
        ... a
        Swapping(current=3, other=1)

        >>> a != ~a
        True

        >>> a == ~~a
        True

        >>> a.current == (~a).other
        True

        >>> b = ~a
        ... a.swap()
        ... a == b
        True
    """

    __slots__ = ("current", "_other", "__weakref__")

    current: T
    _other: T

    @property
    def other(self) -> T:
        return self._other

    def __init__(self, current: T, other: T) -> None:
        self.current = current
        self._other = other

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(current={self.current!r}, other={self._other!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Swapping)
            and self.current == other.current
            and self._other == other._other
        )

    # `__hash__` is implicitly set to None when `__eq__` is redefined, so instances are not hashable.

    def swap(self) -> None:
        self.current, self._other = self._other, self.current

    def __invert__(self) -> Swapping[T]:
        return type(self)(current=self._other, other=self.current)
