# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import functools
import inspect
from collections.abc import Callable
from typing import (
    Annotated,
    ClassVar,
    Concatenate,
    Final,
    Generator,
    Generic,
    ParamSpec,
    TypeVar,
)


__all__ = [
    "chainable",
    "Pair",
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


T = TypeVar("T")
C = TypeVar("C")


class namedproperty(property[C, T]):
    """
    A simple extension of the built-in `property` descriptor storing the name of the
    attribute it is assigned to.

    The name is stored in the `name` attribute of the property instance.

    Examples:
        >>> class A:
        ...     @namedproperty
        ...     def value(self) -> int:
        ...         return self._value
        ...
        ...     @value.setter
        ...     def value(self, value: int) -> None:
        ...         self._value = value
        ...
        ...
        ... a = A()
        ... a.value = 1
        ... print(a.value.name)
        value
    """

    name: str

    def __set_name__(self, owner: C, name: str) -> None:
        self.name = name

    def getter(self: namedproperty[C, T], fget: Callable[[C], T]) -> namedproperty[C, T]:
        result = super().getter(fget)
        result.name = getattr(self, "name", None)
        return result

    def setter(self: namedproperty[C, T], fset: Callable[[C, T], None]) -> namedproperty[C, T]:
        result = super().setter(fset)
        result.name = getattr(self, "name", None)
        return result

    def deleter(self: namedproperty[C, T], fdel: Callable[[C], None]) -> namedproperty[C, T]:
        result = super().deleter(fdel)
        result.name = getattr(self, "name", None)
        return result

    def __copy__(self) -> namedproperty[C, T]:
        result = type(self)(self.fget, self.fset, self.fdel, self.__doc__)
        result.name = self.name
        return result


class Pair(Generic[T]):
    """
    A generic class representing a pair of values.

    The name of the attributes can be customized by using the Pair.FIRST
    and Pair.SECOND constants as type hint annotations of the
    respective attributes in the subclass.

    Examples:
        >>> class MyPair(Pair[T]):
        ...     a: Annotated[T, Pair.FIRST]
        ...     b: Annotated[T, Pair.SECOND]
        ...
        ...
        ... pair = MyPair(1, 2)
        ... print(pair)
        MyPair(a=1, b=2)

        >>> pair.swap()
        ... swapped = Pair(2, 1)
        ... pair == swapped
        True
    """

    FIRST: Final = "FIRST"
    SECOND: Final = "SECOND"

    __first_attr_name: ClassVar[str] = "first"
    __second_attr_name: ClassVar[str] = "second"

    def __init_subclass__(cls) -> None:
        for key, value in inspect.get_annotations(cls).items():
            if (metadata := getattr(value, "__metadata__", None)) is not None:
                if Pair.FIRST in metadata:
                    cls.__first_attr_name = key
                elif Pair.SECOND in metadata:
                    cls.__second_attr_name = key

    __first: T
    __second: T

    def __init__(self, first: T, second: T, /) -> None:
        self.__first = first
        self.__second = second

    @namedproperty
    def first(self) -> T:
        """Property getter for the first element of the pair."""
        return self.__first

    @first.setter
    def first(self, value: T) -> None:
        """Property setter for the first element of the pair."""
        self.__first = value

    @namedproperty
    def second(self) -> T:
        """Property getter for the read-only property returning the second element of the pair."""
        return self.__second

    #: Settable property returning the second element of the pair (for subclassing).
    _second_settable = namedproperty(
        fget=second.fget, fset=lambda self, value: setattr(self, "_Pair__second", value)
    )

    def __repr__(self) -> str:
        first_name = type(self).__first_attr_name
        second_name = type(self).__second_attr_name
        return f"{self.__class__.__name__}({first_name}={self.__first!r}, {second_name}={self.__second!r})"

    def __iter__(self) -> Generator[T, None, None]:
        yield self.__first
        yield self.__second

    def swap(self: Pair[T]) -> Pair[T]:
        """
        Swap the values of the first and second attributes of the instance.

        Returns:
            The instance with swapped values (for fluent interfaces).
        """
        self.__first, self.__second = self.__second, self.__first
        return self


class NextStepPair(Pair[T]):
    current: Annotated[T, Pair.FIRST] = copy.copy(Pair.first)
    next: Annotated[T, Pair.SECOND] = copy.copy(Pair.second)


class PreviousStepPair(Pair[T]):
    current: Annotated[T, Pair.FIRST] = copy.copy(Pair.first)
    previous: Annotated[T, Pair.SECOND] = copy.copy(Pair.second)
