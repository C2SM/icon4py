# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import typing
from typing import Any, ClassVar, Self

from gt4py import next as gtx
from gt4py.next.embedded import nd_array_field

from icon4py.model.common.quantity import Quantity


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
