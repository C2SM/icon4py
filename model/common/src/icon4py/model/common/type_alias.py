# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import inspect
import os
import typing
from typing import Literal, TypeAlias

import gt4py.next as gtx
import numpy as np


DEFAULT_PRECISION = "double"

# wp: working precision, vp: variable precision
wpfloat: TypeAlias = gtx.float64  # noqa: UP040
vpfloat: TypeAlias = gtx.float64  # noqa: UP040
type anyfloat = gtx.float32 | gtx.float64

precision = os.environ.get("ICON4PY_FLOAT_PRECISION", DEFAULT_PRECISION).lower()


def set_precision(new_precision: Literal["double", "mixed", "single"]) -> None:
    global precision, vpfloat, wpfloat  # noqa: PLW0603 [global-statement]

    precision = new_precision.lower()
    match precision:
        case "double":
            wpfloat = gtx.float64
            vpfloat = gtx.float64
        case "mixed":
            wpfloat = gtx.float64
            vpfloat = gtx.float32
        case "single":
            vpfloat = gtx.float32
            wpfloat = gtx.float32
        case _:
            raise ValueError("Only 'double', 'mixed' and 'single' precision are supported.")


set_precision(precision)


def _castable_float_type(t: typing.Any) -> bool:
    while typing.get_origin(t) in (typing.Final, typing.Annotated):
        t = typing.get_args(t)[0]
    return inspect.isclass(t) and issubclass(t, (float, np.floating))


def dataclass_float_to_wp(self, attributes: list[str] | None = None):
    """Cast float attributes of a dataclass instance to `wpfloat` in place.

    Meant as a helper function to call from `__post_init__`.

    Arg `attributes` defaults to all fields whose resolved type is a float type. For union types
    containing floats pass `attributes` explicitly or convert at the point of use.
    """
    if not dataclasses.is_dataclass(self):
        raise ValueError("This function is meant for dataclasses")
    if attributes is None:
        attributes = [
            name
            for name, this_type in typing.get_type_hints(type(self)).items()
            if name in self.__dataclass_fields__ and _castable_float_type(this_type)
        ]
    for name in attributes:
        if not isinstance(v := object.__getattribute__(self, name), wpfloat):
            object.__setattr__(self, name, wpfloat(v))
