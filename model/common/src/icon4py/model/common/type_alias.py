# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Literal, TypeAlias, get_origin, get_args

import gt4py.next as gtx


DEFAULT_PRECISION = "double"

# wp: working precision, vp: variable precision
wpfloat: type[gtx.float32] | type[gtx.float64] = gtx.float64
vpfloat: type[gtx.float32] | type[gtx.float64] = wpfloat
anyfloat: TypeAlias = gtx.float32 | gtx.float64
float64: TypeAlias = gtx.float64

precision = os.environ.get("FLOAT_PRECISION", DEFAULT_PRECISION).lower()


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


# TODO(pstark): Figure out a better name and place for this -> open for suggestions
#               Might be useful for other configs if they are written as dataclasses
def config_scalars_to_wp(self, attributes: list[str] = []):
    for name in attributes:        
        if not isinstance(v := object.__getattribute__(self, name), wpfloat):
            object.__setattr__(self, name, wpfloat(v))