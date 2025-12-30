# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Literal, TypeAlias

import gt4py.next as gtx


DEFAULT_PRECISION = "double"

wpfloat: TypeAlias = gtx.float64
vpfloat: TypeAlias = wpfloat

precision = os.environ.get("FLOAT_PRECISION", DEFAULT_PRECISION).lower()


def set_precision(new_precision: Literal["double", "mixed", "single"]) -> None:
    global precision, vpfloat, wpfloat  # noqa: PLW0603 [global-statement]

    precision = new_precision.lower()
    match precision:
        case "double":
            wpfloat = gtx.float64
            vpfloat = wpfloat
        case "mixed":
            wpfloat = gtx.float64
            vpfloat = gtx.float32
        case "single":
            wpfloat = gtx.float32
            vpfloat = wpfloat
        case _:
            raise ValueError("Only 'double', 'mixed' and 'single' precision are supported.")


set_precision(precision)
