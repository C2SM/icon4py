# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Literal, TypeAlias

from gt4py.next.ffront.fbuiltins import float32, float64


DEFAULT_PRECISION = "double"

wpfloat: TypeAlias = float64
vpfloat: type[float32] | type[float64] = wpfloat

precision = os.environ.get("FLOAT_PRECISION", DEFAULT_PRECISION).lower()


def set_precision(new_precision: Literal["double", "mixed"]) -> None:
    global precision
    global vpfloat

    precision = new_precision.lower()
    match precision:
        case "double":
            vpfloat = wpfloat
        case "mixed":
            vpfloat = float32
        case _:
            raise ValueError("Only 'double' and 'mixed' precision are supported.")


set_precision(precision)
