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
vpfloat: type[gtx.float32] | type[gtx.float64] = wpfloat
anyfloat: TypeAlias = gtx.float32 | gtx.float64

precision = os.environ.get("FLOAT_PRECISION", DEFAULT_PRECISION).lower()


def set_precision(new_precision: Literal["double", "mixed"]) -> None:
    global precision  # noqa: PLW0603 [global-statement]
    global vpfloat  # noqa: PLW0603 [global-statement]

    precision = new_precision.lower()
    match precision:
        case "double":
            vpfloat = wpfloat
        case "mixed":
            vpfloat = gtx.float32
        case _:
            raise ValueError("Only 'double' and 'mixed' precision are supported.")


set_precision(precision)
