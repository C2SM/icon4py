# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import TypeAlias

from gt4py.next.ffront.fbuiltins import float32, float64


DEFAULT_PRECISION = "double"

wpfloat: TypeAlias = float64

precision = os.environ.get("FLOAT_PRECISION", DEFAULT_PRECISION).lower()
match precision:
    case "double":
        vpfloat = wpfloat
    case "mixed":
        vpfloat: TypeAlias = float32
    case other:
        raise ValueError("Only 'double' and 'mixed' precision are supported.")
