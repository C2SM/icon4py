# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import os

from gt4py.next.ffront.fbuiltins import float32, float64
from typing import TypeAlias

wpfloat: TypeAlias = float64

precision = os.environ.get("FLOAT_PRECISION", "double").lower()
if precision == "double":
    vpfloat = wpfloat
elif precision == "mixed":
    vpfloat: TypeAlias = float32
else:
    raise ValueError("Only 'mixed' and 'double' precision are supported.")
