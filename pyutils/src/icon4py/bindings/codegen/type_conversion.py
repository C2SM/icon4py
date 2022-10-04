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

from functional.ffront.common_types import ScalarKind


BUILTIN_TO_ISO_C_TYPE = {
    ScalarKind.FLOAT64: "real(c_double)",
    ScalarKind.FLOAT32: "real(c_float)",
    ScalarKind.BOOL: "logical(c_int)",
    ScalarKind.INT32: "c_int",
    ScalarKind.INT64: "c_long",
}
BUILTIN_TO_CPP_TYPE = {
    ScalarKind.FLOAT64: "double",
    ScalarKind.FLOAT32: "float",
    ScalarKind.BOOL: "int",
    ScalarKind.INT32: "int",
    ScalarKind.INT64: "long",
}
