# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.type_system import type_specifications as ts


BUILTIN_TO_ISO_C_TYPE: dict[ts.ScalarKind, str] = {
    ts.ScalarKind.FLOAT64: "real(c_double)",
    ts.ScalarKind.FLOAT32: "real(c_float)",
    ts.ScalarKind.BOOL: "logical(c_int)",
    ts.ScalarKind.INT32: "integer(c_int)",
    ts.ScalarKind.INT64: "integer(c_long)",
}
BUILTIN_TO_CPP_TYPE: dict[ts.ScalarKind, str] = {
    ts.ScalarKind.FLOAT64: "double",
    ts.ScalarKind.FLOAT32: "float",
    ts.ScalarKind.BOOL: "int",
    ts.ScalarKind.INT32: "int",
    ts.ScalarKind.INT64: "long",
}
BUILTIN_TO_NUMPY_TYPE: dict[ts.ScalarKind, str] = {
    ts.ScalarKind.FLOAT64: "xp.float64",
    ts.ScalarKind.FLOAT32: "xp.float32",
    ts.ScalarKind.BOOL: "xp.int32",
    ts.ScalarKind.INT32: "xp.int32",
    ts.ScalarKind.INT64: "xp.int64",
}
