# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.tools.py2fgen._definitions import (
    ArrayParamDescriptor,
    DeviceType,
    ParamDescriptor,
    ScalarKind as _ScalarKind,
    ScalarParamDescriptor,
)
from icon4py.tools.py2fgen._export import (
    export,
)


BOOL = _ScalarKind.BOOL
INT32 = _ScalarKind.INT32
INT64 = _ScalarKind.INT64
FLOAT32 = _ScalarKind.FLOAT32
FLOAT64 = _ScalarKind.FLOAT64


__all__ = [
    "ArrayParamDescriptor",
    "DeviceType",
    "ParamDescriptor",
    "ScalarParamDescriptor",
    "export",
    "BOOL",
    "INT32",
    "INT64",
    "FLOAT32",
    "FLOAT64",
]
