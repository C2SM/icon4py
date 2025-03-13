# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import TYPE_CHECKING, Mapping, TypeAlias

from gt4py import eve
from gt4py.next.type_system import type_specifications as gtx_ts


# As long as we use gt4py.eve, we can also just use the gt4py.next types.
# Note that the user-facing type should be py2fgen.ScalarKind, so we can
# copy over the gt4py.next types if we want to.
ScalarKind: TypeAlias = gtx_ts.ScalarKind

BOOL = gtx_ts.ScalarKind.BOOL
INT32 = gtx_ts.ScalarKind.INT32
INT64 = gtx_ts.ScalarKind.INT64
FLOAT32 = gtx_ts.ScalarKind.FLOAT32
FLOAT64 = gtx_ts.ScalarKind.FLOAT64


class DeviceType(eve.StrEnum):
    HOST = "host"
    MAYBE_DEVICE = "maybe_device"


class ArrayParamDescriptor(eve.Node):
    rank: int
    dtype: ScalarKind
    device: DeviceType
    is_optional: bool


class ScalarParamDescriptor(eve.Node):
    dtype: ScalarKind


ParamDescriptor: TypeAlias = ArrayParamDescriptor | ScalarParamDescriptor
ParamDescriptors: TypeAlias = Mapping[str, ParamDescriptor]


if TYPE_CHECKING:
    import cffi

    # Note, we use this plain tuple for performance.
    ArrayDescriptor: TypeAlias = tuple[cffi.FFI.CData, tuple[int, ...], bool, bool]
else:
    from typing import Any

    ArrayDescriptor: TypeAlias = tuple[Any, tuple[int, ...], bool, bool]
