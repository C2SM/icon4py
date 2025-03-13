# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Mapping, TypeAlias

from gt4py import eve
from gt4py.next.type_system import type_specifications as gtx_ts


# As long as we use gt4py.eve, we can also just use the gt4py.next types.
# Note that the user-facing type should be py2fgen.ScalarKind, so we can
# copy over the gt4py.next types if we want to.
ScalarKind = gtx_ts.ScalarKind


class DeviceType(eve.StrEnum):
    HOST = "host"
    MAYBE_DEVICE = "maybe_device"


class ArrayParamDescriptor(eve.Node):
    rank: int
    dtype: gtx_ts.ScalarKind
    device: DeviceType
    is_optional: bool


class ScalarParamDescriptor(eve.Node):
    dtype: gtx_ts.ScalarKind


ParamDescriptor: TypeAlias = ArrayParamDescriptor | ScalarParamDescriptor
ParamDescriptors: TypeAlias = Mapping[str, ParamDescriptor]
