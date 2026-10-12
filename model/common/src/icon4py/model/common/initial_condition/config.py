# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import logging

from icon4py.model.common import time
from icon4py.model.common.config import config_io
from icon4py.model.common.grid import vertical
from icon4py.model.common.initial_condition import from_file as from_file_ic
from icon4py.model.common.initial_condition.analytical import (
    gauss3d as gauss_ic,
    jablonowski_williamson as jw_ic,
    linear_horizontal_tracer_advection as lin_hor_adv_ic,
    linear_vertical_tracer_advection as lin_ver_adv_ic,
    weisman_klemp as wk_ic,
)


__all__ = ["IC_CONFIG", "ConfigContext"]


log = logging.getLogger(__name__)


type IC_CONFIG = (
    jw_ic.JablonowskiWilliamsonConfig
    | gauss_ic.Gauss3DConfig
    | wk_ic.WeismanKlempConfig
    | lin_hor_adv_ic.LinearHorizontalAdvectionConfig
    | lin_ver_adv_ic.LinearVerticalAdvectionConfig
    | from_file_ic.FromFileConfig
)


config_io.register_config_union(
    IC_CONFIG.__value__,
    {
        "jablonowski_williamson": jw_ic.JablonowskiWilliamsonConfig,
        "gauss_3d": gauss_ic.Gauss3DConfig,
        "weisman_klemp": wk_ic.WeismanKlempConfig,
        "linear_horizontal_adv": lin_hor_adv_ic.LinearHorizontalAdvectionConfig,
        "linear_vertical_adv": lin_ver_adv_ic.LinearVerticalAdvectionConfig,
        "from_file": from_file_ic.FromFileConfig,
    },
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConfigContext:
    initial_condition: IC_CONFIG
    is_restart: bool
    start_of_timestepping: time.AbsoluteTime
    dtime: time.RelativeTime
    vertical_grid: vertical.VerticalGridConfig
    ntracer: int
