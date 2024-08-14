# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.common.config import Icon4PyConfig


config = Icon4PyConfig()
backend = config.gt4py_runner
xp = config.array_ns
device = config.device
limited_area = config.limited_area
