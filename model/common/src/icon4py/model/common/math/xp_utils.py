# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Backward-compatibility re-export module.

The ``compute_sqrt`` function has been moved to ``math_utils``.
Please update imports to use ``icon4py.model.common.math.math_utils`` directly.
"""

from icon4py.model.common.math.math_utils import compute_sqrt  # noqa: F401
