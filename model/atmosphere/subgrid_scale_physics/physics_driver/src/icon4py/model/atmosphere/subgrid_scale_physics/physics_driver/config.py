# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the physics driver.

Kept apart from the driver itself so that reading or building a configuration does
not import the stencils.
"""

from __future__ import annotations

import enum

from icon4py.model.common.config import config_io


@config_io.register_enum
class CouplingType(enum.Enum):
    """
    Decide how the physics processes see each other's updates within one timestep.

    The choice applies to all processes. Either way the processes run in list order,
    and at the end exner and theta_v are recomputed once from the final temperature
    and moisture, and vn is updated once from the summed (u, v) tendencies.
    """

    #: Every process reads the same prognostic and basic diagnostic (u , v, w, temperature, pressure, exner, ...) state.
    PARALLEL = 1
    #: Every process reads the above state left by the processes before it
    SEQUENTIAL = 2
