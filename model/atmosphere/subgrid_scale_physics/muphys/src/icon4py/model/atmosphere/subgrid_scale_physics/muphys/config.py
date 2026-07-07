# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class MuphysConfig:
    """Configuration for the muphys microphysics component."""

    qnc: float = 50.0 # Cloud droplet number concentration [cm^-3], default value hardcoded
