# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""States of the surface scheme."""

from __future__ import annotations

import dataclasses

from icon4py.model.common import field_type_aliases as fa, type_alias as ta


@dataclasses.dataclass(frozen=True)
class SurfaceInputState:
    """Atmospheric input of the surface scheme: the lowest level of the atmosphere."""

    surface_pressure: fa.CellField[ta.wpfloat]
    """Pressure at the surface, the lowest half level (``psfc``) [Pa]."""
    temperature: fa.CellField[ta.wpfloat]
    """Air temperature at the lowest full level (``ta(nlev)``) [K]."""
