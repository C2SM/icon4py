# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Host-side thermodynamic helper functions.

NumPy/CuPy counterparts of the elemental functions in ICON's
``mo_thdyn_functions.f90``. These operate on host arrays (e.g. during initial
condition setup) and are not GT4Py field operators; the saturation-adjustment
microphysics has its own GT4Py implementation in the muphys package.
"""

from __future__ import annotations

from icon4py.model.common import constants as phy_const
from icon4py.model.common.utils import data_allocation as data_alloc


def sat_pres_water(temperature: data_alloc.NDArray) -> data_alloc.NDArray:
    """Saturation vapour pressure over liquid water [Pa] (Tetens formula).

    Mirrors ``sat_pres_water`` in ``mo_thdyn_functions.f90`` (``ipsat <= 1``).
    """
    array_ns = data_alloc.array_namespace(temperature)
    return phy_const.TETENS_P0 * array_ns.exp(
        phy_const.TETENS_A_WATER
        * (temperature - phy_const.MELTING_TEMPERATURE)
        / (temperature - phy_const.TETENS_B_WATER)
    )


def sat_pres_ice(temperature: data_alloc.NDArray) -> data_alloc.NDArray:
    """Saturation vapour pressure over ice [Pa] (Tetens formula).

    Mirrors ``sat_pres_ice`` in ``mo_thdyn_functions.f90`` (``ipsat <= 1``).
    """
    array_ns = data_alloc.array_namespace(temperature)
    return phy_const.TETENS_P0 * array_ns.exp(
        phy_const.TETENS_A_ICE
        * (temperature - phy_const.MELTING_TEMPERATURE)
        / (temperature - phy_const.TETENS_B_ICE)
    )
