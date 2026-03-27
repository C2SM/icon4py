# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from types import ModuleType

import numpy as np

from icon4py.model.common import constants as phy_const
from icon4py.model.common.utils import data_allocation as data_alloc


def calculate_specific_humidity(
    pressure: data_alloc.NDArray,
    vapor_pressure: data_alloc.NDArray,
) -> data_alloc.NDArray:
    """
    Calculate specific humidity from vapor pressure.

    Args:
        pressure: air pressure [Pa]
        vapor_pressure: vapor pressure [Pa]
    Returns:
        specific humidity (mixing ratio)
    """
    specific_humidity = (
        phy_const.RD_O_RV * vapor_pressure / (pressure + phy_const.RD_O_RV_MINUS_1 * vapor_pressure)
    )
    return specific_humidity


def calculate_saturation_presssure_water(
    temperature: data_alloc.NDArray, xp: ModuleType = np
) -> data_alloc.NDArray:
    """
    Compute saturation water vapour pressure by the Tetens formula.
        psat = p0 exp( aw (T-T0)/(T-bw)) )  [Tetens formula]

    Args:
        temperature: temperature [K]
    Returns:
        saturation water vapour pressure [Pa]
    """
    return phy_const.TETENS_P0 * xp.exp(
        phy_const.TETENS_AW
        * (temperature - phy_const.MELTING_TEMPERATURE)
        / (temperature - phy_const.TETENS_BW)
    )
