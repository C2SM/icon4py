# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.common import constants as phy_const, type_alias as ta


def calculate_specific_humidity(
    pressure: ta.wpfloat,
    vapor_pressure: ta.wpfloat,
) -> ta.wpfloat:
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
