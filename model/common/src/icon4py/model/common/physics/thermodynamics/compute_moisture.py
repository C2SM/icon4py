# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Host-side moisture helpers.

NumPy/CuPy functions on host arrays (e.g. during initial-condition setup), not
GT4Py field operators; the saturation-adjustment microphysics has its own GT4Py
implementation in the muphys package.
"""

from __future__ import annotations

from icon4py.model.common import constants as phy_const
from icon4py.model.common.physics.thermodynamics.compute_pressure import (
    sat_pres_ice,
    sat_pres_water,
)
from icon4py.model.common.utils import data_allocation as data_alloc


def specific_humidity(
    vapor_pressure: data_alloc.NDArray, pressure: data_alloc.NDArray
) -> data_alloc.NDArray:
    """Specific humidity [kg/kg] from the water-vapour partial pressure and the total pressure.

    ``rdv * pv / (p - (1 - rdv) * pv)`` with ``rdv = RD / RV``.
    """
    return (
        phy_const.RD_O_RV * vapor_pressure / (pressure - (1.0 - phy_const.RD_O_RV) * vapor_pressure)
    )


def qv_from_relative_humidity(
    temperature: data_alloc.NDArray,
    pressure: data_alloc.NDArray,
    rho: data_alloc.NDArray,
    relative_humidity: data_alloc.NDArray,
) -> data_alloc.NDArray:
    """Specific humidity [kg/kg] from a relative-humidity field.

    The saturation vapour pressure (ice below the
    melting point, water above; the ice branch clamps the temperature at
    ``MINIMUM_TEMPERATURE_ICE_SATURATION``) is capped so the vapour pressure cannot
    exceed the total pressure, converted to a saturation specific humidity, and
    scaled by the relative humidity.

    This is the general computation only; the relative-humidity *profile* and any
    test-case specific caps are the caller's responsibility.
    """
    array_ns = data_alloc.array_namespace(rho)
    saturation_pressure = array_ns.where(
        temperature <= phy_const.MELTING_TEMPERATURE,
        sat_pres_ice(array_ns.maximum(temperature, phy_const.MINIMUM_TEMPERATURE_ICE_SATURATION)),
        sat_pres_water(temperature),
    )
    # avoid water vapour pressure > total pressure
    vapour_pressure = array_ns.minimum(saturation_pressure, pressure / (relative_humidity + 1.0e-6))
    saturation_qv = vapour_pressure / (rho * phy_const.RV * temperature)
    # cap relative humidity at 1.0 to avoid supersaturation
    return array_ns.minimum(relative_humidity, 1.0) * saturation_qv
