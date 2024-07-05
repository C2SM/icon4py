# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import sys
from enum import IntEnum
from typing import Final

from icon4py.model.common.type_alias import wpfloat


#: Gas constant for dry air [J/K/kg], called 'rd' in ICON (mo_physical_constants.f90),
#: see https://glossary.ametsoc.org/wiki/Gas_constant.
GAS_CONSTANT_DRY_AIR: Final[wpfloat] = 287.04
RD: Final[wpfloat] = GAS_CONSTANT_DRY_AIR

#: Specific heat at constant pressure [J/K/kg]
SPECIFIC_HEAT_CONSTANT_PRESSURE: Final[wpfloat] = 1004.64
CPD = SPECIFIC_HEAT_CONSTANT_PRESSURE

#: [J/K/kg] specific heat at constant volume
SPECIFIC_HEAT_CONSTANT_VOLUME: Final[wpfloat] = CPD - RD
CVD: Final[wpfloat] = SPECIFIC_HEAT_CONSTANT_VOLUME
CVD_O_RD: Final[wpfloat] = CVD / RD
RD_O_CPD: Final[wpfloat] = RD / CPD
CPD_O_RD: Final[wpfloat] = CPD / RD

#: Gas constant for water vapor [J/K/kg], rv in ICON.
GAS_CONSTANT_WATER_VAPOR: Final[wpfloat] = 461.51
RV: Final[wpfloat] = GAS_CONSTANT_WATER_VAPOR

#: Av. gravitational acceleration [m/s^2]
GRAVITATIONAL_ACCELERATION: Final[wpfloat] = 9.80665
GRAV: Final[wpfloat] = GRAVITATIONAL_ACCELERATION
GRAV_O_RD: Final[wpfloat] = GRAV / RD

#: reference pressure for Exner function [Pa]
REFERENCE_PRESSURE: Final[wpfloat] = 100000.0
P0REF: Final[wpfloat] = REFERENCE_PRESSURE

#: sea level pressure [Pa]
SEAL_LEVEL_PRESSURE: Final[wpfloat] = 101325.0
P0SL_BG: Final[wpfloat] = SEAL_LEVEL_PRESSURE

# average earth radius in [m]
EARTH_RADIUS: Final[float] = 6.371229e6

#: Earth angular velocity [rad/s]
EARTH_ANGULAR_VELOCITY: Final[wpfloat] = 7.29212e-5

#: sea level temperature for reference atmosphere [K]
SEA_LEVEL_TEMPERATURE: Final[wpfloat] = 288.15
T0SL_BG: Final[wpfloat] = SEA_LEVEL_TEMPERATURE

#: difference between sea level temperature and asymptotic stratospheric temperature
DELTA_TEMPERATURE: Final[wpfloat] = 75.0
DEL_T_BG: Final[wpfloat] = DELTA_TEMPERATURE

#: height scale for reference atmosphere [m], defined  in mo_vertical_grid
#: scale height [m]
_H_SCAL_BG: Final[wpfloat] = 10000.0

# Math constants
DBL_EPS = sys.float_info.epsilon  # EPSILON(1._wp)

# Implementation constants
#: default physics to dynamics time step ratio
# TODO (magdalena) not a constant, this is a default config parameter
DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO: Final[float] = 5.0


class RayleighType(IntEnum):
    RAYLEIGH_CLASSIC: Final[
        int
    ] = 1  # classical Rayleigh damping, which makes use of a reference state.
    RAYLEIGH_KLEMP: Final[int] = 2  # Klemp (2008) type Rayleigh damping
