# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import enum
import sys
from typing import Final

from gt4py.eve import utils as eve_utils

from icon4py.model.common import type_alias as ta


#: Gas constant for dry air [J/K/kg], called 'rd' in ICON (mo_physical_constants.f90),
#: see https://glossary.ametsoc.org/wiki/Gas_constant.
GAS_CONSTANT_DRY_AIR: Final[ta.wpfloat] = 287.04
RD: Final[ta.wpfloat] = GAS_CONSTANT_DRY_AIR

#: Specific heat at constant pressure [J/K/kg]
SPECIFIC_HEAT_CONSTANT_PRESSURE: Final[ta.wpfloat] = 1004.64
CPD = SPECIFIC_HEAT_CONSTANT_PRESSURE

#: [J/K/kg] specific heat at constant volume
SPECIFIC_HEAT_CONSTANT_VOLUME: Final[ta.wpfloat] = CPD - RD
CVD: Final[ta.wpfloat] = SPECIFIC_HEAT_CONSTANT_VOLUME
CVD_O_RD: Final[ta.wpfloat] = CVD / RD
RD_O_CPD: Final[ta.wpfloat] = RD / CPD
CPD_O_RD: Final[ta.wpfloat] = CPD / RD

#: Gas constant for water vapor [J/K/kg], rv in ICON.
GAS_CONSTANT_WATER_VAPOR: Final[ta.wpfloat] = 461.51
RV: Final[ta.wpfloat] = GAS_CONSTANT_WATER_VAPOR

#: RV/RD - 1, tvmpc1 in ICON.
RV_O_RD_MINUS_1: Final[ta.wpfloat] = GAS_CONSTANT_WATER_VAPOR / GAS_CONSTANT_DRY_AIR - 1.0
TVMPC1: Final[ta.wpfloat] = RV_O_RD_MINUS_1

#: Av. gravitational acceleration [m/s^2]
GRAVITATIONAL_ACCELERATION: Final[ta.wpfloat] = 9.80665
GRAV: Final[ta.wpfloat] = GRAVITATIONAL_ACCELERATION
GRAV_O_RD: Final[ta.wpfloat] = GRAV / RD

#: reference pressure for Exner function [Pa]
REFERENCE_PRESSURE: Final[ta.wpfloat] = 100000.0
P0REF: Final[ta.wpfloat] = REFERENCE_PRESSURE

#: sea level pressure [Pa]
SEAL_LEVEL_PRESSURE: Final[ta.wpfloat] = 101325.0
P0SL_BG: Final[ta.wpfloat] = SEAL_LEVEL_PRESSURE

# average earth radius in [m]
EARTH_RADIUS: Final[float] = 6.371229e6

#: Earth angular velocity [rad/s]
EARTH_ANGULAR_VELOCITY: Final[ta.wpfloat] = 7.29212e-5

#: sea level temperature for reference atmosphere [K]
SEA_LEVEL_TEMPERATURE: Final[ta.wpfloat] = 288.15
T0SL_BG: Final[ta.wpfloat] = SEA_LEVEL_TEMPERATURE

#: difference between sea level temperature and asymptotic stratospheric temperature
DELTA_TEMPERATURE: Final[ta.wpfloat] = 75.0
DEL_T_BG: Final[ta.wpfloat] = DELTA_TEMPERATURE

#: height scale for reference atmosphere [m], defined  in mo_vertical_grid
#: scale height [m]
HEIGHT_SCALE_FOR_REFERENCE_ATMOSPHERE = 10000.0
_H_SCAL_BG: Final[ta.wpfloat] = HEIGHT_SCALE_FOR_REFERENCE_ATMOSPHERE

# Math constants
DBL_EPS = sys.float_info.epsilon  # EPSILON(1._wp)

# Implementation constants
#: default physics to dynamics time step ratio
# TODO (magdalena) not a constant, this is a default config parameter
DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO: Final[float] = 5.0

#: average earth radius in [m]
EARTH_RADIUS: Final[ta.wpfloat] = 6.371229e6


class RayleighType(enum.IntEnum):
    #: classical Rayleigh damping, which makes use of a reference state.
    CLASSIC = 1
    #: Klemp (2008) type Rayleigh damping
    KLEMP = 2


class PhysicsConstants(eve_utils.FrozenNamespace[ta.wpfloat]):
    rd = RD
    rv = RV
    cpd = CPD
    cvd = CVD
    rv_o_rd_minus_1 = RV_O_RD_MINUS_1
    rd_o_cpd = RD_O_CPD
    grav_o_rd = GRAV_O_RD
    p0ref = P0REF
    cpd_o_rd = CPD_O_RD
