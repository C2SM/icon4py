# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import sys
from typing import Final

from gt4py.eve import utils as eve_utils

from icon4py.model.common.type_alias import wpfloat


#: Gas constant for dry air [J/K/kg], called 'rd' in ICON (mo_physical_constants.f90),
#: see https://glossary.ametsoc.org/wiki/Gas_constant.
GAS_CONSTANT_DRY_AIR: Final[wpfloat] = wpfloat(287.04)
RD: Final[wpfloat] = GAS_CONSTANT_DRY_AIR

#: Specific heat capacity of dry air at constant pressure [J/K/kg]
SPECIFIC_HEAT_CAPACITY_PRESSURE_DRY_AIR: Final[wpfloat] = wpfloat(1004.64)
CPD = SPECIFIC_HEAT_CAPACITY_PRESSURE_DRY_AIR

#: [J/K/kg] specific heat capacity at constant volume
SPECIFIC_HEAT_CAPACITY_VOLUME_DRY_AIR: Final[wpfloat] = CPD - RD
CVD: Final[wpfloat] = SPECIFIC_HEAT_CAPACITY_VOLUME_DRY_AIR
CVD_O_RD: Final[wpfloat] = CVD / RD
RD_O_CPD: Final[wpfloat] = RD / CPD
CPD_O_RD: Final[wpfloat] = CPD / RD
RD_O_CVD: Final[wpfloat] = RD / CVD

#: Gas constant for water vapor [J/K/kg], rv in ICON.
GAS_CONSTANT_WATER_VAPOR: Final[wpfloat] = wpfloat(461.51)
RV: Final[wpfloat] = GAS_CONSTANT_WATER_VAPOR

#: Specific heat capacity of water vapor at constant pressure [J/K/kg]
SPECIFIC_HEAT_CAPACITY_PRESSURE_WATER_VAPOR: Final[wpfloat] = wpfloat(1869.46)
CPV = SPECIFIC_HEAT_CAPACITY_PRESSURE_WATER_VAPOR

#: Specific heat capacity of water vapor at constant volume [J/K/kg]
SPECIFIC_HEAT_CAPACITY_VOLUME_WATER_VAPOR: Final[wpfloat] = CPV - RV
CVV = SPECIFIC_HEAT_CAPACITY_VOLUME_WATER_VAPOR

#: cp_dry_air / cp_liquid_water - 1
_RCPL: Final[wpfloat] = wpfloat(3.1733)

#: Specific heat capacity of liquid water [J/K/kg]. Originally expressed as clw in ICON.
SPECIFIC_HEAT_CAPACITY_LIQUID_WATER: Final[wpfloat] = (_RCPL + wpfloat(1.0)) * CPD
CPL = SPECIFIC_HEAT_CAPACITY_LIQUID_WATER

#: density of liquid water. Originally expressed as rhow in ICON. [kg/m3]
WATER_DENSITY: Final[wpfloat] = wpfloat(1.000e3)

#: specific heat capacity of ice. Originally expressed as ci in ICON. [J/K/kg]
SPECIFIC_HEAT_CAPACITY_ICE: Final[wpfloat] = wpfloat(2108.0)

#: Melting temperature of ice/snow [K]. Originally expressed as tmelt in ICON.
MELTING_TEMPERATURE: Final[wpfloat] = wpfloat(273.15)

#: Latent heat of vaporisation for water [J/kg]. Originally expressed as alv in ICON.
LATENT_HEAT_FOR_VAPORISATION: Final[wpfloat] = wpfloat(2.5008e6)

#: Latent heat of sublimation for water [J/kg]. Originally expressed as als in ICON.
LATENT_HEAT_FOR_SUBLIMATION: Final[wpfloat] = wpfloat(2.8345e6)

#: Latent heat of fusion for water [J/kg]. Originally expressed as alf in ICON.
LATENT_HEAT_FOR_FUSION: Final[wpfloat] = (
    LATENT_HEAT_FOR_SUBLIMATION - LATENT_HEAT_FOR_VAPORISATION
)

#: Triple point of water at 611hPa [K]
WATER_TRIPLE_POINT_TEMPERATURE: Final[wpfloat] = wpfloat(273.16)

#: RV/RD - 1, tvmpc1 in ICON.
RV_O_RD_MINUS_1: Final[wpfloat] = GAS_CONSTANT_WATER_VAPOR / GAS_CONSTANT_DRY_AIR - 1.0
TVMPC1: Final[wpfloat] = RV_O_RD_MINUS_1

#: Av. gravitational acceleration [m/s^2]
GRAVITATIONAL_ACCELERATION: Final[wpfloat] = wpfloat(9.80665)
GRAV: Final[wpfloat] = GRAVITATIONAL_ACCELERATION
GRAV_O_RD: Final[wpfloat] = GRAV / RD
GRAV_O_CPD: Final[wpfloat] = GRAV / CPD

#: reference pressure for Exner function [Pa]
REFERENCE_PRESSURE: Final[wpfloat] = wpfloat(100000.0)
P0REF: Final[wpfloat] = REFERENCE_PRESSURE
RD_O_P0REF: Final[wpfloat] = RD / P0REF

#: sea level pressure [Pa]
SEA_LEVEL_PRESSURE: Final[wpfloat] = wpfloat(101325.0)
P0SL_BG: Final[wpfloat] = SEA_LEVEL_PRESSURE

# average earth radius in [m]
EARTH_RADIUS: Final[wpfloat] = wpfloat(6.371229e6)

#: Earth angular velocity [rad/s]
EARTH_ANGULAR_VELOCITY: Final[wpfloat] = wpfloat(7.29212e-5)

#: sea level temperature for reference atmosphere [K]
SEA_LEVEL_TEMPERATURE: Final[wpfloat] = wpfloat(288.15)
T0SL_BG: Final[wpfloat] = SEA_LEVEL_TEMPERATURE

#: difference between sea level temperature and asymptotic stratospheric temperature
DELTA_TEMPERATURE: Final[wpfloat] = wpfloat(75.0)
DEL_T_BG: Final[wpfloat] = DELTA_TEMPERATURE

#: height scale for reference atmosphere [m], defined  in mo_vertical_grid
#: scale height [m]
HEIGHT_SCALE_FOR_REFERENCE_ATMOSPHERE = wpfloat(10000.0)
_H_SCAL_BG: Final[wpfloat] = HEIGHT_SCALE_FOR_REFERENCE_ATMOSPHERE

# Math constants
DBL_EPS = sys.float_info.epsilon  # EPSILON(1._wp)

# Implementation constants
#: default dynamics to physics time step ratio
DEFAULT_DYNAMICS_TO_PHYSICS_TIMESTEP_RATIO: Final[float] = 5.0


class PhysicsConstants(eve_utils.FrozenNamespace[wpfloat]):
    """
    Constants used in gt4py stencils.
    """

    rd = GAS_CONSTANT_DRY_AIR
    rv = GAS_CONSTANT_WATER_VAPOR
    rv_o_rd_minus_1 = RV_O_RD_MINUS_1
    cvd = SPECIFIC_HEAT_CAPACITY_VOLUME_DRY_AIR
    cpd = SPECIFIC_HEAT_CAPACITY_PRESSURE_DRY_AIR
    cpv = SPECIFIC_HEAT_CAPACITY_PRESSURE_WATER_VAPOR
    cvv = SPECIFIC_HEAT_CAPACITY_VOLUME_WATER_VAPOR
    cpl = SPECIFIC_HEAT_CAPACITY_LIQUID_WATER
    cpi = SPECIFIC_HEAT_CAPACITY_ICE
    water_density = WATER_DENSITY
    tmelt = MELTING_TEMPERATURE
    water_triple_point_temperature = WATER_TRIPLE_POINT_TEMPERATURE
    lh_vaporise = LATENT_HEAT_FOR_VAPORISATION
    lh_sublimate = LATENT_HEAT_FOR_SUBLIMATION
    lh_fusion = LATENT_HEAT_FOR_FUSION
    rd_o_cpd = RD_O_CPD
    rd_o_cvd = RD_O_CVD
    cpd_o_rd = CPD_O_RD
    cvd_o_rd = CVD_O_RD
    rd_o_p0ref = RD_O_P0REF
    grav_o_cpd = GRAV_O_CPD
    grav_o_rd = GRAV_O_RD
    p0ref = REFERENCE_PRESSURE
    eps = DBL_EPS


class RayleighType(eve_utils.FrozenNamespace[int]):
    #: classical Rayleigh damping, which makes use of a reference state.
    CLASSIC = 1
    #: Klemp (2008) type Rayleigh damping
    KLEMP = 2
