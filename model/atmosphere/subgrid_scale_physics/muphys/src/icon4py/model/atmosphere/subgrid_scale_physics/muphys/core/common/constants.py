# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import enum

from icon4py.model.common import type_alias as ta


class GraupelConsts(ta.wpfloat, enum.Enum):
    rho_00 = 1.225  # reference air density
    q1 = 8.0e-6
    qmin = 1.0e-15  # threshold for computation
    ams = 0.069  # Formfactor in the mass-size relation of snow particles
    bms = 2.0  # Exponent in the mass-size relation of snow particles
    v0s = 25.0  # prefactor in snow fall speed
    v1s = 0.5  # Exponent in the terminal velocity for snow
    m0_ice = 1.0e-12  # initial crystal mass for cloud ice nucleation
    ci = 2108.0  # specific heat of ice
    tx = 3339.5
    tfrz_het1 = 267.15  # temperature for het. freezing of cloud water with supersat => TMELT - 6.0
    tfrz_het2 = 248.15  # temperature for het. freezing of cloud water => TMELT - 25.0
    tfrz_hom = 236.15  # temperature for hom. freezing of cloud water => TMELT - 37.0
    lvc = 3135383.2031928  # invariant part of vaporization enthalpy => alv - (cpv - clw) * tmelt
    lsc = 2899657.201  # invariant part of vaporization enthalpy => als - (cpv - ci) * tmelt


class ThermodynamicConsts(ta.wpfloat, enum.Enum):
    # Thermodynamic constants for the dry and moist atmosphere
    # Dry air
    rd = 287.04  # [J/K/kg] gas constant
    cpd = 1004.64  # [J/K/kg] specific heat at constant pressure
    cvd = 717.60  # [J/K/kg] specific heat at constant volume => cpd - rd
    con_m = 1.50e-5  # [m^2/s]  kinematic viscosity of dry air
    con_h = 2.20e-5  # [m^2/s]  scalar conductivity of dry air
    con0_h = 2.40e-2  # [J/m/s/K] thermal conductivity of dry air
    eta0d = 1.717e-5  # [N*s/m2] dyn viscosity of dry air at tmelt
    # H2O
    # gas
    rv = 461.51  # [J/K/kg] gas constant for water vapor
    cpv = 1869.46  # [J/K/kg] specific heat at constant pressure
    cvv = 1407.95  # [J/K/kg] specific heat at constant volume => cpv - rv
    dv0 = 2.22e-5  # [m^2/s]  diff coeff of H2O vapor in dry air at tmelt
    # liquid / water
    rhoh2o = 1000.0  # [kg/m3]  density of liquid water
    # solid / ice
    rhoice = 916.7  # [kg/m3]  density of pure ice
    cv_i = 2000.0
    # phase changes
    alv = 2.5008e6  # [J/kg]   latent heat for vaporisation
    als = 2.8345e6  # [J/kg]   latent heat for sublimation
    alf = 333700.0  # [J/kg]   latent heat for fusion => als - alv
    tmelt = 273.15  # [K]      melting temperature of ice/snow
    t3 = 273.16  # [K]      Triple point of water at 611hPa
    # Auxiliary constants
    rdv = 0.6219583540985028  # [ ] rd / rv
    vtmpc1 = 0.6078246934225193  # [ ] rv / rd - 1.0
    vtmpc2 = 0.8608257684344642  # [ ] cpv / cpd - 1.0
    rcpv = -0.46260417446749336  # [ ] cpd / cpv - 1.0
    alvdcp = 2489.2498805542286  # [K] alv / cpd
    alsdcp = 2821.408663799968  # [K] als / cpd
    rcpd = 0.000995381430164039  # [K*kg/J] 1.0 / cpd
    rcvd = 0.0013935340022296545  # [K*kg/J] 1.0 / cvd
    rcpl = 3.1733  # cp_d / cp_l - 1
    clw = 4192.6641119999995  # specific heat capacity of liquid water (rcpl + 1.0) * cpd
    cv_v = 78.37934216297742  # (rcpv + 1.0) * cpd - rv


class IndexConsts(ta.wpfloat, enum.Enum):
    prefactor_r = 14.58
    exponent_r = 0.111
    offset_r = 1.0e-12
    prefactor_i = 1.25
    exponent_i = 0.160
    offset_i = 1.0e-12
    prefactor_s = 57.80
    exponent_s = 0.16666666666666666
    offset_s = 1.0e-12
    prefactor_g = 12.24
    exponent_g = 0.217
    offset_g = 1.0e-08


class IconNwpConsts(ta.wpfloat, enum.Enum):
    # Constants of the newer MPIM rain-microphysics revisions carried by icon-nwp
    # (mo_aes_graupel.f90), used by the ICON_NWP scheme variants only.
    # Hydrometeor-density clamp bounds, shared by all polynomial fits and fall speeds
    rhox_mn = 3.26216e-08
    rhox_mx = 6.97604e-03
    # cloud_to_rain: accretion kernel, degree-4 polynomial in log(clamped rho*qr)
    a_ac_1 = -2.155543e00
    a_ac_2 = -1.148491e00
    a_ac_3 = -1.882563e-02
    a_ac_4 = 2.941391e-03
    a_ac_5 = 5.575598e-05
    # rain_to_vapor: evaporation, exp of degree-4 polynomial in log(clamped rho*qr)
    a_ev_1 = -5.532194e00
    a_ev_2 = 2.432848e-01
    a_ev_3 = -4.145391e-02
    a_ev_4 = -1.798439e-03
    a_ev_5 = -1.405764e-05
    # rain fall speed: degree-4 polynomial in log(clamped rho*qr); the constant term
    # vm_a_r_1 is NOT multiplied by sqrt(rho_00/rho) in the Fortran (operator
    # precedence in mo_aes_graupel.f90 vm()), reproduced faithfully
    vm_a_r_1 = -5.91051e-01
    vm_a_r_2 = -5.37440e00
    vm_a_r_3 = -1.00459e00
    vm_a_r_4 = -6.44895e-02
    vm_a_r_5 = -1.40361e-03
    # ice / snow / graupel fall speeds: prefactor * clamped_rhox**exponent * density correction
    vm_prefactor_i = 0.80
    vm_exponent_i = 0.160
    vm_prefactor_s = 115.60  # 2 * 57.80
    vm_exponent_s = 0.16666666666666666
    vm_prefactor_g = 12.24
    vm_exponent_g = 0.217
    # snow_number: lower clamp on the snow mass density rho*qs
    rho_s_mn = 2.0e-7
