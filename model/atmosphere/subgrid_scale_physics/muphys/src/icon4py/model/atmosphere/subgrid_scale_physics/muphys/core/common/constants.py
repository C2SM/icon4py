# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# noqa: RUF012

class thermodyn:
    # Thermodynamic constants for the dry and moist atmosphere

    # Dry air
    rd = 287.04  # [J/K/kg] gas constant
    cpd = 1004.64  # [J/K/kg] specific heat at constant pressure
    cvd = cpd - rd  # [J/K/kg] specific heat at constant volume
    con_m = 1.50e-5  # [m^2/s]  kinematic viscosity of dry air
    con_h = 2.20e-5  # [m^2/s]  scalar conductivity of dry air
    con0_h = 2.40e-2  # [J/m/s/K] thermal conductivity of dry air
    eta0d = 1.717e-5  # [N*s/m2] dyn viscosity of dry air at tmelt

    # H2O
    # gas
    rv = 461.51  # [J/K/kg] gas constant for water vapor
    cpv = 1869.46  # [J/K/kg] specific heat at constant pressure
    cvv = cpv - rv  # [J/K/kg] specific heat at constant volume
    dv0 = 2.22e-5  # [m^2/s]  diff coeff of H2O vapor in dry air at tmelt
    # liquid / water
    rhoh2o = 1000.0  # [kg/m3]  density of liquid water
    # solid / ice
    rhoice = 916.7  # [kg/m3]  density of pure ice

    cv_i = 2000.0

    # phase changes
    alv = 2.5008e6  # [J/kg]   latent heat for vaporisation
    als = 2.8345e6  # [J/kg]   latent heat for sublimation
    alf = als - alv  # [J/kg]   latent heat for fusion
    tmelt = 273.15  # [K]      melting temperature of ice/snow
    t3 = 273.16  # [K]      Triple point of water at 611hPa

    # Auxiliary constants
    rdv = rd / rv  # [ ]
    vtmpc1 = rv / rd - 1.0  # [ ]
    vtmpc2 = cpv / cpd - 1.0  # [ ]
    rcpv = cpd / cpv - 1.0  # [ ]
    alvdcp = alv / cpd  # [K]
    alsdcp = als / cpd  # [K]
    rcpd = 1.0 / cpd  # [K*kg/J]
    rcvd = 1.0 / cvd  # [K*kg/J]
    rcpl = 3.1733  # cp_d / cp_l - 1

    clw = (rcpl + 1.0) * cpd  # specific heat capacity of liquid water
    cv_v = (rcpv + 1.0) * cpd - rv


class graupel_ct:
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
    tfrz_het1 = thermodyn.tmelt - 6.0  # temperature for het. freezing of cloud water with supersat
    tfrz_het2 = thermodyn.tmelt - 25.0  # temperature for het. freezing of cloud water
    tfrz_hom = thermodyn.tmelt - 37.0  # temperature for hom. freezing of cloud water
    lvc = (
        thermodyn.alv - (thermodyn.cpv - thermodyn.clw) * thermodyn.tmelt
    )  # invariant part of vaporization enthalpy
    lsc = (
        thermodyn.als - (thermodyn.cpv - ci) * thermodyn.tmelt
    )  # invariant part of vaporization enthalpy


class idx:
    nx = 6  # number of water species
    np = 4  # number of precipitating water species
    lqr = 0  # index for rain
    lqi = 1  # index for ice
    lqs = 2  # index for snow
    lqg = 3  # index for graupel
    lqc = 4  # index for cloud
    lqv = 5  # index for vapor

    qx_ind: list = [lqv, lqc, lqr, lqs, lqi, lqg]
    qp_ind: list = [lqr, lqi, lqs, lqg]

    lrain = True  # switch for disabling rain
    lcold = True  # switch for disabling freezing processes

    prefactor_r = 14.58
    exponent_r = 0.111
    offset_r = 1.0e-12
    prefactor_i = 1.25
    exponent_i = 0.160
    offset_i = 1.0e-12
    prefactor_s = 57.80
    exponent_s = 0.5
    offset_s = 1.0e-12
    prefactor_g = 12.24
    exponent_g = 0.217
    offset_g = 1.0e-08

    params: list = [
        [14.58, 0.111, 1.0e-12],
        [1.25, 0.160, 1.0e-12],
        [57.80, 0.5 / 3.0, 1.0e-12],
        [12.24, 0.217, 1.0e-08],
    ]

    cloud_num = 200.00e06  # cloud droplet number concentration (from gscp_data.f90)

    ZERO = 0.0
