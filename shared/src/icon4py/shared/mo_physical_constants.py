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

"""
Physical constants for the ICON general circulation models.

They are grouped as follows:
- Natural constants
- Molar weights
- Earth and Earth orbit constants
- Thermodynamic constants for the dry and moist atmosphere
- Constants used for the computation of lookup tables of the saturation
mixing ratio over liquid water (*c_les*) or ice(*c_ies*)
"""

from typing import Final

from gt4py.eve.utils import FrozenNamespace


class PhysicalConstants(FrozenNamespace):
    # Natural constants (WMO/SI values)
    # ---------------------------------

    avo = 6.02214179e23
    """ Avogadro constant [1/mo] """
    ak = 1.3806504e-23
    """ Boltzmann constant [J/K] """
    argas = 8.314472
    """ molar/universal/ideal gas constant [J/K/mol] """
    stbo = 5.6704e-8
    """ Stephan-Boltzmann constant [W/m2/K4] """

    # Molar weights
    # -------------

    # Pure species
    amco2 = 44.011
    """ Molar weight of CO2 [g/mol] """
    amch4 = 16.043
    """ Molar weight of CH4 [g/mol] """
    amo3 = 47.9982
    """ Molar weight of O3 [g/mol] """
    amo2 = 31.9988
    """ Molar weight of O2 [g/mol] """
    amn2o = 44.013
    """ Molar weight of N2O [g/mol] """
    amc11 = 137.3686
    """ Molar weight of CFC11 [g/mol] """
    amc12 = 120.9140
    """ Molar weight of CFC12 [g/mol] """
    amw = 18.0154
    """ Molar weight of H2O [g/mol] """
    amo = 15.9994
    """ Molar weight of O [g/mol] """
    amno = 30.0061398
    """ Molar weight of NO [g/mol] """
    amn2 = 28.0134
    """ Molar weight of N2 [g/mol] """

    # Mixed species
    amd = 28.970  # [g/mol] dry air
    """ Molar weight of dry air [g/mol] """

    # Auxiliary constants
    # ppmv2gg converts ozone from volume mixing ratio in ppmv
    # to mass mixing ratio in g/g
    ppmv2gg = 1.0e-6 * amo3 / amd
    o3mr2gg = amo3 / amd

    # Earth and Earth orbit constants
    # -------------------------------

    earth_radius = 6.371229e6
    """Average Radius of Earth [m]"""
    inverse_earth_radius = 1.0 / earth_radius
    """Inverse Radius of Earth [1/m]"""
    earth_angular_velocity = 7.29212e-5
    """Angular velocity Earth[rad/s] """

    rae = 0.1277e-2
    """ Ratio of atm. scale height to Earth radius [m/m]"""

    # WMO/SI values
    grav = 9.80665
    """Average gravitational acceleration of Earth [m/s2]"""
    rgrav = 1.0 / grav
    """Inverse gravitational acceleration of Earth [s2/m]"""

    # Thermodynamic constants for the dry and moist atmosphere
    # --------------------------------------------------------

    # Dry air
    rd = 287.04
    """Gas constant of dry air [J/K/kg]"""
    cpd = 1004.64
    """Specific heat of dry air at constant pressure [J/K/kg]"""
    cvd = cpd - rd
    """"Specific heat of dry air at constant volume [J/K/kg]"""
    con_m = 1.50e-5
    """Kinematic viscosity of dry air [m^2/s]"""
    con_h = 2.20e-5
    """calar conductivity of dry air [m^2/s]"""
    con0_h = 2.40e-2
    """Thermal conductivity of dry air [J/m/s/K]"""
    eta0d = 1.717e-5
    """Dynamic viscosity of dry air at tmelt [N*s/m2] """

    # H2O
    # - gas
    rv = 461.51
    """Gas constant of water vapor [J/K/kg] """
    cpv = 1869.46
    """Specific heat of water vapour at constant pressure [J/K/kg] """
    cvv = cpv - rv
    """Specific heat of water vapour at constant volume [J/K/kg] """
    dv0 = 2.22e-5
    """Diffusion coefficient of water vapor in dry air at tmelt [m^2/s]"""
    # - liquid / water
    rhoh2o = 1000.0
    """Density of liquid water [kg/m3]  """
    # - solid / ice
    rhoice = 916.7
    """Density of pure ice [kg/m3]"""
    cv_i = 2000.0

    # - phase changes
    alv = 2.5008e6
    """Latent heat of vaporisation for water [J/kg]"""
    als = 2.8345e6
    """Latent heat of sublimation for water [J/kg]"""
    alf = als - alv
    """latent heat of fusion for water [J/kg] """
    tmelt = 273.15
    """Melting temperature of ice/snow [K]"""
    t3 = 273.16
    """Triple point of water at 611hPa [K]"""

    # Auxiliary constants
    rdv = rd / rv
    """[ ]"""
    vtmpc1 = rv / rd - 1.0
    """[ ]"""
    vtmpc2 = cpv / cpd - 1.0
    """[ ]"""
    rcpv = cpd / cpv - 1.0
    """[ ]"""
    alvdcp = alv / cpd
    """[K]"""
    alsdcp = als / cpd
    """[K]"""
    rcpd = 1.0 / cpd
    """[K*kg/J]"""
    rcvd = 1.0 / cvd
    """[K*kg/J]"""
    rcpl = 3.1733
    """cp_d / cp_l - 1"""

    clw = (rcpl + 1.0) * cpd
    """Specific heat capacity of liquid water"""
    cv_v = (rcpv + 1.0) * cpd - rv

    o_m_rdv = 1.0 - rd / rv
    """[ ]"""
    rd_o_cpd = rd / cpd
    """[ ]"""
    cvd_o_rd = cvd / rd
    """[ ]"""

    p0ref = 100000.0
    """Reference pressure for Exner function [Pa]"""

    # Variables for computing cloud cover in RH scheme
    uc1 = 0.8
    ucl = 1.00

    dtdz_standardatm = -6.5e-3
    """vertical tropospheric temperature gradient of U.S. standard atmosphere [K/m]"""

    # Constants for radiation module
    # ------------------------------
    zemiss_def = 0.996
    """Longwave surface default emissivity factor"""

    #
    salinity_fac = 0.981
    """Salinity factor for reduced saturation vapor pressure over oceans"""

    # For radar reflectivity calculation
    K_w_0 = 0.93
    """dielectric constant at reference temperature 0°C"""
    K_i_0 = 0.176
    """dielectric constant at reference temperature 0°C"""

    # Parameters for ocean model
    # --------------------------

    # Coefficients in linear EOS
    a_T = 2.55e-04
    """Thermal expansion coefficient of liquid water [kg/m3/K]"""
    b_S = 7.64e-01
    """Haline contraction coefficient of liquid water [kg/m3/psu]"""

    # Density reference values, to be constant in Boussinesq ocean models
    rho_ref = 1025.022
    """Reference density of liquid water [kg/m^3]"""
    rho_inv = 0.0009755881663
    """Inverse reference density of liquid water [m^3/kg]"""
    sal_ref = 35.0
    """reference salinity of liquid water [psu] """

    SItodBar = 1.0e-4
    """Conversion from pressure [p] to pressure [bar]"""
    # used in ocean thermodynamics
    sfc_press_pascal = 101300.0
    """ Surface pressure [Pa] """
    sfc_press_bar = 101300.0 * SItodBar
    """ Surface pressure [bar] """

    p0sl_bg = 101325.0
    """Sea level pressure [Pa]"""

    # Parameters for sea-ice and lake model
    # -------------------------------------

    ks = 0.31
    """Heat conductivity snow [J/m/s/K]"""
    ki = 2.1656
    """Heat conductivity ice [J/m/s/K]"""
    rhoi = 917.0
    """Density of sea ice [kg/m3]"""
    rhos = 300.0
    """Density of snow [kg/m3]"""
    ci = 2106.0
    """Heat capacity of ice [J/kg/K]"""
    cs = 2090.0
    """Heat capacity of snow [J/kg/K]"""
    Tf = -1.80
    """Temperature ice bottom [C]"""
    mu = 0.054
    """Constant in linear freezing"""
    albedoW = 0.07
    """Albedo ocean used in atmosphere"""

    fr_fac = 1.1925
    """Frank Roeske energy budget closing factor for OMIP"""

    # CCSM3 albedo scheme - not used for coupling
    alb_ice_vis = 0.73
    """Albedo of dry ice (visible)"""
    alb_ice_nir = 0.33
    """Albedo of dry ice (near-infrared)"""
    alb_sno_vis = 0.96
    """Albedo of dry snow (visible)"""
    alb_sno_nir = 0.68
    """Albedo of dry snow (near-infrared)"""
    I_0 = 0.17
    """Ice-surface penetrating shortwave fraction"""
    Cd_ia = 1.2e-3
    """Ice-atmosphere drag coefficient"""
    Cd_io = 3.0e-3
    """Ice-ocean drag coefficient"""
    Ch_io = 12.0e-3
    """Ice-ocean heat transfer coefficient"""

    # Parameters for NWP sea-ice model
    # --------------------------------
    tf_salt = 271.45  # (Note: Differs from Tf)
    """salt-water freezing point [K] """

    rdaylen = 86400.0
    """Length of day [s] as float [s]"""
    idaylen = int(rdaylen)
    """Length of day [s] as integer [s]"""


phy_const: Final = PhysicalConstants()
