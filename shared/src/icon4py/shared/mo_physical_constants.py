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
This module provides physical constants for the ICON general circulation models.

Physical constants are grouped as follows:
- Natural constants
- Molar weights
- Earth and Earth orbit constants
- Thermodynamic constants for the dry and moist atmosphere
- Constants used for the computation of lookup tables of the saturation
mixing ratio over liquid water (*c_les*) or ice(*c_ies*)
"""


# Natural constants
# -----------------
#

# WMO/SI values
avo = 6.02214179e23  # [1/mo]    Avogadro constant
ak = 1.3806504e-23  # [J/K]     Boltzmann constant
argas = 8.314472  # [J/K/mol] molar/universal/ideal gas constant
stbo = 5.6704e-8  # [W/m2/K4] Stephan-Boltzmann constant


# Molar weights
# -------------

# Pure species
amco2 = 44.011  # [g/mol] CO2
amch4 = 16.043  # [g/mol] CH4
amo3 = 47.9982  # [g/mol] O3
amo2 = 31.9988  # [g/mol] O2
amn2o = 44.013  # [g/mol] N2O
amc11 = 137.3686  # [g/mol] CFC11
amc12 = 120.9140  # [g/mol] CFC12
amw = 18.0154  # [g/mol] H2O
amo = 15.9994  # [g/mol] O
amno = 30.0061398  # [g/mol] NO
amn2 = 28.0134  # [g/mol] N2

# Mixed species
amd = 28.970  # [g/mol] dry air

# Auxiliary constants
# ppmv2gg converts ozone from volume mixing ratio in ppmv
# to mass mixing ratio in g/g
ppmv2gg = 1.0e-6 * amo3 / amd
o3mr2gg = amo3 / amd

# Earth and Earth orbit constants
# -------------------------------

earth_radius = 6.371229e6  # [m]    average radius
inverse_earth_radius = 1.0 / earth_radius  # [1/m]
earth_angular_velocity = 7.29212e-5  # [rad/s]  angular velocity


# WMO/SI value
grav = 9.80665  # [m/s2] av. gravitational acceleration
rgrav = 1.0 / grav  # [s2/m]

rae = 0.1277e-2  # [m/m]  ratio of atm. scale height to Earth radius


# Thermodynamic constants for the dry and moist atmosphere
# --------------------------------------------------------

# Dry air
rd = 287.04  # [J/K/kg] gas constant
cpd = 1004.64  # [J/K/kg] specific heat at constant pressure
cvd = cpd - rd  # [J/K/kg] specific heat at constant volume
con_m = 1.50e-5  # [m^2/s]  kinematic viscosity of dry air
con_h = 2.20e-5  # [m^2/s]  scalar conductivity of dry air
con0_h = 2.40e-2  # [J/m/s/K]thermal conductivity of dry air
eta0d = 1.717e-5  # [N*s/m2] dyn viscosity of dry air at tmelt

# H2O
# - gas
rv = 461.51  # [J/K/kg] gas constant for water vapor
cpv = 1869.46  # [J/K/kg] specific heat at constant pressure
cvv = cpv - rv  # [J/K/kg] specific heat at constant volume
dv0 = 2.22e-5  # [m^2/s]  diff coeff of H2O vapor in dry air at tmelt
# - liquid / water
rhoh2o = 1000.0  # [kg/m3]  density of liquid water
# - solid / ice
rhoice = 916.7  # [kg/m3]  density of pure ice
cv_i = 2000.0
# - phase changes
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

o_m_rdv = 1.0 - rd / rv  # [ ]
rd_o_cpd = rd / cpd  # [ ]
cvd_o_rd = cvd / rd  # [ ]

p0ref = 100000.0  # [Pa]  reference pressure for Exner function

# Variables for computing cloud cover in RH scheme
uc1 = 0.8
ucl = 1.00

# U.S. standard atmosphere vertical tropospheric temperature gradient
dtdz_standardatm = -6.5e-3  # [ K/m ]

# constants for radiation module
zemiss_def = 0.996  # lw sfc default emissivity factor

# salinity factor for reduced saturation vapor pressure over oceans
salinity_fac = 0.981

# dielectric constants at reference temperature 0°C for radar reflectivity calculation:
K_w_0 = 0.93
K_i_0 = 0.176

# ------------below are parameters for ocean model---------------
# coefficients in linear EOS
a_T = 2.55e-04  # thermal expansion coefficient (kg/m3/K)
b_S = 7.64e-01  # haline contraction coefficient (kg/m3/psu)

# density reference values, to be constant in Boussinesq ocean models
rho_ref = 1025.022  # reference density [kg/m^3]
rho_inv = 0.0009755881663  # inverse reference density [m^3/kg]
sal_ref = 35.0  # reference salinity [psu]

SItodBar = 1.0e-4  # Conversion from pressure [p] to pressure [bar]
# used in ocean thermodynamics
sfc_press_pascal = 101300.0
sfc_press_bar = 101300.0 * SItodBar

p0sl_bg = 101325.0  # [Pa]     sea level pressure

# ----------below are parameters for sea-ice and lake model---------------

ks = 0.31  # heat conductivity snow     [J  / (m s K)]
ki = 2.1656  # heat conductivity ice      [J  / (m s K)]
rhoi = 917.0  # density of sea ice         [kg / m**3]
rhos = 300.0  # density of snow            [kg / m**3]
ci = 2106.0  # Heat capacity of ice       [J / (kg K)]
cs = 2090.0  # Heat capacity of snow      [J / (kg K)]

Tf = -1.80  # Temperature ice bottom     [C]
mu = 0.054  # Constant in linear freezing-
albedoW = 0.07  # albedo of the ocean used in atmosphere


fr_fac = 1.1925  # Frank Roeske energy budget closing factor for OMIP

# CCSM3 albedo scheme - not used for coupling
alb_ice_vis = 0.73  # Albedo of dry ice  (visible)
alb_ice_nir = 0.33  # Albedo of dry ice  (near-infrared)
alb_sno_vis = 0.96  # Albedo of dry snow (visible)
alb_sno_nir = 0.68  # Albedo of dry snow (near-infrared)
I_0 = 0.17  # Ice-surface penetrating shortwave fraction
Cd_ia = 1.2e-3  # Ice-atmosphere drag coefficient
Cd_io = 3.0e-3  # Ice-ocean drag coefficient
Ch_io = 12.0e-3  # Ice-ocean heat transfer coefficient

# --------- parameters for NWP sea-ice model (we should agree on a single value)-----
# _cdm>
# The value of the salt-water freezing point is the same as in GME and COSMO (-1.7 dgr C).
# Note that a different value (Tf=-1.8 dgr C) is defined in "mo_physical_constants".
# _cdm<

tf_salt = 271.45  # salt-water freezing point [K] (note that it differs from Tf)

# Length of day in seconds, as integer and real
rdaylen = 86400.0  # [s]
idaylen = int(rdaylen)  # [s]
