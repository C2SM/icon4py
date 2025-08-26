# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Final

from gt4py.eve import utils as eve_utils

from icon4py.model.common import constants as physics_constants, type_alias as ta


phy_const: Final = physics_constants.PhysicsConstants()


class MicrophysicsConstants(eve_utils.FrozenNamespace[ta.wpfloat]):
    """
    Constants used for the computation of saturated pressure in saturation adjustment and microphysics.
    It was originally in mo_lookup_tables_constants.f90.

    This class also contains numerical, physical, and empirical constants for the ICON graupel scheme.
    These constants are not configurable from namelists in ICON.
    If users want to tune the model for better results in specific cases, you may need to change the hard coded constants here.
    Users can find the description of all parameters used in this microphyscs scheme in the COSMO microphysics documentation:
    "A Description of the Nonhydrostatic Regional COSMO-Model Part II Physical Parameterizations",
    which can be downloaded via the link given in the docstring of SingleMomentSixClassIconGraupelConfig.
    """

    #: p0 in Tetens formula for saturation water pressure, see eq. 5.33 in COSMO documentation. Originally expressed as c1es in ICON.
    tetens_p0 = 610.78
    #: aw in Tetens formula for saturation water pressure. Originally expressed as c3les in ICON.
    tetens_aw = 17.269
    #: bw in Tetens formula for saturation water pressure. Originally expressed as c4les in ICON.
    tetens_bw = 35.86
    #: numerator in temperature partial derivative of Tetens formula for saturation water pressure (psat tetens_der / (t - tetens_bw)^2). Originally expressed as c5les in ICON.
    tetens_der = tetens_aw * (phy_const.tmelt - tetens_bw)
    #: ai in Tetens formula for saturation ice water pressure, see eq. 5.35 in the COSMO microphysics documentation, p = p0 exp(ai(T - T_triplepoint)/(T - bi)). Originally expressed as c3ies in ICON.
    tetens_ai: Final[ta.wpfloat] = 21.875
    #: bi in Tetens formula for saturation ice water pressure. Originally expressed as c4ies in ICON.
    tetens_bi: Final[ta.wpfloat] = 7.66

    #: threshold temperature for heterogeneous freezing of raindrops. Originally expressed as trfrz in ICON.
    threshold_freeze_temperature: Final[ta.wpfloat] = 271.15
    #: FR: 1. coefficient for immersion raindrop freezing: alpha_if, see eq. 5.168 in the COSMO microphysics documentation. Originally expressed as crfrz1 in ICON.
    coeff_rain_freeze1: Final[ta.wpfloat] = 9.95e-5
    #: FR: 2. coefficient for immersion raindrop freezing: a_if, see eq. 5.168 in the COSMO microphysics documentation. Originally expressed as crfrz2 in ICON.
    coeff_rain_freeze2: Final[ta.wpfloat] = 0.66
    #: temperature for hom. freezing of cloud water. Originally expressed as thn in ICON.
    homogeneous_freeze_temperature: Final[ta.wpfloat] = 236.15
    #: threshold temperature for mixed-phase cloud freezing of cloud drops (Forbes 2012, Forbes & Ahlgrimm 2014), see eq. 5.166 in the COSMO microphysics documentation. Originally expressed as tmix in ICON.
    threshold_freeze_temperature_mixedphase: Final[ta.wpfloat] = 250.15
    #: threshold for lowest detectable mixing ratios.
    qmin: Final[ta.wpfloat] = 1.0e-15
    #: exponential factor in ice terminal velocity equation v = zvz0i*rhoqi^zbvi, see eq. 5.169 in the COSMO microphysics documentation. Originally expressed as bvi in ICON.
    power_law_exponent_for_ice_mean_fall_speed: Final[ta.wpfloat] = 0.16
    #: reference air density. Originally expressed as rho0 in ICON.
    ref_air_density: Final[ta.wpfloat] = 1.225e0
    #: in m/s; minimum terminal fall velocity of rain particles (applied only near the ground). Originally expressed as v_sedi_rain_min in ICON.
    minimum_rain_fall_speed: Final[ta.wpfloat] = 0.7
    #: in m/s; minimum terminal fall velocity of snow particles (applied only near the ground). Originally expressed as v_sedi_snow_min in ICON.
    minimum_snow_fall_speed: Final[ta.wpfloat] = 0.1
    #: in m/s; minimum terminal fall velocity of graupel particles (applied only near the ground). Originally expressed as v_sedi_graupel_min in ICON.
    minimum_graupel_fall_speed: Final[ta.wpfloat] = 0.4
    #: maximal number concentration of ice crystals, see eq. 5.165.
    nimax_thom: Final[ta.wpfloat] = 250.0e3
    #: Formfactor in the mass-diameter relation of snow particles, see eq. 5.159 in the COSMO microphysics documentation. Originally expressed as ams in ICON.
    power_law_coeff_for_snow_mD_relation: Final[ta.wpfloat] = 0.069
    #: A constant intercept parameter for inverse exponential size distribution of snow particles, see eq. 5.160 in the COSMO microphysics documentation. Originally expressed as n0s0 in ICON.
    snow_default_intercept_param: Final[ta.wpfloat] = 8.0e5
    #: exponent of mixing ratio in the collection equation where cloud or ice particles are rimed by graupel (exp=(3+b)/(1+beta), v=a D^b, m=alpha D^beta), see eqs. 5.152 to 5.154 in the COSMO microphysics documentation. Originally expressed as rimexp_g in ICON.
    graupel_rimexp: Final[ta.wpfloat] = 0.94878
    #: exponent of mixing ratio in the graupel mean terminal velocity-mixing ratio relationship (exp=b/(1+beta)), see eq. 5.156 in the COSMO microphysics documentation. Originally expressed as expsedg in ICON.
    power_law_exponent_for_graupel_mean_fall_speed: Final[ta.wpfloat] = 0.217
    #: power law coefficient in the graupel mean terminal velocity-mixing ratio relationship, see eq. 5.156 in the COSMO microphysics documentation. Originally expressed as vz0g in ICON.
    power_law_coeff_for_graupel_mean_fall_speed: Final[ta.wpfloat] = 12.24
    #: initial crystal mass for cloud ice nucleation, see eq. 5.101 in the COSMO microphysics documentation. Originally expressed as mi0 in ICON.
    ice_initial_mass: Final[ta.wpfloat] = 1.0e-12
    #: maximum mass of cloud ice crystals to avoid too large ice crystals near melting point, see eq. 5.105 in the COSMO microphysics documentation. Originally expressed as mimax in ICON.
    ice_max_mass: Final[ta.wpfloat] = 1.0e-9
    #: initial mass of snow crystals which is used in ice-ice autoconversion to snow particles, see eq. 5.108 in the COSMO microphysics documentation. Originally expressed as msmin in ICON.
    snow_min_mass: Final[ta.wpfloat] = 3.0e-9
    #: Scaling factor [1/K] for temperature-dependent cloud ice sticking efficiency, see eq. 5.163 in the COSMO microphysics documentation. Originally expressed as ceff_min in ICON.
    ice_sticking_eff_factor: Final[ta.wpfloat] = 3.5e-3
    #: Temperature at which cloud ice autoconversion starts, see eq. 5.163 in the COSMO microphysics documentation.
    tmin_iceautoconv: Final[ta.wpfloat] = 188.15
    #: Reference length for distance from cloud top (Forbes 2012), see eq. 5.166 in the COSMO microphysics documentation.
    dist_cldtop_ref: Final[ta.wpfloat] = 500.0
    #: lower bound on snow/ice deposition reduction, see eq. 5.166 in the COSMO microphysics documentation.
    reduce_dep_ref: Final[ta.wpfloat] = 0.1
    #: Howell factor in depositional growth equation, see eq. 5.71 and eqs. 5.103 & 5.104 in the COSMO microphysics documentation. Originally expressed as hw in ICON.
    howell_factor: Final[ta.wpfloat] = 2.270603
    #: Collection efficiency for snow collecting cloud water, see eq. 5.113 in the COSMO microphysics documentation. Originally expressed as ecs in ICON.
    snow_cloud_collection_eff: Final[ta.wpfloat] = 0.9
    #: Exponent in the terminal velocity for snow, see unnumbered eq. (v = 25 D^0.5) below eq. 5.159 in the COSMO microphysics documentation. Originally expressed as v1s in ICON.
    power_law_exponent_for_snow_fall_speed: Final[ta.wpfloat] = 0.5
    #: kinematic viscosity of air. Originally expressed as eta in ICON.
    air_kinemetic_viscosity: Final[ta.wpfloat] = 1.75e-5
    #: molecular diffusion coefficient for water vapour. Originally expressed as dv in ICON.
    diffusion_coeff_for_water_vapor: Final[ta.wpfloat] = 2.22e-5
    #: thermal conductivity of dry air. Originally expressed as lheat in ICON.
    thermal_conductivity_dry_air: Final[ta.wpfloat] = 2.40e-2
    #: Exponent in the mass-diameter relation of snow particles, see eq. 5.159 in the COSMO microphysics documentation. Originally expressed as bms in ICON.
    power_law_exponent_for_snow_mD_relation: Final[ta.wpfloat] = 2.0
    #: Formfactor in the mass-diameter relation of cloud ice, see eq. 5.90 in the COSMO microphysics documentation. Originally expressed as ami in ICON.
    power_law_exponent_for_ice_mD_relation: Final[ta.wpfloat] = 130.0
    #: specific heat of water vapor J, at constant pressure (Landolt-Bornstein). NOTE THAT THIS IS DIFFERENT FROM VALUE USED IN THE MODEL CONSTANTS [J/K/kg]
    cp_v: Final[ta.wpfloat] = 1850.0

    rcpd: Final[ta.wpfloat] = 1.0 / phy_const.cpd
    rcvd: Final[ta.wpfloat] = 1.0 / phy_const.cvd

    #: parameter for snow intercept parameter when snow_intercept_option=FIELD_BEST_FIT_ESTIMATION, see Field et al. (2005). Originally expressed as zn0s1 in ICON.
    snow_intercept_parameter_n0s1: Final[ta.wpfloat] = 13.5 * 5.65e5
    #: parameter for snow intercept parameter when snow_intercept_option=FIELD_BEST_FIT_ESTIMATION, see Field et al. (2005). Originally expressed as zn0s2 in ICON.
    snow_intercept_parameter_n0s2: Final[ta.wpfloat] = -0.107
    #: parameter for snow intercept parameter when snow_intercept_option=FIELD_GENERAL_MOMENT_ESTIMATION. Originally expressed as mma in ICON.
    snow_intercept_parameter_mma: tuple[
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
    ] = (
        5.065339,
        -0.062659,
        -3.032362,
        0.029469,
        -0.000285,
        0.312550,
        0.000204,
        0.003199,
        0.000000,
        -0.015952,
    )
    #: parameter for snow intercept parameter when snow_intercept_option=FIELD_GENERAL_MOMENT_ESTIMATION. Originally expressed as mmb in ICON.
    snow_intercept_parameter_mmb: tuple[
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
    ] = (
        0.476221,
        -0.015896,
        0.165977,
        0.007468,
        -0.000141,
        0.060366,
        0.000079,
        0.000594,
        0.000000,
        -0.003577,
    )

    #: temperature for het. nuc. of cloud ice. Originally expressed as thet in ICON.
    heterogeneous_freeze_temperature: Final[ta.wpfloat] = 248.15
    #: autoconversion coefficient (cloud water to rain). Originally expressed as ccau in ICON.
    kessler_cloud2rain_autoconversion_coeff_for_cloud: Final[ta.wpfloat] = 4.0e-4
    #: (15/32)*(PI**0.5)*(ECR/RHOW)*V0R*AR**(1/8) when Kessler (1969) is used for cloud-cloud autoconversion. Originally expressed as cac in ICON.
    kessler_cloud2rain_autoconversion_coeff_for_rain: Final[ta.wpfloat] = 1.72
    #: constant in phi-function for Seifert-Beheng (2001) autoconversion.
    kphi1: Final[ta.wpfloat] = 6.00e02
    #: exponent in phi-function for Seifert-Beheng (2001) autoconversion.
    kphi2: Final[ta.wpfloat] = 0.68e00
    #: exponent in phi-function for Seifert-Beheng (2001) accretion.
    kphi3: Final[ta.wpfloat] = 5.00e-05
    #: kernel coeff for Seifert-Beheng (2001) autoconversion.
    kcau: Final[ta.wpfloat] = 9.44e09
    #: kernel coeff for Seifert-Beheng (2001) accretion.
    kcac: Final[ta.wpfloat] = 5.25e00
    #: gamma exponent for cloud distribution in Seifert-Beheng (2001) autoconverssion.
    cnue: Final[ta.wpfloat] = 2.00e00
    #: separating mass between cloud and rain in Seifert-Beheng (2001) autoconverssion.
    xstar: Final[ta.wpfloat] = 2.60e-10

    #: p0 in Tetens formula for saturation water pressure, see eq. 5.33 in the COSMO microphysics documentation, p = p0 exp(aw(T - T_triplepoint)/(T - bw)). Originally expressed as c1es in ICON.
    tetens_p0: Final[ta.wpfloat] = 610.78
    #: aw in Tetens formula for saturation water pressure. Originally expressed as c3les in ICON.
    tetens_aw: Final[ta.wpfloat] = 17.269
    #: bw in Tetens formula for saturation water pressure. Originally expressed as c4les in ICON.
    tetens_bw: Final[ta.wpfloat] = 35.86

    #: coefficient for graupel riming
    crim_g: Final[ta.wpfloat] = 4.43
    cagg_g: Final[ta.wpfloat] = 2.46
    #: autoconversion coefficient (cloud ice to snow)
    ciau: Final[ta.wpfloat] = 1.0e-3
    #: initial mass of snow crystals
    msmin: Final[ta.wpfloat] = 3.0e-9
    #: (15/32)*(PI**0.5)*(EIR/RHOW)*V0R*AR**(1/8)
    cicri: Final[ta.wpfloat] = 1.72
    #: (PI/24)*EIR*V0R*Gamma(6.5)*AR**(-5/8)
    crcri: Final[ta.wpfloat] = 1.24e-3
    #: DIFF*LH_v*RHO/LHEAT
    asmel: Final[ta.wpfloat] = 2.95e3
    #: factor in calculation of critical temperature
    tcrit: Final[ta.wpfloat] = 3339.5

    #: minimum specific cloud content [kg/kg]
    qc0: Final[ta.wpfloat] = 0.0
    #: minimum specific ice content [kg/kg]
    qi0: Final[ta.wpfloat] = 0.0

    #: ice crystal number concentration at threshold temperature for mixed-phase cloud
    nimix: Final[ta.wpfloat] = 5.0 * math.exp(
        0.304 * (phy_const.tmelt - threshold_freeze_temperature_mixedphase)
    )

    ccsdep: Final[ta.wpfloat] = (
        0.26
        * math.gamma((power_law_exponent_for_snow_fall_speed + 5.0) / 2.0)
        * math.sqrt(1.0 / air_kinemetic_viscosity)
    )
    _ccsvxp: Final[ta.wpfloat] = -(
        power_law_exponent_for_snow_fall_speed / (power_law_exponent_for_snow_mD_relation + 1.0)
        + 1.0
    )
    ccsvxp: Final[ta.wpfloat] = _ccsvxp + 1.0
    ccslam: Final[ta.wpfloat] = power_law_coeff_for_snow_mD_relation * math.gamma(
        power_law_exponent_for_snow_mD_relation + 1.0
    )
    ccslxp: Final[ta.wpfloat] = 1.0 / (power_law_exponent_for_snow_mD_relation + 1.0)
    ccswxp: Final[ta.wpfloat] = power_law_exponent_for_snow_fall_speed * ccslxp
    ccsaxp: Final[ta.wpfloat] = -(power_law_exponent_for_snow_fall_speed + 3.0)
    ccsdxp: Final[ta.wpfloat] = -(power_law_exponent_for_snow_fall_speed + 1.0) / 2.0
    ccshi1: Final[ta.wpfloat] = (
        phy_const.lh_sublimate
        * phy_const.lh_sublimate
        / (thermal_conductivity_dry_air * phy_const.rv)
    )
    ccdvtp: Final[ta.wpfloat] = 2.22e-5 * phy_const.tmelt ** (-1.94) * 101325.0
    ccidep: Final[ta.wpfloat] = 4.0 * power_law_exponent_for_ice_mD_relation ** (-1.0 / 3.0)
    ccswxp_ln1o2: Final[ta.wpfloat] = math.exp(ccswxp * math.log(0.5))

    pvsw0: Final[ta.wpfloat] = tetens_p0 * math.exp(
        tetens_aw * (phy_const.tmelt - phy_const.tmelt) / (phy_const.tmelt - tetens_bw)
    )
