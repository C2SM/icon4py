# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import enum
import math

from icon4py.model.common import type_alias as ta
from icon4py.model.common.constants import PhysicsConstants


class MicrophysicsConstants(ta.wpfloat, enum.Enum):
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
    TETENS_P0 = ta.wpfloat(610.78)
    #: aw in Tetens formula for saturation water pressure. Originally expressed as c3les in ICON.
    TETENS_AW = ta.wpfloat(17.269)
    #: bw in Tetens formula for saturation water pressure. Originally expressed as c4les in ICON.
    TETENS_BW = ta.wpfloat(35.86)
    #: numerator in temperature partial derivative of Tetens formula for saturation water pressure (psat tetens_der / (t - tetens_bw)^2). Originally expressed as c5les in ICON.
    TETENS_DER = TETENS_AW * (PhysicsConstants.tmelt - TETENS_BW)
    #: ai in Tetens formula for saturation ice water pressure, see eq. 5.35 in the COSMO microphysics documentation, p = p0 exp(ai(T - T_triplepoint)/(T - bi)). Originally expressed as c3ies in ICON.
    TETENS_AI = ta.wpfloat(21.875)
    #: bi in Tetens formula for saturation ice water pressure. Originally expressed as c4ies in ICON.
    TETENS_BI = ta.wpfloat(7.66)

    #: threshold temperature for heterogeneous freezing of raindrops. Originally expressed as trfrz in ICON.
    THRESHOLD_FREEZE_TEMPERATURE = ta.wpfloat(271.15)
    #: FR: 1. coefficient for immersion raindrop freezing: alpha_if, see eq. 5.168 in the COSMO microphysics documentation. Originally expressed as crfrz1 in ICON.
    COEFF_RAIN_FREEZE1 = ta.wpfloat(9.95e-5)
    #: FR: 2. coefficient for immersion raindrop freezing: a_if, see eq. 5.168 in the COSMO microphysics documentation. Originally expressed as crfrz2 in ICON.
    COEFF_RAIN_FREEZE2 = ta.wpfloat(0.66)
    #: temperature for hom. freezing of cloud water. Originally expressed as thn in ICON.
    HOMOGENEOUS_FREEZE_TEMPERATURE = ta.wpfloat(236.15)
    #: threshold temperature for mixed-phase cloud freezing of cloud drops (Forbes 2012, Forbes & Ahlgrimm 2014), see eq. 5.166 in the COSMO microphysics documentation. Originally expressed as tmix in ICON.
    THRESHOLD_FREEZE_TEMPERATURE_MIXEDPHASE = ta.wpfloat(250.15)
    #: threshold for lowest detectable mixing ratios.
    QMIN = (
        5 * PhysicsConstants.eps
    )  # (1.11e-15 for wpfloat==gtx.float64, originally 1.0e-15 for double)
    #: exponential factor in ice terminal velocity equation v = zvz0i*rhoqi^zbvi, see eq. 5.169 in the COSMO microphysics documentation. Originally expressed as bvi in ICON.
    POWER_LAW_EXPONENT_FOR_ICE_MEAN_FALL_SPEED = ta.wpfloat(0.16)
    #: reference air density. Originally expressed as rho0 in ICON.
    REF_AIR_DENSITY = ta.wpfloat(1.225e0)
    #: in m/s; minimum terminal fall velocity of rain particles (applied only near the ground). Originally expressed as v_sedi_rain_min in ICON.
    MINIMUM_RAIN_FALL_SPEED = ta.wpfloat(0.7)
    #: in m/s; minimum terminal fall velocity of snow particles (applied only near the ground). Originally expressed as v_sedi_snow_min in ICON.
    MINIMUM_SNOW_FALL_SPEED = ta.wpfloat(0.1)
    #: in m/s; minimum terminal fall velocity of graupel particles (applied only near the ground). Originally expressed as v_sedi_graupel_min in ICON.
    MINIMUM_GRAUPEL_FALL_SPEED = ta.wpfloat(0.4)
    #: maximal number concentration of ice crystals, see eq. 5.165.
    NIMAX_THOM = ta.wpfloat(250.0e3)
    #: Formfactor in the mass-diameter relation of snow particles, see eq. 5.159 in the COSMO microphysics documentation. Originally expressed as ams in ICON.
    POWER_LAW_COEFF_FOR_SNOW_MD_RELATION = ta.wpfloat(0.069)
    #: A constant intercept parameter for inverse exponential size distribution of snow particles, see eq. 5.160 in the COSMO microphysics documentation. Originally expressed as n0s0 in ICON.
    SNOW_DEFAULT_INTERCEPT_PARAM = ta.wpfloat(8.0e5)
    #: exponent of mixing ratio in the collection equation where cloud or ice particles are rimed by graupel (exp=(3+b)/(1+beta), v=a D^b, m=alpha D^beta), see eqs. 5.152 to 5.154 in the COSMO microphysics documentation. Originally expressed as rimexp_g in ICON.
    GRAUPEL_RIMEXP = ta.wpfloat(0.94878)
    #: exponent of mixing ratio in the graupel mean terminal velocity-mixing ratio relationship (exp=b/(1+beta)), see eq. 5.156 in the COSMO microphysics documentation. Originally expressed as expsedg in ICON.
    POWER_LAW_EXPONENT_FOR_GRAUPEL_MEAN_FALL_SPEED = ta.wpfloat(0.217)
    #: power law coefficient in the graupel mean terminal velocity-mixing ratio relationship, see eq. 5.156 in the COSMO microphysics documentation. Originally expressed as vz0g in ICON.
    POWER_LAW_COEFF_FOR_GRAUPEL_MEAN_FALL_SPEED = ta.wpfloat(12.24)
    #: initial crystal mass for cloud ice nucleation, see eq. 5.101 in the COSMO microphysics documentation. Originally expressed as mi0 in ICON.
    ICE_INITIAL_MASS = ta.wpfloat(1.0e-12)
    #: maximum mass of cloud ice crystals to avoid too large ice crystals near melting point, see eq. 5.105 in the COSMO microphysics documentation. Originally expressed as mimax in ICON.
    ICE_MAX_MASS = ta.wpfloat(1.0e-9)
    #: initial mass of snow crystals which is used in ice-ice autoconversion to snow particles, see eq. 5.108 in the COSMO microphysics documentation. Originally expressed as msmin in ICON.
    SNOW_MIN_MASS = ta.wpfloat(3.0e-9)
    #: Scaling factor [1/K] for temperature-dependent cloud ice sticking efficiency, see eq. 5.163 in the COSMO microphysics documentation. Originally expressed as ceff_min in ICON.
    ICE_STICKING_EFF_FACTOR = ta.wpfloat(3.5e-3)
    #: Temperature at which cloud ice autoconversion starts, see eq. 5.163 in the COSMO microphysics documentation.
    TMIN_ICEAUTOCONV = ta.wpfloat(188.15)
    #: Reference length for distance from cloud top (Forbes 2012), see eq. 5.166 in the COSMO microphysics documentation.
    DIST_CLDTOP_REF = ta.wpfloat(500.0)
    #: lower bound on snow/ice deposition reduction, see eq. 5.166 in the COSMO microphysics documentation.
    REDUCE_DEP_REF = ta.wpfloat(0.1)
    #: Howell factor in depositional growth equation, see eq. 5.71 and eqs. 5.103 & 5.104 in the COSMO microphysics documentation. Originally expressed as hw in ICON.
    HOWELL_FACTOR = ta.wpfloat(2.270603)
    #: Collection efficiency for snow collecting cloud water, see eq. 5.113 in the COSMO microphysics documentation. Originally expressed as ecs in ICON.
    SNOW_CLOUD_COLLECTION_EFF = ta.wpfloat(0.9)
    #: Exponent in the terminal velocity for snow, see unnumbered eq. (v = 25 D^0.5) below eq. 5.159 in the COSMO microphysics documentation. Originally expressed as v1s in ICON.
    POWER_LAW_EXPONENT_FOR_SNOW_FALL_SPEED = ta.wpfloat(0.5)
    #: kinematic viscosity of air. Originally expressed as eta in ICON.
    AIR_KINEMATIC_VISCOSITY = ta.wpfloat(1.75e-5)
    #: molecular diffusion coefficient for water vapour. Originally expressed as dv in ICON.
    DIFFUSION_COEFF_FOR_WATER_VAPOR = ta.wpfloat(2.22e-5)
    #: thermal conductivity of dry air. Originally expressed as lheat in ICON.
    THERMAL_CONDUCTIVITY_DRY_AIR = ta.wpfloat(2.40e-2)
    #: Exponent in the mass-diameter relation of snow particles, see eq. 5.159 in the COSMO microphysics documentation. Originally expressed as bms in ICON.
    POWER_LAW_EXPONENT_FOR_SNOW_MD_RELATION = ta.wpfloat(2.0)
    #: Formfactor in the mass-diameter relation of cloud ice, see eq. 5.90 in the COSMO microphysics documentation. Originally expressed as ami in ICON.
    POWER_LAW_EXPONENT_FOR_ICE_MD_RELATION = ta.wpfloat(130.0)
    #: specific heat of water vapor J, at constant pressure (Landolt-Bornstein). NOTE THAT THIS IS DIFFERENT FROM VALUE USED IN THE MODEL CONSTANTS [J/K/kg]
    CP_V = ta.wpfloat(1850.0)

    RCPD = ta.wpfloat(1.0) / PhysicsConstants.cpd
    RCVD = ta.wpfloat(1.0) / PhysicsConstants.cvd

    #: parameter for snow intercept parameter when snow_intercept_option=FIELD_BEST_FIT_ESTIMATION, see Field et al. (2005). Originally expressed as zn0s1 in ICON.
    SNOW_INTERCEPT_PARAMETER_N0S1 = ta.wpfloat(13.5) * ta.wpfloat(5.65e5)
    #: parameter for snow intercept parameter when snow_intercept_option=FIELD_BEST_FIT_ESTIMATION, see Field et al. (2005). Originally expressed as zn0s2 in ICON.
    SNOW_INTERCEPT_PARAMETER_N0S2 = ta.wpfloat(-0.107)
    #: parameter for snow intercept parameter when snow_intercept_option=FIELD_GENERAL_MOMENT_ESTIMATION. Originally expressed as mma in ICON.
    SNOW_INTERCEPT_PARAMETER_MMA1 = ta.wpfloat(5.065339)
    SNOW_INTERCEPT_PARAMETER_MMA2 = ta.wpfloat(-0.062659)
    SNOW_INTERCEPT_PARAMETER_MMA3 = ta.wpfloat(-3.032362)
    SNOW_INTERCEPT_PARAMETER_MMA4 = ta.wpfloat(0.029469)
    SNOW_INTERCEPT_PARAMETER_MMA5 = ta.wpfloat(-0.000285)
    SNOW_INTERCEPT_PARAMETER_MMA6 = ta.wpfloat(0.312550)
    SNOW_INTERCEPT_PARAMETER_MMA7 = ta.wpfloat(0.000204)
    SNOW_INTERCEPT_PARAMETER_MMA8 = ta.wpfloat(0.003199)
    SNOW_INTERCEPT_PARAMETER_MMA9 = ta.wpfloat(0.000000)
    SNOW_INTERCEPT_PARAMETER_MMA10 = ta.wpfloat(-0.015952)
    # #: parameter for snow intercept parameter when snow_intercept_option=FIELD_GENERAL_MOMENT_ESTIMATION. Originally expressed as mmb in ICON.
    SNOW_INTERCEPT_PARAMETER_MMB1 = ta.wpfloat(0.476221)
    SNOW_INTERCEPT_PARAMETER_MMB2 = ta.wpfloat(-0.015896)
    SNOW_INTERCEPT_PARAMETER_MMB3 = ta.wpfloat(0.165977)
    SNOW_INTERCEPT_PARAMETER_MMB4 = ta.wpfloat(0.007468)
    SNOW_INTERCEPT_PARAMETER_MMB5 = ta.wpfloat(-0.000141)
    SNOW_INTERCEPT_PARAMETER_MMB6 = ta.wpfloat(0.060366)
    SNOW_INTERCEPT_PARAMETER_MMB7 = ta.wpfloat(0.000079)
    SNOW_INTERCEPT_PARAMETER_MMB8 = ta.wpfloat(0.000594)
    SNOW_INTERCEPT_PARAMETER_MMB9 = ta.wpfloat(0.000000)
    SNOW_INTERCEPT_PARAMETER_MMB10 = ta.wpfloat(-0.003577)
    #: temperature for het. nuc. of cloud ice. Originally expressed as thet in ICON.
    HETEROGENEOUS_FREEZE_TEMPERATURE = ta.wpfloat(248.15)
    #: autoconversion coefficient (cloud water to rain). Originally expressed as ccau in ICON.
    KESSLER_CLOUD2RAIN_AUTOCONVERSION_COEFF_FOR_CLOUD = ta.wpfloat(4.0e-4)
    #: (15/32)*(PI**0.5)*(ECR/RHOW)*V0R*AR**(1/8) when Kessler (1969) is used for cloud-cloud autoconversion. Originally expressed as cac in ICON.
    KESSLER_CLOUD2RAIN_AUTOCONVERSION_COEFF_FOR_RAIN = ta.wpfloat(1.72)
    #: constant in phi-function for Seifert-Beheng (2001) autoconversion.
    KPHI1 = ta.wpfloat(6.00e02)
    #: exponent in phi-function for Seifert-Beheng (2001) autoconversion.
    KPHI2 = ta.wpfloat(0.68e00)
    #: exponent in phi-function for Seifert-Beheng (2001) accretion.
    KPHI3 = ta.wpfloat(5.00e-05)
    #: kernel coeff for Seifert-Beheng (2001) autoconversion.
    KCAU = ta.wpfloat(9.44e09)
    #: kernel coeff for Seifert-Beheng (2001) accretion.
    KCAC = ta.wpfloat(5.25e00)
    #: gamma exponent for cloud distribution in Seifert-Beheng (2001) autoconverssion.
    CNUE = ta.wpfloat(2.00e00)
    #: separating mass between cloud and rain in Seifert-Beheng (2001) autoconverssion.
    XSTAR = ta.wpfloat(2.60e-10)

    #: coefficient for graupel riming
    CRIM_G = ta.wpfloat(4.43)
    CAGG_G = ta.wpfloat(2.46)
    #: autoconversion coefficient (cloud ice to snow)
    CIAU = ta.wpfloat(1.0e-3)
    #: initial mass of snow crystals
    MSMIN = ta.wpfloat(3.0e-9)
    #: (15/32)*(PI**0.5)*(EIR/RHOW)*V0R*AR**(1/8)
    CICRI = ta.wpfloat(1.72)
    #: (PI/24)*EIR*V0R*Gamma(6.5)*AR**(-5/8)
    CRCRI = ta.wpfloat(1.24e-3)
    #: DIFF*LH_v*RHO/LHEAT
    ASMEL = ta.wpfloat(2.95e3)
    #: factor in calculation of critical temperature
    TCRIT = ta.wpfloat(3339.5)

    #: minimum specific cloud content [kg/kg]
    QC0 = ta.wpfloat(0.0)
    #: minimum specific ice content [kg/kg]
    QI0 = ta.wpfloat(0.0)

    #: ice crystal number concentration at threshold temperature for mixed-phase cloud
    NIMIX = ta.wpfloat(5.0) * math.exp(
        ta.wpfloat(0.304) * (PhysicsConstants.tmelt - THRESHOLD_FREEZE_TEMPERATURE_MIXEDPHASE)
    )

    CCSDEP = (
        ta.wpfloat(0.26)
        * math.gamma((POWER_LAW_EXPONENT_FOR_SNOW_FALL_SPEED + ta.wpfloat(5.0)) / ta.wpfloat(2.0))
        * math.sqrt(ta.wpfloat(1.0) / AIR_KINEMATIC_VISCOSITY)
    )
    _ccsvxp = -(
        POWER_LAW_EXPONENT_FOR_SNOW_FALL_SPEED
        / (POWER_LAW_EXPONENT_FOR_SNOW_MD_RELATION + ta.wpfloat(1.0))
        + ta.wpfloat(1.0)
    )
    CCSVXP = _ccsvxp + ta.wpfloat(1.0)
    CCSLAM = POWER_LAW_COEFF_FOR_SNOW_MD_RELATION * math.gamma(
        POWER_LAW_EXPONENT_FOR_SNOW_MD_RELATION + ta.wpfloat(1.0)
    )
    CCSLXP = ta.wpfloat(1.0) / (POWER_LAW_EXPONENT_FOR_SNOW_MD_RELATION + ta.wpfloat(1.0))
    CCSWXP = POWER_LAW_EXPONENT_FOR_SNOW_FALL_SPEED * CCSLXP
    CCSAXP = -(POWER_LAW_EXPONENT_FOR_SNOW_FALL_SPEED + ta.wpfloat(3.0))
    CCSDXP = -(POWER_LAW_EXPONENT_FOR_SNOW_FALL_SPEED + ta.wpfloat(1.0)) / ta.wpfloat(2.0)
    CCSHI1 = (
        PhysicsConstants.lh_sublimate
        * PhysicsConstants.lh_sublimate
        / (THERMAL_CONDUCTIVITY_DRY_AIR * PhysicsConstants.rv)
    )
    CCDVTP = (
        ta.wpfloat(2.22e-5) * PhysicsConstants.tmelt ** (ta.wpfloat(-1.94)) * ta.wpfloat(101325.0)
    )
    CCIDEP = ta.wpfloat(4.0) * POWER_LAW_EXPONENT_FOR_ICE_MD_RELATION ** (
        ta.wpfloat(-1.0) / ta.wpfloat(3.0)
    )
    CCSWXP_LN1O2 = math.exp(CCSWXP * math.log(ta.wpfloat(0.5)))

    PVSW0 = TETENS_P0 * math.exp(
        TETENS_AW
        * (PhysicsConstants.tmelt - PhysicsConstants.tmelt)
        / (PhysicsConstants.tmelt - TETENS_BW)
    )
