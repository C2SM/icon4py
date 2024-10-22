# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import dataclasses
import enum
import math
import sys
from typing import Final, Optional

import gt4py.next as gtx
import numpy as np
from gt4py.eve.utils import FrozenNamespace
from gt4py.next import backend
from gt4py.next.ffront.decorator import field_operator, program, scan_operator
from gt4py.next.ffront.fbuiltins import (
    broadcast,
    exp,
    log,
    maximum,
    minimum,
    sqrt,
    where,
)

from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    saturation_adjustment as satad,
)
from icon4py.model.common import constants as phy_const
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, vertical as v_grid
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
    tracer_state as tracers,
)
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc


# TODO (Chia Rui): The limit has to be manually set to a huge value for a big scan operator. Remove it when neccesary.
sys.setrecursionlimit(350000)


class LiquidAutoConversion(enum.IntEnum):
    """
    Options for computing liquid auto conversion rate
    """

    #: Kessler (1969) liquid auto conversion mode
    KESSLER = 0
    #: Seifert & Beheng (2006) liquid auto conversion mode
    SEIFERT_BEHENG = 1


class SnowInterceptParameter(enum.IntEnum):
    """
    Options for deriving snow intercept parameter
    """

    #: Estimated intercept parameter for the snow size distribution from the best-fit line in figure 10(a) of Field et al. (2005)
    FIELD_BEST_FIT_ESTIMATION = 1
    #: Estimated intercept parameter for the snow size distribution from the general moment equation in table 2 of Field et al. (2005)
    FIELD_GENERAL_MOMENT_ESTIMATION = 2


@dataclasses.dataclass(frozen=True)
class SingleMomentSixClassIconGraupelConfig:
    """
    Contains necessary parameters to configure icon graupel microphysics scheme.

    Encapsulates namelist parameters.
    Values should be read from configuration.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    lsedi_ice (option for whether ice sedimendation is performed), lstickeff (option for different empirical formulae of sticking efficiency), and lred_depgrow (option for reduced depositional growth) are removed because they are set to True in the original code gscp_graupel.f90.
    lpres_pri is removed because it is a duplicated option for lsedi_ice.
    ldiag_ttend, and ldiag_qtend are removed because default outputs in icon4py physics granules include tendencies.
    The COSMO microphysics documentation "A Description of the Nonhydrostatic Regional COSMO-Model Part II Physical Parameterizations" can be downloaded via this link: https://www.cosmo-model.org/content/model/cosmo/coreDocumentation/cosmo_physics_6.00.pdf
    """

    #: execute saturation adjustment right after microphysics. Originally defined as lsatad as input to nwp_microphysics in mo_nwp_gscp_interface.f90 in ICON.
    do_saturation_adjustment: bool = True
    #: liquid auto conversion mode. Originally defined as isnow_n0temp (PARAMETER) in gscp_data.f90 in ICON. I keep it because I think the choice depends on resolution.
    liquid_autoconversion_option: LiquidAutoConversion = LiquidAutoConversion.KESSLER
    #: snow size distribution interception parameter. Originally defined as isnow_n0temp (PARAMETER) in gscp_data.f90 in ICON. I keep it because I think the choice depends on resolution.
    snow_intercept_option: SnowInterceptParameter = (
        SnowInterceptParameter.FIELD_GENERAL_MOMENT_ESTIMATION
    )
    #: Determine whether the microphysical processes are isochoric or isobaric. Originally defined as l_cv as input to the graupel in ICON. It is hardcoded to True, I still keep it because I do not understand the reason for its existence and isobaric may have a positive effect on reducing sound waves.
    is_isochoric: bool = True
    #: Do latent heat nudging. Originally defined as dass_lhn in mo_run_config.f90 in ICON.
    do_latent_heat_nudging = False
    #: Whether a fixed latent heat capacities are used for water. Originally defined as ithermo_water in mo_atm_phy_nwp_config.f90 in ICON (0 means True).
    use_constant_water_heat_capacity = True
    #: First parameter in RHS of eq. 5.163 in the COSMO microphysics documentation for the sticking efficiency when lstickeff = True (repricated in icon4py because it is always True in ICON). Originally defined as tune_zceff_min in mo_tuning_nwp_config.f90 in ICON.
    ice_stickeff_min: wpfloat = 0.075
    #: Power law coefficient in v-qi ice terminal velocity-mixing ratio relationship, see eq. 5.169 in the COSMO microphysics documentation. Originally defined as tune_zvz0i in mo_tuning_nwp_config.f90 in ICON.
    power_law_coeff_for_ice_mean_fall_speed: wpfloat = 1.25
    #: Exponent of the density factor in ice terminal velocity equation to account for density (air thermodynamic state) change. Originally defined as tune_icesedi_exp in mo_tuning_nwp_config.f90 in ICON.
    exponent_for_density_factor_in_ice_sedimentation: wpfloat = 0.33
    #: Power law coefficient in v-D snow terminal velocity-Diameter relationship, see eqs. 5.57 (depricated after COSMO 3.0) and unnumbered eq. (v = 25 D^0.5) below eq. 5.159 in the COSMO microphysics documentation. Originally defined as tune_v0snow in mo_tuning_nwp_config.f90 in ICON.
    power_law_coeff_for_snow_fall_speed: wpfloat = 20.0
    #: mu exponential factor in gamma distribution of rain particles. Originally defined as mu_rain in mo_atm_phy_nwp_config.f90 in ICON.
    rain_mu: wpfloat = 0.0
    #: Interception parameter in gamma distribution of rain particles. Originally defined as rain_n0_factor in mo_atm_phy_nwp_config.f90 in ICON.
    rain_n0: wpfloat = 1.0


# TODO (Chia Rui): Refactor these constants when the restriction of FrozenNameSpace for compile-time constants is lifted.
class SingleMomentSixClassIconGraupelParams(FrozenNamespace):
    """
    Contains numerical, physical, and empirical constants for the ICON graupel scheme.

    These constants are not configurable from namelists in ICON.
    If users want to tune the model for better results in specific cases, you may need to change the hard coded constants here.
    Users can find the description of all parameters used in this microphyscs scheme in the COSMO microphysics documentation:
    "A Description of the Nonhydrostatic Regional COSMO-Model Part II Physical Parameterizations",
    which can be downloaded via the link given in the docstring of SingleMomentSixClassIconGraupelConfig.
    """

    #: threshold temperature for heterogeneous freezing of raindrops. Originally expressed as trfrz in ICON.
    threshold_freeze_temperature: wpfloat = 271.15
    #: FR: 1. coefficient for immersion raindrop freezing: alpha_if, see eq. 5.168 in the COSMO microphysics documentation. Originally expressed as crfrz1 in ICON.
    coeff_rain_freeze1: wpfloat = 9.95e-5
    #: FR: 2. coefficient for immersion raindrop freezing: a_if, see eq. 5.168 in the COSMO microphysics documentation. Originally expressed as crfrz2 in ICON.
    coeff_rain_freeze2: wpfloat = 0.66
    #: temperature for hom. freezing of cloud water. Originally expressed as thn in ICON.
    homogeneous_freeze_temperature: wpfloat = 236.15
    #: threshold temperature for mixed-phase cloud freezing of cloud drops (Forbes 2012, Forbes & Ahlgrimm 2014), see eq. 5.166 in the COSMO microphysics documentation. Originally expressed as tmix in ICON.
    threshold_freeze_temperature_mixedphase: wpfloat = 250.15
    #: threshold for lowest detectable mixing ratios.
    qmin: wpfloat = 1.0e-15
    #: a small number for cloud existence criterion.
    eps: wpfloat = phy_const.DBL_EPS
    #: exponential factor in ice terminal velocity equation v = zvz0i*rhoqi^zbvi, see eq. 5.169 in the COSMO microphysics documentation. Originally expressed as bvi in ICON.
    power_law_exponent_for_ice_mean_fall_speed: wpfloat = 0.16
    #: reference air density. Originally expressed as rho0 in ICON.
    ref_air_density: wpfloat = 1.225e0
    #: in m/s; minimum terminal fall velocity of rain particles (applied only near the ground). Originally expressed as v_sedi_rain_min in ICON.
    minimum_rain_fall_speed: wpfloat = 0.7
    #: in m/s; minimum terminal fall velocity of snow particles (applied only near the ground). Originally expressed as v_sedi_snow_min in ICON.
    minimum_snow_fall_speed: wpfloat = 0.1
    #: in m/s; minimum terminal fall velocity of graupel particles (applied only near the ground). Originally expressed as v_sedi_graupel_min in ICON.
    minimum_graupel_fall_speed: wpfloat = 0.4
    #: maximal number concentration of ice crystals, see eq. 5.165.
    nimax_thom: wpfloat = 250.0e3
    #: Formfactor in the mass-diameter relation of snow particles, see eq. 5.159 in the COSMO microphysics documentation. Originally expressed as ams in ICON.
    power_law_coeff_for_snow_mD_relation: wpfloat = 0.069
    #: A constant intercept parameter for inverse exponential size distribution of snow particles, see eq. 5.160 in the COSMO microphysics documentation. Originally expressed as n0s0 in ICON.
    snow_default_intercept_param: wpfloat = 8.0e5
    #: exponent of mixing ratio in the collection equation where cloud or ice particles are rimed by graupel (exp=(3+b)/(1+beta), v=a D^b, m=alpha D^beta), see eqs. 5.152 to 5.154 in the COSMO microphysics documentation. Originally expressed as rimexp_g in ICON.
    graupel_rimexp: wpfloat = 0.94878
    #: exponent of mixing ratio in the graupel mean terminal velocity-mixing ratio relationship (exp=b/(1+beta)), see eq. 5.156 in the COSMO microphysics documentation. Originally expressed as expsedg in ICON.
    power_law_exponent_for_graupel_mean_fall_speed: wpfloat = 0.217
    #: power law coefficient in the graupel mean terminal velocity-mixing ratio relationship, see eq. 5.156 in the COSMO microphysics documentation. Originally expressed as vz0g in ICON.
    power_law_coeff_for_graupel_mean_fall_speed: wpfloat = 12.24
    #: initial crystal mass for cloud ice nucleation, see eq. 5.101 in the COSMO microphysics documentation. Originally expressed as mi0 in ICON.
    ice_initial_mass: wpfloat = 1.0e-12
    #: maximum mass of cloud ice crystals to avoid too large ice crystals near melting point, see eq. 5.105 in the COSMO microphysics documentation. Originally expressed as mimax in ICON.
    ice_max_mass: wpfloat = 1.0e-9
    #: initial mass of snow crystals which is used in ice-ice autoconversion to snow particles, see eq. 5.108 in the COSMO microphysics documentation. Originally expressed as msmin in ICON.
    snow_min_mass: wpfloat = 3.0e-9
    #: Scaling factor [1/K] for temperature-dependent cloud ice sticking efficiency, see eq. 5.163 in the COSMO microphysics documentation. Originally expressed as ceff_min in ICON.
    ice_sticking_eff_factor: wpfloat = 3.5e-3
    #: Temperature at which cloud ice autoconversion starts, see eq. 5.163 in the COSMO microphysics documentation.
    tmin_iceautoconv: wpfloat = 188.15
    #: Reference length for distance from cloud top (Forbes 2012), see eq. 5.166 in the COSMO microphysics documentation.
    dist_cldtop_ref: wpfloat = 500.0
    #: lower bound on snow/ice deposition reduction, see eq. 5.166 in the COSMO microphysics documentation.
    reduce_dep_ref: wpfloat = 0.1
    #: Howell factor in depositional growth equation, see eq. 5.71 and eqs. 5.103 & 5.104 in the COSMO microphysics documentation. Originally expressed as hw in ICON.
    howell_factor: wpfloat = 2.270603
    #: Collection efficiency for snow collecting cloud water, see eq. 5.113 in the COSMO microphysics documentation. Originally expressed as ecs in ICON.
    snow_cloud_collection_eff: wpfloat = 0.9
    #: Exponent in the terminal velocity for snow, see unnumbered eq. (v = 25 D^0.5) below eq. 5.159 in the COSMO microphysics documentation. Originally expressed as v1s in ICON.
    power_law_exponent_for_snow_fall_speed: wpfloat = 0.5
    #: kinematic viscosity of air. Originally expressed as eta in ICON.
    air_kinemetic_viscosity: wpfloat = 1.75e-5
    #: molecular diffusion coefficient for water vapour. Originally expressed as dv in ICON.
    diffusion_coeff_for_water_vapor: wpfloat = 2.22e-5
    #: thermal conductivity of dry air. Originally expressed as lheat in ICON.
    dry_air_latent_heat: wpfloat = 2.40e-2
    #: Exponent in the mass-diameter relation of snow particles, see eq. 5.159 in the COSMO microphysics documentation. Originally expressed as bms in ICON.
    power_law_exponent_for_snow_mD_relation: wpfloat = 2.0
    #: Formfactor in the mass-diameter relation of cloud ice, see eq. 5.90 in the COSMO microphysics documentation. Originally expressed as ami in ICON.
    power_law_exponent_for_ice_mD_relation: wpfloat = 130.0
    #: density of liquid water. Originally expressed as rhow in ICON. [kg/m3]
    water_density: wpfloat = 1.000e3
    #: specific heat of water vapor J, at constant pressure (Landolt-Bornstein). [J/K/kg]
    cp_v: wpfloat = 1850.0
    #: specific heat of ice. Originally expressed as ci in ICON. [J/K/kg]
    specific_heat_capacity_for_ice: wpfloat = 2108.0

    #: parameter for snow intercept parameter when snow_intercept_option=FIELD_BEST_FIT_ESTIMATION, see Field et al. (2005). Originally expressed as zn0s1 in ICON.
    snow_intercept_parameter_n0s1: wpfloat = 13.5 * 5.65e5
    #: parameter for snow intercept parameter when snow_intercept_option=FIELD_BEST_FIT_ESTIMATION, see Field et al. (2005). Originally expressed as zn0s2 in ICON.
    snow_intercept_parameter_n0s2: wpfloat = -0.107
    #: parameter for snow intercept parameter when snow_intercept_option=FIELD_GENERAL_MOMENT_ESTIMATION. Originally expressed as mma in ICON.
    snow_intercept_parameter_mma: tuple[
        wpfloat, wpfloat, wpfloat, wpfloat, wpfloat, wpfloat, wpfloat, wpfloat, wpfloat, wpfloat
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
        wpfloat, wpfloat, wpfloat, wpfloat, wpfloat, wpfloat, wpfloat, wpfloat, wpfloat, wpfloat
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
    heterogeneous_freeze_temperature: wpfloat = 248.15
    #: autoconversion coefficient (cloud water to rain). Originally expressed as ccau in ICON.
    kessler_cloud2rain_autoconversion_coeff_for_cloud: wpfloat = 4.0e-4
    #: (15/32)*(PI**0.5)*(ECR/RHOW)*V0R*AR**(1/8) when Kessler (1969) is used for cloud-cloud autoconversion. Originally expressed as cac in ICON.
    kessler_cloud2rain_autoconversion_coeff_for_rain: wpfloat = 1.72
    #: constant in phi-function for Seifert-Beheng (2001) autoconversion.
    kphi1: wpfloat = 6.00e02
    #: exponent in phi-function for Seifert-Beheng (2001) autoconversion.
    kphi2: wpfloat = 0.68e00
    #: exponent in phi-function for Seifert-Beheng (2001) accretion.
    kphi3: wpfloat = 5.00e-05
    #: kernel coeff for Seifert-Beheng (2001) autoconversion.
    kcau: wpfloat = 9.44e09
    #: kernel coeff for Seifert-Beheng (2001) accretion.
    kcac: wpfloat = 5.25e00
    #: gamma exponent for cloud distribution in Seifert-Beheng (2001) autoconverssion.
    cnue: wpfloat = 2.00e00
    #: separating mass between cloud and rain in Seifert-Beheng (2001) autoconverssion.
    xstar: wpfloat = 2.60e-10

    #: p0 in Tetens formula for saturation water pressure, see eq. 5.33 in the COSMO microphysics documentation, p = p0 exp(aw(T - T_triplepoint)/(T - bw)). Originally expressed as c1es in ICON.
    tetens_p0: wpfloat = 610.78
    #: aw in Tetens formula for saturation water pressure. Originally expressed as c3les in ICON.
    tetens_aw: wpfloat = 17.269
    #: ai in Tetens formula for saturation ice water pressure, see eq. 5.35 in the COSMO microphysics documentation, p = p0 exp(ai(T - T_triplepoint)/(T - bi)). Originally expressed as c3ies in ICON.
    tetens_ai: wpfloat = 21.875
    #: bw in Tetens formula for saturation water pressure. Originally expressed as c4les in ICON.
    tetens_bw: wpfloat = 35.86
    #: bi in Tetens formula for saturation ice water pressure. Originally expressed as c4ies in ICON.
    tetens_bi: wpfloat = 7.66

    #: coefficient for graupel riming
    crim_g: wpfloat = 4.43
    #: coefficient for snow-graupel conversion by riming
    csg: wpfloat = 0.5
    cagg_g: wpfloat = 2.46
    #: autoconversion coefficient (cloud ice to snow)
    ciau: wpfloat = 1.0e-3
    #: initial mass of snow crystals
    msmin: wpfloat = 3.0e-9
    #: (15/32)*(PI**0.5)*(EIR/RHOW)*V0R*AR**(1/8)
    cicri: wpfloat = 1.72
    #: (PI/24)*EIR*V0R*Gamma(6.5)*AR**(-5/8)
    crcri: wpfloat = 1.24e-3
    #: DIFF*LH_v*RHO/LHEAT
    asmel: wpfloat = 2.95e3
    #: factor in calculation of critical temperature
    tcrit: wpfloat = 3339.5

    #: minimum specific cloud content [kg/kg]
    qc0: wpfloat = 0.0
    #: minimum specific ice content [kg/kg]
    qi0: wpfloat = 0.0

    #: Latent heat of vaporisation for water [J/kg]. Originally expressed as alv in ICON.
    latent_heat_for_vaporisation: wpfloat = 2.5008e6
    #: Latent heat of sublimation for water [J/kg]. Originally expressed as als in ICON.
    latent_heat_for_sublimation: wpfloat = 2.8345e6
    #: Melting temperature of ice/snow [K]. Originally expressed as tmelt in ICON.
    melting_temperature: wpfloat = 273.15
    #: Triple point of water at 611hPa [K]
    t3: wpfloat = 273.16

    #: ice crystal number concentration at threshold temperature for mixed-phase cloud
    nimix: wpfloat = 5.0 * np.exp(
        0.304 * (melting_temperature - threshold_freeze_temperature_mixedphase)
    )

    #: Gas constant of dry air [J/K/kg]
    rd: wpfloat = phy_const.RD
    #: Specific heat of dry air at constant pressure [J/K/kg]
    cpd: wpfloat = phy_const.CPD
    #: cpd / cp_water - 1
    rcpl: wpfloat = 3.1733

    #: Gas constant of water vapor [J/K/kg]
    rv: wpfloat = phy_const.RV
    #: Specific heat of water vapour at constant pressure [J/K/kg]
    cpv: wpfloat = 1869.46
    #: Specific heat of water vapour at constant volume [J/K/kg]
    cvv: wpfloat = cpv - rv

    ccsdep: wpfloat = (
        0.26
        * math.gamma((power_law_exponent_for_snow_fall_speed + 5.0) / 2.0)
        * np.sqrt(1.0 / air_kinemetic_viscosity)
    )
    _ccsvxp: wpfloat = -(
        power_law_exponent_for_snow_fall_speed / (power_law_exponent_for_snow_mD_relation + 1.0)
        + 1.0
    )
    ccsvxp: wpfloat = _ccsvxp + 1.0
    ccslam: wpfloat = power_law_coeff_for_snow_mD_relation * math.gamma(
        power_law_exponent_for_snow_mD_relation + 1.0
    )
    ccslxp: wpfloat = 1.0 / (power_law_exponent_for_snow_mD_relation + 1.0)
    ccswxp: wpfloat = power_law_exponent_for_snow_fall_speed * ccslxp
    ccsaxp: wpfloat = -(power_law_exponent_for_snow_fall_speed + 3.0)
    ccsdxp: wpfloat = -(power_law_exponent_for_snow_fall_speed + 1.0) / 2.0
    ccshi1: wpfloat = (
        latent_heat_for_sublimation * latent_heat_for_sublimation / (dry_air_latent_heat * rv)
    )
    ccdvtp: wpfloat = 2.22e-5 * melting_temperature ** (-1.94) * 101325.0
    ccidep: wpfloat = 4.0 * power_law_exponent_for_ice_mD_relation ** (-1.0 / 3.0)
    ccswxp_ln1o2: wpfloat = np.exp(ccswxp * np.log(0.5))

    #: Latent heat of fusion for water [J/kg]. Originally expressed as alf in ICON.
    latent_heat_for_fusion: wpfloat = latent_heat_for_sublimation - latent_heat_for_vaporisation
    #: Specific heat capacity of liquid water [J/kg]. Originally expressed as clw in ICON.
    specific_heat_capacity_for_water: wpfloat = (rcpl + 1.0) * cpd

    #: Specific heat of dry air at constant volume [J/K/kg]
    cvd: wpfloat = phy_const.CVD
    #: [K*kg/J]
    rcpd: wpfloat = 1.0 / cpd
    #: [K*kg/J]"""
    rcvd: wpfloat = 1.0 / cvd

    pvsw0: wpfloat = tetens_p0 * np.exp(
        tetens_aw * (melting_temperature - melting_temperature) / (melting_temperature - tetens_bw)
    )


icon_graupel_params: Final = SingleMomentSixClassIconGraupelParams()


@gtx.field_operator
def _compute_cooper_inp_concentration(temperature: wpfloat) -> wpfloat:
    cnin = 5.0 * exp(0.304 * (icon_graupel_params.melting_temperature - temperature))
    cnin = minimum(cnin, icon_graupel_params.nimax_thom)
    return cnin


@gtx.field_operator
def _compute_snow_interception_and_collision_parameters(
    temperature: wpfloat,
    rho: wpfloat,
    qs: wpfloat,
    ccsvel: wpfloat,
    ccsrim: wpfloat,
    ccsagg: wpfloat,
    power_law_coeff_for_snow_fall_speed: wpfloat,
    llqs: bool,
    snow_intercept_option: gtx.int32,
) -> tuple[wpfloat, wpfloat, wpfloat, wpfloat, wpfloat]:
    """
    Compute the intercept parameter, N0, of the snow exponential size distribution.

    First method: Explained in paragraphs at pages 2008 and 2009 in Field et al. (2005). N0s_23 = (M_2)^4 / (M_3)^3, M_2 = Gamma(3) N0s / lamda^3, M_2 = Gamma(4) N0s / lamda^4, so N0s_23 = 2/27 N0s. And N0s_23 = 5.65E5 exp(-0.107Tc)
    Second method: Eq. 5.160 in the COSMO microphysics documentation, originally in Table 2 in Field et al. (2005).

    Args:
        temperature: air temperature [K]
        rho: air density [kg/m3]
        qs: specific snow content [kg/kg]
        ccsvel: constant for snow sedimentation
        ccsrim: constant for snow riming with clouds
        ccsagg: constant for ice aggregation (becomes snow)
        power_law_coeff_for_snow_fall_speed: power law coefficient in snow v-D relationship
        llqs: snow grid cell
        snow_intercept_option: estimation method for snow intercept parameter
    Returns:

    """
    if llqs:
        # TODO (Chia Rui): SnowInterceptParameter.FIELD_BEST_FIT_ESTIMATION.value does not work.
        if snow_intercept_option == 1:
            # Calculate n0s using the temperature-dependent
            # formula of Field et al. (2005)
            local_tc = temperature - icon_graupel_params.melting_temperature
            local_tc = minimum(local_tc, wpfloat("0.0"))
            local_tc = maximum(local_tc, wpfloat("-40.0"))
            n0s = icon_graupel_params.snow_intercept_parameter_n0s1 * exp(
                icon_graupel_params.snow_intercept_parameter_n0s2 * local_tc
            )
            n0s = minimum(n0s, wpfloat("1.0e9"))
            n0s = maximum(n0s, wpfloat("1.0e6"))

        # TODO (Chia Rui): SnowInterceptParameter.FIELD_GENERAL_MOMENT_ESTIMATION.value does not work.
        elif snow_intercept_option == 2:
            # Calculate n0s using the temperature-dependent moment
            # relations of Field et al. (2005)
            local_tc = temperature - icon_graupel_params.melting_temperature
            local_tc = minimum(local_tc, wpfloat("0.0"))
            local_tc = maximum(local_tc, wpfloat("-40.0"))

            local_nnr = wpfloat("3.0")
            local_hlp = (
                icon_graupel_params.snow_intercept_parameter_mma[0]
                + icon_graupel_params.snow_intercept_parameter_mma[1] * local_tc
                + icon_graupel_params.snow_intercept_parameter_mma[2] * local_nnr
                + icon_graupel_params.snow_intercept_parameter_mma[3] * local_tc * local_nnr
                + icon_graupel_params.snow_intercept_parameter_mma[4] * local_tc**2.0
                + icon_graupel_params.snow_intercept_parameter_mma[5] * local_nnr**2.0
                + icon_graupel_params.snow_intercept_parameter_mma[6] * local_tc**2.0 * local_nnr
                + icon_graupel_params.snow_intercept_parameter_mma[7] * local_tc * local_nnr**2.0
                + icon_graupel_params.snow_intercept_parameter_mma[8] * local_tc**3.0
                + icon_graupel_params.snow_intercept_parameter_mma[9] * local_nnr**3.0
            )
            local_alf = exp(local_hlp * log(wpfloat("10.0")))
            local_bet = (
                icon_graupel_params.snow_intercept_parameter_mmb[0]
                + icon_graupel_params.snow_intercept_parameter_mmb[1] * local_tc
                + icon_graupel_params.snow_intercept_parameter_mmb[2] * local_nnr
                + icon_graupel_params.snow_intercept_parameter_mmb[3] * local_tc * local_nnr
                + icon_graupel_params.snow_intercept_parameter_mmb[4] * local_tc**2.0
                + icon_graupel_params.snow_intercept_parameter_mmb[5] * local_nnr**2.0
                + icon_graupel_params.snow_intercept_parameter_mmb[6] * local_tc**2.0 * local_nnr
                + icon_graupel_params.snow_intercept_parameter_mmb[7] * local_tc * local_nnr**2.0
                + icon_graupel_params.snow_intercept_parameter_mmb[8] * local_tc**3.0
                + icon_graupel_params.snow_intercept_parameter_mmb[9] * local_nnr**3.0
            )

            # Here is the exponent bms=2.0 hardwired# not ideal# (Uli Blahak)
            local_m2s = (
                qs * rho / icon_graupel_params.power_law_coeff_for_snow_mD_relation
            )  # UB rho added as bugfix
            local_m3s = local_alf * exp(local_bet * log(local_m2s))

            local_hlp = icon_graupel_params.snow_intercept_parameter_n0s1 * exp(
                icon_graupel_params.snow_intercept_parameter_n0s2 * local_tc
            )
            n0s = wpfloat("13.50") * local_m2s * (local_m2s / local_m3s) ** 3.0
            n0s = maximum(n0s, wpfloat("0.5") * local_hlp)
            n0s = minimum(n0s, wpfloat("1.0e2") * local_hlp)
            n0s = minimum(n0s, wpfloat("1.0e9"))
            n0s = maximum(n0s, wpfloat("1.0e6"))

        else:
            n0s = icon_graupel_params.snow_default_intercept_param

        # compute integration factor for terminal velocity
        snow_sed0 = ccsvel * exp(icon_graupel_params.ccsvxp * log(n0s))
        # compute constants for riming, aggregation, and deposition processes for snow
        crim = ccsrim * n0s
        cagg = ccsagg * n0s
        cbsdep = icon_graupel_params.ccsdep * sqrt(power_law_coeff_for_snow_fall_speed)
    else:
        n0s = icon_graupel_params.snow_default_intercept_param
        snow_sed0 = wpfloat("0.0")
        crim = wpfloat("0.0")
        cagg = wpfloat("0.0")
        cbsdep = wpfloat("0.0")

    return n0s, snow_sed0, crim, cagg, cbsdep


@gtx.field_operator
def _deposition_nucleation_at_low_temperature_or_in_clouds(
    temperature: wpfloat,
    rho: wpfloat,
    qv: wpfloat,
    qi: wpfloat,
    qvsi: wpfloat,
    cnin: wpfloat,
    dt: wpfloat,
    llqc: bool,
) -> wpfloat:
    """
    Heterogeneous deposition nucleation for low temperatures below a threshold or in clouds.
    When in clouds, we require water saturation for this process (i.e. the existence of cloud water) to exist.
    In this case, heterogeneous deposition nucleation is assumed to occur only when no cloud ice is present and the temperature is below a nucleation threshold.

    ice nucleation rate = ice_initial_mass Ni / rho / dt, Eq. 5.101 in the COSMO microphysics documentation
    ice_initial_mass is the initial ice crystal mass

    Args:
        temperature: air temperature [K]
        rho: air density [kg/m3]
        qv: specific humidity [kg/kg]
        qvsi: saturated vapor mixing ratio over ice
        cnin: number concentration of ice nucleating particles [/m3]
        dt: time step [s]
        llqc: cloud grid cell
    Returns:
        Deposition nucleation rate
    """
    ice_nucleation_rate_v2i = (
        icon_graupel_params.ice_initial_mass / rho * cnin / dt
        if (llqc & (temperature <= wpfloat("267.15")) & (qi <= icon_graupel_params.qmin))
        | (
            (temperature < icon_graupel_params.heterogeneous_freeze_temperature)
            & (qv > wpfloat("8.0e-6"))
            & (qi <= wpfloat("0.0"))
            & (qv > qvsi)
        )
        else wpfloat("0.0")
    )
    return ice_nucleation_rate_v2i


@gtx.field_operator
def _autoconversion_and_rain_accretion(
    temperature: wpfloat,
    qc: wpfloat,
    qr: wpfloat,
    qnc: wpfloat,
    celn7o8qrk: wpfloat,
    llqc: bool,
    liquid_autoconversion_option: gtx.int32,
) -> tuple[wpfloat, wpfloat]:
    """
    Compute the rate of cloud-to-rain autoconversion and the mass of cloud accreted by rain.
    Method 1: liquid_autoconversion_option = LiquidAutoConversion.KESSLER, Kessler (1969)
    Method 2: liquid_autoconversion_option = LiquidAutoConversion.SEIFERT_BEHENG, Seifert and beheng (2001)

    Args:
        temperature: air temperature [K]
        qc: specific cloud content [kg/kg]
        qr: specific rain content [kg/kg]
        qnc: number concentration of CCN [/m3]
        celn7o8qrk: constant (refer to equation or documentation in the docstring above)
        llqc: cloud grid cell
        liquid_autoconversion_option: liquid auto conversion mode
    Returns:
        cloud-to-rain autoconversionn rate, rain-cloud accretion rate
    """
    if llqc & (temperature > icon_graupel_params.homogeneous_freeze_temperature):
        # TODO (Chia Rui): LiquidAutoConversion.KESSLER.value does not work.
        if liquid_autoconversion_option == 0:
            # Kessler(1969) autoconversion rate
            cloud_autoconversion_rate_c2r = (
                icon_graupel_params.kessler_cloud2rain_autoconversion_coeff_for_cloud
                * maximum(qc - icon_graupel_params.qc0, wpfloat("0.0"))
            )
            rain_cloud_collision_rate_c2r = (
                icon_graupel_params.kessler_cloud2rain_autoconversion_coeff_for_rain
                * qc
                * celn7o8qrk
            )

        # TODO (Chia Rui): LiquidAutoConversion.SEIFERT_BEHENG.value does not work.
        elif liquid_autoconversion_option == 1:
            # Seifert and Beheng (2001) autoconversion rate
            local_const = (
                icon_graupel_params.kcau
                / (wpfloat("20.0") * icon_graupel_params.xstar)
                * (icon_graupel_params.cnue + wpfloat("2.0"))
                * (icon_graupel_params.cnue + wpfloat("4.0"))
                / (icon_graupel_params.cnue + wpfloat("1.0")) ** 2.0
            )

            # with constant cloud droplet number concentration qnc
            if qc > wpfloat("1.0e-6"):
                local_tau = minimum(wpfloat("1.0") - qc / (qc + qr), wpfloat("0.9"))
                local_tau = maximum(local_tau, wpfloat("1.0e-30"))
                local_hlp = exp(icon_graupel_params.kphi2 * log(local_tau))
                local_phi = (
                    icon_graupel_params.kphi1 * local_hlp * (wpfloat("1.0") - local_hlp) ** 3.0
                )
                cloud_autoconversion_rate_c2r = (
                    local_const
                    * qc
                    * qc
                    * qc
                    * qc
                    / (qnc * qnc)
                    * (wpfloat("1.0") + local_phi / (wpfloat("1.0") - local_tau) ** 2.0)
                )
                local_phi = (local_tau / (local_tau + icon_graupel_params.kphi3)) ** 4.0
                rain_cloud_collision_rate_c2r = icon_graupel_params.kcac * qc * qr * local_phi
            else:
                cloud_autoconversion_rate_c2r = wpfloat("0.0")
                rain_cloud_collision_rate_c2r = wpfloat("0.0")
        else:
            cloud_autoconversion_rate_c2r = wpfloat("0.0")
            rain_cloud_collision_rate_c2r = wpfloat("0.0")
    else:
        cloud_autoconversion_rate_c2r = wpfloat("0.0")
        rain_cloud_collision_rate_c2r = wpfloat("0.0")

    return cloud_autoconversion_rate_c2r, rain_cloud_collision_rate_c2r


@gtx.field_operator
def _freezing_in_clouds(
    temperature: wpfloat,
    qc: wpfloat,
    qr: wpfloat,
    cscmax: wpfloat,
    csrmax: wpfloat,
    celn7o4qrk: wpfloat,
    llqc: bool,
    llqr: bool,
) -> tuple[wpfloat, wpfloat]:
    """
    Compute the freezing rate of cloud and rain in clouds if there is cloud water and the temperature is above homogeneuous freezing temperature.
    Cloud is frozen to ice. Rain is frozen to graupel.
    (Please refer to the COSMO microphysics documentation via the link given in the docstring of SingleMomentSixClassIconGraupelConfig for all the equations)

    Method 1: rain_freezing_option = 1, Eq. 5.168
    Method 2 (ABANDONED in icon4py): rain_freezing_option = 2, Eq. 5.126, an approximation of combined immersion Eq. 5.80 and contact freezing Eq. 5.83.
    Args:
        temperature: air temperature [K]
        qc: specific cloud content [kg/kg]
        qr: specific rain content [kg/kg]
        cscmax: maximum specific cloud content [kg/kg]
        csrmax maximum specific rain content [kg/kg]
        celn7o4qrk: constant (refer to equation or documentation in the docstring above)
        llqc: cloud grid cell
        llqr: rain grid cell
    Returns:
        cloud freezing rate, rain freezing rate
    """
    if llqc:
        if temperature > icon_graupel_params.homogeneous_freeze_temperature:
            # Calculation of in-cloud rainwater freezing
            if (
                llqr
                & (temperature < icon_graupel_params.threshold_freeze_temperature)
                & (qr > wpfloat("0.1") * qc)
            ):
                rain_freezing_rate_r2g_in_clouds = (
                    icon_graupel_params.coeff_rain_freeze1
                    * (
                        exp(
                            icon_graupel_params.coeff_rain_freeze2
                            * (icon_graupel_params.threshold_freeze_temperature - temperature)
                        )
                        - wpfloat("1.0")
                    )
                    * celn7o4qrk
                )
            else:
                rain_freezing_rate_r2g_in_clouds = wpfloat("0.0")
            cloud_freezing_rate_c2i = wpfloat("0.0")
        else:
            # tg <= tg: ! hom. freezing of cloud and rain water
            cloud_freezing_rate_c2i = cscmax
            rain_freezing_rate_r2g_in_clouds = csrmax
    else:
        cloud_freezing_rate_c2i = wpfloat("0.0")
        rain_freezing_rate_r2g_in_clouds = wpfloat("0.0")

    return cloud_freezing_rate_c2i, rain_freezing_rate_r2g_in_clouds


@gtx.field_operator
def _riming_in_clouds(
    temperature: wpfloat,
    qc: wpfloat,
    crim: wpfloat,
    cslam: wpfloat,
    celnrimexp_g: wpfloat,
    celn3o4qsk: wpfloat,
    llqc: bool,
    llqs: bool,
) -> tuple[wpfloat, wpfloat, wpfloat, wpfloat]:
    """
    Compute the rate of riming by snow and graupel in clouds if there is cloud water and the temperature is above homogeneuous freezing temperature.
    (Please refer to the COSMO microphysics documentation via the link given in the docstring of SingleMomentSixClassIconGraupelConfig for all the equations)

    riming or accretion rate = 1/rho intg_0^inf m_dot f dD (Eq. 5.64)
    m_dot = pi/4 D^2 E(D) v(D) rho qc (Eq. 5.67)

    snow: f = N0 exp(-lamda D), E is constant, m(D) = alpha D^beta, v(D) = v0 D^b
        snow riming = pi/4 qc N0 E v0 Gamma(b+3) / lamda^(b+3), lamda = (alpha N0 Gamma(beta+1) / rhoqs)^(beta+1)

    graupel:
        graupel riming = 4.43 qc rhoqg^0.94878 (Eq 5.152)
        snow to graupel conversion = 0.5 qc rhoqs^0.75 (above Eq 5.132)

    rain shedding is on if temperature is above zero degree celcius. In this case, riming tendencies are converted to rain shedding.

    Args:
        temperature: air temperature [K]
        qc: specific cloud content [kg/kg]
        crim: constant (refer to equation or documentation in the docstring above)
        cslam: constant (refer to equation or documentation in the docstring above)
        celnrimexp_g: constant (refer to equation or documentation in the docstring above)
        celn3o4qsk: constant (refer to equation or documentation in the docstring above)
        llqc: cloud grid cell
        llqs: snow grid cell
    Returns:
        snow-cloud riming rate, graupel-cloud riming rate, rain shed by snow-cloud and graupel-cloud riming rates, snow-graupel autoconversion rate
    """
    if llqc & (temperature > icon_graupel_params.homogeneous_freeze_temperature):
        if llqs:
            snow_riming_rate_c2s = crim * qc * exp(icon_graupel_params.ccsaxp * log(cslam))
        else:
            snow_riming_rate_c2s = wpfloat("0.0")

        graupel_riming_rate_c2g = icon_graupel_params.crim_g * qc * celnrimexp_g

        if temperature >= icon_graupel_params.melting_temperature:
            rain_shedding_rate_c2r = snow_riming_rate_c2s + graupel_riming_rate_c2g
            snow_riming_rate_c2s = wpfloat("0.0")
            graupel_riming_rate_c2g = wpfloat("0.0")
            snow_autoconversion_rate_s2g = wpfloat("0.0")
        else:
            if qc >= icon_graupel_params.qc0:
                snow_autoconversion_rate_s2g = icon_graupel_params.csg * qc * celn3o4qsk
            else:
                snow_autoconversion_rate_s2g = wpfloat("0.0")
            rain_shedding_rate_c2r = wpfloat("0.0")
    else:
        snow_riming_rate_c2s = wpfloat("0.0")
        graupel_riming_rate_c2g = wpfloat("0.0")
        rain_shedding_rate_c2r = wpfloat("0.0")
        snow_autoconversion_rate_s2g = wpfloat("0.0")

    return (
        snow_riming_rate_c2s,
        graupel_riming_rate_c2g,
        rain_shedding_rate_c2r,
        snow_autoconversion_rate_s2g,
    )


@gtx.field_operator
def _reduced_deposition_in_clouds(
    temperature: wpfloat,
    qv_kup: wpfloat,
    qc_kup: wpfloat,
    qi_kup: wpfloat,
    qs_kup: wpfloat,
    qg_kup: wpfloat,
    qvsw_kup: wpfloat,
    dz: wpfloat,
    dist_cldtop_kup: wpfloat,
    k_lev: gtx.int32,
    startmoist_level: gtx.int32,
    is_surface: bool,
    llqc: bool,
) -> tuple[wpfloat, wpfloat]:
    """
    Artificially reduce the deposition rate in clouds.

    Args:
        temperature: air temperature [K]
        qv_kup: specific humidity at k-1 level [kg/kg]
        qc_kup: specific cloud content at k-1 level [kg/kg]
        qi_kup: specific ice content at k-1 level [kg/kg]
        qs_kup: specific snow content at k-1 level [kg/kg]
        qg_kup: specific graupel content at k-1 level [kg/kg]
        qvsw_kup: saturated vapor mixing ratio at k-1 level
        dz: vertical grid spacing
        dist_cldtop_kup: vertical distance to cloud top
        k_lev: current vertical level index
        startmoist_level: the starting vertical level index for moist process
        is_surface: True if the current k level is at the bottom
        llqc: cloud grid cell
    Returns:
        vertical distance to cloud top, reduced factor for ice deposition
    """
    if llqc:
        if (k_lev > startmoist_level) & (not is_surface):
            cqcgk_1 = qi_kup + qs_kup + qg_kup

            # distance from cloud top
            if (qv_kup + qc_kup < qvsw_kup) & (cqcgk_1 < icon_graupel_params.qmin):
                # upper cloud layer
                dist_cldtop = wpfloat("0.0")  # reset distance to upper cloud layer
            else:
                dist_cldtop = dist_cldtop_kup + dz
        else:
            dist_cldtop = dist_cldtop_kup

        if (k_lev > startmoist_level) & (not is_surface):
            # finalizing transfer rates in clouds and calculate depositional growth reduction
            cnin = _compute_cooper_inp_concentration(temperature)
            cfnuc = minimum(cnin / icon_graupel_params.nimix, wpfloat("1.0"))

            # with asymptotic behaviour dz -> 0 (xxx)
            #        reduce_dep = MIN(fnuc + (1.0_wp-fnuc)*(reduce_dep_ref + &
            #                             dist_cldtop(iv)/dist_cldtop_ref + &
            #                             (1.0_wp-reduce_dep_ref)*(zdh/dist_cldtop_ref)**4), 1.0_wp)

            # without asymptotic behaviour dz -> 0
            reduce_dep = cfnuc + (wpfloat("1.0") - cfnuc) * (
                icon_graupel_params.reduce_dep_ref
                + dist_cldtop / icon_graupel_params.dist_cldtop_ref
            )
            reduce_dep = minimum(reduce_dep, wpfloat("1.0"))
        else:
            reduce_dep = wpfloat("1.0")
    else:
        dist_cldtop = dist_cldtop_kup
        reduce_dep = wpfloat("1.0")

    return dist_cldtop, reduce_dep


@gtx.field_operator
def _collision_and_ice_deposition_in_cold_ice_clouds(
    temperature: wpfloat,
    rho: wpfloat,
    qv: wpfloat,
    qi: wpfloat,
    qs: wpfloat,
    qvsi: wpfloat,
    rhoqi_intermediate: wpfloat,
    dt: wpfloat,
    cslam: wpfloat,
    cidep: wpfloat,
    cagg: wpfloat,
    cmi: wpfloat,
    ice_stickeff_min: wpfloat,
    reduce_dep: wpfloat,
    celnrimexp_g: wpfloat,
    celn7o8qrk: wpfloat,
    celn13o8qrk: wpfloat,
    llqi: bool,
) -> tuple[wpfloat, wpfloat, wpfloat, wpfloat, wpfloat, wpfloat, wpfloat, wpfloat, wpfloat]:
    """
    Compute (Please refer to the COSMO microphysics documentation via the link given in the docstring of SingleMomentSixClassIconGraupelConfig for all the equations)
        1. the aggregation of snow and graupel in ice clouds when temperature is below zero degree celcius.
            aggregation rate = 1/rho intg_0^inf m_dot f dD (Eq. 5.64)
            m_dot = pi/4 D^2 E(T) v(D) rho qi (Eq. 5.67)

            snow: f = N0 exp(-lamda D), E changes with temperature, m(D) = alpha D^beta, v(D) = v0 D^b
                snow aggregation = pi/4 qi N0 E(T) v0 Gamma(b+3) / lamda^(b+3), lamda = (alpha N0 Gamma(beta+1) / rhoqs)^(beta+1)

            graupel:
                graupel aggregation = 2.46 qc rhoqg^0.94878 (Eq 5.154)
        2. the autoconversion of ice crystals in ice clouds when temperature is below zero degree celcius and depositional growth.
            iceAutoconversion = max(0.001 (qi - qi0), 0) Eq. 5.133

            ice: f = Ni delta(D-Di), mi = ai Di^3 = rho qi / Nin, v(D) = 0
            ice deposition or sublimation rate = c_dep Ni mi (qv - qvsi), Eq. 5.104

        3. the ice loss and rain loss due to accretion of rain in ice clouds when temperature is below zero degree celcius.

            riming rate = 1/rho intg_0^inf m_dot f dD (Eq. 5.64)
            m_dot(ice loss) = pi/4 D^2 E(T) v(D) rho qi (Eq. 5.67)
            m_dot(rain loss) = pi/4 D^5 E(T) v(D) rho qi (Eq. 5.67)

            rain: f = N0 D^mu exp(-lamda D), E is a constant, m(D) = alpha D^beta, v(D) = v0 D^b, b = 0.5 (Eq. 5.57)
            ice: uniform size=Di and mass=mi

            ice loss = pi/4 qi N0 E v0 Gamma(b+3) / lamda^(b+3), lamda = (alpha N0 Gamma(beta+1) / rhoqs)^(beta+1)

            rain loss = pi/4 qi N0 E v0 Gamma(b+3) / lamda^(b+3), lamda = (alpha N0 Gamma(beta+1) / rhoqs)^(beta+1)

        E(T) = max( 0.02, min( exp(0.09(T - T_0 )), 1.0), C_se (T - T_se )), Eq. 5.163
        Another E(T) = max(0.2, min(exp(0.09(T - T 0 )), 1.0)), Eq. 5.162, based on Lin et al. 1983 is ABANDONNED.

    Args:
        temperature: air temperature [K]
        rho: air density [kg/m3]
        qv: specific humidity [kg/kg]
        qi: specific ice content [kg/kg]
        qs: specific snow content [kg/kg]
        qvsi: saturated vapor mixing ratio over ice
        rhoqi_intermediate: ice mass with sedimendation flux from above [kg/m3]
        dt: time step [s]
        cslam: constant (refer to equation or documentation in the docstring above)
        cidep: constant (refer to equation or documentation in the docstring above)
        cagg: constant (refer to equation or documentation in the docstring above)
        cmi: constant (refer to equation or documentation in the docstring above)
        ice_stickeff_min: constant (refer to equation or documentation in the docstring above)
        reduce_dep: ice deposition reduced factor
        celnrimexp_g: constant (refer to equation or documentation in the docstring above)
        celn7o8qrk: constant (refer to equation or documentation in the docstring above)
        celn13o8qrk: constant (refer to equation or documentation in the docstring above)
        llqi: ice grid cell
    Returns:
        snow-ice accretion rate,
        graupel-ice accretion rate,
        ice-to-snow autoconversion rate,
        depositional growth rate of ice,
        rain-ice accretion ice loss rate,
        rain-ice accretion rain loss rate,
        ice-to-snow deposition autoconversion rate,
        net depositional growth rate of ice,
        net sublimation growth rate of ice
    """
    if (temperature <= icon_graupel_params.melting_temperature) & llqi:
        # Change in sticking efficiency needed in case of cloud ice sedimentation
        # (based on Guenther Zaengls work)
        local_eff = minimum(
            exp(wpfloat("0.09") * (temperature - icon_graupel_params.melting_temperature)),
            wpfloat("1.0"),
        )
        local_eff = maximum(local_eff, ice_stickeff_min)
        local_eff = maximum(
            local_eff,
            icon_graupel_params.ice_sticking_eff_factor
            * (temperature - icon_graupel_params.tmin_iceautoconv),
        )

        local_nid = rho * qi / cmi
        local_lnlogmi = log(cmi)

        local_qvsidiff = qv - qvsi
        local_svmax = local_qvsidiff / dt

        snow_ice_collision_rate_i2s = (
            local_eff * qi * cagg * exp(icon_graupel_params.ccsaxp * log(cslam))
        )
        graupel_ice_collision_rate_i2g = local_eff * qi * icon_graupel_params.cagg_g * celnrimexp_g
        ice_autoconverson_rate_i2s = (
            local_eff
            * icon_graupel_params.ciau
            * maximum(qi - icon_graupel_params.qi0, wpfloat("0.0"))
        )

        rain_ice_2graupel_ice_loss_rate_i2g = icon_graupel_params.cicri * qi * celn7o8qrk
        if qs > wpfloat("1.0e-7"):
            rain_ice_2graupel_rain_loss_rate_r2g = (
                icon_graupel_params.crcri * (qi / cmi) * celn13o8qrk
            )
        else:
            rain_ice_2graupel_rain_loss_rate_r2g = wpfloat("0.0")

        local_icetotaldeposition = (
            cidep * local_nid * exp(wpfloat("0.33") * local_lnlogmi) * local_qvsidiff
        )
        ice_deposition_rate_v2i = local_icetotaldeposition

        # for sedimenting quantities the maximum
        # allowed depletion is determined by the predictor value.
        local_simax = rhoqi_intermediate / rho / dt

        if local_icetotaldeposition > wpfloat("0.0"):
            local_icetotaldeposition = (
                local_icetotaldeposition * reduce_dep
            )  # FR new: depositional growth reduction
            ice_net_deposition_rate_v2i = minimum(local_icetotaldeposition, local_svmax)
            ice_net_sublimation_rate_v2i = wpfloat("0.0")
        elif local_icetotaldeposition < wpfloat("0.0"):
            ice_net_deposition_rate_v2i = wpfloat("0.0")
            ice_net_sublimation_rate_v2i = maximum(local_icetotaldeposition, local_svmax)
            ice_net_sublimation_rate_v2i = -maximum(ice_net_sublimation_rate_v2i, -local_simax)
        else:
            ice_net_deposition_rate_v2i = wpfloat("0.0")
            ice_net_sublimation_rate_v2i = wpfloat("0.0")

        local_lnlogmi = log(icon_graupel_params.msmin / cmi)
        local_ztau = wpfloat("1.5") * (exp(wpfloat("0.66") * local_lnlogmi) - wpfloat("1.0"))
        ice_dep_autoconversion_rate_i2s = ice_net_deposition_rate_v2i / local_ztau
    else:
        snow_ice_collision_rate_i2s = wpfloat("0.0")
        graupel_ice_collision_rate_i2g = wpfloat("0.0")
        ice_autoconverson_rate_i2s = wpfloat("0.0")
        ice_deposition_rate_v2i = wpfloat("0.0")
        rain_ice_2graupel_ice_loss_rate_i2g = wpfloat("0.0")
        rain_ice_2graupel_rain_loss_rate_r2g = wpfloat("0.0")
        ice_dep_autoconversion_rate_i2s = wpfloat("0.0")
        ice_net_deposition_rate_v2i = wpfloat("0.0")
        ice_net_sublimation_rate_v2i = wpfloat("0.0")

    return (
        snow_ice_collision_rate_i2s,
        graupel_ice_collision_rate_i2g,
        ice_autoconverson_rate_i2s,
        ice_deposition_rate_v2i,
        rain_ice_2graupel_ice_loss_rate_i2g,
        rain_ice_2graupel_rain_loss_rate_r2g,
        ice_dep_autoconversion_rate_i2s,
        ice_net_deposition_rate_v2i,
        ice_net_sublimation_rate_v2i,
    )


@gtx.field_operator
def _snow_and_graupel_depositional_growth_in_cold_ice_clouds(
    temperature: wpfloat,
    pres: wpfloat,
    qv: wpfloat,
    qs: wpfloat,
    qvsi: wpfloat,
    dt: wpfloat,
    ice_net_deposition_rate_v2i: wpfloat,
    cslam: wpfloat,
    cbsdep: wpfloat,
    csdep: wpfloat,
    reduce_dep: wpfloat,
    celn6qgk: wpfloat,
    llqi: bool,
    llqs: bool,
    llqg: bool,
) -> tuple[wpfloat, wpfloat]:
    """
    Compute the vapor deposition of ice crystals and snow in ice clouds when temperature is below zero degree celcius.
    (Please refer to the COSMO microphysics documentation via the link given in the docstring of SingleMomentSixClassIconGraupelConfig for all the equations)

    deposition rate = 1/rho intg_0^inf m_dot f dD (Eq. 5.64)
    m_dot = 4 pi C(D) G F(v,D) d (qv - qvsi),
    G = 1/(1+Hw) and d are functions of environment
    F = 1 + 0.26 sqrt(D v(D)/eta/2) is ventilation factor
    C(D) = C0 D is capacitance (C0 = D/2 for a sphere, D/pi for a circular disk)

    snow resulted from fast ice deposition = ice deposition rate / time_scale, Eq. 5.108

    snow: f = N0 exp(-lamda D), v = v0 D^b
    snow deposition rate = Eq. 5.118 (wrong?) , derived from Eq. 5.71 and 5.72
       = 4 G d (qv - qvsi) N0 (1 + 0.26 sqrt(v0/eta/2) Gamma((5+b)/2)) / lamda^((1+b)/2) 1/lamda^(2)

    graupel deposition = Eq. 5.140

    Args:
        temperature: air temperature [K]
        pres: air pressure [Pa]
        qv: specific humidity [kg/kg]
        qs: specific snow content [kg/kg]
        qvsi: saturated vapor mixing ratio over ice
        dt: time step [s]
        ice_net_deposition_rate_v2i: ice deposition transfer rate
        cslam: constant (refer to equation or documentation in the docstring above)
        cbsdep: constant (refer to equation or documentation in the docstring above)
        csdep: constant (refer to equation or documentation in the docstring above)
        reduce_dep: ice deposition reduced factor
        celn6qgk: constant (refer to equation or documentation in the docstring above)
        llqi: ice grid cell
        llqs: snow grid cell
        llqg: graupel grid cell
    Returns:
        depositional growth rate of snow in cold clouds, depositional growth rate of graupel in cold clouds
    """
    if llqi | llqs | llqg:
        if temperature <= icon_graupel_params.melting_temperature:
            local_qvsidiff = qv - qvsi
            local_svmax = local_qvsidiff / dt

            local_xfac = wpfloat("1.0") + cbsdep * exp(icon_graupel_params.ccsdxp * log(cslam))
            snow_deposition_rate_v2s_in_cold_clouds = (
                csdep * local_xfac * local_qvsidiff / (cslam + icon_graupel_params.eps) ** 2.0
            )
            # FR new: depositional growth reduction
            if snow_deposition_rate_v2s_in_cold_clouds > wpfloat("0.0"):
                snow_deposition_rate_v2s_in_cold_clouds = (
                    snow_deposition_rate_v2s_in_cold_clouds * reduce_dep
                )

            # GZ: This limitation, which was missing in the original graupel scheme,
            # is crucial for numerical stability in the tropics!
            if snow_deposition_rate_v2s_in_cold_clouds > wpfloat("0.0"):
                snow_deposition_rate_v2s_in_cold_clouds = minimum(
                    snow_deposition_rate_v2s_in_cold_clouds,
                    local_svmax - ice_net_deposition_rate_v2i,
                )
            # Suppress depositional growth of snow if the existing amount is too small for a
            # a meaningful distiction between cloud ice and snow
            if qs <= wpfloat("1.0e-7"):
                snow_deposition_rate_v2s_in_cold_clouds = minimum(
                    snow_deposition_rate_v2s_in_cold_clouds, wpfloat("0.0")
                )
            # ** GZ: this numerical fit should be replaced with a physically more meaningful formulation **
            graupel_deposition_rate_v2g_in_cold_clouds = (
                (
                    wpfloat("0.398561")
                    - wpfloat("0.00152398") * temperature
                    + wpfloat("2554.99") / pres
                    + wpfloat("2.6531e-7") * pres
                )
                * local_qvsidiff
                * celn6qgk
            )
        else:
            snow_deposition_rate_v2s_in_cold_clouds = wpfloat("0.0")
            graupel_deposition_rate_v2g_in_cold_clouds = wpfloat("0.0")
    else:
        snow_deposition_rate_v2s_in_cold_clouds = wpfloat("0.0")
        graupel_deposition_rate_v2g_in_cold_clouds = wpfloat("0.0")

    return snow_deposition_rate_v2s_in_cold_clouds, graupel_deposition_rate_v2g_in_cold_clouds


@gtx.field_operator
def _melting(
    temperature: wpfloat,
    pres: wpfloat,
    rho: wpfloat,
    qv: wpfloat,
    qvsw: wpfloat,
    rhoqi_intermediate: wpfloat,
    dt: wpfloat,
    cssmax: wpfloat,
    csgmax: wpfloat,
    celn8qsk: wpfloat,
    celn6qgk: wpfloat,
    llqi: bool,
    llqs: bool,
    llqg: bool,
) -> tuple[wpfloat, wpfloat, wpfloat, wpfloat, wpfloat, wpfloat]:
    """
    Compute the vapor deposition of ice crystals, snow, and graupel in ice clouds when temperature is above zero degree celcius.
    When the air is supersubsaturated over both ice and water, depositional growth of snow and graupel is converted to growth of rain.
    (Please refer to the COSMO microphysics documentation via the link given in the docstring of SingleMomentSixClassIconGraupelConfig for all the equations)

    Ice crystals completely melt when temperature is above zero.
    For snow and graupel, follow Eqs. 5.141 - 5.146

    Args:
        temperature: air temperature [K]
        pres: air pressure [Pa]
        rho: air density [kg/m3]
        qv: specific humidity [kg/kg]
        qvsw: saturated vapor mixing ratio
        rhoqi_intermediate: ice mass with sedimendation flux from above [kg/m3]
        dt: time step [s]
        cssmax: maximum specific snow content
        csgmax: maximum specific graupel content
        celn8qsk: constant (refer to equation or documentation in the docstring above)
        celn6qgk: constant (refer to equation or documentation in the docstring above)
        llqi: ice grid cell
        llqs: snow grid cell
        llqg: graupel grid cell
    Returns:
        melting rate of ice,
        melting rate of snow,
        melting rate of graupel,
        depositional growth rate of snow in melting condition,
        depositional growth rate of graupel in melting condition,
        growth rate of rain in melting condition
    """
    if llqi | llqs | llqg:
        if temperature > icon_graupel_params.melting_temperature:
            # cloud ice melts instantaneously
            ice_melting_rate_i2c = rhoqi_intermediate / rho / dt

            local_qvsw0 = icon_graupel_params.pvsw0 / (
                rho * icon_graupel_params.rv * icon_graupel_params.melting_temperature
            )
            local_qvsw0diff = qv - local_qvsw0

            # ** GZ: several numerical fits in this section should be replaced with physically more meaningful formulations **
            if (
                temperature
                > icon_graupel_params.melting_temperature
                - icon_graupel_params.tcrit * local_qvsw0diff
            ):
                # calculate melting rate
                local_x1 = (
                    temperature
                    - icon_graupel_params.melting_temperature
                    + icon_graupel_params.asmel * local_qvsw0diff
                )
                snow_melting_rate_s2r = (
                    (wpfloat("79.6863") / pres + wpfloat("0.612654e-3")) * local_x1 * celn8qsk
                )
                snow_melting_rate_s2r = minimum(snow_melting_rate_s2r, cssmax)
                graupel_melting_rate_g2r = (
                    (wpfloat("12.31698") / pres + wpfloat("7.39441e-05")) * local_x1 * celn6qgk
                )
                graupel_melting_rate_g2r = minimum(graupel_melting_rate_g2r, csgmax)
                # deposition + melting, ice particle temperature: t0
                # calculation without howell-factor!
                snow_deposition_rate_v2s_in_melting_condition = (
                    (wpfloat("31282.3") / pres + wpfloat("0.241897")) * local_qvsw0diff * celn8qsk
                )
                graupel_deposition_rate_v2g_in_melting_condition = (
                    (wpfloat("0.153907") - pres * wpfloat("7.86703e-07"))
                    * local_qvsw0diff
                    * celn6qgk
                )
                if local_qvsw0diff < wpfloat("0.0"):
                    # melting + evaporation of snow/graupel
                    snow_deposition_rate_v2s_in_melting_condition = maximum(
                        -cssmax, snow_deposition_rate_v2s_in_melting_condition
                    )
                    graupel_deposition_rate_v2g_in_melting_condition = maximum(
                        -csgmax, graupel_deposition_rate_v2g_in_melting_condition
                    )
                    # melt water evaporates
                    snow_melting_rate_s2r = (
                        snow_melting_rate_s2r + snow_deposition_rate_v2s_in_melting_condition
                    )
                    graupel_melting_rate_g2r = (
                        graupel_melting_rate_g2r + graupel_deposition_rate_v2g_in_melting_condition
                    )
                    snow_melting_rate_s2r = maximum(snow_melting_rate_s2r, wpfloat("0.0"))
                    graupel_melting_rate_g2r = maximum(graupel_melting_rate_g2r, wpfloat("0.0"))
                    rain_deposition_rate_v2r = wpfloat("0.0")
                else:
                    # deposition on snow/graupel is interpreted as increase in rain water ( qv --> qr, sconr), therefore,  sconr=(zssdep+zsgdep)
                    rain_deposition_rate_v2r = (
                        snow_deposition_rate_v2s_in_melting_condition
                        + graupel_deposition_rate_v2g_in_melting_condition
                    )
                    snow_deposition_rate_v2s_in_melting_condition = wpfloat("0.0")
                    graupel_deposition_rate_v2g_in_melting_condition = wpfloat("0.0")
            else:
                snow_melting_rate_s2r = wpfloat("0.0")
                graupel_melting_rate_g2r = wpfloat("0.0")
                rain_deposition_rate_v2r = wpfloat("0.0")
                # if t<t_crit, no melting, only evaporation of snow/graupel
                local_qvsidiff = qv - qvsw
                snow_deposition_rate_v2s_in_melting_condition = (
                    (wpfloat("0.28003") - pres * wpfloat("0.146293e-6")) * local_qvsidiff * celn8qsk
                )
                graupel_deposition_rate_v2g_in_melting_condition = (
                    (wpfloat("0.0418521") - pres * wpfloat("4.7524e-8")) * local_qvsidiff * celn6qgk
                )
                snow_deposition_rate_v2s_in_melting_condition = maximum(
                    -cssmax, snow_deposition_rate_v2s_in_melting_condition
                )
                graupel_deposition_rate_v2g_in_melting_condition = maximum(
                    -csgmax, graupel_deposition_rate_v2g_in_melting_condition
                )
        else:
            ice_melting_rate_i2c = wpfloat("0.0")
            snow_melting_rate_s2r = wpfloat("0.0")
            graupel_melting_rate_g2r = wpfloat("0.0")
            rain_deposition_rate_v2r = wpfloat("0.0")
            snow_deposition_rate_v2s_in_melting_condition = wpfloat("0.0")
            graupel_deposition_rate_v2g_in_melting_condition = wpfloat("0.0")
    else:
        ice_melting_rate_i2c = wpfloat("0.0")
        snow_melting_rate_s2r = wpfloat("0.0")
        graupel_melting_rate_g2r = wpfloat("0.0")
        rain_deposition_rate_v2r = wpfloat("0.0")
        snow_deposition_rate_v2s_in_melting_condition = wpfloat("0.0")
        graupel_deposition_rate_v2g_in_melting_condition = wpfloat("0.0")

    return (
        ice_melting_rate_i2c,
        snow_melting_rate_s2r,
        graupel_melting_rate_g2r,
        snow_deposition_rate_v2s_in_melting_condition,
        graupel_deposition_rate_v2g_in_melting_condition,
        rain_deposition_rate_v2r,
    )


@gtx.field_operator
def _evaporation_and_freezing_in_subsaturated_air(
    temperature: wpfloat,
    qv: wpfloat,
    qc: wpfloat,
    qvsw: wpfloat,
    rhoqr: wpfloat,
    dt: wpfloat,
    rain_freezing_rate_r2g_in_clouds: wpfloat,
    csrmax: wpfloat,
    bev: wpfloat,
    bevxp: wpfloat,
    cev: wpfloat,
    cevxp: wpfloat,
    celn7o4qrk: wpfloat,
    llqr: bool,
) -> tuple[wpfloat, wpfloat]:
    """
    Compute the evaporation rate of rain in subsaturated condition.
    (Please refer to the COSMO microphysics documentation via the link given in the docstring of SingleMomentSixClassIconGraupelConfig for all the equations)

    deposition rate = 1/rho intg_0^inf m_dot f dD (Eq. 5.64)
    m_dot = 4 pi C(D) G F(v,D) d (qv - qvsw),
    G = 1/(1+Hw) and d are functions of environment
    F = 1 + 0.26 sqrt(D v(D)/eta/2) is ventilation factor
    C(D) = C0 D is capacitance (C0 = D/2 for a sphere, D/pi for a circular disk)

    snow resulted from fast ice deposition = ice deposition rate / time_scale, Eq. 5.108

    rain: gamma distribution f = N0 D^(mu) exp(-lamda D), m = alpha D^beta, v = v0 D^b
        V = v0 Gamma(b + beta + mu + 1) 1 / (alpha N0 Gamma(beta + mu + 1) )^(b/(beta + mu + 1)) (rho q)^(b/(beta + mu + 1)) rho_factor
        rain evaporation rate = Eq. 5.117 (wrong?) , derived from Eq. 5.71 and 5.72
                              = 2 pi G d (qv - qvsw) N0 (Gamma(2+mu) + 0.26 sqrt(v0/eta/2) Gamma((5+b+mu)/2)) / lamda^((1+b)/2) 1/lamda^(2+mu)
        lamda = (alpha N0 Gamma(beta+mu+1) / rhoq )^(1/(beta+mu+1))

        rain freezing rate =
           Method 1: rain_freezing_option = 1, Eq. 5.168
           Method 2 (ABANDONED in icon4py): rain_freezing_option = 2, Eq. 5.126, an approximation of combined immersion Eq. 5.80 and contact freezing Eq. 5.83

    Args:
        temperature: air temperature [K]
        qv: specific humidity [kg/kg]
        qc: specific cloud content [kg/kg]
        qvsw: saturated vapor mixing ratio
        rhoqr: rain mass [kg/m3]
        dt: time step [s]
        rain_freezing_rate_r2g_in_clouds: rain freezing transfer rate in clouds
        csrmax: maximum specific rain content
        bev: constant (refer to equation or documentation in the docstring above)
        bevxp: constant (refer to equation or documentation in the docstring above)
        cev: constant (refer to equation or documentation in the docstring above)
        cevxp: constant (refer to equation or documentation in the docstring above)
        celn7o4qrk: constant (refer to equation or documentation in the docstring above)
        llqr: rain grid cell
    Returns:
        evaporation rate of rain, freezing rate of rain
    """
    rain_freezing_rate_r2g = rain_freezing_rate_r2g_in_clouds
    if llqr & (qv + qc <= qvsw):
        local_lnqr = log(rhoqr)
        local_x1 = wpfloat("1.0") + bev * exp(bevxp * local_lnqr)
        # Limit evaporation rate in order to avoid overshoots towards supersaturation, the pre-factor approximates (esat(T_wb)-e)/(esat(T)-e) at temperatures between 0 degC and 30 degC
        local_temp_c = temperature - icon_graupel_params.melting_temperature
        local_maxevap = (
            (
                wpfloat("0.61")
                - wpfloat("0.0163") * local_temp_c
                + wpfloat("1.111e-4") * local_temp_c**2.0
            )
            * (qvsw - qv)
            / dt
        )
        rain_evaporation_rate_r2v = cev * local_x1 * (qvsw - qv) * exp(cevxp * local_lnqr)
        rain_evaporation_rate_r2v = minimum(rain_evaporation_rate_r2v, local_maxevap)

        if temperature > icon_graupel_params.homogeneous_freeze_temperature:
            # Calculation of below-cloud rainwater freezing
            if temperature < icon_graupel_params.threshold_freeze_temperature:
                # FR new: reduced rain freezing rate
                rain_freezing_rate_r2g = (
                    icon_graupel_params.coeff_rain_freeze1
                    * (
                        exp(
                            icon_graupel_params.coeff_rain_freeze2
                            * (icon_graupel_params.threshold_freeze_temperature - temperature)
                        )
                        - wpfloat("1.0")
                    )
                    * celn7o4qrk
                )
        else:  # Hom. freezing of rain water
            rain_freezing_rate_r2g = csrmax
    else:
        rain_evaporation_rate_r2v = wpfloat("0.0")

    return rain_evaporation_rate_r2v, rain_freezing_rate_r2g


# TODO (Chia Rui): this is a duplicated function for saturated pressure. Move to a common place when the one in saturation adjustment can be used in scan operator.
@gtx.field_operator
def sat_pres_water(temperature: wpfloat) -> wpfloat:
    return icon_graupel_params.tetens_p0 * exp(
        icon_graupel_params.tetens_aw
        * (temperature - icon_graupel_params.melting_temperature)
        / (temperature - icon_graupel_params.tetens_bw)
    )


@gtx.field_operator
def sat_pres_ice(temperature: wpfloat) -> wpfloat:
    return icon_graupel_params.tetens_p0 * exp(
        icon_graupel_params.tetens_ai
        * (temperature - icon_graupel_params.melting_temperature)
        / (temperature - icon_graupel_params.tetens_bi)
    )


@dataclasses.dataclass
class MetricStateIconGraupel:
    ddqz_z_full: gtx.Field[[CellDim, KDim], wpfloat]


class SingleMomentSixClassIconGraupel:
    def __init__(
        self,
        graupel_config: SingleMomentSixClassIconGraupelConfig,
        saturation_adjust_config: satad.SaturationAdjustmentConfig,
        grid: Optional[icon_grid.IconGrid],
        metric_state: Optional[MetricStateIconGraupel],
        vertical_params: Optional[v_grid.VerticalGrid],
        backend: backend.Backend,
    ):
        self.config = graupel_config
        self._initialize_configurable_parameters()
        self.grid = grid
        self.metric_state = metric_state
        self.vertical_params = vertical_params
        self._backend = backend
        self.saturation_adjustment = satad.SaturationAdjustment(
            config=saturation_adjust_config,
            grid=grid,
            vertical_params=vertical_params,
            metric_state=satad.MetricStateSaturationAdjustment(
                ddqz_z_full=metric_state.ddqz_z_full
            ),
            backend=self._backend,
        )

        self._initialize_local_fields()
        self._determine_horizontal_domains()

    def _initialize_configurable_parameters(self):
        # TODO (Chia Rui): clean up the naming system of these parameters
        ccsrim: wpfloat = (
            0.25
            * math.pi
            * icon_graupel_params.snow_cloud_collection_eff
            * self.config.power_law_coeff_for_snow_fall_speed
            * math.gamma(icon_graupel_params.power_law_exponent_for_snow_fall_speed + 3.0)
        )
        ccsagg: wpfloat = (
            0.25
            * math.pi
            * self.config.power_law_coeff_for_snow_fall_speed
            * math.gamma(icon_graupel_params.power_law_exponent_for_snow_fall_speed + 3.0)
        )
        _ccsvxp = -(
            icon_graupel_params.power_law_exponent_for_snow_fall_speed
            / (icon_graupel_params.power_law_exponent_for_snow_mD_relation + 1.0)
            + 1.0
        )
        ccsvel: wpfloat = (
            icon_graupel_params.power_law_coeff_for_snow_mD_relation
            * self.config.power_law_coeff_for_snow_fall_speed
            * math.gamma(
                icon_graupel_params.power_law_exponent_for_snow_mD_relation
                + icon_graupel_params.power_law_exponent_for_snow_fall_speed
                + 1.0
            )
            * (
                icon_graupel_params.power_law_coeff_for_snow_mD_relation
                * math.gamma(icon_graupel_params.power_law_exponent_for_snow_mD_relation + 1.0)
            )
            ** _ccsvxp
        )
        _n0r: wpfloat = (
            8.0e6 * np.exp(3.2 * self.config.rain_mu) * 0.01 ** (-self.config.rain_mu)
        )  # empirical relation adapted from Ulbrich (1983)
        _n0r: wpfloat = _n0r * self.config.rain_n0  # apply tuning factor to zn0r variable
        _ar: wpfloat = (
            math.pi
            * icon_graupel_params.water_density
            / 6.0
            * _n0r
            * math.gamma(self.config.rain_mu + 4.0)
        )  # pre-factor

        power_law_exponent_for_rain_mean_fall_speed: wpfloat = wpfloat(0.5) / (
            self.config.rain_mu + wpfloat(4.0)
        )
        power_law_coeff_for_rain_mean_fall_speed: wpfloat = (
            wpfloat(130.0)
            * math.gamma(self.config.rain_mu + 4.5)
            / math.gamma(self.config.rain_mu + 4.0)
            * _ar ** (-power_law_exponent_for_rain_mean_fall_speed)
        )

        cevxp: wpfloat = (self.config.rain_mu + wpfloat(2.0)) / (self.config.rain_mu + 4.0)
        cev: wpfloat = (
            wpfloat(2.0)
            * math.pi
            * icon_graupel_params.diffusion_coeff_for_water_vapor
            / icon_graupel_params.howell_factor
            * _n0r
            * _ar ** (-cevxp)
            * math.gamma(self.config.rain_mu + 2.0)
        )
        bevxp: wpfloat = (wpfloat(2.0) * self.config.rain_mu + wpfloat(5.5)) / (
            2.0 * self.config.rain_mu + wpfloat(8.0)
        ) - cevxp
        bev: wpfloat = (
            0.26
            * np.sqrt(
                icon_graupel_params.ref_air_density
                * 130.0
                / icon_graupel_params.air_kinemetic_viscosity
            )
            * _ar ** (-bevxp)
            * math.gamma((2.0 * self.config.rain_mu + 5.5) / 2.0)
            / math.gamma(self.config.rain_mu + 2.0)
        )

        # Precomputations for optimization
        power_law_exponent_for_rain_mean_fall_speed_ln1o2: wpfloat = np.exp(
            power_law_exponent_for_rain_mean_fall_speed * np.log(0.5)
        )
        power_law_exponent_for_ice_mean_fall_speed_ln1o2: wpfloat = np.exp(
            icon_graupel_params.power_law_exponent_for_ice_mean_fall_speed * np.log(0.5)
        )
        power_law_exponent_for_graupel_mean_fall_speed_ln1o2: wpfloat = np.exp(
            icon_graupel_params.power_law_exponent_for_graupel_mean_fall_speed * np.log(0.5)
        )

        self._ccs = (ccsrim, ccsagg, ccsvel)
        self._rain_vel_coef = (
            power_law_exponent_for_rain_mean_fall_speed,
            power_law_coeff_for_rain_mean_fall_speed,
            cevxp,
            cev,
            bevxp,
            bev,
        )
        self._sed_dens_factor_coef = (
            power_law_exponent_for_rain_mean_fall_speed_ln1o2,
            power_law_exponent_for_ice_mean_fall_speed_ln1o2,
            power_law_exponent_for_graupel_mean_fall_speed_ln1o2,
        )

    @property
    def ccs(self) -> tuple[wpfloat, wpfloat, wpfloat]:
        return self._ccs

    @property
    def rain_vel_coef(self) -> tuple[wpfloat, wpfloat, wpfloat, wpfloat, wpfloat, wpfloat]:
        return self._rain_vel_coef

    @property
    def sed_dens_factor_coef(self) -> tuple[wpfloat, wpfloat, wpfloat]:
        return self._sed_dens_factor_coef

    def _initialize_local_fields(self):
        self.qnc = field_alloc.allocate_zero_field(CellDim, KDim, grid=self.grid)
        # TODO (Chia Rui): remove these tendency terms when physics inteface infrastructure is ready
        self.temperature_tendency = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=wpfloat
        )
        self.qv_tendency = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=wpfloat
        )
        self.qc_tendency = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=wpfloat
        )
        self.qi_tendency = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=wpfloat
        )
        self.qr_tendency = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=wpfloat
        )
        self.qs_tendency = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=wpfloat
        )
        self.qg_tendency = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=wpfloat
        )
        self.rhoqrv_old_kup = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=wpfloat
        )
        self.rhoqsv_old_kup = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=wpfloat
        )
        self.rhoqgv_old_kup = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=wpfloat
        )
        self.rhoqiv_old_kup = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=wpfloat
        )
        self.vnew_r = field_alloc.allocate_zero_field(CellDim, KDim, grid=self.grid, dtype=wpfloat)
        self.vnew_s = field_alloc.allocate_zero_field(CellDim, KDim, grid=self.grid, dtype=wpfloat)
        self.vnew_g = field_alloc.allocate_zero_field(CellDim, KDim, grid=self.grid, dtype=wpfloat)
        self.vnew_i = field_alloc.allocate_zero_field(CellDim, KDim, grid=self.grid, dtype=wpfloat)
        self.rain_precipitation_flux = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=wpfloat
        )
        self.snow_precipitation_flux = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=wpfloat
        )
        self.graupel_precipitation_flux = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=wpfloat
        )
        self.ice_precipitation_flux = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=wpfloat
        )
        self.total_precipitation_flux = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=wpfloat
        )

    def _determine_horizontal_domains(self):
        cell_domain = h_grid.domain(CellDim)
        self._start_cell_nudging = self.grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self.grid.start_index(cell_domain(h_grid.Zone.END))

    def run(
        self,
        dtime: wpfloat,
        prognostic_state: prognostics.PrognosticState,
        diagnostic_state: diagnostics.DiagnosticState,
        tracer_state: tracers.TracerState,
    ):
        icon_graupel(
            self.vertical_params.kstart_moist,
            self.config.liquid_autoconversion_option.value,
            self.config.snow_intercept_option.value,
            self.config.is_isochoric,
            self.config.use_constant_water_heat_capacity,
            self.config.ice_stickeff_min,
            self.config.power_law_coeff_for_ice_mean_fall_speed,
            self.config.exponent_for_density_factor_in_ice_sedimentation,
            self.config.power_law_coeff_for_snow_fall_speed,
            *self._ccs,
            *self._rain_vel_coef,
            *self._sed_dens_factor_coef,
            dtime,
            self.metric_state.ddqz_z_full,
            diagnostic_state.temperature,
            diagnostic_state.pressure,
            prognostic_state.rho,
            tracer_state.qv,
            tracer_state.qc,
            tracer_state.qi,
            tracer_state.qr,
            tracer_state.qs,
            tracer_state.qg,
            self.qnc,
            self.temperature_tendency,
            self.qv_tendency,
            self.qc_tendency,
            self.qi_tendency,
            self.qr_tendency,
            self.qs_tendency,
            self.qg_tendency,
            self.rhoqrv_old_kup,
            self.rhoqsv_old_kup,
            self.rhoqgv_old_kup,
            self.rhoqiv_old_kup,
            self.vnew_r,
            self.vnew_s,
            self.vnew_g,
            self.vnew_i,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        icon_graupel_flux_above_ground(
            self.config.do_latent_heat_nudging,
            dtime,
            prognostic_state.rho,
            tracer_state.qr,
            tracer_state.qs,
            tracer_state.qg,
            tracer_state.qi,
            self.qr_tendency,
            self.qs_tendency,
            self.qg_tendency,
            self.qi_tendency,
            self.rhoqrv_old_kup,
            self.rhoqsv_old_kup,
            self.rhoqgv_old_kup,
            self.rhoqiv_old_kup,
            self.vnew_r,
            self.vnew_s,
            self.vnew_g,
            self.vnew_i,
            self.rain_precipitation_flux,
            self.snow_precipitation_flux,
            self.graupel_precipitation_flux,
            self.ice_precipitation_flux,
            self.total_precipitation_flux,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels - gtx.int32(1),
            offset_provider={},
        )

        icon_graupel_flux_ground(
            self.config.do_latent_heat_nudging,
            dtime,
            prognostic_state.rho,
            tracer_state.qr,
            tracer_state.qs,
            tracer_state.qg,
            tracer_state.qi,
            self.qr_tendency,
            self.qs_tendency,
            self.qg_tendency,
            self.qi_tendency,
            self.rhoqrv_old_kup,
            self.rhoqsv_old_kup,
            self.rhoqgv_old_kup,
            self.rhoqiv_old_kup,
            self.vnew_r,
            self.vnew_s,
            self.vnew_g,
            self.vnew_i,
            self.rain_precipitation_flux,
            self.snow_precipitation_flux,
            self.graupel_precipitation_flux,
            self.ice_precipitation_flux,
            self.total_precipitation_flux,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=self.grid.num_levels - gtx.int32(1),
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        # TODO (Chia Rui): the entire call of microphysics needs to be tested when slow performance of scan operator is resolved. Until then, the following call of saturation adjusmnt is wrong.
        # TODO (Chia Rui): Change the way saturation adjustment is called when the interface to the saturation adjustment which follows the protocol is completed.
        if self.config.do_saturation_adjustment:
            # self.saturation_adjustment.run( # [commented-out-code]
            #     dtime=dtime, # noqa: ERA001 [commented-out-code]
            #     prognostic_state=prognostic_state, # noqa: ERA001 [commented-out-code]
            #     diagnostic_state=diagnostic_state, # noqa: ERA001 [commented-out-code]
            #     tracer_state=tracer_state, # noqa: ERA001 [commented-out-code]
            # ) # [commented-out-code]
            raise NotImplementedError


@gtx.field_operator
def _icon_graupel_flux_ground(
    do_latent_heat_nudging: bool,
    dtime: wpfloat,
    rho: gtx.Field[[CellDim, KDim], wpfloat],
    qr: gtx.Field[[CellDim, KDim], wpfloat],
    qs: gtx.Field[[CellDim, KDim], wpfloat],
    qg: gtx.Field[[CellDim, KDim], wpfloat],
    qi: gtx.Field[[CellDim, KDim], wpfloat],
    qr_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    qs_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    qg_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    qi_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqrv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqsv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqgv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqiv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_r: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_s: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_g: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_i: gtx.Field[[CellDim, KDim], wpfloat],
) -> tuple[
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
]:
    rain_flux = wpfloat("0.5") * ((qr + qr_tendency * dtime) * rho * vnew_r + rhoqrv_old_kup)
    snow_flux = wpfloat("0.5") * ((qs + qs_tendency * dtime) * rho * vnew_s + rhoqsv_old_kup)
    graupel_flux = wpfloat("0.5") * ((qg + qg_tendency * dtime) * rho * vnew_g + rhoqgv_old_kup)
    ice_flux = wpfloat("0.5") * ((qi + qi_tendency * dtime) * rho * vnew_i + rhoqiv_old_kup)
    zero = broadcast(wpfloat("0.0"), (CellDim, KDim))
    # for the latent heat nudging
    total_flux = rain_flux + snow_flux + graupel_flux if do_latent_heat_nudging else zero
    return rain_flux, snow_flux, graupel_flux, ice_flux, total_flux


@gtx.field_operator
def _icon_graupel_flux_above_ground(
    do_latent_heat_nudging: bool,
    dtime: wpfloat,
    rho: gtx.Field[[CellDim, KDim], wpfloat],
    qr: gtx.Field[[CellDim, KDim], wpfloat],
    qs: gtx.Field[[CellDim, KDim], wpfloat],
    qg: gtx.Field[[CellDim, KDim], wpfloat],
    qi: gtx.Field[[CellDim, KDim], wpfloat],
    qr_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    qs_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    qg_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    qi_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqrv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqsv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqgv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqiv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_r: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_s: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_g: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_i: gtx.Field[[CellDim, KDim], wpfloat],
) -> tuple[
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
]:
    zero = broadcast(wpfloat("0.0"), (CellDim, KDim))

    rain_flux_ = (qr + qr_tendency * dtime) * rho * vnew_r
    snow_flux_ = (qs + qs_tendency * dtime) * rho * vnew_s
    graupel_flux_ = (qg + qg_tendency * dtime) * rho * vnew_g
    ice_flux_ = (qi + qi_tendency * dtime) * rho * vnew_i

    rain_flux_new = where(rain_flux_ <= icon_graupel_params.qmin, zero, rain_flux_)
    snow_flux_new = where(snow_flux_ <= icon_graupel_params.qmin, zero, snow_flux_)
    graupel_flux_new = where(graupel_flux_ <= icon_graupel_params.qmin, zero, graupel_flux_)
    ice_flux_new = where(ice_flux_ <= icon_graupel_params.qmin, zero, ice_flux_)

    rain_flux = wpfloat("0.5") * (rain_flux_new + rhoqrv_old_kup)
    snow_flux = wpfloat("0.5") * (snow_flux_new + rhoqsv_old_kup)
    graupel_flux = wpfloat("0.5") * (graupel_flux_new + rhoqgv_old_kup)
    ice_flux = wpfloat("0.5") * (ice_flux_new + rhoqiv_old_kup)
    total_flux = rain_flux + snow_flux + graupel_flux + ice_flux if do_latent_heat_nudging else zero

    return rain_flux, snow_flux, graupel_flux, ice_flux, total_flux


@program(grid_type=gtx.GridType.UNSTRUCTURED)
def icon_graupel_flux_ground(
    do_latent_heat_nudging: bool,
    dtime: wpfloat,
    rho: gtx.Field[[CellDim, KDim], wpfloat],
    qr: gtx.Field[[CellDim, KDim], wpfloat],
    qs: gtx.Field[[CellDim, KDim], wpfloat],
    qg: gtx.Field[[CellDim, KDim], wpfloat],
    qi: gtx.Field[[CellDim, KDim], wpfloat],
    qr_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    qs_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    qg_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    qi_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqrv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqsv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqgv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqiv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_r: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_s: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_g: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_i: gtx.Field[[CellDim, KDim], wpfloat],
    rain_precipitation_flux: gtx.Field[[CellDim, KDim], wpfloat],
    snow_precipitation_flux: gtx.Field[[CellDim, KDim], wpfloat],
    graupel_precipitation_flux: gtx.Field[[CellDim, KDim], wpfloat],
    ice_precipitation_flux: gtx.Field[[CellDim, KDim], wpfloat],
    total_precipitation_flux: gtx.Field[[CellDim, KDim], wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _icon_graupel_flux_ground(
        do_latent_heat_nudging,
        dtime,
        rho,
        qr,
        qs,
        qg,
        qi,
        qr_tendency,
        qs_tendency,
        qg_tendency,
        qi_tendency,
        rhoqrv_old_kup,
        rhoqsv_old_kup,
        rhoqgv_old_kup,
        rhoqiv_old_kup,
        vnew_r,
        vnew_s,
        vnew_g,
        vnew_i,
        out=(
            rain_precipitation_flux,
            snow_precipitation_flux,
            graupel_precipitation_flux,
            ice_precipitation_flux,
            total_precipitation_flux,
        ),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@program(grid_type=gtx.GridType.UNSTRUCTURED)
def icon_graupel_flux_above_ground(
    do_latent_heat_nudging: bool,
    dtime: wpfloat,
    rho: gtx.Field[[CellDim, KDim], wpfloat],
    qr: gtx.Field[[CellDim, KDim], wpfloat],
    qs: gtx.Field[[CellDim, KDim], wpfloat],
    qg: gtx.Field[[CellDim, KDim], wpfloat],
    qi: gtx.Field[[CellDim, KDim], wpfloat],
    qr_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    qs_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    qg_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    qi_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqrv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqsv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqgv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqiv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_r: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_s: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_g: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_i: gtx.Field[[CellDim, KDim], wpfloat],
    rain_precipitation_flux: gtx.Field[[CellDim, KDim], wpfloat],
    snow_precipitation_flux: gtx.Field[[CellDim, KDim], wpfloat],
    graupel_precipitation_flux: gtx.Field[[CellDim, KDim], wpfloat],
    ice_precipitation_flux: gtx.Field[[CellDim, KDim], wpfloat],
    total_precipitation_flux: gtx.Field[[CellDim, KDim], wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _icon_graupel_flux_above_ground(
        do_latent_heat_nudging,
        dtime,
        rho,
        qr,
        qs,
        qg,
        qi,
        qr_tendency,
        qs_tendency,
        qg_tendency,
        qi_tendency,
        rhoqrv_old_kup,
        rhoqsv_old_kup,
        rhoqgv_old_kup,
        rhoqiv_old_kup,
        vnew_r,
        vnew_s,
        vnew_g,
        vnew_i,
        out=(
            rain_precipitation_flux,
            snow_precipitation_flux,
            graupel_precipitation_flux,
            ice_precipitation_flux,
            total_precipitation_flux,
        ),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@scan_operator(
    axis=KDim,
    forward=True,
    init=(
        wpfloat("0.0"),  # temperature tendency
        wpfloat("0.0"),  # qv tendency
        wpfloat("0.0"),  # qc tendency
        wpfloat("0.0"),  # qi tendency
        wpfloat("0.0"),  # qr tendency
        wpfloat("0.0"),  # qs tendency
        wpfloat("0.0"),  # qg tendency
        wpfloat("0.0"),  # qv
        wpfloat("0.0"),  # qc
        wpfloat("0.0"),  # qi
        wpfloat("0.0"),  # qr
        wpfloat("0.0"),  # qs
        wpfloat("0.0"),  # qg
        wpfloat("0.0"),  # rhoqrv
        wpfloat("0.0"),  # rhoqsv
        wpfloat("0.0"),  # rhoqgv
        wpfloat("0.0"),  # rhoqiv
        wpfloat("0.0"),  # newv_r
        wpfloat("0.0"),  # newv_s
        wpfloat("0.0"),  # newv_g
        wpfloat("0.0"),  # newv_i
        wpfloat("0.0"),  # cloud top distance
        wpfloat("0.0"),  # density
        wpfloat("0.0"),  # density factor
        wpfloat("0.0"),  # density factor for ice
        wpfloat("0.0"),  # snow intercept parameter
        wpfloat("0.0"),  # saturation pressure
        gtx.int32(0),  # k level
    ),
)
def _icon_graupel_scan(
    state_kup: tuple[
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        wpfloat,
        gtx.int32,
    ],
    startmoist_level: gtx.int32,  # k start moist level
    surface_level: gtx.int32,  # k bottom level
    liquid_autoconversion_option: gtx.int32,
    snow_intercept_option: gtx.int32,
    is_isochoric: bool,
    use_constant_water_heat_capacity: bool,
    ice_stickeff_min: wpfloat,
    power_law_coeff_for_ice_mean_fall_speed: wpfloat,
    exponent_for_density_factor_in_ice_sedimentation: wpfloat,
    power_law_coeff_for_snow_fall_speed: wpfloat,
    ccsrim: wpfloat,
    ccsagg: wpfloat,
    ccsvel: wpfloat,
    power_law_exponent_for_rain_mean_fall_speed: wpfloat,
    power_law_coeff_for_rain_mean_fall_speed: wpfloat,
    cevxp: wpfloat,
    cev: wpfloat,
    bevxp: wpfloat,
    bev: wpfloat,
    power_law_exponent_for_rain_mean_fall_speed_ln1o2: wpfloat,
    power_law_exponent_for_ice_mean_fall_speed_ln1o2: wpfloat,
    power_law_exponent_for_graupel_mean_fall_speed_ln1o2: wpfloat,
    dt: wpfloat,
    dz: wpfloat,
    temperature: wpfloat,
    pres: wpfloat,
    rho: wpfloat,
    qv: wpfloat,
    qc: wpfloat,
    qi: wpfloat,
    qr: wpfloat,
    qs: wpfloat,
    qg: wpfloat,
    qnc: wpfloat,
):
    """
    This is the ICON graupel scheme. The structure of the code can be split into several steps as follow:
        1. return the original prognostic, diagnostic, and tracer variables when k is smaller than kstart_moist.
        2. initialize tracer at k-1 level, and some pre-computed coefficients including the snow intercept parameter for later uses.
        3. compute sedimentation fluxes and update rain, snow, and graupel mass at the current k level.
        4. compute pre-computed coefficients after update from sedimentation fluxes to include implicitness of the graupel scheme.
        5. compute all transfer rates.
        6. check if tracers go below 0.
        7. update all tendencies.

    Below is a list of the so-called transfered rates for all microphyiscal processes in ICON graupel scheme:
        ice_net_deposition_rate_v2i         :   vapor   -> ice,     ice vapor deposition
        ice_net_sublimation_rate_v2i        :   vapor   -> ice,     ice vapor sublimation
        ice_deposition_rate_v2i             :   vapor   -> ice,     ice vapor net deposition
        snow_deposition_rate_v2s            :   vapor   -> snow,    snow vapor deposition
        graupel_deposition_rate_v2g         :   vapor   -> graupel, graupel vapor deposition
        ice_nucleation_rate_v2i             :   vapor   -> ice,     ice nucleation
        rain_deposition_rate_v2r            :   vapor   -> rain,    rain condensation on melting snow/graupel
        cloud_autoconversion_rate_c2r       :   cloud   -> rain,    cloud autoconversion into rain
        cloud_freezing_rate_c2i             :   cloud   -> ice,     cloud freezing
        rain_cloud_collision_rate_c2r       :   cloud   -> rain,    rain-cloud accretion
        rain_shedding_rate_c2r              :   cloud   -> rain,    rain shedding from riming above freezing
        snow_riming_rate_c2s                :   cloud   -> snow,    snow riming
        graupel_riming_rate_c2g             :   cloud   -> graupel, graupel riming
        ice_melting_rate_i2c                :   ice     -> cloud,   ice melting
        rain_ice_2graupel_ice_loss_rate_i2g :   ice     -> graupel, ice loss in rain-ice accretion
        ice_dep_autoconversion_rate_i2s     :   ice     -> snow,    ice vapor depositional autoconversion into snow
        snow_ice_collision_rate_i2s         :   ice     -> snow,    snow-ice aggregation
        graupel_ice_collision_rate_i2g      :   ice     -> graupel, graupel-ice aggregation
        ice_autoconverson_rate_i2s          :   ice     -> snow,    ice autoconversion into snow
        rain_ice_2graupel_rain_loss_rate_r2g:   rain    -> graupel, rain loss in rain-ice accretion
        rain_freezing_rate_r2g              :   rain    -> graupel, rain freezing
        rain_evaporation_rate_r2v           :   rain    -> vapor,   rain evaporation
        snow_melting_rate_s2r               :   snow    -> rain,    snow melting
        snow_autoconversion_rate_s2g        :   snow    -> graupel, snow autoconversion into graupel
        graupel_melting_rate_g2r            :   graupel -> rain,    graupel melting
    """

    (
        temperature_tendency_kup,
        qv_tendency_kup,
        qc_tendency_kup,
        qi_tendency_kup,
        qr_tendency_kup,
        qs_tendency_kup,
        qg_tendency_kup,
        qv_old_kup,
        qc_old_kup,
        qi_old_kup,
        qr_old_kup,
        qs_old_kup,
        qg_old_kup,
        rhoqrv_old_kup,
        rhoqsv_old_kup,
        rhoqgv_old_kup,
        rhoqiv_old_kup,
        vnew_r,
        vnew_s,
        vnew_g,
        vnew_i,
        dist_cldtop_kup,
        rho_kup,
        crho1o2_kup,
        crhofac_qi_kup,
        snow_sed0_kup,
        qvsw_kup,
        k_lev,
    ) = state_kup

    # ------------------------------------------------------------------------------
    #  Section 1: Skip microphysics when k < kstart_moist
    # ------------------------------------------------------------------------------

    # TODO (Chia Rui): This scan operator has to be run from top to bottom because bounds are ignored on embedded backend. Consider to remove this when graupel scan can be run on gtfn backend.
    if k_lev < startmoist_level:
        k_lev = k_lev + 1
        return (
            wpfloat("0.0"),
            wpfloat("0.0"),
            wpfloat("0.0"),
            wpfloat("0.0"),
            wpfloat("0.0"),
            wpfloat("0.0"),
            wpfloat("0.0"),
            qv,
            qc,
            qi,
            qr,
            qs,
            qg,
            wpfloat("0.0"),
            wpfloat("0.0"),
            wpfloat("0.0"),
            wpfloat("0.0"),
            wpfloat("0.0"),
            wpfloat("0.0"),
            wpfloat("0.0"),
            wpfloat("0.0"),
            wpfloat("0.0"),
            rho,
            wpfloat("0.0"),
            wpfloat("0.0"),
            wpfloat("0.0"),
            wpfloat("0.0"),
            k_lev,
        )

    # ------------------------------------------------------------------------------
    #  Section 2: Precomputed coefficients
    # ------------------------------------------------------------------------------

    qv_kup = qv_old_kup + qv_tendency_kup * dt
    qc_kup = qc_old_kup + qc_tendency_kup * dt
    qi_kup = qi_old_kup + qi_tendency_kup * dt
    qr_kup = qr_old_kup + qr_tendency_kup * dt
    qs_kup = qs_old_kup + qs_tendency_kup * dt
    qg_kup = qg_old_kup + qg_tendency_kup * dt

    is_surface = True if k_lev == surface_level else False

    # Define reciprocal of heat capacity of dry air (at constant pressure vs at constant volume)
    heat_cap_r = icon_graupel_params.rcvd if is_isochoric else icon_graupel_params.rcpd

    # TODO (Chia Rui): duplicated function for computing latent heat. Saturation adjustment also uses the same function. Move to a common place.
    lhv = (
        icon_graupel_params.latent_heat_for_vaporisation
        if use_constant_water_heat_capacity
        else icon_graupel_params.latent_heat_for_vaporisation
        + (icon_graupel_params.cp_v - icon_graupel_params.specific_heat_capacity_for_water)
        * (temperature - icon_graupel_params.melting_temperature)
        - icon_graupel_params.rv * temperature
    )
    lhs = (
        icon_graupel_params.latent_heat_for_sublimation
        if use_constant_water_heat_capacity
        else icon_graupel_params.latent_heat_for_sublimation
        + (icon_graupel_params.cp_v - icon_graupel_params.specific_heat_capacity_for_ice)
        * (temperature - icon_graupel_params.melting_temperature)
        - icon_graupel_params.rv * temperature
    )

    # for density correction of fall speeds
    chlp = log(icon_graupel_params.ref_air_density / rho)
    crho1o2 = exp(chlp / wpfloat("2.0"))
    crhofac_qi = exp(chlp * exponent_for_density_factor_in_ice_sedimentation)

    cdtdh = wpfloat("0.5") * dt / dz
    cscmax = qc / dt
    cnin = _compute_cooper_inp_concentration(temperature)
    cmi = minimum(rho * qi / cnin, icon_graupel_params.ice_max_mass)
    cmi = maximum(icon_graupel_params.ice_initial_mass, cmi)

    qvsw = sat_pres_water(temperature) / (rho * icon_graupel_params.rv * temperature)
    qvsi = sat_pres_ice(temperature) / (rho * icon_graupel_params.rv * temperature)

    rhoqr = qr * rho
    rhoqs = qs * rho
    rhoqg = qg * rho
    rhoqi = qi * rho

    rhoqrv_new_kup = qr_kup * rho_kup * vnew_r
    rhoqsv_new_kup = qs_kup * rho_kup * vnew_s
    rhoqgv_new_kup = qg_kup * rho_kup * vnew_g
    rhoqiv_new_kup = qi_kup * rho_kup * vnew_i

    if rhoqrv_new_kup <= icon_graupel_params.qmin:
        rhoqrv_new_kup = wpfloat("0.0")
    if rhoqsv_new_kup <= icon_graupel_params.qmin:
        rhoqsv_new_kup = wpfloat("0.0")
    if rhoqgv_new_kup <= icon_graupel_params.qmin:
        rhoqgv_new_kup = wpfloat("0.0")
    if rhoqiv_new_kup <= icon_graupel_params.qmin:
        rhoqiv_new_kup = wpfloat("0.0")

    rhoqr_intermediate = rhoqr / cdtdh + rhoqrv_new_kup + rhoqrv_old_kup
    rhoqs_intermediate = rhoqs / cdtdh + rhoqsv_new_kup + rhoqsv_old_kup
    rhoqg_intermediate = rhoqg / cdtdh + rhoqgv_new_kup + rhoqgv_old_kup
    rhoqi_intermediate = rhoqi / cdtdh + rhoqiv_new_kup + rhoqiv_old_kup

    llqr = True if (rhoqr > icon_graupel_params.qmin) else False
    llqs = True if (rhoqs > icon_graupel_params.qmin) else False
    llqg = True if (rhoqg > icon_graupel_params.qmin) else False
    llqi = True if (rhoqi > icon_graupel_params.qmin) else False

    n0s, snow_sed0, crim, cagg, cbsdep = _compute_snow_interception_and_collision_parameters(
        temperature,
        rho,
        qs,
        ccsvel,
        ccsrim,
        ccsagg,
        power_law_coeff_for_snow_fall_speed,
        llqs,
        snow_intercept_option,
    )

    # ------------------------------------------------------------------------------
    #  Section 3: Sedimentation fluxes
    # ------------------------------------------------------------------------------

    if k_lev > startmoist_level:
        vnew_s = (
            snow_sed0_kup
            * exp(icon_graupel_params.ccswxp * log((qs_kup + qs) * wpfloat("0.5") * rho_kup))
            * crho1o2_kup
            if qs_kup + qs > icon_graupel_params.qmin
            else wpfloat("0.0")
        )
        vnew_r = (
            power_law_coeff_for_rain_mean_fall_speed
            * exp(
                power_law_exponent_for_rain_mean_fall_speed
                * log((qr_kup + qr) * wpfloat("0.5") * rho_kup)
            )
            * crho1o2_kup
            if qr_kup + qr > icon_graupel_params.qmin
            else wpfloat("0.0")
        )
        vnew_g = (
            icon_graupel_params.power_law_coeff_for_graupel_mean_fall_speed
            * exp(
                icon_graupel_params.power_law_exponent_for_graupel_mean_fall_speed
                * log((qg_kup + qg) * wpfloat("0.5") * rho_kup)
            )
            * crho1o2_kup
            if qg_kup + qg > icon_graupel_params.qmin
            else wpfloat("0.0")
        )
        vnew_i = (
            power_law_coeff_for_ice_mean_fall_speed
            * exp(
                icon_graupel_params.power_law_exponent_for_ice_mean_fall_speed
                * log((qi_kup + qi) * wpfloat("0.5") * rho_kup)
            )
            * crhofac_qi_kup
            if qi_kup + qi > icon_graupel_params.qmin
            else wpfloat("0.0")
        )

    if llqs:
        terminal_velocity = snow_sed0 * exp(icon_graupel_params.ccswxp * log(rhoqs)) * crho1o2
        # Prevent terminal fall speed of snow from being zero at the surface level
        if is_surface:
            terminal_velocity = maximum(
                terminal_velocity, icon_graupel_params.minimum_snow_fall_speed
            )

        rhoqsv = rhoqs * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if vnew_s == wpfloat("0.0"):
            vnew_s = terminal_velocity * icon_graupel_params.ccswxp_ln1o2

    else:
        rhoqsv = wpfloat("0.0")

    if llqr:
        terminal_velocity = (
            power_law_coeff_for_rain_mean_fall_speed
            * exp(power_law_exponent_for_rain_mean_fall_speed * log(rhoqr))
            * crho1o2
        )
        # Prevent terminal fall speed of snow from being zero at the surface level
        if is_surface:
            terminal_velocity = maximum(
                terminal_velocity, icon_graupel_params.minimum_rain_fall_speed
            )

        rhoqrv = rhoqr * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if vnew_r == wpfloat("0.0"):
            vnew_r = terminal_velocity * power_law_exponent_for_rain_mean_fall_speed_ln1o2

    else:
        rhoqrv = wpfloat("0.0")

    if llqg:
        terminal_velocity = (
            icon_graupel_params.power_law_coeff_for_graupel_mean_fall_speed
            * exp(icon_graupel_params.power_law_exponent_for_graupel_mean_fall_speed * log(rhoqg))
            * crho1o2
        )
        # Prevent terminal fall speed of snow from being zero at the surface level
        if is_surface:
            terminal_velocity = maximum(
                terminal_velocity, icon_graupel_params.minimum_graupel_fall_speed
            )

        rhoqgv = rhoqg * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if vnew_g == wpfloat("0.0"):
            vnew_g = terminal_velocity * power_law_exponent_for_graupel_mean_fall_speed_ln1o2

    else:
        rhoqgv = wpfloat("0.0")

    if llqi:
        terminal_velocity = (
            power_law_coeff_for_ice_mean_fall_speed
            * exp(icon_graupel_params.power_law_exponent_for_ice_mean_fall_speed * log(rhoqi))
            * crhofac_qi
        )

        rhoqiv = rhoqi * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if vnew_i == wpfloat("0.0"):
            vnew_i = terminal_velocity * power_law_exponent_for_ice_mean_fall_speed_ln1o2

    else:
        rhoqiv = wpfloat("0.0")

    # Prevent terminal fall speeds of precip hydrometeors from being zero at the surface level
    if is_surface:
        vnew_s = maximum(vnew_s, icon_graupel_params.minimum_snow_fall_speed)
        vnew_r = maximum(vnew_r, icon_graupel_params.minimum_rain_fall_speed)
        vnew_g = maximum(vnew_g, icon_graupel_params.minimum_graupel_fall_speed)

    # derive the intermediate density of hydrometeors, Eq. 5.21:
    # limit the precipitation flux at this k level such that mixing ratio won't go below zero
    rhoqrv = minimum(rhoqrv, rhoqr_intermediate)
    rhoqsv = minimum(rhoqsv, rhoqs_intermediate)
    rhoqgv = minimum(rhoqgv, maximum(wpfloat("0.0"), rhoqg_intermediate))
    rhoqiv = minimum(rhoqiv, rhoqi_intermediate)

    rhoqr_intermediate = cdtdh * (rhoqr_intermediate - rhoqrv)
    rhoqs_intermediate = cdtdh * (rhoqs_intermediate - rhoqsv)
    rhoqg_intermediate = cdtdh * (rhoqg_intermediate - rhoqgv)
    rhoqi_intermediate = cdtdh * (rhoqi_intermediate - rhoqiv)

    cimr = wpfloat("1.0") / (wpfloat("1.0") + vnew_r * cdtdh)
    cims = wpfloat("1.0") / (wpfloat("1.0") + vnew_s * cdtdh)
    cimg = wpfloat("1.0") / (wpfloat("1.0") + vnew_g * cdtdh)
    cimi = wpfloat("1.0") / (wpfloat("1.0") + vnew_i * cdtdh)

    # intermediate values
    rhoqr = rhoqr_intermediate * cimr
    rhoqs = rhoqs_intermediate * cims
    rhoqg = rhoqg_intermediate * cimg

    # ------------------------------------------------------------------------------
    #  Section 4: Precomputed coefficients after sedimentation for implicitness
    # ------------------------------------------------------------------------------

    llqr = True if (rhoqr > icon_graupel_params.qmin) else False
    llqs = True if (rhoqs > icon_graupel_params.qmin) else False
    llqg = True if (rhoqg > icon_graupel_params.qmin) else False
    llqi = True if (qi > icon_graupel_params.qmin) else False
    llqc = True if (qc > icon_graupel_params.qmin) else False

    if llqr:
        clnrhoqr = log(rhoqr)
        csrmax = (
            rhoqr_intermediate / rho / dt
        )  # GZ: shifting this computation ahead of the IF condition changes results!
        if qi + qc > icon_graupel_params.qmin:
            celn7o8qrk = exp(wpfloat("7.0") / wpfloat("8.0") * clnrhoqr)
        else:
            celn7o8qrk = wpfloat("0.0")
        if temperature < icon_graupel_params.threshold_freeze_temperature:
            celn7o4qrk = exp(wpfloat("7.0") / wpfloat("4.0") * clnrhoqr)  # FR new
        else:
            celn7o4qrk = wpfloat("0.0")
        if llqi:
            celn13o8qrk = exp(wpfloat("13.0") / wpfloat("8.0") * clnrhoqr)
        else:
            celn13o8qrk = wpfloat("0.0")
    else:
        csrmax = wpfloat("0.0")
        celn7o8qrk = wpfloat("0.0")
        celn7o4qrk = wpfloat("0.0")
        celn13o8qrk = wpfloat("0.0")

    # ** GZ: the following computation differs substantially from the corresponding code in cloudice **
    if llqs:
        clnrhoqs = log(rhoqs)
        cssmax = (
            rhoqs_intermediate / rho / dt
        )  # GZ: shifting this computation ahead of the IF condition changes results#
        if qi + qc > icon_graupel_params.qmin:
            celn3o4qsk = exp(wpfloat("3.0") / wpfloat("4.0") * clnrhoqs)
        else:
            celn3o4qsk = wpfloat("0.0")
        celn8qsk = exp(wpfloat("0.8") * clnrhoqs)
    else:
        cssmax = wpfloat("0.0")
        celn3o4qsk = wpfloat("0.0")
        celn8qsk = wpfloat("0.0")

    if llqg:
        clnrhoqg = log(rhoqg)
        csgmax = rhoqg_intermediate / rho / dt
        if qi + qc > icon_graupel_params.qmin:
            celnrimexp_g = exp(icon_graupel_params.graupel_rimexp * clnrhoqg)
        else:
            celnrimexp_g = wpfloat("0.0")
        celn6qgk = exp(wpfloat("0.6") * clnrhoqg)
    else:
        csgmax = wpfloat("0.0")
        celnrimexp_g = wpfloat("0.0")
        celn6qgk = wpfloat("0.0")

    if llqi | llqs:
        cdvtp = icon_graupel_params.ccdvtp * exp(wpfloat("1.94") * log(temperature)) / pres
        chi = icon_graupel_params.ccshi1 * cdvtp * rho * qvsi / (temperature * temperature)
        chlp = cdvtp / (wpfloat("1.0") + chi)
        cidep = icon_graupel_params.ccidep * chlp

        if llqs:
            cslam = exp(icon_graupel_params.ccslxp * log(icon_graupel_params.ccslam * n0s / rhoqs))
            cslam = minimum(cslam, wpfloat("1.0e15"))
            csdep = wpfloat("4.0") * n0s * chlp
        else:
            cslam = wpfloat("1.0e10")
            csdep = wpfloat("3.367e-2")
    else:
        cidep = wpfloat("1.3e-5")
        cslam = wpfloat("1.0e10")
        csdep = wpfloat("3.367e-2")

    # ------------------------------------------------------------------------------
    #  Section 5: Transfer rates
    # ------------------------------------------------------------------------------

    ice_nucleation_rate_v2i = _deposition_nucleation_at_low_temperature_or_in_clouds(
        temperature, rho, qv, qi, qvsi, cnin, dt, llqc
    )

    (
        cloud_autoconversion_rate_c2r,
        rain_cloud_collision_rate_c2r,
    ) = _autoconversion_and_rain_accretion(
        temperature, qc, qr, qnc, celn7o8qrk, llqc, liquid_autoconversion_option
    )

    cloud_freezing_rate_c2i, rain_freezing_rate_r2g_in_clouds = _freezing_in_clouds(
        temperature, qc, qr, cscmax, csrmax, celn7o4qrk, llqc, llqr
    )

    (
        snow_riming_rate_c2s,
        graupel_riming_rate_c2g,
        rain_shedding_rate_c2r,
        snow_autoconversion_rate_s2g,
    ) = _riming_in_clouds(temperature, qc, crim, cslam, celnrimexp_g, celn3o4qsk, llqc, llqs)

    dist_cldtop, reduce_dep = _reduced_deposition_in_clouds(
        temperature,
        qv_kup,
        qc_kup,
        qi_kup,
        qs_kup,
        qg_kup,
        qvsw_kup,
        dz,
        dist_cldtop_kup,
        k_lev,
        startmoist_level,
        is_surface,
        llqc,
    )

    (
        snow_ice_collision_rate_i2s,
        graupel_ice_collision_rate_i2g,
        ice_autoconverson_rate_i2s,
        ice_deposition_rate_v2i,
        rain_ice_2graupel_ice_loss_rate_i2g,
        rain_ice_2graupel_rain_loss_rate_r2g,
        ice_dep_autoconversion_rate_i2s,
        ice_net_deposition_rate_v2i,
        ice_net_sublimation_rate_v2i,
    ) = _collision_and_ice_deposition_in_cold_ice_clouds(
        temperature,
        rho,
        qv,
        qi,
        qs,
        qvsi,
        rhoqi_intermediate,
        dt,
        cslam,
        cidep,
        cagg,
        cmi,
        ice_stickeff_min,
        reduce_dep,
        celnrimexp_g,
        celn7o8qrk,
        celn13o8qrk,
        llqi,
    )

    (
        snow_deposition_rate_v2s_in_cold_clouds,
        graupel_deposition_rate_v2g_in_cold_clouds,
    ) = _snow_and_graupel_depositional_growth_in_cold_ice_clouds(
        temperature,
        pres,
        qv,
        qs,
        qvsi,
        dt,
        ice_net_deposition_rate_v2i,
        cslam,
        cbsdep,
        csdep,
        reduce_dep,
        celn6qgk,
        llqi,
        llqs,
        llqg,
    )

    (
        ice_melting_rate_i2c,
        snow_melting_rate_s2r,
        graupel_melting_rate_g2r,
        snow_deposition_rate_v2s_in_melting_condition,
        graupel_deposition_rate_v2g_in_melting_condition,
        rain_deposition_rate_v2r,
    ) = _melting(
        temperature,
        pres,
        rho,
        qv,
        qvsw,
        rhoqi_intermediate,
        dt,
        cssmax,
        csgmax,
        celn8qsk,
        celn6qgk,
        llqi,
        llqs,
        llqg,
    )

    (
        rain_evaporation_rate_r2v,
        rain_freezing_rate_r2g,
    ) = _evaporation_and_freezing_in_subsaturated_air(
        temperature,
        qv,
        qc,
        qvsw,
        rhoqr,
        dt,
        rain_freezing_rate_r2g_in_clouds,
        csrmax,
        bev,
        bevxp,
        cev,
        cevxp,
        celn7o4qrk,
        llqr,
    )

    # ------------------------------------------------------------------------------
    #  Section 6: Check for negative mass
    # ------------------------------------------------------------------------------

    snow_deposition_rate_v2s = (
        snow_deposition_rate_v2s_in_cold_clouds + snow_deposition_rate_v2s_in_melting_condition
    )
    graupel_deposition_rate_v2g = (
        graupel_deposition_rate_v2g_in_cold_clouds
        + graupel_deposition_rate_v2g_in_melting_condition
    )

    # finalizing transfer rates in clouds and calculate depositional growth reduction
    if llqc & (temperature > icon_graupel_params.homogeneous_freeze_temperature):
        # Check for maximum depletion of cloud water and adjust the
        # transfer rates accordingly
        csum = (
            cloud_autoconversion_rate_c2r
            + rain_cloud_collision_rate_c2r
            + snow_riming_rate_c2s
            + graupel_riming_rate_c2g
            + rain_shedding_rate_c2r
        )
        ccorr = cscmax / maximum(cscmax, csum)
        cloud_autoconversion_rate_c2r = ccorr * cloud_autoconversion_rate_c2r
        rain_cloud_collision_rate_c2r = ccorr * rain_cloud_collision_rate_c2r
        snow_riming_rate_c2s = ccorr * snow_riming_rate_c2s
        graupel_riming_rate_c2g = ccorr * graupel_riming_rate_c2g
        rain_shedding_rate_c2r = ccorr * rain_shedding_rate_c2r
        snow_autoconversion_rate_s2g = minimum(
            snow_autoconversion_rate_s2g, snow_riming_rate_c2s + cssmax
        )

    if llqi | llqs | llqg:
        if temperature <= icon_graupel_params.melting_temperature:  # cold case
            qvsidiff = qv - qvsi
            csimax = rhoqi_intermediate / rho / dt

            # Check for maximal depletion of cloud ice
            # No check is done for depositional autoconversion (sdau) because
            # this is a always a fraction of the gain rate due to
            # deposition (i.e the sum of this rates is always positive)
            csum = (
                ice_autoconverson_rate_i2s
                + snow_ice_collision_rate_i2s
                + graupel_ice_collision_rate_i2g
                + rain_ice_2graupel_ice_loss_rate_i2g
                + ice_net_sublimation_rate_v2i
            )
            ccorr = csimax / maximum(csimax, csum) if csimax > wpfloat("0.0") else wpfloat("0.0")
            ice_deposition_rate_v2i = (
                ice_net_deposition_rate_v2i - ccorr * ice_net_sublimation_rate_v2i
            )
            ice_autoconverson_rate_i2s = ccorr * ice_autoconverson_rate_i2s
            snow_ice_collision_rate_i2s = ccorr * snow_ice_collision_rate_i2s
            graupel_ice_collision_rate_i2g = ccorr * graupel_ice_collision_rate_i2g
            rain_ice_2graupel_ice_loss_rate_i2g = ccorr * rain_ice_2graupel_ice_loss_rate_i2g
            if qvsidiff < wpfloat("0.0"):
                snow_deposition_rate_v2s = maximum(snow_deposition_rate_v2s, -cssmax)
                graupel_deposition_rate_v2g = maximum(graupel_deposition_rate_v2g, -csgmax)

    csum = rain_evaporation_rate_r2v + rain_freezing_rate_r2g + rain_ice_2graupel_rain_loss_rate_r2g
    ccorr = csrmax / maximum(csrmax, csum) if csum > wpfloat("0.0") else wpfloat("1.0")
    rain_evaporation_rate_r2v = ccorr * rain_evaporation_rate_r2v
    rain_freezing_rate_r2g = ccorr * rain_freezing_rate_r2g
    rain_ice_2graupel_rain_loss_rate_r2g = ccorr * rain_ice_2graupel_rain_loss_rate_r2g

    # limit snow depletion in order to avoid negative values of qs
    ccorr = wpfloat("1.0")
    if snow_deposition_rate_v2s <= wpfloat("0.0"):
        csum = snow_melting_rate_s2r + snow_autoconversion_rate_s2g - snow_deposition_rate_v2s
        if csum > wpfloat("0.0"):
            ccorr = cssmax / maximum(cssmax, csum)
        snow_melting_rate_s2r = ccorr * snow_melting_rate_s2r
        snow_autoconversion_rate_s2g = ccorr * snow_autoconversion_rate_s2g
        snow_deposition_rate_v2s = ccorr * snow_deposition_rate_v2s
    else:
        csum = snow_melting_rate_s2r + snow_autoconversion_rate_s2g
        if csum > wpfloat("0.0"):
            ccorr = cssmax / maximum(cssmax, csum)
        snow_melting_rate_s2r = ccorr * snow_melting_rate_s2r
        snow_autoconversion_rate_s2g = ccorr * snow_autoconversion_rate_s2g

    # ------------------------------------------------------------------------------
    #  Section 7: Update tendencies
    # ------------------------------------------------------------------------------

    cqvt = (
        rain_evaporation_rate_r2v
        - ice_deposition_rate_v2i
        - snow_deposition_rate_v2s
        - graupel_deposition_rate_v2g
        - ice_nucleation_rate_v2i
        - rain_deposition_rate_v2r
    )
    cqct = (
        ice_melting_rate_i2c
        - cloud_autoconversion_rate_c2r
        - cloud_freezing_rate_c2i
        - rain_cloud_collision_rate_c2r
        - rain_shedding_rate_c2r
        - snow_riming_rate_c2s
        - graupel_riming_rate_c2g
    )
    cqit = (
        ice_nucleation_rate_v2i
        + cloud_freezing_rate_c2i
        - ice_melting_rate_i2c
        - rain_ice_2graupel_ice_loss_rate_i2g
        + ice_deposition_rate_v2i
        - ice_dep_autoconversion_rate_i2s
        - snow_ice_collision_rate_i2s
        - graupel_ice_collision_rate_i2g
        - ice_autoconverson_rate_i2s
    )
    cqrt = (
        cloud_autoconversion_rate_c2r
        + rain_shedding_rate_c2r
        + rain_cloud_collision_rate_c2r
        + snow_melting_rate_s2r
        + graupel_melting_rate_g2r
        - rain_evaporation_rate_r2v
        - rain_ice_2graupel_rain_loss_rate_r2g
        - rain_freezing_rate_r2g
        + rain_deposition_rate_v2r
    )
    cqst = (
        ice_autoconverson_rate_i2s
        + ice_dep_autoconversion_rate_i2s
        - snow_melting_rate_s2r
        + snow_riming_rate_c2s
        + snow_deposition_rate_v2s
        + snow_ice_collision_rate_i2s
        - snow_autoconversion_rate_s2g
    )
    cqgt = (
        graupel_ice_collision_rate_i2g
        - graupel_melting_rate_g2r
        + rain_ice_2graupel_ice_loss_rate_i2g
        + rain_ice_2graupel_rain_loss_rate_r2g
        + graupel_deposition_rate_v2g
        + rain_freezing_rate_r2g
        + graupel_riming_rate_c2g
        + snow_autoconversion_rate_s2g
    )

    temperature_tendency = heat_cap_r * (lhv * (cqct + cqrt) + lhs * (cqit + cqst + cqgt))
    qi_tendency = maximum((rhoqi_intermediate / rho * cimi - qi) / dt + cqit * cimi, -qi / dt)
    qr_tendency = maximum((rhoqr_intermediate / rho * cimr - qr) / dt + cqrt * cimr, -qr / dt)
    qs_tendency = maximum((rhoqs_intermediate / rho * cims - qs) / dt + cqst * cims, -qs / dt)
    qg_tendency = maximum((rhoqg_intermediate / rho * cimg - qg) / dt + cqgt * cimg, -qg / dt)
    qc_tendency = maximum(cqct, -qc / dt)
    qv_tendency = maximum(cqvt, -qv / dt)

    k_lev = k_lev + 1

    return (
        temperature_tendency,
        qv_tendency,
        qc_tendency,
        qi_tendency,
        qr_tendency,
        qs_tendency,
        qg_tendency,
        qv,
        qc,
        qi,
        qr,
        qs,
        qg,
        rhoqrv,
        rhoqsv,
        rhoqgv,
        rhoqiv,
        vnew_r,
        vnew_s,
        vnew_g,
        vnew_i,
        dist_cldtop,
        rho,
        crho1o2,
        crhofac_qi,
        snow_sed0,
        qvsw,
        k_lev,
    )


@field_operator
def _icon_graupel(
    startmoist_level: gtx.int32,
    ground_level: gtx.int32,
    liquid_autoconversion_option: gtx.int32,
    snow_intercept_option: gtx.int32,
    is_isochoric: bool,
    use_constant_water_heat_capacity: bool,
    ice_stickeff_min: wpfloat,
    power_law_coeff_for_ice_mean_fall_speed: wpfloat,
    exponent_for_density_factor_in_ice_sedimentation: wpfloat,
    power_law_coeff_for_snow_fall_speed: wpfloat,
    ccsrim: wpfloat,
    ccsagg: wpfloat,
    ccsvel: wpfloat,
    power_law_exponent_for_rain_mean_fall_speed: wpfloat,
    power_law_coeff_for_rain_mean_fall_speed: wpfloat,
    cevxp: wpfloat,
    cev: wpfloat,
    bevxp: wpfloat,
    bev: wpfloat,
    power_law_exponent_for_rain_mean_fall_speed_ln1o2: wpfloat,
    power_law_exponent_for_ice_mean_fall_speed_ln1o2: wpfloat,
    power_law_exponent_for_graupel_mean_fall_speed_ln1o2: wpfloat,
    dt: wpfloat,
    dz: gtx.Field[[CellDim, KDim], wpfloat],
    temperature: gtx.Field[[CellDim, KDim], wpfloat],
    pres: gtx.Field[[CellDim, KDim], wpfloat],
    rho: gtx.Field[[CellDim, KDim], wpfloat],
    qv: gtx.Field[[CellDim, KDim], wpfloat],
    qc: gtx.Field[[CellDim, KDim], wpfloat],
    qi: gtx.Field[[CellDim, KDim], wpfloat],
    qr: gtx.Field[[CellDim, KDim], wpfloat],
    qs: gtx.Field[[CellDim, KDim], wpfloat],
    qg: gtx.Field[[CellDim, KDim], wpfloat],
    qnc: gtx.Field[[CellDim], wpfloat],
) -> tuple[
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
    gtx.Field[[CellDim, KDim], wpfloat],
]:
    (
        temperature_tendency,
        qv_tendency,
        qc_tendency,
        qi_tendency,
        qr_tendency,
        qs_tendency,
        qg_tendency,
        qv_,
        qc_,
        qi_,
        qr_,
        qs_,
        qg_,
        rhoqrv_old_kup,
        rhoqsv_old_kup,
        rhoqgv_old_kup,
        rhoqiv_old_kup,
        vnew_r,
        vnew_s,
        vnew_g,
        vnew_i,
        dist_cldtop,
        rho_kup,
        crho1o2_kup,
        crhofac_qi_kup,
        snow_sed0_kup,
        qvsw_kup,
        k_lev,
    ) = _icon_graupel_scan(
        startmoist_level,
        ground_level,
        liquid_autoconversion_option,
        snow_intercept_option,
        is_isochoric,
        use_constant_water_heat_capacity,
        ice_stickeff_min,
        power_law_coeff_for_ice_mean_fall_speed,
        exponent_for_density_factor_in_ice_sedimentation,
        power_law_coeff_for_snow_fall_speed,
        ccsrim,
        ccsagg,
        ccsvel,
        power_law_exponent_for_rain_mean_fall_speed,
        power_law_coeff_for_rain_mean_fall_speed,
        cevxp,
        cev,
        bevxp,
        bev,
        power_law_exponent_for_rain_mean_fall_speed_ln1o2,
        power_law_exponent_for_ice_mean_fall_speed_ln1o2,
        power_law_exponent_for_graupel_mean_fall_speed_ln1o2,
        dt,
        dz,
        temperature,
        pres,
        rho,
        qv,
        qc,
        qi,
        qr,
        qs,
        qg,
        qnc,
    )

    return (
        temperature_tendency,
        qv_tendency,
        qc_tendency,
        qi_tendency,
        qr_tendency,
        qs_tendency,
        qg_tendency,
        rhoqrv_old_kup,
        rhoqsv_old_kup,
        rhoqgv_old_kup,
        rhoqiv_old_kup,
        vnew_r,
        vnew_s,
        vnew_g,
        vnew_i,
    )


@program(grid_type=gtx.GridType.UNSTRUCTURED)
def icon_graupel(
    startmoist_level: gtx.int32,
    ground_level: gtx.int32,
    liquid_autoconversion_option: gtx.int32,
    snow_intercept_option: gtx.int32,
    is_isochoric: bool,
    use_constant_water_heat_capacity: bool,
    ice_stickeff_min: wpfloat,
    power_law_coeff_for_ice_mean_fall_speed: wpfloat,
    exponent_for_density_factor_in_ice_sedimentation: wpfloat,
    power_law_coeff_for_snow_fall_speed: wpfloat,
    ccsrim: wpfloat,
    ccsagg: wpfloat,
    ccsvel: wpfloat,
    power_law_exponent_for_rain_mean_fall_speed: wpfloat,
    power_law_coeff_for_rain_mean_fall_speed: wpfloat,
    cevxp: wpfloat,
    cev: wpfloat,
    bevxp: wpfloat,
    bev: wpfloat,
    power_law_exponent_for_rain_mean_fall_speed_ln1o2: wpfloat,
    power_law_exponent_for_ice_mean_fall_speed_ln1o2: wpfloat,
    power_law_exponent_for_graupel_mean_fall_speed_ln1o2: wpfloat,
    dt: wpfloat,
    dz: gtx.Field[[CellDim, KDim], wpfloat],
    temperature: gtx.Field[[CellDim, KDim], wpfloat],
    pres: gtx.Field[[CellDim, KDim], wpfloat],
    rho: gtx.Field[[CellDim, KDim], wpfloat],
    qv: gtx.Field[[CellDim, KDim], wpfloat],
    qc: gtx.Field[[CellDim, KDim], wpfloat],
    qi: gtx.Field[[CellDim, KDim], wpfloat],
    qr: gtx.Field[[CellDim, KDim], wpfloat],
    qs: gtx.Field[[CellDim, KDim], wpfloat],
    qg: gtx.Field[[CellDim, KDim], wpfloat],
    qnc: gtx.Field[[CellDim], wpfloat],
    temperature_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    qv_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    qc_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    qi_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    qr_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    qs_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    qg_tendency: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqrv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqsv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqgv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    rhoqiv_old_kup: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_r: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_s: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_g: gtx.Field[[CellDim, KDim], wpfloat],
    vnew_i: gtx.Field[[CellDim, KDim], wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _icon_graupel(
        startmoist_level,
        ground_level,
        liquid_autoconversion_option,
        snow_intercept_option,
        is_isochoric,
        use_constant_water_heat_capacity,
        ice_stickeff_min,
        power_law_coeff_for_ice_mean_fall_speed,
        exponent_for_density_factor_in_ice_sedimentation,
        power_law_coeff_for_snow_fall_speed,
        ccsrim,
        ccsagg,
        ccsvel,
        power_law_exponent_for_rain_mean_fall_speed,
        power_law_coeff_for_rain_mean_fall_speed,
        cevxp,
        cev,
        bevxp,
        bev,
        power_law_exponent_for_rain_mean_fall_speed_ln1o2,
        power_law_exponent_for_ice_mean_fall_speed_ln1o2,
        power_law_exponent_for_graupel_mean_fall_speed_ln1o2,
        dt,
        dz,
        temperature,
        pres,
        rho,
        qv,
        qc,
        qi,
        qr,
        qs,
        qg,
        qnc,
        out=(
            temperature_tendency,
            qv_tendency,
            qc_tendency,
            qi_tendency,
            qr_tendency,
            qs_tendency,
            qg_tendency,
            rhoqrv_old_kup,
            rhoqsv_old_kup,
            rhoqgv_old_kup,
            rhoqiv_old_kup,
            vnew_r,
            vnew_s,
            vnew_g,
            vnew_i,
        ),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
