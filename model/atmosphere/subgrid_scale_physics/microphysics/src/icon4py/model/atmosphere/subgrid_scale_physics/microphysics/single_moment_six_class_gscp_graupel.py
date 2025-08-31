# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import dataclasses
import math
import sys
from typing import Final

import gt4py.next as gtx
from gt4py.eve import utils as eve_utils
from gt4py.next import backend
from gt4py.next.ffront.decorator import field_operator, program, scan_operator
from gt4py.next.ffront.fbuiltins import broadcast, exp, log, maximum, minimum, sqrt, where

from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import microphysics_constants
from icon4py.model.common import (
    constants as physics_constants,
    dimension as dims,
    model_options,
    type_alias as ta,
)
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, vertical as v_grid
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc


# TODO (Chia Rui): The limit has to be manually set to a huge value for a big scan operator. Remove it when neccesary.
sys.setrecursionlimit(350000)

phy_const: Final = physics_constants.PhysicsConstants()
microphy_const: Final = microphysics_constants.MicrophysicsConstants()


class LiquidAutoConversionType(eve_utils.FrozenNamespace[gtx.int32]):
    """
    Options for computing liquid auto conversion rate
    """

    #: Kessler (1969) liquid auto conversion mode
    KESSLER = 0
    #: Seifert & Beheng (2006) liquid auto conversion mode
    SEIFERT_BEHENG = 1


class SnowInterceptParametererization(eve_utils.FrozenNamespace[gtx.int32]):
    """
    Options for deriving snow intercept parameter
    """

    #: Estimated intercept parameter for the snow size distribution from the best-fit line in figure 10(a) of Field et al. (2005)
    FIELD_BEST_FIT_ESTIMATION = 1
    #: Estimated intercept parameter for the snow size distribution from the general moment equation in table 2 of Field et al. (2005)
    FIELD_GENERAL_MOMENT_ESTIMATION = 2


liquid_auto_conversion_type = LiquidAutoConversionType()
snow_intercept_parameterization = SnowInterceptParametererization()


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

    l_cv is removed. This option is to swtich between whether the microphysical processes are isochoric or isobaric. Originally defined as  as input to the graupel in ICON. It is hardcoded to True in ICON.
    The reason for its existence and isobaric may have a positive effect on reducing sound waves. In case it needs to be restored, is_isochoric is a better name.

    The COSMO microphysics documentation "A Description of the Nonhydrostatic Regional COSMO-Model Part II Physical Parameterizations" can be downloaded via this link: https://www.cosmo-model.org/content/model/cosmo/coreDocumentation/cosmo_physics_6.00.pdf
    """

    #: liquid auto conversion mode. Originally defined as isnow_n0temp (PARAMETER) in gscp_data.f90 in ICON. I keep it because I think the choice depends on resolution.
    liquid_autoconversion_option: LiquidAutoConversionType = LiquidAutoConversionType.KESSLER
    #: snow size distribution interception parameter. Originally defined as isnow_n0temp (PARAMETER) in gscp_data.f90 in ICON. I keep it because I think the choice depends on resolution.
    snow_intercept_option: SnowInterceptParametererization = (
        SnowInterceptParametererization.FIELD_GENERAL_MOMENT_ESTIMATION
    )
    #: Do latent heat nudging. Originally defined as dass_lhn in mo_run_config.f90 in ICON.
    do_latent_heat_nudging = False
    #: Whether a fixed latent heat capacities are used for water. Originally defined as ithermo_water in mo_nwp_tuning_config.f90 in ICON (0 means True).
    use_constant_latent_heat = True
    #: First parameter in RHS of eq. 5.163 in the COSMO microphysics documentation for the sticking efficiency when lstickeff = True (repricated in icon4py because it is always True in ICON). Originally defined as tune_zceff_min in mo_tuning_nwp_config.f90 in ICON.
    ice_stickeff_min: ta.wpfloat = 0.075
    #: Power law coefficient in v-qi ice terminal velocity-mixing ratio relationship, see eq. 5.169 in the COSMO microphysics documentation. Originally defined as tune_zvz0i in mo_tuning_nwp_config.f90 in ICON.
    power_law_coeff_for_ice_mean_fall_speed: ta.wpfloat = 1.25
    #: Exponent of the density factor in ice terminal velocity equation to account for density (air thermodynamic state) change. Originally defined as tune_icesedi_exp in mo_tuning_nwp_config.f90 in ICON.
    exponent_for_density_factor_in_ice_sedimentation: ta.wpfloat = 0.33
    #: Power law coefficient in v-D snow terminal velocity-Diameter relationship, see eqs. 5.57 (depricated after COSMO 3.0) and unnumbered eq. (v = 25 D^0.5) below eq. 5.159 in the COSMO microphysics documentation. Originally defined as tune_v0snow in mo_tuning_nwp_config.f90 in ICON.
    power_law_coeff_for_snow_fall_speed: ta.wpfloat = 20.0
    #: mu exponential factor in gamma distribution of rain particles. Originally defined as mu_rain in mo_nwp_tuning_config.f90 in ICON.
    rain_mu: ta.wpfloat = 0.0
    #: Interception parameter in gamma distribution of rain particles. Originally defined as rain_n0_factor in mo_nwp_tuning_config.f90 in ICON.
    rain_n0: ta.wpfloat = 1.0
    #: coefficient for snow-graupel conversion by riming. Originally defined as csg in mo_nwp_tuning_config.f90 in ICON.
    snow2graupel_riming_coeff: ta.wpfloat = 0.5


@gtx.field_operator
def _compute_cooper_inp_concentration(temperature: ta.wpfloat) -> ta.wpfloat:
    cnin = 5.0 * exp(0.304 * (phy_const.tmelt - temperature))
    cnin = minimum(cnin, microphy_const.nimax_thom)
    return cnin


@gtx.field_operator
def _compute_snow_interception_and_collision_parameters(
    temperature: ta.wpfloat,
    rho: ta.wpfloat,
    qs: ta.wpfloat,
    precomputed_riming_coef: ta.wpfloat,
    precomputed_agg_coef: ta.wpfloat,
    precomputed_snow_sed_coef: ta.wpfloat,
    power_law_coeff_for_snow_fall_speed: ta.wpfloat,
    llqs: bool,
    snow_intercept_option: gtx.int32,
) -> tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat, ta.wpfloat, ta.wpfloat]:
    """
    Compute the intercept parameter, N0, of the snow exponential size distribution.

    First method: Explained in paragraphs at pages 2008 and 2009 in Field et al. (2005). N0s_23 = (M_2)^4 / (M_3)^3, M_2 = Gamma(3) N0s / lamda^3, M_2 = Gamma(4) N0s / lamda^4, so N0s_23 = 2/27 N0s. And N0s_23 = 5.65E5 exp(-0.107Tc)
    Second method: Eq. 5.160 in the COSMO microphysics documentation, originally in Table 2 in Field et al. (2005).

    Args:
        temperature: air temperature [K]
        rho: air density [kg/m3]
        qs: specific snow content [kg/kg]
        precomputed_riming_coef: parameter for snow riming with clouds
        precomputed_agg_coef: parameter for ice aggregation (becomes snow)
        precomputed_snow_sed_coef: parameter for snow sedimentation
        power_law_coeff_for_snow_fall_speed: power law coefficient in snow v-D relationship
        llqs: snow grid cell
        snow_intercept_option: estimation method for snow intercept parameter
    Returns:
        n0s: snow size distribution intercept parameter
        snow_sed0: integration factor for snow sedimendation
        crim: riming parameter
        cagg: aggregation parameter
        cbsdep: deposition parameter
    """
    if llqs:
        if snow_intercept_option == snow_intercept_parameterization.FIELD_BEST_FIT_ESTIMATION:
            # Calculate n0s using the temperature-dependent
            # formula of Field et al. (2005)
            local_tc = temperature - phy_const.tmelt
            local_tc = minimum(local_tc, wpfloat("0.0"))
            local_tc = maximum(local_tc, wpfloat("-40.0"))
            n0s = microphy_const.snow_intercept_parameter_n0s1 * exp(
                microphy_const.snow_intercept_parameter_n0s2 * local_tc
            )
            n0s = minimum(n0s, wpfloat("1.0e9"))
            n0s = maximum(n0s, wpfloat("1.0e6"))

        elif (
            snow_intercept_option == snow_intercept_parameterization.FIELD_GENERAL_MOMENT_ESTIMATION
        ):
            # Calculate n0s using the temperature-dependent moment
            # relations of Field et al. (2005)
            local_tc = temperature - phy_const.tmelt
            local_tc = minimum(local_tc, wpfloat("0.0"))
            local_tc = maximum(local_tc, wpfloat("-40.0"))

            local_nnr = wpfloat("3.0")
            local_hlp = (
                microphy_const.snow_intercept_parameter_mma[0]
                + microphy_const.snow_intercept_parameter_mma[1] * local_tc
                + microphy_const.snow_intercept_parameter_mma[2] * local_nnr
                + microphy_const.snow_intercept_parameter_mma[3] * local_tc * local_nnr
                + microphy_const.snow_intercept_parameter_mma[4] * local_tc**2.0
                + microphy_const.snow_intercept_parameter_mma[5] * local_nnr**2.0
                + microphy_const.snow_intercept_parameter_mma[6] * local_tc**2.0 * local_nnr
                + microphy_const.snow_intercept_parameter_mma[7] * local_tc * local_nnr**2.0
                + microphy_const.snow_intercept_parameter_mma[8] * local_tc**3.0
                + microphy_const.snow_intercept_parameter_mma[9] * local_nnr**3.0
            )
            local_alf = exp(local_hlp * log(wpfloat("10.0")))
            local_bet = (
                microphy_const.snow_intercept_parameter_mmb[0]
                + microphy_const.snow_intercept_parameter_mmb[1] * local_tc
                + microphy_const.snow_intercept_parameter_mmb[2] * local_nnr
                + microphy_const.snow_intercept_parameter_mmb[3] * local_tc * local_nnr
                + microphy_const.snow_intercept_parameter_mmb[4] * local_tc**2.0
                + microphy_const.snow_intercept_parameter_mmb[5] * local_nnr**2.0
                + microphy_const.snow_intercept_parameter_mmb[6] * local_tc**2.0 * local_nnr
                + microphy_const.snow_intercept_parameter_mmb[7] * local_tc * local_nnr**2.0
                + microphy_const.snow_intercept_parameter_mmb[8] * local_tc**3.0
                + microphy_const.snow_intercept_parameter_mmb[9] * local_nnr**3.0
            )

            # Here is the exponent bms=2.0 hardwired# not ideal# (Uli Blahak)
            local_m2s = (
                qs * rho / microphy_const.power_law_coeff_for_snow_mD_relation
            )  # UB rho added as bugfix
            local_m3s = local_alf * exp(local_bet * log(local_m2s))

            local_hlp = microphy_const.snow_intercept_parameter_n0s1 * exp(
                microphy_const.snow_intercept_parameter_n0s2 * local_tc
            )
            n0s = wpfloat("13.50") * local_m2s * (local_m2s / local_m3s) ** 3.0
            n0s = maximum(n0s, wpfloat("0.5") * local_hlp)
            n0s = minimum(n0s, wpfloat("1.0e2") * local_hlp)
            n0s = minimum(n0s, wpfloat("1.0e9"))
            n0s = maximum(n0s, wpfloat("1.0e6"))

        else:
            n0s = microphy_const.snow_default_intercept_param

        # compute integration factor for terminal velocity
        snow_sed0 = precomputed_snow_sed_coef * exp(microphy_const.ccsvxp * log(n0s))
        # compute constants for riming, aggregation, and deposition processes for snow
        crim = precomputed_riming_coef * n0s
        cagg = precomputed_agg_coef * n0s
        cbsdep = microphy_const.ccsdep * sqrt(power_law_coeff_for_snow_fall_speed)
    else:
        n0s = microphy_const.snow_default_intercept_param
        snow_sed0 = wpfloat("0.0")
        crim = wpfloat("0.0")
        cagg = wpfloat("0.0")
        cbsdep = wpfloat("0.0")

    return n0s, snow_sed0, crim, cagg, cbsdep


@gtx.field_operator
def _deposition_nucleation_at_low_temperature_or_in_clouds(
    temperature: ta.wpfloat,
    rho: ta.wpfloat,
    qv: ta.wpfloat,
    qi: ta.wpfloat,
    qvsi: ta.wpfloat,
    cnin: ta.wpfloat,
    dtime: ta.wpfloat,
    llqc: bool,
) -> ta.wpfloat:
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
        dtime: time step [s]
        llqc: cloud grid cell
    Returns:
        Deposition nucleation rate
    """
    ice_nucleation_rate_v2i = (
        microphy_const.ice_initial_mass / rho * cnin / dtime
        if (llqc & (temperature <= wpfloat("267.15")) & (qi <= microphy_const.qmin))
        | (
            (temperature < microphy_const.heterogeneous_freeze_temperature)
            & (qv > wpfloat("8.0e-6"))
            & (qi <= wpfloat("0.0"))
            & (qv > qvsi)
        )
        else wpfloat("0.0")
    )
    return ice_nucleation_rate_v2i


@gtx.field_operator
def _autoconversion_and_rain_accretion(
    temperature: ta.wpfloat,
    qc: ta.wpfloat,
    qr: ta.wpfloat,
    qnc: ta.wpfloat,
    celn7o8qrk: ta.wpfloat,
    llqc: bool,
    liquid_autoconversion_option: gtx.int32,
) -> tuple[ta.wpfloat, ta.wpfloat]:
    """
    Compute the rate of cloud-to-rain autoconversion and the mass of cloud accreted by rain.
    Method 1: liquid_autoconversion_option = LiquidAutoConversionType.KESSLER, Kessler (1969)
    Method 2: liquid_autoconversion_option = LiquidAutoConversionType.SEIFERT_BEHENG, Seifert and beheng (2001)

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
    if llqc & (temperature > microphy_const.homogeneous_freeze_temperature):
        if liquid_autoconversion_option == liquid_auto_conversion_type.KESSLER:
            # Kessler(1969) autoconversion rate
            cloud_autoconversion_rate_c2r = (
                microphy_const.kessler_cloud2rain_autoconversion_coeff_for_cloud
                * maximum(qc - microphy_const.qc0, wpfloat("0.0"))
            )
            rain_cloud_collision_rate_c2r = (
                microphy_const.kessler_cloud2rain_autoconversion_coeff_for_rain * qc * celn7o8qrk
            )

        elif liquid_autoconversion_option == liquid_auto_conversion_type.SEIFERT_BEHENG:
            # Seifert and Beheng (2001) autoconversion rate
            local_const = (
                microphy_const.kcau
                / (wpfloat("20.0") * microphy_const.xstar)
                * (microphy_const.cnue + wpfloat("2.0"))
                * (microphy_const.cnue + wpfloat("4.0"))
                / (microphy_const.cnue + wpfloat("1.0")) ** 2.0
            )

            # with constant cloud droplet number concentration qnc
            if qc > wpfloat("1.0e-6"):
                local_tau = minimum(wpfloat("1.0") - qc / (qc + qr), wpfloat("0.9"))
                local_tau = maximum(local_tau, wpfloat("1.0e-30"))
                local_hlp = exp(microphy_const.kphi2 * log(local_tau))
                local_phi = microphy_const.kphi1 * local_hlp * (wpfloat("1.0") - local_hlp) ** 3.0
                cloud_autoconversion_rate_c2r = (
                    local_const
                    * qc
                    * qc
                    * qc
                    * qc
                    / (qnc * qnc)
                    * (wpfloat("1.0") + local_phi / (wpfloat("1.0") - local_tau) ** 2.0)
                )
                local_phi = (local_tau / (local_tau + microphy_const.kphi3)) ** 4.0
                rain_cloud_collision_rate_c2r = microphy_const.kcac * qc * qr * local_phi
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
    temperature: ta.wpfloat,
    qc: ta.wpfloat,
    qr: ta.wpfloat,
    cscmax: ta.wpfloat,
    csrmax: ta.wpfloat,
    celn7o4qrk: ta.wpfloat,
    llqc: bool,
    llqr: bool,
) -> tuple[ta.wpfloat, ta.wpfloat]:
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
        if temperature > microphy_const.homogeneous_freeze_temperature:
            # Calculation of in-cloud rainwater freezing
            if (
                llqr
                & (temperature < microphy_const.threshold_freeze_temperature)
                & (qr > wpfloat("0.1") * qc)
            ):
                rain_freezing_rate_r2g_in_clouds = (
                    microphy_const.coeff_rain_freeze1
                    * (
                        exp(
                            microphy_const.coeff_rain_freeze2
                            * (microphy_const.threshold_freeze_temperature - temperature)
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
    temperature: ta.wpfloat,
    qc: ta.wpfloat,
    crim: ta.wpfloat,
    cslam: ta.wpfloat,
    celnrimexp_g: ta.wpfloat,
    celn3o4qsk: ta.wpfloat,
    snow2graupel_riming_coeff: ta.wpfloat,
    llqc: bool,
    llqs: bool,
) -> tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat, ta.wpfloat]:
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
    if llqc & (temperature > microphy_const.homogeneous_freeze_temperature):
        if llqs:
            snow_riming_rate_c2s = crim * qc * exp(microphy_const.ccsaxp * log(cslam))
        else:
            snow_riming_rate_c2s = wpfloat("0.0")

        graupel_riming_rate_c2g = microphy_const.crim_g * qc * celnrimexp_g

        if temperature >= phy_const.tmelt:
            rain_shedding_rate_c2r = snow_riming_rate_c2s + graupel_riming_rate_c2g
            snow_riming_rate_c2s = wpfloat("0.0")
            graupel_riming_rate_c2g = wpfloat("0.0")
            snow_autoconversion_rate_s2g = wpfloat("0.0")
        else:
            if qc >= microphy_const.qc0:
                snow_autoconversion_rate_s2g = snow2graupel_riming_coeff * qc * celn3o4qsk
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
    temperature: ta.wpfloat,
    qv_kup: ta.wpfloat,
    qc_kup: ta.wpfloat,
    qi_kup: ta.wpfloat,
    qs_kup: ta.wpfloat,
    qg_kup: ta.wpfloat,
    qvsw_kup: ta.wpfloat,
    dz: ta.wpfloat,
    dist_cldtop_kup: ta.wpfloat,
    k_lev: gtx.int32,
    is_surface: bool,
    llqc: bool,
) -> tuple[ta.wpfloat, ta.wpfloat]:
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
        is_surface: True if the current k level is at the bottom
        llqc: cloud grid cell
    Returns:
        vertical distance to cloud top, reduced factor for ice deposition
    """
    if llqc:
        if (k_lev > 0) & (not is_surface):
            cqcgk_1 = qi_kup + qs_kup + qg_kup

            # distance from cloud top
            if (qv_kup + qc_kup < qvsw_kup) & (cqcgk_1 < microphy_const.qmin):
                # upper cloud layer
                dist_cldtop = wpfloat("0.0")  # reset distance to upper cloud layer
            else:
                dist_cldtop = dist_cldtop_kup + dz
        else:
            dist_cldtop = dist_cldtop_kup

        if (k_lev > 0) & (not is_surface):
            # finalizing transfer rates in clouds and calculate depositional growth reduction
            cnin = _compute_cooper_inp_concentration(temperature)
            cfnuc = minimum(cnin / microphy_const.nimix, wpfloat("1.0"))

            # with asymptotic behaviour dz -> 0 (xxx)
            #        reduce_dep = MIN(fnuc + (1.0_wp-fnuc)*(reduce_dep_ref + &
            #                             dist_cldtop(iv)/dist_cldtop_ref + &
            #                             (1.0_wp-reduce_dep_ref)*(zdh/dist_cldtop_ref)**4), 1.0_wp)

            # without asymptotic behaviour dz -> 0
            reduce_dep = cfnuc + (wpfloat("1.0") - cfnuc) * (
                microphy_const.reduce_dep_ref + dist_cldtop / microphy_const.dist_cldtop_ref
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
    temperature: ta.wpfloat,
    rho: ta.wpfloat,
    qv: ta.wpfloat,
    qi: ta.wpfloat,
    qs: ta.wpfloat,
    qvsi: ta.wpfloat,
    rhoqi_intermediate: ta.wpfloat,
    dtime: ta.wpfloat,
    cslam: ta.wpfloat,
    cidep: ta.wpfloat,
    cagg: ta.wpfloat,
    cmi: ta.wpfloat,
    ice_stickeff_min: ta.wpfloat,
    reduce_dep: ta.wpfloat,
    celnrimexp_g: ta.wpfloat,
    celn7o8qrk: ta.wpfloat,
    celn13o8qrk: ta.wpfloat,
    llqi: bool,
) -> tuple[
    ta.wpfloat,
    ta.wpfloat,
    ta.wpfloat,
    ta.wpfloat,
    ta.wpfloat,
    ta.wpfloat,
    ta.wpfloat,
    ta.wpfloat,
    ta.wpfloat,
]:
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
        dtime: time step [s]
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
    if (temperature <= phy_const.tmelt) & llqi:
        # Change in sticking efficiency needed in case of cloud ice sedimentation
        # (based on Guenther Zaengls work)
        local_eff = minimum(
            exp(wpfloat("0.09") * (temperature - phy_const.tmelt)),
            wpfloat("1.0"),
        )
        local_eff = maximum(local_eff, ice_stickeff_min)
        local_eff = maximum(
            local_eff,
            microphy_const.ice_sticking_eff_factor
            * (temperature - microphy_const.tmin_iceautoconv),
        )

        local_nid = rho * qi / cmi
        local_lnlogmi = log(cmi)

        local_qvsidiff = qv - qvsi
        local_svmax = local_qvsidiff / dtime

        snow_ice_collision_rate_i2s = (
            local_eff * qi * cagg * exp(microphy_const.ccsaxp * log(cslam))
        )
        graupel_ice_collision_rate_i2g = local_eff * qi * microphy_const.cagg_g * celnrimexp_g
        ice_autoconverson_rate_i2s = (
            local_eff * microphy_const.ciau * maximum(qi - microphy_const.qi0, wpfloat("0.0"))
        )

        rain_ice_2graupel_ice_loss_rate_i2g = microphy_const.cicri * qi * celn7o8qrk
        if qs > wpfloat("1.0e-7"):
            rain_ice_2graupel_rain_loss_rate_r2g = microphy_const.crcri * (qi / cmi) * celn13o8qrk
        else:
            rain_ice_2graupel_rain_loss_rate_r2g = wpfloat("0.0")

        local_icetotaldeposition = (
            cidep * local_nid * exp(wpfloat("0.33") * local_lnlogmi) * local_qvsidiff
        )
        ice_deposition_rate_v2i = local_icetotaldeposition

        # for sedimenting quantities the maximum
        # allowed depletion is determined by the predictor value.
        local_simax = rhoqi_intermediate / rho / dtime

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

        local_lnlogmi = log(microphy_const.msmin / cmi)
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
    temperature: ta.wpfloat,
    pressure: ta.wpfloat,
    qv: ta.wpfloat,
    qs: ta.wpfloat,
    qvsi: ta.wpfloat,
    dtime: ta.wpfloat,
    ice_net_deposition_rate_v2i: ta.wpfloat,
    cslam: ta.wpfloat,
    cbsdep: ta.wpfloat,
    csdep: ta.wpfloat,
    reduce_dep: ta.wpfloat,
    celn6qgk: ta.wpfloat,
    llqi: bool,
    llqs: bool,
    llqg: bool,
) -> tuple[ta.wpfloat, ta.wpfloat]:
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
        pressure: air pressure [Pa]
        qv: specific humidity [kg/kg]
        qs: specific snow content [kg/kg]
        qvsi: saturated vapor mixing ratio over ice
        dtime: time step [s]
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
        if temperature <= phy_const.tmelt:
            local_qvsidiff = qv - qvsi
            local_svmax = local_qvsidiff / dtime

            local_xfac = wpfloat("1.0") + cbsdep * exp(microphy_const.ccsdxp * log(cslam))
            snow_deposition_rate_v2s_in_cold_clouds = (
                csdep * local_xfac * local_qvsidiff / (cslam + phy_const.eps) ** 2.0
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
                    + wpfloat("2554.99") / pressure
                    + wpfloat("2.6531e-7") * pressure
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
    temperature: ta.wpfloat,
    pressure: ta.wpfloat,
    rho: ta.wpfloat,
    qv: ta.wpfloat,
    qvsw: ta.wpfloat,
    rhoqi_intermediate: ta.wpfloat,
    dtime: ta.wpfloat,
    cssmax: ta.wpfloat,
    csgmax: ta.wpfloat,
    celn8qsk: ta.wpfloat,
    celn6qgk: ta.wpfloat,
    llqi: bool,
    llqs: bool,
    llqg: bool,
) -> tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat, ta.wpfloat, ta.wpfloat, ta.wpfloat]:
    """
    Compute the vapor deposition of ice crystals, snow, and graupel in ice clouds when temperature is above zero degree celcius.
    When the air is supersubsaturated over both ice and water, depositional growth of snow and graupel is converted to growth of rain.
    (Please refer to the COSMO microphysics documentation via the link given in the docstring of SingleMomentSixClassIconGraupelConfig for all the equations)

    Ice crystals completely melt when temperature is above zero.
    For snow and graupel, follow Eqs. 5.141 - 5.146

    Args:
        temperature: air temperature [K]
        pressure: air pressure [Pa]
        rho: air density [kg/m3]
        qv: specific humidity [kg/kg]
        qvsw: saturated vapor mixing ratio
        rhoqi_intermediate: ice mass with sedimendation flux from above [kg/m3]
        dtime: time step [s]
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
        if temperature > phy_const.tmelt:
            # cloud ice melts instantaneously
            ice_melting_rate_i2c = rhoqi_intermediate / rho / dtime

            local_qvsw0 = microphy_const.pvsw0 / (rho * phy_const.rv * phy_const.tmelt)
            local_qvsw0diff = qv - local_qvsw0

            # ** GZ: several numerical fits in this section should be replaced with physically more meaningful formulations **
            if temperature > phy_const.tmelt - microphy_const.tcrit * local_qvsw0diff:
                # calculate melting rate
                local_x1 = temperature - phy_const.tmelt + microphy_const.asmel * local_qvsw0diff
                snow_melting_rate_s2r = (
                    (wpfloat("79.6863") / pressure + wpfloat("0.612654e-3")) * local_x1 * celn8qsk
                )
                snow_melting_rate_s2r = minimum(snow_melting_rate_s2r, cssmax)
                graupel_melting_rate_g2r = (
                    (wpfloat("12.31698") / pressure + wpfloat("7.39441e-05")) * local_x1 * celn6qgk
                )
                graupel_melting_rate_g2r = minimum(graupel_melting_rate_g2r, csgmax)
                # deposition + melting, ice particle temperature: t0
                # calculation without howell-factor!
                snow_deposition_rate_v2s_in_melting_condition = (
                    (wpfloat("31282.3") / pressure + wpfloat("0.241897"))
                    * local_qvsw0diff
                    * celn8qsk
                )
                graupel_deposition_rate_v2g_in_melting_condition = (
                    (wpfloat("0.153907") - pressure * wpfloat("7.86703e-07"))
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
                    (wpfloat("0.28003") - pressure * wpfloat("0.146293e-6"))
                    * local_qvsidiff
                    * celn8qsk
                )
                graupel_deposition_rate_v2g_in_melting_condition = (
                    (wpfloat("0.0418521") - pressure * wpfloat("4.7524e-8"))
                    * local_qvsidiff
                    * celn6qgk
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
    temperature: ta.wpfloat,
    qv: ta.wpfloat,
    qc: ta.wpfloat,
    qvsw: ta.wpfloat,
    rhoqr: ta.wpfloat,
    dtime: ta.wpfloat,
    rain_freezing_rate_r2g_in_clouds: ta.wpfloat,
    csrmax: ta.wpfloat,
    precomputed_evaporation_alpha_exp_coeff: ta.wpfloat,
    precomputed_evaporation_alpha_coeff: ta.wpfloat,
    precomputed_evaporation_beta_exp_coeff: ta.wpfloat,
    precomputed_evaporation_beta_coeff: ta.wpfloat,
    celn7o4qrk: ta.wpfloat,
    llqr: bool,
) -> tuple[ta.wpfloat, ta.wpfloat]:
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
        dtime: time step [s]
        rain_freezing_rate_r2g_in_clouds: rain freezing transfer rate in clouds
        csrmax: maximum specific rain content
        precomputed_evaporation_alpha_exp_coeff: constant (refer to equation or documentation in the docstring above)
        precomputed_evaporation_alpha_coeff: constant (refer to equation or documentation in the docstring above)
        precomputed_evaporation_beta_exp_coeff: constant (refer to equation or documentation in the docstring above)
        precomputed_evaporation_beta_coeff: constant (refer to equation or documentation in the docstring above)
        celn7o4qrk: constant (refer to equation or documentation in the docstring above)
        llqr: rain grid cell
    Returns:
        evaporation rate of rain, freezing rate of rain
    """
    rain_freezing_rate_r2g = rain_freezing_rate_r2g_in_clouds
    if llqr & (qv + qc <= qvsw):
        local_lnqr = log(rhoqr)
        local_x1 = wpfloat("1.0") + precomputed_evaporation_beta_coeff * exp(
            precomputed_evaporation_beta_exp_coeff * local_lnqr
        )
        # Limit evaporation rate in order to avoid overshoots towards supersaturation, the pre-factor approximates (esat(T_wb)-e)/(esat(T)-e) at temperatures between 0 degC and 30 degC
        local_temp_c = temperature - phy_const.tmelt
        local_maxevap = (
            (
                wpfloat("0.61")
                - wpfloat("0.0163") * local_temp_c
                + wpfloat("1.111e-4") * local_temp_c**2.0
            )
            * (qvsw - qv)
            / dtime
        )
        rain_evaporation_rate_r2v = (
            precomputed_evaporation_alpha_coeff
            * local_x1
            * (qvsw - qv)
            * exp(precomputed_evaporation_alpha_exp_coeff * local_lnqr)
        )
        rain_evaporation_rate_r2v = minimum(rain_evaporation_rate_r2v, local_maxevap)

        if temperature > microphy_const.homogeneous_freeze_temperature:
            # Calculation of below-cloud rainwater freezing
            if temperature < microphy_const.threshold_freeze_temperature:
                # FR new: reduced rain freezing rate
                rain_freezing_rate_r2g = (
                    microphy_const.coeff_rain_freeze1
                    * (
                        exp(
                            microphy_const.coeff_rain_freeze2
                            * (microphy_const.threshold_freeze_temperature - temperature)
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


@gtx.field_operator
def sat_pres_water(temperature: ta.wpfloat) -> ta.wpfloat:
    return microphy_const.tetens_p0 * exp(
        microphy_const.tetens_aw
        * (temperature - phy_const.tmelt)
        / (temperature - microphy_const.tetens_bw)
    )


@gtx.field_operator
def sat_pres_ice(temperature: ta.wpfloat) -> ta.wpfloat:
    return microphy_const.tetens_p0 * exp(
        microphy_const.tetens_ai
        * (temperature - phy_const.tmelt)
        / (temperature - microphy_const.tetens_bi)
    )


@dataclasses.dataclass
class MetricStateIconGraupel:
    ddqz_z_full: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat]


class SingleMomentSixClassIconGraupel:
    def __init__(
        self,
        graupel_config: SingleMomentSixClassIconGraupelConfig,
        grid: icon_grid.IconGrid | None,
        metric_state: MetricStateIconGraupel | None,
        vertical_params: v_grid.VerticalGrid | None,
        backend: backend.Backend,
    ):
        self.config = graupel_config
        self._initialize_configurable_parameters()
        self._grid = grid
        self.metric_state = metric_state
        self.vertical_params = vertical_params
        self._backend = backend

        self._initialize_local_fields()
        self._determine_horizontal_domains()
        self._initialize_gt4py_programs()

    def _initialize_configurable_parameters(self):
        precomputed_riming_coef: ta.wpfloat = (
            0.25
            * math.pi
            * microphy_const.snow_cloud_collection_eff
            * self.config.power_law_coeff_for_snow_fall_speed
            * math.gamma(microphy_const.power_law_exponent_for_snow_fall_speed + 3.0)
        )
        precomputed_agg_coef: ta.wpfloat = (
            0.25
            * math.pi
            * self.config.power_law_coeff_for_snow_fall_speed
            * math.gamma(microphy_const.power_law_exponent_for_snow_fall_speed + 3.0)
        )
        _ccsvxp = -(
            microphy_const.power_law_exponent_for_snow_fall_speed
            / (microphy_const.power_law_exponent_for_snow_mD_relation + 1.0)
            + 1.0
        )
        precomputed_snow_sed_coef: ta.wpfloat = (
            microphy_const.power_law_coeff_for_snow_mD_relation
            * self.config.power_law_coeff_for_snow_fall_speed
            * math.gamma(
                microphy_const.power_law_exponent_for_snow_mD_relation
                + microphy_const.power_law_exponent_for_snow_fall_speed
                + 1.0
            )
            * (
                microphy_const.power_law_coeff_for_snow_mD_relation
                * math.gamma(microphy_const.power_law_exponent_for_snow_mD_relation + 1.0)
            )
            ** _ccsvxp
        )
        _n0r: ta.wpfloat = (
            8.0e6 * math.exp(3.2 * self.config.rain_mu) * 0.01 ** (-self.config.rain_mu)
        )  # empirical relation adapted from Ulbrich (1983)
        _n0r: ta.wpfloat = _n0r * self.config.rain_n0  # apply tuning factor to rain_n0 variable
        _ar: ta.wpfloat = (
            math.pi * phy_const.water_density / 6.0 * _n0r * math.gamma(self.config.rain_mu + 4.0)
        )  # pre-factor

        power_law_exponent_for_rain_mean_fall_speed: ta.wpfloat = 0.5 / (self.config.rain_mu + 4.0)
        power_law_coeff_for_rain_mean_fall_speed: ta.wpfloat = (
            130.0
            * math.gamma(self.config.rain_mu + 4.5)
            / math.gamma(self.config.rain_mu + 4.0)
            * _ar ** (-power_law_exponent_for_rain_mean_fall_speed)
        )

        precomputed_evaporation_alpha_exp_coeff: ta.wpfloat = (self.config.rain_mu + 2.0) / (
            self.config.rain_mu + 4.0
        )
        precomputed_evaporation_alpha_coeff: ta.wpfloat = (
            2.0
            * math.pi
            * microphy_const.diffusion_coeff_for_water_vapor
            / microphy_const.howell_factor
            * _n0r
            * _ar ** (-precomputed_evaporation_alpha_exp_coeff)
            * math.gamma(self.config.rain_mu + 2.0)
        )
        precomputed_evaporation_beta_exp_coeff: ta.wpfloat = (2.0 * self.config.rain_mu + 5.5) / (
            2.0 * self.config.rain_mu + 8.0
        ) - precomputed_evaporation_alpha_exp_coeff
        precomputed_evaporation_beta_coeff: ta.wpfloat = (
            0.26
            * math.sqrt(
                microphy_const.ref_air_density * 130.0 / microphy_const.air_kinemetic_viscosity
            )
            * _ar ** (-precomputed_evaporation_beta_exp_coeff)
            * math.gamma((2.0 * self.config.rain_mu + 5.5) / 2.0)
            / math.gamma(self.config.rain_mu + 2.0)
        )

        # Precomputations for optimization
        power_law_exponent_for_rain_mean_fall_speed_ln1o2: ta.wpfloat = math.exp(
            power_law_exponent_for_rain_mean_fall_speed * math.log(0.5)
        )
        power_law_exponent_for_ice_mean_fall_speed_ln1o2: ta.wpfloat = math.exp(
            microphy_const.power_law_exponent_for_ice_mean_fall_speed * math.log(0.5)
        )
        power_law_exponent_for_graupel_mean_fall_speed_ln1o2: ta.wpfloat = math.exp(
            microphy_const.power_law_exponent_for_graupel_mean_fall_speed * math.log(0.5)
        )

        self._ice_collision_precomputed_coef = (
            precomputed_riming_coef,
            precomputed_agg_coef,
            precomputed_snow_sed_coef,
        )
        self._rain_precomputed_coef = (
            power_law_exponent_for_rain_mean_fall_speed,
            power_law_coeff_for_rain_mean_fall_speed,
            precomputed_evaporation_alpha_exp_coeff,
            precomputed_evaporation_alpha_coeff,
            precomputed_evaporation_beta_exp_coeff,
            precomputed_evaporation_beta_coeff,
        )
        self._sed_dens_factor_coef = (
            power_law_exponent_for_rain_mean_fall_speed_ln1o2,
            power_law_exponent_for_ice_mean_fall_speed_ln1o2,
            power_law_exponent_for_graupel_mean_fall_speed_ln1o2,
        )

    @property
    def ice_collision_precomputed_coef(self) -> tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat]:
        return self._ice_collision_precomputed_coef

    @property
    def rain_precomputed_coef(
        self,
    ) -> tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat, ta.wpfloat, ta.wpfloat, ta.wpfloat]:
        return self._rain_precomputed_coef

    @property
    def sed_dens_factor_coef(self) -> tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat]:
        return self._sed_dens_factor_coef

    def _initialize_local_fields(self):
        self.rhoqrv_old_kup = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        self.rhoqsv_old_kup = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        self.rhoqgv_old_kup = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        self.rhoqiv_old_kup = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        self.vnew_r = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        self.vnew_s = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        self.vnew_g = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        self.vnew_i = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        self.rain_precipitation_flux = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        self.snow_precipitation_flux = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        self.graupel_precipitation_flux = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        self.ice_precipitation_flux = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        self.total_precipitation_flux = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )

    def _determine_horizontal_domains(self):
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self._grid.start_index(cell_domain(h_grid.Zone.END))

    def _initialize_gt4py_programs(self):
        self._icon_graupel = model_options.setup_program(
            backend=self._backend,
            program=icon_graupel,
            constant_args={
                "ground_level": gtx.int32(self._grid.num_levels - 1),
                "liquid_autoconversion_option": self.config.liquid_autoconversion_option,
                "snow_intercept_option": self.config.snow_intercept_option,
                "use_constant_latent_heat": self.config.use_constant_latent_heat,
                "ice_stickeff_min": self.config.ice_stickeff_min,
                "snow2graupel_riming_coeff": self.config.snow2graupel_riming_coeff,
                "power_law_coeff_for_ice_mean_fall_speed": self.config.power_law_coeff_for_ice_mean_fall_speed,
                "exponent_for_density_factor_in_ice_sedimentation": self.config.exponent_for_density_factor_in_ice_sedimentation,
                "power_law_coeff_for_snow_fall_speed": self.config.power_law_coeff_for_snow_fall_speed,
                "precomputed_riming_coef": self._ice_collision_precomputed_coef[0],
                "precomputed_agg_coef": self._ice_collision_precomputed_coef[1],
                "precomputed_snow_sed_coef": self._ice_collision_precomputed_coef[2],
                "power_law_exponent_for_rain_mean_fall_speed": self._rain_precomputed_coef[0],
                "power_law_coeff_for_rain_mean_fall_speed": self._rain_precomputed_coef[1],
                "precomputed_evaporation_alpha_exp_coeff": self._rain_precomputed_coef[2],
                "precomputed_evaporation_alpha_coeff": self._rain_precomputed_coef[3],
                "precomputed_evaporation_beta_exp_coeff": self._rain_precomputed_coef[4],
                "precomputed_evaporation_beta_coeff": self._rain_precomputed_coef[5],
                "power_law_exponent_for_rain_mean_fall_speed_ln1o2": self._sed_dens_factor_coef[0],
                "power_law_exponent_for_ice_mean_fall_speed_ln1o2": self._sed_dens_factor_coef[1],
                "power_law_exponent_for_graupel_mean_fall_speed_ln1o2": self._sed_dens_factor_coef[
                    2
                ],
            },
            horizontal_sizes={
                "horizontal_start": self._start_cell_nudging,
                "horizontal_end": self._end_cell_local,
            },
            vertical_sizes={
                "vertical_start": self.vertical_params.kstart_moist,
                "vertical_end": self._grid.num_levels,
            },
        )

        self._icon_graupel_flux_above_ground = model_options.setup_program(
            backend=self._backend,
            program=icon_graupel_flux_above_ground,
            constant_args={
                "do_latent_heat_nudging": self.config.do_latent_heat_nudging,
            },
            horizontal_sizes={
                "horizontal_start": self._start_cell_nudging,
                "horizontal_end": self._end_cell_local,
            },
            vertical_sizes={
                "model_top": gtx.int32(0),
                "ground_level": gtx.int32(self._grid.num_levels - 1),
            },
        )

        self._icon_graupel_flux_at_ground = model_options.setup_program(
            backend=self._backend,
            program=icon_graupel_flux_at_ground,
            constant_args={
                "do_latent_heat_nudging": self.config.do_latent_heat_nudging,
            },
            horizontal_sizes={
                "horizontal_start": self._start_cell_nudging,
                "horizontal_end": self._end_cell_local,
            },
            vertical_sizes={
                "ground_level": gtx.int32(self._grid.num_levels - 1),
                "model_num_levels": self._grid.num_levels,
            },
        )

        # self._icon_graupel = icon_graupel.with_backend(self._backend).compile(
        #     enable_jit=False,
        #     ground_level=[gtx.int32(self._grid.num_levels - 1)],
        #     liquid_autoconversion_option=[self.config.liquid_autoconversion_option],
        #     snow_intercept_option=[self.config.snow_intercept_option],
        #     use_constant_latent_heat=[self.config.use_constant_latent_heat],
        #     ice_stickeff_min=[self.config.ice_stickeff_min],
        #     snow2graupel_riming_coeff=[self.config.snow2graupel_riming_coeff],
        #     power_law_coeff_for_ice_mean_fall_speed=[
        #         self.config.power_law_coeff_for_ice_mean_fall_speed
        #     ],
        #     exponent_for_density_factor_in_ice_sedimentation=[
        #         self.config.exponent_for_density_factor_in_ice_sedimentation
        #     ],
        #     power_law_coeff_for_snow_fall_speed=[self.config.power_law_coeff_for_snow_fall_speed],
        #     precomputed_riming_coef=[self._ice_collision_precomputed_coef[0]],
        #     precomputed_agg_coef=[self._ice_collision_precomputed_coef[1]],
        #     precomputed_snow_sed_coef=[self._ice_collision_precomputed_coef[2]],
        #     power_law_exponent_for_rain_mean_fall_speed=[self._rain_precomputed_coef[0]],
        #     power_law_coeff_for_rain_mean_fall_speed=[self._rain_precomputed_coef[1]],
        #     precomputed_evaporation_alpha_exp_coeff=[self._rain_precomputed_coef[2]],
        #     precomputed_evaporation_alpha_coeff=[self._rain_precomputed_coef[3]],
        #     precomputed_evaporation_beta_exp_coeff=[self._rain_precomputed_coef[4]],
        #     precomputed_evaporation_beta_coeff=[self._rain_precomputed_coef[5]],
        #     power_law_exponent_for_rain_mean_fall_speed_ln1o2=[self._sed_dens_factor_coef[0]],
        #     power_law_exponent_for_ice_mean_fall_speed_ln1o2=[self._sed_dens_factor_coef[1]],
        #     power_law_exponent_for_graupel_mean_fall_speed_ln1o2=[self._sed_dens_factor_coef[2]],
        #     vertical_start=[gtx.int32(self.vertical_params.kstart_moist)],
        #     vertical_end=[gtx.int32(self._grid.num_levels)],
        #     offset_provider={},
        # )

        # self._icon_graupel_flux_above_ground = icon_graupel_flux_above_ground.with_backend(
        #     self._backend
        # ).compile(
        #     enable_jit=False,
        #     do_latent_heat_nudging=[self.config.do_latent_heat_nudging],
        #     vertical_start=[gtx.int32(0)],
        #     vertical_end=[gtx.int32(self._grid.num_levels - 1)],
        #     offset_provider={},
        # )

        # self._icon_graupel_flux_at_ground = icon_graupel_flux_at_ground.with_backend(
        #     self._backend
        # ).compile(
        #     enable_jit=False,
        #     do_latent_heat_nudging=[self.config.do_latent_heat_nudging],
        #     vertical_start=[gtx.int32(self._grid.num_levels - 1)],
        #     vertical_end=[gtx.int32(self._grid.num_levels)],
        #     offset_provider={},
        # )

    def run(
        self,
        dtime: ta.wpfloat,
        rho: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
        temperature: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
        pressure: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
        qv: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
        qc: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
        qr: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
        qi: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
        qs: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
        qg: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
        qnc: gtx.Field[[dims.CellDim], ta.wpfloat],
        temperature_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
        qv_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
        qc_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
        qr_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
        qi_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
        qs_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
        qg_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    ):
        """
        Run the ICON single-moment graupel (nwp) microphysics. Precipitation flux is also computed.
        Args:
            dtime: microphysics time step [s]
            temperature: air temperature [K]
            pressure: air pressure [Pa]
            qv: specific humidity [kg/kg]
            qc: specific cloud content [kg/kg]
            qr: specific rain content [kg/kg]
            qi: specific ice content [kg/kg]
            qs: specific snow content [kg/kg]
            qg: specific graupel content [kg/kg]
            qnc: cloud number concentration [/m3]
        Returns:
            temperature_tendency: air temperature tendency [K/s]
            qv_tendency: specific humidity tendency [kg/kg/s]
            qc_tendency: specific cloud content tendency [kg/kg/s]
            qr_tendency: specific rain content tendency [kg/kg/s]
            qi_tendency: specific ice content tendency [kg/kg/s]
            qs_tendency: specific snow content tendency [kg/kg/s]
            qg_tendency: specific graupel content tendency [kg/kg/s]
        """
        self._icon_graupel(
            # gtx.int32(self._grid.num_levels - 1),
            # self.config.liquid_autoconversion_option,
            # self.config.snow_intercept_option,
            # self.config.use_constant_latent_heat,
            # self.config.ice_stickeff_min,
            # self.config.snow2graupel_riming_coeff,
            # self.config.power_law_coeff_for_ice_mean_fall_speed,
            # self.config.exponent_for_density_factor_in_ice_sedimentation,
            # self.config.power_law_coeff_for_snow_fall_speed,
            # *self._ice_collision_precomputed_coef,
            # *self._rain_precomputed_coef,
            # *self._sed_dens_factor_coef,
            dtime=dtime,
            dz=self.metric_state.ddqz_z_full,
            temperature=temperature,
            pressure=pressure,
            rho=rho,
            qv=qv,
            qc=qc,
            qi=qi,
            qr=qr,
            qs=qs,
            qg=qg,
            qnc=qnc,
            temperature_tendency=temperature_tendency,
            qv_tendency=qv_tendency,
            qc_tendency=qc_tendency,
            qi_tendency=qi_tendency,
            qr_tendency=qr_tendency,
            qs_tendency=qs_tendency,
            qg_tendency=qg_tendency,
            rhoqrv_old_kup=self.rhoqrv_old_kup,
            rhoqsv_old_kup=self.rhoqsv_old_kup,
            rhoqgv_old_kup=self.rhoqgv_old_kup,
            rhoqiv_old_kup=self.rhoqiv_old_kup,
            vnew_r=self.vnew_r,
            vnew_s=self.vnew_s,
            vnew_g=self.vnew_g,
            vnew_i=self.vnew_i,
            # horizontal_start=self._start_cell_nudging,
            # horizontal_end=self._end_cell_local,
            # vertical_start=self.vertical_params.kstart_moist,
            # vertical_end=self._grid.num_levels,
            # offset_provider={},
        )

        self._icon_graupel_flux_above_ground(
            # self.config.do_latent_heat_nudging,
            dtime=dtime,
            rho=rho,
            qr=qr,
            qs=qs,
            qg=qg,
            qi=qi,
            qr_tendency=qr_tendency,
            qs_tendency=qs_tendency,
            qg_tendency=qg_tendency,
            qi_tendency=qi_tendency,
            rhoqrv_old_kup=self.rhoqrv_old_kup,
            rhoqsv_old_kup=self.rhoqsv_old_kup,
            rhoqgv_old_kup=self.rhoqgv_old_kup,
            rhoqiv_old_kup=self.rhoqiv_old_kup,
            vnew_r=self.vnew_r,
            vnew_s=self.vnew_s,
            vnew_g=self.vnew_g,
            vnew_i=self.vnew_i,
            rain_precipitation_flux=self.rain_precipitation_flux,
            snow_precipitation_flux=self.snow_precipitation_flux,
            graupel_precipitation_flux=self.graupel_precipitation_flux,
            ice_precipitation_flux=self.ice_precipitation_flux,
            total_precipitation_flux=self.total_precipitation_flux,
            # horizontal_start=self._start_cell_nudging,
            # horizontal_end=self._end_cell_local,
            # vertical_start=gtx.int32(0),
            # vertical_end=gtx.int32(self._grid.num_levels - 1),
            # offset_provider={},
        )

        self._icon_graupel_flux_at_ground(
            # self.config.do_latent_heat_nudging,
            dtime=dtime,
            rho=rho,
            qr=qr,
            qs=qs,
            qg=qg,
            qi=qi,
            qr_tendency=qr_tendency,
            qs_tendency=qs_tendency,
            qg_tendency=qg_tendency,
            qi_tendency=qi_tendency,
            rhoqrv_old_kup=self.rhoqrv_old_kup,
            rhoqsv_old_kup=self.rhoqsv_old_kup,
            rhoqgv_old_kup=self.rhoqgv_old_kup,
            rhoqiv_old_kup=self.rhoqiv_old_kup,
            vnew_r=self.vnew_r,
            vnew_s=self.vnew_s,
            vnew_g=self.vnew_g,
            vnew_i=self.vnew_i,
            rain_precipitation_flux=self.rain_precipitation_flux,
            snow_precipitation_flux=self.snow_precipitation_flux,
            graupel_precipitation_flux=self.graupel_precipitation_flux,
            ice_precipitation_flux=self.ice_precipitation_flux,
            total_precipitation_flux=self.total_precipitation_flux,
            # horizontal_start=self._start_cell_nudging,
            # horizontal_end=self._end_cell_local,
            # vertical_start=gtx.int32(self._grid.num_levels - 1),
            # vertical_end=self._grid.num_levels,
            # offset_provider={},
        )


@gtx.field_operator
def _icon_graupel_flux_at_ground(
    do_latent_heat_nudging: bool,
    dtime: ta.wpfloat,
    rho: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qr: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qs: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qg: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qi: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qr_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qs_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qg_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qi_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqrv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqsv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqgv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqiv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_r: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_s: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_g: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_i: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
) -> tuple[
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
]:
    rain_flux = wpfloat("0.5") * ((qr + qr_tendency * dtime) * rho * vnew_r + rhoqrv_old_kup)
    snow_flux = wpfloat("0.5") * ((qs + qs_tendency * dtime) * rho * vnew_s + rhoqsv_old_kup)
    graupel_flux = wpfloat("0.5") * ((qg + qg_tendency * dtime) * rho * vnew_g + rhoqgv_old_kup)
    ice_flux = wpfloat("0.5") * ((qi + qi_tendency * dtime) * rho * vnew_i + rhoqiv_old_kup)
    zero = broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim))
    # for the latent heat nudging
    total_flux = rain_flux + snow_flux + graupel_flux if do_latent_heat_nudging else zero
    return rain_flux, snow_flux, graupel_flux, ice_flux, total_flux


@gtx.field_operator
def _icon_graupel_flux_above_ground(
    do_latent_heat_nudging: bool,
    dtime: ta.wpfloat,
    rho: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qr: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qs: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qg: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qi: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qr_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qs_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qg_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qi_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqrv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqsv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqgv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqiv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_r: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_s: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_g: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_i: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
) -> tuple[
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
]:
    zero = broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim))

    rain_flux_ = (qr + qr_tendency * dtime) * rho * vnew_r
    snow_flux_ = (qs + qs_tendency * dtime) * rho * vnew_s
    graupel_flux_ = (qg + qg_tendency * dtime) * rho * vnew_g
    ice_flux_ = (qi + qi_tendency * dtime) * rho * vnew_i

    rain_flux_new = where(rain_flux_ <= microphy_const.qmin, zero, rain_flux_)
    snow_flux_new = where(snow_flux_ <= microphy_const.qmin, zero, snow_flux_)
    graupel_flux_new = where(graupel_flux_ <= microphy_const.qmin, zero, graupel_flux_)
    ice_flux_new = where(ice_flux_ <= microphy_const.qmin, zero, ice_flux_)

    rain_flux = wpfloat("0.5") * (rain_flux_new + rhoqrv_old_kup)
    snow_flux = wpfloat("0.5") * (snow_flux_new + rhoqsv_old_kup)
    graupel_flux = wpfloat("0.5") * (graupel_flux_new + rhoqgv_old_kup)
    ice_flux = wpfloat("0.5") * (ice_flux_new + rhoqiv_old_kup)
    total_flux = rain_flux + snow_flux + graupel_flux + ice_flux if do_latent_heat_nudging else zero

    return rain_flux, snow_flux, graupel_flux, ice_flux, total_flux


@program(grid_type=gtx.GridType.UNSTRUCTURED)
def icon_graupel_flux_at_ground(
    do_latent_heat_nudging: bool,
    dtime: ta.wpfloat,
    rho: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qr: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qs: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qg: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qi: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qr_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qs_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qg_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qi_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqrv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqsv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqgv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqiv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_r: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_s: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_g: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_i: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rain_precipitation_flux: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    snow_precipitation_flux: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    graupel_precipitation_flux: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    ice_precipitation_flux: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    total_precipitation_flux: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    ground_level: gtx.int32,
    model_num_levels: gtx.int32,
):
    _icon_graupel_flux_at_ground(
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
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (ground_level, model_num_levels),
        },
    )


@program(grid_type=gtx.GridType.UNSTRUCTURED)
def icon_graupel_flux_above_ground(
    do_latent_heat_nudging: bool,
    dtime: ta.wpfloat,
    rho: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qr: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qs: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qg: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qi: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qr_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qs_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qg_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qi_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqrv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqsv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqgv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqiv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_r: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_s: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_g: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_i: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rain_precipitation_flux: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    snow_precipitation_flux: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    graupel_precipitation_flux: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    ice_precipitation_flux: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    total_precipitation_flux: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    model_top: gtx.int32,
    ground_level: gtx.int32,
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
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (model_top, ground_level),
        },
    )


@scan_operator(
    axis=dims.KDim,
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
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        gtx.int32,
    ],
    ground_level: gtx.int32,  # k bottom level
    liquid_autoconversion_option: gtx.int32,
    snow_intercept_option: gtx.int32,
    use_constant_latent_heat: bool,
    ice_stickeff_min: ta.wpfloat,
    snow2graupel_riming_coeff: ta.wpfloat,
    power_law_coeff_for_ice_mean_fall_speed: ta.wpfloat,
    exponent_for_density_factor_in_ice_sedimentation: ta.wpfloat,
    power_law_coeff_for_snow_fall_speed: ta.wpfloat,
    precomputed_riming_coef: ta.wpfloat,
    precomputed_agg_coef: ta.wpfloat,
    precomputed_snow_sed_coef: ta.wpfloat,
    power_law_exponent_for_rain_mean_fall_speed: ta.wpfloat,
    power_law_coeff_for_rain_mean_fall_speed: ta.wpfloat,
    precomputed_evaporation_alpha_exp_coeff: ta.wpfloat,
    precomputed_evaporation_alpha_coeff: ta.wpfloat,
    precomputed_evaporation_beta_exp_coeff: ta.wpfloat,
    precomputed_evaporation_beta_coeff: ta.wpfloat,
    power_law_exponent_for_rain_mean_fall_speed_ln1o2: ta.wpfloat,
    power_law_exponent_for_ice_mean_fall_speed_ln1o2: ta.wpfloat,
    power_law_exponent_for_graupel_mean_fall_speed_ln1o2: ta.wpfloat,
    dtime: ta.wpfloat,
    dz: ta.wpfloat,
    temperature: ta.wpfloat,
    pressure: ta.wpfloat,
    rho: ta.wpfloat,
    qv: ta.wpfloat,
    qc: ta.wpfloat,
    qi: ta.wpfloat,
    qr: ta.wpfloat,
    qs: ta.wpfloat,
    qg: ta.wpfloat,
    qnc: ta.wpfloat,
):
    """
    This is the ICON graupel scheme. The structure of the code can be split into several steps as follow:
        1. initialize tracer at k-1 level, and some pre-computed coefficients including the snow intercept parameter for later uses.
        2. compute sedimentation fluxes and update rain, snow, and graupel mass at the current k level.
        3. compute pre-computed coefficients after update from sedimentation fluxes to include implicitness of the graupel scheme.
        4. compute all transfer rates.
        5. check if tracers go below 0.
        6. update all tendencies.

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
        _,
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
    #  Section 1: Precomputed coefficients
    # ------------------------------------------------------------------------------

    qv_kup = qv_old_kup + qv_tendency_kup * dtime
    qc_kup = qc_old_kup + qc_tendency_kup * dtime
    qi_kup = qi_old_kup + qi_tendency_kup * dtime
    qr_kup = qr_old_kup + qr_tendency_kup * dtime
    qs_kup = qs_old_kup + qs_tendency_kup * dtime
    qg_kup = qg_old_kup + qg_tendency_kup * dtime

    is_surface = True if k_lev == ground_level else False

    # TODO (Chia Rui): duplicated function for computing latent heat. Saturation adjustment also uses the same function. Move to a common place.
    lhv = (
        phy_const.lh_vaporise
        if use_constant_latent_heat
        else phy_const.lh_vaporise
        + (microphy_const.cp_v - phy_const.cpl) * (temperature - phy_const.tmelt)
        - phy_const.rv * temperature
    )
    lhs = (
        phy_const.lh_sublimate
        if use_constant_latent_heat
        else phy_const.lh_sublimate
        + (microphy_const.cp_v - phy_const.cpi) * (temperature - phy_const.tmelt)
        - phy_const.rv * temperature
    )

    # for density correction of fall speeds
    chlp = log(microphy_const.ref_air_density / rho)
    crho1o2 = exp(chlp / wpfloat("2.0"))
    crhofac_qi = exp(chlp * exponent_for_density_factor_in_ice_sedimentation)

    cdtdh = wpfloat("0.5") * dtime / dz
    cscmax = qc / dtime
    cnin = _compute_cooper_inp_concentration(temperature)
    cmi = minimum(rho * qi / cnin, microphy_const.ice_max_mass)
    cmi = maximum(microphy_const.ice_initial_mass, cmi)

    qvsw = sat_pres_water(temperature) / (rho * phy_const.rv * temperature)
    qvsi = sat_pres_ice(temperature) / (rho * phy_const.rv * temperature)

    rhoqr = qr * rho
    rhoqs = qs * rho
    rhoqg = qg * rho
    rhoqi = qi * rho

    rhoqrv_new_kup = qr_kup * rho_kup * vnew_r
    rhoqsv_new_kup = qs_kup * rho_kup * vnew_s
    rhoqgv_new_kup = qg_kup * rho_kup * vnew_g
    rhoqiv_new_kup = qi_kup * rho_kup * vnew_i

    if rhoqrv_new_kup <= microphy_const.qmin:
        rhoqrv_new_kup = wpfloat("0.0")
    if rhoqsv_new_kup <= microphy_const.qmin:
        rhoqsv_new_kup = wpfloat("0.0")
    if rhoqgv_new_kup <= microphy_const.qmin:
        rhoqgv_new_kup = wpfloat("0.0")
    if rhoqiv_new_kup <= microphy_const.qmin:
        rhoqiv_new_kup = wpfloat("0.0")

    rhoqr_intermediate = rhoqr / cdtdh + rhoqrv_new_kup + rhoqrv_old_kup
    rhoqs_intermediate = rhoqs / cdtdh + rhoqsv_new_kup + rhoqsv_old_kup
    rhoqg_intermediate = rhoqg / cdtdh + rhoqgv_new_kup + rhoqgv_old_kup
    rhoqi_intermediate = rhoqi / cdtdh + rhoqiv_new_kup + rhoqiv_old_kup

    llqr = True if (rhoqr > microphy_const.qmin) else False
    llqs = True if (rhoqs > microphy_const.qmin) else False
    llqg = True if (rhoqg > microphy_const.qmin) else False
    llqi = True if (rhoqi > microphy_const.qmin) else False

    n0s, snow_sed0, crim, cagg, cbsdep = _compute_snow_interception_and_collision_parameters(
        temperature,
        rho,
        qs,
        precomputed_riming_coef,
        precomputed_agg_coef,
        precomputed_snow_sed_coef,
        power_law_coeff_for_snow_fall_speed,
        llqs,
        snow_intercept_option,
    )

    # ------------------------------------------------------------------------------
    #  Section 2: Sedimentation fluxes
    # ------------------------------------------------------------------------------

    if k_lev > 0:
        vnew_s = (
            snow_sed0_kup
            * exp(microphy_const.ccswxp * log((qs_kup + qs) * wpfloat("0.5") * rho_kup))
            * crho1o2_kup
            if qs_kup + qs > microphy_const.qmin
            else wpfloat("0.0")
        )
        vnew_r = (
            power_law_coeff_for_rain_mean_fall_speed
            * exp(
                power_law_exponent_for_rain_mean_fall_speed
                * log((qr_kup + qr) * wpfloat("0.5") * rho_kup)
            )
            * crho1o2_kup
            if qr_kup + qr > microphy_const.qmin
            else wpfloat("0.0")
        )
        vnew_g = (
            microphy_const.power_law_coeff_for_graupel_mean_fall_speed
            * exp(
                microphy_const.power_law_exponent_for_graupel_mean_fall_speed
                * log((qg_kup + qg) * wpfloat("0.5") * rho_kup)
            )
            * crho1o2_kup
            if qg_kup + qg > microphy_const.qmin
            else wpfloat("0.0")
        )
        vnew_i = (
            power_law_coeff_for_ice_mean_fall_speed
            * exp(
                microphy_const.power_law_exponent_for_ice_mean_fall_speed
                * log((qi_kup + qi) * wpfloat("0.5") * rho_kup)
            )
            * crhofac_qi_kup
            if qi_kup + qi > microphy_const.qmin
            else wpfloat("0.0")
        )

    if llqs:
        terminal_velocity = snow_sed0 * exp(microphy_const.ccswxp * log(rhoqs)) * crho1o2
        # Prevent terminal fall speed of snow from being zero at the surface level
        if is_surface:
            terminal_velocity = maximum(terminal_velocity, microphy_const.minimum_snow_fall_speed)

        rhoqsv = rhoqs * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if vnew_s == wpfloat("0.0"):
            vnew_s = terminal_velocity * microphy_const.ccswxp_ln1o2

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
            terminal_velocity = maximum(terminal_velocity, microphy_const.minimum_rain_fall_speed)

        rhoqrv = rhoqr * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if vnew_r == wpfloat("0.0"):
            vnew_r = terminal_velocity * power_law_exponent_for_rain_mean_fall_speed_ln1o2

    else:
        rhoqrv = wpfloat("0.0")

    if llqg:
        terminal_velocity = (
            microphy_const.power_law_coeff_for_graupel_mean_fall_speed
            * exp(microphy_const.power_law_exponent_for_graupel_mean_fall_speed * log(rhoqg))
            * crho1o2
        )
        # Prevent terminal fall speed of snow from being zero at the surface level
        if is_surface:
            terminal_velocity = maximum(
                terminal_velocity, microphy_const.minimum_graupel_fall_speed
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
            * exp(microphy_const.power_law_exponent_for_ice_mean_fall_speed * log(rhoqi))
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
        vnew_s = maximum(vnew_s, microphy_const.minimum_snow_fall_speed)
        vnew_r = maximum(vnew_r, microphy_const.minimum_rain_fall_speed)
        vnew_g = maximum(vnew_g, microphy_const.minimum_graupel_fall_speed)

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
    #  Section 3: Precomputed coefficients after sedimentation for implicitness
    # ------------------------------------------------------------------------------

    llqr = True if (rhoqr > microphy_const.qmin) else False
    llqs = True if (rhoqs > microphy_const.qmin) else False
    llqg = True if (rhoqg > microphy_const.qmin) else False
    llqi = True if (qi > microphy_const.qmin) else False
    llqc = True if (qc > microphy_const.qmin) else False

    if llqr:
        clnrhoqr = log(rhoqr)
        csrmax = (
            rhoqr_intermediate / rho / dtime
        )  # GZ: shifting this computation ahead of the IF condition changes results!
        if qi + qc > microphy_const.qmin:
            celn7o8qrk = exp(wpfloat("7.0") / wpfloat("8.0") * clnrhoqr)
        else:
            celn7o8qrk = wpfloat("0.0")
        if temperature < microphy_const.threshold_freeze_temperature:
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
            rhoqs_intermediate / rho / dtime
        )  # GZ: shifting this computation ahead of the IF condition changes results#
        if qi + qc > microphy_const.qmin:
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
        csgmax = rhoqg_intermediate / rho / dtime
        if qi + qc > microphy_const.qmin:
            celnrimexp_g = exp(microphy_const.graupel_rimexp * clnrhoqg)
        else:
            celnrimexp_g = wpfloat("0.0")
        celn6qgk = exp(wpfloat("0.6") * clnrhoqg)
    else:
        csgmax = wpfloat("0.0")
        celnrimexp_g = wpfloat("0.0")
        celn6qgk = wpfloat("0.0")

    if llqi | llqs:
        cdvtp = microphy_const.ccdvtp * exp(wpfloat("1.94") * log(temperature)) / pressure
        chi = microphy_const.ccshi1 * cdvtp * rho * qvsi / (temperature * temperature)
        chlp = cdvtp / (wpfloat("1.0") + chi)
        cidep = microphy_const.ccidep * chlp

        if llqs:
            cslam = exp(microphy_const.ccslxp * log(microphy_const.ccslam * n0s / rhoqs))
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
    #  Section 4: Transfer rates
    # ------------------------------------------------------------------------------

    ice_nucleation_rate_v2i = _deposition_nucleation_at_low_temperature_or_in_clouds(
        temperature, rho, qv, qi, qvsi, cnin, dtime, llqc
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
    ) = _riming_in_clouds(
        temperature,
        qc,
        crim,
        cslam,
        celnrimexp_g,
        celn3o4qsk,
        snow2graupel_riming_coeff,
        llqc,
        llqs,
    )

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
        dtime,
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
        pressure,
        qv,
        qs,
        qvsi,
        dtime,
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
        pressure,
        rho,
        qv,
        qvsw,
        rhoqi_intermediate,
        dtime,
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
        dtime,
        rain_freezing_rate_r2g_in_clouds,
        csrmax,
        precomputed_evaporation_alpha_exp_coeff,
        precomputed_evaporation_alpha_coeff,
        precomputed_evaporation_beta_exp_coeff,
        precomputed_evaporation_beta_coeff,
        celn7o4qrk,
        llqr,
    )

    # ------------------------------------------------------------------------------
    #  Section 5: Check for negative mass
    # ------------------------------------------------------------------------------

    snow_deposition_rate_v2s = (
        snow_deposition_rate_v2s_in_cold_clouds + snow_deposition_rate_v2s_in_melting_condition
    )
    graupel_deposition_rate_v2g = (
        graupel_deposition_rate_v2g_in_cold_clouds
        + graupel_deposition_rate_v2g_in_melting_condition
    )

    # finalizing transfer rates in clouds and calculate depositional growth reduction
    if llqc & (temperature > microphy_const.homogeneous_freeze_temperature):
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
        if temperature <= phy_const.tmelt:  # cold case
            qvsidiff = qv - qvsi
            csimax = rhoqi_intermediate / rho / dtime

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
    #  Section 6: Update tendencies
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

    # l_cv (is_isochoric) is removed in icon4py. So, heat_cap_r (reciprocal of heat capacity of dry air) = microphy_const.rcvd (at constant volume)
    temperature_tendency = microphy_const.rcvd * (lhv * (cqct + cqrt) + lhs * (cqit + cqst + cqgt))
    qi_tendency = maximum((rhoqi_intermediate / rho * cimi - qi) / dtime + cqit * cimi, -qi / dtime)
    qr_tendency = maximum((rhoqr_intermediate / rho * cimr - qr) / dtime + cqrt * cimr, -qr / dtime)
    qs_tendency = maximum((rhoqs_intermediate / rho * cims - qs) / dtime + cqst * cims, -qs / dtime)
    qg_tendency = maximum((rhoqg_intermediate / rho * cimg - qg) / dtime + cqgt * cimg, -qg / dtime)
    qc_tendency = maximum(cqct, -qc / dtime)
    qv_tendency = maximum(cqvt, -qv / dtime)

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
    ground_level: gtx.int32,
    liquid_autoconversion_option: gtx.int32,
    snow_intercept_option: gtx.int32,
    use_constant_latent_heat: bool,
    ice_stickeff_min: ta.wpfloat,
    snow2graupel_riming_coeff: ta.wpfloat,
    power_law_coeff_for_ice_mean_fall_speed: ta.wpfloat,
    exponent_for_density_factor_in_ice_sedimentation: ta.wpfloat,
    power_law_coeff_for_snow_fall_speed: ta.wpfloat,
    precomputed_riming_coef: ta.wpfloat,
    precomputed_agg_coef: ta.wpfloat,
    precomputed_snow_sed_coef: ta.wpfloat,
    power_law_exponent_for_rain_mean_fall_speed: ta.wpfloat,
    power_law_coeff_for_rain_mean_fall_speed: ta.wpfloat,
    precomputed_evaporation_alpha_exp_coeff: ta.wpfloat,
    precomputed_evaporation_alpha_coeff: ta.wpfloat,
    precomputed_evaporation_beta_exp_coeff: ta.wpfloat,
    precomputed_evaporation_beta_coeff: ta.wpfloat,
    power_law_exponent_for_rain_mean_fall_speed_ln1o2: ta.wpfloat,
    power_law_exponent_for_ice_mean_fall_speed_ln1o2: ta.wpfloat,
    power_law_exponent_for_graupel_mean_fall_speed_ln1o2: ta.wpfloat,
    dtime: ta.wpfloat,
    dz: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    temperature: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    pressure: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rho: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qv: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qc: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qi: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qr: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qs: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qg: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qnc: gtx.Field[[dims.CellDim], ta.wpfloat],
) -> tuple[
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
]:
    (
        temperature_tendency,
        qv_tendency,
        qc_tendency,
        qi_tendency,
        qr_tendency,
        qs_tendency,
        qg_tendency,
        _,
        _,
        _,
        _,
        _,
        _,
        rhoqrv_old_kup,
        rhoqsv_old_kup,
        rhoqgv_old_kup,
        rhoqiv_old_kup,
        vnew_r,
        vnew_s,
        vnew_g,
        vnew_i,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _icon_graupel_scan(
        ground_level,
        liquid_autoconversion_option,
        snow_intercept_option,
        use_constant_latent_heat,
        ice_stickeff_min,
        snow2graupel_riming_coeff,
        power_law_coeff_for_ice_mean_fall_speed,
        exponent_for_density_factor_in_ice_sedimentation,
        power_law_coeff_for_snow_fall_speed,
        precomputed_riming_coef,
        precomputed_agg_coef,
        precomputed_snow_sed_coef,
        power_law_exponent_for_rain_mean_fall_speed,
        power_law_coeff_for_rain_mean_fall_speed,
        precomputed_evaporation_alpha_exp_coeff,
        precomputed_evaporation_alpha_coeff,
        precomputed_evaporation_beta_exp_coeff,
        precomputed_evaporation_beta_coeff,
        power_law_exponent_for_rain_mean_fall_speed_ln1o2,
        power_law_exponent_for_ice_mean_fall_speed_ln1o2,
        power_law_exponent_for_graupel_mean_fall_speed_ln1o2,
        dtime,
        dz,
        temperature,
        pressure,
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
    ground_level: gtx.int32,
    liquid_autoconversion_option: gtx.int32,
    snow_intercept_option: gtx.int32,
    use_constant_latent_heat: bool,
    ice_stickeff_min: ta.wpfloat,
    snow2graupel_riming_coeff: ta.wpfloat,
    power_law_coeff_for_ice_mean_fall_speed: ta.wpfloat,
    exponent_for_density_factor_in_ice_sedimentation: ta.wpfloat,
    power_law_coeff_for_snow_fall_speed: ta.wpfloat,
    precomputed_riming_coef: ta.wpfloat,
    precomputed_agg_coef: ta.wpfloat,
    precomputed_snow_sed_coef: ta.wpfloat,
    power_law_exponent_for_rain_mean_fall_speed: ta.wpfloat,
    power_law_coeff_for_rain_mean_fall_speed: ta.wpfloat,
    precomputed_evaporation_alpha_exp_coeff: ta.wpfloat,
    precomputed_evaporation_alpha_coeff: ta.wpfloat,
    precomputed_evaporation_beta_exp_coeff: ta.wpfloat,
    precomputed_evaporation_beta_coeff: ta.wpfloat,
    power_law_exponent_for_rain_mean_fall_speed_ln1o2: ta.wpfloat,
    power_law_exponent_for_ice_mean_fall_speed_ln1o2: ta.wpfloat,
    power_law_exponent_for_graupel_mean_fall_speed_ln1o2: ta.wpfloat,
    dtime: ta.wpfloat,
    dz: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    temperature: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    pressure: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rho: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qv: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qc: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qi: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qr: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qs: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qg: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qnc: gtx.Field[[dims.CellDim], ta.wpfloat],
    temperature_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qv_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qc_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qi_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qr_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qs_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    qg_tendency: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqrv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqsv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqgv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    rhoqiv_old_kup: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_r: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_s: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_g: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    vnew_i: gtx.Field[[dims.CellDim, dims.KDim], ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _icon_graupel(
        ground_level,
        liquid_autoconversion_option,
        snow_intercept_option,
        use_constant_latent_heat,
        ice_stickeff_min,
        snow2graupel_riming_coeff,
        power_law_coeff_for_ice_mean_fall_speed,
        exponent_for_density_factor_in_ice_sedimentation,
        power_law_coeff_for_snow_fall_speed,
        precomputed_riming_coef,
        precomputed_agg_coef,
        precomputed_snow_sed_coef,
        power_law_exponent_for_rain_mean_fall_speed,
        power_law_coeff_for_rain_mean_fall_speed,
        precomputed_evaporation_alpha_exp_coeff,
        precomputed_evaporation_alpha_coeff,
        precomputed_evaporation_beta_exp_coeff,
        precomputed_evaporation_beta_coeff,
        power_law_exponent_for_rain_mean_fall_speed_ln1o2,
        power_law_exponent_for_ice_mean_fall_speed_ln1o2,
        power_law_exponent_for_graupel_mean_fall_speed_ln1o2,
        dtime,
        dz,
        temperature,
        pressure,
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
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
