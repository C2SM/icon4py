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
"""Prognostic one-moment bulk microphysical parameterization.

"""
import sys
from typing import Final, Optional
import numpy as np
import dataclasses

import gt4py.next as gtx
from gt4py.eve.utils import FrozenNamespace
from gt4py.next.ffront.decorator import program, field_operator, scan_operator
from gt4py.next.ffront.fbuiltins import (
    exp,
    tanh,
    log,
    maximum,
    minimum,
    sqrt
)
#from gt4py.next.iterator.embedded import np_as_located_field
#from gt4py.next.program_processors.runners import roundtrip

from icon4py.model.common.math.math_utilities import gamma_fct
from icon4py.model.common.math.math_constants import math_const
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, VertexDim
from icon4py.model.common.grid.horizontal import (
    CellParams,
    EdgeParams,
    HorizontalMarkerIndex,
)
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.states import prognostic_state as prognostics, diagnostic_state as diagnostics, tracer_state as tracers
from icon4py.model.atmosphere.dycore.state_utils.utils import _allocate
from icon4py.model.common.type_alias import vpfloat, wpfloat


sys.setrecursionlimit(350000)


@dataclasses.dataclass(frozen=True)
class SingleMomentSixClassIconGraupelConfig:
    """
    Contains necessary parameter to configure icon graupel microphysics scheme.

    Encapsulates namelist parameters.
    Values should be read from configuration.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    lpres_pri is removed because it is a duplicated parameter with do_ice_sedimentation.
    ldiag_ttend, and ldiag_qtend are removed because default outputs in icon4py physics granules include tendencies.
    """
    # TODO (Chia RUi): Remove unwanted options
    #: liquid auto conversion mode. 1: Kessler (1969), 2: Seifert & Beheng (2006). Originally defined as iautocon in gscp_data.f90 in ICON.
    liquid_autoconversion_option: gtx.int32 = gtx.int32(1)
    #: Option for deriving snow size distribution interception parameter.
    snow_intercept_option: gtx.int32 = gtx.int32(2)
    #: Rain freezing mode. Originally defined as super_coolw (a bool) in ICON.
    rain_freezing_option: gtx.int32 = gtx.int32(1)
    #: Number concentration of ice particles. 1: Cooper (1986) formula, see eq. 5.165. 2: Hobbs and Rangno (1985) and Meyers et al. (1992). see eq. 5.89.
    ice_concentration_option: gtx.int32 = gtx.int32(1)
    #: Do ice particle sedimentation (Heymsfield & Donner 1990 *1/3). Originally defined as lsedi_ice in ICON.
    do_ice_sedimentation: bool = True
    #: Options for different sticking efficiency in ice-ice auto conversion (work from Guenther Zaengl). Originall defined as lstickeff in ICON.
    ice_autocon_sticking_efficiency_option: gtx.int32 = gtx.int32(1)
    #: Whether ice depositional growth is reduced near water cloud top.
    do_reduced_icedeposition: bool = True
    #: Determine whether the microphysical processes are isochoric or isobaric. Originally defined as l_cv in ICON.
    is_isochoric: bool = True
    #: Do latent heat nudging. Originally defined as dass_lhn in ICON.
    do_latent_heat_nudging = False
    #: Whether a fixed latent heat capacities are used for water. Originally defined as ithermo_water in ICON (0 means True).
    use_constant_water_heat_capacity = True
    #: First parameter in RHS of eq. 5.163 for the sticking efficiency when ice_autocon_sticking_efficiency = 1. Default seems to be 0.075.
    ice_stickeff_min: wpfloat = 0.01
    #: Constant in v-qi ice terminal velocity-mixing ratio relationship, see eq. 5.169.
    ice_v0: wpfloat = 1.25
    #: Exponent of the density factor in ice terminal velocity equation to account for density (air thermodynamic state) change. Default seems to be 0.33.
    ice_sedi_density_factor_exp: wpfloat = 0.3
    #: Constant in v-D snow terminal velocity-Diameter relationship, see eqs. 5.57 (depricated after COSMO 3.0) and unnumbered eq. (v = 25 D^0.5) below eq. 5.159. default seems to be 20.0.
    snow_v0: wpfloat = 20.0
    #: mu exponential factor in gamma distribution of rain particles.
    rain_mu: wpfloat = 0.0
    #: Interception parameter in gamma distribution of rain particles.
    rain_n0: wpfloat = 1.0


#@dataclasses.dataclass(frozen=True)
#class SingleMomentSixClassIconGraupelParams:
class SingleMomentSixClassIconGraupelParams(FrozenNamespace):
    """
    Contains numerical, physical, and empirical constants for the ICON graupel scheme.

    These constants are not configurable from namelists in ICON.
    If users want to tune the model for better results in specific cases, you may need to change the hard coded constants here.
    All the equations can be found in A Description of the Nonhydrostatic Regional COSMO-Model Part II Physical Parameterizations.
    """

    # config: dataclasses.InitVar[SingleMomentSixClassIconGraupelConfig]
    #: threshold temperature for heterogeneous freezing of raindrops
    threshold_freeze_temperature: wpfloat = 271.15
    #: coefficient for raindrop freezing, see eq. 5.126
    coeff_rain_freeze_mode2: wpfloat = 1.68
    #: FR: 1. coefficient for immersion raindrop freezing: alpha_if, see eq. 5.168
    coeff_rain_freeze1_mode1: wpfloat = 9.95e-5
    #: FR: 2. coefficient for immersion raindrop freezing: a_if, see eq. 5.168
    coeff_rain_freeze2_mode1: wpfloat = 0.66
    #: temperature for hom. freezing of cloud water
    homogeneous_freeze_temperature: wpfloat = 236.15
    #: threshold temperature for mixed-phase cloud freezing of cloud drops (Forbes 2012, Forbes & Ahlgrimm 2014), see eq. 5.166.
    threshold_freeze_temperature_mixedphase: wpfloat = 250.15
    #: threshold for lowest detectable mixing ratios
    qmin: wpfloat = 1.0e-15
    #: a small number for cloud existence criterion
    eps: wpfloat = 1.0e-15
    #: exponential factor in ice terminal velocity equation v = zvz0i*rhoqi^zbvi, see eq. 5.169
    ice_exp_v: wpfloat = 0.16
    #: reference air density
    ref_air_density: wpfloat = 1.225e+0
    #: in m/s; minimum terminal fall velocity of rain particles (applied only near the ground)
    rain_v_sedi_min: wpfloat = 0.7
    #: in m/s; minimum terminal fall velocity of snow particles (applied only near the ground)
    snow_v_sedi_min: wpfloat = 0.1
    #: in m/s; minimum terminal fall velocity of graupel particles (applied only near the ground)
    graupel_v_sedi_min: wpfloat = 0.4
    #: maximal number of ice crystals, see eq. 5.165.
    nimax_Thom: wpfloat = 250.0e+3
    # TODO (Chia Rui): What are these two parameters for? Why they are not used but exist in ICON
    # zams_ci= 0.069           # Formfactor in the mass-size relation of snow particles for cloud ice scheme
    # zams_gr= 0.069           # Formfactor in the mass-size relation of snow particles for graupel scheme
    #: Formfactor in the mass-diameter relation of snow particles, see eq. 5.159.
    snow_m0: wpfloat = 0.069
    #: A constant intercept parameter for inverse exponential size distribution of snow particles, see eq. 5.160.
    snow_n0: wpfloat = 8.0e5
    #: exponent of mixing ratio in the collection equation where cloud or ice particles are rimed by graupel (exp=(3+b)/(1+beta), v=a D^b, m=alpha D^beta), see eqs. 5.152 to 5.154.
    graupel_exp_rim: wpfloat = 0.94878
    #: exponent of mixing ratio in the graupel mean terminal velocity-mixing ratio relationship (exp=b/(1+beta)), see eq. 5.156.
    graupel_exp_sed: wpfloat = 0.217
    #: constant in the graupel mean terminal velocity-mixing ratio relationship, see eq. 5.156.
    graupel_sed0: wpfloat = 12.24
    #: initial crystal mass for cloud ice nucleation, see eq. 5.101
    ice_initial_mass: wpfloat = 1.0e-12
    #: maximum mass of cloud ice crystals to avoid too large ice crystals near melting point, see eq. 5.105
    ice_max_mass: wpfloat = 1.0e-9
    #: initial mass of snow crystals which is used in ice-ice autoconversion to snow particles, see eq. 5.108
    snow_min_mass: wpfloat = 3.0e-9
    #: Scaling factor [1/K] for temperature-dependent cloud ice sticking efficiency (only used when ice_autocon_sticking_efficiency_option=1), see eq. 5.163
    stick_eff_fac: wpfloat = 3.5e-3
    #: Temperature at which cloud ice autoconversion starts (only used when ice_autocon_sticking_efficiency_option=1), see eq. 5.163
    tmin_iceautoconv: wpfloat = 188.15
    #: Reference length for distance from cloud top (Forbes 2012) (only used when do_reduced_icedeposition=True), see eq. 5.166
    dist_cldtop_ref: wpfloat = 500.0
    #: lower bound on snow/ice deposition reduction (only used when do_reduced_icedeposition=True), see eq. 5.166
    reduce_dep_ref: wpfloat = 0.1
    #: Howell factor in depositional growth equation, see eq. 5.71 and eqs. 5.103 & 5.104 (for ice? TODO (Chia Rui): check)
    howell_factor: wpfloat = 2.270603
    #: Collection efficiency for snow collecting cloud water, see eq. 5.113
    snow_cloud_collection_eff: wpfloat = 0.9
    #: Exponent in the terminal velocity for snow, see unnumbered eq. (v = 25 D^0.5) below eq. 5.159
    snow_exp_v: wpfloat = 0.50
    #: kinematic viscosity of air
    eta: wpfloat = 1.75e-5
    #: molecular diffusion coefficient for water vapour
    diffusion_coeff_water_vapor: wpfloat = 2.22e-5
    #: thermal conductivity of dry air
    dry_air_latent_heat: wpfloat = 2.40e-2
    #: Exponent in the mass-diameter relation of snow particles, see eq. 5.159
    snow_exp_m: wpfloat = 2.000
    #: Formfactor in the mass-diameter relation of cloud ice, see eq. 5.90
    ice_m0: wpfloat = 130.0
    #: density of liquid water
    water_density: wpfloat = 1.000e+3
    #: specific heat of water vapor J, at constant pressure (Landolt-Bornstein)
    cp_v: wpfloat = 1850.0
    #: specific heat of ice
    ci: wpfloat = 2108.0

    snow_n0s1: wpfloat = 13.5 * 5.65e5  # parameter in N0S(T)
    snow_n0s2: wpfloat = -0.107  # parameter in N0S(T), Field et al
    snow_mma: tuple[wpfloat,wpfloat,wpfloat,wpfloat,wpfloat,wpfloat,wpfloat,wpfloat,wpfloat,wpfloat] = (
        5.065339, -0.062659, -3.032362, 0.029469, -0.000285, 0.312550, 0.000204, 0.003199, 0.000000, -0.015952
    )
    snow_mmb: tuple[wpfloat,wpfloat,wpfloat,wpfloat,wpfloat,wpfloat,wpfloat,wpfloat,wpfloat,wpfloat] = (
        0.476221, -0.015896, 0.165977, 0.007468, -0.000141, 0.060366, 0.000079, 0.000594, 0.000000, -0.003577
    )

    #: temperature for het. nuc. of cloud ice
    heterogeneous_freeze_temperature: wpfloat = 248.15
    #: autoconversion coefficient (cloud water to rain)
    kessler_cloud2rain_autoconversion_coeff_for_cloud: wpfloat = 4.0e-4
    #: (15/32)*(PI**0.5)*(ECR/RHOW)*V0R*AR**(1/8) when Kessler (1969) is used for cloud-cloud autoconversion
    kessler_cloud2rain_autoconversion_coeff_for_rain: wpfloat = 1.72
    #: constant in phi-function for Seifert-Beheng (2001) autoconversion
    kphi1: wpfloat = 6.00e+02
    #: exponent in phi-function for Seifert-Beheng (2001) autoconversion
    kphi2: wpfloat = 0.68e+00
    #: exponent in phi-function for Seifert-Beheng (2001) accretion
    kphi3: wpfloat = 5.00e-05
    #: kernel coeff for Seifert-Beheng (2001) autoconversion
    kcau: wpfloat = 9.44e+09
    #: kernel coeff for Seifert-Beheng (2001) accretion
    kcac: wpfloat = 5.25e+00
    #: gamma exponent for cloud distribution in Seifert-Beheng (2001) autoconverssion
    cnue: wpfloat = 2.00e+00
    #: separating mass between cloud and rain in Seifert-Beheng (2001) autoconverssion
    xstar: wpfloat = 2.60e-10

    #: = b1
    c1es: wpfloat = 610.78
    #: = b2w
    c3les: wpfloat = 17.269
    #: = b2i
    c3ies: wpfloat = 21.875
    #: = b4w
    c4les: wpfloat = 35.86
    #: = b4i
    c4ies: wpfloat = 7.66

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

    qc0: wpfloat = 0.0
    qi0: wpfloat = 0.0

    # TODO (Chia Rui): move parameters below to common/constants? But they need to be declared under FrozenNameSpace to be used in the big graupel scan operator.
    #: Latent heat of vaporisation for water [J/kg]
    alv: wpfloat = 2.5008e6
    #: Latent heat of sublimation for water [J/kg]
    als: wpfloat = 2.8345e6
    #: Melting temperature of ice/snow [K]
    tmelt: wpfloat = 273.15
    #: Triple point of water at 611hPa [K]
    t3: wpfloat = 273.16

    #: Gas constant of dry air [J/K/kg]
    rd: wpfloat = 287.04
    #: Specific heat of dry air at constant pressure [J/K/kg]
    cpd: wpfloat = 1004.64
    #: cp_d / cp_l - 1
    rcpl: wpfloat = 3.1733

    #: Gas constant of water vapor [J/K/kg] """
    rv: wpfloat = 461.51
    #: Specific heat of water vapour at constant pressure [J/K/kg]
    cpv: wpfloat = 1869.46
    #: Specific heat of water vapour at constant volume [J/K/kg]
    cvv: wpfloat = cpv - rv

    ccsdep: wpfloat = 0.26 * gamma_fct((snow_exp_v + 5.0) / 2.0) * np.sqrt(1.0 / eta)
    _ccsvxp: wpfloat = -(snow_exp_v / (snow_exp_m + 1.0) + 1.0)
    ccsvxp: wpfloat = _ccsvxp + 1.0
    ccslam: wpfloat = snow_m0 * gamma_fct(snow_exp_m + 1.0)
    ccslxp: wpfloat = 1.0 / (snow_exp_m + 1.0)
    ccswxp: wpfloat = snow_exp_v * ccslxp
    ccsaxp: wpfloat = -(snow_exp_v + 3.0)
    ccsdxp: wpfloat = -(snow_exp_v + 1.0) / 2.0
    ccshi1: wpfloat = als * als / (dry_air_latent_heat * rv)
    ccdvtp: wpfloat = 2.22e-5 * tmelt ** (-1.94) * 101325.0
    ccidep: wpfloat = 4.0 * ice_m0 ** (-1.0 / 3.0)
    # log_10 = np.log(10.0)  # logarithm of 10
    ccswxp_ln1o2: wpfloat = np.exp(ccswxp * np.log(0.5))

    #: latent heat of fusion for water [J/kg]
    alf: wpfloat = als - alv
    #: Specific heat capacity of liquid water
    clw: wpfloat = (rcpl + 1.0) * cpd

    #: Specific heat of dry air at constant volume [J/K/kg]
    cvd: wpfloat = cpd - rd
    #: [K*kg/J]
    rcpd: wpfloat = 1.0 / cpd
    #: [K*kg/J]"""
    rcvd: wpfloat = 1.0 / cvd

    c2es: wpfloat = c1es * rd / rv
    #: = b234w
    c5les: wpfloat = c3les * (tmelt - c4les)
    #: = b234i
    c5ies: wpfloat = c3ies * (tmelt - c4ies)
    c5alvcp: wpfloat = c5les * alv / cpd
    c5alscp: wpfloat = c5ies * als / cpd
    alvdcp: wpfloat = alv / cpd
    alsdcp: wpfloat = als / cpd

    pvsw0: wpfloat = c1es * np.exp(c3les * (tmelt - tmelt) / (tmelt - c4les)) # _fpvsw(tmelt)  # sat. vap. pressure for t = t0

#_icon_graupel_params = SingleMomentSixClassIconGraupelParams()
#icon_graupel_params: Final = FrozenNamespace(**vars(_icon_graupel_params))
icon_graupel_params: Final = SingleMomentSixClassIconGraupelParams()

# Statement functions
# -------------------

def _fpvsw(
    ztx: wpfloat
) -> wpfloat:
    # Return saturation vapour pressure over water from temperature
    return icon_graupel_params.c1es * np.exp(
        icon_graupel_params.c3les * (ztx - icon_graupel_params.tmelt) / (ztx - icon_graupel_params.c4les)
    )


@dataclasses.dataclass
class MetricStateIconGraupel:
    ddqz_z_full: gtx.Field[[CellDim, KDim], wpfloat]


class SingleMomentSixClassIconGraupel:

    def __init__(
        self,
        config: SingleMomentSixClassIconGraupelConfig,
        grid: Optional[icon_grid.IconGrid],
        metric_state: Optional[MetricStateIconGraupel],
        vertical_params: Optional[v_grid.VerticalGridParams]
    ):
        self.config = config
        self._validate_and_initialize_configurable_parameters()
        self.grid = grid
        self.metric_state = metric_state
        self.vertical_params = vertical_params

        self._initialize_local_fields()

    def _validate_and_initialize_configurable_parameters(self):
        if self.config.liquid_autoconversion_option != gtx.int32(1) and self.config.liquid_autoconversion_option != gtx.int32(2):
            raise NotImplementedError("liquid_autoconversion_option can only be 1 or 2.")
        if self.config.snow_intercept_option != gtx.int32(1) and self.config.snow_intercept_option != gtx.int32(2):
            raise NotImplementedError("snow_intercept_option can only be 1 or 2.")
        if self.config.rain_freezing_option != gtx.int32(1) and self.config.rain_freezing_option != gtx.int32(2):
            raise NotImplementedError("rain_freezing_option can only be 1 or 2.")
        if self.config.ice_autocon_sticking_efficiency_option != gtx.int32(1) and self.config.ice_autocon_sticking_efficiency_option != gtx.int32(2):
            raise NotImplementedError("ice_autocon_sticking_efficiency_option can only be 1 or 2.")
        # TODO (Chia Rui): clean up the naming system of these parameters and categorize them in a tuple
        ccsrim: wpfloat = 0.25 * math_const.pi * icon_graupel_params.snow_cloud_collection_eff * self.config.snow_v0 * gamma_fct(
            icon_graupel_params.snow_exp_v + 3.0)
        ccsagg: wpfloat = 0.25 * math_const.pi * self.config.snow_v0 * gamma_fct(icon_graupel_params.snow_exp_v + 3.0)
        _ccsvxp = -(icon_graupel_params.snow_exp_v / (icon_graupel_params.snow_exp_m + 1.0) + 1.0)
        ccsvel: wpfloat = icon_graupel_params.snow_m0 * self.config.snow_v0 * gamma_fct(
            icon_graupel_params.snow_exp_m + icon_graupel_params.snow_exp_v + 1.0) * (
                     icon_graupel_params.snow_m0 * gamma_fct(icon_graupel_params.snow_exp_m + 1.0)) ** _ccsvxp
        if self.config.ice_concentration_option == gtx.int32(1):
            nimax: wpfloat = icon_graupel_params.nimax_Thom
            nimix: wpfloat = self._fxna_cooper(icon_graupel_params.threshold_freeze_temperature_mixedphase)
        elif self.config.ice_concentration_option == gtx.int32(2):
            nimax: wpfloat = self._fxna(icon_graupel_params.homogeneous_freeze_temperature)
            nimix: wpfloat = self._fxna(icon_graupel_params.threshold_freeze_temperature_mixedphase)
        else:
            raise NotImplementedError("ice_concentration_option can only be 1 or 2.")

        _n0r: wpfloat = 8.0e6 * np.exp(3.2 * self.config.rain_mu) * 0.01 ** (
            -self.config.rain_mu)  # empirical relation adapted from Ulbrich (1983)
        _n0r: wpfloat = _n0r * self.config.rain_n0  # apply tuning factor to zn0r variable
        _ar: wpfloat = math_const.pi * icon_graupel_params.water_density / 6.0 * _n0r * gamma_fct(
            self.config.rain_mu + 4.0)  # pre-factor

        rain_exp_v: wpfloat = wpfloat(0.5) / (self.config.rain_mu + wpfloat(4.0))
        rain_v0: wpfloat = wpfloat(130.0) * gamma_fct(self.config.rain_mu + 4.5) / gamma_fct(self.config.rain_mu + 4.0) * _ar ** (
            -rain_exp_v)

        cevxp: wpfloat = (self.config.rain_mu + wpfloat(2.0)) / (self.config.rain_mu + 4.0)
        cev: wpfloat = wpfloat(2.0) * math_const.pi * icon_graupel_params.diffusion_coeff_water_vapor / icon_graupel_params.howell_factor * _n0r * _ar ** (
            -cevxp) * gamma_fct(self.config.rain_mu + 2.0)
        bevxp: wpfloat = (wpfloat(2.0) * self.config.rain_mu + wpfloat(5.5)) / (2.0 * self.config.rain_mu + wpfloat(8.0)) - cevxp
        bev: wpfloat = 0.26 * np.sqrt(icon_graupel_params.ref_air_density * 130.0 / icon_graupel_params.eta) * _ar ** (
            -bevxp) * gamma_fct((2.0 * self.config.rain_mu + 5.5) / 2.0) / gamma_fct(self.config.rain_mu + 2.0)

        # Precomputations for optimization
        # vzxp_ln1o2 = np.exp(vzxp * np.log(0.5))
        # bvi_ln1o2 = np.exp(icon_graupel_params.ice_exp_v * np.log(0.5))
        # expsedg_ln1o2 = np.exp(icon_graupel_params.graupel_exp_sed * np.log(0.5))
        rain_exp_v_ln1o2: wpfloat = np.exp(rain_exp_v * np.log(0.5))
        ice_exp_v_ln1o2: wpfloat = np.exp(icon_graupel_params.ice_exp_v * np.log(0.5))
        graupel_exp_sed_ln1o2: wpfloat = np.exp(icon_graupel_params.graupel_exp_sed * np.log(0.5))

        self._ccs = (ccsrim, ccsagg, ccsvel)
        self._nimax = nimax
        self._nimix = nimix
        self._rain_vel_coef = (rain_exp_v, rain_v0, cevxp, cev, bevxp, bev)
        self._sed_dens_factor_coef = (rain_exp_v_ln1o2, ice_exp_v_ln1o2, graupel_exp_sed_ln1o2)

    @property
    def ccs(self) -> tuple[wpfloat, wpfloat, wpfloat]:
        return self._ccs

    @property
    def nimax(self) -> wpfloat:
        return self._nimax

    @property
    def nimix(self) -> wpfloat:
        return self._nimax

    @property
    def rain_vel_coef(self) -> tuple[wpfloat, wpfloat, wpfloat, wpfloat, wpfloat, wpfloat]:
        return self._rain_vel_coef

    @property
    def sed_dens_factor_coef(self) -> tuple[wpfloat, wpfloat, wpfloat]:
        return self._sed_dens_factor_coef

    def _initialize_local_fields(self):
        self.qnc = _allocate(CellDim, KDim, grid=self.grid)
        # TODO (Chia Rui): remove these tendency terms when physics inteface infrastructure is ready
        self.temperature_tendency = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.qv_tendency = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.qc_tendency = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.qi_tendency = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.qr_tendency = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.qs_tendency = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.qg_tendency = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.rhoqrv_old_kup = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.rhoqsv_old_kup = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.rhoqgv_old_kup = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.rhoqiv_old_kup = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.vnew_r = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.vnew_s = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.vnew_g = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.vnew_i = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.rain_precipitation_flux = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.snow_precipitation_flux = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.graupel_precipitation_flux = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.ice_precipitation_flux = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.total_precipitation_flux = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)

    @classmethod
    def _fxna(
        self,
        ztx: wpfloat
    ) -> wpfloat:
        # Return number of activate ice crystals from temperature
        return 1.0e2 * np.exp(0.2 * (icon_graupel_params.tmelt - ztx))

    @classmethod
    def _fxna_cooper(
        self,
        ztx: wpfloat
    ) -> wpfloat:
        # Return number of activate ice crystals from temperature

        # Method: Cooper (1986) used by Greg Thompson(2008)

        return 5.0 * np.exp(0.304 * (icon_graupel_params.tmelt - ztx))

    def __str__(self):
        # TODO (Chia Rui): Print out the configuration and derived empirical parameters
        pass

    def run(
        self,
        dtime: wpfloat,
        prognostic_state: prognostics.PrognosticState,
        diagnostic_state: diagnostics.DiagnosticState,
        tracer_state: tracers.TracerState
    ):

        start_cell_nudging = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.nudging(CellDim)
        )
        end_cell_local = self.grid.get_end_index(CellDim, HorizontalMarkerIndex.local(CellDim))

        icon_graupel(
            self.vertical_params.kstart_moist,
            self.config.liquid_autoconversion_option,
            self.config.snow_intercept_option,
            self.config.rain_freezing_option,
            self.config.ice_concentration_option,
            self.config.do_ice_sedimentation,
            self.config.ice_autocon_sticking_efficiency_option,
            self.config.do_reduced_icedeposition,
            self.config.is_isochoric,
            self.config.use_constant_water_heat_capacity,
            self.config.ice_stickeff_min,
            self.config.ice_v0,
            self.config.ice_sedi_density_factor_exp,
            self.config.snow_v0,
            *self._ccs,
            self._nimax,
            self._nimix,
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
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        icon_graupel_flux(
            self.vertical_params.kstart_moist,
            prognostic_state.rho,
            tracer_state.qr,
            tracer_state.qs,
            tracer_state.qg,
            tracer_state.qi,
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
            self.config.do_ice_sedimentation,
            self.config.do_latent_heat_nudging,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )


@scan_operator(
    axis=KDim,
    forward=True,
    init=(
        vpfloat("0.0"),  # temperature
        vpfloat("0.0"),  # qv
        vpfloat("0.0"),  # qc
        vpfloat("0.0"),  # qi
        vpfloat("0.0"),  # qr
        vpfloat("0.0"),  # qs
        vpfloat("0.0"),  # qg
        vpfloat("0.0"),  # rhoqrv
        vpfloat("0.0"),  # rhoqsv
        vpfloat("0.0"),  # rhoqgv
        vpfloat("0.0"),  # rhoqiv
        vpfloat("0.0"),  # newv_r
        vpfloat("0.0"),  # newv_s
        vpfloat("0.0"),  # newv_g
        vpfloat("0.0"),  # newv_i
        vpfloat("0.0"),  # cloud top distance
        vpfloat("0.0"),  # density
        vpfloat("0.0"),  # density factor
        vpfloat("0.0"),  # density factor for ice
        vpfloat("0.0"),  # snow intercept parameter
        vpfloat("0.0"),  # saturation pressure
        gtx.int32(0)     # k level
    )
)
def _icon_graupel_scan(
    state_kup: tuple[
        vpfloat,
        vpfloat,
        vpfloat,
        vpfloat,
        vpfloat,
        vpfloat,
        vpfloat,
        vpfloat,
        vpfloat,
        vpfloat,
        vpfloat,
        vpfloat,
        vpfloat,
        vpfloat,
        vpfloat,
        vpfloat,
        vpfloat,
        vpfloat,
        vpfloat,
        vpfloat,
        vpfloat,
        gtx.int32
    ],
    # Grid information
    kstart_moist: gtx.int32, # k starting level
    kend: gtx.int32,     # k bottom level
    liquid_autoconversion_option: gtx.int32,
    snow_intercept_option: gtx.int32,
    rain_freezing_option: gtx.int32,
    ice_concentration_option: gtx.int32,
    do_ice_sedimentation: bool,
    ice_autocon_sticking_efficiency_option: gtx.int32,
    do_reduced_icedeposition: bool,
    is_isochoric: bool,
    use_constant_water_heat_capacity: bool,
    ice_stickeff_min: wpfloat,
    ice_v0: wpfloat,
    ice_sedi_density_factor_exp: wpfloat,
    snow_v0: wpfloat,
    ccsrim: wpfloat,
    ccsagg: wpfloat,
    ccsvel: wpfloat,
    nimax: wpfloat,
    nimix: wpfloat,
    rain_exp_v: wpfloat,
    rain_v0: wpfloat,
    cevxp: wpfloat,
    cev: wpfloat,
    bevxp: wpfloat,
    bev: wpfloat,
    rain_exp_v_ln1o2: wpfloat,
    ice_exp_v_ln1o2: wpfloat,
    graupel_exp_sed_ln1o2: wpfloat,
    dt: wpfloat,     # time step
    dz: wpfloat,     # level thickness
    # Prognostics
    temperature: vpfloat,
    pres: vpfloat,
    rho: vpfloat,
    qv: vpfloat,
    qc: vpfloat,
    qi: vpfloat,
    qr: vpfloat,
    qs: vpfloat,
    qg: vpfloat,
    # Number Densities
    qnc: vpfloat,  # originally 2D Field, now 3D Field
):
    # TODO (Chia Rui): The graupel scheme here is a plain translation from Fortran code. Clean-up and refactor with functional calls when slow compilation is resolved.

    # unpack carry
    # temporary variables for storing variables at k-1 (upper) grid
    (
        temperature_kup,
        qv_kup,
        qc_kup,
        qi_kup,
        qr_kup,
        qs_kup,
        qg_kup,
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
        k_lev
    ) = state_kup

    # ------------------------------------------------------------------------------
    #  Section 1: Initial setting of physical constants
    # ------------------------------------------------------------------------------

    if k_lev < kstart_moist:
        # tracing current k level
        k_lev = k_lev + 1
        return (
            temperature,
            qv,
            qc,
            qi,
            qr,
            qs,
            qg,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            rho,
            0.0,
            0.0,
            0.0,
            0.0,
            k_lev
        )

    is_surface = True if k_lev == kend else False

    # Define reciprocal of heat capacity of dry air (at constant pressure vs at constant volume)
    heat_cap_r = icon_graupel_params.rcvd if is_isochoric else icon_graupel_params.rcpd

    # timestep for calculations
    dtr = 1.0 / dt

    # Latent heats
    # Default themodynamic is constant latent heat
    # tg = make_normalized(t(iv,k))
    # Calculate Latent heats if necessary
    # function called: CLHv = latent_heat_vaporization(temperature)
    # function called: CLHs = latent_heat_sublimation(temperature)
    lhv = icon_graupel_params.alv if use_constant_water_heat_capacity else icon_graupel_params.alv + (icon_graupel_params.cp_v - icon_graupel_params.clw) * (temperature - icon_graupel_params.tmelt) - icon_graupel_params.rv * temperature
    lhs = icon_graupel_params.als if use_constant_water_heat_capacity else icon_graupel_params.als + (icon_graupel_params.cp_v - icon_graupel_params.ci) * (temperature - icon_graupel_params.tmelt) - icon_graupel_params.rv * temperature

    #----------------------------------------------------------------------------
    # Section 2: Check for existence of rain and snow
    #            Initialize microphysics and sedimentation scheme
    #----------------------------------------------------------------------------

    # initialization of source terms (all set to zero)
    szdep_v2i = 0.0 # vapor   -> ice,     ice vapor deposition
    szsub_v2i = 0.0 # vapor   -> ice,     ice vapor sublimation
    sidep_v2i = 0.0 # vapor   -> ice,     ice vapor net deposition
    #ssdpc_v2s = 0.0 # vapor   -> snow,    snow vapor deposition below freezing temperature
    #sgdpc_v2g = 0.0 # vapor   -> graupel, graupel vapor deposition below freezing temperature
    #ssdph_v2s = 0.0 # vapor   -> snow,    snow vapor deposition above freezing temperature
    #sgdph_v2g = 0.0 # vapor   -> graupel, graupel vapor deposition above freezing temperature
    ssdep_v2s = 0.0 # vapor   -> snow,    snow vapor deposition
    sgdep_v2g = 0.0 # vapor   -> graupel, graupel vapor deposition
    #sdnuc_v2i = 0.0 # vapor   -> ice,     low temperature heterogeneous ice deposition nucleation
    #scnuc_v2i = 0.0 # vapor   -> ice,     incloud ice nucleation
    snucl_v2i = 0.0 # vapor   -> ice,     ice nucleation
    sconr_v2r = 0.0 # vapor   -> rain,    rain condensation on melting snow/graupel
    scaut_c2r = 0.0 # cloud   -> rain,    cloud autoconversion into rain
    scfrz_c2i = 0.0 # cloud   -> ice,     cloud freezing
    scacr_c2r = 0.0 # cloud   -> rain,    rain-cloud accretion
    sshed_c2r = 0.0 # cloud   -> rain,    rain shedding from riming above freezing
    srims_c2s = 0.0 # cloud   -> snow,    snow riming
    srimg_c2g = 0.0 # cloud   -> graupel, graupel riming
    simlt_i2c = 0.0 # ice     -> cloud,   ice melting
    sicri_i2g = 0.0 # ice     -> graupel, ice loss in rain-ice accretion
    sdaut_i2s = 0.0 # ice     -> snow,    ice vapor depositional autoconversion into snow
    saggs_i2s = 0.0 # ice     -> snow,    snow-ice aggregation
    saggg_i2g = 0.0 # ice     -> graupel, graupel-ice aggregation
    siaut_i2s = 0.0 # ice     -> snow,    ice autoconversion into snow
    srcri_r2g = 0.0 # rain    -> graupel, rain loss in rain-ice accretion
    #scrfr_r2g = 0.0 # rain    -> graupel, rain freezing in clouds
    #ssrfr_r2g = 0.0 # rain    -> graupel, rain freezing in clear sky
    srfrz_r2g = 0.0 # rain    -> graupel, rain freezing
    sevap_r2v = 0.0 # rain    -> vapor,   rain evaporation
    ssmlt_s2r = 0.0 # snow    -> rain,    snow melting
    scosg_s2g = 0.0 # snow    -> graupel, snow autoconversion into graupel
    sgmlt_g2r = 0.0 # graupel -> rain,    graupel melting

    reduce_dep = 1.0 # FR: Reduction coeff. for dep. growth of rain and ice

    #----------------------------------------------------------------------------
    # 2.1: Preparations for computations and to check the different conditions
    #----------------------------------------------------------------------------

    #..for density correction of fall speeds
    c1orho = 1.0 / rho
    chlp = log(icon_graupel_params.ref_air_density * c1orho)
    crho1o2 = exp(chlp / 2.0)
    crhofac_qi = exp(chlp * ice_sedi_density_factor_exp)

    # NOT WORKING, NOT SURE WHY THIS WILL LEAD TO DIVISION BY ZERO
    # if rho_kup == 0.0:
    # crho1o2_kup = 1.0
    # crhofac_qi_kup = 1.0
    # else:
    #    c1orho_kup     = 1.0 / rho_kup
    #    chlp_kup       = log(icon_graupel_params.ref_air_density * c1orho_kup)
    #    crho1o2_kup    = exp(Chlp_kup / 2.0)
    #    crhofac_qi_kup = exp(Chlp_kup * ice_sedi_density_factor_exp)

    rhoqr = qr * rho
    rhoqs = qs * rho
    rhoqg = qg * rho
    rhoqi = qi * rho

    llqr = True if (rhoqr > icon_graupel_params.qmin) else False
    llqs = True if (rhoqs > icon_graupel_params.qmin) else False
    llqg = True if (rhoqg > icon_graupel_params.qmin) else False
    llqi = True if (rhoqi > icon_graupel_params.qmin) else False

    cdtdh = 0.5 * dt / dz

    #-------------------------------------------------------------------------
    # qs_prepare:
    #-------------------------------------------------------------------------
    if llqs:
        # function called: Cn0s = snow_intercept(qs,temperature,rho)
        #------------------------------------------------------------------------------
        # Description:
        #   This subroutine computes the intercept parameter, N0, of the snow exponential size distribution.
        #
        #   First method: Explained in paragraphs at pages 2008 and 2009 in Field et al. (2005). N0s_23 = (M_2)^4 / (M_3)^3, M_2 = Gamma(3) N0s / lamda^3, M_2 = Gamma(4) N0s / lamda^4, so N0s_23 = 2/27 N0s. And N0s_23 = 5.65E5 exp(-0.107Tc)
        #
        #   Second method: Eq. 5.160 in the documentation, originally in Table 2 in Field et al. (2005).
        #
        #------------------------------------------------------------------------------

        if snow_intercept_option == 1:
            # Calculate n0s using the temperature-dependent
            # formula of Field et al. (2005)
            local_tc = temperature - icon_graupel_params.tmelt
            local_tc = minimum( local_tc , 0.0 )
            local_tc = maximum( local_tc , -40.0 )
            n0s = icon_graupel_params.snow_n0s1 * exp(icon_graupel_params.snow_n0s2 * local_tc)
            n0s = minimum( n0s , 1.0e9 )
            n0s = maximum( n0s , 1.0e6 )

        elif snow_intercept_option == 2:
            # Calculate n0s using the temperature-dependent moment
            # relations of Field et al. (2005)
            local_tc = temperature - icon_graupel_params.tmelt
            local_tc = minimum( local_tc , 0.0 )
            local_tc = maximum( local_tc , -40.0 )

            local_nnr = 3.0
            local_hlp = (
                icon_graupel_params.snow_mma[0] +
                icon_graupel_params.snow_mma[1]*local_tc +
                icon_graupel_params.snow_mma[2]*local_nnr +
                icon_graupel_params.snow_mma[3]*local_tc*local_nnr +
                icon_graupel_params.snow_mma[4]*local_tc**2.0 +
                icon_graupel_params.snow_mma[5]*local_nnr**2.0 +
                icon_graupel_params.snow_mma[6]*local_tc**2.0*local_nnr +
                icon_graupel_params.snow_mma[7]*local_tc*local_nnr**2.0 +
                icon_graupel_params.snow_mma[8]*local_tc**3.0 +
                icon_graupel_params.snow_mma[9]*local_nnr**3.0
            )
            local_alf = exp(local_hlp * log(10.0))
            local_bet = (
                icon_graupel_params.snow_mmb[0] +
                icon_graupel_params.snow_mmb[1]*local_tc +
                icon_graupel_params.snow_mmb[2]*local_nnr +
                icon_graupel_params.snow_mmb[3]*local_tc*local_nnr +
                icon_graupel_params.snow_mmb[4]*local_tc**2.0 +
                icon_graupel_params.snow_mmb[5]*local_nnr**2.0 +
                icon_graupel_params.snow_mmb[6]*local_tc**2.0*local_nnr +
                icon_graupel_params.snow_mmb[7]*local_tc*local_nnr**2.0 +
                icon_graupel_params.snow_mmb[8]*local_tc**3.0 +
                icon_graupel_params.snow_mmb[9]*local_nnr**3.0
            )

            # Here is the exponent bms=2.0 hardwired# not ideal# (Uli Blahak)
            local_m2s = qs * rho / icon_graupel_params.snow_m0  # UB rho added as bugfix
            local_m3s = local_alf * exp(local_bet * log(local_m2s))

            local_hlp = icon_graupel_params.snow_n0s1 * exp(icon_graupel_params.snow_n0s2 * local_tc)
            n0s = 13.50 * local_m2s * (local_m2s / local_m3s) ** 3.0
            n0s = maximum(n0s, 0.5 * local_hlp)
            n0s = minimum(n0s, 1.0e2 * local_hlp)
            n0s = minimum(n0s, 1.0e9)
            n0s = maximum(n0s, 1.0e6)

        else:
            n0s = icon_graupel_params.snow_n0

        # compute integration factor for terminal velocity
        #vz0s = ccsvel * exp(icon_graupel_params.ccsvxp * log(n0s))
        snow_sed0 = ccsvel * exp(icon_graupel_params.ccsvxp * log(n0s))
        # compute constants for riming, aggregation, and deposition processes for snow
        crim = ccsrim * n0s
        cagg = ccsagg * n0s
        cbsdep = icon_graupel_params.ccsdep * sqrt(snow_v0)
    else:
        n0s = icon_graupel_params.snow_n0
        snow_sed0 = 0.0
        crim = 0.0
        cagg = 0.0
        cbsdep = 0.0


    #----------------------------------------------------------------------------
    # 2.2: sedimentation fluxes
    #----------------------------------------------------------------------------

    rhoqrv_new_kup = qr_kup * rho_kup * vnew_r
    rhoqsv_new_kup = qs_kup * rho_kup * vnew_s
    rhoqgv_new_kup = qg_kup * rho_kup * vnew_g
    rhoqiv_new_kup = qi_kup * rho_kup * vnew_i
    if rhoqrv_new_kup <= icon_graupel_params.qmin: rhoqrv_new_kup = 0.0
    if rhoqsv_new_kup <= icon_graupel_params.qmin: rhoqsv_new_kup = 0.0
    if rhoqgv_new_kup <= icon_graupel_params.qmin: rhoqgv_new_kup = 0.0
    if rhoqiv_new_kup <= icon_graupel_params.qmin: rhoqiv_new_kup = 0.0

    rhoqr_intermediate = rhoqr / cdtdh + rhoqrv_new_kup + rhoqrv_old_kup
    rhoqs_intermediate = rhoqs / cdtdh + rhoqsv_new_kup + rhoqsv_old_kup
    rhoqg_intermediate = rhoqg / cdtdh + rhoqgv_new_kup + rhoqgv_old_kup
    rhoqi_intermediate = rhoqi / cdtdh + rhoqiv_new_kup + rhoqiv_old_kup

    #-------------------------------------------------------------------------
    # qs_sedi, qr_sedi, qg_sedi, qi_sedi:
    #-------------------------------------------------------------------------
    if k_lev > kstart_moist:
        vnew_s = snow_sed0_kup * exp(icon_graupel_params.ccswxp * log((qs_kup + qs) * 0.5 * rho_kup)) * crho1o2_kup if qs_kup + qs > icon_graupel_params.qmin else 0.0
        vnew_r = rain_v0 * exp(rain_exp_v * log((qr_kup + qr) * 0.5 * rho_kup)) * crho1o2_kup if qr_kup + qr > icon_graupel_params.qmin else 0.0
        vnew_g = icon_graupel_params.graupel_sed0 * exp(icon_graupel_params.graupel_exp_sed * log((qg_kup + qg) * 0.5 * rho_kup)) * crho1o2_kup if qg_kup + qg > icon_graupel_params.qmin else 0.0
        vnew_i = ice_v0 * exp(icon_graupel_params.ice_exp_v * log((qi_kup + qi) * 0.5 * rho_kup)) * crhofac_qi_kup if qi_kup + qi > icon_graupel_params.qmin else 0.0

    if llqs:
        terminal_velocity = snow_sed0 * exp(icon_graupel_params.ccswxp * log(rhoqs)) * crho1o2
        # Prevent terminal fall speed of snow from being zero at the surface level
        if is_surface: terminal_velocity = maximum(terminal_velocity, icon_graupel_params.snow_v_sedi_min)

        rhoqsv = rhoqs * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if vnew_s == 0.0: vnew_s = terminal_velocity * icon_graupel_params.ccswxp_ln1o2

    else:
        rhoqsv = 0.0

    if llqr:
        terminal_velocity = rain_v0 * exp(rain_exp_v * log(rhoqr)) * crho1o2
        # Prevent terminal fall speed of snow from being zero at the surface level
        if is_surface: terminal_velocity = maximum(terminal_velocity, icon_graupel_params.rain_v_sedi_min)

        rhoqrv = rhoqr * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if vnew_r == 0.0: vnew_r = terminal_velocity * rain_exp_v_ln1o2

    else:
        rhoqrv = 0.0

    if llqg:
        terminal_velocity = icon_graupel_params.graupel_sed0 * exp(icon_graupel_params.graupel_exp_sed * log(rhoqg)) * crho1o2
        # Prevent terminal fall speed of snow from being zero at the surface level
        if is_surface: terminal_velocity = maximum(terminal_velocity, icon_graupel_params.graupel_v_sedi_min)

        rhoqgv = rhoqg * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if vnew_g == 0.0: vnew_g = terminal_velocity * graupel_exp_sed_ln1o2

    else:
        rhoqgv = 0.0

    if llqi:
        terminal_velocity = ice_v0 * exp(icon_graupel_params.ice_exp_v * log(rhoqi)) * crhofac_qi

        rhoqiv = rhoqi * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if vnew_i == 0.0: vnew_i = terminal_velocity * ice_exp_v_ln1o2

    else:
        rhoqiv = 0.0

    # Prevent terminal fall speeds of precip hydrometeors from being zero at the surface level
    if is_surface:
        vnew_s = maximum(vnew_s, icon_graupel_params.snow_v_sedi_min)
        vnew_r = maximum( vnew_r, icon_graupel_params.rain_v_sedi_min )
        vnew_g = maximum(vnew_g, icon_graupel_params.graupel_v_sedi_min)

    #-------------------------------------------------------------------------
    # derive the intermediate density of hydrometeors, Eq. 5.21:
    #-------------------------------------------------------------------------


    # limit the precipitation flux at this k level such that mixing ratio won't go below zero
    rhoqrv = minimum( rhoqrv , rhoqr_intermediate )
    rhoqsv = minimum( rhoqsv , rhoqs_intermediate )
    rhoqgv = minimum( rhoqgv , maximum(0.0 , rhoqg_intermediate) )
    rhoqiv = minimum( rhoqiv , rhoqi_intermediate )

    rhoqr_intermediate = cdtdh * (rhoqr_intermediate - rhoqrv)
    rhoqs_intermediate = cdtdh * (rhoqs_intermediate - rhoqsv)
    rhoqg_intermediate = cdtdh * (rhoqg_intermediate - rhoqgv)
    rhoqi_intermediate = cdtdh * (rhoqi_intermediate - rhoqiv)

    cimr = 1.0 / (1.0 + vnew_r * cdtdh)
    cims = 1.0 / (1.0 + vnew_s * cdtdh)
    cimg = 1.0 / (1.0 + vnew_g * cdtdh)
    cimi = 1.0 / (1.0 + vnew_i * cdtdh)

    # intermediate values
    rhoqr = rhoqr_intermediate * cimr
    rhoqs = rhoqs_intermediate * cims
    rhoqg = rhoqg_intermediate * cimg
    rhoqi = rhoqi_intermediate * cimi


    #--------------------------------------------------------------------------
    # 2.3: Second part of preparations
    #--------------------------------------------------------------------------

    #FR old
    #   Csdep    = 3.2E-2
    #Csdep        = 3.367e-2
    #Cidep        = 1.3e-5
    #Cslam        = 1.0e10

    cscmax = qc * dtr
    # TODO (Chia Rui): can we create an interface for the computation of ice number concentration for different methods to remove if statement here?
    if ice_concentration_option == 1:
        # function called: cnin = _fxna_cooper(temperature)
        cnin = 5.0 * exp(0.304 * (icon_graupel_params.tmelt - temperature))
        cnin = minimum( cnin , nimax )
    elif ice_concentration_option == 2:
        # function called: cnin = _fxna(temperature)
        cnin = 1.0e2 * exp(0.2 * (icon_graupel_params.tmelt - temperature))
        cnin = minimum( cnin , nimax )
    else:
        cnin = 1.0
    cmi = minimum( rho * qi / cnin , icon_graupel_params.ice_max_mass )
    cmi = maximum( icon_graupel_params.ice_initial_mass , cmi )

    # function called: qvsw = sat_pres_water(temperature) / (rho * icon_graupel_params.rv * temperature)
    # function called: qvsi = sat_pres_ice(temperature) / (rho * icon_graupel_params.rv * temperature)
    qvsw = icon_graupel_params.c1es * exp( icon_graupel_params.c3les * (temperature - icon_graupel_params.tmelt) / (temperature - icon_graupel_params.c4les) ) / (rho * icon_graupel_params.rv * temperature)
    qvsi = icon_graupel_params.c1es * exp( icon_graupel_params.c3ies * (temperature - icon_graupel_params.tmelt) / (temperature - icon_graupel_params.c4ies) ) / (rho * icon_graupel_params.rv * temperature)
    llqr = True if (rhoqr > icon_graupel_params.qmin) else False
    llqs = True if (rhoqs > icon_graupel_params.qmin) else False
    llqg = True if (rhoqg > icon_graupel_params.qmin) else False
    llqi = True if (qi > icon_graupel_params.qmin) else False
    llqc = True if (qc > icon_graupel_params.qmin) else False


    ##----------------------------------------------------------------------------
    ## 2.4: IF (llqr): ic1
    ##----------------------------------------------------------------------------

    if llqr:
        clnrhoqr = log(rhoqr)
        csrmax   = rhoqr_intermediate / rho * dtr  # GZ: shifting this computation ahead of the IF condition changes results!
        if ( qi + qc > icon_graupel_params.qmin ):
            celn7o8qrk   = exp(7.0/8.0 * clnrhoqr)
        else:
            celn7o8qrk = 0.0
        if temperature < icon_graupel_params.threshold_freeze_temperature:
            celn7o4qrk   = exp(7.0/4.0 * clnrhoqr) #FR new
            celn27o16qrk = exp(27.0/16.0 * clnrhoqr)
        else:
            celn7o4qrk = 0.0
            celn27o16qrk = 0.0
        if llqi:
            celn13o8qrk  = exp(13.0/8.0 * clnrhoqr)
        else:
            celn13o8qrk = 0.0
    else:
        csrmax = 0.0
        celn7o8qrk = 0.0
        celn7o4qrk = 0.0
        celn27o16qrk = 0.0
        celn13o8qrk = 0.0

    ##----------------------------------------------------------------------------
    ## 2.5: IF (llqs): ic2
    ##----------------------------------------------------------------------------

    # ** GZ: the following computation differs substantially from the corresponding code in cloudice **
    if llqs:
        clnrhoqs = log(rhoqs)
        cssmax = rhoqs_intermediate / rho * dtr  # GZ: shifting this computation ahead of the IF condition changes results#
        if qi + qc > icon_graupel_params.qmin:
            celn3o4qsk = exp(3.0/4.0 * clnrhoqs)
        else:
            celn3o4qsk = 0.0
        celn8qsk = exp(0.8 * clnrhoqs)
    else:
        cssmax = 0.0
        celn3o4qsk = 0.0
        celn8qsk = 0.0

    ##----------------------------------------------------------------------------
    ## 2.6: IF (llqg): ic3
    ##----------------------------------------------------------------------------

    if llqg:
        clnrhoqg = log(rhoqg)
        csgmax = rhoqg_intermediate / rho * dtr
        if qi + qc > icon_graupel_params.qmin:
            celnrimexp_g = exp(icon_graupel_params.graupel_exp_rim * clnrhoqg)
        else:
            celnrimexp_g = 0.0
        celn6qgk = exp(0.6 * clnrhoqg)
    else:
        csgmax = 0.0
        celnrimexp_g = 0.0
        celn6qgk = 0.0

        ##----------------------------------------------------------------------------
    ## 2.7:  slope of snow PSD and coefficients for depositional growth (llqi,llqs)
    ##----------------------------------------------------------------------------

    if llqi | llqs:
        cdvtp = icon_graupel_params.ccdvtp * exp(1.94 * log(temperature)) / pres
        chi = icon_graupel_params.ccshi1 * cdvtp * rho * qvsi/(temperature * temperature)
        chlp = cdvtp / (1.0 + chi)
        cidep = icon_graupel_params.ccidep * chlp

        if llqs:
            cslam = exp(icon_graupel_params.ccslxp * log(icon_graupel_params.ccslam * n0s / rhoqs ))
            cslam = minimum( cslam , 1.0e15 )
            csdep = 4.0 * n0s * chlp
        else:
            cslam = 1.0e10
            csdep = 3.367e-2
    else:
        cidep = 1.3e-5
        cslam = 1.0e10
        csdep = 3.367e-2


    ##----------------------------------------------------------------------------
    ## 2.8: Deposition nucleation for low temperatures below a threshold (llqv)
    ##----------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    # Description:
    #   This subroutine computes the heterogeneous ice deposition nucleation rate.
    #
    #   ice nucleation rate = ice_initial_mass Ni / rho / dt, Eq. 5.101
    #   ice_initial_mass is the initial ice crystal mass
    #
    #------------------------------------------------------------------------------

    if ( (temperature < icon_graupel_params.heterogeneous_freeze_temperature) & (qv > 8.e-6) & (qi <= 0.0) & (qv > qvsi) ):
        snucl_v2i = icon_graupel_params.ice_initial_mass * c1orho * cnin * dtr


    #--------------------------------------------------------------------------
    # Section 3: Search for cloudy grid points with cloud water and
    #            calculation of the conversion rates involving qc (ic6)
    #--------------------------------------------------------------------------

    ##----------------------------------------------------------------------------
    ## 3.1: Autoconversion of clouds and accretion of rain
    ##----------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    # Description:
    #   This subroutine computes the rate of autoconversion and accretion by rain.
    #
    #   Method 1: liquid_autoconversion_option = 1, Kessler (1969)
    #   Method 2: liquid_autoconversion_option = 2, Seifert and beheng (2001)
    #
    #------------------------------------------------------------------------------

    # if there is cloud water and the temperature is above homogeneuous freezing temperature
    if ( llqc & (temperature > icon_graupel_params.homogeneous_freeze_temperature) ):

        if liquid_autoconversion_option == 1:
            # Kessler(1969) autoconversion rate
            scaut_c2r = icon_graupel_params.kessler_cloud2rain_autoconversion_coeff_for_cloud * maximum( qc - icon_graupel_params.qc0 , 0.0 )
            scacr_c2r = icon_graupel_params.kessler_cloud2rain_autoconversion_coeff_for_rain * qc * celn7o8qrk

        elif liquid_autoconversion_option == 2:
            # Seifert and Beheng (2001) autoconversion rate
            local_const = icon_graupel_params.kcau / (20.0 * icon_graupel_params.xstar) * (icon_graupel_params.cnue + 2.0) * (icon_graupel_params.cnue + 4.0) / (icon_graupel_params.cnue + 1.0)**2.0

            # with constant cloud droplet number concentration qnc
            if qc > 1.0e-6:
                local_tau = minimum( 1.0 - qc / (qc + qr) , 0.9 )
                local_tau = maximum( local_tau , 1.e-30 )
                local_hlp = exp(icon_graupel_params.kphi2 * log(local_tau))
                local_phi = icon_graupel_params.kphi1 * local_hlp * (1.0 - local_hlp)**3.0
                scaut_c2r = local_const * qc * qc * qc * qc / (qnc * qnc) * (1.0 + local_phi / (1.0 - local_tau)**2.0)
                local_phi = (local_tau / (local_tau + icon_graupel_params.kphi3))**4.0
                scacr_c2r = icon_graupel_params.kcac * qc * qr * local_phi


    ##----------------------------------------------------------------------------
    ## 3.2: Cloud and rain freezing in clouds
    ##----------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    # Description:
    #   This subroutine computes the freezing rate of rain in clouds.
    #
    #   Method 1: rain_freezing_option = 1, Eq. 5.168
    #   Method 2: rain_freezing_option = 2, Eq. 5.126, an approximation of combined immersion Eq. 5.80 and contact freezing Eq. 5.83 (ABANDONED)
    #
    #------------------------------------------------------------------------------

    # if there is cloud water, and the temperature is above homogeneuous freezing temperature
    if llqc:
        if temperature > icon_graupel_params.homogeneous_freeze_temperature:
            # Calculation of in-cloud rainwater freezing
            if llqr & (temperature < icon_graupel_params.threshold_freeze_temperature) & (qr > 0.1 * qc):
                if rain_freezing_option == 1:
                    srfrz_r2g = icon_graupel_params.coeff_rain_freeze1_mode1 * ( exp(icon_graupel_params.coeff_rain_freeze2_mode1 * (icon_graupel_params.threshold_freeze_temperature - temperature)) - 1.0 ) * celn7o4qrk
                elif rain_freezing_option == 2:
                    local_tfrzdiff = icon_graupel_params.threshold_freeze_temperature - temperature
                    srfrz_r2g = icon_graupel_params.coeff_rain_freeze_mode2 * local_tfrzdiff * sqrt(local_tfrzdiff) * celn27o16qrk
        else:
            # tg <= tg: ! hom. freezing of cloud and rain water
            scfrz_c2i = cscmax
            srfrz_r2g = csrmax


    ##----------------------------------------------------------------------------
    ## 3.3: Riming in clouds
    ##----------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    # Description:
    #   This subroutine computes the rate of riming by snow and graupel in clouds.
    #
    #   riming or accretion rate = 1/rho intg_0^inf m_dot f dD (Eq. 5.64)
    #   m_dot = pi/4 D^2 E(D) v(D) rho qc (Eq. 5.67)
    #
    #   snow: f = N0 exp(-lamda D), E is constant, m(D) = alpha D^beta, v(D) = v0 D^b
    #      snow riming = pi/4 qc N0 E v0 Gamma(b+3) / lamda^(b+3), lamda = (alpha N0 Gamma(beta+1) / rhoqs)^(beta+1)
    #
    #   graupel:
    #      graupel riming = 4.43 qc rhoqg^0.94878 (Eq 5.152)
    #      snow to graupel coqc = qc + 3.0nversion = 0.5 qc rhoqs^0.75 (above Eq 5.132)
    #
    #   rain shedding is on if temperature is above zero degree celcius. In this case, riming tendencies are converted to rain shedding.
    #
    #------------------------------------------------------------------------------

    # if there is cloud water and the temperature is above homogeneuous freezing temperature
    if llqc & (temperature > icon_graupel_params.homogeneous_freeze_temperature):

        if llqs:
            srims_c2s = crim * qc * exp(icon_graupel_params.ccsaxp * log(cslam))

        srimg_c2g = icon_graupel_params.crim_g * qc * celnrimexp_g

        if temperature >= icon_graupel_params.tmelt:
            sshed_c2r = srims_c2s + srimg_c2g
            srims_c2s = 0.0
            srimg_c2g = 0.0
        else:
            if qc >= icon_graupel_params.qc0:
                scosg_s2g = icon_graupel_params.csg * qc * celn3o4qsk


    ##----------------------------------------------------------------------------
    ## 3.4: Ice nucleation in clouds
    ##----------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    # Description:
    #   This subroutine computes the ice nucleation in clouds.
    #
    #   Calculation of heterogeneous nucleation of cloud ice in clouds.
    #   This is done in this section, because we require water saturation
    #   for this process (i.e. the existence of cloud water) to exist.
    #   Heterogeneous nucleation is assumed to occur only when no
    #   cloud ice is present and the temperature is below a nucleation
    #   threshold.
    #
    #------------------------------------------------------------------------------

    # if there is cloud water
    if llqc & (temperature <= 267.15) & (qi <= icon_graupel_params.qmin):
        # TODO (Chia Rui): This originally corresponds to super_coolw =True/False, and it fails the test if I write it into one expression...
        if ice_concentration_option == 1:
            snucl_v2i = icon_graupel_params.ice_initial_mass * c1orho * cnin * dtr
        elif ice_concentration_option == 2:
            snucl_v2i = icon_graupel_params.ice_initial_mass / rho * cnin * dtr

    ##----------------------------------------------------------------------------
    ## 3.5: Reduced deposition in clouds
    ##----------------------------------------------------------------------------

    if do_reduced_icedeposition & llqc:
        if (k_lev > kstart_moist) & (k_lev < kend):

            cqcgk_1 = qi_kup + qs_kup + qg_kup

            # distance from cloud top
            if (qv_kup + qc_kup < qvsw_kup) & (cqcgk_1 < icon_graupel_params.qmin):
                # upper cloud layer
                dist_cldtop = 0.0  # reset distance to upper cloud layer
            else:
                dist_cldtop = dist_cldtop + dz

    if do_reduced_icedeposition & llqc:
        if (k_lev > kstart_moist) & (k_lev < kend):
            # finalizing transfer rates in clouds and calculate depositional growth reduction
            # function called: Cnin_cooper = _fxna_cooper(temperature)
            cnin_cooper = 5.0 * exp(0.304 * (icon_graupel_params.tmelt - temperature))
            cnin_cooper = minimum(cnin_cooper, nimax)
            cfnuc = minimum(cnin_cooper / nimix, 1.0)

            # with asymptotic behaviour dz -> 0 (xxx)
            #        reduce_dep = MIN(fnuc + (1.0_wp-fnuc)*(reduce_dep_ref + &
            #                             dist_cldtop(iv)/dist_cldtop_ref + &
            #                             (1.0_wp-reduce_dep_ref)*(zdh/dist_cldtop_ref)**4), 1.0_wp)

            # without asymptotic behaviour dz -> 0
            reduce_dep = cfnuc + (1.0 - cfnuc) * (icon_graupel_params.reduce_dep_ref + dist_cldtop / icon_graupel_params.dist_cldtop_ref)
            reduce_dep = minimum(reduce_dep, 1.0)

    #------------------------------------------------------------------------
    # Section 4: Search for cold grid points with cloud ice and/or snow and
    #            calculation of the conversion rates involving qi, qs and qg
    #------------------------------------------------------------------------

    ##----------------------------------------------------------------------------
    ## 4.1: Aggregation in ice clouds
    ## 4.2: Autoconversion of ice
    ## 4.3: Riming between rain and ice in ice clouds
    ##----------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    # Description:
    #   This subroutine computes the aggregation of snow and graupel in ice clouds when temperature is below zero degree celcius.
    #
    #
    #   aggregation rate = 1/rho intg_0^inf m_dot f dD (Eq. 5.64)
    #   m_dot = pi/4 D^2 E(T) v(D) rho qi (Eq. 5.67)
    #
    #   snow: f = N0 exp(-lamda D), E changes with temperature, m(D) = alpha D^beta, v(D) = v0 D^b
    #      snow aggregation = pi/4 qi N0 E(T) v0 Gamma(b+3) / lamda^(b+3), lamda = (alpha N0 Gamma(beta+1) / rhoqs)^(beta+1)
    #
    #   graupel:
    #      graupel aggregation = 2.46 qc rhoqg^0.94878 (Eq 5.154)
    #
    # Description:
    #   This subroutine computes the autoconversion of ice crystals in ice clouds when temperature is below zero degree celcius.
    #
    #
    #   iceAutoconversion = max(0.001 (qi - qi0), 0) Eq. 5.133
    #
    # Description:
    #   This subroutine computes the ice loss and rain loss due to accretion of rain in ice clouds when temperature is below zero degree celcius.
    #
    #   riming rate = 1/rho intg_0^inf m_dot f dD (Eq. 5.64)
    #   m_dot(ice loss) = pi/4 D^2 E(T) v(D) rho qi (Eq. 5.67)
    #   m_dot(rain loss) = pi/4 D^5 E(T) v(D) rho qi (Eq. 5.67)
    #
    #   rain: f = N0 D^mu exp(-lamda D), E is a constant, m(D) = alpha D^beta, v(D) = v0 D^b, b = 0.5 (Eq. 5.57)
    #   ice: uniform size=Di and mass=mi
    #
    #   ice loss = pi/4 qi N0 E v0 Gamma(b+3) / lamda^(b+3), lamda = (alpha N0 Gamma(beta+1) / rhoqs)^(beta+1)
    #
    #   rain loss = pi/4 qi N0 E v0 Gamma(b+3) / lamda^(b+3), lamda = (alpha N0 Gamma(beta+1) / rhoqs)^(beta+1)
    #
    # ------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------

    #local_eff = 0.0
    if (temperature <= icon_graupel_params.tmelt) & llqi:

        # Change in sticking efficiency needed in case of cloud ice sedimentation
        # (based on Guenther Zaengls work)
        if ice_autocon_sticking_efficiency_option == 1:
            local_eff = minimum( exp(0.09 * (temperature - icon_graupel_params.tmelt)) , 1.0 )
            local_eff = maximum( local_eff , ice_stickeff_min )
            local_eff = maximum( local_eff , icon_graupel_params.stick_eff_fac * (temperature - icon_graupel_params.tmin_iceautoconv) )
        elif ice_autocon_sticking_efficiency_option == 2: #original sticking efficiency of cloud ice
            local_eff = minimum( exp(0.09 * (temperature - icon_graupel_params.tmelt)) , 1.0 )
            local_eff = maximum( local_eff , 0.2 )
        else:
            local_eff = 0.0

        local_nid = rho * qi / cmi
        local_lnlogmi = log(cmi)

        local_qvsidiff = qv - qvsi
        local_svmax = local_qvsidiff * dtr

        saggs_i2s = local_eff * qi * cagg * exp(icon_graupel_params.ccsaxp * log(cslam))
        saggg_i2g = local_eff * qi * icon_graupel_params.cagg_g * celnrimexp_g
        siaut_i2s = local_eff * icon_graupel_params.ciau * maximum( qi - icon_graupel_params.qi0 , 0.0 )

        sicri_i2g = icon_graupel_params.cicri * qi * celn7o8qrk
        if (qs > 1.e-7):
            srcri_r2g = icon_graupel_params.crcri * (qi / cmi) * celn13o8qrk

        local_icetotaldeposition = cidep * local_nid * exp(0.33 * local_lnlogmi) * local_qvsidiff
        sidep_v2i = local_icetotaldeposition
        # szdep_v2i = 0.0
        # szsub_v2i = 0.0

        # for sedimenting quantities the maximum
        # allowed depletion is determined by the predictor value.
        local_simax = rhoqi_intermediate * c1orho * dtr if do_ice_sedimentation else qi * dtr

        if local_icetotaldeposition > 0.0:
            if do_reduced_icedeposition:
                local_iceTotalDeposition = local_icetotaldeposition * reduce_dep  # FR new: depositional growth reduction
            szdep_v2i = minimum(local_icetotaldeposition, local_svmax)
        elif local_icetotaldeposition < 0.0:
            szsub_v2i = maximum(local_icetotaldeposition, local_svmax)
            szsub_v2i = - maximum(szsub_v2i, -local_simax)

        local_lnlogmi = log(icon_graupel_params.msmin / cmi)
        local_ztau = 1.5 * (exp(0.66 * local_lnlogmi) - 1.0)
        sdaut_i2s = szdep_v2i / local_ztau

    ##----------------------------------------------------------------------------
    ## 4.4: Vapor deposition in ice clouds
    ##----------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    # Description:
    #   This subroutine computes the vapor deposition of ice crystals and snow in ice clouds when temperature is below zero degree celcius.
    #
    #
    #   deposition rate = 1/rho intg_0^inf m_dot f dD (Eq. 5.64)
    #   m_dot = 4 pi C(D) G F(v,D) d (qv - qvsi),
    #   G = 1/(1+Hw) and d are functions of environment
    #   F = 1 + 0.26 sqrt(D v(D)/eta/2) is ventilation factor
    #   C(D) = C0 D is capacitance (C0 = D/2 for a sphere, D/pi for a circular disk)
    #
    #   ice: f = Ni delta(D-Di), mi = ai Di^3 = rho qi / Nin, v(D) = 0
    #   ice deposition or sublimation rate = c_dep Ni mi (qv - qvsi), Eq. 5.104
    #
    #   snow resulted from fast ice deposition = ice deposition rate / time_scale, Eq. 5.108
    #
    #   snow: f = N0 exp(-lamda D), v = v0 D^b
    #   snow deposition rate = Eq. 5.118 (wrong?) , derived from Eq. 5.71 and 5.72
    #      = 4 G d (qv - qvsi) N0 (1 + 0.26 sqrt(v0/eta/2) Gamma((5+b)/2)) / lamda^((1+b)/2) 1/lamda^(2)
    #
    #   graupel deposition = Eq. 5.140
    #
    #------------------------------------------------------------------------------

    if llqi | llqs | llqg:

        if temperature <= icon_graupel_params.tmelt:

            local_qvsidiff = qv - qvsi
            local_svmax = local_qvsidiff * dtr

            local_xfac = 1.0 + cbsdep * exp(icon_graupel_params.ccsdxp * log(cslam))
            ssdep_v2s = csdep * local_xfac * local_qvsidiff / (cslam + icon_graupel_params.eps)**2.0
            #FR new: depositional growth reduction
            if do_reduced_icedeposition & (ssdep_v2s > 0.0):
                ssdep_v2s = ssdep_v2s * reduce_dep

            # GZ: This limitation, which was missing in the original graupel scheme,
            # is crucial for numerical stability in the tropics!
            if ssdep_v2s > 0.0:
                ssdep_v2s = minimum( ssdep_v2s , local_svmax - szdep_v2i )
            # Suppress depositional growth of snow if the existing amount is too small for a
            # a meaningful distiction between cloud ice and snow
            if qs <= 1.e-7:
                ssdep_v2s = minimum( ssdep_v2s , 0.0 )
            # ** GZ: this numerical fit should be replaced with a physically more meaningful formulation **
            sgdep_v2g = (
                0.398561 -
                0.00152398 * temperature +
                2554.99 / pres +
                2.6531e-7 * pres
            ) * local_qvsidiff * celn6qgk


    #------------------------------------------------------------------------
    # Section 5: Search for warm grid points with cloud ice and/or snow and
    #            calculation of the melting rates of qi and ps
    #------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    # Description:
    #   This subroutine computes the vapor deposition of ice crystals, snow, and graupel in ice clouds when temperature is above zero degree celcius.
    #
    #
    #   Ice crystals completely melt when temperature is above zero.
    #
    #   For snow and graupel, follow Eqs. 5.141 - 5.146
    #
    #------------------------------------------------------------------------------

    if llqi | llqs | llqg:

        if temperature > icon_graupel_params.tmelt:

            # cloud ice melts instantaneously
            simlt_i2c = rhoqi_intermediate * c1orho * dtr if do_ice_sedimentation else qi * dtr

            local_qvsw0 = icon_graupel_params.pvsw0 / (rho * icon_graupel_params.rv * icon_graupel_params.tmelt)
            local_qvsw0diff = qv - local_qvsw0

            # ** GZ: several numerical fits in this section should be replaced with physically more meaningful formulations **
            if temperature > icon_graupel_params.tmelt - icon_graupel_params.tcrit * local_qvsw0diff:
                #calculate melting rate
                local_x1  = temperature - icon_graupel_params.tmelt + icon_graupel_params.asmel * local_qvsw0diff
                ssmlt_s2r = (79.6863 / pres + 0.612654e-3) * local_x1 * celn8qsk
                ssmlt_s2r = minimum( ssmlt_s2r , cssmax )
                sgmlt_g2r = (12.31698 / pres + 7.39441e-05) * local_x1 * celn6qgk
                sgmlt_g2r = minimum( sgmlt_g2r , csgmax )
                #deposition + melting, ice particle temperature: t0
                #calculation without howell-factor!
                ssdep_v2s  = (31282.3 / pres + 0.241897) * local_qvsw0diff * celn8qsk
                sgdep_v2g  = (0.153907 - pres * 7.86703e-07) * local_qvsw0diff * celn6qgk
                if local_qvsw0diff < 0.0:
                    #melting + evaporation of snow/graupel
                    ssdep_v2s = maximum( -cssmax , ssdep_v2s )
                    sgdep_v2g = maximum( -csgmax , sgdep_v2g )
                    #melt water evaporates
                    ssmlt_s2r = ssmlt_s2r + ssdep_v2s
                    sgmlt_g2r = sgmlt_g2r + sgdep_v2g
                    ssmlt_s2r = maximum( ssmlt_s2r , 0.0 )
                    sgmlt_g2r = maximum( sgmlt_g2r , 0.0 )
                else:
                    #deposition on snow/graupel is interpreted as increase
                    #in rain water ( qv --> qr, sconr)
                    #therefore,  sconr=(zssdep+zsgdep)
                    sconr_v2r = ssdep_v2s + sgdep_v2g
                    ssdep_v2s = 0.0
                    sgdep_v2g = 0.0
            else:
                #if t<t_crit
                #no melting, only evaporation of snow/graupel
                #local_qvsw      = sat_pres_water(input_t) / (input_rho * icon_graupel_params.rv * input_t)
                #output_qvsw_kup = local_qvsw ! redundant in the original code?
                local_qvsidiff  = qv - qvsw
                ssdep_v2s = (0.28003 - pres * 0.146293e-6) * local_qvsidiff * celn8qsk
                sgdep_v2g = (0.0418521 - pres * 4.7524e-8) * local_qvsidiff * celn6qgk
                ssdep_v2s = maximum( -cssmax , ssdep_v2s )
                sgdep_v2g = maximum( -csgmax , sgdep_v2g )

    #--------------------------------------------------------------------------
    # Section 6: Search for grid points with rain in subsaturated areas
    #            and calculation of the evaporation rate of rain
    #--------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    # Description:
    #   This subroutine computes the evaporation rate of rain in subsaturated condition.
    #
    #
    #   deposition rate = 1/rho intg_0^inf m_dot f dD (Eq. 5.64)
    #   m_dot = 4 pi C(D) G F(v,D) d (qv - qvsw),
    #   G = 1/(1+Hw) and d are functions of environment
    #   F = 1 + 0.26 sqrt(D v(D)/eta/2) is ventilation factor
    #   C(D) = C0 D is capacitance (C0 = D/2 for a sphere, D/pi for a circular disk)
    #
    #   snow resulted from fast ice deposition = ice deposition rate / time_scale, Eq. 5.108
    #
    #   rain: gamma distribution f = N0 D^(mu) exp(-lamda D), m = alpha D^beta, v = v0 D^b
    #       V = v0 Gamma(b + beta + mu + 1) 1 / (alpha N0 Gamma(beta + mu + 1) )^(b/(beta + mu + 1)) (rho q)^(b/(beta + mu + 1)) rho_factor
    #       rain evaporation rate = Eq. 5.117 (wrong?) , derived from Eq. 5.71 and 5.72
    #                             = 2 pi G d (qv - qvsw) N0 (Gamma(2+mu) + 0.26 sqrt(v0/eta/2) Gamma((5+b+mu)/2)) / lamda^((1+b)/2) 1/lamda^(2+mu)
    #       lamda = (alpha N0 Gamma(beta+mu+1) / rhoq )^(1/(beta+mu+1))
    #
    #       rain freezing rate =
    #          Method 1: rain_freezing_option = 1, Eq. 5.168
    #          Method 2: rain_freezing_option = 2, Eq. 5.126, an approximation of combined immersion Eq. 5.80 and contact freezing Eq. 5.83 (ABANDONED)
    #
    #------------------------------------------------------------------------------

    if llqr & (qv + qc <= qvsw):

        local_lnqr = log(rhoqr)
        local_x1 = 1.0 + bev * exp(bevxp * local_lnqr)
        #sev  = zcev*zx1*(zqvsw - qvg) * EXP (zcevxp  * zlnqrk)
        # Limit evaporation rate in order to avoid overshoots towards supersaturation
        # the pre-factor approximates (esat(T_wb)-e)/(esat(T)-e) at temperatures between 0 degC and 30 degC
        local_temp_c = temperature - icon_graupel_params.tmelt
        local_maxevap = (0.61 - 0.0163 * local_temp_c + 1.111e-4 * local_temp_c**2.0) * (qvsw - qv) / dt
        sevap_r2v = cev * local_x1 * (qvsw - qv) * exp(cevxp * local_lnqr)
        sevap_r2v = minimum( sevap_r2v , local_maxevap )

        if temperature > icon_graupel_params.homogeneous_freeze_temperature:
            # Calculation of below-cloud rainwater freezing
            if temperature < icon_graupel_params.threshold_freeze_temperature:
                if rain_freezing_option == 1:
                    #FR new: reduced rain freezing rate
                    srfrz_r2g = icon_graupel_params.coeff_rain_freeze1_mode1 * (exp(icon_graupel_params.coeff_rain_freeze2_mode1 * (icon_graupel_params.threshold_freeze_temperature - temperature)) - 1.0 ) * celn7o4qrk
                elif rain_freezing_option == 2:
                    srfrz_r2g = icon_graupel_params.coeff_rain_freeze_mode2 * sqrt( (icon_graupel_params.threshold_freeze_temperature - temperature)**3.0 ) * celn27o16qrk
        else: # Hom. freezing of rain water
            srfrz_r2g = csrmax

    #--------------------------------------------------------------------------
    # Section 7: Calculate the total tendencies of the prognostic variables.
    #            Update the prognostic variables in the interior domain.
    #--------------------------------------------------------------------------

    # finalizing transfer rates in clouds and calculate depositional growth reduction
    if llqc & (temperature > icon_graupel_params.homogeneous_freeze_temperature):
        # Check for maximum depletion of cloud water and adjust the
        # transfer rates accordingly
        csum = scaut_c2r + scacr_c2r + srims_c2s + srimg_c2g + sshed_c2r
        ccorr = cscmax / maximum(cscmax, csum)
        scaut_c2r = ccorr * scaut_c2r
        scacr_c2r = ccorr * scacr_c2r
        srims_c2s = ccorr * srims_c2s
        srimg_c2g = ccorr * srimg_c2g
        sshed_c2r = ccorr * sshed_c2r
        scosg_s2g = minimum(scosg_s2g, srims_c2s + cssmax)

    if llqi | llqs | llqg:
        if temperature <= icon_graupel_params.tmelt:  # cold case

            qvsidiff = qv - qvsi
            csimax = rhoqi_intermediate * c1orho * dtr if do_ice_sedimentation else qi * dtr

            # Check for maximal depletion of cloud ice
            # No check is done for depositional autoconversion (sdau) because
            # this is a always a fraction of the gain rate due to
            # deposition (i.e the sum of this rates is always positive)
            csum = siaut_i2s + saggs_i2s + saggg_i2g + sicri_i2g + szsub_v2i
            ccorr = csimax / maximum(csimax, csum) if csimax > 0.0 else 0.0
            sidep_v2i = szdep_v2i - ccorr * szsub_v2i
            siaut_i2s = ccorr * siaut_i2s
            saggs_i2s = ccorr * saggs_i2s
            saggg_i2g = ccorr * saggg_i2g
            sicri_i2g = ccorr * sicri_i2g
            if qvsidiff < 0.0:
                ssdep_v2s = maximum(ssdep_v2s, -cssmax)
                sgdep_v2g = maximum(sgdep_v2g, -csgmax)

    csum = sevap_r2v + srfrz_r2g + srcri_r2g
    ccorr = csrmax / maximum(csrmax, csum) if csum > 0.0 else 1.0
    sevap_r2v = ccorr * sevap_r2v
    srfrz_r2g = ccorr * srfrz_r2g
    srcri_r2g = ccorr * srcri_r2g

    # limit snow depletion in order to avoid negative values of qs
    ccorr = 1.0
    if ssdep_v2s <= 0.0:
        Csum = ssmlt_s2r + scosg_s2g - ssdep_v2s
        if Csum > 0.0: ccorr = cssmax / maximum(cssmax, csum)
        ssmlt_s2r = ccorr * ssmlt_s2r
        scosg_s2g = ccorr * scosg_s2g
        ssdep_v2s = ccorr * ssdep_v2s
    else:
        csum = ssmlt_s2r + scosg_s2g
        if csum > 0.0: ccorr = cssmax / maximum(cssmax, csum)
        ssmlt_s2r = ccorr * ssmlt_s2r
        scosg_s2g = ccorr * scosg_s2g

    cqvt = sevap_r2v - sidep_v2i - ssdep_v2s - sgdep_v2g - snucl_v2i - sconr_v2r
    cqct = simlt_i2c - scaut_c2r - scfrz_c2i - scacr_c2r - sshed_c2r - srims_c2s - srimg_c2g
    cqit = snucl_v2i + scfrz_c2i - simlt_i2c - sicri_i2g + sidep_v2i - sdaut_i2s - saggs_i2s - saggg_i2g - siaut_i2s
    cqrt = scaut_c2r + sshed_c2r + scacr_c2r + ssmlt_s2r + sgmlt_g2r - sevap_r2v - srcri_r2g - srfrz_r2g + sconr_v2r
    cqst = siaut_i2s + sdaut_i2s - ssmlt_s2r + srims_c2s + ssdep_v2s + saggs_i2s - scosg_s2g
    cqgt = saggg_i2g - sgmlt_g2r + sicri_i2g + srcri_r2g + sgdep_v2g + srfrz_r2g + srimg_c2g + scosg_s2g

    # First method:
    ctt = heat_cap_r * (lhv * (cqct + cqrt) + lhs * (cqit + cqst + cqgt))
    # Second method:
    #solid_phase_change = snucl_v2i + scfrz_c2i - simlt_i2c + sidep_v2i - ssmlt_s2r + srims_c2s - sgmlt_g2r + srcri_r2g + sgdep_v2g + srfrz_r2g + srimg_c2g
    #liquid_phase_change = simlt_i2c - scfrz_c2i - srims_c2s - srimg_c2g + ssmlt_s2r + sgmlt_g2r - sevap_r2v - srcri_r2g - srfrz_r2g + sconr_v2r
    #Ctt = heat_cap_r * (liquid_phase_change * lhv + solid_phase_change * CLHs)
    # Third method
    #v2s_phase_change = snucl_v2i + sidep_v2i + sgdep_v2g
    #l2s_phase_change = scfrz_c2i - simlt_i2c - ssmlt_s2r + srims_c2s - sgmlt_g2r + srcri_r2g + srfrz_r2g + srimg_c2g
    #v2l_phase_change = sconr_v2r - sevap_r2v
    #Ctt = heat_cap_r * CLHs * (v2s_phase_change + l2s_phase_change) + heat_cap_r * lhv * (v2l_phase_change - l2s_phase_change)

    # Update variables and add qi to qrs for water loading
    qi = maximum( 0.0 , (rhoqi_intermediate * c1orho + cqit * dt) * cimi ) if do_ice_sedimentation else maximum( 0.0 , qi + cqit * dt )
    qr = maximum( 0.0 , (rhoqr_intermediate * c1orho + cqrt * dt) * cimr )
    qs = maximum( 0.0 , (rhoqs_intermediate * c1orho + cqst * dt) * cims )
    qg = maximum( 0.0 , (rhoqg_intermediate * c1orho + cqgt * dt) * cimg )

    # Update of prognostic variables or tendencies
    temperature = temperature + ctt * dt
    qv = maximum( 0.0 , qv + cqvt * dt )
    qc = maximum( 0.0 , qc + cqct * dt )

    # tracing current k level
    k_lev = k_lev + 1


    return (
        temperature,
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
        k_lev
    )

@field_operator
def _compute_icon_graupel_t_tendency(
    dt: wpfloat,
    temperature_new: gtx.Field[[CellDim,KDim], vpfloat],
    temperature_old: gtx.Field[[CellDim,KDim], vpfloat]
) -> gtx.Field[[CellDim,KDim], vpfloat]:
    dtr = 1.0 / dt
    return (temperature_new - temperature_old) * dtr

@field_operator
def _compute_icon_graupel_q_tendency(
    dt: wpfloat,
    qv_new: gtx.Field[[CellDim, KDim], vpfloat],
    qc_new: gtx.Field[[CellDim, KDim], vpfloat],
    qi_new: gtx.Field[[CellDim, KDim], vpfloat],
    qr_new: gtx.Field[[CellDim, KDim], vpfloat],
    qs_new: gtx.Field[[CellDim, KDim], vpfloat],
    qg_new: gtx.Field[[CellDim, KDim], vpfloat],
    qv_old: gtx.Field[[CellDim, KDim], vpfloat],
    qc_old: gtx.Field[[CellDim, KDim], vpfloat],
    qi_old: gtx.Field[[CellDim, KDim], vpfloat],
    qr_old: gtx.Field[[CellDim, KDim], vpfloat],
    qs_old: gtx.Field[[CellDim, KDim], vpfloat],
    qg_old: gtx.Field[[CellDim, KDim], vpfloat]
) -> tuple[
    gtx.Field[[CellDim,KDim], vpfloat],
    gtx.Field[[CellDim,KDim], vpfloat],
    gtx.Field[[CellDim,KDim], vpfloat],
    gtx.Field[[CellDim,KDim], vpfloat],
    gtx.Field[[CellDim,KDim], vpfloat],
    gtx.Field[[CellDim,KDim], vpfloat]
]:
    dtr = 1.0 / dt
    return (
        maximum( -qv_old * dtr , (qv_new - qv_old) * dtr ),
        maximum( -qc_old * dtr , (qc_new - qc_old) * dtr ),
        maximum( -qi_old * dtr , (qi_new - qi_old) * dtr ),
        maximum( -qr_old * dtr , (qr_new - qr_old) * dtr ),
        maximum( -qs_old * dtr , (qs_new - qs_old) * dtr ),
        maximum( -qg_old * dtr , (qg_new - qg_old) * dtr)
    )

@field_operator
def _icon_graupel(
    kstart_moist: gtx.int32,
    kend: gtx.int32,
    liquid_autoconversion_option: gtx.int32,
    snow_intercept_option: gtx.int32,
    rain_freezing_option: gtx.int32,
    ice_concentration_option: gtx.int32,
    do_ice_sedimentation: bool,
    ice_autocon_sticking_efficiency_option: gtx.int32,
    do_reduced_icedeposition: bool,
    is_isochoric: bool,
    use_constant_water_heat_capacity: bool,
    ice_stickeff_min: wpfloat,
    ice_v0: wpfloat,
    ice_sedi_density_factor_exp: wpfloat,
    snow_v0: wpfloat,
    ccsrim: wpfloat,
    ccsagg: wpfloat,
    ccsvel: wpfloat,
    nimax: wpfloat,
    nimix: wpfloat,
    rain_exp_v: wpfloat,
    rain_v0: wpfloat,
    cevxp: wpfloat,
    cev: wpfloat,
    bevxp: wpfloat,
    bev: wpfloat,
    rain_exp_v_ln1o2: wpfloat,
    ice_exp_v_ln1o2: wpfloat,
    graupel_exp_sed_ln1o2: wpfloat,
    dt: wpfloat,  # time step
    dz: gtx.Field[[CellDim, KDim], wpfloat],
    temperature: gtx.Field[[CellDim, KDim], vpfloat],
    pres: gtx.Field[[CellDim, KDim], vpfloat],
    rho: gtx.Field[[CellDim, KDim], vpfloat],
    qv: gtx.Field[[CellDim, KDim], vpfloat],
    qc: gtx.Field[[CellDim, KDim], vpfloat],
    qi: gtx.Field[[CellDim, KDim], vpfloat],
    qr: gtx.Field[[CellDim, KDim], vpfloat],
    qs: gtx.Field[[CellDim, KDim], vpfloat],
    qg: gtx.Field[[CellDim, KDim], vpfloat],
    qnc: gtx.Field[[CellDim, KDim], vpfloat], # originally 2D Field, now 3D Field
) -> tuple[
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat]
]:
    (
        temperature_,
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
        k_lev
    ) = _icon_graupel_scan(
        kstart_moist,
        kend,
        liquid_autoconversion_option,
        snow_intercept_option,
        rain_freezing_option,
        ice_concentration_option,
        do_ice_sedimentation,
        ice_autocon_sticking_efficiency_option,
        do_reduced_icedeposition,
        is_isochoric,
        use_constant_water_heat_capacity,
        ice_stickeff_min,
        ice_v0,
        ice_sedi_density_factor_exp,
        snow_v0,
        ccsrim,
        ccsagg,
        ccsvel,
        nimax,
        nimix,
        rain_exp_v,
        rain_v0,
        cevxp,
        cev,
        bevxp,
        bev,
        rain_exp_v_ln1o2,
        ice_exp_v_ln1o2,
        graupel_exp_sed_ln1o2,
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
        qnc
    )

    temperature_tendency = _compute_icon_graupel_t_tendency(
        dt,
        temperature_,
        temperature
    )

    qv_tendency, qc_tendency, qi_tendency, qr_tendency, qs_tendency, qg_tendency = _compute_icon_graupel_q_tendency(
        dt,
        qv_,
        qc_,
        qi_,
        qr_,
        qs_,
        qg_,
        qv,
        qc,
        qi,
        qr,
        qs,
        qg
    )

    return(
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
        vnew_i
    )


@program
def icon_graupel(
    kstart_moist: gtx.int32,
    liquid_autoconversion_option: gtx.int32,
    snow_intercept_option: gtx.int32,
    rain_freezing_option: gtx.int32,
    ice_concentration_option: gtx.int32,
    do_ice_sedimentation: bool,
    ice_autocon_sticking_efficiency_option: gtx.int32,
    do_reduced_icedeposition: bool,
    is_isochoric: bool,
    use_constant_water_heat_capacity: bool,
    ice_stickeff_min: wpfloat,
    ice_v0: wpfloat,
    ice_sedi_density_factor_exp: wpfloat,
    snow_v0: wpfloat,
    ccsrim: wpfloat,
    ccsagg: wpfloat,
    ccsvel: wpfloat,
    nimax: wpfloat,
    nimix: wpfloat,
    rain_exp_v: wpfloat,
    rain_v0: wpfloat,
    cevxp: wpfloat,
    cev: wpfloat,
    bevxp: wpfloat,
    bev: wpfloat,
    rain_exp_v_ln1o2: wpfloat,
    ice_exp_v_ln1o2: wpfloat,
    graupel_exp_sed_ln1o2: wpfloat,
    dt: wpfloat,  # time step
    dz: gtx.Field[[CellDim, KDim], wpfloat],
    temperature: gtx.Field[[CellDim, KDim], vpfloat],
    pres: gtx.Field[[CellDim, KDim], vpfloat],
    rho: gtx.Field[[CellDim, KDim], vpfloat],
    qv: gtx.Field[[CellDim, KDim], vpfloat],
    qc: gtx.Field[[CellDim, KDim], vpfloat],
    qi: gtx.Field[[CellDim, KDim], vpfloat],
    qr: gtx.Field[[CellDim, KDim], vpfloat],
    qs: gtx.Field[[CellDim, KDim], vpfloat],
    qg: gtx.Field[[CellDim, KDim], vpfloat],
    qnc: gtx.Field[[CellDim, KDim], vpfloat],  # originally 2D Field, now 3D Field
    temperature_tendency: gtx.Field[[CellDim, KDim], vpfloat],
    qv_tendency: gtx.Field[[CellDim, KDim], vpfloat],
    qc_tendency: gtx.Field[[CellDim, KDim], vpfloat],
    qi_tendency: gtx.Field[[CellDim, KDim], vpfloat],
    qr_tendency: gtx.Field[[CellDim, KDim], vpfloat],
    qs_tendency: gtx.Field[[CellDim, KDim], vpfloat],
    qg_tendency: gtx.Field[[CellDim, KDim], vpfloat],
    rhoqrv_old_kup: gtx.Field[[CellDim, KDim], vpfloat],
    rhoqsv_old_kup: gtx.Field[[CellDim, KDim], vpfloat],
    rhoqgv_old_kup: gtx.Field[[CellDim, KDim], vpfloat],
    rhoqiv_old_kup: gtx.Field[[CellDim, KDim], vpfloat],
    vnew_r: gtx.Field[[CellDim, KDim], vpfloat],
    vnew_s: gtx.Field[[CellDim, KDim], vpfloat],
    vnew_g: gtx.Field[[CellDim, KDim], vpfloat],
    vnew_i: gtx.Field[[CellDim, KDim], vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _icon_graupel(
        kstart_moist,
        vertical_end,
        liquid_autoconversion_option,
        snow_intercept_option,
        rain_freezing_option,
        ice_concentration_option,
        do_ice_sedimentation,
        ice_autocon_sticking_efficiency_option,
        do_reduced_icedeposition,
        is_isochoric,
        use_constant_water_heat_capacity,
        ice_stickeff_min,
        ice_v0,
        ice_sedi_density_factor_exp,
        snow_v0,
        ccsrim,
        ccsagg,
        ccsvel,
        nimax,
        nimix,
        rain_exp_v,
        rain_v0,
        cevxp,
        cev,
        bevxp,
        bev,
        rain_exp_v_ln1o2,
        ice_exp_v_ln1o2,
        graupel_exp_sed_ln1o2,
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
            vnew_i
        ),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        }
    )

# TODO (Chia Rui): replace this scan operator with a field operator since it needs no information from other levels
@scan_operator(
    axis=KDim,
    forward=True,
    init=(
        *(0.0,) * 5,  # rain, snow, graupel, ice, solid predicitation fluxes
        0
    ),
)
def _icon_graupel_flux_scan(
    state_kup: tuple[
        vpfloat,  # rain flux
        vpfloat,  # snow flux
        vpfloat,  # graupel flux
        vpfloat,  # ice flux
        vpfloat,  # solid precipitation flux
        gtx.int32  # k level
    ],
    # Grid information
    kstart_moist: gtx.int32,  # k starting level
    kend: gtx.int32,  # k bottom level
    rho: vpfloat,
    qr: vpfloat,
    qs: vpfloat,
    qg: vpfloat,
    qi: vpfloat,
    rhoqrv_old_kup: vpfloat,
    rhoqsv_old_kup: vpfloat,
    rhoqgv_old_kup: vpfloat,
    rhoqiv_old_kup: vpfloat,
    vnew_r: vpfloat,
    vnew_s: vpfloat,
    vnew_g: vpfloat,
    vnew_i: vpfloat,
    do_ice_sedimentation: bool,
    do_latent_heat_nudging: bool
):
    # unpack carry
    (
        rain_flux_kup,
        snow_flux_kup,
        graupel_flux_kup,
        ice_flux_kup,
        total_flux_kup,
        k_lev
    ) = state_kup

    # | (k_lev > kend_moist)
    if k_lev < kstart_moist:
        # tracing current k level
        k_lev = k_lev + 1
        return(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            k_lev
        )

    if k_lev == kend:
        # Precipitation fluxes at the ground
        rain_flux = 0.5 * (qr * rho * vnew_r + rhoqrv_old_kup)
        snow_flux = 0.5 * (qs * rho * vnew_s + rhoqsv_old_kup)
        graupel_flux = 0.5 * (qg * rho * vnew_g + rhoqgv_old_kup)
        ice_flux = 0.5 * (qi * rho * vnew_i + rhoqiv_old_kup) if do_ice_sedimentation else 0.0
        # TODO (CHia Rui): check whether lpres_ice is redundant, prs_gsp = 0.5 * (rho * (qs * Vnew_s + qi * Vnew_i) + rhoqsV_old_kup + rhoqiV_old_kup)
        # for the latent heat nudging
        total_flux = rain_flux + snow_flux + graupel_flux if do_latent_heat_nudging else 0.0
    else:
        rain_flux = qr * rho * vnew_r
        snow_flux = qs * rho * vnew_s
        graupel_flux = qg * rho * vnew_g
        ice_flux = qi * rho * vnew_i if do_ice_sedimentation else 0.0
        if rain_flux <= icon_graupel_params.qmin: rain_flux = 0.0
        if snow_flux <= icon_graupel_params.qmin: snow_flux = 0.0
        if graupel_flux <= icon_graupel_params.qmin: graupel_flux = 0.0
        if ice_flux <= icon_graupel_params.qmin: ice_flux = 0.0

        if do_latent_heat_nudging:
            if do_ice_sedimentation:
                total_flux = rain_flux + snow_flux + graupel_flux + ice_flux
                total_flux = 0.5 * (total_flux + rhoqrv_old_kup + rhoqsv_old_kup + rhoqgv_old_kup + rhoqiv_old_kup)
            else:
                total_flux = rain_flux + snow_flux + graupel_flux
                total_flux = 0.5 * (total_flux + rhoqrv_old_kup + rhoqsv_old_kup + rhoqgv_old_kup)
        else:
            total_flux = 0.0

    # tracing current k level
    k_lev = k_lev + 1

    return(
        rain_flux,
        snow_flux,
        graupel_flux,
        ice_flux,
        total_flux,
        k_lev
    )

@field_operator
def _icon_graupel_flux(
    kstart_moist: gtx.int32,
    kend: gtx.int32,
    rho: gtx.Field[[CellDim, KDim], vpfloat],
    qr: gtx.Field[[CellDim, KDim], vpfloat],
    qs: gtx.Field[[CellDim, KDim], vpfloat],
    qg: gtx.Field[[CellDim, KDim], vpfloat],
    qi: gtx.Field[[CellDim, KDim], vpfloat],
    rhoqrv_old_kup: gtx.Field[[CellDim, KDim], vpfloat],
    rhoqsv_old_kup: gtx.Field[[CellDim, KDim], vpfloat],
    rhoqgv_old_kup: gtx.Field[[CellDim, KDim], vpfloat],
    rhoqiv_old_kup: gtx.Field[[CellDim, KDim], vpfloat],
    vnew_r: gtx.Field[[CellDim, KDim], vpfloat],
    vnew_s: gtx.Field[[CellDim, KDim], vpfloat],
    vnew_g: gtx.Field[[CellDim, KDim], vpfloat],
    vnew_i: gtx.Field[[CellDim, KDim], vpfloat],
    do_ice_sedimentation: bool,
    do_latent_heat_nudging: bool
) -> tuple[
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
]:
    rain_flux, snow_flux, graupel_flux, ice_flux, total_flux, _ = _icon_graupel_flux_scan(
        kstart_moist,
        kend,
        rho,
        qr,
        qs,
        qg,
        qi,
        rhoqrv_old_kup,
        rhoqsv_old_kup,
        rhoqgv_old_kup,
        rhoqiv_old_kup,
        vnew_r,
        vnew_s,
        vnew_g,
        vnew_i,
        do_ice_sedimentation,
        do_latent_heat_nudging
    )
    return rain_flux, snow_flux, graupel_flux, ice_flux, total_flux


@program
def icon_graupel_flux(
    kstart_moist: gtx.int32,
    kend: gtx.int32,
    rho: gtx.Field[[CellDim, KDim], vpfloat],
    qr: gtx.Field[[CellDim, KDim], vpfloat],
    qs: gtx.Field[[CellDim, KDim], vpfloat],
    qg: gtx.Field[[CellDim, KDim], vpfloat],
    qi: gtx.Field[[CellDim, KDim], vpfloat],
    rhoqrv_old_kup: gtx.Field[[CellDim, KDim], vpfloat],
    rhoqsv_old_kup: gtx.Field[[CellDim, KDim], vpfloat],
    rhoqgv_old_kup: gtx.Field[[CellDim, KDim], vpfloat],
    rhoqiv_old_kup: gtx.Field[[CellDim, KDim], vpfloat],
    vnew_r: gtx.Field[[CellDim, KDim], vpfloat],
    vnew_s: gtx.Field[[CellDim, KDim], vpfloat],
    vnew_g: gtx.Field[[CellDim, KDim], vpfloat],
    vnew_i: gtx.Field[[CellDim, KDim], vpfloat],
    rain_precipitation_flux: gtx.Field[[CellDim, KDim], vpfloat],
    snow_precipitation_flux: gtx.Field[[CellDim, KDim], vpfloat],
    graupel_precipitation_flux: gtx.Field[[CellDim, KDim], vpfloat],
    ice_precipitation_flux: gtx.Field[[CellDim, KDim], vpfloat],
    total_precipitation_flux: gtx.Field[[CellDim, KDim], vpfloat],
    do_ice_sedimentation: bool,
    do_latent_heat_nudging: bool,  # if true, latent heat nudging is applied
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _icon_graupel_flux(
        kstart_moist,
        vertical_end,
        rho,
        qr,
        qs,
        qg,
        qi,
        rhoqrv_old_kup,
        rhoqsv_old_kup,
        rhoqgv_old_kup,
        rhoqiv_old_kup,
        vnew_r,
        vnew_s,
        vnew_g,
        vnew_i,
        do_ice_sedimentation,
        do_latent_heat_nudging,
        out=(
            rain_precipitation_flux,
            snow_precipitation_flux,
            graupel_precipitation_flux,
            ice_precipitation_flux,
            total_precipitation_flux
        ),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        }
    )

#(backend=roundtrip.executor)
