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

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.math.math_utilities import gamma_fct
from icon4py.model.common.math.math_constants import math_const
from icon4py.model.common.settings import xp
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


sys.setrecursionlimit(350000)


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
    liquid_autoconversion_option: int = 1
    #: Option for deriving snow size distribution interception parameter.
    snow_intercept_option: int = 2
    #: Rain freezing mode. 1: . 2: super_coolw = True  # switch for supercooled liquid water (work from Felix Rieper)
    rain_freezing_option: int = 1
    #: Number concentration of ice particles. 1: 2:
    ice_concentration_option: int = 1
    # GrConst_lsedi_ice = True  # switch for sedimentation of cloud ice (Heymsfield & Donner 1990 *1/3)
    #: Do ice particle sedimentation.
    do_ice_sedimentation: bool = True
    # GrConst_lstickeff = True  # switch for sticking coeff. (work from Guenther Zaengl)
    #: Sticking efficiency in ice-ice auto conversion.
    ice_autocon_sticking_efficiency_option: int = 1
    # GrConst_lred_depgrow = True  # separate switch for reduced depositional growth near tops of water clouds
    #: Whether ice deposition is reduced near cloud top.
    do_reduced_icedeposition: bool = True
    #: Determine whether the microphysical processes are isochoric or isobaric. Originally defined as l_cv in ICON.
    is_isochoric: bool = True
    #: Do latent heat nudging. Originally defined as dass_lhn in ICON.
    do_latent_heat_nudging = False
    #: Whether a fixed latent heat capacities are used for water. Originally defined as ithermo_water in ICON (0 means True).
    use_constant_water_heat_capacity = True

    #: First parameter in RHS of eq. 5.163 for the sticking efficiency when ice_autocon_sticking_efficiency = 2. Default seems to be 0.075.
    ice_stickeff_min: float = 0.01
    #: Constant in v-D snow terminal velocity-Diameter relationship, see eqs. 5.57 (depricated after COSMO 3.0) and unnumbered eq. (v = 25 D^0.5) below eq. 5.159. default seems to be 20.0.
    snow_v0: float = 20.0
    #: Constant in v-qi ice terminal velocity-mixing ratio relationship, see eq. 5.169.
    ice_v0: float = 1.25 # TODO remove vz0i
    #: Density factor in ice terminal velocity equation to account for density (air thermodynamic state) change. Default seems to be 0.33.
    ice_sedi_density_factor: float = 0.3 # TODO remove ice_sedi_exp
    #: mu exponential factor in gamma distribution of rain particles.
    rain_mu: float = 0.0 # TODO remove mu_rain
    #: Interception parameter in gamma distribution of rain particles.
    rain_n0: float = 1.0 # TODO remove _factor


@dataclasses.dataclass(frozen=True)
class SingleMomentSixClassIconGraupelParams:
    """
    Contains numerical, physical, and empirical constants for the ICON graupel scheme.

    These constants are not configurable from namelists in ICON.
    If users want to tune the model for better results in specific cases, you may need to change the hard coded constants here.
    All the equations can be found in A Description of the Nonhydrostatic Regional COSMO-Model Part II Physical Parameterizations.
    """

    # config: dataclasses.InitVar[SingleMomentSixClassIconGraupelConfig]
    #: threshold temperature for heterogeneous freezing of raindrops
    # trfrz = 271.15,
    threshold_freeze_temperature: float = 271.15
    #: coefficient for raindrop freezing, see eq. 5.126
    # crfrz = 1.68,
    coeff_rain_freeze_mode1: float = 1.68
    #: FR: 1. coefficient for immersion raindrop freezing: alpha_if, see eq. 5.168
    # crfrz1 = 9.95e-5,
    coeff_rain_freeze1_mode2: float = 9.95e-5
    #: FR: 2. coefficient for immersion raindrop freezing: a_if, see eq. 5.168
    # crfrz2 = 0.66,
    coeff_rain_freeze2_mode2: float = 0.66
    #: temperature for hom. freezing of cloud water
    # thn = 236.15,
    homogeneous_freeze_temperature: float = 236.15
    #: threshold temperature for mixed-phase cloud freezing of cloud drops (Forbes 2012, Forbes & Ahlgrimm 2014), see eq. 5.166.
    # tmix = 250.15,
    threshold_freeze_temperature_mixedphase: float = 250.15
    #: threshold for lowest detectable mixing ratios
    qmin: float = 1.0e-15
    #: a small number for cloud existence criterion
    eps: float = 1.0e-15
    #: exponential factor in ice terminal velocity equation v = zvz0i*rhoqi^zbvi, see eq. 5.169
    # bvi: float = 0.16
    ice_exp_v: float = 0.16
    #: reference air density
    # rho0: float = 1.225e+0
    ref_air_density: float = 1.225e+0
    #: in m/s; minimum terminal fall velocity of rain particles (applied only near the ground)
    rain_v_sedi_min: float = 0.7
    #: in m/s; minimum terminal fall velocity of snow particles (applied only near the ground)
    snow_v_sedi_min: float = 0.1
    #: in m/s; minimum terminal fall velocity of graupel particles (applied only near the ground)
    graupel_v_sedi_min: float = 0.4
    #: maximal number of ice crystals, see eq. 5.165.
    nimax_Thom: float = 250.0e+3
    # TODO (Chia Rui): What are these two parameters for? Why they are not used but exist in ICON
    # zams_ci= 0.069           # Formfactor in the mass-size relation of snow particles for cloud ice scheme
    # zams_gr= 0.069           # Formfactor in the mass-size relation of snow particles for graupel scheme
    #: Formfactor in the mass-diameter relation of snow particles, see eq. 5.159.
    # ams: float = 0.069
    snow_m0: float = 0.069
    #: A constant intercept parameter for inverse exponential size distribution of snow particles, see eq. 5.160.
    # n0s0: float = 8.0e5
    snow_n0: float = 8.0e5
    #: exponent of mixing ratio in the collection equation where cloud or ice particles are rimed by graupel (exp=(3+b)/(1+beta), v=a D^b, m=alpha D^beta), see eqs. 5.152 to 5.154.
    #rimexp_g: float = 0.94878
    graupel_exp_rim: float = 0.94878
    #: exponent of mixing ratio in the graupel mean terminal velocity-mixing ratio relationship (exp=b/(1+beta)), see eq. 5.156.
    #expsedg: float = 0.217
    graupel_exp_sed: float = 0.217
    #: constant in the graupel mean terminal velocity-mixing ratio relationship, see eq. 5.156.
    #vz0g: float = 12.24
    graupel_sed0: float = 12.24
    #: initial crystal mass for cloud ice nucleation, see eq. 5.101
    #mi0: float = 1.0e-12
    ice_initial_mass: float = 1.0e-12
    #: maximum mass of cloud ice crystals to avoid too large ice crystals near melting point, see eq. 5.105
    #mimax: float = 1.0e-9
    ice_max_mass: float = 1.0e-9
    #: initial mass of snow crystals which is used in ice-ice autoconversion to snow particles, see eq. 5.108
    #msmin: float = 3.0e-9
    snow_min_mass: float = 3.0e-9
    #: Scaling factor [1/K] for temperature-dependent cloud ice sticking efficiency (only used when ice_autocon_sticking_efficiency_option=1), see eq. 5.163
    #ceff_fac: float = 3.5e-3
    stick_eff_fac: float = 3.5e-3
    #: Temperature at which cloud ice autoconversion starts (only used when ice_autocon_sticking_efficiency_option=1), see eq. 5.163
    tmin_iceautoconv: float = 188.15
    #: Reference length for distance from cloud top (Forbes 2012) (only used when do_reduced_icedeposition=True), see eq. 5.166
    dist_cldtop_ref: float = 500.0
    #: lower bound on snow/ice deposition reduction (only used when do_reduced_icedeposition=True), see eq. 5.166
    reduce_dep_ref: float = 0.1
    #: Howell factor in depositional growth equation, see eq. 5.71 and eqs. 5.103 & 5.104 (for ice? TODO (Chia Rui): check)
    #hw: float = 2.270603
    howwll_factor: float = 2.270603
    #: Collection efficiency for snow collecting cloud water, see eq. 5.113
    #ecs: float = 0.9
    snow_cloud_collection_eff: float = 0.9
    #: Exponent in the terminal velocity for snow, see unnumbered eq. (v = 25 D^0.5) below eq. 5.159
    #v1s: float = 0.50
    snow_exp_v: float = 0.50
    #: kinematic viscosity of air
    eta: float = 1.75e-5
    #: molecular diffusion coefficient for water vapour
    #dv: float = 2.22e-5
    diffusion_coeff_water_vapor: float = 2.22e-5
    #: thermal conductivity of dry air
    #lheat: float = 2.40e-2
    dry_air_latent_heat: float = 2.40e-2
    #: Exponent in the mass-diameter relation of snow particles, see eq. 5.159
    #bms: float = 2.000
    snow_exp_m: float = 2.000
    #: Formfactor in the mass-diameter relation of cloud ice, see eq. 5.90
    #ami: float = 130.0
    ice_m0: float = 130.0
    #: density of liquid water
    #rhow: float = 1.000e+3
    water_density: float = 1.000e+3
    #: specific heat of water vapor J, at constant pressure (Landolt-Bornstein)
    cp_v: float = 1850.0
    #: specific heat of ice
    ci: float = 2108.0
    # TODO Chia Rui: remove the constants below
    #x1o3: float = 1.0 / 3.0
    #x7o8: float = 7.0 / 8.0
    #x13o8: float = 13.0 / 8.0
    #x27o16: float = 27.0 / 16.0
    #x1o2: float = 1.0 / 2.0
    #x3o4: float = 0.75
    #x7o4: float = 7.0 / 4.0

    n0s1 = 13.5 * 5.65e5  # parameter in N0S(T)
    n0s2 = -0.107  # parameter in N0S(T), Field et al
    mma = (
    5.065339, -0.062659, -3.032362, 0.029469, -0.000285, 0.312550, 0.000204, 0.003199, 0.000000, -0.015952)
    mmb = (
    0.476221, -0.015896, 0.165977, 0.007468, -0.000141, 0.060366, 0.000079, 0.000594, 0.000000, -0.003577)

    #: temperature for het. nuc. of cloud ice
    # thet = 248.15
    heterogeneous_freeze_temperature = 248.15
    #: autoconversion coefficient (cloud water to rain)
    # ccau = 4.0e-4
    cloud2rain_autoconversion_coeff = 4.0e-4
    #: (15/32)*(PI**0.5)*(ECR/RHOW)*V0R*AR**(1/8) when Kessler (1969) is used for cloud-cloud autoconversion
    # cac = 1.72
    cac = 1.72
    #: constant in phi-function for Seifert-Beheng (2001) autoconversion
    kphi1 = 6.00e+02
    #: exponent in phi-function for Seifert-Beheng (2001) autoconversion
    kphi2 = 0.68e+00
    #: exponent in phi-function for Seifert-Beheng (2001) accretion
    kphi3 = 5.00e-05
    #: kernel coeff for Seifert-Beheng (2001) autoconversion
    kcau = 9.44e+09
    #: kernel coeff for Seifert-Beheng (2001) accretion
    kcac = 5.25e+00
    #: gamma exponent for cloud distribution in Seifert-Beheng (2001) autoconverssion
    cnue = 2.00e+00
    #: separating mass between cloud and rain in Seifert-Beheng (2001) autoconverssion
    xstar = 2.60e-10

    #: = b1
    c1es = 610.78
    #: = b2w
    c3les = 17.269
    #: = b2i
    c3ies = 21.875
    #: = b4w
    c4les = 35.86
    #: = b4i
    c4ies = 7.66

    crim_g = 4.43  # coefficient for graupel riming
    csg = 0.5  # coefficient for snow-graupel conversion by riming
    cagg_g = 2.46
    ciau = 1.0e-3  # autoconversion coefficient (cloud ice to snow)
    msmin = 3.0e-9  # initial mass of snow crystals
    cicri = 1.72  # (15/32)*(PI**0.5)*(EIR/RHOW)*V0R*AR**(1/8)
    crcri = 1.24e-3  # (PI/24)*EIR*V0R*Gamma(6.5)*AR**(-5/8)
    asmel = 2.95e3  # DIFF*LH_v*RHO/LHEAT
    tcrit = 3339.5  # factor in calculation of critical temperature

    qc0 = 0.0  # param_qc0
    qi0 = 0.0  # param_qi0

    # TODO (Chia Rui): move parameters below to common/constants? But they need to be declared under FrozenNameSpace to be used in the big graupel scan operator.
    #: Latent heat of vaporisation for water [J/kg]
    alv = 2.5008e6
    #: Latent heat of sublimation for water [J/kg]
    als = 2.8345e6
    #: Melting temperature of ice/snow [K]
    tmelt = 273.15
    #: Triple point of water at 611hPa [K]
    t3 = 273.16

    #: Gas constant of dry air [J/K/kg]
    rd = 287.04
    #: Specific heat of dry air at constant pressure [J/K/kg]
    cpd = 1004.64
    #: cp_d / cp_l - 1
    rcpl = 3.1733

    #: Gas constant of water vapor [J/K/kg] """
    rv = 461.51
    #: Specific heat of water vapour at constant pressure [J/K/kg]
    cpv = 1869.46
    #: Specific heat of water vapour at constant volume [J/K/kg]
    cvv = cpv - rv

    def __post_init__(self):
        ccsdep = 0.26 * gamma_fct((self.snow_exp_v + 5.0) / 2.0) * np.sqrt(
            1.0 / self.eta)
        _ccsvxp = -(self.snow_exp_v / (self.snow_exp_m + 1.0) + 1.0)
        ccsvxp = _ccsvxp + 1.0
        ccslam = self.snow_m0 * gamma_fct(self.snow_exp_m + 1.0)
        ccslxp = 1.0 / (self.snow_exp_m + 1.0)
        ccswxp = self.snow_exp_v * ccslxp
        ccsaxp = -(self.snow_exp_v + 3.0)
        ccsdxp = -(self.snow_exp_v + 1.0) / 2.0
        ccshi1 = self.als * self.als / (self.dry_air_latent_heat * self.rv)
        ccdvtp = 2.22e-5 * self.tmelt ** (-1.94) * 101325.0
        ccidep = 4.0 * self.ice_m0 ** (-1.0 / 3.0)
        pvsw0 = fpvsw(self.tmelt)  # sat. vap. pressure for t = t0
        #log_10 = np.log(10.0)  # logarithm of 10
        ccswxp_ln1o2 = np.exp(icon_graupel_params.ccswxp * np.log(0.5))

        #: latent heat of fusion for water [J/kg]
        alf = self.als - self.alv
        #: Specific heat capacity of liquid water
        clw = (self.rcpl + 1.0) * self.cpd

        #: Specific heat of dry air at constant volume [J/K/kg]
        cvd = self.cpd - self.rd
        #: [K*kg/J]
        rcpd = 1.0 / self.cpd
        #: [K*kg/J]"""
        rcvd = 1.0 / cvd

        c2es = self.c1es * self.rd / self.rv
        #: = b234w
        c5les = self.c3les * (self.tmelt - self.c4les)
        #: = b234i
        c5ies = self.c3ies * (self.tmelt - self.c4ies)
        c5alvcp = c5les * self.alv / self.cpd
        c5alscp = c5ies * self.als / self.cpd
        alvdcp = self.alv / self.cpd
        alsdcp = self.als / self.cpd

        object.__setattr__(
            self,
            "ccsdep",
            ccsdep,
        )
        object.__setattr__(
            self,
            "ccsvxp",
            ccsvxp,
        )
        object.__setattr__(
            self,
            "ccslam",
            ccslam,
        )
        object.__setattr__(
            self,
            "ccslxp",
            ccslxp,
        )
        object.__setattr__(
            self,
            "ccswxp",
            ccswxp,
        )
        object.__setattr__(
            self,
            "ccsaxp",
            ccsaxp,
        )
        object.__setattr__(
            self,
            "ccsdxp",
            ccsdxp,
        )
        object.__setattr__(
            self,
            "ccshi1",
            ccshi1,
        )
        object.__setattr__(
            self,
            "ccdvtp",
            ccdvtp,
        )
        object.__setattr__(
            self,
            "ccidep",
            ccidep,
        )
        object.__setattr__(
            self,
            "pvsw0",
            pvsw0,
        )
        object.__setattr__(
            self,
            "ccswxp_ln1o2",
            ccswxp_ln1o2,
        )
        object.__setattr__(
            self,
            "alf",
            alf,
        )
        object.__setattr__(
            self,
            "clw",
            clw,
        )
        object.__setattr__(
            self,
            "cvd",
            cvd,
        )
        object.__setattr__(
            self,
            "rcpd",
            rcpd,
        )
        object.__setattr__(
            self,
            "rcvd",
            rcvd,
        )
        object.__setattr__(
            self,
            "c2es",
            c2es,
        )
        object.__setattr__(
            self,
            "c5les",
            c5les,
        )
        object.__setattr__(
            self,
            "c5ies",
            c5ies,
        )
        object.__setattr__(
            self,
            "c5alvcp",
            c5alvcp,
        )
        object.__setattr__(
            self,
            "c5alscp",
            c5alscp,
        )
        object.__setattr__(
            self,
            "alvdcp",
            alvdcp,
        )
        object.__setattr__(
            self,
            "alsdcp",
            alsdcp,
        )


icon_graupel_params: Final = FrozenNamespace(**vars(SingleMomentSixClassIconGraupelParams))


# Statement functions
# -------------------

def fpvsw(
   ztx: float
   ) -> float:
   # Return saturation vapour pressure over water from temperature
   return icon_graupel_params.GrFuncConst_c1es * np.exp(
      icon_graupel_params.GrFuncConst_c3les * (ztx - icon_graupel_params.tmelt) / (ztx - icon_graupel_params.GrFuncConst_c4les)
   )


# Field operation
# -------------------
@field_operator
def _fxna(
   ztx: gtx.float64
   ) -> gtx.float64:
   # Return number of activate ice crystals from temperature
   return 1.0e2 * exp(0.2 * (icon_graupel_params.tmelt - ztx))


@field_operator
def _fxna_cooper(
   ztx: gtx.float64
   ) -> gtx.float64:
   # Return number of activate ice crystals from temperature

   # Method: Cooper (1986) used by Greg Thompson(2008)

   return 5.0 * exp(0.304 * (icon_graupel_params.tmelt - ztx))


@field_operator
def latent_heat_vaporization(
   input_t: float64
) -> float64:
   '''Return latent heat of vaporization.

   Computed as internal energy and taking into account Kirchoff's relations
   '''
   # specific heat of water vapor at constant pressure (Landolt-Bornstein)
   #cp_v = 1850.0

   return (
      icon_graupel_params.alv
      + (icon_graupel_params.GrConst_cp_v - icon_graupel_params.clw) * (input_t - icon_graupel_params.tmelt)
      - icon_graupel_params.rv * input_t
   )

@field_operator
def latent_heat_sublimation(
   input_t: gtx.float64
) -> gtx.float64:

   #-------------------------------------------------------------------------------
   #>
   #! Description:
   #!   Latent heat of sublimation as internal energy and taking into account
   #!   Kirchoff's relations
   #-------------------------------------------------------------------------------

   # specific heat of water vapor at constant pressure (Landolt-Bornstein)
   #cp_v = 1850.0
   # specific heat of ice
   #ci = 2108.0

   return(
      icon_graupel_params.als + (icon_graupel_params.GrConst_cp_v - icon_graupel_params.GrConst_ci) * (input_t - icon_graupel_params.tmelt) - icon_graupel_params.rv * input_t
   )


@field_operator
def sat_pres_water(
   input_t: gtx.float64
) -> gtx.float64:

   # Tetens formula
   return (
      icon_graupel_params.GrFuncConst_c1es * exp( icon_graupel_params.GrFuncConst_c3les * (input_t - icon_graupel_params.tmelt) / (input_t - icon_graupel_params.GrFuncConst_c4les) )
   )


@field_operator
def sat_pres_ice(
   input_t: gtx.float64
) -> gtx.float64:

   # Tetens formula
   return (
      icon_graupel_params.GrFuncConst_c1es * exp( icon_graupel_params.GrFuncConst_c3ies * (input_t - icon_graupel_params.tmelt) / (input_t - icon_graupel_params.GrFuncConst_c4ies) )
   )

@field_operator
def sat_pres_water_murphykoop(
   input_t: gtx.float64
) -> gtx.float64:

  # Eq. (10) of Murphy and Koop (2005)
  return (
     exp( 54.842763 - 6763.22 / input_t - 4.210 * log(input_t) + 0.000367 * input_t  + tanh(0.0415 * (input_t - 218.8)) * (53.878 - 1331.22 / input_t - 9.44523 * log(input_t) + 0.014025 * input_t) )
  )

@field_operator
def sat_pres_ice_murphykoop(
   input_t: gtx.float64
) -> gtx.float64:

  # Eq. (7) of Murphy and Koop (2005)
  return (
     exp(9.550426 - 5723.265 / input_t + 3.53068 * log(input_t) - 0.00728332 * input_t )
  )

@field_operator
def TV(
   input_t: gtx.float64
) -> gtx.float64:

   # Tetens formula
   return (
      icon_graupel_params.GrFuncConst_c1es * exp( icon_graupel_params.GrFuncConst_c3les * (input_t - icon_graupel_params.tmelt) / (input_t - icon_graupel_params.GrFuncConst_c4les) )
   )


@dataclasses.dataclass
class MetricStateIconGraupel:
    ddqz_z_full: gtx.Field[[CellDim, KDim], float]


class SingleMomentSixClassIconGraupel:

    def __init__(
        self,
        config: SingleMomentSixClassIconGraupelConfig,
        grid: Optional[icon_grid.IconGrid],
        metric_state: Optional[MetricStateIconGraupel],
        vertical_params: Optional[v_grid.VerticalGridParams]
    ):
        self.config = config
        self._initialize_configurable_parameters()
        self.grid = grid
        self.metric_state = metric_state
        self.vertical_params = vertical_params

        self._initialize_local_fields()

    def _initialize_configurable_parameters(self):
        # TODO (Chia Rui): clean up the naming system of these parameters and categorize them in a tuple
        ccsrim = 0.25 * math_const.pi * icon_graupel_params.snow_cloud_collection_eff * self.config.snow_v0 * gamma_fct(
            icon_graupel_params.snow_exp_v + 3.0)
        ccsagg = 0.25 * math_const.pi * self.config.snow_v0 * gamma_fct(icon_graupel_params.snow_exp_v + 3.0)
        _ccsvxp = -(icon_graupel_params.snow_exp_v / (icon_graupel_params.snow_exp_m + 1.0) + 1.0)
        ccsvel = icon_graupel_params.snow_m0 * self.config.snow_v0 * gamma_fct(
            icon_graupel_params.snow_exp_m + icon_graupel_params.snow_exp_v + 1.0) * (
                     icon_graupel_params.snow_m0 * gamma_fct(icon_graupel_params.snow_exp_m + 1.0)) ** _ccsvxp
        if self.config.ice_concentration_option == 1:
            nimax = icon_graupel_params.nimax_Thom
            nimix = self._fxna_cooper(icon_graupel_params.threshold_freeze_temperature_mixedphase)
        elif self.config.ice_concentration_option == 2:
            nimax = self._fxna(icon_graupel_params.homogeneous_freeze_temperature)
            nimix = self._fxna(icon_graupel_params.threshold_freeze_temperature_mixedphase)
        else:
            raise NotImplementedError("ice_concentration_option can only be 1 or 2.")

        _n0r = 8.0e6 * np.exp(3.2 * self.config.rain_mu) * (0.01) ** (
            -self.config.rain_mu)  # empirical relation adapted from Ulbrich (1983)
        _n0r = _n0r * self.config.rain_n0  # apply tuning factor to zn0r variable
        _ar = math_const.pi * icon_graupel_params.water_density / 6.0 * _n0r * gamma_fct(
            self.config.rain_mu + 4.0)  # pre-factor

        vzxp = 0.5 / (self.config.rain_mu + 4.0)
        vz0r = 130.0 * gamma_fct(self.config.rain_mu + 4.5) / gamma_fct(self.config.rain_mu + 4.0) * _ar ** (
            -vzxp)

        cevxp = (self.config.rain_mu + 2.0) / (self.config.rain_mu + 4.0)
        cev = 2.0 * math_const.pi * icon_graupel_params.diffusion_coeff_water_vapor / icon_graupel_params.howwll_factor * _n0r * _ar ** (
            -cevxp) * gamma_fct(self.config.rain_mu + 2.0)
        bevxp = (2.0 * self.config.rain_mu + 5.5) / (2.0 * self.config.rain_mu + 8.0) - cevxp
        bev = 0.26 * np.sqrt(icon_graupel_params.ref_air_density * 130.0 / icon_graupel_params.eta) * _ar ** (
            -bevxp) * gamma_fct((2.0 * self.config.rain_mu + 5.5) / 2.0) / gamma_fct(self.config.rain_mu + 2.0)

        # Precomputations for optimization
        vzxp_ln1o2 = np.exp(vzxp * np.log(0.5))
        bvi_ln1o2 = np.exp(icon_graupel_params.ice_exp_v * np.log(0.5))
        expsedg_ln1o2 = np.exp(icon_graupel_params.graupel_exp_sed * np.log(0.5))

        self.ccs = (ccsrim, ccsagg, ccsvel)
        self.nimax = nimax
        self.nimix = nimix
        self.rain_vel_coef = (vzxp, vz0r, cevxp, cev, bevxp, bev)
        self.sef_dens_factor_coef = (vzxp_ln1o2, bvi_ln1o2, expsedg_ln1o2)

    def _initialize_local_fields(self):
        self.qnc = _allocate(CellDim, KDim, grid=self.grid)
        # TODO (Chia Rui): remove these tendency terms when physics inteface infrastructure is ready
        self.temperature_tendency = _allocate(CellDim, KDim, grid=self.grid)
        self.qv_tendency = _allocate(CellDim, KDim, grid=self.grid)
        self.qc_tendency = _allocate(CellDim, KDim, grid=self.grid)
        self.qi_tendency = _allocate(CellDim, KDim, grid=self.grid)
        self.qr_tendency = _allocate(CellDim, KDim, grid=self.grid)
        self.qs_tendency = _allocate(CellDim, KDim, grid=self.grid)
        self.qg_tendency = _allocate(CellDim, KDim, grid=self.grid)
        self.rhoqrv_old_kup = _allocate(CellDim, KDim, grid=self.grid)
        self.rhoqsv_old_kup = _allocate(CellDim, KDim, grid=self.grid)
        self.rhoqgv_old_kup = _allocate(CellDim, KDim, grid=self.grid)
        self.rhoqiv_old_kup = _allocate(CellDim, KDim, grid=self.grid)
        self.vnew_r = _allocate(CellDim, KDim, grid=self.grid)
        self.vnew_s = _allocate(CellDim, KDim, grid=self.grid)
        self.vnew_g = _allocate(CellDim, KDim, grid=self.grid)
        self.vnew_i = _allocate(CellDim, KDim, grid=self.grid)

    @classmethod
    def _fxna(
        self,
        ztx: float
    ) -> float:
        # Return number of activate ice crystals from temperature
        return 1.0e2 * np.exp(0.2 * (icon_graupel_params.tmelt - ztx))

    @classmethod
    def _fxna_cooper(
        self,
        ztx: float
    ) -> float:
        # Return number of activate ice crystals from temperature

        # Method: Cooper (1986) used by Greg Thompson(2008)

        return 5.0 * np.exp(0.304 * (icon_graupel_params.tmelt - ztx))

    def __str__(self):
        # TODO (Chia Rui): Print out the configuration and derived empirical parameters
        pass

    def run(
        self,
        dtime: float,
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
            dtime,
            self.config.is_isochoric,
            self.config.use_constant_water_heat_capacity,
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


@scan_operator(
    axis=KDim,
    forward=True,
    init=(
        0.0,  # temperature
        0.0,  # qv
        0.0,  # qc
        0.0,  # qi
        0.0,  # qr
        0.0,  # qs
        0.0,  # qg
        0.0,  # rhoqrV
        0.0,  # rhoqsV
        0.0,  # rhoqgV
        0.0,  # rhoqiV
        0.0,  # newV_r
        0.0,  # newV_s
        0.0,  # newV_g
        0.0,  # newV_i
        0.0,  # cloud top distance
        0.0,  # density
        0.0,  # density factor
        0.0,  # density factor for ice
        0.0,  # snow intercept parameter
        0.0,  # saturation pressure
        0     # k level
    ),
)
def _icon_graupel_scan(
    state_kup: tuple[
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        int32
    ],
    # Grid information
    kstart_moist: int32, # k starting level
    kend: int32,     # k bottom level
    dt: float64,     # time step
    dz: float64,     # level thickness
    # Prognostics
    temperature: float64,
    pres: float64,
    rho: float64,
    qv: float64,
    qc: float64,
    qi: float64,
    qr: float64,
    qs: float64,
    qg: float64,
    # Number Densities
    qnc: float64,  # originally 2D Field, now 3D Field
    # Option Switches
    is_isochoric: bool,
    ithermo_water: int32
    ):

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
        rhoqrV_old_kup,
        rhoqsV_old_kup,
        rhoqgV_old_kup,
        rhoqiV_old_kup,
        Vnew_r,
        Vnew_s,
        Vnew_g,
        Vnew_i,
        dist_cldtop,
        rho_kup,
        Crho1o2_kup,
        Crhofac_qi_kup,
        Cvz0s_kup,
        qvsw_kup,
        k_lev
    ) = state_kup

    # ------------------------------------------------------------------------------
    #  Section 1: Initial setting of physical constants
    # ------------------------------------------------------------------------------


    if ( k_lev < kstart_moist ):
        # tracing current k level
        k_lev = k_lev + int32(1)
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
    Cheat_cap_r = icon_graupel_params.rcvd if is_isochoric else icon_graupel_params.rcpd

    # timestep for calculations
    Cdtr  = 1.0 / dt

    # Latent heats
    # Default themodynamic is constant latent heat
    # tg = make_normalized(t(iv,k))
    # Calculate Latent heats if necessary
    # function called: CLHv = latent_heat_vaporization(temperature)
    # function called: CLHs = latent_heat_sublimation(temperature)
    CLHv = icon_graupel_params.alv + (icon_graupel_params.GrConst_cp_v - icon_graupel_params.clw) * (temperature - icon_graupel_params.tmelt) - icon_graupel_params.rv * temperature if (ithermo_water != int32(0)) else icon_graupel_params.alv
    CLHs = icon_graupel_params.als + (icon_graupel_params.GrConst_cp_v - icon_graupel_params.GrConst_ci) * (temperature - icon_graupel_params.tmelt) - icon_graupel_params.rv * temperature if (ithermo_water != int32(0)) else icon_graupel_params.als

    #----------------------------------------------------------------------------
    # Section 2: Check for existence of rain and snow
    #            Initialize microphysics and sedimentation scheme
    #----------------------------------------------------------------------------

    # initialization of source terms (all set to zero)
    Szdep_v2i = 0.0 # vapor   -> ice,     ice vapor deposition
    Szsub_v2i = 0.0 # vapor   -> ice,     ice vapor sublimation
    Sidep_v2i = 0.0 # vapor   -> ice,     ice vapor net deposition
    #Ssdpc_v2s = 0.0 # vapor   -> snow,    snow vapor deposition below freezing temperature
    #Sgdpc_v2g = 0.0 # vapor   -> graupel, graupel vapor deposition below freezing temperature
    #Ssdph_v2s = 0.0 # vapor   -> snow,    snow vapor deposition above freezing temperature
    #Sgdph_v2g = 0.0 # vapor   -> graupel, graupel vapor deposition above freezing temperature
    Ssdep_v2s = 0.0 # vapor   -> snow,    snow vapor deposition
    Sgdep_v2g = 0.0 # vapor   -> graupel, graupel vapor deposition
    #Sdnuc_v2i = 0.0 # vapor   -> ice,     low temperature heterogeneous ice deposition nucleation
    #Scnuc_v2i = 0.0 # vapor   -> ice,     incloud ice nucleation
    Snucl_v2i = 0.0 # vapor   -> ice,     ice nucleation
    Sconr_v2r = 0.0 # vapor   -> rain,    rain condensation on melting snow/graupel
    Scaut_c2r = 0.0 # cloud   -> rain,    cloud autoconversion into rain
    Scfrz_c2i = 0.0 # cloud   -> ice,     cloud freezing
    Scacr_c2r = 0.0 # cloud   -> rain,    rain-cloud accretion
    Sshed_c2r = 0.0 # cloud   -> rain,    rain shedding from riming above freezing
    Srims_c2s = 0.0 # cloud   -> snow,    snow riming
    Srimg_c2g = 0.0 # cloud   -> graupel, graupel riming
    Simlt_i2c = 0.0 # ice     -> cloud,   ice melting
    Sicri_i2g = 0.0 # ice     -> graupel, ice loss in rain-ice accretion
    Sdaut_i2s = 0.0 # ice     -> snow,    ice vapor depositional autoconversion into snow
    Saggs_i2s = 0.0 # ice     -> snow,    snow-ice aggregation
    Saggg_i2g = 0.0 # ice     -> graupel, graupel-ice aggregation
    Siaut_i2s = 0.0 # ice     -> snow,    ice autoconversion into snow
    Srcri_r2g = 0.0 # rain    -> graupel, rain loss in rain-ice accretion
    #Scrfr_r2g = 0.0 # rain    -> graupel, rain freezing in clouds
    #Ssrfr_r2g = 0.0 # rain    -> graupel, rain freezing in clear sky
    Srfrz_r2g = 0.0 # rain    -> graupel, rain freezing
    Sevap_r2v = 0.0 # rain    -> vapor,   rain evaporation
    Ssmlt_s2r = 0.0 # snow    -> rain,    snow melting
    Scosg_s2g = 0.0 # snow    -> graupel, snow autoconversion into graupel
    Sgmlt_g2r = 0.0 # graupel -> rain,    graupel melting

    reduce_dep = 1.0 # FR: Reduction coeff. for dep. growth of rain and ice

    #----------------------------------------------------------------------------
    # 2.1: Preparations for computations and to check the different conditions
    #----------------------------------------------------------------------------

    #qrg  = make_normalized(qr(iv,k))
    #qsg  = make_normalized(qs(iv,k))
    #qgg  = make_normalized(qg(iv,k))
    #qvg  = make_normalized(qv(iv,k))
    #qcg  = make_normalized(qc(iv,k))
    #qig  = make_normalized(qi(iv,k))
    #tg   = t(iv,k)
    #ppg  = p(iv,k)
    #rhog = rho(iv,k)

    #..for density correction of fall speeds
    C1orho     = 1.0 / rho
    Chlp       = log(icon_graupel_params.GrConst_rho0 * C1orho)
    Crho1o2    = exp(Chlp * icon_graupel_params.GrConst_x1o2)
    Crhofac_qi = exp(Chlp * icon_graupel_params.GrConst_icesedi_exp)

    # NOT WORKING, NOT SURE WHY THIS WILL LEAD TO DIVISION BY ZERO
    #if (rho_kup == 0.0):
    # Crho1o2_kup = 1.0
    # Crhofac_qi_kup = 1.0
    #else:
    #    C1orho_kup     = 1.0 / rho_kup
    #    Chlp_kup       = log(icon_graupel_params.GrConst_rho0 * C1orho_kup)
    #    Crho1o2_kup    = exp(Chlp_kup * icon_graupel_params.GrConst_x1o2)
    #    Crhofac_qi_kup = exp(Chlp_kup * icon_graupel_params.GrConst_icesedi_exp)

    rhoqr = qr * rho
    rhoqs = qs * rho
    rhoqg = qg * rho
    rhoqi = qi * rho

    llqr = True if (rhoqr > icon_graupel_params.GrConst_qmin) else False
    llqs = True if (rhoqs > icon_graupel_params.GrConst_qmin) else False
    llqg = True if (rhoqg > icon_graupel_params.GrConst_qmin) else False
    llqi = True if (rhoqi > icon_graupel_params.GrConst_qmin) else False

    Cdtdh = 0.5 * dt / dz

    #-------------------------------------------------------------------------
    # qs_prepare:
    #-------------------------------------------------------------------------
    if (llqs):
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

        if ( icon_graupel_params.GrConst_isnow_n0temp == 1 ):
            # Calculate n0s using the temperature-dependent
            # formula of Field et al. (2005)
            local_tc = temperature - icon_graupel_params.tmelt
            local_tc = minimum( local_tc , 0.0 )
            local_tc = maximum( local_tc , -40.0 )
            Cn0s = icon_graupel_params.n0s1 * exp(icon_graupel_params.n0s2 * local_tc)
            Cn0s = minimum( Cn0s , 1.0e9 )
            Cn0s = maximum( Cn0s , 1.0e6 )

        elif ( icon_graupel_params.GrConst_isnow_n0temp == 2 ):
            # Calculate n0s using the temperature-dependent moment
            # relations of Field et al. (2005)
            local_tc = temperature - icon_graupel_params.tmelt
            local_tc = minimum( local_tc , 0.0 )
            local_tc = maximum( local_tc , -40.0 )

            local_nnr  = 3.0
            local_hlp = (
                icon_graupel_params.GrFuncConst_mma[0] +
                icon_graupel_params.GrFuncConst_mma[1]*local_tc +
                icon_graupel_params.GrFuncConst_mma[2]*local_nnr +
                icon_graupel_params.GrFuncConst_mma[3]*local_tc*local_nnr +
                icon_graupel_params.GrFuncConst_mma[4]*local_tc**2.0 +
                icon_graupel_params.GrFuncConst_mma[5]*local_nnr**2.0 +
                icon_graupel_params.GrFuncConst_mma[6]*local_tc**2.0*local_nnr +
                icon_graupel_params.GrFuncConst_mma[7]*local_tc*local_nnr**2.0 +
                icon_graupel_params.GrFuncConst_mma[8]*local_tc**3.0 +
                icon_graupel_params.GrFuncConst_mma[9]*local_nnr**3.0
            )
            local_alf = exp(local_hlp*icon_graupel_params.GrConst_log_10) # 10.0_wp**hlp
            local_bet = (
                icon_graupel_params.GrFuncConst_mmb[0] +
                icon_graupel_params.GrFuncConst_mmb[1]*local_tc +
                icon_graupel_params.GrFuncConst_mmb[2]*local_nnr +
                icon_graupel_params.GrFuncConst_mmb[3]*local_tc*local_nnr +
                icon_graupel_params.GrFuncConst_mmb[4]*local_tc**2.0 +
                icon_graupel_params.GrFuncConst_mmb[5]*local_nnr**2.0 +
                icon_graupel_params.GrFuncConst_mmb[6]*local_tc**2.0*local_nnr +
                icon_graupel_params.GrFuncConst_mmb[7]*local_tc*local_nnr**2.0 +
                icon_graupel_params.GrFuncConst_mmb[8]*local_tc**3.0 +
                icon_graupel_params.GrFuncConst_mmb[9]*local_nnr**3.0
            )

            # Here is the exponent bms=2.0 hardwired# not ideal# (Uli Blahak)
            local_m2s = qs * rho / icon_graupel_params.GrConst_ams  # UB rho added as bugfix
            local_m3s = local_alf * exp(local_bet * log(local_m2s))

            local_hlp = icon_graupel_params.GrFuncConst_n0s1 * exp(icon_graupel_params.GrFuncConst_n0s2 * local_tc)
            Cn0s = 13.50 * local_m2s * (local_m2s / local_m3s) ** 3.0
            Cn0s = maximum(Cn0s, 0.5 * local_hlp)
            Cn0s = minimum(Cn0s, 1.0e2 * local_hlp)
            Cn0s = minimum(Cn0s, 1.0e9)
            Cn0s = maximum(Cn0s, 1.0e6)

        else:
            Cn0s = icon_graupel_params.GrConst_n0s0

        # compute integration factor for terminal velocity
        Cvz0s = icon_graupel_params.GrConst_ccsvel * exp(icon_graupel_params.GrConst_ccsvxp * log(Cn0s))
        # compute constants for riming, aggregation, and deposition processes for snow
        Crim = icon_graupel_params.GrConst_ccsrim * Cn0s
        Cagg = icon_graupel_params.GrConst_ccsagg * Cn0s
        Cbsdep = icon_graupel_params.GrConst_ccsdep * sqrt(icon_graupel_params.GrConst_v0snow)

    else:
        Cn0s = icon_graupel_params.GrConst_n0s0
        Cvz0s = 0.0
        Crim = 0.0
        Cagg = 0.0
        Cbsdep = 0.0


    #----------------------------------------------------------------------------
    # 2.2: sedimentation fluxes
    #----------------------------------------------------------------------------

    rhoqrV_new_kup = qr_kup * rho_kup * Vnew_r
    rhoqsV_new_kup = qs_kup * rho_kup * Vnew_s
    rhoqgV_new_kup = qg_kup * rho_kup * Vnew_g
    rhoqiV_new_kup = qi_kup * rho_kup * Vnew_i
    if (rhoqrV_new_kup <= icon_graupel_params.GrConst_qmin): rhoqrV_new_kup = 0.0
    if (rhoqsV_new_kup <= icon_graupel_params.GrConst_qmin): rhoqsV_new_kup = 0.0
    if (rhoqgV_new_kup <= icon_graupel_params.GrConst_qmin): rhoqgV_new_kup = 0.0
    if (rhoqiV_new_kup <= icon_graupel_params.GrConst_qmin): rhoqiV_new_kup = 0.0

    rhoqr_intermediate = rhoqr / Cdtdh + rhoqrV_new_kup + rhoqrV_old_kup
    rhoqs_intermediate = rhoqs / Cdtdh + rhoqsV_new_kup + rhoqsV_old_kup
    rhoqg_intermediate = rhoqg / Cdtdh + rhoqgV_new_kup + rhoqgV_old_kup
    rhoqi_intermediate = rhoqi / Cdtdh + rhoqiV_new_kup + rhoqiV_old_kup

    #-------------------------------------------------------------------------
    # qs_sedi, qr_sedi, qg_sedi, qi_sedi:
    #-------------------------------------------------------------------------
    if ( k_lev > kstart_moist ):
        Vnew_s = Cvz0s_kup * exp(icon_graupel_params.GrConst_ccswxp * log((qs_kup + qs) * 0.5 * rho_kup)) * Crho1o2_kup if (qs_kup + qs > icon_graupel_params.GrConst_qmin) else 0.0
        Vnew_r = icon_graupel_params.GrConst_vz0r * exp(icon_graupel_params.GrConst_vzxp * log((qr_kup + qr) * 0.5 * rho_kup)) * Crho1o2_kup if (qr_kup + qr > icon_graupel_params.GrConst_qmin) else 0.0
        Vnew_g = icon_graupel_params.GrConst_vz0g * exp(icon_graupel_params.GrConst_expsedg * log((qg_kup + qg) * 0.5 * rho_kup)) * Crho1o2_kup if (qg_kup + qg > icon_graupel_params.GrConst_qmin) else 0.0
        Vnew_i = icon_graupel_params.GrConst_vz0i * exp(icon_graupel_params.GrConst_bvi * log((qi_kup + qi) * 0.5 * rho_kup)) * Crhofac_qi_kup if (qi_kup + qi > icon_graupel_params.GrConst_qmin) else 0.0

    if (llqs):
        terminal_velocity = Cvz0s * exp(icon_graupel_params.GrConst_ccswxp * log(rhoqs)) * Crho1o2
        # Prevent terminal fall speed of snow from being zero at the surface level
        if (is_surface): terminal_velocity = maximum(terminal_velocity, icon_graupel_params.GrConst_v_sedi_snow_min)

        rhoqsV = rhoqs * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if ( Vnew_s == 0.0 ): Vnew_s = terminal_velocity * icon_graupel_params.GrConst_ccswxp_ln1o2

    else:
        rhoqsV = 0.0

    if (llqr):
        terminal_velocity = icon_graupel_params.GrConst_vz0r * exp(icon_graupel_params.GrConst_vzxp * log(rhoqr)) * Crho1o2
        # Prevent terminal fall speed of snow from being zero at the surface level
        if (is_surface): terminal_velocity = maximum(terminal_velocity, icon_graupel_params.GrConst_v_sedi_rain_min)

        rhoqrV = rhoqr * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if ( Vnew_r == 0.0 ): Vnew_r = terminal_velocity * icon_graupel_params.GrConst_vzxp_ln1o2

    else:
        rhoqrV = 0.0

    if (llqg):
        terminal_velocity = icon_graupel_params.GrConst_vz0g * exp(icon_graupel_params.GrConst_expsedg * log(rhoqg)) * Crho1o2
        # Prevent terminal fall speed of snow from being zero at the surface level
        if (is_surface): terminal_velocity = maximum(terminal_velocity, icon_graupel_params.GrConst_v_sedi_graupel_min)

        rhoqgV = rhoqg * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if ( Vnew_g == 0.0 ): Vnew_g = terminal_velocity * icon_graupel_params.GrConst_expsedg_ln1o2

    else:
        rhoqgV = 0.0

    if (llqi):
        terminal_velocity = icon_graupel_params.GrConst_vz0i * exp(icon_graupel_params.GrConst_bvi * log(rhoqi)) * Crhofac_qi

        rhoqiV = rhoqi * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if (Vnew_i == 0.0): Vnew_i = terminal_velocity * icon_graupel_params.GrConst_bvi_ln1o2

    else:
        rhoqiV = 0.0

    # Prevent terminal fall speeds of precip hydrometeors from being zero at the surface level
    if (is_surface):
        Vnew_s = maximum(Vnew_s, icon_graupel_params.GrConst_v_sedi_snow_min)
        Vnew_r = maximum( Vnew_r, icon_graupel_params.GrConst_v_sedi_rain_min )
        Vnew_g = maximum(Vnew_g, icon_graupel_params.GrConst_v_sedi_graupel_min)

    #-------------------------------------------------------------------------
    # derive the intermediate density of hydrometeors, Eq. 5.21:
    #-------------------------------------------------------------------------


    # limit the precipitation flux at this k level such that mixing ratio won't go below zero
    rhoqrV   = minimum( rhoqrV , rhoqr_intermediate )
    rhoqsV   = minimum( rhoqsV , rhoqs_intermediate )
    rhoqgV   = minimum( rhoqgV , maximum(0.0 , rhoqg_intermediate) )
    rhoqiV   = minimum( rhoqiV , rhoqi_intermediate )

    rhoqr_intermediate = Cdtdh * (rhoqr_intermediate - rhoqrV)
    rhoqs_intermediate = Cdtdh * (rhoqs_intermediate - rhoqsV)
    rhoqg_intermediate = Cdtdh * (rhoqg_intermediate - rhoqgV)
    rhoqi_intermediate = Cdtdh * (rhoqi_intermediate - rhoqiV)

    Cimr = 1.0 / (1.0 + Vnew_r * Cdtdh)
    Cims = 1.0 / (1.0 + Vnew_s * Cdtdh)
    Cimg = 1.0 / (1.0 + Vnew_g * Cdtdh)
    Cimi = 1.0 / (1.0 + Vnew_i * Cdtdh)

    # intermediate values
    rhoqr = rhoqr_intermediate * Cimr
    rhoqs = rhoqs_intermediate * Cims
    rhoqg = rhoqg_intermediate * Cimg
    rhoqi = rhoqi_intermediate * Cimi


    #--------------------------------------------------------------------------
    # 2.3: Second part of preparations
    #--------------------------------------------------------------------------

    #FR old
    #   Csdep    = 3.2E-2
    #Csdep        = 3.367e-2
    #Cidep        = 1.3e-5
    #Cslam        = 1.0e10

    Cscmax = qc * Cdtr
    if ( icon_graupel_params.GrConst_lsuper_coolw ):
        # function called: Cnin = _fxna_cooper(temperature)
        Cnin = 5.0 * exp(0.304 * (icon_graupel_params.tmelt - temperature))
        Cnin = minimum( Cnin , icon_graupel_params.GrConst_nimax )
    else:
        # function called: Cnin = _fxna(temperature)
        Cnin = 1.0e2 * exp(0.2 * (icon_graupel_params.tmelt - temperature))
        Cnin = minimum( Cnin , icon_graupel_params.GrConst_nimax )
    Cmi = minimum( rho * qi / Cnin , icon_graupel_params.GrConst_mimax )
    Cmi = maximum( icon_graupel_params.GrConst_mi0 , Cmi )

    # function called: Cqvsw = sat_pres_water(temperature) / (rho * icon_graupel_params.rv * temperature)
    # function called: Cqvsi = sat_pres_ice(temperature) / (rho * icon_graupel_params.rv * temperature)
    Cqvsw = icon_graupel_params.GrFuncConst_c1es * exp( icon_graupel_params.GrFuncConst_c3les * (temperature - icon_graupel_params.tmelt) / (temperature - icon_graupel_params.GrFuncConst_c4les) ) / (rho * icon_graupel_params.rv * temperature)
    Cqvsi = icon_graupel_params.GrFuncConst_c1es * exp( icon_graupel_params.GrFuncConst_c3ies * (temperature - icon_graupel_params.tmelt) / (temperature - icon_graupel_params.GrFuncConst_c4ies) ) / (rho * icon_graupel_params.rv * temperature)
    llqr = True if (rhoqr > icon_graupel_params.GrConst_qmin) else False
    llqs = True if (rhoqs > icon_graupel_params.GrConst_qmin) else False
    llqg = True if (rhoqg > icon_graupel_params.GrConst_qmin) else False
    llqi = True if (qi > icon_graupel_params.GrConst_qmin) else False
    llqc = True if (qc > icon_graupel_params.GrConst_qmin) else False


    ##----------------------------------------------------------------------------
    ## 2.4: IF (llqr): ic1
    ##----------------------------------------------------------------------------

    if (llqr):
        Clnrhoqr = log(rhoqr)
        Csrmax   = rhoqr_intermediate / rho * Cdtr  # GZ: shifting this computation ahead of the IF condition changes results!
        if ( qi + qc > icon_graupel_params.GrConst_qmin ):
            Celn7o8qrk   = exp(icon_graupel_params.GrConst_x7o8   * Clnrhoqr)
        else:
            Celn7o8qrk = 0.0
        if ( temperature < icon_graupel_params.GrConst_trfrz ):
            Celn7o4qrk   = exp(icon_graupel_params.GrConst_x7o4   * Clnrhoqr) #FR new
            Celn27o16qrk = exp(icon_graupel_params.GrConst_x27o16 * Clnrhoqr)
        else:
            Celn7o4qrk = 0.0
            Celn27o16qrk = 0.0
        if (llqi):
            Celn13o8qrk  = exp(icon_graupel_params.GrConst_x13o8  * Clnrhoqr)
        else:
            Celn13o8qrk = 0.0
    else:
        Csrmax = 0.0
        Celn7o8qrk = 0.0
        Celn7o4qrk = 0.0
        Celn27o16qrk = 0.0
        Celn13o8qrk = 0.0

    ##----------------------------------------------------------------------------
    ## 2.5: IF (llqs): ic2
    ##----------------------------------------------------------------------------

    # ** GZ: the following computation differs substantially from the corresponding code in cloudice **
    if (llqs):
        Clnrhoqs = log(rhoqs)
        Cssmax   = rhoqs_intermediate / rho * Cdtr  # GZ: shifting this computation ahead of the IF condition changes results#
        if ( qi + qc > icon_graupel_params.GrConst_qmin ):
            Celn3o4qsk = exp(icon_graupel_params.GrConst_x3o4 * Clnrhoqs)
        else:
            Celn3o4qsk = 0.0
        Celn8qsk = exp(0.8 * Clnrhoqs)
    else:
        Cssmax = 0.0
        Celn3o4qsk = 0.0
        Celn8qsk = 0.0

    ##----------------------------------------------------------------------------
    ## 2.6: IF (llqg): ic3
    ##----------------------------------------------------------------------------

    if (llqg):
        Clnrhoqg = log(rhoqg)
        Csgmax   = rhoqg_intermediate / rho * Cdtr
        if ( qi + qc > icon_graupel_params.GrConst_qmin ):
            Celnrimexp_g = exp(icon_graupel_params.GrConst_rimexp_g * Clnrhoqg)
        else:
            Celnrimexp_g =0.0
        Celn6qgk = exp(0.6 * Clnrhoqg)
    else:
        Csgmax = 0.0
        Celnrimexp_g = 0.0
        Celn6qgk = 0.0

        ##----------------------------------------------------------------------------
    ## 2.7:  slope of snow PSD and coefficients for depositional growth (llqi,llqs)
    ##----------------------------------------------------------------------------

    if ( llqi | llqs ):
        Cdvtp  = icon_graupel_params.GrConst_ccdvtp * exp(1.94 * log(temperature)) / pres
        Chi    = icon_graupel_params.GrConst_ccshi1 * Cdvtp * rho * Cqvsi/(temperature * temperature)
        Chlp    = Cdvtp / (1.0 + Chi)
        Cidep = icon_graupel_params.GrConst_ccidep * Chlp

        if (llqs):
            Cslam = exp(icon_graupel_params.GrConst_ccslxp * log(icon_graupel_params.GrConst_ccslam * Cn0s / rhoqs ))
            Cslam = minimum( Cslam , 1.0e15 )
            Csdep = 4.0 * Cn0s * Chlp
        else:
            Cslam = 1.0e10
            Csdep = 3.367e-2
    else:
        Cidep = 1.3e-5
        Cslam = 1.0e10
        Csdep = 3.367e-2


    ##----------------------------------------------------------------------------
    ## 2.8: Deposition nucleation for low temperatures below a threshold (llqv)
    ##----------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    # Description:
    #   This subroutine computes the heterogeneous ice deposition nucleation rate.
    #
    #   ice nucleation rate = mi0 Ni / rho / dt, Eq. 5.101
    #   mi0 is the initial ice crystal mass
    #
    #------------------------------------------------------------------------------

    if ( (temperature < icon_graupel_params.GrFuncConst_thet) & (qv > 8.e-6) & (qi <= 0.0) & (qv > Cqvsi) ):
        Snucl_v2i = icon_graupel_params.GrConst_mi0 * C1orho * Cnin * Cdtr


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
    #   Method 1: iautocon = 0, Kessler (1969)
    #   Method 2: iautocon = 1, Seifert and beheng (2001)
    #
    #------------------------------------------------------------------------------

    # if there is cloud water and the temperature is above homogeneuous freezing temperature
    if ( llqc & (temperature > icon_graupel_params.GrConst_thn) ):

        if (icon_graupel_params.GrConst_iautocon == 0):
            # Kessler(1969) autoconversion rate
            Scaut_c2r = icon_graupel_params.GrFuncConst_ccau  * maximum( qc - icon_graupel_params.GrFuncConst_qc0 , 0.0 )
            Scacr_c2r = icon_graupel_params.GrFuncConst_cac * qc * Celn7o8qrk

        elif (icon_graupel_params.GrConst_iautocon == 1):
            # Seifert and Beheng (2001) autoconversion rate
            local_const = icon_graupel_params.GrFuncConst_kcau / (20.0 * icon_graupel_params.GrFuncConst_xstar) * (icon_graupel_params.GrFuncConst_cnue + 2.0) * (icon_graupel_params.GrFuncConst_cnue + 4.0) / (icon_graupel_params.GrFuncConst_cnue + 1.0)**2.0

            # with constant cloud droplet number concentration qnc
            if ( qc > 1.0e-6 ):
                local_tau = minimum( 1.0 - qc / (qc + qr) , 0.9 )
                local_tau = maximum( local_tau , 1.e-30 )
                local_hlp  = exp(icon_graupel_params.GrFuncConst_kphi2 * log(local_tau))
                local_phi = icon_graupel_params.GrFuncConst_kphi1 * local_hlp * (1.0 - local_hlp)**3.0
                Scaut_c2r = local_const * qc * qc * qc * qc / (qnc * qnc) * (1.0 + local_phi / (1.0 - local_tau)**2.0)
                local_phi = (local_tau / (local_tau + icon_graupel_params.GrFuncConst_kphi3))**4.0
                Scacr_c2r = icon_graupel_params.GrFuncConst_kcac * qc * qr * local_phi


    ##----------------------------------------------------------------------------
    ## 3.2: Cloud and rain freezing in clouds
    ##----------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    # Description:
    #   This subroutine computes the freezing rate of rain in clouds.
    #
    #   Method 1: rain_freezing_mode = 2, Eq. 5.126, an approximation of combined immersion Eq. 5.80 and contact freezing Eq. 5.83 (ABANDONED)
    #   Method 2: rain_freezing_mode = 1, Eq. 5.168
    #
    #------------------------------------------------------------------------------

    # if there is cloud water, and the temperature is above homogeneuous freezing temperature
    if ( llqc ):
        if ( temperature > icon_graupel_params.GrConst_thn ):
            # Calculation of in-cloud rainwater freezing
            if ( llqr & (temperature < icon_graupel_params.GrConst_trfrz) & (qr > 0.1 * qc) ):
                if rain_freezing_mode == 1:
                    Srfrz_r2g = icon_graupel_params.GrConst_crfrz1 * ( exp(icon_graupel_params.GrConst_crfrz2 * (icon_graupel_params.GrConst_trfrz - temperature)) - 1.0 ) * Celn7o4qrk
                elif rain_freezing_mode == 2:
                    local_tfrzdiff = icon_graupel_params.GrConst_trfrz - temperature
                    Srfrz_r2g = icon_graupel_params.GrConst_crfrz * local_tfrzdiff * sqrt(local_tfrzdiff) * Celn27o16qrk
        else:
            # tg <= tg: ! hom. freezing of cloud and rain water
            Scfrz_c2i = Cscmax
            Srfrz_r2g = Csrmax


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
    if ( llqc & (temperature > icon_graupel_params.GrConst_thn) ):

        if ( llqs ):
            Srims_c2s = Crim * qc * exp(icon_graupel_params.GrConst_ccsaxp * log(Cslam))

        Srimg_c2g = icon_graupel_params.GrFuncConst_crim_g * qc * Celnrimexp_g

        if ( temperature >= icon_graupel_params.tmelt ):
            Sshed_c2r = Srims_c2s + Srimg_c2g
            Srims_c2s = 0.0
            Srimg_c2g = 0.0
        else:
            if ( qc >= icon_graupel_params.GrFuncConst_qc0 ):
                Scosg_s2g = icon_graupel_params.GrFuncConst_csg * qc * Celn3o4qsk


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
    if ( llqc & (temperature <= 267.15) & (qi <= icon_graupel_params.GrConst_qmin) ):
        if ( icon_graupel_params.GrConst_lsuper_coolw ):
            Snucl_v2i = icon_graupel_params.GrConst_mi0 * C1orho * Cnin * Cdtr
        else:
            Snucl_v2i = icon_graupel_params.GrConst_mi0 / rho * Cnin * Cdtr

    ##----------------------------------------------------------------------------
    ## 3.5: Reduced deposition in clouds
    ##----------------------------------------------------------------------------

    if ( icon_graupel_params.GrConst_lred_depgrow & llqc ):
        if ((k_lev > kstart_moist) & (k_lev < kend)):

            Cqcgk_1 = qi_kup + qs_kup + qg_kup

            # distance from cloud top
            if ( (qv_kup + qc_kup < qvsw_kup) & (Cqcgk_1 < icon_graupel_params.GrConst_qmin) ):
                # upper cloud layer
                dist_cldtop = 0.0  # reset distance to upper cloud layer
            else:
                dist_cldtop = dist_cldtop + dz

    if ( icon_graupel_params.GrConst_lred_depgrow & llqc ):
        if ((k_lev > kstart_moist) & (k_lev < kend)):
            # finalizing transfer rates in clouds and calculate depositional growth reduction
            # function called: Cnin_cooper = _fxna_cooper(temperature)
            Cnin_cooper = 5.0 * exp(0.304 * (icon_graupel_params.tmelt - temperature))
            Cnin_cooper = minimum(Cnin_cooper, icon_graupel_params.GrConst_nimax)
            Cfnuc = minimum(Cnin_cooper / icon_graupel_params.GrConst_nimix, 1.0)

            # with asymptotic behaviour dz -> 0 (xxx)
            #        reduce_dep = MIN(fnuc + (1.0_wp-fnuc)*(reduce_dep_ref + &
            #                             dist_cldtop(iv)/dist_cldtop_ref + &
            #                             (1.0_wp-reduce_dep_ref)*(zdh/dist_cldtop_ref)**4), 1.0_wp)

            # without asymptotic behaviour dz -> 0
            reduce_dep = Cfnuc + (1.0 - Cfnuc) * (icon_graupel_params.GrConst_reduce_dep_ref + dist_cldtop / icon_graupel_params.GrConst_dist_cldtop_ref)
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
    if ( (temperature <= icon_graupel_params.tmelt) & llqi ):

        # Change in sticking efficiency needed in case of cloud ice sedimentation
        # (based on Guenther Zaengls work)
        if ( icon_graupel_params.GrConst_lstickeff ):
            local_eff = minimum( exp(0.09 * (temperature - icon_graupel_params.tmelt)) , 1.0 )
            local_eff = maximum( local_eff , icon_graupel_params.GrConst_ceff_min )
            local_eff = maximum( local_eff , icon_graupel_params.GrConst_ceff_fac * (temperature - icon_graupel_params.GrConst_tmin_iceautoconv) )
        else: #original sticking efficiency of cloud ice
            local_eff = minimum( exp(0.09 * (temperature - icon_graupel_params.tmelt)) , 1.0 )
            local_eff = maximum( local_eff , 0.2 )

        local_nid = rho * qi / Cmi
        local_lnlogmi = log(Cmi)

        local_qvsidiff = qv - Cqvsi
        local_svmax = local_qvsidiff * Cdtr

        Saggs_i2s = local_eff * qi * Cagg * exp(icon_graupel_params.GrConst_ccsaxp * log(Cslam))
        Saggg_i2g = local_eff * qi * icon_graupel_params.GrFuncConst_cagg_g * Celnrimexp_g
        Siaut_i2s = local_eff * icon_graupel_params.GrFuncConst_ciau * maximum( qi - icon_graupel_params.GrFuncConst_qi0 , 0.0 )

        Sicri_i2g = icon_graupel_params.GrFuncConst_cicri * qi * Celn7o8qrk
        if (qs > 1.e-7):
            Srcri_r2g = icon_graupel_params.GrFuncConst_crcri * (qi / Cmi) * Celn13o8qrk


        local_iceTotalDeposition = Cidep * local_nid * exp(0.33 * local_lnlogmi) * local_qvsidiff
        Sidep_v2i = local_iceTotalDeposition
        # Szdep_v2i = 0.0
        # Szsub_v2i = 0.0

        # for sedimenting quantities the maximum
        # allowed depletion is determined by the predictor value.
        if (icon_graupel_params.GrConst_lsedi_ice):
            local_simax = rhoqi_intermediate * C1orho * Cdtr
        else:
            local_simax = qi * Cdtr

        if (local_iceTotalDeposition > 0.0):
            if (icon_graupel_params.GrConst_lred_depgrow):
                local_iceTotalDeposition = local_iceTotalDeposition * reduce_dep  # FR new: depositional growth reduction
            Szdep_v2i = minimum(local_iceTotalDeposition, local_svmax)
        elif (local_iceTotalDeposition < 0.0):
            Szsub_v2i = maximum(local_iceTotalDeposition, local_svmax)
            Szsub_v2i = - maximum(Szsub_v2i, -local_simax)

        local_lnlogmi = log(icon_graupel_params.GrFuncConst_msmin / Cmi)
        local_ztau = 1.5 * (exp(0.66 * local_lnlogmi) - 1.0)
        Sdaut_i2s = Szdep_v2i / local_ztau

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

    if ( llqi | llqs | llqg ):

        if ( temperature <= icon_graupel_params.tmelt ):

            local_qvsidiff = qv - Cqvsi
            local_svmax    = local_qvsidiff * Cdtr

            local_xfac = 1.0 + Cbsdep * exp(icon_graupel_params.GrConst_ccsdxp * log(Cslam))
            Ssdep_v2s = Csdep * local_xfac * local_qvsidiff / (Cslam + icon_graupel_params.GrConst_eps)**2.0
            #FR new: depositional growth reduction
            if ( (icon_graupel_params.GrConst_lred_depgrow) & (Ssdep_v2s > 0.0) ):
                Ssdep_v2s = Ssdep_v2s * reduce_dep

            # GZ: This limitation, which was missing in the original graupel scheme,
            # is crucial for numerical stability in the tropics!
            if ( Ssdep_v2s > 0.0 ):
                Ssdep_v2s = minimum( Ssdep_v2s , local_svmax - Szdep_v2i )
            # Suppress depositional growth of snow if the existing amount is too small for a
            # a meaningful distiction between cloud ice and snow
            if ( qs <= 1.e-7 ):
                Ssdep_v2s = minimum( Ssdep_v2s , 0.0 )
            # ** GZ: this numerical fit should be replaced with a physically more meaningful formulation **
            Sgdep_v2g = (
                0.398561 -
                0.00152398 * temperature +
                2554.99 / pres +
                2.6531e-7 * pres
            ) * local_qvsidiff * Celn6qgk


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

    if ( llqi | llqs | llqg ):

        if ( temperature > icon_graupel_params.tmelt ):

            # cloud ice melts instantaneously
            if ( icon_graupel_params.GrConst_lsedi_ice ):
                Simlt_i2c = rhoqi_intermediate * C1orho * Cdtr
            else:
                Simlt_i2c = qi * Cdtr

            local_qvsw0     = icon_graupel_params.GrConst_pvsw0 / (rho * icon_graupel_params.rv * icon_graupel_params.tmelt)
            local_qvsw0diff = qv - local_qvsw0

            # ** GZ: several numerical fits in this section should be replaced with physically more meaningful formulations **
            if ( temperature > icon_graupel_params.tmelt - icon_graupel_params.GrFuncConst_tcrit * local_qvsw0diff ):
                #calculate melting rate
                local_x1  = temperature - icon_graupel_params.tmelt + icon_graupel_params.GrFuncConst_asmel * local_qvsw0diff
                Ssmlt_s2r = (79.6863 / pres + 0.612654e-3) * local_x1 * Celn8qsk
                Ssmlt_s2r = minimum( Ssmlt_s2r , Cssmax )
                Sgmlt_g2r = (12.31698 / pres + 7.39441e-05) * local_x1 * Celn6qgk
                Sgmlt_g2r = minimum( Sgmlt_g2r , Csgmax )
                #deposition + melting, ice particle temperature: t0
                #calculation without howell-factor!
                Ssdep_v2s  = (31282.3 / pres + 0.241897) * local_qvsw0diff * Celn8qsk
                Sgdep_v2g  = (0.153907 - pres * 7.86703e-07) * local_qvsw0diff * Celn6qgk
                if ( local_qvsw0diff < 0.0 ):
                    #melting + evaporation of snow/graupel
                    Ssdep_v2s = maximum( -Cssmax , Ssdep_v2s )
                    Sgdep_v2g = maximum( -Csgmax , Sgdep_v2g )
                    #melt water evaporates
                    Ssmlt_s2r = Ssmlt_s2r + Ssdep_v2s
                    Sgmlt_g2r = Sgmlt_g2r + Sgdep_v2g
                    Ssmlt_s2r = maximum( Ssmlt_s2r , 0.0 )
                    Sgmlt_g2r = maximum( Sgmlt_g2r , 0.0 )
                else:
                    #deposition on snow/graupel is interpreted as increase
                    #in rain water ( qv --> qr, sconr)
                    #therefore,  sconr=(zssdep+zsgdep)
                    Sconr_v2r = Ssdep_v2s + Sgdep_v2g
                    Ssdep_v2s = 0.0
                    Sgdep_v2g = 0.0
            else:
                #if t<t_crit
                #no melting, only evaporation of snow/graupel
                #local_qvsw      = sat_pres_water(input_t) / (input_rho * icon_graupel_params.rv * input_t)
                #output_qvsw_kup = local_qvsw ! redundant in the original code?
                local_qvsidiff  = qv - Cqvsw
                Ssdep_v2s = (0.28003 - pres * 0.146293e-6) * local_qvsidiff * Celn8qsk
                Sgdep_v2g = (0.0418521 - pres * 4.7524e-8) * local_qvsidiff * Celn6qgk
                Ssdep_v2s = maximum( -Cssmax , Ssdep_v2s )
                Sgdep_v2g = maximum( -Csgmax , Sgdep_v2g )

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
    #          Method 1: lsuper_coolw = true, Eq. 5.80
    #          Method 2: lsuper_coolw = false, Eq. 5.126, an approximation of combined immersion Eq. 5.80 and contact freezing Eq. 5.83
    #
    #------------------------------------------------------------------------------

    if ( llqr & (qv + qc <= Cqvsw) ):

        local_lnqr = log(rhoqr)
        local_x1   = 1.0 + icon_graupel_params.GrConst_bev * exp(icon_graupel_params.GrConst_bevxp * local_lnqr)
        #sev  = zcev*zx1*(zqvsw - qvg) * EXP (zcevxp  * zlnqrk)
        # Limit evaporation rate in order to avoid overshoots towards supersaturation
        # the pre-factor approximates (esat(T_wb)-e)/(esat(T)-e) at temperatures between 0 degC and 30 degC
        local_temp_c  = temperature - icon_graupel_params.tmelt
        local_maxevap = (0.61 - 0.0163 * local_temp_c + 1.111e-4 * local_temp_c**2.0) * (Cqvsw - qv) / dt
        Sevap_r2v     = icon_graupel_params.GrConst_cev * local_x1 * (Cqvsw - qv) * exp(icon_graupel_params.GrConst_cevxp * local_lnqr)
        Sevap_r2v     = minimum( Sevap_r2v , local_maxevap )

        if ( temperature > icon_graupel_params.GrConst_thn ):
            # Calculation of below-cloud rainwater freezing
            if ( temperature < icon_graupel_params.GrConst_trfrz ):
                if (icon_graupel_params.GrConst_lsuper_coolw):
                    #FR new: reduced rain freezing rate
                    Srfrz_r2g = icon_graupel_params.GrConst_crfrz1 * (exp(icon_graupel_params.GrConst_crfrz2 * (icon_graupel_params.GrConst_trfrz - temperature)) - 1.0 ) * Celn7o4qrk
                else:
                    Srfrz_r2g = icon_graupel_params.GrConst_crfrz * sqrt( (icon_graupel_params.GrConst_trfrz - temperature)**3.0 ) * Celn27o16qrk
        else: # Hom. freezing of rain water
            Srfrz_r2g = Csrmax

    #--------------------------------------------------------------------------
    # Section 7: Calculate the total tendencies of the prognostic variables.
    #            Update the prognostic variables in the interior domain.
    #--------------------------------------------------------------------------

    # finalizing transfer rates in clouds and calculate depositional growth reduction
    if ( llqc & (temperature > icon_graupel_params.GrConst_thn)):
        # Check for maximum depletion of cloud water and adjust the
        # transfer rates accordingly
        Csum = Scaut_c2r + Scacr_c2r + Srims_c2s + Srimg_c2g + Sshed_c2r
        Ccorr = Cscmax / maximum(Cscmax, Csum)
        Scaut_c2r = Ccorr * Scaut_c2r
        Scacr_c2r = Ccorr * Scacr_c2r
        Srims_c2s = Ccorr * Srims_c2s
        Srimg_c2g = Ccorr * Srimg_c2g
        Sshed_c2r = Ccorr * Sshed_c2r
        Scosg_s2g = minimum(Scosg_s2g, Srims_c2s + Cssmax)

    if ( llqi | llqs | llqg ):
        if (temperature <= icon_graupel_params.tmelt):  # cold case

            Cqvsidiff = qv - Cqvsi
            if (icon_graupel_params.GrConst_lsedi_ice):
                Csimax = rhoqi_intermediate * C1orho * Cdtr
            else:
                Csimax = qi * Cdtr

            # Check for maximal depletion of cloud ice
            # No check is done for depositional autoconversion (sdau) because
            # this is a always a fraction of the gain rate due to
            # deposition (i.e the sum of this rates is always positive)
            Csum = Siaut_i2s + Saggs_i2s + Saggg_i2g + Sicri_i2g + Szsub_v2i
            Ccorr = 0.0
            if (Csimax > 0.0): Ccorr = Csimax / maximum(Csimax, Csum)
            Sidep_v2i = Szdep_v2i - Ccorr * Szsub_v2i
            Siaut_i2s = Ccorr * Siaut_i2s
            Saggs_i2s = Ccorr * Saggs_i2s
            Saggg_i2g = Ccorr * Saggg_i2g
            Sicri_i2g = Ccorr * Sicri_i2g
            if (Cqvsidiff < 0.0):
                Ssdep_v2s = maximum(Ssdep_v2s, -Cssmax)
                Sgdep_v2g = maximum(Sgdep_v2g, -Csgmax)

    Csum = Sevap_r2v + Srfrz_r2g + Srcri_r2g
    Ccorr = 1.0
    if (Csum > 0.0): Ccorr = Csrmax / maximum(Csrmax, Csum)
    Sevap_r2v = Ccorr * Sevap_r2v
    Srfrz_r2g = Ccorr * Srfrz_r2g
    Srcri_r2g = Ccorr * Srcri_r2g

    # limit snow depletion in order to avoid negative values of qs
    Ccorr = 1.0
    if (Ssdep_v2s <= 0.0):
        Csum = Ssmlt_s2r + Scosg_s2g - Ssdep_v2s
        if (Csum > 0.0): Ccorr = Cssmax / maximum(Cssmax, Csum)
        Ssmlt_s2r = Ccorr * Ssmlt_s2r
        Scosg_s2g = Ccorr * Scosg_s2g
        Ssdep_v2s = Ccorr * Ssdep_v2s
    else:
        Csum = Ssmlt_s2r + Scosg_s2g
        if (Csum > 0.0): Ccorr = Cssmax / maximum(Cssmax, Csum)
        Ssmlt_s2r = Ccorr * Ssmlt_s2r
        Scosg_s2g = Ccorr * Scosg_s2g


    Cqvt = Sevap_r2v - Sidep_v2i - Ssdep_v2s - Sgdep_v2g - Snucl_v2i - Sconr_v2r
    Cqct = Simlt_i2c - Scaut_c2r - Scfrz_c2i - Scacr_c2r - Sshed_c2r - Srims_c2s - Srimg_c2g
    Cqit = Snucl_v2i + Scfrz_c2i - Simlt_i2c - Sicri_i2g + Sidep_v2i - Sdaut_i2s - Saggs_i2s - Saggg_i2g - Siaut_i2s
    Cqrt = Scaut_c2r + Sshed_c2r + Scacr_c2r + Ssmlt_s2r + Sgmlt_g2r - Sevap_r2v - Srcri_r2g - Srfrz_r2g + Sconr_v2r
    Cqst = Siaut_i2s + Sdaut_i2s - Ssmlt_s2r + Srims_c2s + Ssdep_v2s + Saggs_i2s - Scosg_s2g
    Cqgt = Saggg_i2g - Sgmlt_g2r + Sicri_i2g + Srcri_r2g + Sgdep_v2g + Srfrz_r2g + Srimg_c2g + Scosg_s2g

    # First method:
    Ctt = Cheat_cap_r * ( CLHv * (Cqct + Cqrt) + CLHs * (Cqit + Cqst + Cqgt) )
    # Second method:
    #solid_phase_change = Snucl_v2i + Scfrz_c2i - Simlt_i2c + Sidep_v2i - Ssmlt_s2r + Srims_c2s - Sgmlt_g2r + Srcri_r2g + Sgdep_v2g + Srfrz_r2g + Srimg_c2g
    #liquid_phase_change = Simlt_i2c - Scfrz_c2i - Srims_c2s - Srimg_c2g + Ssmlt_s2r + Sgmlt_g2r - Sevap_r2v - Srcri_r2g - Srfrz_r2g + Sconr_v2r
    #Ctt = Cheat_cap_r * (liquid_phase_change * CLHv + solid_phase_change * CLHs)
    # Third method
    #v2s_phase_change = Snucl_v2i + Sidep_v2i + Sgdep_v2g
    #l2s_phase_change = Scfrz_c2i - Simlt_i2c - Ssmlt_s2r + Srims_c2s - Sgmlt_g2r + Srcri_r2g + Srfrz_r2g + Srimg_c2g
    #v2l_phase_change = Sconr_v2r - Sevap_r2v
    #Ctt = Cheat_cap_r * CLHs * (v2s_phase_change + l2s_phase_change) + Cheat_cap_r * CLHv * (v2l_phase_change - l2s_phase_change)

    # Update variables and add qi to qrs for water loading
    qi = maximum( 0.0 , (rhoqi_intermediate * C1orho + Cqit * dt) * Cimi ) if ( icon_graupel_params.GrConst_lsedi_ice ) else maximum( 0.0 , qi + Cqit * dt )
    qr = maximum( 0.0 , (rhoqr_intermediate * C1orho + Cqrt * dt) * Cimr )
    qs = maximum( 0.0 , (rhoqs_intermediate * C1orho + Cqst * dt) * Cims )
    qg = maximum( 0.0 , (rhoqg_intermediate * C1orho + Cqgt * dt) * Cimg )

    # Update of prognostic variables or tendencies
    temperature = temperature + Ctt * dt
    qv = maximum( 0.0 , qv + Cqvt * dt )
    qc = maximum( 0.0 , qc + Cqct * dt )

    # tracing current k level
    k_lev = k_lev + int32(1)


    return (
        temperature,
        qv,
        qc,
        qi,
        qr,
        qs,
        qg,
        rhoqrV,
        rhoqsV,
        rhoqgV,
        rhoqiV,
        Vnew_r,
        Vnew_s,
        Vnew_g,
        Vnew_i,
        dist_cldtop,
        rho,
        Crho1o2,
        Crhofac_qi,
        Cvz0s,
        Cqvsw,
        k_lev
    )

@field_operator
def _compute_icon_graupel_t_tendency(
    dt: float64,
    temperature_new: Field[[CellDim,KDim], float64],
    temperature_old: Field[[CellDim,KDim], float64]
) -> Field[[CellDim,KDim], float64]:
    Cdtr = 1.0 / dt
    return (temperature_new - temperature_old) * Cdtr

@field_operator
def _compute_icon_graupel_q_tendency(
    dt: float64,
    qv_new: Field[[CellDim, KDim], float64],
    qc_new: Field[[CellDim, KDim], float64],
    qi_new: Field[[CellDim, KDim], float64],
    qr_new: Field[[CellDim, KDim], float64],
    qs_new: Field[[CellDim, KDim], float64],
    qg_new: Field[[CellDim, KDim], float64],
    qv_old: Field[[CellDim, KDim], float64],
    qc_old: Field[[CellDim, KDim], float64],
    qi_old: Field[[CellDim, KDim], float64],
    qr_old: Field[[CellDim, KDim], float64],
    qs_old: Field[[CellDim, KDim], float64],
    qg_old: Field[[CellDim, KDim], float64]
) -> tuple[
    Field[[CellDim,KDim], float64],
    Field[[CellDim,KDim], float64],
    Field[[CellDim,KDim], float64],
    Field[[CellDim,KDim], float64],
    Field[[CellDim,KDim], float64],
    Field[[CellDim,KDim], float64]
]:
    Cdtr = 1.0 / dt
    return (
        maximum( -qv_old * Cdtr , (qv_new - qv_old) * Cdtr ),
        maximum( -qc_old * Cdtr , (qc_new - qc_old) * Cdtr ),
        maximum( -qi_old * Cdtr , (qi_new - qi_old) * Cdtr ),
        maximum( -qr_old * Cdtr , (qr_new - qr_old) * Cdtr ),
        maximum( -qs_old * Cdtr , (qs_new - qs_old) * Cdtr ),
        maximum( -qg_old * Cdtr , (qg_new - qg_old) * Cdtr)
    )

@field_operator
def _icon_graupel(
   kstart_moist: int32,
   kend: int32,
   dt: float64,  # time step
   dz: Field[[CellDim, KDim], float64],
   temperature: Field[[CellDim, KDim], float64],
   pres: Field[[CellDim, KDim], float64],
   rho: Field[[CellDim, KDim], float64],
   qv: Field[[CellDim, KDim], float64],
   qc: Field[[CellDim, KDim], float64],
   qi: Field[[CellDim, KDim], float64],
   qr: Field[[CellDim, KDim], float64],
   qs: Field[[CellDim, KDim], float64],
   qg: Field[[CellDim, KDim], float64],
   qnc: Field[[CellDim, KDim], float64], # originally 2D Field, now 3D Field
   is_isochoric: bool,
   ithermo_water: int32
) -> tuple[
    Field[[CellDim, KDim], float64],
    Field[[CellDim, KDim], float64],
    Field[[CellDim, KDim], float64],
    Field[[CellDim, KDim], float64],
    Field[[CellDim, KDim], float64],
    Field[[CellDim, KDim], float64],
    Field[[CellDim, KDim], float64],
    Field[[CellDim, KDim], float64],
    Field[[CellDim, KDim], float64],
    Field[[CellDim, KDim], float64],
    Field[[CellDim, KDim], float64],
    Field[[CellDim, KDim], float64],
    Field[[CellDim, KDim], float64],
    Field[[CellDim, KDim], float64],
    Field[[CellDim, KDim], float64]
]:
    (
        temperature_,
        qv_,
        qc_,
        qi_,
        qr_,
        qs_,
        qg_,
        rhoqrV_old_kup,
        rhoqsV_old_kup,
        rhoqgV_old_kup,
        rhoqiV_old_kup,
        Vnew_r,
        Vnew_s,
        Vnew_g,
        Vnew_i,
        dist_cldtop,
        rho_kup,
        Crho1o2_kup,
        Crhofac_qi_kup,
        Cvz0s_kup,
        qvsw_kup,
        k_lev
    ) = _icon_graupel_scan(
        kstart_moist,
        kend,
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
        is_isochoric,
        ithermo_water
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
        rhoqrV_old_kup,
        rhoqsV_old_kup,
        rhoqgV_old_kup,
        rhoqiV_old_kup,
        Vnew_r,
        Vnew_s,
        Vnew_g,
        Vnew_i
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
        float64,  # rain flux
        float64,  # snow flux
        float64,  # graupel flux
        float64,  # ice flux
        float64,  # solid precipitation flux
        int32  # k level
    ],
    # Grid information
    kstart_moist: int32,  # k starting level
    kend: int32,  # k bottom level
    rho: float64,
    qr: float64,
    qs: float64,
    qg: float64,
    qi: float64,
    rhoqrv_old_kup: float64,
    rhoqsv_old_kup: float64,
    rhoqgv_old_kup: float64,
    rhoqiv_old_kup: float64,
    vnew_r: float64,
    vnew_s: float64,
    vnew_g: float64,
    vnew_i: float64,
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
        k_lev = k_lev + int32(1)
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
    k_lev = k_lev + int32(1)

    return(
        rain_flux,
        snow_flux,
        graupel_flux,
        ice_flux,
        total_flux,
        k_lev
    )


@program
def icon_graupel(
    kstart_moist: int32,
    dt: float64,  # time step
    dz: Field[[CellDim, KDim], float64],
    temperature: Field[[CellDim, KDim], float64],
    pres: Field[[CellDim, KDim], float64],
    rho: Field[[CellDim, KDim], float64],
    qv: Field[[CellDim, KDim], float64],
    qc: Field[[CellDim, KDim], float64],
    qi: Field[[CellDim, KDim], float64],
    qr: Field[[CellDim, KDim], float64],
    qs: Field[[CellDim, KDim], float64],
    qg: Field[[CellDim, KDim], float64],
    qnc: Field[[CellDim, KDim], float64],  # originally 2D Field, now 3D Field
    is_isochoric: bool,
    ithermo_water: int32,
    temperature_tendency: Field[[CellDim, KDim], float64],
    qv_tendency: Field[[CellDim, KDim], float64],
    qc_tendency: Field[[CellDim, KDim], float64],
    qi_tendency: Field[[CellDim, KDim], float64],
    qr_tendency: Field[[CellDim, KDim], float64],
    qs_tendency: Field[[CellDim, KDim], float64],
    qg_tendency: Field[[CellDim, KDim], float64],
    rhoqrv_old_kup: Field[[CellDim, KDim], float64],
    rhoqsv_old_kup: Field[[CellDim, KDim], float64],
    rhoqgv_old_kup: Field[[CellDim, KDim], float64],
    rhoqiv_old_kup: Field[[CellDim, KDim], float64],
    vnew_r: Field[[CellDim, KDim], float64],
    vnew_s: Field[[CellDim, KDim], float64],
    vnew_g: Field[[CellDim, KDim], float64],
    vnew_i: Field[[CellDim, KDim], float64],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _icon_graupel(
        kstart_moist,
        vertical_end,
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
        is_isochoric,
        ithermo_water,
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
        },
    )

@field_operator
def _icon_graupel_flux(
    kstart_moist: int32,
    kend: int32,
    rho: Field[[CellDim, KDim], float64],
    qr: Field[[CellDim, KDim], float64],
    qs: Field[[CellDim, KDim], float64],
    qg: Field[[CellDim, KDim], float64],
    qi: Field[[CellDim, KDim], float64],
    rhoqrv_old_kup: Field[[CellDim, KDim], float64],
    rhoqsv_old_kup: Field[[CellDim, KDim], float64],
    rhoqgv_old_kup: Field[[CellDim, KDim], float64],
    rhoqiv_old_kup: Field[[CellDim, KDim], float64],
    vnew_r: Field[[CellDim, KDim], float64],
    vnew_s: Field[[CellDim, KDim], float64],
    vnew_g: Field[[CellDim, KDim], float64],
    vnew_i: Field[[CellDim, KDim], float64],
    do_ice_sedimentation: bool,
    do_latent_heat_nudging: bool
) -> tuple[
    Field[[CellDim, KDim], float64],
    Field[[CellDim, KDim], float64],
    Field[[CellDim, KDim], float64],
    Field[[CellDim, KDim], float64],
    Field[[CellDim, KDim], float64],
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
    kstart_moist: int32,
    kend: int32,
    rho: Field[[CellDim, KDim], float64],
    qr: Field[[CellDim, KDim], float64],
    qs: Field[[CellDim, KDim], float64],
    qg: Field[[CellDim, KDim], float64],
    qi: Field[[CellDim, KDim], float64],
    rhoqrv_old_kup: Field[[CellDim, KDim], float64],
    rhoqsv_old_kup: Field[[CellDim, KDim], float64],
    rhoqgv_old_kup: Field[[CellDim, KDim], float64],
    rhoqiv_old_kup: Field[[CellDim, KDim], float64],
    vnew_r: Field[[CellDim, KDim], float64],
    vnew_s: Field[[CellDim, KDim], float64],
    vnew_g: Field[[CellDim, KDim], float64],
    vnew_i: Field[[CellDim, KDim], float64],
    do_ice_sedimentation: bool,
    do_latent_heat_nudging: bool,  # if true, latent heat nudging is applied
    rain_precipitation_flux: Field[[CellDim, KDim], float64],
    snow_precipitation_flux: Field[[CellDim, KDim], float64],
    graupel_precipitation_flux: Field[[CellDim, KDim], float64],
    ice_precipitation_flux: Field[[CellDim, KDim], float64],
    total_precipitation_flux: Field[[CellDim, KDim], float64]
):
    _icon_graupel_flux(
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
        do_latent_heat_nudging,
        out=(
            rain_precipitation_flux,
            snow_precipitation_flux,
            graupel_precipitation_flux,
            ice_precipitation_flux,
            total_precipitation_flux
        )
    )

#(backend=roundtrip.executor)
