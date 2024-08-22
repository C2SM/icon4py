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
from typing import Final
from numpy import sqrt as numpy_sqrt
from numpy import log as numpy_log
from numpy import exp as numpy_exp

from gt4py.eve.utils import FrozenNamespace
from gt4py.next.ffront.decorator import program, field_operator, scan_operator
from gt4py.next.ffront.fbuiltins import (
    Field,
    exp,
    tanh,
    int32,
    float64,
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
#from icon4py.atm_phy_schemes.mo_convect_tables import conv_table
from icon4py.model.common.mo_physical_constants import phy_const


sys.setrecursionlimit(350000)


# This class contains all constants that are used in the transfer rate calculations
class GraupelFunctionConstants(FrozenNamespace):

   GrFuncConst_n0s1  = 13.5 * 5.65e5 # parameter in N0S(T)
   GrFuncConst_n0s2  = -0.107        # parameter in N0S(T), Field et al
   GrFuncConst_mma   = (5.065339, -0.062659, -3.032362, 0.029469, -0.000285, 0.312550,  0.000204,  0.003199, 0.000000, -0.015952)
   GrFuncConst_mmb   = (0.476221, -0.015896,  0.165977, 0.007468, -0.000141, 0.060366,  0.000079,  0.000594, 0.000000, -0.003577)

   GrFuncConst_thet  = 248.15 # temperature for het. nuc. of cloud ice

   GrFuncConst_ccau  = 4.0e-4     # autoconversion coefficient (cloud water to rain)
   GrFuncConst_cac   = 1.72       # (15/32)*(PI**0.5)*(ECR/RHOW)*V0R*AR**(1/8)
   GrFuncConst_kphi1 = 6.00e+02   # constant in phi-function for autoconversion
   GrFuncConst_kphi2 = 0.68e+00   # exponent in phi-function for autoconversion
   GrFuncConst_kphi3 = 5.00e-05   # exponent in phi-function for accretion
   GrFuncConst_kcau  = 9.44e+09   # kernel coeff for SB2001 autoconversion
   GrFuncConst_kcac  = 5.25e+00   # kernel coeff for SB2001 accretion
   GrFuncConst_cnue  = 2.00e+00   # gamma exponent for cloud distribution
   GrFuncConst_xstar = 2.60e-10   # separating mass between cloud and rain

   GrFuncConst_c1es    = 610.78                                               # = b1
   GrFuncConst_c2es    = GrFuncConst_c1es*phy_const.rd/phy_const.rv           #
   GrFuncConst_c3les   = 17.269                                               # = b2w
   GrFuncConst_c3ies   = 21.875                                               # = b2i
   GrFuncConst_c4les   = 35.86                                                # = b4w
   GrFuncConst_c4ies   = 7.66                                                 # = b4i
   GrFuncConst_c5les   = GrFuncConst_c3les*(phy_const.tmelt - GrFuncConst_c4les)      # = b234w
   GrFuncConst_c5ies   = GrFuncConst_c3ies*(phy_const.tmelt - GrFuncConst_c4ies)      # = b234i
   GrFuncConst_c5alvcp = GrFuncConst_c5les*phy_const.alv/phy_const.cpd            #
   GrFuncConst_c5alscp = GrFuncConst_c5ies*phy_const.als/phy_const.cpd            #
   GrFuncConst_alvdcp  = phy_const.alv/phy_const.cpd                          #
   GrFuncConst_alsdcp  = phy_const.als/phy_const.cpd                          #

   GrFuncConst_crim_g  = 4.43       # coefficient for graupel riming
   GrFuncConst_csg     = 0.5        # coefficient for snow-graupel conversion by riming
   GrFuncConst_cagg_g  = 2.46
   GrFuncConst_ciau    = 1.0e-3     # autoconversion coefficient (cloud ice to snow)
   GrFuncConst_msmin   = 3.0e-9     # initial mass of snow crystals
   GrFuncConst_cicri   = 1.72       # (15/32)*(PI**0.5)*(EIR/RHOW)*V0R*AR**(1/8)
   GrFuncConst_crcri   = 1.24e-3    # (PI/24)*EIR*V0R*Gamma(6.5)*AR**(-5/8)
   GrFuncConst_asmel   = 2.95e3     # DIFF*LH_v*RHO/LHEAT
   GrFuncConst_tcrit   = 3339.5     # factor in calculation of critical temperature

   GrFuncConst_qc0 = 0.0            # param_qc0
   GrFuncConst_qi0 = 0.0            # param_qi0


graupel_funcConst : Final = GraupelFunctionConstants()


# Statement functions
# -------------------

def fpvsw(
   ztx: float64
   ) -> float64:
   # Return saturation vapour pressure over water from temperature
   return graupel_funcConst.GrFuncConst_c1es * numpy_exp(
      graupel_funcConst.GrFuncConst_c3les * (ztx - phy_const.tmelt) / (ztx - graupel_funcConst.GrFuncConst_c4les)
   )


def fxna(
   ztx: float64
   ) -> float64:
   # Return number of activate ice crystals from temperature
   return 1.0e2 * numpy_exp(0.2 * (phy_const.tmelt - ztx))


def fxna_cooper(
   ztx: float64
   ) -> float64:
   # Return number of activate ice crystals from temperature

   # Method: Cooper (1986) used by Greg Thompson(2008)

   return 5.0 * numpy_exp(0.304 * (phy_const.tmelt - ztx))

class GraupelGlobalConstants(FrozenNamespace):
   """Constants for the graupel scheme."""

   GrConst_trfrz              = 271.15           # threshold temperature for heterogeneous freezing of raindrops
   GrConst_crfrz              = 1.68             # coefficient for raindrop freezing
   GrConst_crfrz1             = 9.95e-5          # FR: 1. coefficient for immersion raindrop freezing: alpha_if
   GrConst_crfrz2             = 0.66             # FR: 2. coefficient for immersion raindrop freezing: a_if
   GrConst_thn                = 236.15           # temperature for hom. freezing of cloud water
   GrConst_tmix               = 250.15           # threshold temperature for mixed-phase cloud freezing of cloud drops (Forbes 2012)
   GrConst_qmin               = 1.0e-15          # threshold for computations
   GrConst_eps                = 1.0e-15          # small number
   GrConst_bvi                = 0.16             # v = zvz0i*rhoqi^zbvi
   GrConst_rho0               = 1.225e+0         # reference air density
   GrConst_v_sedi_rain_min    = 0.7              # in m/s; minimum terminal fall velocity of rain    particles (applied only near the ground)
   GrConst_v_sedi_snow_min    = 0.1              # in m/s; minimum terminal fall velocity of snow    particles (applied only near the ground)
   GrConst_v_sedi_graupel_min = 0.4              # in m/s; minimum terminal fall velocity of graupel particles (applied only near the ground)
   GrConst_nimax_Thom         = 250.0e+3          # FR: maximal number of ice crystals
   #zams_ci= 0.069           # Formfactor in the mass-size relation of snow particles for cloud ice scheme
   #zams_gr= 0.069           # Formfactor in the mass-size relation of snow particles for graupel scheme
   GrConst_ams                = 0.069            # Formfactor in the mass-size relation of snow particles for graupel scheme
   GrConst_n0s0               = 8.0e5            #
   GrConst_rimexp_g           = 0.94878          #
   GrConst_expsedg            = 0.217            # exponent for graupel sedimentation
   GrConst_vz0g               = 12.24            # coefficient of sedimentation velocity for graupel
   GrConst_mi0                = 1.0e-12          # initial crystal mass for cloud ice nucleation
   GrConst_mimax              = 1.0e-9           # maximum mass of cloud ice crystals
   GrConst_msmin              = 3.0e-9           # initial mass of snow crystals
   GrConst_ceff_fac           = 3.5e-3           # Scaling factor [1/K] for temperature-dependent cloud ice sticking efficiency
   GrConst_tmin_iceautoconv   = 188.15           # Temperature at which cloud ice autoconversion starts
   GrConst_dist_cldtop_ref    = 500.0            # Reference length for distance from cloud top (Forbes 2012)
   GrConst_reduce_dep_ref     = 0.1              # lower bound on snow/ice deposition reduction
   GrConst_hw                 = 2.270603         # Howell factor
   GrConst_ecs                = 0.9              # Collection efficiency for snow collecting cloud water
   GrConst_v1s                = 0.50             # Exponent in the terminal velocity for snow
   GrConst_eta                = 1.75e-5          # kinematic viscosity of air
   GrConst_dv                 = 2.22e-5          # molecular diffusion coefficient for water vapour
   GrConst_lheat              = 2.40e-2          # thermal conductivity of dry air
   GrConst_bms                = 2.000            # Exponent in the mass-size relation of snow particles
   GrConst_ami                = 130.0            # Formfactor in the mass-size relation of cloud ice
   GrConst_rhow               = 1.000e+3         # density of liquid water
   GrConst_cp_v               = 1850.0           # specific heat of water vapor J, at constant pressure (Landolt-Bornstein)
   GrConst_ci                 = 2108.0           # specific heat of ice

   # option switches (remove in the future, not used)
   GrConst_iautocon       = 1
   GrConst_isnow_n0temp   = 2
   GrConst_lsuper_coolw   = True  # switch for supercooled liquid water (work from Felix Rieper)
   GrConst_lsedi_ice      = True  # switch for sedimentation of cloud ice (Heymsfield & Donner 1990 *1/3)
   GrConst_lstickeff      = True  # switch for sticking coeff. (work from Guenther Zaengl)
   GrConst_lred_depgrow   = True  # separate switch for reduced depositional growth near tops of water clouds
   #GrConst_ithermo_water  = False #
   #GrConst_l_cv           = True
   #GrConst_lpres_pri      = True
   #GrConst_ldass_lhn      = True # if true, latent heat nudging is applied
   #GrConst_ldiag_ttend    = True # if true, temperature tendency shall be diagnosed
   #GrConst_ldiag_qtend    = True # if true, moisture tendencies shall be diagnosed

   #if (graupel_const.GrConst_lsuper_coolw):
   GrConst_nimax = GrConst_nimax_Thom
   GrConst_nimix = 5.0 * numpy_exp(0.304 * (phy_const.tmelt - GrConst_tmix))
   #else:
   #    GrConst_nimax = 1.0e2 * exp(0.2 * (phy_const.tmelt - GrConst_thn))
   #    GrConst_nimix = 1.0e2 * exp(0.2 * (phy_const.tmelt - GrConst_tmix))

   GrConst_x1o3   =  1.0/ 3.0
   GrConst_x7o8   =  7.0/ 8.0
   GrConst_x13o8  = 13.0/ 8.0
   GrConst_x27o16 = 27.0/16.0
   GrConst_x1o2   =  1.0/ 2.0
   GrConst_x3o4   =  0.75
   GrConst_x7o4   =  7.0/ 4.0

   GrConst_ceff_min = 0.01 # default: 0.075
   GrConst_v0snow = 20.0 # default: 20.0
   GrConst_vz0i = 1.25
   GrConst_icesedi_exp = 0.3 # default: 0.33
   GrConst_mu_rain = 0.0
   GrConst_rain_n0_factor = 1.0
   '''
   GrConst_ceff_min = graupel_optional.GOC_ceff_min
   GrConst_v0snow = graupel_optional.GOC_v0snow
   GrConst_vz0i = graupel_optional.GOC_vz0i
   GrConst_mu_rain = graupel_optional.GOC_mu_rain
   GrConst_rain_n0_factor = graupel_optional.GOC_rain_n0_factor
   '''

   GrConst_ccsrim = 0.25 * math_const.pi * GrConst_ecs * GrConst_v0snow * gamma_fct(GrConst_v1s + 3.0)
   GrConst_ccsagg = 0.25 * math_const.pi * GrConst_v0snow * gamma_fct(GrConst_v1s + 3.0)
   GrConst_ccsdep = 0.26 * gamma_fct((GrConst_v1s + 5.0) / 2.0) * numpy_sqrt(1.0 / GrConst_eta)
   GrConst_ccsvxp_ = -(GrConst_v1s / (GrConst_bms + 1.0) + 1.0)
   GrConst_ccsvel = GrConst_ams * GrConst_v0snow * gamma_fct(GrConst_bms + GrConst_v1s + 1.0) * (GrConst_ams * gamma_fct(GrConst_bms + 1.0))**GrConst_ccsvxp_
   GrConst_ccsvxp = GrConst_ccsvxp_ + 1.0
   GrConst_ccslam = GrConst_ams * gamma_fct(GrConst_bms + 1.0)
   GrConst_ccslxp = 1.0 / (GrConst_bms + 1.0)
   GrConst_ccswxp = GrConst_v1s * GrConst_ccslxp
   GrConst_ccsaxp = -(GrConst_v1s + 3.0)
   GrConst_ccsdxp = -(GrConst_v1s + 1.0) / 2.0
   GrConst_ccshi1 = phy_const.als * phy_const.als / (GrConst_lheat * phy_const.rv)
   GrConst_ccdvtp = 2.22e-5 * phy_const.tmelt**(-1.94) * 101325.0
   GrConst_ccidep = 4.0 * GrConst_ami**(-GrConst_x1o3)

   #if ( GrConst_lsuper_coolw ):
   #   GrConst_nimax = GrConst_nimax_Thom
   #   GrConst_nimix = fxna_cooper(GrConst_tmix)
   #else:
   #   GrConst_nimax = fxna(GrConst_thn)
   #   GrConst_nimix = fxna(GrConst_tmix)

   GrConst_pvsw0 = fpvsw(phy_const.tmelt)  # sat. vap. pressure for t = t0
   GrConst_log_10 = numpy_log(10.0) # logarithm of 10

   GrConst_n0r   = 8.0e6 * numpy_exp(3.2 * GrConst_mu_rain) * (0.01)**(-GrConst_mu_rain)  # empirical relation adapted from Ulbrich (1983)
   GrConst_n0r   = GrConst_n0r * GrConst_rain_n0_factor                           # apply tuning factor to zn0r variable
   GrConst_ar    = math_const.pi * GrConst_rhow / 6.0 * GrConst_n0r * gamma_fct(GrConst_mu_rain + 4.0)      # pre-factor

   GrConst_vzxp  = 0.5 / (GrConst_mu_rain + 4.0)
   GrConst_vz0r  = 130.0 * gamma_fct(GrConst_mu_rain + 4.5) / gamma_fct(GrConst_mu_rain + 4.0) * GrConst_ar**(-GrConst_vzxp)

   GrConst_cevxp = (GrConst_mu_rain + 2.0) / (GrConst_mu_rain + 4.0)
   GrConst_cev   = 2.0 * math_const.pi * GrConst_dv / GrConst_hw * GrConst_n0r*GrConst_ar**(-GrConst_cevxp) * gamma_fct(GrConst_mu_rain + 2.0)
   GrConst_bevxp = (2.0 * GrConst_mu_rain + 5.5) / (2.0 * GrConst_mu_rain + 8.0) - GrConst_cevxp
   GrConst_bev   =  0.26 * numpy_sqrt(GrConst_rho0 * 130.0 / GrConst_eta) * GrConst_ar**(-GrConst_bevxp) * gamma_fct((2.0 * GrConst_mu_rain + 5.5) / 2.0) / gamma_fct(GrConst_mu_rain + 2.0)

   # Precomputations for optimization
   GrConst_ccswxp_ln1o2  = numpy_exp(GrConst_ccswxp * numpy_log(0.5))
   GrConst_vzxp_ln1o2    = numpy_exp(GrConst_vzxp * numpy_log(0.5))
   GrConst_bvi_ln1o2     = numpy_exp(GrConst_bvi * numpy_log(0.5))
   GrConst_expsedg_ln1o2 = numpy_exp(GrConst_expsedg * numpy_log(0.5))


graupel_const : Final = GraupelGlobalConstants()


# Field operation
# -------------------

@field_operator
def _fxna(
   ztx: float64
   ) -> float64:
   # Return number of activate ice crystals from temperature
   return 1.0e2 * exp(0.2 * (phy_const.tmelt - ztx))


@field_operator
def _fxna_cooper(
   ztx: float64
   ) -> float64:
   # Return number of activate ice crystals from temperature

   # Method: Cooper (1986) used by Greg Thompson(2008)

   return 5.0 * exp(0.304 * (phy_const.tmelt - ztx))


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
      phy_const.alv
      + (graupel_const.GrConst_cp_v - phy_const.clw) * (input_t - phy_const.tmelt)
      - phy_const.rv * input_t
   )

@field_operator
def latent_heat_sublimation(
   input_t: float64
) -> float64:

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
      phy_const.als + (graupel_const.GrConst_cp_v - graupel_const.GrConst_ci) * (input_t - phy_const.tmelt) - phy_const.rv * input_t
   )


@field_operator
def sat_pres_water(
   input_t: float64
) -> float64:

   # Tetens formula
   return (
      graupel_funcConst.GrFuncConst_c1es * exp( graupel_funcConst.GrFuncConst_c3les * (input_t - phy_const.tmelt) / (input_t - graupel_funcConst.GrFuncConst_c4les) )
   )


@field_operator
def sat_pres_ice(
   input_t: float64
) -> float64:

   # Tetens formula
   return (
      graupel_funcConst.GrFuncConst_c1es * exp( graupel_funcConst.GrFuncConst_c3ies * (input_t - phy_const.tmelt) / (input_t - graupel_funcConst.GrFuncConst_c4ies) )
   )

@field_operator
def sat_pres_water_murphykoop(
   input_t: float64
) -> float64:

  # Eq. (10) of Murphy and Koop (2005)
  return (
     exp( 54.842763 - 6763.22 / input_t - 4.210 * log(input_t) + 0.000367 * input_t  + tanh(0.0415 * (input_t - 218.8)) * (53.878 - 1331.22 / input_t - 9.44523 * log(input_t) + 0.014025 * input_t) )
  )

@field_operator
def sat_pres_ice_murphykoop(
   input_t: float64
) -> float64:

  # Eq. (7) of Murphy and Koop (2005)
  return (
     exp(9.550426 - 5723.265 / input_t + 3.53068 * log(input_t) - 0.00728332 * input_t )
  )

@field_operator
def TV(
   input_t: float64
) -> float64:

   # Tetens formula
   return (
      graupel_funcConst.GrFuncConst_c1es * exp( graupel_funcConst.GrFuncConst_c3les * (input_t - phy_const.tmelt) / (input_t - graupel_funcConst.GrFuncConst_c4les) )
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
        gtx.int32(0)     # k level
    )
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
        gtx.int32
    ],
    # Grid information
    startmoist_level: gtx.int32,     # k start moist level
    surface_level: gtx.int32,     # k bottom level
    liquid_autoconversion_option: gtx.int32,
    snow_intercept_option: gtx.int32,
    is_isochoric: bool,
    use_constant_water_heat_capacity: bool,
    ice_stickeff_min: wpfloat,
    ice_v0: wpfloat,
    ice_sedi_density_factor_exp: wpfloat,
    snow_v0: wpfloat,
    ccsrim: wpfloat,
    ccsagg: wpfloat,
    ccsvel: wpfloat,
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
    dz: wpfloat,
    # Prognostics
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

    # ------------------------------------------------------------------------------
    #  Section 1: Initial setting of physical constants
    # ------------------------------------------------------------------------------

    # unpack carry
    # temporary variables for storing variables at k-1 (upper) grid
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
        dist_cldtop,
        rho_kup,
        crho1o2_kup,
        crhofac_qi_kup,
        snow_sed0_kup,
        qvsw_kup,
        k_lev
    ) = state_kup

    qv_kup = qv_old_kup + qv_tendency_kup * dt
    qc_kup = qc_old_kup + qc_tendency_kup * dt
    qi_kup = qi_old_kup + qi_tendency_kup * dt
    qr_kup = qr_old_kup + qr_tendency_kup * dt
    qs_kup = qs_old_kup + qs_tendency_kup * dt
    qg_kup = qg_old_kup + qg_tendency_kup * dt

    is_surface = True if k_lev + startmoist_level == surface_level else False

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
    lhv = icon_graupel_params.alv if use_constant_water_heat_capacity else icon_graupel_params.alv + (
            icon_graupel_params.cp_v - icon_graupel_params.clw) * (temperature - icon_graupel_params.tmelt) - icon_graupel_params.rv * temperature
    lhs = icon_graupel_params.als if use_constant_water_heat_capacity else icon_graupel_params.als + (
            icon_graupel_params.cp_v - icon_graupel_params.ci) * (temperature - icon_graupel_params.tmelt) - icon_graupel_params.rv * temperature

    # ----------------------------------------------------------------------------
    # Section 2: Check for existence of rain and snow
    #            Initialize microphysics and sedimentation scheme
    # ----------------------------------------------------------------------------

    # initialization of source terms (all set to zero)
    szdep_v2i = wpfloat("0.0")  # vapor   -> ice,     ice vapor deposition
    szsub_v2i = wpfloat("0.0")  # vapor   -> ice,     ice vapor sublimation
    sidep_v2i = wpfloat("0.0")  # vapor   -> ice,     ice vapor net deposition
    # ssdpc_v2s = wpfloat("0.0") # vapor   -> snow,    snow vapor deposition below freezing temperature
    # sgdpc_v2g = wpfloat("0.0") # vapor   -> graupel, graupel vapor deposition below freezing temperature
    # ssdph_v2s = wpfloat("0.0") # vapor   -> snow,    snow vapor deposition above freezing temperature
    # sgdph_v2g = wpfloat("0.0") # vapor   -> graupel, graupel vapor deposition above freezing temperature
    ssdep_v2s = wpfloat("0.0")  # vapor   -> snow,    snow vapor deposition
    sgdep_v2g = wpfloat("0.0")  # vapor   -> graupel, graupel vapor deposition
    # sdnuc_v2i = wpfloat("0.0") # vapor   -> ice,     low temperature heterogeneous ice deposition nucleation
    # scnuc_v2i = wpfloat("0.0") # vapor   -> ice,     incloud ice nucleation
    snucl_v2i = wpfloat("0.0")  # vapor   -> ice,     ice nucleation
    sconr_v2r = wpfloat("0.0")  # vapor   -> rain,    rain condensation on melting snow/graupel
    scaut_c2r = wpfloat("0.0")  # cloud   -> rain,    cloud autoconversion into rain
    scfrz_c2i = wpfloat("0.0")  # cloud   -> ice,     cloud freezing
    scacr_c2r = wpfloat("0.0")  # cloud   -> rain,    rain-cloud accretion
    sshed_c2r = wpfloat("0.0")  # cloud   -> rain,    rain shedding from riming above freezing
    srims_c2s = wpfloat("0.0")  # cloud   -> snow,    snow riming
    srimg_c2g = wpfloat("0.0")  # cloud   -> graupel, graupel riming
    simlt_i2c = wpfloat("0.0")  # ice     -> cloud,   ice melting
    sicri_i2g = wpfloat("0.0")  # ice     -> graupel, ice loss in rain-ice accretion
    sdaut_i2s = wpfloat("0.0")  # ice     -> snow,    ice vapor depositional autoconversion into snow
    saggs_i2s = wpfloat("0.0")  # ice     -> snow,    snow-ice aggregation
    saggg_i2g = wpfloat("0.0")  # ice     -> graupel, graupel-ice aggregation
    siaut_i2s = wpfloat("0.0")  # ice     -> snow,    ice autoconversion into snow
    srcri_r2g = wpfloat("0.0")  # rain    -> graupel, rain loss in rain-ice accretion
    # scrfr_r2g = wpfloat("0.0") # rain    -> graupel, rain freezing in clouds
    # ssrfr_r2g = wpfloat("0.0") # rain    -> graupel, rain freezing in clear sky
    srfrz_r2g = wpfloat("0.0")  # rain    -> graupel, rain freezing
    sevap_r2v = wpfloat("0.0")  # rain    -> vapor,   rain evaporation
    ssmlt_s2r = wpfloat("0.0")  # snow    -> rain,    snow melting
    scosg_s2g = wpfloat("0.0")  # snow    -> graupel, snow autoconversion into graupel
    sgmlt_g2r = wpfloat("0.0")  # graupel -> rain,    graupel melting

    reduce_dep = wpfloat("1.0")  # FR: Reduction coeff. for dep. growth of rain and ice

    # ----------------------------------------------------------------------------
    # 2.1: Preparations for computations and to check the different conditions
    # ----------------------------------------------------------------------------

    # ..for density correction of fall speeds
    c1orho = wpfloat("1.0") / rho
    chlp = log(icon_graupel_params.ref_air_density * c1orho)
    crho1o2 = exp(chlp / wpfloat("2.0"))
    crhofac_qi = exp(chlp * ice_sedi_density_factor_exp)

    rhoqr = qr * rho
    rhoqs = qs * rho
    rhoqg = qg * rho
    rhoqi = qi * rho

    llqr = True if (rhoqr > icon_graupel_params.qmin) else False
    llqs = True if (rhoqs > icon_graupel_params.qmin) else False
    llqg = True if (rhoqg > icon_graupel_params.qmin) else False
    llqi = True if (rhoqi > icon_graupel_params.qmin) else False

    cdtdh = 0.5 * dt / dz

    # -------------------------------------------------------------------------
    # qs_prepare:
    # -------------------------------------------------------------------------
    if llqs:
        # function called: Cn0s = snow_intercept(qs,temperature,rho)
        # ------------------------------------------------------------------------------
        # Description:
        #   This subroutine computes the intercept parameter, N0, of the snow exponential size distribution.
        #
        #   First method: Explained in paragraphs at pages 2008 and 2009 in Field et al. (2005). N0s_23 = (M_2)^4 / (M_3)^3, M_2 = Gamma(3) N0s / lamda^3, M_2 = Gamma(4) N0s / lamda^4, so N0s_23 = 2/27 N0s. And N0s_23 = 5.65E5 exp(-0.107Tc)
        #
        #   Second method: Eq. 5.160 in the documentation, originally in Table 2 in Field et al. (2005).
        #
        # ------------------------------------------------------------------------------

        if snow_intercept_option == 1:
            # Calculate n0s using the temperature-dependent
            # formula of Field et al. (2005)
            local_tc = temperature - icon_graupel_params.tmelt
            local_tc = minimum(local_tc, wpfloat("0.0"))
            local_tc = maximum(local_tc, -40.0)
            n0s = icon_graupel_params.snow_n0s1 * exp(icon_graupel_params.snow_n0s2 * local_tc)
            n0s = minimum(n0s, 1.0e9)
            n0s = maximum(n0s, 1.0e6)

        elif snow_intercept_option == 2:
            # Calculate n0s using the temperature-dependent moment
            # relations of Field et al. (2005)
            local_tc = temperature - icon_graupel_params.tmelt
            local_tc = minimum(local_tc, wpfloat("0.0"))
            local_tc = maximum(local_tc, -40.0)

            local_nnr = 3.0
            local_hlp = (
                icon_graupel_params.snow_mma[0] +
                icon_graupel_params.snow_mma[1] * local_tc +
                icon_graupel_params.snow_mma[2] * local_nnr +
                icon_graupel_params.snow_mma[3] * local_tc * local_nnr +
                icon_graupel_params.snow_mma[4] * local_tc ** 2.0 +
                icon_graupel_params.snow_mma[5] * local_nnr ** 2.0 +
                icon_graupel_params.snow_mma[6] * local_tc ** 2.0 * local_nnr +
                icon_graupel_params.snow_mma[7] * local_tc * local_nnr ** 2.0 +
                icon_graupel_params.snow_mma[8] * local_tc ** 3.0 +
                icon_graupel_params.snow_mma[9] * local_nnr ** 3.0
            )
            local_alf = exp(local_hlp * log(10.0))
            local_bet = (
                icon_graupel_params.snow_mmb[0] +
                icon_graupel_params.snow_mmb[1] * local_tc +
                icon_graupel_params.snow_mmb[2] * local_nnr +
                icon_graupel_params.snow_mmb[3] * local_tc * local_nnr +
                icon_graupel_params.snow_mmb[4] * local_tc ** 2.0 +
                icon_graupel_params.snow_mmb[5] * local_nnr ** 2.0 +
                icon_graupel_params.snow_mmb[6] * local_tc ** 2.0 * local_nnr +
                icon_graupel_params.snow_mmb[7] * local_tc * local_nnr ** 2.0 +
                icon_graupel_params.snow_mmb[8] * local_tc ** 3.0 +
                icon_graupel_params.snow_mmb[9] * local_nnr ** 3.0
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
        snow_sed0 = ccsvel * exp(icon_graupel_params.ccsvxp * log(n0s))
        # compute constants for riming, aggregation, and deposition processes for snow
        crim = ccsrim * n0s
        cagg = ccsagg * n0s
        cbsdep = icon_graupel_params.ccsdep * sqrt(snow_v0)
    else:
        n0s = icon_graupel_params.snow_n0
        snow_sed0 = wpfloat("0.0")
        crim = wpfloat("0.0")
        cagg = wpfloat("0.0")
        cbsdep = wpfloat("0.0")


    # ----------------------------------------------------------------------------
    # 2.2: sedimentation fluxes
    # ----------------------------------------------------------------------------

    rhoqrv_new_kup = qr_kup * rho_kup * vnew_r
    rhoqsv_new_kup = qs_kup * rho_kup * vnew_s
    rhoqgv_new_kup = qg_kup * rho_kup * vnew_g
    rhoqiv_new_kup = qi_kup * rho_kup * vnew_i
    if rhoqrv_new_kup <= icon_graupel_params.qmin: rhoqrv_new_kup = wpfloat("0.0")
    if rhoqsv_new_kup <= icon_graupel_params.qmin: rhoqsv_new_kup = wpfloat("0.0")
    if rhoqgv_new_kup <= icon_graupel_params.qmin: rhoqgv_new_kup = wpfloat("0.0")
    if rhoqiv_new_kup <= icon_graupel_params.qmin: rhoqiv_new_kup = wpfloat("0.0")

    rhoqr_intermediate = rhoqr / cdtdh + rhoqrv_new_kup + rhoqrv_old_kup
    rhoqs_intermediate = rhoqs / cdtdh + rhoqsv_new_kup + rhoqsv_old_kup
    rhoqg_intermediate = rhoqg / cdtdh + rhoqgv_new_kup + rhoqgv_old_kup
    rhoqi_intermediate = rhoqi / cdtdh + rhoqiv_new_kup + rhoqiv_old_kup

    # -------------------------------------------------------------------------
    # qs_sedi, qr_sedi, qg_sedi, qi_sedi:
    # -------------------------------------------------------------------------

    if k_lev > 0:
        vnew_s = snow_sed0_kup * exp(icon_graupel_params.ccswxp * log(
            (qs_kup + qs) * 0.5 * rho_kup)) * crho1o2_kup if qs_kup + qs > icon_graupel_params.qmin else wpfloat(
            "0.0")
        vnew_r = rain_v0 * exp(rain_exp_v * log(
            (qr_kup + qr) * 0.5 * rho_kup)) * crho1o2_kup if qr_kup + qr > icon_graupel_params.qmin else wpfloat(
            "0.0")
        vnew_g = icon_graupel_params.graupel_sed0 * exp(icon_graupel_params.graupel_exp_sed * log(
            (qg_kup + qg) * 0.5 * rho_kup)) * crho1o2_kup if qg_kup + qg > icon_graupel_params.qmin else wpfloat(
            "0.0")
        vnew_i = ice_v0 * exp(icon_graupel_params.ice_exp_v * log(
            (qi_kup + qi) * 0.5 * rho_kup)) * crhofac_qi_kup if qi_kup + qi > icon_graupel_params.qmin else wpfloat(
            "0.0")

    if llqs:
        terminal_velocity = snow_sed0 * exp(icon_graupel_params.ccswxp * log(rhoqs)) * crho1o2
        # Prevent terminal fall speed of snow from being zero at the surface level
        if is_surface: terminal_velocity = maximum(terminal_velocity, icon_graupel_params.snow_v_sedi_min)

        rhoqsv = rhoqs * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if vnew_s == wpfloat("0.0"): vnew_s = terminal_velocity * icon_graupel_params.ccswxp_ln1o2

    else:
        rhoqsv = wpfloat("0.0")

    if llqr:
        terminal_velocity = rain_v0 * exp(rain_exp_v * log(rhoqr)) * crho1o2
        # Prevent terminal fall speed of snow from being zero at the surface level
        if is_surface: terminal_velocity = maximum(terminal_velocity, icon_graupel_params.rain_v_sedi_min)

        rhoqrv = rhoqr * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if vnew_r == wpfloat("0.0"): vnew_r = terminal_velocity * rain_exp_v_ln1o2

    else:
        rhoqrv = wpfloat("0.0")

    if llqg:
        terminal_velocity = icon_graupel_params.graupel_sed0 * exp(icon_graupel_params.graupel_exp_sed * log(rhoqg)) * crho1o2
        # Prevent terminal fall speed of snow from being zero at the surface level
        if is_surface: terminal_velocity = maximum(terminal_velocity, icon_graupel_params.graupel_v_sedi_min)

        rhoqgv = rhoqg * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if vnew_g == wpfloat("0.0"): vnew_g = terminal_velocity * graupel_exp_sed_ln1o2

    else:
        rhoqgv = wpfloat("0.0")

    if llqi:
        terminal_velocity = ice_v0 * exp(icon_graupel_params.ice_exp_v * log(rhoqi)) * crhofac_qi

        rhoqiv = rhoqi * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if vnew_i == wpfloat("0.0"): vnew_i = terminal_velocity * ice_exp_v_ln1o2

    else:
        rhoqiv = wpfloat("0.0")

    # Prevent terminal fall speeds of precip hydrometeors from being zero at the surface level
    if is_surface:
        vnew_s = maximum(vnew_s, icon_graupel_params.snow_v_sedi_min)
        vnew_r = maximum(vnew_r, icon_graupel_params.rain_v_sedi_min)
        vnew_g = maximum(vnew_g, icon_graupel_params.graupel_v_sedi_min)

    # -------------------------------------------------------------------------
    # derive the intermediate density of hydrometeors, Eq. 5.21:
    # -------------------------------------------------------------------------

    # limit the precipitation flux at this k level such that mixing ratio won't go below zero
    rhoqrv = minimum(rhoqrv, rhoqr_intermediate)
    rhoqsv = minimum(rhoqsv, rhoqs_intermediate)
    rhoqgv = minimum(rhoqgv, maximum(wpfloat("0.0"), rhoqg_intermediate))
    rhoqiv = minimum(rhoqiv, rhoqi_intermediate)

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

    # --------------------------------------------------------------------------
    # 2.3: Second part of preparations
    # --------------------------------------------------------------------------

    # TODO (Chia Rui): remove the comments below when the test pass
    # FR old
    #   Csdep    = 3.2E-2
    # Csdep        = 3.367e-2
    # Cidep        = 1.3e-5
    # Cslam        = 1.0e10

    cscmax = qc * dtr
    # TODO (Chia Rui): define a field operator for the ice nuclei number concentration, cnin = _fxna_cooper(temperature) when slow compilation time issue is resolved
    cnin = 5.0 * exp(0.304 * (icon_graupel_params.tmelt - temperature))
    cnin = minimum(cnin, icon_graupel_params.nimax_Thom)
    cmi = minimum(rho * qi / cnin, icon_graupel_params.ice_max_mass)
    cmi = maximum(icon_graupel_params.ice_initial_mass, cmi)

    # function called: qvsw = sat_pres_water(temperature) / (rho * icon_graupel_params.rv * temperature)
    # function called: qvsi = sat_pres_ice(temperature) / (rho * icon_graupel_params.rv * temperature)
    qvsw = icon_graupel_params.c1es * exp(icon_graupel_params.c3les * (temperature - icon_graupel_params.tmelt) / (
            temperature - icon_graupel_params.c4les)) / (rho * icon_graupel_params.rv * temperature)
    qvsi = icon_graupel_params.c1es * exp(icon_graupel_params.c3ies * (temperature - icon_graupel_params.tmelt) / (
            temperature - icon_graupel_params.c4ies)) / (rho * icon_graupel_params.rv * temperature)
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
        csrmax = rhoqr_intermediate / rho * dtr  # GZ: shifting this computation ahead of the IF condition changes results!
        if (qi + qc > icon_graupel_params.qmin):
            celn7o8qrk = exp(7.0 / 8.0 * clnrhoqr)
        else:
            celn7o8qrk = wpfloat("0.0")
        if temperature < icon_graupel_params.threshold_freeze_temperature:
            celn7o4qrk = exp(7.0 / 4.0 * clnrhoqr)  # FR new
            celn27o16qrk = exp(27.0 / 16.0 * clnrhoqr)
        else:
            celn7o4qrk = wpfloat("0.0")
            celn27o16qrk = wpfloat("0.0")
        if llqi:
            celn13o8qrk = exp(13.0 / 8.0 * clnrhoqr)
        else:
            celn13o8qrk = wpfloat("0.0")
    else:
        csrmax = wpfloat("0.0")
        celn7o8qrk = wpfloat("0.0")
        celn7o4qrk = wpfloat("0.0")
        celn27o16qrk = wpfloat("0.0")
        celn13o8qrk = wpfloat("0.0")

    ##----------------------------------------------------------------------------
    ## 2.5: IF (llqs): ic2
    ##----------------------------------------------------------------------------

    # ** GZ: the following computation differs substantially from the corresponding code in cloudice **
    if llqs:
        clnrhoqs = log(rhoqs)
        cssmax = rhoqs_intermediate / rho * dtr  # GZ: shifting this computation ahead of the IF condition changes results#
        if qi + qc > icon_graupel_params.qmin:
            celn3o4qsk = exp(3.0 / 4.0 * clnrhoqs)
        else:
            celn3o4qsk = wpfloat("0.0")
        celn8qsk = exp(0.8 * clnrhoqs)
    else:
        cssmax = wpfloat("0.0")
        celn3o4qsk = wpfloat("0.0")
        celn8qsk = wpfloat("0.0")

    ##----------------------------------------------------------------------------
    ## 2.6: IF (llqg): ic3
    ##----------------------------------------------------------------------------

    if llqg:
        clnrhoqg = log(rhoqg)
        csgmax = rhoqg_intermediate / rho * dtr
        if qi + qc > icon_graupel_params.qmin:
            celnrimexp_g = exp(icon_graupel_params.graupel_exp_rim * clnrhoqg)
        else:
            celnrimexp_g = wpfloat("0.0")
        celn6qgk = exp(0.6 * clnrhoqg)
    else:
        csgmax = wpfloat("0.0")
        celnrimexp_g = wpfloat("0.0")
        celn6qgk = wpfloat("0.0")

        ##----------------------------------------------------------------------------
    ## 2.7:  slope of snow PSD and coefficients for depositional growth (llqi,llqs)
    ##----------------------------------------------------------------------------

    if llqi | llqs:
        cdvtp = icon_graupel_params.ccdvtp * exp(1.94 * log(temperature)) / pres
        chi = icon_graupel_params.ccshi1 * cdvtp * rho * qvsi / (temperature * temperature)
        chlp = cdvtp / (1.0 + chi)
        cidep = icon_graupel_params.ccidep * chlp

        if llqs:
            cslam = exp(icon_graupel_params.ccslxp * log(icon_graupel_params.ccslam * n0s / rhoqs))
            cslam = minimum(cslam, 1.0e15)
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

    # ------------------------------------------------------------------------------
    # Description:
    #   This subroutine computes the heterogeneous ice deposition nucleation rate.
    #
    #   ice nucleation rate = ice_initial_mass Ni / rho / dt, Eq. 5.101
    #   ice_initial_mass is the initial ice crystal mass
    #
    # ------------------------------------------------------------------------------

    if ((temperature < icon_graupel_params.heterogeneous_freeze_temperature) & (qv > 8.e-6) & (qi <= wpfloat("0.0")) & (
        qv > qvsi)):
        snucl_v2i = icon_graupel_params.ice_initial_mass * c1orho * cnin * dtr

    # --------------------------------------------------------------------------
    # Section 3: Search for cloudy grid points with cloud water and
    #            calculation of the conversion rates involving qc (ic6)
    # --------------------------------------------------------------------------

    ##----------------------------------------------------------------------------
    ## 3.1: Autoconversion of clouds and accretion of rain
    ##----------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # Description:
    #   This subroutine computes the rate of autoconversion and accretion by rain.
    #
    #   Method 1: liquid_autoconversion_option = 1, Kessler (1969)
    #   Method 2: liquid_autoconversion_option = 2, Seifert and beheng (2001)
    #
    # ------------------------------------------------------------------------------

    # if there is cloud water and the temperature is above homogeneuous freezing temperature
    if (llqc & (temperature > icon_graupel_params.homogeneous_freeze_temperature)):

        if liquid_autoconversion_option == 1:
            # Kessler(1969) autoconversion rate
            scaut_c2r = icon_graupel_params.kessler_cloud2rain_autoconversion_coeff_for_cloud * maximum(
                qc - icon_graupel_params.qc0, wpfloat("0.0"))
            scacr_c2r = icon_graupel_params.kessler_cloud2rain_autoconversion_coeff_for_rain * qc * celn7o8qrk

        elif liquid_autoconversion_option == 2:
            # Seifert and Beheng (2001) autoconversion rate
            local_const = icon_graupel_params.kcau / (20.0 * icon_graupel_params.xstar) * (
                    icon_graupel_params.cnue + 2.0) * (icon_graupel_params.cnue + 4.0) / (
                                  icon_graupel_params.cnue + 1.0) ** 2.0

            # with constant cloud droplet number concentration qnc
            if qc > 1.0e-6:
                local_tau = minimum(1.0 - qc / (qc + qr), 0.9)
                local_tau = maximum(local_tau, 1.e-30)
                local_hlp = exp(icon_graupel_params.kphi2 * log(local_tau))
                local_phi = icon_graupel_params.kphi1 * local_hlp * (1.0 - local_hlp) ** 3.0
                scaut_c2r = local_const * qc * qc * qc * qc / (qnc * qnc) * (1.0 + local_phi / (1.0 - local_tau) ** 2.0)
                local_phi = (local_tau / (local_tau + icon_graupel_params.kphi3)) ** 4.0
                scacr_c2r = icon_graupel_params.kcac * qc * qr * local_phi

    ##----------------------------------------------------------------------------
    ## 3.2: Cloud and rain freezing in clouds
    ##----------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
    # Description:
    #   This subroutine computes the freezing rate of rain in clouds.
    #
    #   Method 1: rain_freezing_option = 1, Eq. 5.168
    #   Method 2 (discarded): rain_freezing_option = 2, Eq. 5.126, an approximation of combined immersion Eq. 5.80 and contact freezing Eq. 5.83 (ABANDONED)
    #
    # ------------------------------------------------------------------------------

    # if there is cloud water, and the temperature is above homogeneuous freezing temperature
    if llqc:
        if temperature > icon_graupel_params.homogeneous_freeze_temperature:
            # Calculation of in-cloud rainwater freezing
            if llqr & (temperature < icon_graupel_params.threshold_freeze_temperature) & (qr > 0.1 * qc):
                srfrz_r2g = icon_graupel_params.coeff_rain_freeze1_mode1 * (
                        exp(icon_graupel_params.coeff_rain_freeze2_mode1 * (
                                icon_graupel_params.threshold_freeze_temperature - temperature)) - 1.0) * celn7o4qrk
        else:
            # tg <= tg: ! hom. freezing of cloud and rain water
            scfrz_c2i = cscmax
            srfrz_r2g = csrmax

    ##----------------------------------------------------------------------------
    ## 3.3: Riming in clouds
    ##----------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
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
    # ------------------------------------------------------------------------------

    # if there is cloud water and the temperature is above homogeneuous freezing temperature
    if llqc & (temperature > icon_graupel_params.homogeneous_freeze_temperature):

        if llqs:
            srims_c2s = crim * qc * exp(icon_graupel_params.ccsaxp * log(cslam))

        srimg_c2g = icon_graupel_params.crim_g * qc * celnrimexp_g

        if temperature >= icon_graupel_params.tmelt:
            sshed_c2r = srims_c2s + srimg_c2g
            srims_c2s = wpfloat("0.0")
            srimg_c2g = wpfloat("0.0")
        else:
            if qc >= icon_graupel_params.qc0:
                scosg_s2g = icon_graupel_params.csg * qc * celn3o4qsk

    ##----------------------------------------------------------------------------
    ## 3.4: Ice nucleation in clouds
    ##----------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
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
    # ------------------------------------------------------------------------------

    # if there is cloud water
    if llqc & (temperature <= 267.15) & (qi <= icon_graupel_params.qmin):
        snucl_v2i = icon_graupel_params.ice_initial_mass * c1orho * cnin * dtr

    ##----------------------------------------------------------------------------
    ## 3.5: Reduced deposition in clouds
    ##----------------------------------------------------------------------------

    if llqc:
        if (k_lev > 0) & (not is_surface):

            cqcgk_1 = qi_kup + qs_kup + qg_kup

            # distance from cloud top
            if (qv_kup + qc_kup < qvsw_kup) & (cqcgk_1 < icon_graupel_params.qmin):
                # upper cloud layer
                dist_cldtop = wpfloat("0.0")  # reset distance to upper cloud layer
            else:
                dist_cldtop = dist_cldtop + dz

    if llqc:
        if (k_lev > 0) & (not is_surface):
            # finalizing transfer rates in clouds and calculate depositional growth reduction
            # TODO (Chia Rui): define a field operator for the ice nuclei number concentration, cnin = _fxna_cooper(temperature)
            cnin_cooper = 5.0 * exp(0.304 * (icon_graupel_params.tmelt - temperature))
            cnin_cooper = minimum(cnin_cooper, icon_graupel_params.nimax_Thom)
            cfnuc = minimum(cnin_cooper / icon_graupel_params.nimix, 1.0)

            # with asymptotic behaviour dz -> 0 (xxx)
            #        reduce_dep = MIN(fnuc + (1.0_wp-fnuc)*(reduce_dep_ref + &
            #                             dist_cldtop(iv)/dist_cldtop_ref + &
            #                             (1.0_wp-reduce_dep_ref)*(zdh/dist_cldtop_ref)**4), 1.0_wp)

            # without asymptotic behaviour dz -> 0
            reduce_dep = cfnuc + (1.0 - cfnuc) * (
                    icon_graupel_params.reduce_dep_ref + dist_cldtop / icon_graupel_params.dist_cldtop_ref)
            reduce_dep = minimum(reduce_dep, 1.0)

    # ------------------------------------------------------------------------
    # Section 4: Search for cold grid points with cloud ice and/or snow and
    #            calculation of the conversion rates involving qi, qs and qg
    # ------------------------------------------------------------------------

    ##----------------------------------------------------------------------------
    ## 4.1: Aggregation in ice clouds
    ## 4.2: Autoconversion of ice
    ## 4.3: Riming between rain and ice in ice clouds
    ##----------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
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

    # local_eff = 0.0
    if (temperature <= icon_graupel_params.tmelt) & llqi:

        # Change in sticking efficiency needed in case of cloud ice sedimentation
        # (based on Guenther Zaengls work)
        local_eff = minimum(exp(0.09 * (temperature - icon_graupel_params.tmelt)), 1.0)
        local_eff = maximum(local_eff, ice_stickeff_min)
        local_eff = maximum(local_eff, icon_graupel_params.stick_eff_fac * (
                temperature - icon_graupel_params.tmin_iceautoconv))

        local_nid = rho * qi / cmi
        local_lnlogmi = log(cmi)

        local_qvsidiff = qv - qvsi
        local_svmax = local_qvsidiff * dtr

        saggs_i2s = local_eff * qi * cagg * exp(icon_graupel_params.ccsaxp * log(cslam))
        saggg_i2g = local_eff * qi * icon_graupel_params.cagg_g * celnrimexp_g
        siaut_i2s = local_eff * icon_graupel_params.ciau * maximum(qi - icon_graupel_params.qi0, wpfloat("0.0"))

        sicri_i2g = icon_graupel_params.cicri * qi * celn7o8qrk
        if (qs > 1.e-7):
            srcri_r2g = icon_graupel_params.crcri * (qi / cmi) * celn13o8qrk

        local_icetotaldeposition = cidep * local_nid * exp(0.33 * local_lnlogmi) * local_qvsidiff
        sidep_v2i = local_icetotaldeposition
        # szdep_v2i = 0.0
        # szsub_v2i = 0.0

        # for sedimenting quantities the maximum
        # allowed depletion is determined by the predictor value.
        local_simax = rhoqi_intermediate * c1orho * dtr

        if local_icetotaldeposition > wpfloat("0.0"):
            local_icetotaldeposition = local_icetotaldeposition * reduce_dep  # FR new: depositional growth reduction
            szdep_v2i = minimum(local_icetotaldeposition, local_svmax)
        elif local_icetotaldeposition < wpfloat("0.0"):
            szsub_v2i = maximum(local_icetotaldeposition, local_svmax)
            szsub_v2i = - maximum(szsub_v2i, -local_simax)

        local_lnlogmi = log(icon_graupel_params.msmin / cmi)
        local_ztau = 1.5 * (exp(0.66 * local_lnlogmi) - 1.0)
        sdaut_i2s = szdep_v2i / local_ztau

    ##----------------------------------------------------------------------------
    ## 4.4: Vapor deposition in ice clouds
    ##----------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
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
    # ------------------------------------------------------------------------------

    if llqi | llqs | llqg:

        if temperature <= icon_graupel_params.tmelt:

            local_qvsidiff = qv - qvsi
            local_svmax = local_qvsidiff * dtr

            local_xfac = 1.0 + cbsdep * exp(icon_graupel_params.ccsdxp * log(cslam))
            ssdep_v2s = csdep * local_xfac * local_qvsidiff / (cslam + icon_graupel_params.eps) ** 2.0
            # FR new: depositional growth reduction
            if ssdep_v2s > wpfloat("0.0"):
                ssdep_v2s = ssdep_v2s * reduce_dep

            # GZ: This limitation, which was missing in the original graupel scheme,
            # is crucial for numerical stability in the tropics!
            if ssdep_v2s > wpfloat("0.0"):
                ssdep_v2s = minimum(ssdep_v2s, local_svmax - szdep_v2i)
            # Suppress depositional growth of snow if the existing amount is too small for a
            # a meaningful distiction between cloud ice and snow
            if qs <= 1.e-7:
                ssdep_v2s = minimum(ssdep_v2s, wpfloat("0.0"))
            # ** GZ: this numerical fit should be replaced with a physically more meaningful formulation **
            sgdep_v2g = (
                            0.398561 -
                            0.00152398 * temperature +
                            2554.99 / pres +
                            2.6531e-7 * pres
                        ) * local_qvsidiff * celn6qgk

    # ------------------------------------------------------------------------
    # Section 5: Search for warm grid points with cloud ice and/or snow and
    #            calculation of the melting rates of qi and ps
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
    # Description:
    #   This subroutine computes the vapor deposition of ice crystals, snow, and graupel in ice clouds when temperature is above zero degree celcius.
    #
    #
    #   Ice crystals completely melt when temperature is above zero.
    #
    #   For snow and graupel, follow Eqs. 5.141 - 5.146
    #
    # ------------------------------------------------------------------------------

    if llqi | llqs | llqg:

        if temperature > icon_graupel_params.tmelt:

            # cloud ice melts instantaneously
            simlt_i2c = rhoqi_intermediate * c1orho * dtr

            local_qvsw0 = icon_graupel_params.pvsw0 / (rho * icon_graupel_params.rv * icon_graupel_params.tmelt)
            local_qvsw0diff = qv - local_qvsw0

            # ** GZ: several numerical fits in this section should be replaced with physically more meaningful formulations **
            if temperature > icon_graupel_params.tmelt - icon_graupel_params.tcrit * local_qvsw0diff:
                # calculate melting rate
                local_x1 = temperature - icon_graupel_params.tmelt + icon_graupel_params.asmel * local_qvsw0diff
                ssmlt_s2r = (79.6863 / pres + 0.612654e-3) * local_x1 * celn8qsk
                ssmlt_s2r = minimum(ssmlt_s2r, cssmax)
                sgmlt_g2r = (12.31698 / pres + 7.39441e-05) * local_x1 * celn6qgk
                sgmlt_g2r = minimum(sgmlt_g2r, csgmax)
                # deposition + melting, ice particle temperature: t0
                # calculation without howell-factor!
                ssdep_v2s = (31282.3 / pres + 0.241897) * local_qvsw0diff * celn8qsk
                sgdep_v2g = (0.153907 - pres * 7.86703e-07) * local_qvsw0diff * celn6qgk
                if local_qvsw0diff < wpfloat("0.0"):
                    # melting + evaporation of snow/graupel
                    ssdep_v2s = maximum(-cssmax, ssdep_v2s)
                    sgdep_v2g = maximum(-csgmax, sgdep_v2g)
                    # melt water evaporates
                    ssmlt_s2r = ssmlt_s2r + ssdep_v2s
                    sgmlt_g2r = sgmlt_g2r + sgdep_v2g
                    ssmlt_s2r = maximum(ssmlt_s2r, wpfloat("0.0"))
                    sgmlt_g2r = maximum(sgmlt_g2r, wpfloat("0.0"))
                else:
                    # deposition on snow/graupel is interpreted as increase
                    # in rain water ( qv --> qr, sconr)
                    # therefore,  sconr=(zssdep+zsgdep)
                    sconr_v2r = ssdep_v2s + sgdep_v2g
                    ssdep_v2s = wpfloat("0.0")
                    sgdep_v2g = wpfloat("0.0")
            else:
                # if t<t_crit
                # no melting, only evaporation of snow/graupel
                # local_qvsw      = sat_pres_water(input_t) / (input_rho * icon_graupel_params.rv * input_t)
                # output_qvsw_kup = local_qvsw ! redundant in the original code?
                local_qvsidiff = qv - qvsw
                ssdep_v2s = (0.28003 - pres * 0.146293e-6) * local_qvsidiff * celn8qsk
                sgdep_v2g = (0.0418521 - pres * 4.7524e-8) * local_qvsidiff * celn6qgk
                ssdep_v2s = maximum(-cssmax, ssdep_v2s)
                sgdep_v2g = maximum(-csgmax, sgdep_v2g)

    # --------------------------------------------------------------------------
    # Section 6: Search for grid points with rain in subsaturated areas
    #            and calculation of the evaporation rate of rain
    # --------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
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
    #          Method 2 (discarded): rain_freezing_option = 2, Eq. 5.126, an approximation of combined immersion Eq. 5.80 and contact freezing Eq. 5.83 (ABANDONED)
    #
    # ------------------------------------------------------------------------------

    if llqr & (qv + qc <= qvsw):

        local_lnqr = log(rhoqr)
        local_x1 = 1.0 + bev * exp(bevxp * local_lnqr)
        # sev  = zcev*zx1*(zqvsw - qvg) * EXP (zcevxp  * zlnqrk)
        # Limit evaporation rate in order to avoid overshoots towards supersaturation
        # the pre-factor approximates (esat(T_wb)-e)/(esat(T)-e) at temperatures between 0 degC and 30 degC
        local_temp_c = temperature - icon_graupel_params.tmelt
        local_maxevap = (0.61 - 0.0163 * local_temp_c + 1.111e-4 * local_temp_c ** 2.0) * (qvsw - qv) / dt
        sevap_r2v = cev * local_x1 * (qvsw - qv) * exp(cevxp * local_lnqr)
        sevap_r2v = minimum(sevap_r2v, local_maxevap)

        if temperature > icon_graupel_params.homogeneous_freeze_temperature:
            # Calculation of below-cloud rainwater freezing
            if temperature < icon_graupel_params.threshold_freeze_temperature:
                # FR new: reduced rain freezing rate
                srfrz_r2g = icon_graupel_params.coeff_rain_freeze1_mode1 * (
                        exp(icon_graupel_params.coeff_rain_freeze2_mode1 * (
                                icon_graupel_params.threshold_freeze_temperature - temperature)) - 1.0) * celn7o4qrk
        else:  # Hom. freezing of rain water
            srfrz_r2g = csrmax

    # --------------------------------------------------------------------------
    # Section 7: Calculate the total tendencies of the prognostic variables.
    #            Update the prognostic variables in the interior domain.
    # --------------------------------------------------------------------------

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
            csimax = rhoqi_intermediate * c1orho * dtr

            # Check for maximal depletion of cloud ice
            # No check is done for depositional autoconversion (sdau) because
            # this is a always a fraction of the gain rate due to
            # deposition (i.e the sum of this rates is always positive)
            csum = siaut_i2s + saggs_i2s + saggg_i2g + sicri_i2g + szsub_v2i
            ccorr = csimax / maximum(csimax, csum) if csimax > wpfloat("0.0") else wpfloat("0.0")
            sidep_v2i = szdep_v2i - ccorr * szsub_v2i
            siaut_i2s = ccorr * siaut_i2s
            saggs_i2s = ccorr * saggs_i2s
            saggg_i2g = ccorr * saggg_i2g
            sicri_i2g = ccorr * sicri_i2g
            if qvsidiff < wpfloat("0.0"):
                ssdep_v2s = maximum(ssdep_v2s, -cssmax)
                sgdep_v2g = maximum(sgdep_v2g, -csgmax)

    csum = sevap_r2v + srfrz_r2g + srcri_r2g
    ccorr = csrmax / maximum(csrmax, csum) if csum > wpfloat("0.0") else wpfloat("1.0")
    sevap_r2v = ccorr * sevap_r2v
    srfrz_r2g = ccorr * srfrz_r2g
    srcri_r2g = ccorr * srcri_r2g

    # limit snow depletion in order to avoid negative values of qs
    ccorr = wpfloat("1.0")
    if ssdep_v2s <= wpfloat("0.0"):
        Csum = ssmlt_s2r + scosg_s2g - ssdep_v2s
        if Csum > wpfloat("0.0"): ccorr = cssmax / maximum(cssmax, csum)
        ssmlt_s2r = ccorr * ssmlt_s2r
        scosg_s2g = ccorr * scosg_s2g
        ssdep_v2s = ccorr * ssdep_v2s
    else:
        csum = ssmlt_s2r + scosg_s2g
        if csum > wpfloat("0.0"): ccorr = cssmax / maximum(cssmax, csum)
        ssmlt_s2r = ccorr * ssmlt_s2r
        scosg_s2g = ccorr * scosg_s2g

    cqvt = sevap_r2v - sidep_v2i - ssdep_v2s - sgdep_v2g - snucl_v2i - sconr_v2r
    cqct = simlt_i2c - scaut_c2r - scfrz_c2i - scacr_c2r - sshed_c2r - srims_c2s - srimg_c2g
    cqit = snucl_v2i + scfrz_c2i - simlt_i2c - sicri_i2g + sidep_v2i - sdaut_i2s - saggs_i2s - saggg_i2g - siaut_i2s
    cqrt = scaut_c2r + sshed_c2r + scacr_c2r + ssmlt_s2r + sgmlt_g2r - sevap_r2v - srcri_r2g - srfrz_r2g + sconr_v2r
    cqst = siaut_i2s + sdaut_i2s - ssmlt_s2r + srims_c2s + ssdep_v2s + saggs_i2s - scosg_s2g
    cqgt = saggg_i2g - sgmlt_g2r + sicri_i2g + srcri_r2g + sgdep_v2g + srfrz_r2g + srimg_c2g + scosg_s2g

    temperature_tendency = heat_cap_r * (lhv * (cqct + cqrt) + lhs * (cqit + cqst + cqgt))
    qi_tendency = maximum((rhoqi_intermediate * c1orho * cimi - qi) / dt + cqit, -qi / dt)
    qr_tendency = maximum((rhoqr_intermediate * c1orho * cimr - qr) / dt + cqrt, -qr / dt)
    qs_tendency = maximum((rhoqs_intermediate * c1orho * cims - qs) / dt + cqst, -qs / dt)
    qg_tendency = maximum((rhoqg_intermediate * c1orho * cimg - qg) / dt + cqgt, -qg / dt)
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
        k_lev
    )


@field_operator
def _graupel(
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
   l_cv: bool,
   ithermo_water: int32
)-> tuple[
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
        k_lev,
        Szdep_v2i,
        Szsub_v2i,
        Snucl_v2i,
        Scfrz_c2i,
        Simlt_i2c,
        Sicri_i2g,
        Sidep_v2i,
        Sdaut_i2s,
        Saggs_i2s,
        Saggg_i2g,
        Siaut_i2s,
        Ssmlt_s2r,
        Srims_c2s,
        Ssdep_v2s,
        Scosg_s2g,
        Sgmlt_g2r,
        Srcri_r2g,
        Sgdep_v2g,
        Srfrz_r2g,
        Srimg_c2g
    ) = _graupel_scan(
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
        l_cv,
        ithermo_water
    )

    return(
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
        Vnew_i
    )




@field_operator
def _graupel_t_tendency(
    dt: float64,
    temperature_new: Field[[CellDim,KDim], float64],
    temperature_old: Field[[CellDim,KDim], float64]
) -> Field[[CellDim,KDim], float64]:
    Cdtr = 1.0 / dt
    return (temperature_new - temperature_old) * Cdtr

@field_operator
def _graupel_q_tendency(
    dt: float64,
    qv_new: Field[[CellDim, KDim], float64],
    qc_new: Field[[CellDim, KDim], float64],
    qi_new: Field[[CellDim, KDim], float64],
    qr_new: Field[[CellDim, KDim], float64],
    qs_new: Field[[CellDim, KDim], float64],
    qv_old: Field[[CellDim, KDim], float64],
    qc_old: Field[[CellDim, KDim], float64],
    qi_old: Field[[CellDim, KDim], float64],
    qr_old: Field[[CellDim, KDim], float64],
    qs_old: Field[[CellDim, KDim], float64]
) -> tuple[
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
        maximum( -qs_old * Cdtr , (qs_new - qs_old) * Cdtr )
    )


@scan_operator(
    axis=KDim,
    forward=True,
    init=(
        *(0.0,) * 5,  # rain, snow, graupel, ice, solid predicitation fluxes
        0
    ),
)
def _graupel_flux_scan(
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
    Vnew_r: float64,
    Vnew_s: float64,
    Vnew_g: float64,
    Vnew_i: float64,
    rhoqrV_old_kup: float64,
    rhoqsV_old_kup: float64,
    rhoqgV_old_kup: float64,
    rhoqiV_old_kup: float64,
    lpres_pri: bool,
    ldass_lhn: bool  # if true, latent heat nudging is applied
):
    # unpack carry
    (
        prr_gsp_kup,
        prs_gsp_kup,
        prg_gsp_kup,
        pri_gsp_kup,
        qrsflux_kup,
        k_lev
    ) = state_kup

    # | (k_lev > kend_moist)
    if ( k_lev < kstart_moist ):
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


    qrsflux = 0.0
    prr_gsp = 0.0
    prs_gsp = 0.0
    prg_gsp = 0.0
    pri_gsp = 0.0

    rhoqrV_new_kup = qr * rho * Vnew_r
    rhoqsV_new_kup = qs * rho * Vnew_s
    rhoqgV_new_kup = qg * rho * Vnew_g
    rhoqiV_new_kup = qi * rho * Vnew_i
    if ( rhoqrV_new_kup <= graupel_const.GrConst_qmin ): rhoqrV_new_kup = 0.0
    if ( rhoqsV_new_kup <= graupel_const.GrConst_qmin ): rhoqsV_new_kup = 0.0
    if ( rhoqgV_new_kup <= graupel_const.GrConst_qmin ): rhoqgV_new_kup = 0.0
    if ( rhoqiV_new_kup <= graupel_const.GrConst_qmin ): rhoqiV_new_kup = 0.0

    # good solution provided by Nikki to know where I am along the KDim axis
    if ( k_lev == kend ):

        # Precipitation fluxes at the ground
        prr_gsp = 0.5 * (qr * rho * Vnew_r + rhoqrV_old_kup)
        if ( graupel_const.GrConst_lsedi_ice & lpres_pri ):
            prs_gsp = 0.5 * (rho * qs * Vnew_s + rhoqsV_old_kup)
            pri_gsp = 0.5 * (rho * qi * Vnew_i + rhoqiV_old_kup)
        elif ( graupel_const.GrConst_lsedi_ice ):
            prs_gsp = 0.5 * (rho * (qs * Vnew_s + qi * Vnew_i) + rhoqsV_old_kup + rhoqiV_old_kup)
        else:
            prs_gsp = 0.5 * (qs * rho * Vnew_s + rhoqsV_old_kup)
        prg_gsp = 0.5 * (qg * rho * Vnew_g + rhoqgV_old_kup)

        # for the latent heat nudging
        if ( ldass_lhn ):  # THEN default: true
            qrsflux = prr_gsp + prs_gsp + prg_gsp

    else:

        # for the latent heat nudging
        if ( ldass_lhn ):  # THEN default: true
            if (graupel_const.GrConst_lsedi_ice):
                qrsflux = rhoqrV_new_kup + rhoqsV_new_kup + rhoqgV_new_kup + rhoqiV_new_kup
                qrsflux = 0.5 * (qrsflux + rhoqrV_old_kup + rhoqsV_old_kup + rhoqgV_old_kup + rhoqiV_old_kup)
            else:
                qrsflux = rhoqrV_new_kup + rhoqsV_new_kup + rhoqgV_new_kup
                qrsflux = 0.5 * (qrsflux + rhoqrV_old_kup + rhoqsV_old_kup + rhoqgV_old_kup)

    # tracing current k level
    k_lev = k_lev + int32(1)

    return(
        prr_gsp,
        prs_gsp,
        prg_gsp,
        pri_gsp,
        qrsflux,
        k_lev
    )

'''
def graupel_wrapper(
   # Grid information
   dt: float64,  # time step
   dz: Field[[CellDim, KDim], float64],  # level thickness
   # Prognostics
   temperature: Field[[CellDim, KDim], float64],
   pres: Field[[CellDim, KDim], float64],
   rho: Field[[CellDim, KDim], float64],
   qv: Field[[CellDim, KDim], float64],
   qc: Field[[CellDim, KDim], float64],
   qi: Field[[CellDim, KDim], float64],
   qr: Field[[CellDim, KDim], float64],
   qs: Field[[CellDim, KDim], float64],
   qg: Field[[CellDim, KDim], float64],
   # Number Densities
   qnc: Field[[CellDim, KDim], float64], # originally 2D Field, now 3D Field
   # Option Switches
   l_cv: bool,
   lpres_pri: bool,
   ithermo_water: int32,
   ldass_lhn: bool, # if true, latent heat nudging is applied
   ldiag_ttend: bool,  # if true, temperature tendency shall be diagnosed
   ldiag_qtend: bool,  # if true, moisture tendencies shall be diagnosed
   # Grid indices
   cell_size: int32,
   k_size: int32,
   kstart_moist: int32,
   kend: int32,
   kend_moist: int32,
):

    temperature_ = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    qv_ = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    qc_ = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    qi_ = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    qr_ = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    qs_ = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    qg_ = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))

    #rho_kup = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    #Crho1o2_kup = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    #Crhofac_qi_kup = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    #Cvz0s_kup = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    rhoqrV_old_kup = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    rhoqsV_old_kup = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    rhoqgV_old_kup = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    rhoqiV_old_kup = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    Vnew_r = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    Vnew_s = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    Vnew_g = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    Vnew_i = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    #dist_cldtop = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    #qvsw_kup = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    #k_lev = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=int32))


    _graupel(
        kstart_moist,
        kend,
        dt,
        dz,
        temperature,
        pres,
        rho,
        qv,
        qi,
        qc,
        qr,
        qs,
        qg,
        qnc,
        l_cv,
        ithermo_water,
        out=(
            temperature_,
            qv_,
            qc_,
            qi_,
            qr_,
            qs_,
            qg_,
            # used in graupel scheme, do not output to outer world
            rhoqrV_old_kup,
            rhoqsV_old_kup,
            rhoqgV_old_kup,
            rhoqiV_old_kup,
            Vnew_r,
            Vnew_s,
            Vnew_g,
            Vnew_i
        ),
        offset_provider={}
    )


    ddt_tend_t = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    ddt_tend_qv = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    ddt_tend_qc = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    ddt_tend_qi = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    ddt_tend_qr = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    ddt_tend_qs = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    ddt_tend_qg = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))

    prr_gsp = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    prs_gsp = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    pri_gsp = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    prg_gsp = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    qrsflux = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))


    if ( ldiag_ttend ):
        _graupel_t_tendency(
            dt,
            temperature_,
            temperature,
            out=(
                ddt_tend_t
            ),
            offset_provider={}
        )

    if ( ldiag_qtend ):
        _graupel_q_tendency(
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
            qg,
            out=(
                ddt_tend_qv,
                ddt_tend_qc,
                ddt_tend_qi,
                ddt_tend_qr,
                ddt_tend_qs,
                ddt_tend_qg
            ),
            offset_provider={}
        )


    _graupel_flux(
        kstart_moist,
        kend,
        rho,
        qr_,
        qs_,
        qi_,
        qg_,
        Vnew_r,
        Vnew_s,
        Vnew_i,
        Vnew_g,
        rhoqrV_old_kup,
        rhoqsV_old_kup,
        rhoqiV_old_kup,
        rhoqgV_old_kup,
        lpres_pri,
        ldass_lhn,
        out=(
            prr_gsp,
            prs_gsp,
            pri_gsp,
            prg_gsp,
            qrsflux
        ),
        offset_provider={}
    )


    return (
        temperature_,
        qv_,
        qc_,
        qi_,
        qr_,
        qs_,
        qg_,
        ddt_tend_t,
        ddt_tend_qv,
        ddt_tend_qc,
        ddt_tend_qi,
        ddt_tend_qr,
        ddt_tend_qs,
        ddt_tend_qg,
        prr_gsp,
        prs_gsp,
        pri_gsp,
        prg_gsp,
        qrsflux
    )
'''

@program
def graupel(
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
    qnc: Field[[CellDim, KDim], float64],  # originally 2D Field, now 3D Field
    l_cv: bool,
    ithermo_water: int32,
    rhoqrV_old_kup: Field[[CellDim, KDim], float64],
    rhoqsV_old_kup: Field[[CellDim, KDim], float64],
    rhoqgV_old_kup: Field[[CellDim, KDim], float64],
    rhoqiV_old_kup: Field[[CellDim, KDim], float64],
    Vnew_r: Field[[CellDim, KDim], float64],
    Vnew_s: Field[[CellDim, KDim], float64],
    Vnew_g: Field[[CellDim, KDim], float64],
    Vnew_i: Field[[CellDim, KDim], float64]
):
    _graupel(
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
        l_cv,
        ithermo_water,
        out = (
            temperature,
            qv,
            qc,
            qi,
            qr,
            qs,
            qg,
            rhoqrV_old_kup,
            rhoqsV_old_kup,
            rhoqgV_old_kup,
            rhoqiV_old_kup,
            Vnew_r,
            Vnew_s,
            Vnew_g,
            Vnew_i
        )
    )

#(backend=roundtrip.executor)
