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
        0.0, # temperature
        0.0, # qv
        0.0, # qc
        0.0, # qi
        0.0, # qr
        0.0, # qs
        0.0, # qg
        0.0, # rhoqrV
        0.0, # rhoqsV
        0.0, # rhoqgV
        0.0, # rhoqiV
        0.0, # newV_r
        0.0, # newV_s
        0.0, # newV_g
        0.0, # newV_i
        0.0, # cloud top distance
        0.0, # density
        0.0, # density factor
        0.0, # density factor for ice
        0.0, # snow intercept parameter
        0.0, # saturation pressure
        0,   # k level
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ),
)
def _graupel_scan(
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
        int32,
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
        float64
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
    l_cv: bool,
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
        k_lev,
        Szdep_v2i_kup,
        Szsub_v2i_kup,
        Snucl_v2i_kup,
        Scfrz_c2i_kup,
        Simlt_i2c_kup,
        Sicri_i2g_kup,
        Sidep_v2i_kup,
        Sdaut_i2s_kup,
        Saggs_i2s_kup,
        Saggg_i2g_kup,
        Siaut_i2s_kup,
        Ssmlt_s2r_kup,
        Srims_c2s_kup,
        Ssdep_v2s_kup,
        Scosg_s2g_kup,
        Sgmlt_g2r_kup,
        Srcri_r2g_kup,
        Sgdep_v2g_kup,
        Srfrz_r2g_kup,
        Srimg_c2g_kup,
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
            k_lev,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        )

    is_surface = True if ( k_lev == kend ) else False


    # Define reciprocal of heat capacity of dry air (at constant pressure vs at constant volume)
    Cheat_cap_r = phy_const.rcvd if ( l_cv ) else phy_const.rcpd

    # timestep for calculations
    Cdtr  = 1.0 / dt

    # Latent heats
    # Default themodynamic is constant latent heat
    # tg = make_normalized(t(iv,k))
    # Calculate Latent heats if necessary
    # function called: CLHv = latent_heat_vaporization(temperature)
    # function called: CLHs = latent_heat_sublimation(temperature)
    CLHv = phy_const.alv + (graupel_const.GrConst_cp_v - phy_const.clw) * (temperature - phy_const.tmelt) - phy_const.rv * temperature if (ithermo_water != int32(0)) else phy_const.alv
    CLHs = phy_const.als + (graupel_const.GrConst_cp_v - graupel_const.GrConst_ci) * (temperature - phy_const.tmelt) - phy_const.rv * temperature if (ithermo_water != int32(0)) else phy_const.als

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
    Chlp       = log(graupel_const.GrConst_rho0 * C1orho)
    Crho1o2    = exp(Chlp * graupel_const.GrConst_x1o2)
    Crhofac_qi = exp(Chlp * graupel_const.GrConst_icesedi_exp)

    # NOT WORKING, NOT SURE WHY THIS WILL LEAD TO DIVISION BY ZERO
    #if (rho_kup == 0.0):
    # Crho1o2_kup = 1.0
    # Crhofac_qi_kup = 1.0
    #else:
    #    C1orho_kup     = 1.0 / rho_kup
    #    Chlp_kup       = log(graupel_const.GrConst_rho0 * C1orho_kup)
    #    Crho1o2_kup    = exp(Chlp_kup * graupel_const.GrConst_x1o2)
    #    Crhofac_qi_kup = exp(Chlp_kup * graupel_const.GrConst_icesedi_exp)

    rhoqr = qr * rho
    rhoqs = qs * rho
    rhoqg = qg * rho
    rhoqi = qi * rho

    llqr = True if (rhoqr > graupel_const.GrConst_qmin) else False
    llqs = True if (rhoqs > graupel_const.GrConst_qmin) else False
    llqg = True if (rhoqg > graupel_const.GrConst_qmin) else False
    llqi = True if (rhoqi > graupel_const.GrConst_qmin) else False

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

        if ( graupel_const.GrConst_isnow_n0temp == 1 ):
            # Calculate n0s using the temperature-dependent
            # formula of Field et al. (2005)
            local_tc = temperature - phy_const.tmelt
            local_tc = minimum( local_tc , 0.0 )
            local_tc = maximum( local_tc , -40.0 )
            Cn0s = graupel_funcConst.GrFuncConst_n0s1 * exp(graupel_funcConst.GrFuncConst_n0s2 * local_tc)
            Cn0s = minimum( Cn0s , 1.0e9 )
            Cn0s = maximum( Cn0s , 1.0e6 )

        elif ( graupel_const.GrConst_isnow_n0temp == 2 ):
            # Calculate n0s using the temperature-dependent moment
            # relations of Field et al. (2005)
            local_tc = temperature - phy_const.tmelt
            local_tc = minimum( local_tc , 0.0 )
            local_tc = maximum( local_tc , -40.0 )

            local_nnr  = 3.0
            local_hlp = (
                graupel_funcConst.GrFuncConst_mma[0] +
                graupel_funcConst.GrFuncConst_mma[1]*local_tc +
                graupel_funcConst.GrFuncConst_mma[2]*local_nnr +
                graupel_funcConst.GrFuncConst_mma[3]*local_tc*local_nnr +
                graupel_funcConst.GrFuncConst_mma[4]*local_tc**2.0 +
                graupel_funcConst.GrFuncConst_mma[5]*local_nnr**2.0 +
                graupel_funcConst.GrFuncConst_mma[6]*local_tc**2.0*local_nnr +
                graupel_funcConst.GrFuncConst_mma[7]*local_tc*local_nnr**2.0 +
                graupel_funcConst.GrFuncConst_mma[8]*local_tc**3.0 +
                graupel_funcConst.GrFuncConst_mma[9]*local_nnr**3.0
            )
            local_alf = exp(local_hlp*graupel_const.GrConst_log_10) # 10.0_wp**hlp
            local_bet = (
                graupel_funcConst.GrFuncConst_mmb[0] +
                graupel_funcConst.GrFuncConst_mmb[1]*local_tc +
                graupel_funcConst.GrFuncConst_mmb[2]*local_nnr +
                graupel_funcConst.GrFuncConst_mmb[3]*local_tc*local_nnr +
                graupel_funcConst.GrFuncConst_mmb[4]*local_tc**2.0 +
                graupel_funcConst.GrFuncConst_mmb[5]*local_nnr**2.0 +
                graupel_funcConst.GrFuncConst_mmb[6]*local_tc**2.0*local_nnr +
                graupel_funcConst.GrFuncConst_mmb[7]*local_tc*local_nnr**2.0 +
                graupel_funcConst.GrFuncConst_mmb[8]*local_tc**3.0 +
                graupel_funcConst.GrFuncConst_mmb[9]*local_nnr**3.0
            )

            # Here is the exponent bms=2.0 hardwired# not ideal# (Uli Blahak)
            local_m2s = qs * rho / graupel_const.GrConst_ams  # UB rho added as bugfix
            local_m3s = local_alf * exp(local_bet * log(local_m2s))

            local_hlp = graupel_funcConst.GrFuncConst_n0s1 * exp(graupel_funcConst.GrFuncConst_n0s2 * local_tc)
            Cn0s = 13.50 * local_m2s * (local_m2s / local_m3s) ** 3.0
            Cn0s = maximum(Cn0s, 0.5 * local_hlp)
            Cn0s = minimum(Cn0s, 1.0e2 * local_hlp)
            Cn0s = minimum(Cn0s, 1.0e9)
            Cn0s = maximum(Cn0s, 1.0e6)

        else:
            Cn0s = graupel_const.GrConst_n0s0

        # compute integration factor for terminal velocity
        Cvz0s = graupel_const.GrConst_ccsvel * exp(graupel_const.GrConst_ccsvxp * log(Cn0s))
        # compute constants for riming, aggregation, and deposition processes for snow
        Crim = graupel_const.GrConst_ccsrim * Cn0s
        Cagg = graupel_const.GrConst_ccsagg * Cn0s
        Cbsdep = graupel_const.GrConst_ccsdep * sqrt(graupel_const.GrConst_v0snow)

    else:
        Cn0s = graupel_const.GrConst_n0s0
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
    if (rhoqrV_new_kup <= graupel_const.GrConst_qmin): rhoqrV_new_kup = 0.0
    if (rhoqsV_new_kup <= graupel_const.GrConst_qmin): rhoqsV_new_kup = 0.0
    if (rhoqgV_new_kup <= graupel_const.GrConst_qmin): rhoqgV_new_kup = 0.0
    if (rhoqiV_new_kup <= graupel_const.GrConst_qmin): rhoqiV_new_kup = 0.0

    rhoqr_intermediate = rhoqr / Cdtdh + rhoqrV_new_kup + rhoqrV_old_kup
    rhoqs_intermediate = rhoqs / Cdtdh + rhoqsV_new_kup + rhoqsV_old_kup
    rhoqg_intermediate = rhoqg / Cdtdh + rhoqgV_new_kup + rhoqgV_old_kup
    rhoqi_intermediate = rhoqi / Cdtdh + rhoqiV_new_kup + rhoqiV_old_kup

    #-------------------------------------------------------------------------
    # qs_sedi, qr_sedi, qg_sedi, qi_sedi:
    #-------------------------------------------------------------------------
    if ( k_lev > kstart_moist ):
        Vnew_s = Cvz0s_kup * exp(graupel_const.GrConst_ccswxp * log((qs_kup + qs) * 0.5 * rho_kup)) * Crho1o2_kup if (qs_kup + qs > graupel_const.GrConst_qmin) else 0.0
        Vnew_r = graupel_const.GrConst_vz0r * exp(graupel_const.GrConst_vzxp * log((qr_kup + qr) * 0.5 * rho_kup)) * Crho1o2_kup if (qr_kup + qr > graupel_const.GrConst_qmin) else 0.0
        Vnew_g = graupel_const.GrConst_vz0g * exp(graupel_const.GrConst_expsedg * log((qg_kup + qg) * 0.5 * rho_kup)) * Crho1o2_kup if (qg_kup + qg > graupel_const.GrConst_qmin) else 0.0
        Vnew_i = graupel_const.GrConst_vz0i * exp(graupel_const.GrConst_bvi * log((qi_kup + qi) * 0.5 * rho_kup)) * Crhofac_qi_kup if (qi_kup + qi > graupel_const.GrConst_qmin) else 0.0

    if (llqs):
        terminal_velocity = Cvz0s * exp(graupel_const.GrConst_ccswxp * log(rhoqs)) * Crho1o2
        # Prevent terminal fall speed of snow from being zero at the surface level
        if (is_surface): terminal_velocity = maximum(terminal_velocity, graupel_const.GrConst_v_sedi_snow_min)

        rhoqsV = rhoqs * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if ( Vnew_s == 0.0 ): Vnew_s = terminal_velocity * graupel_const.GrConst_ccswxp_ln1o2

    else:
        rhoqsV = 0.0

    if (llqr):
        terminal_velocity = graupel_const.GrConst_vz0r * exp(graupel_const.GrConst_vzxp * log(rhoqr)) * Crho1o2
        # Prevent terminal fall speed of snow from being zero at the surface level
        if (is_surface): terminal_velocity = maximum(terminal_velocity, graupel_const.GrConst_v_sedi_rain_min)

        rhoqrV = rhoqr * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if ( Vnew_r == 0.0 ): Vnew_r = terminal_velocity * graupel_const.GrConst_vzxp_ln1o2

    else:
        rhoqrV = 0.0

    if (llqg):
        terminal_velocity = graupel_const.GrConst_vz0g * exp(graupel_const.GrConst_expsedg * log(rhoqg)) * Crho1o2
        # Prevent terminal fall speed of snow from being zero at the surface level
        if (is_surface): terminal_velocity = maximum(terminal_velocity, graupel_const.GrConst_v_sedi_graupel_min)

        rhoqgV = rhoqg * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if ( Vnew_g == 0.0 ): Vnew_g = terminal_velocity * graupel_const.GrConst_expsedg_ln1o2

    else:
        rhoqgV = 0.0

    if (llqi):
        terminal_velocity = graupel_const.GrConst_vz0i * exp(graupel_const.GrConst_bvi * log(rhoqi)) * Crhofac_qi

        rhoqiV = rhoqi * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if (Vnew_i == 0.0): Vnew_i = terminal_velocity * graupel_const.GrConst_bvi_ln1o2

    else:
        rhoqiV = 0.0

    # Prevent terminal fall speeds of precip hydrometeors from being zero at the surface level
    if (is_surface):
        Vnew_s = maximum(Vnew_s, graupel_const.GrConst_v_sedi_snow_min)
        Vnew_r = maximum( Vnew_r, graupel_const.GrConst_v_sedi_rain_min )
        Vnew_g = maximum(Vnew_g, graupel_const.GrConst_v_sedi_graupel_min)

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
    if ( graupel_const.GrConst_lsuper_coolw ):
        # function called: Cnin = _fxna_cooper(temperature)
        Cnin = 5.0 * exp(0.304 * (phy_const.tmelt - temperature))
        Cnin = minimum( Cnin , graupel_const.GrConst_nimax )
    else:
        # function called: Cnin = _fxna(temperature)
        Cnin = 1.0e2 * exp(0.2 * (phy_const.tmelt - temperature))
        Cnin = minimum( Cnin , graupel_const.GrConst_nimax )
    Cmi = minimum( rho * qi / Cnin , graupel_const.GrConst_mimax )
    Cmi = maximum( graupel_const.GrConst_mi0 , Cmi )

    # function called: Cqvsw = sat_pres_water(temperature) / (rho * phy_const.rv * temperature)
    # function called: Cqvsi = sat_pres_ice(temperature) / (rho * phy_const.rv * temperature)
    Cqvsw = graupel_funcConst.GrFuncConst_c1es * exp( graupel_funcConst.GrFuncConst_c3les * (temperature - phy_const.tmelt) / (temperature - graupel_funcConst.GrFuncConst_c4les) ) / (rho * phy_const.rv * temperature)
    Cqvsi = graupel_funcConst.GrFuncConst_c1es * exp( graupel_funcConst.GrFuncConst_c3ies * (temperature - phy_const.tmelt) / (temperature - graupel_funcConst.GrFuncConst_c4ies) ) / (rho * phy_const.rv * temperature)
    llqr = True if (rhoqr > graupel_const.GrConst_qmin) else False
    llqs = True if (rhoqs > graupel_const.GrConst_qmin) else False
    llqg = True if (rhoqg > graupel_const.GrConst_qmin) else False
    llqi = True if (qi > graupel_const.GrConst_qmin) else False
    llqc = True if (qc > graupel_const.GrConst_qmin) else False


    ##----------------------------------------------------------------------------
    ## 2.4: IF (llqr): ic1
    ##----------------------------------------------------------------------------

    if (llqr):
        Clnrhoqr = log(rhoqr)
        Csrmax   = rhoqr_intermediate / rho * Cdtr  # GZ: shifting this computation ahead of the IF condition changes results!
        if ( qi + qc > graupel_const.GrConst_qmin ):
            Celn7o8qrk   = exp(graupel_const.GrConst_x7o8   * Clnrhoqr)
        else:
            Celn7o8qrk = 0.0
        if ( temperature < graupel_const.GrConst_trfrz ):
            Celn7o4qrk   = exp(graupel_const.GrConst_x7o4   * Clnrhoqr) #FR new
            Celn27o16qrk = exp(graupel_const.GrConst_x27o16 * Clnrhoqr)
        else:
            Celn7o4qrk = 0.0
            Celn27o16qrk = 0.0
        if (llqi):
            Celn13o8qrk  = exp(graupel_const.GrConst_x13o8  * Clnrhoqr)
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
        if ( qi + qc > graupel_const.GrConst_qmin ):
            Celn3o4qsk = exp(graupel_const.GrConst_x3o4 * Clnrhoqs)
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
        if ( qi + qc > graupel_const.GrConst_qmin ):
            Celnrimexp_g = exp(graupel_const.GrConst_rimexp_g * Clnrhoqg)
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
        Cdvtp  = graupel_const.GrConst_ccdvtp * exp(1.94 * log(temperature)) / pres
        Chi    = graupel_const.GrConst_ccshi1 * Cdvtp * rho * Cqvsi/(temperature * temperature)
        Chlp    = Cdvtp / (1.0 + Chi)
        Cidep = graupel_const.GrConst_ccidep * Chlp

        if (llqs):
            Cslam = exp(graupel_const.GrConst_ccslxp * log(graupel_const.GrConst_ccslam * Cn0s / rhoqs ))
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

    if ( (temperature < graupel_funcConst.GrFuncConst_thet) & (qv > 8.e-6) & (qi <= 0.0) & (qv > Cqvsi) ):
        Snucl_v2i = graupel_const.GrConst_mi0 * C1orho * Cnin * Cdtr


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
    if ( llqc & (temperature > graupel_const.GrConst_thn) ):

        if (graupel_const.GrConst_iautocon == 0):
            # Kessler(1969) autoconversion rate
            Scaut_c2r = graupel_funcConst.GrFuncConst_ccau  * maximum( qc - graupel_funcConst.GrFuncConst_qc0 , 0.0 )
            Scacr_c2r = graupel_funcConst.GrFuncConst_cac * qc * Celn7o8qrk

        elif (graupel_const.GrConst_iautocon == 1):
            # Seifert and Beheng (2001) autoconversion rate
            local_const = graupel_funcConst.GrFuncConst_kcau / (20.0 * graupel_funcConst.GrFuncConst_xstar) * (graupel_funcConst.GrFuncConst_cnue + 2.0) * (graupel_funcConst.GrFuncConst_cnue + 4.0) / (graupel_funcConst.GrFuncConst_cnue + 1.0)**2.0

            # with constant cloud droplet number concentration qnc
            if ( qc > 1.0e-6 ):
                local_tau = minimum( 1.0 - qc / (qc + qr) , 0.9 )
                local_tau = maximum( local_tau , 1.e-30 )
                local_hlp  = exp(graupel_funcConst.GrFuncConst_kphi2 * log(local_tau))
                local_phi = graupel_funcConst.GrFuncConst_kphi1 * local_hlp * (1.0 - local_hlp)**3.0
                Scaut_c2r = local_const * qc * qc * qc * qc / (qnc * qnc) * (1.0 + local_phi / (1.0 - local_tau)**2.0)
                local_phi = (local_tau / (local_tau + graupel_funcConst.GrFuncConst_kphi3))**4.0
                Scacr_c2r = graupel_funcConst.GrFuncConst_kcac * qc * qr * local_phi


    ##----------------------------------------------------------------------------
    ## 3.2: Cloud and rain freezing in clouds
    ##----------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    # Description:
    #   This subroutine computes the freezing rate of rain in clouds.
    #
    #   Method 1: lsuper_coolw = true, Eq. 5.80
    #   Method 2: lsuper_coolw = false, Eq. 5.126, an approximation of combined immersion Eq. 5.80 and contact freezing Eq. 5.83 (ABANDONED)
    #
    #------------------------------------------------------------------------------

    # if there is cloud water, and the temperature is above homogeneuous freezing temperature
    if ( llqc ):
        if ( temperature > graupel_const.GrConst_thn ):
            # Calculation of in-cloud rainwater freezing
            if ( llqr & (temperature < graupel_const.GrConst_trfrz) & (qr > 0.1 * qc) ):
                if ( graupel_const.GrConst_lsuper_coolw ):
                    Srfrz_r2g = graupel_const.GrConst_crfrz1 * ( exp(graupel_const.GrConst_crfrz2 * (graupel_const.GrConst_trfrz - temperature)) - 1.0 ) * Celn7o4qrk
                else:
                    local_tfrzdiff = graupel_const.GrConst_trfrz - temperature
                    Srfrz_r2g = graupel_const.GrConst_crfrz * local_tfrzdiff * sqrt(local_tfrzdiff) * Celn27o16qrk
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
    if ( llqc & (temperature > graupel_const.GrConst_thn) ):

        if ( llqs ):
            Srims_c2s = Crim * qc * exp(graupel_const.GrConst_ccsaxp * log(Cslam))

        Srimg_c2g = graupel_funcConst.GrFuncConst_crim_g * qc * Celnrimexp_g

        if ( temperature >= phy_const.tmelt ):
            Sshed_c2r = Srims_c2s + Srimg_c2g
            Srims_c2s = 0.0
            Srimg_c2g = 0.0
        else:
            if ( qc >= graupel_funcConst.GrFuncConst_qc0 ):
                Scosg_s2g = graupel_funcConst.GrFuncConst_csg * qc * Celn3o4qsk


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
    if ( llqc & (temperature <= 267.15) & (qi <= graupel_const.GrConst_qmin) ):
        if ( graupel_const.GrConst_lsuper_coolw ):
            Snucl_v2i = graupel_const.GrConst_mi0 * C1orho * Cnin * Cdtr
        else:
            Snucl_v2i = graupel_const.GrConst_mi0 / rho * Cnin * Cdtr

    ##----------------------------------------------------------------------------
    ## 3.5: Reduced deposition in clouds
    ##----------------------------------------------------------------------------

    if ( graupel_const.GrConst_lred_depgrow & llqc ):
        if ((k_lev > kstart_moist) & (k_lev < kend)):

            Cqcgk_1 = qi_kup + qs_kup + qg_kup

            # distance from cloud top
            if ( (qv_kup + qc_kup < qvsw_kup) & (Cqcgk_1 < graupel_const.GrConst_qmin) ):
                # upper cloud layer
                dist_cldtop = 0.0  # reset distance to upper cloud layer
            else:
                dist_cldtop = dist_cldtop + dz

    if ( graupel_const.GrConst_lred_depgrow & llqc ):
        if ((k_lev > kstart_moist) & (k_lev < kend)):
            # finalizing transfer rates in clouds and calculate depositional growth reduction
            # function called: Cnin_cooper = _fxna_cooper(temperature)
            Cnin_cooper = 5.0 * exp(0.304 * (phy_const.tmelt - temperature))
            Cnin_cooper = minimum(Cnin_cooper, graupel_const.GrConst_nimax)
            Cfnuc = minimum(Cnin_cooper / graupel_const.GrConst_nimix, 1.0)

            # with asymptotic behaviour dz -> 0 (xxx)
            #        reduce_dep = MIN(fnuc + (1.0_wp-fnuc)*(reduce_dep_ref + &
            #                             dist_cldtop(iv)/dist_cldtop_ref + &
            #                             (1.0_wp-reduce_dep_ref)*(zdh/dist_cldtop_ref)**4), 1.0_wp)

            # without asymptotic behaviour dz -> 0
            reduce_dep = Cfnuc + (1.0 - Cfnuc) * (graupel_const.GrConst_reduce_dep_ref + dist_cldtop / graupel_const.GrConst_dist_cldtop_ref)
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
    if ( (temperature <= phy_const.tmelt) & llqi ):

        # Change in sticking efficiency needed in case of cloud ice sedimentation
        # (based on Guenther Zaengls work)
        if ( graupel_const.GrConst_lstickeff ):
            local_eff = minimum( exp(0.09 * (temperature - phy_const.tmelt)) , 1.0 )
            local_eff = maximum( local_eff , graupel_const.GrConst_ceff_min )
            local_eff = maximum( local_eff , graupel_const.GrConst_ceff_fac * (temperature - graupel_const.GrConst_tmin_iceautoconv) )
        else: #original sticking efficiency of cloud ice
            local_eff = minimum( exp(0.09 * (temperature - phy_const.tmelt)) , 1.0 )
            local_eff = maximum( local_eff , 0.2 )

        local_nid = rho * qi / Cmi
        local_lnlogmi = log(Cmi)

        local_qvsidiff = qv - Cqvsi
        local_svmax = local_qvsidiff * Cdtr

        Saggs_i2s = local_eff * qi * Cagg * exp(graupel_const.GrConst_ccsaxp * log(Cslam))
        Saggg_i2g = local_eff * qi * graupel_funcConst.GrFuncConst_cagg_g * Celnrimexp_g
        Siaut_i2s = local_eff * graupel_funcConst.GrFuncConst_ciau * maximum( qi - graupel_funcConst.GrFuncConst_qi0 , 0.0 )

        Sicri_i2g = graupel_funcConst.GrFuncConst_cicri * qi * Celn7o8qrk
        if (qs > 1.e-7):
            Srcri_r2g = graupel_funcConst.GrFuncConst_crcri * (qi / Cmi) * Celn13o8qrk


        local_iceTotalDeposition = Cidep * local_nid * exp(0.33 * local_lnlogmi) * local_qvsidiff
        Sidep_v2i = local_iceTotalDeposition
        # Szdep_v2i = 0.0
        # Szsub_v2i = 0.0

        # for sedimenting quantities the maximum
        # allowed depletion is determined by the predictor value.
        if (graupel_const.GrConst_lsedi_ice):
            local_simax = rhoqi_intermediate * C1orho * Cdtr
        else:
            local_simax = qi * Cdtr

        if (local_iceTotalDeposition > 0.0):
            if (graupel_const.GrConst_lred_depgrow):
                local_iceTotalDeposition = local_iceTotalDeposition * reduce_dep  # FR new: depositional growth reduction
            Szdep_v2i = minimum(local_iceTotalDeposition, local_svmax)
        elif (local_iceTotalDeposition < 0.0):
            Szsub_v2i = maximum(local_iceTotalDeposition, local_svmax)
            Szsub_v2i = - maximum(Szsub_v2i, -local_simax)

        local_lnlogmi = log(graupel_funcConst.GrFuncConst_msmin / Cmi)
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

        if ( temperature <= phy_const.tmelt ):

            local_qvsidiff = qv - Cqvsi
            local_svmax    = local_qvsidiff * Cdtr

            local_xfac = 1.0 + Cbsdep * exp(graupel_const.GrConst_ccsdxp * log(Cslam))
            Ssdep_v2s = Csdep * local_xfac * local_qvsidiff / (Cslam + graupel_const.GrConst_eps)**2.0
            #FR new: depositional growth reduction
            if ( (graupel_const.GrConst_lred_depgrow) & (Ssdep_v2s > 0.0) ):
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

        if ( temperature > phy_const.tmelt ):

            # cloud ice melts instantaneously
            if ( graupel_const.GrConst_lsedi_ice ):
                Simlt_i2c = rhoqi_intermediate * C1orho * Cdtr
            else:
                Simlt_i2c = qi * Cdtr

            local_qvsw0     = graupel_const.GrConst_pvsw0 / (rho * phy_const.rv * phy_const.tmelt)
            local_qvsw0diff = qv - local_qvsw0

            # ** GZ: several numerical fits in this section should be replaced with physically more meaningful formulations **
            if ( temperature > phy_const.tmelt - graupel_funcConst.GrFuncConst_tcrit * local_qvsw0diff ):
                #calculate melting rate
                local_x1  = temperature - phy_const.tmelt + graupel_funcConst.GrFuncConst_asmel * local_qvsw0diff
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
                #local_qvsw      = sat_pres_water(input_t) / (input_rho * phy_const.rv * input_t)
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
        local_x1   = 1.0 + graupel_const.GrConst_bev * exp(graupel_const.GrConst_bevxp * local_lnqr)
        #sev  = zcev*zx1*(zqvsw - qvg) * EXP (zcevxp  * zlnqrk)
        # Limit evaporation rate in order to avoid overshoots towards supersaturation
        # the pre-factor approximates (esat(T_wb)-e)/(esat(T)-e) at temperatures between 0 degC and 30 degC
        local_temp_c  = temperature - phy_const.tmelt
        local_maxevap = (0.61 - 0.0163 * local_temp_c + 1.111e-4 * local_temp_c**2.0) * (Cqvsw - qv) / dt
        Sevap_r2v     = graupel_const.GrConst_cev * local_x1 * (Cqvsw - qv) * exp(graupel_const.GrConst_cevxp * local_lnqr)
        Sevap_r2v     = minimum( Sevap_r2v , local_maxevap )

        if ( temperature > graupel_const.GrConst_thn ):
            # Calculation of below-cloud rainwater freezing
            if ( temperature < graupel_const.GrConst_trfrz ):
                if (graupel_const.GrConst_lsuper_coolw):
                    #FR new: reduced rain freezing rate
                    Srfrz_r2g = graupel_const.GrConst_crfrz1 * (exp(graupel_const.GrConst_crfrz2 * (graupel_const.GrConst_trfrz - temperature)) - 1.0 ) * Celn7o4qrk
                else:
                    Srfrz_r2g = graupel_const.GrConst_crfrz * sqrt( (graupel_const.GrConst_trfrz - temperature)**3.0 ) * Celn27o16qrk
        else: # Hom. freezing of rain water
            Srfrz_r2g = Csrmax

    #--------------------------------------------------------------------------
    # Section 7: Calculate the total tendencies of the prognostic variables.
    #            Update the prognostic variables in the interior domain.
    #--------------------------------------------------------------------------

    # finalizing transfer rates in clouds and calculate depositional growth reduction
    if ( llqc & (temperature > graupel_const.GrConst_thn)):
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
        if (temperature <= phy_const.tmelt):  # cold case

            Cqvsidiff = qv - Cqvsi
            if (graupel_const.GrConst_lsedi_ice):
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
    qi = maximum( 0.0 , (rhoqi_intermediate * C1orho + Cqit * dt) * Cimi ) if ( graupel_const.GrConst_lsedi_ice ) else maximum( 0.0 , qi + Cqit * dt )
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
