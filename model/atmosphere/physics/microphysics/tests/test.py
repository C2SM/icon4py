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
import dataclasses

from gt4py.next.embedded.context import offset_provider
from gt4py.next.program_processors.runners.double_roundtrip import backend
from gt4py.next.program_processors.runners.gtfn import (
    run_gtfn_cached,
    run_gtfn_gpu_cached,
)
import numpy as np
import gt4py.next as gtx
from gt4py.eve.utils import FrozenNamespace
from gt4py.next.ffront.fbuiltins import log, exp, maximum, minimum, sqrt
from icon4py.model.common import constants as global_const
from gt4py.next.ffront.decorator import program, field_operator, scan_operator
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.atmosphere.physics.microphysics.single_moment_six_class_gscp_graupel import icon_graupel_params
from icon4py.model.atmosphere.physics.microphysics import single_moment_six_class_gscp_graupel as graupel
from icon4py.model.atmosphere.physics.microphysics import saturation_adjustment
from icon4py.model.common.type_alias import wpfloat, vpfloat


@scan_operator(axis=KDim, forward=True, init=(0.0, gtx.int32(0)))
def test_multiple_return(
    state: tuple[float, gtx.int32],
    input_arg: float
):
    if state[1] < 5:
        return state[0], state[1] + 1

    return state[0] + input_arg, state[1] + 1

cell_size = 3
k_size = 10

in_field = gtx.as_field((CellDim,KDim), np.full((cell_size,k_size),fill_value=1.0, dtype=float))
out_field1 = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
out_field2 = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=gtx.int32))

test_multiple_return(in_field, out=(out_field1, out_field2), offset_provider={})

print(out_field1.ndarray)
print(out_field2.ndarray)


class MyConstants(FrozenNamespace):
    a: wpfloat = 2.0


my_constants: Final = MyConstants()

class MyAnotherConstants(FrozenNamespace):
    a: wpfloat = 1.0 + my_constants.a

my_anotherconstants: Final = MyAnotherConstants()

my_a: Final[wpfloat] = wpfloat("1.0")

@field_operator
def test_constant(
    input_arg: gtx.Field[[CellDim, KDim], wpfloat]
):
    return input_arg + my_constants.a

@program(backend=run_gtfn_cached)
def program_test_constant(
    input_arg: gtx.Field[[CellDim, KDim], wpfloat],
    output_arg: gtx.Field[[CellDim, KDim], wpfloat],
    cstart: gtx.int32,
    cend: gtx.int32,
    kstart: gtx.int32,
    kend: gtx.int32,
):
    test_constant(
        input_arg,
        out=output_arg,
        domain={
            CellDim: (cstart, cend),
            KDim: (kstart, kend),
        }
    )

input_field = gtx.as_field((CellDim,KDim), np.full((cell_size,k_size),fill_value=0.0, dtype=float))
output_field1 = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
output_field2 = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))

test_constant.with_backend(run_gtfn_cached)(input_field, out=output_field1, offset_provider={})
#program_test_constant(input_field,output_field, gtx.int32(0), gtx.int32(cell_size), gtx.int32(0), gtx.int32(k_size),offset_provider={})
print()
print(output_field1.ndarray)

my_constants: Final = MyAnotherConstants()

test_constant.with_backend(run_gtfn_cached)(input_field, out=output_field2, offset_provider={})
print()
print(output_field2.ndarray)


'''
@scan_operator(axis=KDim, forward=True, init=(0.0, 0.0))
def test_trans(
    state: tuple[float, float],
    temperature: float,
    rho: float,
    qs: float,
    input_arg: float
):

    llqs = True if input_arg > 0.0 else False
    if llqs:
        # Calculate n0s using the temperature-dependent moment
        # relations of Field et al. (2005)
        local_tc = temperature - icon_graupel_params.tmelt
        local_tc = minimum(local_tc, 0.0)
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
        n0s_ = 13.50 * local_m2s * (local_m2s / local_m3s) ** 3.0
        n0s_ = maximum(n0s_, 0.5 * local_hlp)
        n0s_ = minimum(n0s_, 1.0e2 * local_hlp)
        n0s_ = minimum(n0s_, 1.0e9)
        n0s_ = maximum(n0s_, 1.0e6)
    else:
        n0s_ = icon_graupel_params.snow_n0

    result = 3.5 * exp(icon_graupel_params.ccsvxp * log(n0s_)) if llqs else 0.0
    final_result = state[0] + result
    return state[0] + input_arg, final_result

cell_size = 10
k_size = 4


in_field = gtx.as_field((CellDim,KDim), np.full((cell_size,k_size),fill_value=2.0, dtype=float))
t_field = gtx.as_field((CellDim,KDim), np.full((cell_size,k_size),fill_value=260.0, dtype=float))
rho_field = gtx.as_field((CellDim,KDim), np.full((cell_size,k_size),fill_value=1.0, dtype=float))
qs_field = gtx.as_field((CellDim,KDim), np.full((cell_size,k_size),fill_value=0.001, dtype=float))
out_field1 = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
out_field2 = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))

test_trans.with_backend(run_gtfn_cached)(t_field, rho_field, qs_field, in_field, out=(out_field1,out_field2), offset_provider={})

print(out_field1.ndarray)
print(out_field2.ndarray)
'''



'''
graupel_config = graupel.SingleMomentSixClassIconGraupelConfig(
        do_saturation_adjustment=False,
        liquid_autoconversion_option=gtx.int32(1),
        ice_stickeff_min=wpfloat(0.01),
        ice_v0=wpfloat(1.25),
        ice_sedi_density_factor_exp=wpfloat(0.3),
        snow_v0=wpfloat(20.0),
        rain_mu=wpfloat(0.0),
        rain_n0=wpfloat(1.0),
    )

saturation_adjustment_config = saturation_adjustment.SaturationAdjustmentConfig()
graupel_microphysics = graupel.SingleMomentSixClassIconGraupel(
    graupel_config=graupel_config,
    saturation_adjust_config=saturation_adjustment_config,
    grid=None,
    metric_state=None,
    vertical_params=None,
)

dtime = 5.0
cell_size = 30
k_size = 65

ddqz_z_full = gtx.as_field((CellDim,KDim), np.full((cell_size,k_size),fill_value=1.0, dtype=float))
qnc = gtx.as_field((CellDim,KDim), np.full((cell_size,k_size),fill_value=1000000.0, dtype=float))
pressure = gtx.as_field((CellDim,KDim), np.full((cell_size,k_size),fill_value=100000.0, dtype=float))
temperature = gtx.as_field((CellDim,KDim), np.full((cell_size,k_size),fill_value=260.0, dtype=float))
rho = gtx.as_field((CellDim,KDim), np.full((cell_size,k_size),fill_value=1.0, dtype=float))

k_startmoist = 10
qv_array = np.full((cell_size,k_size),fill_value=0.001, dtype=float)
qc_array = np.full((cell_size,k_size),fill_value=0.00001, dtype=float)
qi_array = np.full((cell_size,k_size),fill_value=0.00001, dtype=float)
qr_array = np.full((cell_size,k_size),fill_value=0.00001, dtype=float)
qs_array = np.full((cell_size,k_size),fill_value=0.00001, dtype=float)
qg_array = np.full((cell_size,k_size),fill_value=0.00001, dtype=float)
for k in range(k_size):
    if k < k_startmoist:
        qv_array[:, k] = 0.0
        qv_array[:, k] = 0.0
        qc_array[:, k] = 0.0
        qi_array[:, k] = 0.0
        qr_array[:, k] = 0.0
        qs_array[:, k] = 0.0
        qg_array[:, k] = 0.0
qv = gtx.as_field((CellDim,KDim), qv_array)
qc = gtx.as_field((CellDim,KDim), qc_array)
qi = gtx.as_field((CellDim,KDim), qi_array)
qr = gtx.as_field((CellDim,KDim), qr_array)
qs = gtx.as_field((CellDim,KDim), qs_array)
qg = gtx.as_field((CellDim,KDim), qg_array)
rhoqrv_old_kup = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
rhoqsv_old_kup = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
rhoqgv_old_kup = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
rhoqiv_old_kup = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
vnew_r = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
vnew_s = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
vnew_g = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
vnew_i = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
temperature_ = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
qv_ = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
qc_ = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
qi_ = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
qr_ = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
qs_ = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
qg_ = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
dist_cldtop = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
rho_kup = gtx.as_field((CellDim,KDim), np.full((cell_size,k_size), fill_value=1.0, dtype=float))
crho1o2_kup = gtx.as_field((CellDim,KDim), np.full((cell_size,k_size), fill_value=1.0, dtype=float))
crhofac_qi_kup = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
snow_sed0_kup = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
qvsw_kup = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=float))
k_lev = gtx.as_field((CellDim,KDim), np.zeros((cell_size,k_size), dtype=gtx.int32))
from devtools import Timer
timer = Timer("embedded graupel")

for time_step in range(1):
    timer.start()
    graupel.icon_graupel(
        gtx.int32(k_size),
        graupel_microphysics.config.liquid_autoconversion_option,
        graupel_microphysics.config.snow_intercept_option,
        graupel_microphysics.config.rain_freezing_option,
        graupel_microphysics.config.ice_concentration_option,
        graupel_microphysics.config.do_ice_sedimentation,
        graupel_microphysics.config.ice_autocon_sticking_efficiency_option,
        graupel_microphysics.config.do_reduced_icedeposition,
        graupel_microphysics.config.is_isochoric,
        graupel_microphysics.config.use_constant_water_heat_capacity,
        graupel_microphysics.config.ice_stickeff_min,
        graupel_microphysics.config.ice_v0,
        graupel_microphysics.config.ice_sedi_density_factor_exp,
        graupel_microphysics.config.snow_v0,
        *graupel_microphysics.ccs,
        graupel_microphysics.nimax,
        graupel_microphysics.nimix,
        *graupel_microphysics.rain_vel_coef,
        *graupel_microphysics.sed_dens_factor_coef,
        dtime,
        ddqz_z_full,
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
        k_lev,
        gtx.int32(0),
        gtx.int32(10),
        gtx.int32(0),
        gtx.int32(k_size),
        offset_provider={},
    )
    timer.capture()

timer.summary(True)

for cell in range(cell_size):
    print(cell, temperature_.ndarray[cell,:])
'''
