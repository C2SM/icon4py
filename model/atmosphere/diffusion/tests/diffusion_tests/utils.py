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

import numpy as np

from icon4py.model.atmosphere.diffusion.diffusion_states import DiffusionDiagnosticState
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils.serialbox_utils import IconDiffusionExitSavepoint


def verify_diffusion_fields(
    diagnostic_state: DiffusionDiagnosticState,
    prognostic_state: PrognosticState,
    diffusion_savepoint: IconDiffusionExitSavepoint,
):
    ref_div_ic = np.asarray(diffusion_savepoint.div_ic())
    val_div_ic = np.asarray(diagnostic_state.div_ic)
    ref_hdef_ic = np.asarray(diffusion_savepoint.hdef_ic())
    val_hdef_ic = np.asarray(diagnostic_state.hdef_ic)
    assert np.allclose(ref_div_ic, val_div_ic)
    assert np.allclose(ref_hdef_ic, val_hdef_ic)
    ref_w = np.asarray(diffusion_savepoint.w())
    val_w = np.asarray(prognostic_state.w)
    ref_dwdx = np.asarray(diffusion_savepoint.dwdx())
    val_dwdx = np.asarray(diagnostic_state.dwdx)
    ref_dwdy = np.asarray(diffusion_savepoint.dwdy())
    val_dwdy = np.asarray(diagnostic_state.dwdy)
    assert np.allclose(ref_dwdx, val_dwdx)
    assert np.allclose(ref_dwdy, val_dwdy)

    ref_vn = np.asarray(diffusion_savepoint.vn())
    val_vn = np.asarray(prognostic_state.vn)
    assert np.allclose(ref_vn, val_vn)
    assert np.allclose(ref_w, val_w)
    ref_exner = np.asarray(diffusion_savepoint.exner())
    ref_theta_v = np.asarray(diffusion_savepoint.theta_v())
    val_theta_v = np.asarray(prognostic_state.theta_v)
    val_exner = np.asarray(prognostic_state.exner)
    assert np.allclose(ref_theta_v, val_theta_v)
    assert np.allclose(ref_exner, val_exner)


def smag_limit_numpy(func, *args):
    return 0.125 - 4.0 * func(*args)


def diff_multfac_vn_numpy(shape, k4, substeps):
    factor = min(1.0 / 128.0, k4 * substeps / 3.0)
    return factor * np.ones(shape)


def enhanced_smagorinski_factor_numpy(factor_in, heigths_in, a_vec):
    alin = (factor_in[1] - factor_in[0]) / (heigths_in[1] - heigths_in[0])
    df32 = factor_in[2] - factor_in[1]
    df42 = factor_in[3] - factor_in[1]
    dz32 = heigths_in[2] - heigths_in[1]
    dz42 = heigths_in[3] - heigths_in[1]
    bqdr = (df42 * dz32 - df32 * dz42) / (dz32 * dz42 * (dz42 - dz32))
    aqdr = df32 / dz32 - bqdr * dz32
    zf = 0.5 * (a_vec[:-1] + a_vec[1:])
    max0 = np.maximum(0.0, zf - heigths_in[0])
    dzlin = np.minimum(heigths_in[1] - heigths_in[0], max0)
    max1 = np.maximum(0.0, zf - heigths_in[1])
    dzqdr = np.minimum(heigths_in[3] - heigths_in[1], max1)
    return factor_in[0] + dzlin * alin + dzqdr * (aqdr + dzqdr * bqdr)
