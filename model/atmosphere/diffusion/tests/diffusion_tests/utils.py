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
    ref_div_ic = diffusion_savepoint.div_ic().asnumpy()
    val_div_ic = diagnostic_state.div_ic.asnumpy()
    ref_hdef_ic = diffusion_savepoint.hdef_ic().asnumpy()
    val_hdef_ic = diagnostic_state.hdef_ic.asnumpy()
    ref_w = diffusion_savepoint.w().asnumpy()
    val_w = prognostic_state.w.asnumpy()
    ref_dwdx = diffusion_savepoint.dwdx().asnumpy()
    val_dwdx = diagnostic_state.dwdx.asnumpy()
    ref_dwdy = diffusion_savepoint.dwdy().asnumpy()
    val_dwdy = diagnostic_state.dwdy.asnumpy()
    ref_exner = diffusion_savepoint.exner().asnumpy()
    ref_theta_v = diffusion_savepoint.theta_v().asnumpy()
    val_theta_v = prognostic_state.theta_v.asnumpy()
    val_exner = prognostic_state.exner.asnumpy()
    ref_vn = diffusion_savepoint.vn().asnumpy()
    val_vn = prognostic_state.vn.asnumpy()

    assert np.allclose(ref_div_ic, val_div_ic)
    assert np.allclose(ref_hdef_ic, val_hdef_ic)
    assert np.allclose(ref_vn, val_vn)
    assert np.allclose(ref_w, val_w)
    assert np.allclose(ref_dwdx, val_dwdx)
    assert np.allclose(ref_dwdy, val_dwdy)

    assert np.allclose(ref_theta_v, val_theta_v)
    assert np.allclose(ref_exner, val_exner)


def smag_limit_numpy(func, *args):
    return 0.125 - 4.0 * func(*args)


def diff_multfac_vn_numpy(shape, k4, substeps):
    factor = min(1.0 / 128.0, k4 * substeps / 3.0)
    return factor * np.ones(shape)
