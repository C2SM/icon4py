# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.common.components import framework as fw, quantities as qty, states
from icon4py.model.testing import serialbox as sb, test_utils


def construct_diagnostics(savepoint: sb.IconDiffusionInitSavepoint) -> states.DiffusionDiagnostics:
    return states.DiffusionDiagnostics(
        hdef_ic=fw.Field(qty.HorizontalWindDeformationOnCellKHalf, savepoint.hdef_ic()),
        div_ic=fw.Field(qty.DivergenceOnCellKHalf, savepoint.div_ic()),
        dwdx=fw.Field(qty.ZonalGradientOfWOnCellKHalf, savepoint.dwdx()),
        dwdy=fw.Field(qty.MeridionalGradientOfWOnCellKHalf, savepoint.dwdy()),
    )


def diffusion_views(
    prognostic_state: states.PrognosticState,
    diagnostic_state: states.DiffusionDiagnostics,
    dtime: float,
    initial_run: bool = False,
) -> tuple[diffusion.Diffusion.Input, diffusion.Diffusion.Output]:
    """The diffusion's views over the prognostics (diffused in place) and its diagnostics."""
    inputs = diffusion.Diffusion.Input(
        vn=prognostic_state.vn,
        w=prognostic_state.w,
        exner=prognostic_state.exner,
        theta_v=prognostic_state.theta_v,
        dtime=dtime,
        initial_run=initial_run,
    )
    out = diffusion.Diffusion.Output(
        vn=prognostic_state.vn,
        w=prognostic_state.w,
        exner=prognostic_state.exner,
        theta_v=prognostic_state.theta_v,
        hdef_ic=diagnostic_state.hdef_ic,
        div_ic=diagnostic_state.div_ic,
        dwdx=diagnostic_state.dwdx,
        dwdy=diagnostic_state.dwdy,
    )
    return inputs, out


def verify_diffusion_fields(
    config: diffusion.DiffusionConfig,
    diagnostic_state: states.DiffusionDiagnostics,
    prognostic_state: states.PrognosticState,
    diffusion_savepoint: sb.IconDiffusionExitSavepoint,
):
    ref_w = diffusion_savepoint.w().asnumpy()
    val_w = prognostic_state.w.data.asnumpy()
    ref_exner = diffusion_savepoint.exner().asnumpy()
    ref_theta_v = diffusion_savepoint.theta_v().asnumpy()
    val_theta_v = prognostic_state.theta_v.data.asnumpy()
    val_exner = prognostic_state.exner.data.asnumpy()
    ref_vn = diffusion_savepoint.vn().asnumpy()
    val_vn = prognostic_state.vn.data.asnumpy()

    validate_diagnostics = (
        config.shear_type
        >= diffusion.TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND
    )
    if validate_diagnostics:
        ref_div_ic = diffusion_savepoint.div_ic().asnumpy()
        val_div_ic = diagnostic_state.div_ic.data.asnumpy()
        ref_hdef_ic = diffusion_savepoint.hdef_ic().asnumpy()
        val_hdef_ic = diagnostic_state.hdef_ic.data.asnumpy()
        ref_dwdx = diffusion_savepoint.dwdx().asnumpy()
        val_dwdx = diagnostic_state.dwdx.data.asnumpy()
        ref_dwdy = diffusion_savepoint.dwdy().asnumpy()
        val_dwdy = diagnostic_state.dwdy.data.asnumpy()

        test_utils.assert_dallclose(
            val_div_ic, ref_div_ic, atol=1e-16 if test_utils.wp_is_dp else 4e-9
        )
        test_utils.assert_dallclose(
            val_hdef_ic, ref_hdef_ic, atol=1e-13 if test_utils.wp_is_dp else 2e-12
        )
        test_utils.assert_dallclose(val_dwdx, ref_dwdx, atol=1e-18 if test_utils.wp_is_dp else 2e-9)
        test_utils.assert_dallclose(val_dwdy, ref_dwdy, atol=1e-18 if test_utils.wp_is_dp else 2e-9)

    test_utils.assert_dallclose(
        val_vn, ref_vn, atol=1.0e-8 if test_utils.wp_is_dp else 4e-6, rtol=1.0e-9
    )
    test_utils.assert_dallclose(val_w, ref_w, atol=1e-14 if test_utils.wp_is_dp else 2e-7)
    test_utils.assert_dallclose(val_theta_v, ref_theta_v)
    test_utils.assert_dallclose(val_exner, ref_exner)


def smag_limit_numpy(func, *args):
    return 0.125 - 4.0 * func(*args)


def diff_multfac_vn_numpy(shape, k4, substeps):
    factor = min(1.0 / 128.0, k4 * substeps / 3.0)
    return np.full(shape, factor)
