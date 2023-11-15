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

from icon4py.model.atmosphere.diffusion.diffusion import DiffusionConfig, DiffusionType
from icon4py.model.atmosphere.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
)
from icon4py.model.common.dimension import CEDim
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field, dallclose
from icon4py.model.common.test_utils.serialbox_utils import (
    IconDiffusionExitSavepoint,
    InterpolationSavepoint,
    MetricSavepoint,
    IconDiffusionInitSavepoint,
)


def exclaim_ape_diffusion_config(ndyn_substeps):
    """Create DiffusionConfig matching EXCLAIM_APE_R04B02.

    Set values to the ones used in the  EXCLAIM_APE_R04B02 experiment where they differ
    from the default.
    """
    return DiffusionConfig(
        diffusion_type=DiffusionType.SMAGORINSKY_4TH_ORDER,
        hdiff_w=True,
        hdiff_vn=True,
        type_t_diffu=2,
        type_vn_diffu=1,
        hdiff_efdt_ratio=24.0,
        smagorinski_scaling_factor=0.025,
        hdiff_temp=True,
        n_substeps=ndyn_substeps,
    )


def r04b09_diffusion_config(
    ndyn_substeps,  # noqa: F811 # imported `ndyn_substeps` fixture
) -> DiffusionConfig:
    """
    Create DiffusionConfig matching MCH_CH_r04b09_dsl.

    Set values to the ones used in the  MCH_CH_r04b09_dsl experiment where they differ
    from the default.
    """
    return DiffusionConfig(
        diffusion_type=DiffusionType.SMAGORINSKY_4TH_ORDER,
        hdiff_w=True,
        hdiff_vn=True,
        type_t_diffu=2,
        type_vn_diffu=1,
        hdiff_efdt_ratio=24.0,
        hdiff_w_efdt_ratio=15.0,
        smagorinski_scaling_factor=0.025,
        zdiffu_t=True,
        velocity_boundary_diffusion_denom=150.0,
        max_nudging_coeff=0.075,
        n_substeps=ndyn_substeps,
    )


def construct_config(name: str, ndyn_substeps: int = 5):
    if name.lower() in "mch_ch_r04b09_dsl":
        return r04b09_diffusion_config(ndyn_substeps)
    elif name.lower() in "exclaim_ape_r02b04":
        return exclaim_ape_diffusion_config(ndyn_substeps)


def verify_diffusion_fields(
    diagnostic_state: DiffusionDiagnosticState,
    prognostic_state: PrognosticState,
    diffusion_savepoint: IconDiffusionExitSavepoint,
):
    ref_div_ic = np.asarray(diffusion_savepoint.div_ic())
    val_div_ic = np.asarray(diagnostic_state.div_ic)
    ref_hdef_ic = np.asarray(diffusion_savepoint.hdef_ic())
    val_hdef_ic = np.asarray(diagnostic_state.hdef_ic)
    assert dallclose(ref_div_ic, val_div_ic, atol=5.0e-18)
    assert dallclose(ref_hdef_ic, val_hdef_ic)
    ref_w = np.asarray(diffusion_savepoint.w())
    val_w = np.asarray(prognostic_state.w)
    ref_dwdx = np.asarray(diffusion_savepoint.dwdx())
    val_dwdx = np.asarray(diagnostic_state.dwdx)
    ref_dwdy = np.asarray(diffusion_savepoint.dwdy())
    val_dwdy = np.asarray(diagnostic_state.dwdy)
    assert dallclose(ref_dwdx, val_dwdx, atol=1e-19)
    assert dallclose(ref_dwdy, val_dwdy, atol=1e-19)

    ref_vn = np.asarray(diffusion_savepoint.vn())
    val_vn = np.asarray(prognostic_state.vn)
    assert dallclose(ref_vn, val_vn, atol=1e-15)
    assert dallclose(ref_w, val_w, atol=1e-16)
    ref_exner = np.asarray(diffusion_savepoint.exner())
    ref_theta_v = np.asarray(diffusion_savepoint.theta_v())
    val_theta_v = np.asarray(prognostic_state.theta_v)
    val_exner = np.asarray(prognostic_state.exner)
    assert dallclose(ref_theta_v, val_theta_v)
    assert dallclose(ref_exner, val_exner)


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


def construct_interpolation_state(
    savepoint: InterpolationSavepoint,
) -> DiffusionInterpolationState:
    grg = savepoint.geofac_grg()
    return DiffusionInterpolationState(
        e_bln_c_s=as_1D_sparse_field(savepoint.e_bln_c_s(), CEDim),
        rbf_coeff_1=savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=savepoint.rbf_vec_coeff_v2(),
        geofac_div=as_1D_sparse_field(savepoint.geofac_div(), CEDim),
        geofac_n2s=savepoint.geofac_n2s(),
        geofac_grg_x=grg[0],
        geofac_grg_y=grg[1],
        nudgecoeff_e=savepoint.nudgecoeff_e(),
    )


def construct_metric_state_for_diffusion(savepoint: MetricSavepoint) -> DiffusionMetricState:
    return DiffusionMetricState(
        mask_hdiff=savepoint.mask_hdiff(),
        theta_ref_mc=savepoint.theta_ref_mc(),
        wgtfac_c=savepoint.wgtfac_c(),
        zd_intcoef=savepoint.zd_intcoef(),
        zd_vertoffset=savepoint.zd_vertoffset(),
        zd_diffcoef=savepoint.zd_diffcoef(),
    )


def construct_diagnostics(
    savepoint: IconDiffusionInitSavepoint,
) -> DiffusionDiagnosticState:
    return DiffusionDiagnosticState(
        hdef_ic=savepoint.hdef_ic(),
        div_ic=savepoint.div_ic(),
        dwdx=savepoint.dwdx(),
        dwdy=savepoint.dwdy(),
    )
