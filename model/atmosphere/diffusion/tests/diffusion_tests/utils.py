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

from icon4py.model.atmosphere.diffusion.diffusion import (
    DiffusionConfig,
    DiffusionType,
    TurbulenceShearForcingType,
)
from icon4py.model.atmosphere.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
)
from icon4py.model.common.dimension import CEDim, CellDim, KDim
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field, dallclose, zero_field
from icon4py.model.common.test_utils.serialbox_utils import (
    IconDiffusionExitSavepoint,
    IconDiffusionInitSavepoint,
    IconGridSavepoint,
    InterpolationSavepoint,
    MetricSavepoint,
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
        zdiffu_t=False,
        type_t_diffu=2,
        type_vn_diffu=1,
        hdiff_efdt_ratio=24.0,
        smagorinski_scaling_factor=0.025,
        hdiff_temp=True,
        n_substeps=ndyn_substeps,
    )


def r04b09_diffusion_config(
    ndyn_substeps,  # imported `ndyn_substeps` fixture
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
        shear_type=TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND,
    )


def construct_config(name: str, ndyn_substeps: int = 5):
    if name.lower() in "mch_ch_r04b09_dsl":
        return r04b09_diffusion_config(ndyn_substeps)
    elif name.lower() in "exclaim_ape_r02b04":
        return exclaim_ape_diffusion_config(ndyn_substeps)


def verify_diffusion_fields(
    config: DiffusionConfig,
    diagnostic_state: DiffusionDiagnosticState,
    prognostic_state: PrognosticState,
    diffusion_savepoint: IconDiffusionExitSavepoint,
):
    ref_w = diffusion_savepoint.w().asnumpy()
    val_w = prognostic_state.w.asnumpy()
    ref_exner = diffusion_savepoint.exner().asnumpy()
    ref_theta_v = diffusion_savepoint.theta_v().asnumpy()
    val_theta_v = prognostic_state.theta_v.asnumpy()
    val_exner = prognostic_state.exner.asnumpy()
    ref_vn = diffusion_savepoint.vn().asnumpy()
    val_vn = prognostic_state.vn.asnumpy()

    validate_diagnostics = (
        config.shear_type >= TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND
    )
    if validate_diagnostics:
        ref_div_ic = diffusion_savepoint.div_ic().asnumpy()
        val_div_ic = diagnostic_state.div_ic.asnumpy()
        ref_hdef_ic = diffusion_savepoint.hdef_ic().asnumpy()
        val_hdef_ic = diagnostic_state.hdef_ic.asnumpy()
        ref_dwdx = diffusion_savepoint.dwdx().asnumpy()
        val_dwdx = diagnostic_state.dwdx.asnumpy()
        ref_dwdy = diffusion_savepoint.dwdy().asnumpy()
        val_dwdy = diagnostic_state.dwdy.asnumpy()

        assert np.allclose(ref_div_ic, val_div_ic)
        assert np.allclose(ref_hdef_ic, val_hdef_ic)
        assert np.allclose(ref_dwdx, val_dwdx)
        assert np.allclose(ref_dwdy, val_dwdy)

    assert np.allclose(ref_vn, val_vn)
    assert np.allclose(ref_w, val_w)
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
    grid_savepoint: IconGridSavepoint,
) -> DiffusionDiagnosticState:
    grid = grid_savepoint.construct_icon_grid()
    dwdx = savepoint.dwdx() if savepoint.dwdx() else zero_field(grid, CellDim, KDim)
    dwdy = savepoint.dwdy() if savepoint.dwdy() else zero_field(grid, CellDim, KDim)
    return DiffusionDiagnosticState(
        hdef_ic=savepoint.hdef_ic(),
        div_ic=savepoint.div_ic(),
        dwdx=dwdx,
        dwdy=dwdy,
    )
