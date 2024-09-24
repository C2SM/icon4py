# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.test_utils import helpers, serialbox_utils as sb


def vertical_grid(vertical_config: v_grid.VerticalGridConfig, grid_savepoint: sb.IconGridSavepoint):
    return v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )


def construct_diagnostics(
    savepoint: sb.IconDiffusionInitSavepoint,
) -> diffusion_states.DiffusionDiagnosticState:
    dwdx = savepoint.dwdx()
    dwdy = savepoint.dwdy()
    return diffusion_states.DiffusionDiagnosticState(
        hdef_ic=savepoint.hdef_ic(),
        div_ic=savepoint.div_ic(),
        dwdx=dwdx,
        dwdy=dwdy,
    )


def construct_metric_state(savepoint: sb.MetricSavepoint) -> diffusion_states.DiffusionMetricState:
    return diffusion_states.DiffusionMetricState(
        mask_hdiff=savepoint.mask_hdiff(),
        theta_ref_mc=savepoint.theta_ref_mc(),
        wgtfac_c=savepoint.wgtfac_c(),
        zd_intcoef=savepoint.zd_intcoef(),
        zd_vertoffset=savepoint.zd_vertoffset(),
        zd_diffcoef=savepoint.zd_diffcoef(),
    )


def construct_interpolation_state(
    savepoint: sb.InterpolationSavepoint,
) -> diffusion_states.DiffusionInterpolationState:
    grg = savepoint.geofac_grg()
    return diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=helpers.as_1D_sparse_field(savepoint.e_bln_c_s(), dims.CEDim),
        rbf_coeff_1=savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=savepoint.rbf_vec_coeff_v2(),
        geofac_div=helpers.as_1D_sparse_field(savepoint.geofac_div(), dims.CEDim),
        geofac_n2s=savepoint.geofac_n2s(),
        geofac_grg_x=grg[0],
        geofac_grg_y=grg[1],
        nudgecoeff_e=savepoint.nudgecoeff_e(),
    )


def construct_config(name: str, ndyn_substeps: int = 5):
    if name.lower() in "mch_ch_r04b09_dsl":
        return r04b09_diffusion_config(ndyn_substeps)
    elif name.lower() in "exclaim_ape_r02b04":
        return exclaim_ape_diffusion_config(ndyn_substeps)


def r04b09_diffusion_config(
    ndyn_substeps,  # imported `ndyn_substeps` fixture
) -> diffusion.DiffusionConfig:
    """
    Create DiffusionConfig matching MCH_CH_r04b09_dsl.

    Set values to the ones used in the  MCH_CH_r04b09_dsl experiment where they differ
    from the default.
    """
    return diffusion.DiffusionConfig(
        diffusion_type=diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER,
        hdiff_w=True,
        hdiff_vn=True,
        type_t_diffu=2,
        type_vn_diffu=1,
        hdiff_efdt_ratio=24.0,
        hdiff_w_efdt_ratio=15.0,
        smagorinski_scaling_factor=0.025,
        zdiffu_t=True,
        thslp_zdiffu=0.02,
        thhgtd_zdiffu=125.0,
        velocity_boundary_diffusion_denom=150.0,
        max_nudging_coeff=0.075,
        n_substeps=ndyn_substeps,
        shear_type=diffusion.TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND,
    )


def exclaim_ape_diffusion_config(ndyn_substeps):
    """Create DiffusionConfig matching EXCLAIM_APE_R04B02.

    Set values to the ones used in the  EXCLAIM_APE_R04B02 experiment where they differ
    from the default.
    """
    return diffusion.DiffusionConfig(
        diffusion_type=diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER,
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
