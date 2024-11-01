# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import icon4py.model.common.dimension as dims
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.common import settings
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.test_utils import helpers, serialbox_utils as sb


def verify_diffusion_fields(
    config: diffusion.DiffusionConfig,
    diagnostic_state: diffusion_states.DiffusionDiagnosticState,
    prognostic_state: prognostics.PrognosticState,
    diffusion_savepoint: sb.IconDiffusionExitSavepoint,
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
        config.shear_type
        >= diffusion.TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND
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
        # edge length rtol 1e-9
        assert helpers.dallclose(val_div_ic, ref_div_ic, atol=1e-16)
        assert helpers.dallclose(val_hdef_ic, ref_hdef_ic, atol=1e-13)  # atol=1e-13, rtol=5e-9
        assert helpers.dallclose(val_dwdx, ref_dwdx, atol=1e-18)
        assert helpers.dallclose(val_dwdy, ref_dwdy, atol=1e-18)

    # rtol 5e-9
    assert helpers.dallclose(val_vn, ref_vn, atol=1e-9)  # initial 1e-18, # atol= 1e-9, rtol=1e-6
    assert helpers.dallclose(val_w, ref_w, atol=1e-14)  # initial 1e-18,
    assert helpers.dallclose(val_theta_v, ref_theta_v)
    assert helpers.dallclose(val_exner, ref_exner)


def smag_limit_numpy(func, *args):
    return 0.125 - 4.0 * func(*args)


def diff_multfac_vn_numpy(shape, k4, substeps):
    factor = min(1.0 / 128.0, k4 * substeps / 3.0)
    return factor * np.ones(shape)


# TODO: this code is replicated across the codebase currently. The configuration should be read from an external file.
def construct_diffusion_config(name: str, ndyn_substeps: int = 5):
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


def compare_dace_orchestration_multiple_steps(
    non_orch_field: diffusion_states.DiffusionDiagnosticState | prognostics.PrognosticState,
    orch_field: diffusion_states.DiffusionDiagnosticState | prognostics.PrognosticState,
):
    if isinstance(non_orch_field, diffusion_states.DiffusionDiagnosticState):
        div_ic_dace_non_orch = non_orch_field.div_ic.asnumpy()
        hdef_ic_dace_non_orch = non_orch_field.hdef_ic.asnumpy()
        dwdx_dace_non_orch = non_orch_field.dwdx.asnumpy()
        dwdy_dace_non_orch = non_orch_field.dwdy.asnumpy()

        div_ic_dace_orch = orch_field.div_ic.asnumpy()
        hdef_ic_dace_orch = orch_field.hdef_ic.asnumpy()
        dwdx_dace_orch = orch_field.dwdx.asnumpy()
        dwdy_dace_orch = orch_field.dwdy.asnumpy()

        assert np.allclose(div_ic_dace_non_orch, div_ic_dace_orch)
        assert np.allclose(hdef_ic_dace_non_orch, hdef_ic_dace_orch)
        assert np.allclose(dwdx_dace_non_orch, dwdx_dace_orch)
        assert np.allclose(dwdy_dace_non_orch, dwdy_dace_orch)
    elif isinstance(non_orch_field, prognostics.PrognosticState):
        w_dace_orch = non_orch_field.w.asnumpy()
        theta_v_dace_orch = non_orch_field.theta_v.asnumpy()
        exner_dace_orch = non_orch_field.exner.asnumpy()
        vn_dace_orch = non_orch_field.vn.asnumpy()

        w_dace_non_orch = orch_field.w.asnumpy()
        theta_v_dace_non_orch = orch_field.theta_v.asnumpy()
        exner_dace_non_orch = orch_field.exner.asnumpy()
        vn_dace_non_orch = orch_field.vn.asnumpy()

        assert np.allclose(w_dace_non_orch, w_dace_orch)
        assert np.allclose(theta_v_dace_non_orch, theta_v_dace_orch)
        assert np.allclose(exner_dace_non_orch, exner_dace_orch)
        assert np.allclose(vn_dace_non_orch, vn_dace_orch)
    else:
        raise ValueError("Field type not recognized")


@pytest.fixture
def diffusion_instance(
    icon_grid,
    metrics_savepoint,
    interpolation_savepoint,
    ndyn_substeps,
    experiment,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    grid_savepoint,
    backend,
    processor_props,  # fixture
    decomposition_info,  # fixture
):
    """Fixture to create a diffusion instance and clear the orchestration cache properly -if applicable-."""
    exchange = definitions.create_exchange(processor_props, decomposition_info)
    edge_geometry = grid_savepoint.construct_edge_geometry()
    cell_geometry = grid_savepoint.construct_cell_geometry()
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )
    config = construct_diffusion_config(experiment, ndyn_substeps=ndyn_substeps)
    additional_parameters = diffusion.DiffusionParams(config)
    metric_state = diffusion_states.DiffusionMetricState(
        mask_hdiff=metrics_savepoint.mask_hdiff(),
        theta_ref_mc=metrics_savepoint.theta_ref_mc(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        zd_intcoef=metrics_savepoint.zd_intcoef(),
        zd_vertoffset=metrics_savepoint.zd_vertoffset(),
        zd_diffcoef=metrics_savepoint.zd_diffcoef(),
    )
    interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=helpers.as_1D_sparse_field(interpolation_savepoint.e_bln_c_s(), dims.CEDim),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=helpers.as_1D_sparse_field(interpolation_savepoint.geofac_div(), dims.CEDim),
        geofac_n2s=interpolation_savepoint.geofac_n2s(),
        geofac_grg_x=interpolation_savepoint.geofac_grg()[0],
        geofac_grg_y=interpolation_savepoint.geofac_grg()[1],
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )

    diffusion_instance_ = diffusion.Diffusion(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_grid=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        backend=backend,
        exchange=exchange,
    )

    yield diffusion_instance_

    if settings.dace_orchestration is not None:
        diffusion_instance_._do_diffusion_step.clear_cache()
