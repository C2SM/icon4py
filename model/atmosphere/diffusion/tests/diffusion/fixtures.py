# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from icon4py.model.atmosphere.diffusion import diffusion_states
from icon4py.model.testing import serialbox as sb

from icon4py.model.testing.fixtures.stencil_tests import grid

from icon4py.model.testing.fixtures.datatest import (
    backend,
    damping_height,
    data_provider,
    decomposition_info,
    download_ser_data,
    experiment,
    flat_height,
    grid_savepoint,
    htop_moist_proc,
    icon_grid,
    interpolation_savepoint,
    linit,
    maximal_layer_thickness,
    metrics_savepoint,
    lowest_layer_thickness,
    rayleigh_coeff,
    exner_expol,
    vwind_offctr,
    rayleigh_type,
    model_top_height,
    ndyn_substeps,
    processor_props,
    ranked_data_path,
    savepoint_diffusion_exit,
    savepoint_diffusion_init,
    step_date_exit,
    step_date_init,
    stretch_factor,
    top_height_limit_for_maximal_layer_thickness,
)


@pytest.fixture
def interpolation_state(
    interpolation_savepoint: sb.InterpolationSavepoint,
) -> diffusion_states.DiffusionInterpolationState:
    return diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=interpolation_savepoint.e_bln_c_s(),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=interpolation_savepoint.geofac_div(),
        geofac_n2s=interpolation_savepoint.geofac_n2s(),
        geofac_grg_x=interpolation_savepoint.geofac_grg()[0],
        geofac_grg_y=interpolation_savepoint.geofac_grg()[1],
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )


@pytest.fixture
def metric_state(
    metrics_savepoint: sb.MetricSavepoint,
) -> diffusion_states.DiffusionMetricState:
    return diffusion_states.DiffusionMetricState(
        mask_hdiff=metrics_savepoint.mask_hdiff(),
        theta_ref_mc=metrics_savepoint.theta_ref_mc(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        zd_intcoef=metrics_savepoint.zd_intcoef(),
        zd_vertoffset=metrics_savepoint.zd_vertoffset(),
        zd_diffcoef=metrics_savepoint.zd_diffcoef(),
    )
