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
import pytest

from icon4py.model.atmosphere.diffusion import diffusion, diffusion_utils
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams
from icon4py.model.common.settings import backend
from icon4py.model.common.test_utils import (
    datatest_utils as dt_utils,
    helpers,
    reference_funcs as ref_funcs,
    serialbox_utils as sb,
)

from .utils import (
    construct_config,
    construct_diagnostics,
    construct_interpolation_state,
    construct_metric_state,
    diff_multfac_vn_numpy,
    smag_limit_numpy,
    verify_diffusion_fields,
)


def test_diffusion_coefficients_with_hdiff_efdt_ratio(experiment):
    config = construct_config(experiment, ndyn_substeps=5)
    config.hdiff_efdt_ratio = 1.0
    config.hdiff_w_efdt_ratio = 2.0

    params = diffusion.DiffusionParams(config)

    assert params.K2 == pytest.approx(0.125, abs=1e-12)
    assert params.K4 == pytest.approx(0.125 / 8.0, abs=1e-12)
    assert params.K6 == pytest.approx(0.125 / 64.0, abs=1e-12)
    assert params.K4W == pytest.approx(1.0 / 72.0, abs=1e-12)


def test_diffusion_coefficients_without_hdiff_efdt_ratio(experiment):
    config = construct_config(experiment)
    config.hdiff_efdt_ratio = 0.0
    config.hdiff_w_efdt_ratio = 0.0

    params = diffusion.DiffusionParams(config)

    assert params.K2 == 0.0
    assert params.K4 == 0.0
    assert params.K6 == 0.0
    assert params.K4W == 0.0


def test_smagorinski_factor_for_diffusion_type_4(experiment):
    config = construct_config(experiment, ndyn_substeps=5)
    config.smagorinski_scaling_factor = 0.15
    config.diffusion_type = 4

    params = diffusion.DiffusionParams(config)
    assert len(params.smagorinski_factor) == 1
    assert params.smagorinski_factor[0] == pytest.approx(0.15, abs=1e-16)
    assert params.smagorinski_height is None


def test_smagorinski_heights_diffusion_type_5_are_consistent(
    experiment,
):
    config = construct_config(experiment, ndyn_substeps=5)
    config.smagorinski_scaling_factor = 0.15
    config.diffusion_type = 5

    params = diffusion.DiffusionParams(config)
    assert len(params.smagorinski_height) == 4
    assert min(params.smagorinski_height) == params.smagorinski_height[0]
    assert max(params.smagorinski_height) == params.smagorinski_height[-1]
    assert params.smagorinski_height[0] < params.smagorinski_height[1]
    assert params.smagorinski_height[1] < params.smagorinski_height[3]
    assert params.smagorinski_height[2] != params.smagorinski_height[1]
    assert params.smagorinski_height[2] != params.smagorinski_height[3]


def test_smagorinski_factor_diffusion_type_5(experiment):
    params = diffusion.DiffusionParams(construct_config(experiment, ndyn_substeps=5))
    assert len(params.smagorinski_factor) == len(params.smagorinski_height)
    assert len(params.smagorinski_factor) == 4
    assert np.all(params.smagorinski_factor >= np.zeros(len(params.smagorinski_factor)))


def create_vertical_params(vertical_config, grid_savepoint):
    return v_grid.VerticalGridParams(
        vertical_config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )


@pytest.mark.datatest
def test_diffusion_init(
    diffusion_savepoint_init,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    icon_grid,
    experiment,
    step_date_init,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    ndyn_substeps,
):
    config = construct_config(experiment, ndyn_substeps=ndyn_substeps)
    additional_parameters = diffusion.DiffusionParams(config)

    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)

    meta = diffusion_savepoint_init.get_metadata("linit", "date")

    assert meta["linit"] is False
    assert meta["date"] == step_date_init

    interpolation_state = construct_interpolation_state(interpolation_savepoint)
    metric_state = construct_metric_state(metrics_savepoint)
    edge_params = grid_savepoint.construct_edge_geometry()
    cell_params = grid_savepoint.construct_cell_geometry()

    diffusion_granule = diffusion.Diffusion()
    diffusion_granule.init(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_params=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_params,
        cell_params=cell_params,
    )

    assert diffusion_granule.diff_multfac_w == min(
        1.0 / 48.0, additional_parameters.K4W * config.substep_as_float
    )

    assert helpers.dallclose(diffusion_granule.v_vert.asnumpy(), 0.0)
    assert helpers.dallclose(diffusion_granule.u_vert.asnumpy(), 0.0)
    assert helpers.dallclose(diffusion_granule.kh_smag_ec.asnumpy(), 0.0)
    assert helpers.dallclose(diffusion_granule.kh_smag_e.asnumpy(), 0.0)

    shape_k = (icon_grid.num_levels,)
    expected_smag_limit = smag_limit_numpy(
        diff_multfac_vn_numpy,
        shape_k,
        additional_parameters.K4,
        config.substep_as_float,
    )

    assert (
        diffusion_granule.smag_offset == 0.25 * additional_parameters.K4 * config.substep_as_float
    )
    assert helpers.dallclose(diffusion_granule.smag_limit.asnumpy(), expected_smag_limit)

    expected_diff_multfac_vn = diff_multfac_vn_numpy(
        shape_k, additional_parameters.K4, config.substep_as_float
    )
    assert helpers.dallclose(diffusion_granule.diff_multfac_vn.asnumpy(), expected_diff_multfac_vn)
    expected_enh_smag_fac = ref_funcs.enhanced_smagorinski_factor_numpy(
        additional_parameters.smagorinski_factor,
        additional_parameters.smagorinski_height,
        grid_savepoint.vct_a().asnumpy(),
    )
    assert helpers.dallclose(diffusion_granule.enh_smag_fac.asnumpy(), expected_enh_smag_fac)


def _verify_init_values_against_savepoint(
    savepoint: sb.IconDiffusionInitSavepoint, diffusion_granule: diffusion.Diffusion
):
    dtime = savepoint.get_metadata("dtime")["dtime"]

    assert savepoint.nudgezone_diff() == diffusion_granule.nudgezone_diff
    assert savepoint.bdy_diff() == diffusion_granule.bdy_diff
    assert savepoint.fac_bdydiff_v() == diffusion_granule.fac_bdydiff_v
    assert savepoint.smag_offset() == diffusion_granule.smag_offset
    assert savepoint.diff_multfac_w() == diffusion_granule.diff_multfac_w

    # this is done in diffusion.run(...) because it depends on the dtime
    diffusion_utils.scale_k.with_backend(backend)(
        diffusion_granule.enh_smag_fac,
        dtime,
        diffusion_granule.diff_multfac_smag,
        offset_provider={},
    )
    assert helpers.dallclose(
        diffusion_granule.enh_smag_fac.asnumpy(), savepoint.enh_smag_fac(), rtol=1e-7
    )
    assert helpers.dallclose(
        diffusion_granule.diff_multfac_smag.asnumpy(), savepoint.diff_multfac_smag(), rtol=1e-7
    )

    assert helpers.dallclose(diffusion_granule.smag_limit.asnumpy(), savepoint.smag_limit())
    assert helpers.dallclose(
        diffusion_granule.diff_multfac_n2w.asnumpy(), savepoint.diff_multfac_n2w()
    )
    assert helpers.dallclose(
        diffusion_granule.diff_multfac_vn.asnumpy(), savepoint.diff_multfac_vn()
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment,step_date_init",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000"),
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:20.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:04.000"),
    ],
)
@pytest.mark.parametrize("ndyn_substeps", (2,))
def test_verify_diffusion_init_against_savepoint(
    experiment,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    diffusion_savepoint_init,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    ndyn_substeps,
):
    config = construct_config(experiment, ndyn_substeps=ndyn_substeps)
    additional_parameters = diffusion.DiffusionParams(config)
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)
    interpolation_state = construct_interpolation_state(interpolation_savepoint)
    metric_state = construct_metric_state(metrics_savepoint)
    edge_params = grid_savepoint.construct_edge_geometry()
    cell_params = grid_savepoint.construct_cell_geometry()

    diffusion_granule = diffusion.Diffusion()
    diffusion_granule.init(
        icon_grid,
        config,
        additional_parameters,
        vertical_params,
        metric_state,
        interpolation_state,
        edge_params,
        cell_params,
    )

    _verify_init_values_against_savepoint(diffusion_savepoint_init, diffusion_granule)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
@pytest.mark.parametrize("ndyn_substeps", (2,))
def test_run_diffusion_single_step(
    diffusion_savepoint_init,
    diffusion_savepoint_exit,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    icon_grid,
    experiment,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    ndyn_substeps,
):
    dtime = diffusion_savepoint_init.get_metadata("dtime").get("dtime")
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()
    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    interpolation_state = construct_interpolation_state(interpolation_savepoint)
    metric_state = construct_metric_state(metrics_savepoint)
    diagnostic_state = construct_diagnostics(diffusion_savepoint_init)
    prognostic_state = diffusion_savepoint_init.construct_prognostics()
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)
    config = construct_config(experiment, ndyn_substeps)
    additional_parameters = diffusion.DiffusionParams(config)

    diffusion_granule = diffusion.Diffusion()
    diffusion_granule.init(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_params=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )

    verify_diffusion_fields(config, diagnostic_state, prognostic_state, diffusion_savepoint_init)
    assert diffusion_savepoint_init.fac_bdydiff_v() == diffusion_granule.fac_bdydiff_v

    diffusion_granule.run(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        dtime=dtime,
    )

    verify_diffusion_fields(config, diagnostic_state, prognostic_state, diffusion_savepoint_exit)


@pytest.mark.datatest
@pytest.mark.parametrize("linit, experiment", [(True, dt_utils.REGIONAL_EXPERIMENT)])
def test_run_diffusion_initial_step(
    experiment,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    diffusion_savepoint_init,
    diffusion_savepoint_exit,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    icon_grid,
):
    dtime = diffusion_savepoint_init.get_metadata("dtime").get("dtime")
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()
    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    interpolation_state = construct_interpolation_state(interpolation_savepoint)
    metric_state = construct_metric_state(metrics_savepoint)
    diagnostic_state = construct_diagnostics(diffusion_savepoint_init)
    prognostic_state = diffusion_savepoint_init.construct_prognostics()
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)
    config = construct_config(experiment, ndyn_substeps=2)
    additional_parameters = diffusion.DiffusionParams(config)

    diffusion_granule = diffusion.Diffusion()
    diffusion_granule.init(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_params=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )
    assert diffusion_savepoint_init.fac_bdydiff_v() == diffusion_granule.fac_bdydiff_v

    diffusion_granule.initial_run(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        dtime=dtime,
    )

    verify_diffusion_fields(
        config=config,
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        diffusion_savepoint=diffusion_savepoint_exit,
    )
