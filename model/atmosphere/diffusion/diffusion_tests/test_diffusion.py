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

from icon4py.model.common.test_utils.serialbox_utils import (
    IconDiffusionExitSavepoint,
    IconDiffusionInitSavepoint,
)
from icon4py.model.atmosphere.diffusion.diffusion import Diffusion, DiffusionParams
from icon4py.model.atmosphere.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    PrognosticState,
)
from icon4py.model.atmosphere.diffusion.diffusion_utils import scale_k
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams
from icon4py.model.common.grid.vertical import VerticalModelParams

from .utils import (
    diff_multfac_vn_numpy,
    enhanced_smagorinski_factor_numpy,
    smag_limit_numpy,
    verify_diffusion_fields,
)


def test_diffusion_coefficients_with_hdiff_efdt_ratio(r04b09_diffusion_config):
    config = r04b09_diffusion_config
    config.hdiff_efdt_ratio = 1.0
    config.hdiff_w_efdt_ratio = 2.0

    params = DiffusionParams(config)

    assert params.K2 == pytest.approx(0.125, abs=1e-12)
    assert params.K4 == pytest.approx(0.125 / 8.0, abs=1e-12)
    assert params.K6 == pytest.approx(0.125 / 64.0, abs=1e-12)
    assert params.K4W == pytest.approx(1.0 / 72.0, abs=1e-12)


def test_diffusion_coefficients_without_hdiff_efdt_ratio(r04b09_diffusion_config):
    config = r04b09_diffusion_config
    config.hdiff_efdt_ratio = 0.0
    config.hdiff_w_efdt_ratio = 0.0

    params = DiffusionParams(config)

    assert params.K2 == 0.0
    assert params.K4 == 0.0
    assert params.K6 == 0.0
    assert params.K4W == 0.0


def test_smagorinski_factor_for_diffusion_type_4(r04b09_diffusion_config):
    config = r04b09_diffusion_config
    config.smagorinski_scaling_factor = 0.15
    config.diffusion_type = 4

    params = DiffusionParams(config)
    assert len(params.smagorinski_factor) == 1
    assert params.smagorinski_factor[0] == pytest.approx(0.15, abs=1e-16)
    assert params.smagorinski_height is None


def test_smagorinski_heights_diffusion_type_5_are_consistent(
    r04b09_diffusion_config,
):
    config = r04b09_diffusion_config
    config.smagorinski_scaling_factor = 0.15
    config.diffusion_type = 5

    params = DiffusionParams(config)
    assert len(params.smagorinski_height) == 4
    assert min(params.smagorinski_height) == params.smagorinski_height[0]
    assert max(params.smagorinski_height) == params.smagorinski_height[-1]
    assert params.smagorinski_height[0] < params.smagorinski_height[1]
    assert params.smagorinski_height[1] < params.smagorinski_height[3]
    assert params.smagorinski_height[2] != params.smagorinski_height[1]
    assert params.smagorinski_height[2] != params.smagorinski_height[3]


def test_smagorinski_factor_diffusion_type_5(r04b09_diffusion_config):
    params = DiffusionParams(r04b09_diffusion_config)
    assert len(params.smagorinski_factor) == len(params.smagorinski_height)
    assert len(params.smagorinski_factor) == 4
    assert np.all(params.smagorinski_factor >= np.zeros(len(params.smagorinski_factor)))


@pytest.mark.datatest
def test_diffusion_init(
    diffusion_savepoint_init,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    icon_grid,
    r04b09_diffusion_config,
    step_date_init,
    damping_height,
):
    config = r04b09_diffusion_config
    additional_parameters = DiffusionParams(config)
    vertical_params = VerticalModelParams(
        grid_savepoint.vct_a(), damping_height, nflatlev=0, nflat_gradp=0
    )

    meta = diffusion_savepoint_init.get_metadata("linit", "date")

    assert meta["linit"] is False
    assert meta["date"] == step_date_init

    interpolation_state = (
        interpolation_savepoint.construct_interpolation_state_for_diffusion()
    )
    metric_state = metrics_savepoint.construct_metric_state_for_diffusion()
    edge_params = grid_savepoint.construct_edge_geometry()
    cell_params = grid_savepoint.construct_cell_geometry()

    diffusion = Diffusion()
    diffusion.init(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_params=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_params,
        cell_params=cell_params,
    )

    assert diffusion.diff_multfac_w == min(
        1.0 / 48.0, additional_parameters.K4W * config.substep_as_float
    )

    assert np.allclose(0.0, np.asarray(diffusion.v_vert))
    assert np.allclose(0.0, np.asarray(diffusion.u_vert))
    assert np.allclose(0.0, np.asarray(diffusion.kh_smag_ec))
    assert np.allclose(0.0, np.asarray(diffusion.kh_smag_e))

    shape_k = (icon_grid.n_lev(),)
    expected_smag_limit = smag_limit_numpy(
        diff_multfac_vn_numpy,
        shape_k,
        additional_parameters.K4,
        config.substep_as_float,
    )

    assert (
        diffusion.smag_offset
        == 0.25 * additional_parameters.K4 * config.substep_as_float
    )
    assert np.allclose(expected_smag_limit, diffusion.smag_limit)

    expected_diff_multfac_vn = diff_multfac_vn_numpy(
        shape_k, additional_parameters.K4, config.substep_as_float
    )
    assert np.allclose(expected_diff_multfac_vn, diffusion.diff_multfac_vn)
    expected_enh_smag_fac = enhanced_smagorinski_factor_numpy(
        additional_parameters.smagorinski_factor,
        additional_parameters.smagorinski_height,
        grid_savepoint.vct_a(),
    )
    assert np.allclose(expected_enh_smag_fac, np.asarray(diffusion.enh_smag_fac))


def _verify_init_values_against_savepoint(
    savepoint: IconDiffusionInitSavepoint, diffusion: Diffusion
):
    dtime = savepoint.get_metadata("dtime")["dtime"]

    assert savepoint.nudgezone_diff() == diffusion.nudgezone_diff
    assert savepoint.bdy_diff() == diffusion.bdy_diff
    assert savepoint.fac_bdydiff_v() == diffusion.fac_bdydiff_v
    assert savepoint.smag_offset() == diffusion.smag_offset
    assert savepoint.diff_multfac_w() == diffusion.diff_multfac_w

    # this is done in diffusion.run(...) because it depends on the dtime
    scale_k(
        diffusion.enh_smag_fac, dtime, diffusion.diff_multfac_smag, offset_provider={}
    )
    assert np.allclose(savepoint.diff_multfac_smag(), diffusion.diff_multfac_smag)

    assert np.allclose(savepoint.smag_limit(), diffusion.smag_limit)
    assert np.allclose(
        savepoint.diff_multfac_n2w(), np.asarray(diffusion.diff_multfac_n2w)
    )
    assert np.allclose(savepoint.diff_multfac_vn(), diffusion.diff_multfac_vn)


@pytest.mark.datatest
def test_verify_diffusion_init_against_first_regular_savepoint(
    diffusion_savepoint_init,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    r04b09_diffusion_config,
    icon_grid,
    damping_height,
):
    config = r04b09_diffusion_config
    additional_parameters = DiffusionParams(config)
    vct_a = grid_savepoint.vct_a()
    cell_geometry = grid_savepoint.construct_cell_geometry()
    edge_geometry = grid_savepoint.construct_edge_geometry()

    interpolation_state = (
        interpolation_savepoint.construct_interpolation_state_for_diffusion()
    )
    metric_state = metrics_savepoint.construct_metric_state_for_diffusion()

    diffusion = Diffusion()
    diffusion.init(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_params=VerticalModelParams(
            vct_a, damping_height, nflatlev=0, nflat_gradp=0
        ),
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )

    _verify_init_values_against_savepoint(diffusion_savepoint_init, diffusion)


@pytest.mark.datatest
@pytest.mark.parametrize("step_date_init", ["2021-06-20T12:00:50.000"])
def test_verify_diffusion_init_against_other_regular_savepoint(
    r04b09_diffusion_config,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    diffusion_savepoint_init,
    damping_height,
):
    config = r04b09_diffusion_config
    additional_parameters = DiffusionParams(config)

    vertical_params = VerticalModelParams(
        grid_savepoint.vct_a(), damping_height, nflat_gradp=0, nflatlev=0
    )
    interpolation_state = (
        interpolation_savepoint.construct_interpolation_state_for_diffusion()
    )
    metric_state = metrics_savepoint.construct_metric_state_for_diffusion()
    edge_params = grid_savepoint.construct_edge_geometry()
    cell_params = grid_savepoint.construct_cell_geometry()

    diffusion = Diffusion()
    diffusion.init(
        icon_grid,
        config,
        additional_parameters,
        vertical_params,
        metric_state,
        interpolation_state,
        edge_params,
        cell_params,
    )

    _verify_init_values_against_savepoint(diffusion_savepoint_init, diffusion)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "step_date_init, step_date_exit",
    [
        ("2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        ("2021-06-20T12:00:20.000", "2021-06-20T12:00:20.000"),
        ("2021-06-20T12:01:00.000", "2021-06-20T12:01:00.000"),
    ],
)
def test_run_diffusion_single_step(
    diffusion_savepoint_init,
    diffusion_savepoint_exit,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    icon_grid,
    r04b09_diffusion_config,
    damping_height,
):
    dtime = diffusion_savepoint_init.get_metadata("dtime").get("dtime")
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()
    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    interpolation_state = (
        interpolation_savepoint.construct_interpolation_state_for_diffusion()
    )
    metric_state = metrics_savepoint.construct_metric_state_for_diffusion()
    diagnostic_state = diffusion_savepoint_init.construct_diagnostics_for_diffusion()
    prognostic_state = diffusion_savepoint_init.construct_prognostics()
    vct_a = grid_savepoint.vct_a()
    vertical_params = VerticalModelParams(
        vct_a=vct_a, rayleigh_damping_height=damping_height, nflatlev=0, nflat_gradp=0
    )
    config = r04b09_diffusion_config
    additional_parameters = DiffusionParams(config)

    diffusion = Diffusion()
    diffusion.init(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_params=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )
    verify_diffusion_fields(diagnostic_state, prognostic_state, diffusion_savepoint_exit)
    assert diffusion_savepoint_init.fac_bdydiff_v() == diffusion.fac_bdydiff_v
    diffusion.run(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        dtime=dtime,
    )
    verify_diffusion_fields(diagnostic_state, prognostic_state, diffusion_savepoint_exit)

@pytest.mark.datatest
@pytest.mark.parametrize("linit", [True])
def test_run_diffusion_initial_step(
    diffusion_savepoint_init,
    diffusion_savepoint_exit,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    icon_grid,
    r04b09_diffusion_config,
    damping_height,
):
    dtime = diffusion_savepoint_init.get_metadata("dtime").get("dtime")
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()
    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    interpolation_state = (
        interpolation_savepoint.construct_interpolation_state_for_diffusion()
    )
    metric_state = metrics_savepoint.construct_metric_state_for_diffusion()
    diagnostic_state = diffusion_savepoint_init.construct_diagnostics_for_diffusion()
    prognostic_state = diffusion_savepoint_init.construct_prognostics()
    vct_a = grid_savepoint.vct_a()
    vertical_params = VerticalModelParams(
        vct_a=vct_a, rayleigh_damping_height=damping_height, nflatlev=0, nflat_gradp=0
    )
    config = r04b09_diffusion_config
    additional_parameters = DiffusionParams(config)

    diffusion = Diffusion()
    diffusion.init(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_params=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )
    assert diffusion_savepoint_init.fac_bdydiff_v() == diffusion.fac_bdydiff_v

    diffusion.initial_run(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        dtime=dtime,
    )

    verify_diffusion_fields(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        diffusion_savepoint=diffusion_savepoint_exit,
    )
