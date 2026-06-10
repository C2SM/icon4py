# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import pytest

from icon4py.model.common import model_backends, model_options
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.standalone_driver import driver_utils, standalone_driver
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    grid_utils,
    serialbox as sb,
    test_utils,
)

from ..fixtures import *  # noqa: F403


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_description, istep_exit, substep_exit, timeloop_date_init, timeloop_date_exit, step_date_exit, timeloop_diffusion_linit_init, timeloop_diffusion_linit_exit",
    [
        (
            test_defs.Experiments.JW,
            2,
            5,
            "2008-09-01T00:00:00.000",
            "2008-09-01T00:15:00.000",  # TODO (jcanton) restore 1-timestep dates in https://github.com/C2SM/icon4py/pull/1304
            "2008-09-01T00:15:00.000",  # TODO (jcanton) restore 1-timestep dates in https://github.com/C2SM/icon4py/pull/1304
            False,
            False,
        ),
        (
            test_defs.Experiments.GAUSS3D,
            2,
            5,
            "2001-01-01T00:00:00.000",
            "2001-01-01T00:00:04.000",
            "2001-01-01T00:00:04.000",
            False,
            False,
        ),
        # TODO (jcanton,msimberg) add MCH_CH_R04B09 Experiment here in https://github.com/C2SM/icon4py/pull/1281
    ],
)
def test_standalone_driver(
    experiment_description: test_defs.ExperimentDescription,
    timeloop_date_init: str,
    timeloop_date_exit: str,
    timeloop_diffusion_linit_init: bool,
    *,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend_like: model_backends.BackendLike,
    savepoint_nonhydro_exit: sb.IconNonHydroExitSavepoint,
    substep_exit: int,
    savepoint_diffusion_exit: sb.IconDiffusionExitSavepoint,
) -> None:

    grid_file_path = grid_utils._download_grid_file(experiment_description.grid)
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)

    backend = model_options.customize_backend(program=None, backend=backend_like)
    config = standalone_driver.build_config(config_file_path)
    config = config.with_driver_overrides(output_path=tmp_path / "ci_driver_output")
    allocator = model_backends.get_allocator(backend)
    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
    ds, _ = standalone_driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )

    rho_sp = savepoint_nonhydro_exit.rho_new()
    exner_sp = savepoint_diffusion_exit.exner()
    theta_sp = savepoint_diffusion_exit.theta_v()
    vn_sp = savepoint_diffusion_exit.vn()
    w_sp = savepoint_diffusion_exit.w()

    test_utils.assert_dallclose(
        ds.prognostics.current.vn.asnumpy(),
        vn_sp.asnumpy(),
        atol=5e-4,  # TODO (jcanton) restore or parameterize tolerances in https://github.com/C2SM/icon4py/pull/1304
    )

    test_utils.assert_dallclose(
        ds.prognostics.current.w.asnumpy(),
        w_sp.asnumpy(),
        atol=3e-6,  # TODO (jcanton) restore or parameterize tolerances in https://github.com/C2SM/icon4py/pull/1304
    )

    test_utils.assert_dallclose(
        ds.prognostics.current.exner.asnumpy(),
        exner_sp.asnumpy(),
        atol=2e-7,  # TODO (jcanton) restore or parameterize tolerances in https://github.com/C2SM/icon4py/pull/1304
    )

    test_utils.assert_dallclose(
        ds.prognostics.current.theta_v.asnumpy(),
        theta_sp.asnumpy(),
        atol=3e-5,  # TODO (jcanton) restore or parameterize tolerances in https://github.com/C2SM/icon4py/pull/1304
    )

    test_utils.assert_dallclose(
        ds.prognostics.current.rho.asnumpy(), rho_sp.asnumpy(), atol=4e-7
    )  # TODO (jcanton) restore or parameterize tolerances in https://github.com/C2SM/icon4py/pull/1304
