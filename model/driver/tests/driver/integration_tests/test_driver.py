# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import pathlib

import gt4py.next.typing as gtx_typing
import pytest

from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.driver import config as driver_config, driver, driver_utils
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    grid_utils,
    serialbox as sb,
    test_utils,
)

from ..fixtures import *  # noqa: F403


# Tolerances (atol, rtol) per experiment, measured across the CSCS CI backends
# (gtfn_cpu, gtfn_gpu, dace_cpu, dace_gpu).
_TOLERANCES: dict[test_defs.ExperimentDescription, dict[str, tuple[float, float]]] = {
    test_defs.Experiments.JW: {
        "vn": (5.3e-7, 0.0),
        "w": (8e-9, 0.0),
        "exner": (4.5e-11, 5.5e-11),
        "theta_v": (5.5e-8, 1.3e-10),
        "rho": (1.5e-10, 2.2e-10),
    },
    test_defs.Experiments.GAUSS3D: {
        "vn": (4.1e-13, 0.0),
        "w": (8.1e-14, 0.0),
        "exner": (1.3e-10, 1.3e-10),
        "theta_v": (9.3e-8, 3.1e-10),
        "rho": (1.8e-15, 3.7e-15),
    },
    test_defs.Experiments.MCH_CH_R04B09: {
        "vn": (3.5e-3, 0.0),
        "w": (1e-3, 0.0),
        "exner": (6.8e-7, 9.9e-7),
        "theta_v": (1.2e-3, 3.6e-6),
        "rho": (3.5e-6, 3.7e-6),
    },
}


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
            "2008-09-01T00:05:00.000",
            "2008-09-01T00:05:00.000",
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
        (
            test_defs.Experiments.MCH_CH_R04B09,
            2,
            2,
            "2021-06-20T12:00:00.000",
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
            True,
            False,
        ),
        (
            test_defs.Experiments.MCH_CH_R04B09,
            2,
            2,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:20.000",
            "2021-06-20T12:00:20.000",
            False,
            False,
        ),
    ],
)
def test_driver(
    experiment_description: test_defs.ExperimentDescription,
    timeloop_date_init: str,
    timeloop_date_exit: str,
    timeloop_diffusion_linit_init: bool,
    *,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
    savepoint_nonhydro_exit: sb.IconNonHydroExitSavepoint,
    substep_exit: int,
    savepoint_diffusion_exit: sb.IconDiffusionExitSavepoint,
) -> None:
    allocator = model_backends.get_allocator(backend)

    grid_file_path = grid_utils._download_grid_file(experiment_description.grid)
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)

    config = driver_config.read_experiment_config_from_fortran(config_file_path)
    config = config.with_overrides(
        driver={
            "output_path": tmp_path / "ci_driver_output",
            # 'start_of_simulation' stays at the beginning of the experiment: the second
            # MCH_CH_R04B09 case starts the time loop later, i.e. it restarts.
            "start_of_timestepping": datetime.datetime.fromisoformat(timeloop_date_init).replace(
                tzinfo=datetime.UTC
            ),
            "end_of_simulation": datetime.datetime.fromisoformat(timeloop_date_exit).replace(
                tzinfo=datetime.UTC
            ),
        }
    )

    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
    ds, _ = driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )

    prognostics = ds.prognostics.current
    computed = {
        "vn": prognostics.vn,
        "w": prognostics.w,
        "exner": prognostics.exner,
        "theta_v": prognostics.theta_v,
        "rho": prognostics.rho,
    }
    references = {
        "vn": savepoint_diffusion_exit.vn(),
        "w": savepoint_diffusion_exit.w(),
        "exner": savepoint_diffusion_exit.exner(),
        "theta_v": savepoint_diffusion_exit.theta_v(),
        "rho": savepoint_nonhydro_exit.rho_new(),
    }

    tolerances = _TOLERANCES[experiment_description]

    for name, reference in references.items():
        atol, rtol = tolerances[name]
        test_utils.assert_dallclose(
            computed[name].asnumpy(),
            reference.asnumpy(),
            atol=atol,
            rtol=rtol,
            err_msg=name,
        )
