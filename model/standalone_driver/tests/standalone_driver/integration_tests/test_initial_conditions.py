# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import gt4py.next.typing as gtx_typing
import pytest

from icon4py.model.common import initial_condition, model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.states import data, prognostic_state as prognostics
from icon4py.model.standalone_driver import driver_utils, standalone_driver
from icon4py.model.testing import definitions, grid_utils, serialbox as sb, test_utils
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    process_props,
)


# Tolerances (atol, rtol) per experiment.
# rtol is 0.0 where the reference field contains zeros or near-zeros: there no
# meaningful rtol can cover the difference, only atol.
_TOLERANCES: dict[definitions.ExperimentDescription, dict[str, tuple[float, float]]] = {
    definitions.Experiments.JW: {
        "rho": (0.0, 1e-12),
        "exner": (1e-14, 1e-12),
        "theta_v": (1e-11, 1e-12),
        "vn": (1e-12, 1e-12),
        "w": (1e-12, 1e-12),
    },
    definitions.Experiments.EXCLAIM_APE_AES: {
        "rho": (0.0, 1e-12),
        "exner": (1e-14, 1e-12),
        "theta_v": (1e-11, 1e-12),
        "vn": (1e-12, 1e-12),
        "w": (1e-12, 1e-12),
        "qv": (0.0, 1e-12),
    },
    definitions.Experiments.GAUSS3D: {
        "rho": (0.0, 1e-12),
        "exner": (1e-14, 1e-12),
        "theta_v": (1e-11, 1e-12),
        "vn": (1e-12, 1e-12),
        "w": (1e-12, 1e-12),
    },
    definitions.Experiments.MCH_CH_R04B09: {
        "rho": (0.0, 1e-12),
        "exner": (1e-14, 1e-12),
        "theta_v": (1e-11, 1e-12),
        "vn": (1e-12, 1e-12),
        "w": (1e-12, 1e-12),
    },
}


@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_description",
    [
        definitions.Experiments.JW,
        definitions.Experiments.EXCLAIM_APE_AES,
        definitions.Experiments.GAUSS3D,
        definitions.Experiments.MCH_CH_R04B09,
    ],
)
@pytest.mark.datatest
def test_initial_conditions(
    experiment_description: definitions.ExperimentDescription,
    experiment: definitions.Experiment,
    *,
    data_provider: sb.IconSerialDataProvider,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    allocator = model_backends.get_allocator(backend)

    grid_file_path = grid_utils._download_grid_file(experiment.grid)

    config = experiment.config.with_overrides(driver={"output_path": tmp_path / "ci_driver_output"})

    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
    icon4py_driver: standalone_driver.Icon4pyDriver = standalone_driver.initialize_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )

    prognostic_state_now = prognostics.initialize_prognostic_state(
        grid=icon4py_driver.grid,
        allocator=allocator,
        tracer_config=icon4py_driver.config.tracer_config,
    )
    initial_condition.create(
        config=icon4py_driver.config.initial_condition,
        vertical_config=icon4py_driver.config.vertical_grid,
        grid=icon4py_driver.grid,
        static_fields=icon4py_driver.static_field_factories,
        prognostic_state_now=prognostic_state_now,
        backend=icon4py_driver.backend,
        exchange=icon4py_driver.exchange,
        global_reductions=icon4py_driver.global_reductions,
    )
    prognostics_savepoint = data_provider.from_savepoint_prognostics_initial()

    computed = {
        "rho": prognostic_state_now.rho,
        "exner": prognostic_state_now.exner,
        "theta_v": prognostic_state_now.theta_v,
        "vn": prognostic_state_now.vn,
        "w": prognostic_state_now.w,
    }
    references = {
        "rho": prognostics_savepoint.rho_now(),
        "exner": prognostics_savepoint.exner_now(),
        "theta_v": prognostics_savepoint.theta_v_now(),
        "vn": prognostics_savepoint.vn_now(),
        "w": prognostics_savepoint.w_now(),
    }

    # Moist experiments (e.g. APE) initialize the water-vapour tracer
    if prognostic_state_now.tracer.qv is not None:
        computed["qv"] = prognostic_state_now.tracer.qv
        references["qv"] = prognostics_savepoint.tracer_now(data.QV)

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
