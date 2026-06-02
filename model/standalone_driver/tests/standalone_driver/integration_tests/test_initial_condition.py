# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pathlib

import pytest

from icon4py.model.common import model_backends
from icon4py.model.standalone_driver import driver_utils, standalone_driver
from icon4py.model.standalone_driver.testcases import initial_condition
from icon4py.model.testing import definitions, grid_utils, serialbox as sb, test_utils
from icon4py.model.testing.fixtures.datatest import (
    backend,
    backend_like,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    process_props,
)


@pytest.mark.embedded_remap_error
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.JW, definitions.Experiments.GAUSS3D])
@pytest.mark.datatest
def test_initial_conditions(
    backend_like: model_backends.BackendLike,
    experiment: definitions.Experiment,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    icon4py_driver: standalone_driver.Icon4pyDriver = standalone_driver.initialize_driver(
        grid_file_path=grid_utils._download_grid_file(experiment.grid),
        config_file_path=experiment.config.file_path,
        log_level=next(iter(driver_utils._LOGGING_LEVELS.keys())),
        backend_like=backend_like,
    )

    ds = initial_condition.create(
        experiment_name=experiment.description.name,
        grid=icon4py_driver.grid,
        geometry_field_source=icon4py_driver.static_field_factories.geometry_field_source,
        interpolation_field_source=icon4py_driver.static_field_factories.interpolation_field_source,
        metrics_field_source=icon4py_driver.static_field_factories.metrics_field_source,
        backend=icon4py_driver.backend,
        lowest_layer_thickness=icon4py_driver.vertical_grid_config.lowest_layer_thickness,
        model_top_height=icon4py_driver.vertical_grid_config.model_top_height,
        stretch_factor=icon4py_driver.vertical_grid_config.stretch_factor,
        damping_height=icon4py_driver.vertical_grid_config.rayleigh_damping_height,
        exchange=icon4py_driver.exchange,
    )
    prognostics_savepoint = data_provider.from_savepoint_prognostics_initial()

    assert test_utils.dallclose(
        ds.prognostics.current.rho.asnumpy(),
        prognostics_savepoint.rho_now().asnumpy(),
    )

    assert test_utils.dallclose(
        ds.prognostics.current.exner.asnumpy(),
        prognostics_savepoint.exner_now().asnumpy(),
        atol=1e-14,
    )

    assert test_utils.dallclose(
        ds.prognostics.current.theta_v.asnumpy(),
        prognostics_savepoint.theta_v_now().asnumpy(),
        atol=1e-11,
    )

    assert test_utils.dallclose(
        ds.prognostics.current.vn.asnumpy(),
        prognostics_savepoint.vn_now().asnumpy(),
        atol=1e-12,
    )

    assert test_utils.dallclose(
        ds.prognostics.current.w.asnumpy(),
        prognostics_savepoint.w_now().asnumpy(),
        atol=1e-12,
    )
