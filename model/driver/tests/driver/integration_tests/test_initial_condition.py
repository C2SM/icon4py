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
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.driver import driver, driver_utils
from icon4py.model.driver.testcases import initial_condition
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions,
    grid_utils,
    serialbox as sb,
    test_utils,
)
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
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.JW])
@pytest.mark.datatest
def test_driver_initial_condition(
    backend_like: model_backends.BackendLike,
    tmp_path: pathlib.Path,
    experiment_description: definitions.ExperimentDescription,
    process_props: decomp_defs.ProcessProperties,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    grid_file_path = grid_utils._download_grid_file(experiment_description.grid)
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)
    icon4py_driver: driver.Icon4pyDriver = driver.initialize_driver(
        grid_file_path=grid_file_path,
        config_file_path=config_file_path,
        output_path=tmp_path / "ci_driver_output",
        log_level=next(iter(driver_utils._LOGGING_LEVELS.keys())),
        backend_like=backend_like,
    )

    ds = initial_condition.jablonowski_williamson(
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
    jabw_exit_savepoint = data_provider.from_savepoint_jabw_exit()

    assert test_utils.dallclose(
        ds.prognostics.current.rho.asnumpy(),
        jabw_exit_savepoint.rho().asnumpy(),
    )

    assert test_utils.dallclose(
        ds.prognostics.current.vn.asnumpy(),
        jabw_exit_savepoint.vn().asnumpy(),
        atol=1e-12,
    )

    assert test_utils.dallclose(
        ds.prognostics.current.w.asnumpy(),
        jabw_exit_savepoint.w().asnumpy(),
        atol=1e-12,
    )

    assert test_utils.dallclose(
        ds.prognostics.current.exner.asnumpy(), jabw_exit_savepoint.exner().asnumpy(), atol=1e-14
    )

    assert test_utils.dallclose(
        ds.prognostics.current.theta_v.asnumpy(),
        jabw_exit_savepoint.theta_v().asnumpy(),
        atol=1e-11,
    )
