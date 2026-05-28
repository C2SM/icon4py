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
from icon4py.model.standalone_driver import driver_utils, initial_condition, standalone_driver
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


@pytest.mark.parametrize(
    "experiment_description",
    [
        definitions.Experiments.JW,
        definitions.Experiments.GAUSS3D,
        definitions.Experiments.MCH_CH_R04B09,
        # TODO (jcanton): open a separate PR to enable EXCLAIM_APE which currently does not verify vn
    ],
)
@pytest.mark.datatest
def test_initial_conditions(
    experiment_description: definitions.ExperimentDescription,
    data_provider: sb.IconSerialDataProvider,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend_like: model_backends.BackendLike,
) -> None:
    grid_file_path = grid_utils._download_grid_file(experiment_description.grid)
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)

    icon4py_driver: standalone_driver.Icon4pyDriver = standalone_driver.initialize_driver(
        grid_file_path=grid_file_path,
        config_file_path=config_file_path,
        log_level=next(iter(driver_utils._LOGGING_LEVELS.keys())),
        output_path=tmp_path / "ci_driver_output",
        backend_like=backend_like,
    )

    ds = initial_condition.create(
        config=icon4py_driver.config.initial_condition,
        vertical_config=icon4py_driver.config.vertical_grid,
        grid=icon4py_driver.grid,
        geometry_field_source=icon4py_driver.static_field_factories.geometry_field_source,
        interpolation_field_source=icon4py_driver.static_field_factories.interpolation_field_source,
        metrics_field_source=icon4py_driver.static_field_factories.metrics_field_source,
        backend=icon4py_driver.backend,
        exchange=icon4py_driver.exchange,
    )
    prognostics_savepoint = data_provider.from_savepoint_prognostics_initial()

    test_utils.assert_dallclose(
        ds.prognostics.current.rho.asnumpy(),
        prognostics_savepoint.rho_now().asnumpy(),
    )

    test_utils.assert_dallclose(
        ds.prognostics.current.exner.asnumpy(),
        prognostics_savepoint.exner_now().asnumpy(),
        atol=1e-14,
    )

    test_utils.assert_dallclose(
        ds.prognostics.current.theta_v.asnumpy(),
        prognostics_savepoint.theta_v_now().asnumpy(),
        atol=1e-11,
    )

    test_utils.assert_dallclose(
        ds.prognostics.current.vn.asnumpy(),
        prognostics_savepoint.vn_now().asnumpy(),
        atol=1e-12,
    )

    test_utils.assert_dallclose(
        ds.prognostics.current.w.asnumpy(),
        prognostics_savepoint.w_now().asnumpy(),
        atol=1e-12,
    )
