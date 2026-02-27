# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pathlib

import pytest

from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import driver_states, driver_utils, standalone_driver
from icon4py.model.standalone_driver.testcases import initial_condition
from icon4py.model.testing import definitions, grid_utils, serialbox as sb, test_utils
from icon4py.model.testing.fixtures.datatest import (
    backend,
    backend_like,
    damping_height,
    data_provider,
    download_ser_data,
    processor_props,
)


@pytest.mark.embedded_remap_error
@pytest.mark.parametrize("experiment, rank", [(definitions.Experiments.JW, 0)])
@pytest.mark.datatest
def test_standalone_driver_initial_conditions(
    backend_like,
    backend,
    tmp_path: pathlib.Path,
    experiment: definitions.Experiments,
    data_provider,
    rank: int,
) -> None:
    backend_name = None
    for k, v in model_backends.BACKENDS.items():
        if backend_like == v:
            backend_name = k
    icon4py_driver: standalone_driver.Icon4pyDriver = standalone_driver.initialize_driver(
        configuration_file_path="./",
        output_path=tmp_path / f"ci_driver_output_for_backend_{backend_name}",
        grid_file_path=grid_utils._download_grid_file(definitions.Grids.R02B04_GLOBAL),
        log_level=next(iter(driver_utils._LOGGING_LEVELS.keys())),
        backend_name=backend_name,
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
    )
    jabw_exit_savepoint = data_provider.from_savepoint_jabw_exit()
    default_w_1 = data_alloc.zero_field(icon4py_driver.grid, dims.CellDim)

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

    assert test_utils.dallclose(
        ds.prognostics.current.w_1.asnumpy(),
        default_w_1.asnumpy(),
    )
