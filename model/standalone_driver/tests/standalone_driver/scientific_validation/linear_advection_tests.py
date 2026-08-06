# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.resources
import pathlib

import gt4py.next.typing as gtx_typing
import pytest

from icon4py.model.common import model_backends
from icon4py.model.common.config import config_io
from icon4py.model.common.initial_condition.analytical import linear_advection
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.standalone_driver import config as driver_config, driver_utils, standalone_driver
from icon4py.model.testing import (
    definitions as test_defs,
    grid_utils,
    config as test_config,
)
from ..fixtures import *  # noqa: F403


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_case, grid_description",
    [
        (
            "linear_adv",
            (test_defs.Grids.TORUS_1000X1000_100M, test_defs.Grids.TORUS_1000X1000_50M, test_defs.Grids.TORUS_1000X1000_25M),
        ),
    ],
)
def horizontal_advection_test(
    experiment_case: str,
    grid_description: test_defs.GridDescription,
    *,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    allocator = model_backends.get_allocator(backend)

    grid_file_path = (grid_utils._download_grid_file(grid) for grid in grid_description)
    error_l2 = []

    config_path = (
        test_config.EXPERIMENT_CONFIG_PATH
        / f"{experiment_case}.yaml"
    )
    
    for grid in grid_file_path:
        experiment_config = config_io.read_yaml_str(
            config_path.read_text(), driver_config.ExperimentConfig
        ).with_overrides(
            driver={"output_path": tmp_path / "ci_driver_output"},
        )

        grid_manager = driver_utils.create_grid_manager(
            grid_file_path=grid,
            vertical_grid_config=experiment_config.vertical_grid,
            allocator=allocator,
            process_props=process_props,
        )
        
        vel_max = linear_advection.compute_max_velocity(
            experiment_config.initial_condition.velocity_field,
            grid_manager.grid.grid_params.domain_length,
            grid_manager.grid.grid_params.domain_height,
        )
        end_time = experiment_config.driver.end_of_simulation.total_seconds()
        dtime = min(experiment_config.initial_condition.cfl_number * grid_manager.grid.grid_params.domain_length / vel_max / (4.0 * (3.0**0.5)), experiment_config.driver.end_of_simulation.total_seconds())
        print("debugging num of steps: ", end_time / dtime, int(end_time / dtime), dtime, end_time)
        experiment_config.with_overrides(
            driver={"dtime": dtime},
        )

        ds, _ = standalone_driver.run_driver(
            config=experiment_config,
            grid_manager=grid_manager,
            process_props=process_props,
            backend=backend,
        )
        simulated_tracer = ds.tracers.current.qv.asumpy()

        # get reference solution
        tracer_reference = linear_advection.construct_reference_tracer_numpy(
            test_config,
            icon_grid,
            x_center,
            y_center,
            x_range,
            y_range,
            edges_center_x,
            edges_center_y,
            node_x,
            node_y,
            time,
            time_end,
            weights,
            nodes,
            tracer_reference_high,
            cell_center_x_high,
            cell_center_y_high,
        )

        error_l2.append()