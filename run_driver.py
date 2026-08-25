import dataclasses
import datetime
import pathlib
from typing import Callable

import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx as tmx_module
from icon4py.model.common import model_backends
from icon4py.model.common.config import config_io
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.driver import config as driver_config, driver, driver_utils
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    grid_utils,
    serialbox as sb,
    test_utils,
    config as test_config,
)
from icon4py.model.common import model_backends, model_options
from icon4py.model.common.decomposition import (
    definitions as decomposition_defs,
)


def test_warm_bubble(
    *,
    experiment_case: str,
    experiment_description: test_defs.ExperimentDescription,
    tmp_path: pathlib.Path,
    generate_torus_grid: Callable[..., pathlib.Path],
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    allocator = model_backends.get_allocator(backend)
    grid_path = pathlib.Path("/capstor/store/cscs/userlab/cwd01/cong/grids/Torus_Triangles_100km_x_100km_res500m.nc")
    # grid_path = generate_torus_grid(
    #     n_rows=116,
    #     n_cols=100,
    #     edge_length=100.0,
    # )

    driver_utils.configure_logging(
        logging_level="warning",
        print_distributed_debug_msg=False,
        process_props=process_props,
    )

    my_config_path = test_config.EXPERIMENT_CONFIG_PATH / f"{experiment_case}.yaml"

    experiment_config = config_io.read_yaml_str(
        my_config_path.read_text(), driver_config.ExperimentConfig
    )
    # config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)
    
    # test_experiment_config = driver_config.read_experiment_config_from_fortran(config_file_path)
    # breakpoint()
    grid_managers = driver_utils.create_grid_manager(
        grid_file_path=grid_path,
        vertical_grid_config=experiment_config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
 
    ds, icon4py_driver = driver.run_driver(
        config=experiment_config,
        grid_manager=grid_managers,
        process_props=process_props,
        backend=backend,
    )


if __name__ == "__main__":
    backend = model_options.customize_backend(
        program=None, backend=driver_utils.get_backend_from_name("gtfn_cpu")
    )

    process_props = decomposition_defs.get_process_properties(
        decomposition_defs.get_runtype(with_mpi=False)
    )
    test_warm_bubble(
        experiment_case="warm_bubble",
        experiment_description=test_defs.Experiments.WEISMAN_KLEMP_TORUS,
        tmp_path=pathlib.Path("/capstor/scratch/cscs/cong/icon4py/tmp"),
        generate_torus_grid=None,
        process_props=process_props,
        backend=backend
    )