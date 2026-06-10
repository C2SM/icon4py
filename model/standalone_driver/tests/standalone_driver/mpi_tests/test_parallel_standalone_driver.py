# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import pathlib

import pytest

from icon4py.model.common import model_backends, model_options
from icon4py.model.common.decomposition import definitions as decomp_defs, mpi_decomposition
from icon4py.model.standalone_driver import driver_utils, standalone_driver
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    grid_utils,
    parallel_helpers,
    test_utils,
)
from icon4py.model.testing.fixtures.datatest import (
    backend_like,
    download_ser_data,
    experiment,
    experiment_description,
    process_props,
)


if mpi_decomposition.mpi4py is None:
    pytest.skip("Skipping parallel tests on single node installation", allow_module_level=True)

_log = logging.getLogger(__file__)


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_description",
    [
        test_defs.Experiments.JW,
        test_defs.Experiments.GAUSS3D,
    ],
)
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_standalone_driver_compare_single_multi_rank(
    experiment_description: test_defs.ExperimentDescription,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend_like: model_backends.BackendLike,
) -> None:
    if experiment_description.grid.limited_area:
        pytest.xfail("Limited-area grids not yet supported")

    backend = model_options.customize_backend(program=None, backend=backend_like)
    allocator = model_backends.get_allocator(backend)

    if model_backends.is_cpu_backend(backend_like) and test_utils.is_gtfn_backend(backend):
        atol = 1e-13
        rtol = 1e-14
    else:
        atol = 2e-12
        rtol = 0.0

    _log.info(
        f"running on {process_props.comm} with {process_props.comm_size} ranks and atol = {atol}, rtol = {rtol}"
    )

    grid_file_path = grid_utils._download_grid_file(experiment_description.grid)
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)

    config = standalone_driver.build_config(config_file_path)

    serial_process_props = decomp_defs.get_process_properties(
        decomp_defs.get_runtype(with_mpi=False)
    )
    serial_config = config.with_driver_overrides(
        output_path=tmp_path / "ci_driver_output_serial_rank0"
    )
    serial_grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=serial_process_props,
    )
    single_rank_ds, _ = standalone_driver.run_driver(
        config=serial_config,
        grid_manager=serial_grid_manager,
        process_props=serial_process_props,
        backend=backend,
    )

    mpi_config = config.with_driver_overrides(
        output_path=tmp_path / f"ci_driver_output_mpi_rank_{process_props.rank}"
    )
    mpi_grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
    multi_rank_ds, multi_rank_driver = standalone_driver.run_driver(
        config=mpi_config,
        grid_manager=mpi_grid_manager,
        process_props=process_props,
        backend=backend,
    )

    fields = ["vn", "w", "exner", "theta_v", "rho"]
    serial_reference_fields: dict[str, object] = {
        field_name: getattr(single_rank_ds.prognostics.current, field_name).asnumpy()
        for field_name in fields
    }

    for field_name in fields:
        print(f"\nverifying field {field_name}")
        global_reference_field = process_props.comm.bcast(
            serial_reference_fields.get(field_name),
            root=0,
        )
        local_field = getattr(multi_rank_ds.prognostics.current, field_name)
        dim = local_field.domain.dims[0]
        parallel_helpers.check_local_global_field(
            decomposition_info=multi_rank_driver.decomposition_info,
            process_props=process_props,
            dim=dim,
            global_reference_field=global_reference_field,
            local_field=local_field.asnumpy(),
            check_halos=True,
            atol=atol,
            rtol=rtol,
        )
