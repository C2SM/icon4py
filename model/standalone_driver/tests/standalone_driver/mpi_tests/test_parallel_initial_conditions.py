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

from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs, mpi_decomposition
from icon4py.model.standalone_driver import driver_states, standalone_driver
from icon4py.model.standalone_driver.testcases import initial_condition
from icon4py.model.testing import definitions as test_defs, grid_utils, parallel_helpers
from icon4py.model.testing.fixtures.datatest import backend_like, experiment, processor_props


if mpi_decomposition.mpi4py is None:
    pytest.skip("Skipping parallel tests on single node installation", allow_module_level=True)

_log = logging.getLogger(__file__)


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment",
    [
        test_defs.Experiments.JW,
    ],
)
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_initial_condition_jablonowski_williamson_compare_single_multi_rank(
    experiment: test_defs.Experiment,
    tmp_path: pathlib.Path,
    processor_props: decomp_defs.ProcessProperties,
    backend_like: model_backends.BackendLike,
) -> None:
    if experiment.grid.params.limited_area:
        pytest.xfail("Limited-area grids not yet supported")

    _log.info(f"running on {processor_props.comm} with {processor_props.comm_size} ranks")

    backend_name = "embedded"  # shut up pyright/mypy
    for k, v in model_backends.BACKENDS.items():
        if backend_like == v:
            backend_name = k

    grid_file_path = grid_utils._download_grid_file(experiment.grid)

    single_rank_icon4py_driver: standalone_driver.Icon4pyDriver = (
        standalone_driver.initialize_driver(
            output_path=tmp_path / f"ci_driver_output_for_backend_{backend_name}_serial_rank0",
            grid_file_path=grid_file_path,
            log_level="info",
            backend_name=backend_name,
            force_serial_run=True,
        )
    )

    single_rank_ds: driver_states.DriverStates = initial_condition.jablonowski_williamson(
        grid=single_rank_icon4py_driver.grid,
        geometry_field_source=single_rank_icon4py_driver.static_field_factories.geometry_field_source,
        interpolation_field_source=single_rank_icon4py_driver.static_field_factories.interpolation_field_source,
        metrics_field_source=single_rank_icon4py_driver.static_field_factories.metrics_field_source,
        backend=single_rank_icon4py_driver.backend,
        lowest_layer_thickness=single_rank_icon4py_driver.vertical_grid_config.lowest_layer_thickness,
        model_top_height=single_rank_icon4py_driver.vertical_grid_config.model_top_height,
        stretch_factor=single_rank_icon4py_driver.vertical_grid_config.stretch_factor,
        damping_height=single_rank_icon4py_driver.vertical_grid_config.rayleigh_damping_height,
        exchange=single_rank_icon4py_driver.exchange,
    )

    multi_rank_icon4py_driver: standalone_driver.Icon4pyDriver = (
        standalone_driver.initialize_driver(
            output_path=tmp_path / f"ci_driver_output_for_backend_{backend_name}_serial_rank0",
            grid_file_path=grid_file_path,
            log_level="info",
            backend_name=backend_name,
        )
    )

    multi_rank_ds: driver_states.DriverStates = initial_condition.jablonowski_williamson(
        grid=multi_rank_icon4py_driver.grid,
        geometry_field_source=multi_rank_icon4py_driver.static_field_factories.geometry_field_source,
        interpolation_field_source=multi_rank_icon4py_driver.static_field_factories.interpolation_field_source,
        metrics_field_source=multi_rank_icon4py_driver.static_field_factories.metrics_field_source,
        backend=multi_rank_icon4py_driver.backend,
        lowest_layer_thickness=multi_rank_icon4py_driver.vertical_grid_config.lowest_layer_thickness,
        model_top_height=multi_rank_icon4py_driver.vertical_grid_config.model_top_height,
        stretch_factor=multi_rank_icon4py_driver.vertical_grid_config.stretch_factor,
        damping_height=multi_rank_icon4py_driver.vertical_grid_config.rayleigh_damping_height,
        exchange=multi_rank_icon4py_driver.exchange,
    )

    # TODO (jcanton/msimberg): unify the two checks below and remove code duplication
    fields = ["w", "vn", "exner", "theta_v", "rho"]
    serial_reference_fields: dict[str, object] = {
        field_name: getattr(single_rank_ds.prognostics.current, field_name).asnumpy()
        for field_name in fields
    }

    for field_name in fields:
        print(f"verifying field {field_name}")
        global_reference_field = processor_props.comm.bcast(
            serial_reference_fields.get(field_name),
            root=0,
        )
        local_field = getattr(multi_rank_ds.prognostics.current, field_name)
        dim = local_field.domain.dims[0]
        parallel_helpers.check_local_global_field(
            decomposition_info=multi_rank_icon4py_driver.decomposition_info,
            processor_props=processor_props,
            dim=dim,
            global_reference_field=global_reference_field,
            local_field=local_field.asnumpy(),
            check_halos=True,
            atol=0.0,
        )

    fields = ["u", "v"]
    serial_reference_fields: dict[str, object] = {
        field_name: getattr(single_rank_ds.diagnostic, field_name).asnumpy()
        for field_name in fields
    }
    for field_name in fields:
        print(f"verifying diagnostic field {field_name}")
        global_reference_field = processor_props.comm.bcast(
            serial_reference_fields.get(field_name),
            root=0,
        )
        local_field = getattr(multi_rank_ds.diagnostic, field_name)
        dim = local_field.domain.dims[0]
        parallel_helpers.check_local_global_field(
            decomposition_info=multi_rank_icon4py_driver.decomposition_info,
            processor_props=processor_props,
            dim=dim,
            global_reference_field=global_reference_field,
            local_field=local_field.asnumpy(),
            check_halos=True,
            atol=0.0,
        )

