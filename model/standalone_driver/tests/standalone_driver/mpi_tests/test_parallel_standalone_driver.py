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
from icon4py.model.standalone_driver import driver_utils, main
from icon4py.model.testing import (
    data_handling,
    datatest_utils as dt_utils,
    definitions as test_defs,
    grid_utils,
    parallel_helpers,
)
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
def test_standalone_driver_compare_single_multi_rank(
    experiment: test_defs.Experiment,
    tmp_path: pathlib.Path,
    processor_props: decomp_defs.ProcessProperties,
    backend_like: model_backends.BackendLike,
) -> None:
    if experiment.grid.params.limited_area:
        pytest.xfail("Limited-area grids not yet supported")

    atol = 0.0 if model_backends.is_cpu_backend(backend_like) else 1e-6

    _log.info(
        f"running on {processor_props.comm} with {processor_props.comm_size} ranks and tolerance = {atol}"
    )

    grid_file_path = grid_utils._download_grid_file(experiment.grid)

    single_rank_ds, _ = main.main(
        grid_file_path=grid_file_path,
        icon4py_backend=backend_like,
        output_path=tmp_path / "ci_driver_output_serial_rank0",
        force_serial_run=True,
    )

    multi_rank_ds, decomposition_info = main.main(
        grid_file_path=grid_file_path,
        icon4py_backend=backend_like,
        output_path=tmp_path / f"ci_driver_output_mpi_rank_{processor_props.rank}",
    )

    fields = ["vn", "w", "exner", "theta_v", "rho"]
    serial_reference_fields: dict[str, object] = {
        field_name: getattr(single_rank_ds.prognostics.current, field_name).asnumpy()
        for field_name in fields
    }

    for field_name in fields:
        print(f"\nverifying field {field_name}")
        global_reference_field = processor_props.comm.bcast(
            serial_reference_fields.get(field_name),
            root=0,
        )
        local_field = getattr(multi_rank_ds.prognostics.current, field_name)
        dim = local_field.domain.dims[0]
        parallel_helpers.check_local_global_field(
            decomposition_info=decomposition_info,
            processor_props=processor_props,
            dim=dim,
            global_reference_field=global_reference_field,
            local_field=local_field.asnumpy(),
            check_halos=True,
            atol=atol,
        )
