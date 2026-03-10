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

from icon4py.model.common import dimension as dims, model_backends, model_options
from icon4py.model.common.decomposition import definitions as decomp_defs, mpi_decomposition
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import driver_utils, main
from icon4py.model.testing import definitions as test_defs, grid_utils
from icon4py.model.testing.fixtures.datatest import backend_like, experiment, processor_props
from icon4py.model.testing.parallel_helpers import check_local_global_field


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

    _log.info(f"running on {processor_props.comm} with {processor_props.comm_size} ranks")

    backend_name = "embedded"  # shut up pyright/mypy
    for k, v in model_backends.BACKENDS.items():
        if backend_like == v:
            backend_name = k
    backend = model_options.customize_backend(
        program=None, backend=driver_utils.get_backend_from_name(backend_name)
    )
    array_ns = data_alloc.import_array_ns(backend)  # type: ignore[arg-type] # backend type is correct

    grid_file_path = grid_utils._download_grid_file(experiment.grid)

    fields = ["vn", "w", "exner", "theta_v", "rho"]
    serial_reference_fields: dict[str, object] = {}
    if processor_props.rank == 0:
        single_rank_ds, _ = main.main(
            grid_file_path=grid_file_path,
            icon4py_backend=backend_name,
            output_path=tmp_path / f"ci_driver_output_for_backend_{backend_name}_serial_rank0",
            array_ns=array_ns,
            force_serial_run=True,
        )
        serial_reference_fields = {
            field_name: getattr(single_rank_ds.prognostics.current, field_name).asnumpy()
            for field_name in fields
        }

    multi_rank_ds, decomposition_info = main.main(
        grid_file_path=grid_file_path,
        icon4py_backend=backend_name,
        output_path=tmp_path / f"ci_driver_output_for_backend_{backend_name}_mpi_rank_{processor_props.rank}",
        array_ns=array_ns,
    )

    for field_name in fields:
        global_reference_field = processor_props.comm.bcast(
            serial_reference_fields.get(field_name),
            root=0,
        )
        local_field = getattr(multi_rank_ds.prognostics.current, field_name)
        dim = local_field.domain.dims[0]
        check_local_global_field(
            decomposition_info=decomposition_info,
            processor_props=processor_props,
            dim=dim,
            global_reference_field=global_reference_field,
            local_field=local_field.asnumpy(),
            check_halos=True,
        )
