# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Parallel output tests for the driver.

A single-rank run provides the reference output; multi-rank runs must reproduce it
(within the multi-rank comparison tolerances) through every parallel output path:
gathered netCDF, gathered zarr and rank-block distributed zarr and netCDF (reassembled
to global order via the stored global-index coordinates). The distributed netCDF path
is exercised only on an MPI-parallel netCDF4 installation (PyPI wheels are serial
builds; see "Parallel netCDF" in ``icon4py.model.common.io``).
"""

import logging
import pathlib

import gt4py.next.typing as gtx_typing
import numpy as np
import pytest
import xarray as xr

from icon4py.model.common import model_backends, time
from icon4py.model.common.decomposition import definitions as decomp_defs, mpi_decomposition
from icon4py.model.common.io import io as common_io, netcdf_writers, writers
from icon4py.model.driver import config as driver_config, driver, driver_io, driver_utils
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    grid_utils,
    test_utils,
)
from icon4py.model.testing.fixtures.datatest import (
    backend,
    backend_like,
    download_ser_data,
    experiment_description,
    process_props,
)


if mpi_decomposition.mpi4py is None:
    pytest.skip("Skipping parallel tests on single node installation", allow_module_level=True)

_log = logging.getLogger(__file__)

#: Multi-rank output paths verified against the single-rank reference. Distributed
#: netCDF joins only on an MPI-parallel netCDF4 installation; the data-free MPI tests
#: (``common/io/mpi_tests/test_parallel_io.py``) report it as an explicit skip instead.
_MULTI_RANK_OUTPUT_COMBINATIONS: list[tuple[common_io.OutputBackend, common_io.OutputMode]] = [
    (common_io.OutputBackend.NETCDF, common_io.OutputMode.GATHER),
    (common_io.OutputBackend.ZARR, common_io.OutputMode.GATHER),
    (common_io.OutputBackend.ZARR, common_io.OutputMode.DISTRIBUTED),
]
if netcdf_writers.missing_parallel_support() is None:
    _MULTI_RANK_OUTPUT_COMBINATIONS.append(
        (common_io.OutputBackend.NETCDF, common_io.OutputMode.DISTRIBUTED)
    )


def _run_driver_with_output(
    experiment_description: test_defs.ExperimentDescription,
    output_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
    *,
    output_backend: common_io.OutputBackend,
    output_mode: common_io.OutputMode,
    config_file_path: pathlib.Path,
) -> pathlib.Path:
    """Run the JW testcase for one step with output enabled; return the output directory."""
    allocator = model_backends.get_allocator(backend)
    grid_file_path = grid_utils._download_grid_file(experiment_description.grid)

    config = driver_config.read_experiment_config_from_fortran(config_file_path)
    config = config.with_overrides(
        driver={
            "output_path": output_path,
            "enable_output": True,
            "output_backend": output_backend,
            "output_mode": output_mode,
            "end_of_simulation": time.NumTimeSteps(1),
        }
    )
    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
    _, icon4py_driver = driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )
    return icon4py_driver.config.driver.output_path


def _open_single_output(
    output_path: pathlib.Path, output_backend: common_io.OutputBackend
) -> xr.Dataset:
    suffix = common_io.FILE_SUFFIXES[output_backend]
    matches = sorted(output_path.rglob(f"{driver_io.DEFAULT_OUTPUT_BASENAME}_*{suffix}"))
    assert len(matches) == 1, f"expected exactly one output under {output_path}, got {matches}"
    match output_backend:
        case common_io.OutputBackend.NETCDF:
            return xr.open_dataset(matches[0], decode_times=False)
        case common_io.OutputBackend.ZARR:
            return xr.open_zarr(matches[0], decode_times=False, mask_and_scale=False)


def _assert_dataset_matches_reference(
    dataset: xr.Dataset, reference: xr.Dataset, atol: float, rtol: float
) -> None:
    assert dataset.sizes["time"] == reference.sizes["time"]
    for name in driver_io.DEFAULT_OUTPUT_VARIABLES:
        test_utils.assert_dallclose(
            dataset[name].values, reference[name].values, atol=atol, rtol=rtol, err_msg=name
        )


def _reassemble_global_order(dataset: xr.Dataset, reference: xr.Dataset) -> xr.Dataset:
    """Undo the rank-block layout of a distributed store via its global indices."""
    reassembled = {}
    for name in driver_io.DEFAULT_OUTPUT_VARIABLES:
        variable = dataset[name]
        horizontal_name = str(variable.dims[-1])
        global_index = dataset[f"{writers.GLOBAL_INDEX_PREFIX}_{horizontal_name}"].values
        valid = global_index >= 0
        global_size = reference.sizes[horizontal_name]
        assert int(valid.sum()) == global_size
        global_values = np.empty((*variable.shape[:-1], global_size), dtype=variable.dtype)
        global_values[..., global_index[valid]] = variable.values[..., valid]
        reassembled[name] = xr.DataArray(global_values, dims=variable.dims)
    return xr.Dataset(reassembled)


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.mpi
@pytest.mark.level("integration")
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize("experiment_description", [test_defs.Experiments.JW])
def test_parallel_output_matches_single_rank_reference(
    download_ser_data: None,
    experiment_description: test_defs.ExperimentDescription,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    if experiment_description.grid.limited_area:
        pytest.xfail("Limited-area grids not yet supported")

    atol, rtol = test_utils.get_mpi_comparison_tolerance(backend, atol=1e-10, rtol=0.0)
    _log.info(f"running on {process_props.comm_size} ranks with atol={atol}, rtol={rtol}")

    # resolved once with the session (multi-rank) process_props: the fortran config is
    # decomposition-independent, and only the mpitask<comm_size> archive is downloaded
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)

    single_rank_output = _run_driver_with_output(
        experiment_description,
        tmp_path / f"serial_reference_rank_{process_props.rank}",
        decomp_defs.SingleNodeProcessProperties(),
        backend,
        output_backend=common_io.OutputBackend.NETCDF,
        output_mode=common_io.OutputMode.GATHER,
        config_file_path=config_file_path,
    )

    multi_rank_outputs = {
        (output_backend, output_mode): _run_driver_with_output(
            experiment_description,
            tmp_path / f"mpi_{output_backend.value}_{output_mode.value}_rank_{process_props.rank}",
            process_props,
            backend,
            output_backend=output_backend,
            output_mode=output_mode,
            config_file_path=config_file_path,
        )
        for output_backend, output_mode in _MULTI_RANK_OUTPUT_COMBINATIONS
    }

    # all ranks must have finished writing before the root rank reads the stores
    process_props.comm.Barrier()
    if process_props.rank != 0:
        return

    with _open_single_output(single_rank_output, common_io.OutputBackend.NETCDF) as reference:
        for (output_backend, output_mode), output_path in multi_rank_outputs.items():
            _log.info(f"verifying {output_backend.value}/{output_mode.value} against reference")
            with _open_single_output(output_path, output_backend) as dataset:
                if output_mode == common_io.OutputMode.DISTRIBUTED:
                    reassembled = _reassemble_global_order(dataset, reference)
                    _assert_dataset_matches_reference(reassembled, reference, atol, rtol)
                    assert dataset.sizes["time"] == reference.sizes["time"]
                else:
                    _assert_dataset_matches_reference(dataset, reference, atol, rtol)
