# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import datetime as dt
import pathlib
import re
import uuid
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest
import uxarray as ux  # type: ignore[import-untyped]  # uxarray has no type hints
import zarr

import icon4py.model.common.exceptions as errors
from icon4py.model.common import dimension as dims, time
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import base, vertical as v_grid
from icon4py.model.common.io import distributed, netcdf_writers, ugrid, writers
from icon4py.model.common.io.io import (
    PHASE_DISTRIBUTE,
    PHASE_WRITE,
    FieldGroupIOConfig,
    FieldGroupMonitor,
    IOConfig,
    IOMonitor,
    OutputBackend,
    OutputInterval,
    OutputMode,
    generate_name,
)
from icon4py.model.common.states import data
from icon4py.model.testing import datatest_utils, definitions as test_defs, grid_utils

from ...fixtures import test_path
from .. import utils as test_io_utils
from .test_distributed import synthetic_decomposition_info


# setting backend to fieldview embedded here.
backend = None


@pytest.mark.parametrize(
    "name, suffix, expected",
    [
        ("output", ".nc", "output_0002.nc"),
        ("outxxput_20220101", ".nc", "outxxput_20220101_0002.nc"),
        ("output_20220101T000000_x", ".zarr", "output_20220101T000000_x_0002.zarr"),
    ],
)
def test_generate_name(name: str, suffix: str, expected: str) -> None:
    counter = 2
    assert expected == generate_name(name, counter, suffix)


def is_valid_uxgrid(file: pathlib.Path | str) -> bool:
    import uxarray as ux  # noqa: PLC0415 [import-outside-top-level]

    grid = ux.open_grid(file)
    try:
        grid.validate()
        return True
    except RuntimeError:
        return False


def test_io_monitor_create_output_path(test_path: pathlib.Path) -> None:
    path_name = test_path.absolute().as_posix() + "/output"
    vertical_config = v_grid.VerticalGridConfig(num_levels=test_io_utils.simple_grid.num_levels)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=gtx.as_field(
            (dims.KHalfDim,),
            np.linspace(12000.0, 0.0, test_io_utils.simple_grid.num_levels + 1),  # type: ignore[arg-type]
        ),
        vct_b=None,
    )
    config = IOConfig(field_groups=[], output_path=path_name)
    monitor = IOMonitor(
        config=config,
        vertical_size=vertical_params,
        horizontal_size=test_io_utils.simple_grid.config.horizontal_config,
        grid_file_name=test_io_utils.grid_file,
        grid_id=uuid.UUID(test_io_utils.simple_grid.id),
        dtime=time.RelativeTime(hours=1),
        process_props=decomposition_defs.SingleNodeProcessProperties(),
        decomposition_info=None,
    )
    assert monitor.path.exists()
    assert monitor.path.is_dir()


def test_io_monitor_write_ugrid_file(test_path: pathlib.Path) -> None:
    path_name = test_path.absolute().as_posix() + "/output"
    vertical_config = v_grid.VerticalGridConfig(num_levels=test_io_utils.simple_grid.num_levels)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=gtx.as_field(
            (dims.KHalfDim,),
            np.linspace(12000.0, 0.0, test_io_utils.simple_grid.num_levels + 1),  # type: ignore[arg-type]
        ),
        vct_b=None,
    )

    config = IOConfig(field_groups=[], output_path=path_name)
    monitor = IOMonitor(
        config=config,
        vertical_size=vertical_params,
        horizontal_size=test_io_utils.simple_grid.config.horizontal_config,
        grid_file_name=test_io_utils.grid_file,
        grid_id=uuid.UUID(test_io_utils.simple_grid.id),
        dtime=time.RelativeTime(hours=1),
        process_props=decomposition_defs.SingleNodeProcessProperties(),
        decomposition_info=None,
    )
    ugrid_file = monitor.path.iterdir().__next__().absolute()
    assert "ugrid.nc" in ugrid_file.name
    assert is_valid_uxgrid(ugrid_file)


@pytest.mark.parametrize(
    "variables",
    (
        ["air_density", "exner_function", "upward_air_velocity"],
        ["normal_velocity", "upward_air_velocity", "theta_v"],
    ),
)
def test_io_monitor_write_and_read_ugrid_dataset(
    test_path: pathlib.Path, variables: list[str]
) -> None:
    path_name = test_path.absolute().as_posix() + "/output"
    grid = grid_utils.get_grid_manager_from_identifier(
        test_defs.Experiments.EXCLAIM_APE.grid,
        num_levels=60,
        keep_skip_values=True,
        allocator=backend,  # type: ignore[arg-type]  # None selects the embedded backend
    ).grid
    vertical_config = v_grid.VerticalGridConfig(num_levels=grid.num_levels)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=gtx.as_field((dims.KHalfDim,), np.linspace(12000.0, 0.0, grid.num_levels + 1)),  # type: ignore[arg-type]
        vct_b=None,
    )

    state = test_io_utils.model_state(grid)
    field_configs = [
        FieldGroupIOConfig(
            output_interval=time.NumTimeSteps(1),
            basename="icon4py_dummy_output",
            variables=variables,
            # uxarray reads the data back together with the ugrid file: netCDF only
            backend=OutputBackend.NETCDF,
            mode=OutputMode.GATHER,
            nc_comment="Writing dummy data from icon4py for testing.",
        )
    ]
    config = IOConfig(field_groups=field_configs, output_path=path_name)
    monitor = IOMonitor(
        config=config,
        vertical_size=vertical_params,
        horizontal_size=grid.config.horizontal_config,
        grid_file_name=test_io_utils.grid_file,
        grid_id=uuid.UUID(grid.id),
        dtime=time.RelativeTime(hours=1),
        process_props=decomposition_defs.SingleNodeProcessProperties(),
        decomposition_info=None,
    )
    current_time = dt.datetime.fromisoformat("2024-01-01T12:00:00")
    for _ in range(3):
        monitor.store(state, current_time)
        current_time = current_time + dt.timedelta(hours=1)
    monitor.close()

    assert len([f for f in monitor.path.iterdir() if f.is_file()]) == 1 + len(field_configs)
    uxds = read_back_as_uxarray(monitor.path)
    for var in variables:
        assert var in uxds.variables
        if var in ["air_density", "exner_function", "theta_v"]:
            assert uxds[var].shape == (3, grid.num_levels, grid.num_cells)
        elif var == "upward_air_velocity":
            assert uxds[var].shape == (3, grid.num_levels + 1, grid.num_cells)
        elif var == "normal_velocity":
            assert uxds[var].shape == (3, grid.num_levels, grid.num_edges)


def test_fieldgroup_monitor_write_dataset_file_roll(test_path: pathlib.Path) -> None:
    grid = grid_utils.get_grid_manager_from_identifier(
        test_defs.Experiments.EXCLAIM_APE.grid,
        num_levels=60,
        keep_skip_values=True,
        allocator=backend,  # type: ignore[arg-type]  # None selects the embedded backend
    ).grid
    vertical_config = v_grid.VerticalGridConfig(num_levels=grid.num_levels)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=gtx.as_field((dims.KHalfDim,), np.linspace(12000.0, 0.0, grid.num_levels + 1)),  # type: ignore[arg-type]
        vct_b=None,
    )

    state = test_io_utils.model_state(grid)
    basename_stub = "icon4py_dummy_output"
    config = FieldGroupIOConfig(
        output_interval=time.NumTimeSteps(1),
        basename=basename_stub,
        variables=["air_density", "exner_function", "upward_air_velocity"],
        timesteps_per_file=1,
        # the test pins the netCDF file-roll naming (``..._000N.nc``)
        backend=OutputBackend.NETCDF,
        mode=OutputMode.GATHER,
    )
    monitor = FieldGroupMonitor(
        config=config,
        vertical=vertical_params,
        distribution=distributed.SingleNodeDistribution(grid.config.horizontal_config),
        grid_id=uuid.UUID(grid.id),
        output_path=test_path,
        dtime=time.RelativeTime(hours=1),
        process_props=decomposition_defs.SingleNodeProcessProperties(),
    )
    current_time = dt.datetime.fromisoformat("2024-01-01T12:00:00")
    for _ in range(4):
        monitor.store(state, current_time)
        current_time = current_time + dt.timedelta(hours=1)
    assert len([f for f in monitor.output_path.iterdir() if f.is_file()]) == 4
    expected_name = re.compile(basename_stub + "_\\d{4}.nc")
    for f in monitor.output_path.iterdir():
        if f.is_file():
            assert expected_name.match(f.name)

            with ugrid.load_data_file(f) as ds:
                assert ds.sizes["time"] == 1
                assert ds.sizes["level"] == grid.num_levels
                assert ds.sizes["cell"] == grid.num_cells
                assert ds.sizes["half_level"] == grid.num_levels + 1
                assert ds.variables["air_density"].shape == (
                    1,
                    grid.num_levels,
                    grid.num_cells,
                )
                assert ds.variables["exner_function"].shape == (
                    1,
                    grid.num_levels,
                    grid.num_cells,
                )
                assert ds.variables["upward_air_velocity"].shape == (
                    1,
                    grid.num_levels + 1,
                    grid.num_cells,
                )


def test_fieldgroup_monitor_refuses_to_overwrite_existing_output(test_path: pathlib.Path) -> None:
    # a first run writes ..._0001.nc; a second run sharing the directory must not
    # silently overwrite it -- the per-run file counter restarts at 0.
    state = test_io_utils.model_state(test_io_utils.simple_grid)
    current_time = dt.datetime.fromisoformat("2024-01-01T00:00:00")

    _, first_monitor = create_field_group_monitor(test_path, test_io_utils.simple_grid)
    first_monitor.store(state, current_time)
    first_monitor.close()

    _, second_monitor = create_field_group_monitor(test_path, test_io_utils.simple_grid)
    with pytest.raises(errors.InvalidConfigError, match="already exists"):
        second_monitor.store(state, current_time)


def read_back_as_uxarray(path: pathlib.Path) -> Any:
    ugrid_file = None
    data_files = []
    for f in path.iterdir():
        if f.is_file():
            if "_ugrid.nc" in f.name:
                ugrid_file = f.absolute()
            else:
                data_files.append(f.absolute())
    uxds = ux.open_dataset(ugrid_file, data_files[0])
    return uxds


def test_fieldgroup_monitor_no_output_between_step_intervals(test_path: pathlib.Path) -> None:
    # output every 3rd step: the first two stores must not produce any output
    _, group_monitor = create_field_group_monitor(
        test_path, test_io_utils.simple_grid, output_interval=time.NumTimeSteps(3)
    )
    state = test_io_utils.model_state(test_io_utils.simple_grid)
    step_time = dt.datetime.fromisoformat("2024-01-01T00:00:00")
    group_monitor.store(state, step_time)
    group_monitor.store(state, step_time + dt.timedelta(hours=1))
    group_monitor.close()
    assert len([f for f in group_monitor.output_path.iterdir() if f.is_file()]) == 0


def test_fieldgroup_monitor_records_phase_timings_per_capture(test_path: pathlib.Path) -> None:
    # one (distribute, write) sample pair per capture step, none for skipped steps
    _, group_monitor = create_field_group_monitor(
        test_path, test_io_utils.simple_grid, output_interval=time.NumTimeSteps(2)
    )
    state = test_io_utils.model_state(test_io_utils.simple_grid)
    step_time = dt.datetime.fromisoformat("2024-01-01T00:00:00")
    for step in range(4):
        group_monitor.store(state, step_time + step * dt.timedelta(hours=1))
    group_monitor.close()
    assert len(group_monitor.phase_seconds[PHASE_DISTRIBUTE]) == 2
    assert len(group_monitor.phase_seconds[PHASE_WRITE]) == 2


def create_field_group_monitor(
    test_path: pathlib.Path,
    grid: base.Grid,
    output_interval: OutputInterval = time.NumTimeSteps(1),
    dtime: time.RelativeTime = time.RelativeTime(hours=1),
) -> tuple[FieldGroupIOConfig, FieldGroupMonitor]:
    config = FieldGroupIOConfig(
        basename="test_empty",
        output_interval=output_interval,
        variables=["exner_function", "air_density"],
        # the tests built on this helper count plain files: netCDF (zarr stores are
        # directories)
        backend=OutputBackend.NETCDF,
        mode=OutputMode.GATHER,
    )
    vertical_config = v_grid.VerticalGridConfig(num_levels=test_io_utils.simple_grid.num_levels)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=gtx.as_field(
            (dims.KHalfDim,),
            np.linspace(12000.0, 0.0, test_io_utils.simple_grid.num_levels + 1),  # type: ignore[arg-type]
        ),
        vct_b=None,
    )

    group_monitor = FieldGroupMonitor(
        config=config,
        vertical=vertical_params,
        distribution=distributed.SingleNodeDistribution(grid.config.horizontal_config),
        grid_id=uuid.UUID(grid.id),
        output_path=test_path,
        dtime=dtime,
        process_props=decomposition_defs.SingleNodeProcessProperties(),
    )
    return config, group_monitor


@pytest.mark.parametrize(
    "basename, output_interval, variables, message",
    [
        (
            "",
            1,
            ["exner_function", "air_density"],
            "Output basename is missing.",
        ),
        (
            "/vars/prognostics",
            1,
            ["exner_function", "air_density"],
            "absolute path",
        ),
        (
            "vars/prognostics",
            1,
            [],
            "No variables provided for output.",
        ),
        (
            "vars/prognostics",
            0,
            ["air_density, exner_function"],
            "Output interval must be positive",
        ),
    ],
)
def test_fieldgroup_config_validate_basename(
    basename: str, output_interval: OutputInterval, variables: list[str], message: str
) -> None:
    with pytest.raises(errors.InvalidConfigError) as err:
        FieldGroupIOConfig(
            basename=basename,
            output_interval=output_interval,
            variables=variables,
        )
    assert message in str(err.value)


def test_fieldgroup_monitor_constructs_output_path_and_filepattern(test_path: pathlib.Path) -> None:
    config = FieldGroupIOConfig(
        basename="vars/prognostics",
        output_interval=time.NumTimeSteps(1),
        variables=["exner_function", "air_density"],
    )
    vertical_size = test_io_utils.simple_grid.config.vertical_size
    horizontal_size = test_io_utils.simple_grid.config.horizontal_config
    group_monitor = FieldGroupMonitor(
        config=config,
        vertical=vertical_size,  # type: ignore[arg-type]  # vertical is unused in this test
        distribution=distributed.SingleNodeDistribution(horizontal_size),
        grid_id=uuid.UUID(test_io_utils.simple_grid.id),
        output_path=test_path,
        dtime=time.RelativeTime(hours=1),
        process_props=decomposition_defs.SingleNodeProcessProperties(),
    )
    assert group_monitor.output_path == test_path.joinpath("vars")
    assert group_monitor.output_path.exists()
    assert group_monitor.output_path.is_dir()
    assert "prognostics" in group_monitor._file_basename


class _SingleRankBlockDistribution:
    """Rank-block layout on a single-rank communicator (serial file handle).

    Stands in for ``RankBlockDistribution``, which the monitor only builds in
    multi-rank runs: the monitor must hand the distribution's ``rank_blocks`` and its
    own communicator to the writer, whatever the communicator size.
    """

    def __init__(self, horizontal_size: base.HorizontalGridSize) -> None:
        self._horizontal_size = horizontal_size
        self._rank_blocks = {
            dim_name: distributed.RankBlock(
                start=0,
                count=size,
                size=size,
                padded_size=size,
                global_size=size,
                global_index=np.arange(size, dtype=np.int64),
            )
            for dim_name, size in writers.horizontal_axis_sizes(horizontal_size).items()
        }

    @property
    def writes_output(self) -> bool:
        return True

    @property
    def output_horizontal_size(self) -> base.HorizontalGridSize:
        return self._horizontal_size

    @property
    def rank_blocks(self) -> dict[str, distributed.RankBlock]:
        return self._rank_blocks

    def prepare(self, state: dict) -> dict:
        return state


def test_fieldgroup_monitor_wires_rank_blocks_into_netcdf_writer(
    test_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin the monitor-to-writer wiring of distributed netCDF output.

    The end-to-end coverage of this wiring lives in the MPI tests, which run only on
    MPI-parallel netCDF4 installations (i.e. on no plain PyPI-wheel setup); this test
    pins on any installation that the writer receives the distribution's rank blocks
    (global-index coordinates end up in the file) and the monitor's communicator.
    """
    monkeypatch.setattr(netcdf_writers, "missing_parallel_support", lambda: None)
    grid = test_io_utils.simple_grid
    config = FieldGroupIOConfig(
        basename="rank_block",
        output_interval=time.NumTimeSteps(1),
        variables=["exner_function", "air_density"],
        backend=OutputBackend.NETCDF,
        mode=OutputMode.DISTRIBUTED,
    )
    vertical_config = v_grid.VerticalGridConfig(num_levels=grid.num_levels)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=gtx.as_field((dims.KHalfDim,), np.linspace(12000.0, 0.0, grid.num_levels + 1)),  # type: ignore[arg-type]
        vct_b=None,
    )
    group_monitor = FieldGroupMonitor(
        config=config,
        vertical=vertical_params,
        distribution=_SingleRankBlockDistribution(grid.config.horizontal_config),
        grid_id=uuid.UUID(grid.id),
        output_path=test_path,
        dtime=time.RelativeTime(hours=1),
        process_props=decomposition_defs.SingleNodeProcessProperties(),
    )
    group_monitor.store(
        test_io_utils.model_state(grid), dt.datetime.fromisoformat("2024-01-01T00:00:00")
    )
    dataset = group_monitor._dataset
    assert isinstance(dataset, netcdf_writers.NETCDFWriter)
    assert dataset._process_props is group_monitor._process_props
    group_monitor.close()
    with ugrid.load_data_file(group_monitor.output_path / "rank_block_0001.nc") as ds:
        for dim_name in ("cell", "edge", "vertex"):
            assert f"{writers.GLOBAL_INDEX_PREFIX}_{dim_name}" in ds.variables
        assert ds.sizes["cell"] == grid.num_cells
        assert ds["air_density"].shape == (1, grid.num_levels, grid.num_cells)


def test_fieldgroup_monitor_throw_exception_on_missing_field(test_path: pathlib.Path) -> None:
    config = FieldGroupIOConfig(
        basename="vars/prognostics",
        output_interval=time.NumTimeSteps(1),
        variables=["exner_function", "air_density", "foo"],
    )
    vertical_size = test_io_utils.simple_grid.config.vertical_size
    horizontal_size = test_io_utils.simple_grid.config.horizontal_config
    group_monitor = FieldGroupMonitor(
        config=config,
        vertical=vertical_size,  # type: ignore[arg-type]  # vertical is unused in this test
        distribution=distributed.SingleNodeDistribution(horizontal_size),
        grid_id=uuid.UUID(test_io_utils.simple_grid.id),
        output_path=test_path,
        dtime=time.RelativeTime(hours=1),
        process_props=decomposition_defs.SingleNodeProcessProperties(),
    )
    with pytest.raises(errors.IncompleteStateError, match="Field 'foo' is missing"):
        group_monitor.store(
            test_io_utils.model_state(test_io_utils.simple_grid),
            dt.datetime.fromisoformat("2023-04-04T11:00:00"),
        )


def test_fieldgroup_config_rejects_invalid_interval() -> None:
    # a string interval is no longer supported: only int (steps) or timedelta
    with pytest.raises(errors.InvalidConfigError, match="must be of type"):
        FieldGroupIOConfig(
            basename="a",
            variables=["air_density"],
            output_interval="1 HOUR",  # type: ignore[arg-type]
        )


def test_fieldgroup_monitor_time_interval_normalized_to_steps(test_path: pathlib.Path) -> None:
    # a 3-hour interval with a 1-hour time step fires every 3rd step
    _, group_monitor = create_field_group_monitor(
        test_path,
        test_io_utils.simple_grid,
        output_interval=time.RelativeTime(hours=3),
        dtime=time.RelativeTime(hours=1),
    )
    state = test_io_utils.model_state(test_io_utils.simple_grid)
    step_time = dt.datetime.fromisoformat("2024-01-01T00:00:00")
    # first two steps: no output
    group_monitor.store(state, step_time)
    group_monitor.store(state, step_time + dt.timedelta(hours=1))
    assert len([f for f in group_monitor.output_path.iterdir() if f.is_file()]) == 0
    # third step: output is written
    group_monitor.store(state, step_time + dt.timedelta(hours=2))
    group_monitor.close()
    assert len([f for f in group_monitor.output_path.iterdir() if f.is_file()]) == 1


def test_fieldgroup_monitor_interval_shorter_than_dtime_raises(test_path: pathlib.Path) -> None:
    with pytest.raises(errors.InvalidConfigError, match="shorter than the model time step"):
        create_field_group_monitor(
            test_path,
            test_io_utils.simple_grid,
            output_interval=time.RelativeTime(minutes=30),
            dtime=time.RelativeTime(hours=1),
        )


@pytest.mark.parametrize(
    "basename, backend",
    [
        ("a.nc", OutputBackend.ZARR),
        ("a.zarr", OutputBackend.NETCDF),
        ("a.nc", OutputBackend.NETCDF),
        ("a.zarr", OutputBackend.ZARR),
    ],
)
def test_fieldgroup_config_rejects_basename_with_extension(
    basename: str, backend: OutputBackend
) -> None:
    """The extension is appended from the backend; a configured one would nest."""
    with pytest.raises(errors.InvalidConfigError, match="extension"):
        FieldGroupIOConfig(basename=basename, variables=["air_density"], backend=backend)


def test_fieldgroup_config_rejects_backend_and_mode_strings() -> None:
    """The config takes enum members only; strings belong to the config-file boundary."""
    with pytest.raises(errors.InvalidConfigError, match="OutputBackend"):
        FieldGroupIOConfig(
            basename="a",
            variables=["air_density"],
            backend="netcdf",  # type: ignore[arg-type]  # invalid on purpose
        )
    with pytest.raises(errors.InvalidConfigError, match="OutputMode"):
        FieldGroupIOConfig(
            basename="a",
            variables=["air_density"],
            backend=OutputBackend.NETCDF,
            mode="gather",  # type: ignore[arg-type]  # invalid on purpose
        )


@pytest.mark.parametrize("field", ["horizontal_chunk_size", "horizontal_shard_size"])
@pytest.mark.parametrize("value", [0, -3, True, 2.5])
def test_fieldgroup_config_rejects_invalid_horizontal_chunking(
    field: str, value: int | float
) -> None:
    with pytest.raises(errors.InvalidConfigError, match="positive integer"):
        FieldGroupIOConfig(
            basename="a",
            variables=["air_density"],
            **{field: value},  # type: ignore[arg-type]  # invalid on purpose
        )


def test_fieldgroup_config_rejects_shard_for_netcdf() -> None:
    with pytest.raises(errors.InvalidConfigError, match="'zarr' backend"):
        FieldGroupIOConfig(
            basename="a",
            variables=["air_density"],
            backend=OutputBackend.NETCDF,
            mode=OutputMode.GATHER,
            horizontal_chunk_size=4,
            horizontal_shard_size=8,
        )


def test_fieldgroup_config_rejects_shard_without_chunk() -> None:
    with pytest.raises(errors.InvalidConfigError, match="requires 'horizontal_chunk_size'"):
        FieldGroupIOConfig(
            basename="a",
            variables=["air_density"],
            horizontal_shard_size=8,
        )


def test_fieldgroup_config_rejects_shard_not_multiple_of_chunk() -> None:
    with pytest.raises(errors.InvalidConfigError, match="not a multiple"):
        FieldGroupIOConfig(
            basename="a",
            variables=["air_density"],
            horizontal_chunk_size=4,
            horizontal_shard_size=10,
        )


def test_fieldgroup_config_block_alignment_is_shard_then_chunk_then_one() -> None:
    default = FieldGroupIOConfig(basename="a", variables=["air_density"])
    assert default.block_alignment == 1
    chunked = FieldGroupIOConfig(basename="a", variables=["air_density"], horizontal_chunk_size=4)
    assert chunked.block_alignment == 4
    sharded = FieldGroupIOConfig(
        basename="a",
        variables=["air_density"],
        horizontal_chunk_size=4,
        horizontal_shard_size=8,
    )
    assert sharded.block_alignment == 8


def test_fieldgroup_monitor_wires_chunking_into_zarr_writer(test_path: pathlib.Path) -> None:
    """Pin that the configured chunk/shard sizes reach the store layout."""
    grid = test_io_utils.simple_grid
    config = FieldGroupIOConfig(
        basename="chunked",
        output_interval=time.NumTimeSteps(1),
        variables=["air_density"],
        backend=OutputBackend.ZARR,
        mode=OutputMode.GATHER,
        horizontal_chunk_size=4,
        horizontal_shard_size=8,
    )
    vertical_config = v_grid.VerticalGridConfig(num_levels=grid.num_levels)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=gtx.as_field((dims.KHalfDim,), np.linspace(12000.0, 0.0, grid.num_levels + 1)),  # type: ignore[arg-type]
        vct_b=None,
    )
    group_monitor = FieldGroupMonitor(
        config=config,
        vertical=vertical_params,
        distribution=distributed.SingleNodeDistribution(grid.config.horizontal_config),
        grid_id=uuid.UUID(grid.id),
        output_path=test_path,
        dtime=time.RelativeTime(hours=1),
        process_props=decomposition_defs.SingleNodeProcessProperties(),
    )
    group_monitor.store(
        test_io_utils.model_state(grid), dt.datetime.fromisoformat("2024-01-01T00:00:00")
    )
    group_monitor.close()
    air_density = zarr.open_group(group_monitor.output_path / "chunked_0001.zarr", mode="r")[
        "air_density"
    ]
    assert isinstance(air_density, zarr.Array)
    assert air_density.chunks == (1, grid.num_levels, 4)
    assert air_density.shards == (1, grid.num_levels, 8)


def test_fieldgroup_config_accepts_distributed_netcdf_on_any_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The config never rejects distributed netCDF: the check is rank-aware.

    Whether the combination is writable depends on the run's rank count (single-rank
    runs use a serial file handle), so it is checked when the writer is created --
    a static config-time rejection would break serial runs on serial installations.
    """
    monkeypatch.setattr(netcdf_writers, "missing_parallel_support", lambda: "<serial build>")
    config = FieldGroupIOConfig(
        basename="a",
        variables=["air_density"],
        backend=OutputBackend.NETCDF,
        mode=OutputMode.DISTRIBUTED,
    )
    assert config.backend is OutputBackend.NETCDF
    assert config.mode is OutputMode.DISTRIBUTED


def test_fieldgroup_monitor_interval_not_multiple_of_dtime_raises(test_path: pathlib.Path) -> None:
    with pytest.raises(errors.InvalidConfigError, match="not a multiple"):
        create_field_group_monitor(
            test_path,
            test_io_utils.simple_grid,
            output_interval=time.RelativeTime(minutes=90),
            dtime=time.RelativeTime(hours=1),
        )


def _simple_grid_vertical() -> v_grid.VerticalGrid:
    num_levels = test_io_utils.simple_grid.num_levels
    return v_grid.VerticalGrid(
        config=v_grid.VerticalGridConfig(num_levels=num_levels),
        vct_a=gtx.as_field((dims.KHalfDim,), np.linspace(12000.0, 0.0, num_levels + 1)),  # type: ignore[arg-type]
        vct_b=None,
    )


def test_io_monitor_ugrid_failure_raises_runtime_error(test_path: pathlib.Path) -> None:
    # a broken grid file must fail loudly (and, in a distributed run, on all ranks)
    config = IOConfig(field_groups=[], output_path=str(test_path / "output"))
    with pytest.raises(RuntimeError, match="UGRID topology"):
        IOMonitor(
            config=config,
            vertical_size=_simple_grid_vertical(),
            horizontal_size=test_io_utils.simple_grid.config.horizontal_config,
            grid_file_name=test_path / "does_not_exist.nc",
            grid_id=uuid.UUID(test_io_utils.simple_grid.id),
            dtime=time.RelativeTime(hours=1),
            process_props=decomposition_defs.SingleNodeProcessProperties(),
            decomposition_info=None,
        )


def test_io_config_time_properties_reach_field_group_monitors(test_path: pathlib.Path) -> None:
    config = IOConfig(
        field_groups=[FieldGroupIOConfig(basename="t", variables=["air_density"])],
        output_path=str(test_path / "output"),
        time_units="hours since 2000-01-01",
        calendar="standard",
    )
    monitor = IOMonitor(
        config=config,
        vertical_size=_simple_grid_vertical(),
        horizontal_size=test_io_utils.simple_grid.config.horizontal_config,
        grid_file_name=test_io_utils.grid_file,
        grid_id=uuid.UUID(test_io_utils.simple_grid.id),
        dtime=time.RelativeTime(hours=1),
        process_props=decomposition_defs.SingleNodeProcessProperties(),
        decomposition_info=None,
    )
    assert monitor._group_monitors[0]._time_properties == writers.TimeProperties(
        "hours since 2000-01-01", "standard"
    )


class _TwoRankComm:
    """Rank-0 view of a two-rank communicator, faking the collectives used at setup."""

    def allgather(self, value: int) -> list[int]:
        return [value, value]

    def bcast(self, value: str, root: int = 0) -> str:
        return value

    def Gatherv(self, send: np.ndarray, recv: list[np.ndarray] | None, root: int = 0) -> None:
        """Pretend the second rank owns the complementary global indices.

        Used only by the setup-time partition validation: the gathered indices of the
        two "ranks" then form a permutation of the global grid, as a real
        decomposition's would.
        """
        if recv is None:
            return
        buffer = recv[0]
        buffer[: send.shape[0]] = send
        buffer[send.shape[0] :] = np.setdiff1d(np.arange(buffer.shape[0]), send)


class _TwoRankProcessProperties:
    comm_name = ""
    rank = 0
    comm_size = 2

    def __init__(self) -> None:
        self.comm = _TwoRankComm()

    def is_single_rank(self) -> bool:
        return False


def test_io_monitor_builds_alignment_aware_rank_block_distributions(
    test_path: pathlib.Path,
) -> None:
    """Distributed groups get alignment-rounded blocks; equal alignments share one instance."""
    sharded = dict(
        variables=["air_density"],
        backend=OutputBackend.ZARR,
        mode=OutputMode.DISTRIBUTED,
        horizontal_chunk_size=4,
        horizontal_shard_size=8,
    )
    config = IOConfig(
        field_groups=[
            FieldGroupIOConfig(basename="a", **sharded),  # type: ignore[arg-type]
            FieldGroupIOConfig(basename="b", **sharded),  # type: ignore[arg-type]
            FieldGroupIOConfig(
                basename="c",
                variables=["air_density"],
                backend=OutputBackend.ZARR,
                mode=OutputMode.DISTRIBUTED,
            ),
        ],
        output_path=str(test_path / "output"),
    )
    monitor = IOMonitor(
        config=config,
        vertical_size=_simple_grid_vertical(),
        horizontal_size=test_io_utils.simple_grid.config.horizontal_config,
        grid_file_name=test_io_utils.grid_file,
        grid_id=uuid.UUID(test_io_utils.simple_grid.id),
        dtime=time.RelativeTime(hours=1),
        process_props=_TwoRankProcessProperties(),
        decomposition_info=synthetic_decomposition_info(),
    )
    aligned, aligned_twin, unaligned = (m._distribution for m in monitor._group_monitors)
    assert aligned is aligned_twin
    assert aligned is not unaligned
    # 4 owned cells: shard alignment 8 rounds the block up, the default keeps it at 4
    assert aligned.rank_blocks is not None and unaligned.rank_blocks is not None
    assert aligned.rank_blocks["cell"].size == 8
    assert aligned.rank_blocks["cell"].padded_size == 16
    assert unaligned.rank_blocks["cell"].size == 4
