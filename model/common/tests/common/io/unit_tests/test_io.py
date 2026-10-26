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
import textwrap
import uuid
from typing import Any

import cattrs
import gt4py.next as gtx
import numpy as np
import pytest
import uxarray as ux  # type: ignore[import-untyped]  # uxarray has no type hints
import zarr

import icon4py.model.common.exceptions as errors
from icon4py.model.common import dimension as dims, time
from icon4py.model.common.config import config_io
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import base, vertical as v_grid
from icon4py.model.common.io import distributed, netcdf_writers, ugrid, utils as io_utils, writers
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
    OutputSchedule,
    generate_name,
)
from icon4py.model.common.states import data
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils, definitions as test_defs, grid_utils

from ...fixtures import test_path
from .. import utils as test_io_utils
from .test_distributed import synthetic_decomposition_info


# setting backend to fieldview embedded here.
backend = None


@pytest.fixture
def single_group_config(test_path: pathlib.Path) -> IOConfig:
    """The smallest valid output configuration: one field group, writing below ``test_path``."""
    return IOConfig(
        output_path=test_path / "output",
        field_groups=[FieldGroupIOConfig(basename="single", variables=["air_density"])],
    )


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


def test_io_monitor_create_output_path(single_group_config: IOConfig) -> None:
    vertical_config = v_grid.VerticalGridConfig(num_levels=test_io_utils.simple_grid.num_levels)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=gtx.as_field(
            (dims.KHalfDim,),
            np.linspace(12000.0, 0.0, test_io_utils.simple_grid.num_levels + 1),  # type: ignore[arg-type]
        ),
        vct_b=None,
    )
    monitor = IOMonitor(
        config=single_group_config,
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


def test_io_monitor_write_ugrid_file(single_group_config: IOConfig) -> None:
    vertical_config = v_grid.VerticalGridConfig(num_levels=test_io_utils.simple_grid.num_levels)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=gtx.as_field(
            (dims.KHalfDim,),
            np.linspace(12000.0, 0.0, test_io_utils.simple_grid.num_levels + 1),  # type: ignore[arg-type]
        ),
        vct_b=None,
    )

    monitor = IOMonitor(
        config=single_group_config,
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
    config = IOConfig(output_path=test_path / "output", field_groups=field_configs)
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


def test_fieldgroup_monitor_writes_only_on_interval_steps(test_path: pathlib.Path) -> None:
    # output every 3rd step: output is written at first time step, no output at second and third time steps
    _, group_monitor = create_field_group_monitor(
        test_path, test_io_utils.simple_grid, output_interval=time.NumTimeSteps(3)
    )
    state = test_io_utils.model_state(test_io_utils.simple_grid)
    step_time = dt.datetime.fromisoformat("2024-01-01T00:00:00")
    group_monitor.store(state, step_time)
    assert len([f for f in group_monitor.output_path.iterdir() if f.is_file()]) == 1
    group_monitor.store(state, step_time + dt.timedelta(hours=1))
    group_monitor.store(state, step_time + dt.timedelta(hours=2))
    group_monitor.close()
    assert len([f for f in group_monitor.output_path.iterdir() if f.is_file()]) == 1


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
    # first step: output is written
    group_monitor.store(state, step_time)
    assert len([f for f in group_monitor.output_path.iterdir() if f.is_file()]) == 1
    group_monitor.store(state, step_time + dt.timedelta(hours=1))
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


def test_io_monitor_ugrid_failure_raises_runtime_error(
    test_path: pathlib.Path, single_group_config: IOConfig
) -> None:
    # a broken grid file must fail loudly (and, in a distributed run, on all ranks)
    with pytest.raises(RuntimeError, match="UGRID topology"):
        IOMonitor(
            config=single_group_config,
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
        output_path=test_path / "output",
        field_groups=[FieldGroupIOConfig(basename="t", variables=["air_density"])],
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
        output_path=test_path / "output",
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


def test_io_config_requires_field_groups() -> None:
    with pytest.raises(errors.InvalidConfigError, match="at least one output field group"):
        IOConfig(field_groups=[])


def test_io_config_rejects_duplicate_basenames() -> None:
    # two groups sharing a basename would write the same file series
    with pytest.raises(errors.InvalidConfigError, match="must be unique, duplicated: a"):
        IOConfig(
            field_groups=[
                FieldGroupIOConfig(basename="a", variables=["air_density"]),
                FieldGroupIOConfig(basename="b", variables=["air_density"]),
                FieldGroupIOConfig(basename="a", variables=["exner_function"]),
            ]
        )


def test_io_config_compares_basenames_as_paths() -> None:
    # './sub/../winds' and 'winds' name the same file series, whatever the backend
    with pytest.raises(errors.InvalidConfigError, match="must be unique, duplicated: winds"):
        IOConfig(
            field_groups=[
                FieldGroupIOConfig(basename="winds", variables=["air_density"]),
                FieldGroupIOConfig(
                    basename="./sub/../winds",
                    variables=["air_density"],
                    backend=OutputBackend.NETCDF,
                ),
            ]
        )


@pytest.mark.parametrize("basename", [".", "./", "..", "../winds", "sub/../../winds"])
def test_fieldgroup_config_rejects_basename_outside_the_output_directory(basename: str) -> None:
    with pytest.raises(errors.InvalidConfigError, match="below the output directory"):
        FieldGroupIOConfig(basename=basename, variables=["air_density"])


def test_fieldgroup_config_rejects_duplicate_variables() -> None:
    with pytest.raises(errors.InvalidConfigError, match="duplicated names: air_density"):
        FieldGroupIOConfig(basename="a", variables=["air_density", "exner_function", "air_density"])


@pytest.mark.parametrize("value", [True, 2.5, "3"])
def test_fieldgroup_config_rejects_non_integer_timesteps_per_file(value: object) -> None:
    with pytest.raises(errors.InvalidConfigError, match="'timesteps_per_file': must be an integer"):
        FieldGroupIOConfig(
            basename="a",
            variables=["air_density"],
            timesteps_per_file=value,  # type: ignore[arg-type]  # invalid on purpose
        )


def test_io_monitor_variables_at_capture_time(test_path: pathlib.Path) -> None:
    """The union of the variables of the groups capturing at the current store call."""
    config = IOConfig(
        output_path=test_path / "output",
        field_groups=[
            FieldGroupIOConfig(
                basename="every_step",
                variables=["air_density"],
                backend=OutputBackend.NETCDF,
                mode=OutputMode.GATHER,
            ),
            FieldGroupIOConfig(
                basename="every_other_step",
                variables=["exner_function", "air_density"],
                output_interval=time.NumTimeSteps(2),
                backend=OutputBackend.NETCDF,
                mode=OutputMode.GATHER,
            ),
        ],
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
    state = test_io_utils.model_state(test_io_utils.simple_grid)
    step_time = dt.datetime.fromisoformat("2024-01-01T00:00:00")

    # the predicate is pure: asking twice gives the same answer
    assert monitor.variables_at_capture_time() == {"air_density", "exner_function"}
    assert monitor.variables_at_capture_time() == {"air_density", "exner_function"}
    monitor.store(state, step_time)
    assert monitor.variables_at_capture_time() == {"air_density"}
    monitor.store(state, step_time + dt.timedelta(hours=1))
    assert monitor.variables_at_capture_time() == {"air_density", "exner_function"}
    monitor.close()


def test_io_monitor_writes_horizontal_only_fields(test_path: pathlib.Path) -> None:
    """A purely horizontal (surface) field is written as (time, cell) next to the 3D ones."""
    grid = test_io_utils.simple_grid
    state = test_io_utils.model_state(grid)
    state["surface_pressure"] = io_utils.to_data_array(
        data_alloc.random_field(grid, dims.CellDim),
        data.DIAGNOSTIC_CF_ATTRIBUTES["surface_pressure"],
    )
    config = IOConfig(
        output_path=test_path / "output",
        field_groups=[
            FieldGroupIOConfig(
                basename="surface",
                variables=["surface_pressure", "air_density"],
                backend=OutputBackend.NETCDF,
                mode=OutputMode.GATHER,
            )
        ],
    )
    monitor = IOMonitor(
        config=config,
        vertical_size=_simple_grid_vertical(),
        horizontal_size=grid.config.horizontal_config,
        grid_file_name=test_io_utils.grid_file,
        grid_id=uuid.UUID(grid.id),
        dtime=time.RelativeTime(hours=1),
        process_props=decomposition_defs.SingleNodeProcessProperties(),
        decomposition_info=None,
    )
    step_time = dt.datetime.fromisoformat("2024-01-01T00:00:00")
    monitor.store(state, step_time)
    monitor.store(state, step_time + dt.timedelta(hours=1))
    monitor.close()

    (output_file,) = sorted(monitor.path.glob("surface_*.nc"))
    with ugrid.load_data_file(output_file) as ds:
        assert ds["surface_pressure"].dims == ("time", "cell")
        assert ds["surface_pressure"].shape == (2, grid.num_cells)
        assert ds["surface_pressure"].attrs["standard_name"] == "air_pressure_at_ground_level"
        np.testing.assert_array_equal(
            ds["surface_pressure"].values[1], state["surface_pressure"].values
        )
        assert ds["air_density"].dims == ("time", "level", "cell")


@pytest.mark.parametrize(
    ("interval_yaml", "interval"),
    [
        pytest.param("  type: numsteps\n  value: 10\n", time.NumTimeSteps(10), id="numsteps"),
        pytest.param(
            "  type: relative\n  value: 1800 seconds\n",
            time.RelativeTime(minutes=30),
            id="relative",
        ),
    ],
)
def test_fieldgroup_config_yaml_roundtrip(interval_yaml: str, interval: OutputInterval) -> None:
    """The YAML form of a field group: tagged interval, lower-case enum names."""
    yaml_str = (
        "basename: a\n"
        "variables:\n"
        "  - air_density\n"
        "output_interval:\n" + interval_yaml + "timesteps_per_file: 10\n"
        "backend: zarr\n"
        "mode: distributed\n"
        "horizontal_chunk_size:\n"
        "horizontal_shard_size:\n"
        "nc_title: ICON4Py Simulation\n"
        "nc_comment: ICON inspired code in Python and GT4Py\n"
    )
    config = config_io.read_yaml_str(yaml_str, FieldGroupIOConfig)
    assert config == FieldGroupIOConfig(
        basename="a", variables=["air_density"], output_interval=interval
    )
    assert config_io.write_yaml_str(config) == yaml_str


def test_fieldgroup_config_yaml_rejects_untagged_interval() -> None:
    # the interval is tagged like 'end_of_simulation': a bare number is ambiguous
    with pytest.raises(cattrs.ClassValidationError):
        config_io.read_yaml_str(
            "basename: a\nvariables: [air_density]\noutput_interval: 10\n", FieldGroupIOConfig
        )


def test_io_config_yaml_roundtrip_two_groups() -> None:
    yaml_str = textwrap.dedent(
        """\
        output_path: output
        field_groups:
          - basename: winds
            variables:
              - eastward_wind
              - northward_wind
            output_interval:
              type: numsteps
              value: 10
            timesteps_per_file: 10
            backend: zarr
            mode: distributed
            horizontal_chunk_size:
            horizontal_shard_size:
            nc_title: ICON4Py Simulation
            nc_comment: ICON inspired code in Python and GT4Py
          - basename: precipitation
            variables:
              - precipitation_flux
            output_interval:
              type: numsteps
              value: 1
            timesteps_per_file: 0
            backend: netcdf
            mode: gather
            horizontal_chunk_size:
            horizontal_shard_size:
            nc_title: ICON4Py Simulation
            nc_comment: ICON inspired code in Python and GT4Py
        time_units: seconds since 1970-01-01 00:00:00
        calendar: proleptic_gregorian
        io_ranks_per_node: 0
        """
    )
    config = config_io.read_yaml_str(yaml_str, IOConfig)
    assert config.output_path == pathlib.Path("output")
    assert [group.basename for group in config.field_groups] == ["winds", "precipitation"]
    assert config.field_groups[0].output_interval == 10
    assert config.field_groups[1].backend is OutputBackend.NETCDF
    assert config.field_groups[1].mode is OutputMode.GATHER
    assert config.field_groups[1].timesteps_per_file == 0
    assert config.io_ranks_per_node == 0
    assert config_io.write_yaml_str(config) == yaml_str


def test_output_schedule_counts_steps() -> None:
    schedule = OutputSchedule(time.NumTimeSteps(3), time.RelativeTime(hours=1))
    captures = []
    for _ in range(7):
        # the predicate is pure: asking twice gives the same answer
        assert schedule.at_capture_time() == schedule.at_capture_time()
        captures.append(schedule.at_capture_time())
        schedule.advance()
    assert captures == [True, False, False, True, False, False, True]


def test_output_schedule_normalizes_time_deltas() -> None:
    dtime = time.RelativeTime(hours=1)
    schedule = OutputSchedule(time.RelativeTime(hours=2), dtime)
    captures = []
    for _ in range(4):
        captures.append(schedule.at_capture_time())
        schedule.advance()
    assert captures == [True, False, True, False]
    with pytest.raises(errors.InvalidConfigError, match="not a multiple"):
        OutputSchedule(time.RelativeTime(minutes=90), dtime)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_io_config_rejects_invalid_io_ranks_per_node(value: object) -> None:
    group = FieldGroupIOConfig(basename="single", variables=["air_density"])
    with pytest.raises(errors.InvalidConfigError, match="'io_ranks_per_node'"):
        IOConfig(field_groups=[group], io_ranks_per_node=value)  # type: ignore[arg-type]  # invalid on purpose
    assert IOConfig(field_groups=[group]).io_ranks_per_node == 0
    assert IOConfig(field_groups=[group], io_ranks_per_node=2).io_ranks_per_node == 2
