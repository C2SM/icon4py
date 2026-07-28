# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``standalone_driver.gridlook`` (data-free)."""

import functools
import http.server
import pathlib
import threading
import urllib.request
from typing import Literal

import netCDF4 as nc
import numpy as np
import pytest
import zarr

from icon4py.model.common.io import cf_utils, writers
from icon4py.model.standalone_driver import gridlook


GRID_UUID = "6717c462-6f36-11f0-8dfd-c9f61e2d6a2e"
NUM_TIMES = 2
NUM_LEVELS = 2
NUM_GLOBAL_CELLS = 5
NUM_VERTICES = 4
#: Rank-block layout of the source store: two blocks of chunk 3 holding
#: 3 and 2 owned cells; -1 marks padding.
CELL_GLOBAL_INDEX = np.array([4, 0, 2, 1, 3, -1], dtype=np.int64)
#: Spherical vertex coordinates (radians) of the spherical grid-file variant.
SPHERICAL_VLON = np.array([0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi])
SPHERICAL_VLAT = np.array([0.0, 0.25 * np.pi, -0.25 * np.pi, 0.5 * np.pi])


def _cell_values(global_index: np.ndarray) -> np.ndarray:
    """Field values encoding (time, level, global cell) at each store position."""
    values = np.full((NUM_TIMES, NUM_LEVELS, global_index.shape[0]), np.nan, dtype=np.float32)
    owned = global_index >= 0
    for time_index in range(NUM_TIMES):
        for level in range(NUM_LEVELS):
            values[time_index, level, owned] = 100 * time_index + 10 * level + global_index[owned]
    return values


def _write_source_store(path: pathlib.Path, *, rank_block: bool) -> None:
    group = zarr.open_group(str(path), mode="w-", zarr_format=3)
    group.attrs.update(
        {"title": "test run", "institution": "EXCLAIM - ETH Zurich", "uuidOfHGrid": GRID_UUID}
    )

    times = group.create_array(
        writers.TIME, shape=(NUM_TIMES,), chunks=(1,), dtype="f8", dimension_names=[writers.TIME]
    )
    times[:] = np.arange(NUM_TIMES, dtype=np.float64) * 60.0
    times.attrs.update({"units": cf_utils.DEFAULT_TIME_UNIT, "calendar": cf_utils.DEFAULT_CALENDAR})
    levels = group.create_array(
        writers.MODEL_LEVEL,
        shape=(NUM_LEVELS,),
        dtype=np.int32,
        dimension_names=[writers.MODEL_LEVEL],
    )
    levels[:] = np.arange(NUM_LEVELS, dtype=np.int32)
    half_levels = group.create_array(
        writers.MODEL_HALF_LEVEL,
        shape=(NUM_LEVELS + 1,),
        dtype=np.int32,
        dimension_names=[writers.MODEL_HALF_LEVEL],
    )
    half_levels[:] = np.arange(NUM_LEVELS + 1, dtype=np.int32)
    heights = group.create_array(
        "height",
        shape=(NUM_LEVELS + 1,),
        dtype=np.float64,
        dimension_names=[writers.MODEL_HALF_LEVEL],
    )
    heights[:] = np.linspace(1000.0, 0.0, NUM_LEVELS + 1)

    if rank_block:
        cell_global_index = CELL_GLOBAL_INDEX
        chunk = 3
        index_array = group.create_array(
            f"{writers.GLOBAL_INDEX_PREFIX}_{writers.CELL}",
            shape=cell_global_index.shape,
            chunks=(chunk,),
            dtype=np.int64,
            fill_value=-1,
            dimension_names=[writers.CELL],
        )
        index_array[:] = cell_global_index
        edge_global_index = np.array([0, 2, -1, 1, 3, -1], dtype=np.int64)
        edge_index_array = group.create_array(
            f"{writers.GLOBAL_INDEX_PREFIX}_{writers.EDGE}",
            shape=edge_global_index.shape,
            chunks=(3,),
            dtype=np.int64,
            fill_value=-1,
            dimension_names=[writers.EDGE],
        )
        edge_index_array[:] = edge_global_index
        num_edges = edge_global_index.shape[0]
    else:
        cell_global_index = np.arange(NUM_GLOBAL_CELLS, dtype=np.int64)
        chunk = NUM_GLOBAL_CELLS
        num_edges = 4

    temperature = group.create_array(
        "temperature",
        shape=(NUM_TIMES, NUM_LEVELS, cell_global_index.shape[0]),
        chunks=(1, NUM_LEVELS, chunk),
        dtype=np.float32,
        fill_value=float("nan"),
        dimension_names=[writers.TIME, writers.MODEL_LEVEL, writers.CELL],
    )
    temperature[:] = _cell_values(cell_global_index)
    temperature.attrs.update({"units": "K", "standard_name": "air_temperature"})

    upward_air_velocity = group.create_array(
        "upward_air_velocity",
        shape=(NUM_TIMES, NUM_LEVELS + 1, cell_global_index.shape[0]),
        chunks=(1, NUM_LEVELS + 1, chunk),
        dtype=np.float32,
        fill_value=float("nan"),
        dimension_names=[writers.TIME, writers.MODEL_HALF_LEVEL, writers.CELL],
    )
    w_values = np.full(upward_air_velocity.shape, np.nan, dtype=np.float32)
    owned = cell_global_index >= 0
    w_values[:, :, owned] = cell_global_index[owned].astype(np.float32)
    upward_air_velocity[:] = w_values

    normal_velocity = group.create_array(
        "normal_velocity",
        shape=(NUM_TIMES, NUM_LEVELS, num_edges),
        chunks=(1, NUM_LEVELS, num_edges),
        dtype=np.float32,
        dimension_names=[writers.TIME, writers.MODEL_LEVEL, writers.EDGE],
    )
    normal_velocity[:] = np.zeros((NUM_TIMES, NUM_LEVELS, num_edges), dtype=np.float32)


def _write_grid_file(
    path: pathlib.Path,
    *,
    num_cells: int = NUM_GLOBAL_CELLS,
    grid_uuid: str = GRID_UUID,
    one_based: bool = True,
    spherical: bool = False,
) -> None:
    with nc.Dataset(path, "w") as dataset:
        dataset.createDimension("nv", 3)
        dataset.createDimension("cell", num_cells)
        dataset.createDimension("vertex", NUM_VERTICES)
        vertex_of_cell = dataset.createVariable(gridlook.VERTEX_OF_CELL, "i4", ("nv", "cell"))
        indices = np.stack([(np.arange(num_cells) + offset) % NUM_VERTICES for offset in range(3)])
        vertex_of_cell[:] = indices + (1 if one_based else 0)
        if spherical:
            for name, values in zip(
                gridlook.SPHERICAL_VERTEX_COORDINATE_VARIABLES,
                (SPHERICAL_VLON, SPHERICAL_VLAT),
                strict=True,
            ):
                coordinate = dataset.createVariable(name, "f8", ("vertex",))
                coordinate.units = "radian"
                coordinate[:] = values
        else:
            for offset, name in enumerate(gridlook.VERTEX_COORDINATE_VARIABLES):
                coordinate = dataset.createVariable(name, "f8", ("vertex",))
                coordinate[:] = np.linspace(0.0, 1.0, NUM_VERTICES) + offset
        dataset.uuidOfHGrid = grid_uuid


def _write_source_netcdf(path: pathlib.Path, *, numeric_attrs: bool = False) -> None:
    """netCDF twin of the serial source store (netCDF output is always in global order).

    With ``numeric_attrs`` the temperature variable carries non-string attributes such
    as a foreign netCDF source may attach; ``netCDF4`` returns those as numpy scalars
    and arrays, which zarr cannot serialize unless the exporter coerces them.
    """
    with nc.Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.setncatts(
            {"title": "test run", "institution": "EXCLAIM - ETH Zurich", "uuidOfHGrid": GRID_UUID}
        )
        dataset.createDimension(writers.TIME, None)
        dataset.createDimension(writers.MODEL_LEVEL, NUM_LEVELS)
        dataset.createDimension(writers.MODEL_HALF_LEVEL, NUM_LEVELS + 1)
        dataset.createDimension(writers.CELL, NUM_GLOBAL_CELLS)
        dataset.createDimension(writers.EDGE, 4)

        times = dataset.createVariable(writers.TIME, "f8", (writers.TIME,))
        times.setncatts(
            {"units": cf_utils.DEFAULT_TIME_UNIT, "calendar": cf_utils.DEFAULT_CALENDAR}
        )
        times[:] = np.arange(NUM_TIMES, dtype=np.float64) * 60.0
        levels = dataset.createVariable(writers.MODEL_LEVEL, "i4", (writers.MODEL_LEVEL,))
        levels[:] = np.arange(NUM_LEVELS, dtype=np.int32)
        half_levels = dataset.createVariable(
            writers.MODEL_HALF_LEVEL, "i4", (writers.MODEL_HALF_LEVEL,)
        )
        half_levels[:] = np.arange(NUM_LEVELS + 1, dtype=np.int32)
        heights = dataset.createVariable("height", "f8", (writers.MODEL_HALF_LEVEL,))
        heights[:] = np.linspace(1000.0, 0.0, NUM_LEVELS + 1)

        temperature = dataset.createVariable(
            "temperature", "f4", (writers.TIME, writers.MODEL_LEVEL, writers.CELL)
        )
        temperature[:] = _expected_global_values()
        temperature.setncatts({"units": "K", "standard_name": "air_temperature"})
        if numeric_attrs:
            # non-CF numeric attributes (netCDF4 returns numpy scalars/arrays for these)
            temperature.setncattr("tuning_parameter", np.float32(0.5))
            temperature.setncattr("ensemble_member", np.int32(3))
            temperature.setncattr("sampled_levels", np.array([0, 1], dtype=np.int64))
        upward_air_velocity = dataset.createVariable(
            "upward_air_velocity", "f4", (writers.TIME, writers.MODEL_HALF_LEVEL, writers.CELL)
        )
        upward_air_velocity[:] = np.broadcast_to(
            np.arange(NUM_GLOBAL_CELLS, dtype=np.float32),
            (NUM_TIMES, NUM_LEVELS + 1, NUM_GLOBAL_CELLS),
        )
        normal_velocity = dataset.createVariable(
            "normal_velocity", "f4", (writers.TIME, writers.MODEL_LEVEL, writers.EDGE)
        )
        normal_velocity[:] = np.zeros((NUM_TIMES, NUM_LEVELS, 4), dtype=np.float32)


@pytest.fixture
def source_store(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "icon4py_output_0000.zarr"
    _write_source_store(path, rank_block=True)
    return path


@pytest.fixture
def grid_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "grid.nc"
    _write_grid_file(path)
    return path


def _export(
    source: pathlib.Path, grid_file: pathlib.Path, output: pathlib.Path
) -> tuple[list[str], list[str]]:
    return gridlook.export_store(source=source, grid_file=grid_file, output=output)


def _open_array(store: pathlib.Path, name: str, *, mode: Literal["r", "r+"] = "r") -> zarr.Array:
    array = zarr.open_group(str(store), mode=mode)[name]
    assert isinstance(array, zarr.Array)
    return array


def _expected_global_values() -> np.ndarray:
    return _cell_values(np.arange(NUM_GLOBAL_CELLS, dtype=np.int64))


def test_export_rank_block_reorders_to_global_order(
    source_store: pathlib.Path, grid_file: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    output = tmp_path / "viz.zarr"
    exported, skipped = _export(source_store, grid_file, output)

    assert exported == ["temperature", "upward_air_velocity"]
    assert skipped == ["normal_velocity"]
    temperature = _open_array(output, "temperature")
    assert temperature.shape == (NUM_TIMES, NUM_LEVELS, NUM_GLOBAL_CELLS)
    values = np.asarray(temperature[:])
    assert not np.isnan(values).any()
    np.testing.assert_array_equal(values, _expected_global_values())

    upward_air_velocity = _open_array(output, "upward_air_velocity")
    assert upward_air_velocity.shape == (NUM_TIMES, NUM_LEVELS + 1, NUM_GLOBAL_CELLS)
    np.testing.assert_array_equal(
        np.asarray(upward_air_velocity[:])[0, 0], np.arange(NUM_GLOBAL_CELLS, dtype=np.float32)
    )


def test_export_serial_store_copies_identically(
    grid_file: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    source = tmp_path / "serial.zarr"
    _write_source_store(source, rank_block=False)
    output = tmp_path / "viz.zarr"
    _export(source, grid_file, output)

    temperature = _open_array(output, "temperature")
    np.testing.assert_array_equal(temperature[:], _expected_global_values())


def test_export_netcdf_source_copies_identically(
    grid_file: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    source = tmp_path / "icon4py_output_0000.nc"
    _write_source_netcdf(source)
    output = tmp_path / "viz.zarr"
    exported, skipped = _export(source, grid_file, output)

    assert exported == ["temperature", "upward_air_velocity"]
    assert skipped == ["normal_velocity"]
    temperature = _open_array(output, "temperature")
    assert temperature.chunks == (1, 1, NUM_GLOBAL_CELLS)
    assert temperature.attrs["units"] == "K"
    np.testing.assert_array_equal(temperature[:], _expected_global_values())
    group = zarr.open_group(str(output), mode="r")
    assert group.attrs["uuidOfHGrid"] == GRID_UUID
    times = _open_array(output, writers.TIME)
    assert times.attrs["calendar"] == cf_utils.DEFAULT_CALENDAR
    np.testing.assert_array_equal(times[:], np.arange(NUM_TIMES, dtype=np.float64) * 60.0)


def test_export_netcdf_source_coerces_numeric_attributes(
    grid_file: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    source = tmp_path / "foreign.nc"
    _write_source_netcdf(source, numeric_attrs=True)
    output = tmp_path / "viz.zarr"
    _export(source, grid_file, output)

    # the numeric attributes survive as JSON-safe values, and the store re-reads cleanly
    temperature = _open_array(output, "temperature")
    assert temperature.attrs["tuning_parameter"] == 0.5
    assert temperature.attrs["ensemble_member"] == 3
    assert temperature.attrs["sampled_levels"] == [0, 1]
    reopened = zarr.open_group(str(output), mode="r")["temperature"]
    assert isinstance(reopened, zarr.Array)
    np.testing.assert_array_equal(np.asarray(reopened[:]), _expected_global_values())


def test_export_writes_gridlook_geometry_and_consolidates(
    source_store: pathlib.Path, grid_file: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    output = tmp_path / "viz.zarr"
    _export(source_store, grid_file, output)

    group = zarr.open_group(str(output), mode="r")
    assert group.metadata.consolidated_metadata is not None
    assert "normal_velocity" not in group
    assert "height" not in group
    assert f"{writers.GLOBAL_INDEX_PREFIX}_{writers.CELL}" not in group
    assert f"{writers.GLOBAL_INDEX_PREFIX}_{writers.EDGE}" not in group

    vertex_of_cell = _open_array(output, gridlook.VERTEX_OF_CELL)
    assert vertex_of_cell.dtype == np.int32
    assert vertex_of_cell.shape == (3, NUM_GLOBAL_CELLS)
    assert np.asarray(vertex_of_cell[:]).min() == 1
    # no dimension names: gridlook hides such arrays from its variable selector
    assert gridlook._dimension_names(vertex_of_cell) is None
    for name in gridlook.VERTEX_COORDINATE_VARIABLES:
        coordinate = _open_array(output, name)
        assert coordinate.shape == (NUM_VERTICES,)
        assert gridlook._dimension_names(coordinate) is None


def test_export_preserves_attributes_coordinates_and_chunks(
    source_store: pathlib.Path, grid_file: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    output = tmp_path / "viz.zarr"
    _export(source_store, grid_file, output)

    group = zarr.open_group(str(output), mode="r")
    assert group.attrs["title"] == "test run"
    assert group.attrs["uuidOfHGrid"] == GRID_UUID
    temperature = _open_array(output, "temperature")
    assert temperature.attrs["units"] == "K"
    assert temperature.chunks == (1, 1, NUM_GLOBAL_CELLS)
    times = _open_array(output, writers.TIME)
    assert times.attrs["units"] == cf_utils.DEFAULT_TIME_UNIT
    np.testing.assert_array_equal(times[:], np.arange(NUM_TIMES, dtype=np.float64) * 60.0)
    np.testing.assert_array_equal(
        _open_array(output, writers.MODEL_LEVEL)[:], np.arange(NUM_LEVELS, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        _open_array(output, writers.MODEL_HALF_LEVEL)[:],
        np.arange(NUM_LEVELS + 1, dtype=np.int32),
    )


def test_export_refuses_existing_output(
    source_store: pathlib.Path, grid_file: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    output = tmp_path / "viz.zarr"
    output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        _export(source_store, grid_file, output)


def test_export_rejects_grid_size_mismatch(
    source_store: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    grid_path = tmp_path / "grid.nc"
    _write_grid_file(grid_path, num_cells=NUM_GLOBAL_CELLS + 2)
    with pytest.raises(ValueError, match="global cells"):
        _export(source_store, grid_path, tmp_path / "viz.zarr")


def test_export_serial_store_rejects_grid_size_mismatch(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "serial.zarr"
    _write_source_store(source, rank_block=False)
    grid_path = tmp_path / "grid.nc"
    _write_grid_file(grid_path, num_cells=NUM_GLOBAL_CELLS + 2)
    with pytest.raises(ValueError, match=f"has {NUM_GLOBAL_CELLS + 2} cells"):
        _export(source, grid_path, tmp_path / "viz.zarr")


def test_export_failure_leaves_no_partial_store(
    source_store: pathlib.Path,
    grid_file: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "viz.zarr"

    def _fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Interrupted copy.")

    monkeypatch.setattr(gridlook, "_copy_cell_variable", _fail)
    with pytest.raises(RuntimeError, match="Interrupted copy"):
        _export(source_store, grid_file, output)
    assert not output.exists()


def test_export_rejects_grid_uuid_mismatch(
    source_store: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    grid_path = tmp_path / "grid.nc"
    _write_grid_file(grid_path, grid_uuid="00000000-0000-0000-0000-000000000000")
    with pytest.raises(ValueError, match="uuidOfHGrid"):
        _export(source_store, grid_path, tmp_path / "viz.zarr")


def test_read_grid_geometry_derives_cartesian_from_spherical(tmp_path: pathlib.Path) -> None:
    grid_path = tmp_path / "grid.nc"
    _write_grid_file(grid_path, spherical=True)
    geometry = gridlook.read_grid_geometry(grid_path)

    # vertices at (lon, lat): (0, 0), (pi/2, pi/4), (pi, -pi/4), (-pi/2, pi/2)
    x_name, y_name, z_name = gridlook.VERTEX_COORDINATE_VARIABLES
    s = np.sqrt(0.5)
    np.testing.assert_allclose(geometry.vertex_coordinates[x_name], [1.0, 0.0, -s, 0.0], atol=1e-15)
    np.testing.assert_allclose(geometry.vertex_coordinates[y_name], [0.0, s, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(geometry.vertex_coordinates[z_name], [0.0, s, -s, 1.0], atol=1e-15)


def test_read_grid_geometry_rejects_zero_based_indices(tmp_path: pathlib.Path) -> None:
    grid_path = tmp_path / "grid.nc"
    _write_grid_file(grid_path, one_based=False)
    with pytest.raises(ValueError, match="1-based"):
        gridlook.read_grid_geometry(grid_path)


def test_export_rejects_corrupt_global_index(
    grid_file: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    source = tmp_path / "corrupt.zarr"
    _write_source_store(source, rank_block=True)
    index_name = f"{writers.GLOBAL_INDEX_PREFIX}_{writers.CELL}"
    corrupt = _open_array(source, index_name, mode="r+")
    corrupt[0] = 0  # duplicates the global index 0
    with pytest.raises(ValueError, match="exactly once"):
        _export(source, grid_file, tmp_path / "viz.zarr")


def test_serve_handler_sends_cors_header(tmp_path: pathlib.Path) -> None:
    (tmp_path / "payload.json").write_text("{}")
    handler = functools.partial(gridlook._CORSHTTPRequestHandler, directory=str(tmp_path))
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/payload.json") as response:
                assert response.headers["Access-Control-Allow-Origin"] == "*"
                assert response.read() == b"{}"
        finally:
            server.shutdown()
            thread.join()
