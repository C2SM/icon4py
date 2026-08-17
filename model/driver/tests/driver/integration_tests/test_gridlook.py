# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the gridlook export against real ICON grid files.

The unit tests build synthetic grid files and therefore can only check the
exporter against our own idea of a grid file; these tests pin the actual
external contract: the icosahedral grid files carry spherical vertex
coordinates only (the Cartesian ones are derived), the torus grid files carry
Cartesian vertex coordinates.
"""

import pathlib

import netCDF4 as nc
import numpy as np
import pytest
import zarr

from icon4py.model.common.io import writers
from icon4py.model.driver import gridlook
from icon4py.model.testing import definitions, grid_utils, test_utils


@pytest.mark.datatest
@pytest.mark.parametrize(
    ("grid_description", "on_unit_sphere"),
    [
        (definitions.Grids.R02B04_GLOBAL, True),
        (definitions.Grids.TORUS_100X116_1000M, False),
    ],
    ids=lambda value: value.name if isinstance(value, definitions.GridDescription) else "",
)
def test_read_grid_geometry_from_real_grid_file(
    grid_description: definitions.GridDescription, on_unit_sphere: bool
) -> None:
    grid_file = grid_utils._download_grid_file(grid_description)
    geometry = gridlook.read_grid_geometry(grid_file)

    with nc.Dataset(grid_file, "r") as dataset:
        num_cells = dataset.dimensions[writers.CELL].size
        num_vertices = dataset.dimensions[writers.VERTEX].size

    assert geometry.num_cells == num_cells
    assert geometry.vertex_of_cell.shape == (3, num_cells)
    assert geometry.vertex_of_cell.min() >= 1
    assert geometry.vertex_of_cell.max() <= num_vertices
    for values in geometry.vertex_coordinates.values():
        assert values.shape == (num_vertices,)
        assert np.isfinite(values).all()
    if on_unit_sphere:
        x, y, z = (
            geometry.vertex_coordinates[name] for name in gridlook.VERTEX_COORDINATE_VARIABLES
        )
        test_utils.assert_dallclose(np.sqrt(x**2 + y**2 + z**2), 1.0, rtol=1e-6)


@pytest.mark.datatest
def test_export_against_real_grid_file(tmp_path: pathlib.Path) -> None:
    grid_file = grid_utils._download_grid_file(definitions.Grids.R02B04_GLOBAL)
    with nc.Dataset(grid_file, "r") as dataset:
        num_cells = dataset.dimensions[writers.CELL].size
        grid_uuid = dataset.uuidOfHGrid

    source = tmp_path / "source.zarr"
    group = zarr.open_group(str(source), mode="w-", zarr_format=3)
    group.attrs.update({"uuidOfHGrid": grid_uuid})
    times = group.create_array(
        writers.TIME, shape=(1,), chunks=(1,), dtype="f8", dimension_names=[writers.TIME]
    )
    times[:] = np.zeros(1, dtype=np.float64)
    temperature = group.create_array(
        "temperature",
        shape=(1, 1, num_cells),
        chunks=(1, 1, num_cells),
        dtype=np.float32,
        fill_value=float("nan"),
        dimension_names=[writers.TIME, writers.MODEL_LEVEL, writers.CELL],
    )
    temperature[:] = np.arange(num_cells, dtype=np.float32).reshape(1, 1, num_cells)

    output = tmp_path / "viz.zarr"
    exported, skipped = gridlook.export_store(source=source, grid_file=grid_file, output=output)

    assert exported == ["temperature"]
    assert skipped == []
    exported_group = zarr.open_group(str(output), mode="r")
    exported_temperature = exported_group["temperature"]
    assert isinstance(exported_temperature, zarr.Array)
    np.testing.assert_array_equal(
        np.asarray(exported_temperature[:]).ravel(), np.arange(num_cells, dtype=np.float32)
    )
    vertex_of_cell = exported_group[gridlook.VERTEX_OF_CELL]
    assert isinstance(vertex_of_cell, zarr.Array)
    assert vertex_of_cell.shape == (3, num_cells)
