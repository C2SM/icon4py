# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import pytest
from xarray import Dataset

from icon4py.model.common.test_utils.datatest_utils import (
    GRIDS_PATH,
    R02B04_GLOBAL,
    REGIONAL_EXPERIMENT,
)
from icon4py.model.common.test_utils.grid_utils import GLOBAL_GRIDFILE, REGIONAL_GRIDFILE
from icon4py.model.driver.io.ugrid import (
    FILL_VALUE,
    IconUGridPatch,
    dump_ugrid_file,
    extract_horizontal_coordinates,
    load_data_file,
)


def grid_files():
    files = [(R02B04_GLOBAL, GLOBAL_GRIDFILE), (REGIONAL_EXPERIMENT, REGIONAL_GRIDFILE)]

    for ff in files:
        yield GRIDS_PATH.joinpath(ff[0]).joinpath(ff[1])


@pytest.mark.parametrize("file", grid_files())
def test_convert_to_ugrid(file):
    with load_data_file(file) as ds:
        patch = IconUGridPatch()
        uxds = patch(ds, validate=True)
        assert uxds.attrs["title"] == "ICON grid description"
        assert uxds.attrs["ellipsoid_name"] == "Sphere"
        assert uxds["mesh"].attrs["cf_role"] == "mesh_topology"
        assert uxds["mesh"].attrs["topology_dimension"] == 2
        assert uxds["mesh"].attrs["face_dimension"] == "cell"
        assert uxds["mesh"].attrs["edge_dimension"] == "edge"
        assert uxds["mesh"].attrs["node_dimension"] == "vertex"
        assert uxds["mesh"].attrs["node_coordinates"] == "vlon vlat"
        assert uxds["mesh"].attrs["face_node_connectivity"] == "vertex_of_cell"


@pytest.mark.parametrize("file", grid_files())
def test_dump_ugrid_file(file, test_path):
    with load_data_file(file) as ds:
        patch = IconUGridPatch()
        uxds = patch(ds)
        output_dir = test_path.joinpath("output")
        output_dir.mkdir(0o755, exist_ok=True)
        dump_ugrid_file(uxds, file, output_path=output_dir)
        fname = output_dir.iterdir().__next__().name
        assert fname == file.stem + "_ugrid.nc"


@pytest.mark.parametrize("file", grid_files())
def test_icon_ugrid_patch_index_transformation(file):
    with load_data_file(file) as ds:
        patch = IconUGridPatch()
        uxds = patch(ds)
        for name in patch.index_lists:
            assert_start_index(uxds, name)
            assert uxds[name].dtype == "int32"


@pytest.mark.parametrize("file", grid_files())
def test_icon_ugrid_patch_transposed_index_lists(file):
    with load_data_file(file) as ds:
        patch = IconUGridPatch()
        uxds = patch(ds)
        horizontal_dims = ("cell", "edge", "vertex")
        horizontal_sizes = list(uxds.sizes[k] for k in horizontal_dims)
        for name in patch.connectivities:
            assert len(uxds[name].shape) == 2
            assert uxds[name].shape[0] > uxds[name].shape[1]
            assert uxds[name].dims[0] in horizontal_dims
            assert uxds[name].shape[0] in horizontal_sizes


@pytest.mark.parametrize("file", grid_files())
def test_icon_ugrid_patch_fill_value(file):
    with load_data_file(file) as ds:
        patch = IconUGridPatch()
        uxds = patch(ds)
        patch._set_fill_value(uxds)
        for name in patch.connectivities:
            assert uxds[name].attrs["_FillValue"] == FILL_VALUE


def assert_start_index(uxds: Dataset, name: str):
    assert uxds[name].attrs["start_index"] == 0
    assert np.min(np.where(uxds[name].data > FILL_VALUE)) == 0


@pytest.mark.parametrize("file", grid_files())
def test_extract_horizontal_coordinates(file):
    with load_data_file(file) as ds:
        dim_sizes = ds.sizes
        coords = extract_horizontal_coordinates(ds)
        # TODO (halungge) fix:
        #  - 'long_name', 'standard_name' of attributes fx cell center latitudes
        # - 'units' of lat, lon are conventionally in degrees not in radians as ICON provides themconvert to degrees_north, degrees_east..
        for k in ("cell", "edge", "vertex"):
            assert k in coords
            assert coords[k][0].shape[0] == dim_sizes[k]
