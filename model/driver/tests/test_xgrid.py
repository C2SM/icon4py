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

from pathlib import Path

import pytest

from icon4py.model.common.test_utils.datatest_utils import (
    GRIDS_PATH,
    R02B04_GLOBAL,
    REGIONAL_EXPERIMENT,
)
from icon4py.model.common.test_utils.grid_utils import GLOBAL_GRIDFILE, REGIONAL_GRIDFILE
from icon4py.model.driver.io.xgrid import (
    IconUGridPatch,
    dump_ugrid_file,
    extract_horizontal_coordinates,
    load_icon_grid_data,
)


def grid_files():
    files = [(R02B04_GLOBAL,GLOBAL_GRIDFILE) , (REGIONAL_EXPERIMENT, REGIONAL_GRIDFILE)]
    
    for ff in files:
        yield GRIDS_PATH.joinpath(ff[0]).joinpath(ff[1])

@pytest.mark.parametrize("file", grid_files())
def test_convert_to_ugrid(file):
    with load_icon_grid_data(file) as ds:
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
def test_dump_ugrid_file(file, tmpdir):
    with load_icon_grid_data(file) as ds:
        patch = IconUGridPatch()
        uxds = patch(ds)
        output_dir = Path(tmpdir).joinpath("output")
        #output_dir = Path(__file__).parent.joinpath("output")
        output_dir.mkdir(0o755, exist_ok=True)
        dump_ugrid_file(uxds, file, output_path=output_dir)
        assert tmpdir.compare(file.stem +'_ugrid.nc', path='output/')
        
        
@pytest.mark.parametrize("file", grid_files())
def test_extract_horizontal_coordinates(file):
    with load_icon_grid_data(file) as ds:
        dims = ds.dims    
        coords = extract_horizontal_coordinates(ds)
        # TODO (halungge) fix data  
        #  - 'long_name', 'standard_name' of attributes fx cell center latitudes
        # - 'units' convert to degrees_north, degrees_east.. 
        # - get the bounds
       for k in ("cell", "edge", "vertex"):
            assert k in coords
            assert coords[k][0].shape[0] == dims[k]
            
