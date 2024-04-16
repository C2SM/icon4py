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
import contextlib
import logging
from pathlib import Path
from typing import Union

import uxarray
import xarray as xa
from gt4py.next import Dimension, DimensionKind

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, VertexDim
from icon4py.model.common.grid.grid_manager import GridFile
from icon4py.model.driver.io.exceptions import ValidationError


FILL_VALUE = GridFile.INVALID_INDEX

log = logging.getLogger(__name__)

MESH = "mesh"


def extract_horizontal_coordinates(ds: xa.Dataset):
    """
    Extract the coordinates from the ICON grid file.
    TODO (@halungge) does it  work for decomposed grids?
    """
    return dict(
        cell=(ds["clat"], ds["clon"]),
        vertex=(ds["vlat"], ds["vlon"]),
        edge=(ds["elat"], ds["elon"]),
    )


dimension_mapping = {
    CellDim: "cell",
    KDim: "level",
    EdgeDim:"edge",
    VertexDim:"vertex"
}

coordinates_mapping={
    CellDim: "clon clat",
    VertexDim: "vlon vlat",
    EdgeDim: "elon elat",
}

location_mapping = {
    CellDim:"face",
    VertexDim:"node",
    EdgeDim:"edge",
}

def ugrid_attributes(dim:Dimension) -> dict:
    if dim.kind == DimensionKind.HORIZONTAL:
        return dict(location=location_mapping[dim], coordinates=coordinates_mapping[dim], mesh=MESH)
    else:
        return {}
def extract_bounds(ds: xa.Dataset):
    """
    Extract the bounds from the ICON grid file.
    TODO (@halungge) does it  work for decomposed grids?
    TODO (@halungge) these are not present in the mch grid file 
    """
    return dict(
        cell=(ds["clat_vertices"], ds["clon_vertices"]),
        vertex=(ds["vlat_vertices"], ds["vlon_vertices"]),
        edge=(ds["elat_vertices"], ds["elon_vertices"]),
    )

class IconUGridPatch:
    """
    Patch an ICON grid file with necessary information for UGRID.

    TODO: (magdalena) should all the unnecessary data fields be removed.
    """

    def __init__(self):        
        self.connectivities = (
            "edge_of_cell",  # E2C connectivity
            "vertex_of_cell",  # V2C connectivity
            "adjacent_cell_of_edge",  # C2E connectivity
            "edge_vertices",  # E2V connectivity
            "cells_of_vertex",  # C2V connectivity
            "edges_of_vertex",  # E2V connectivity
            "vertices_of_vertex",  # V2E2V connectivity
            "neighbor_cell_index",  # C2E2C connectivity

        )
        self.domain_bounds = (
            "start_idx_c",  # start and end indices for refin_ctl_levels
            "end_idx_c",
            "start_idx_e",
            "end_idx_e",
            "start_idx_v",
            "end_idx_v",
        )
        self.index_lists = self.connectivities + self.domain_bounds
        # do not exist on local grid.nc of mch_ch_r04b09_dsl what do they contain?
        # "edge_index",  
        # "vertex_index",
        # "cell_index",
        # only for nested grids, ignored for now
        # "child_cell_index",
        # "child_edge_index",
        

    def _add_mesh_var(self, ds: xa.Dataset):
        ds["mesh"] = xa.DataArray(
            -1,  # A dummy value for creating the DataArray with the actual attributes
            attrs=dict(
                cf_role="mesh_topology",
                topology_dimension=2,
                node_coordinates="vlon vlat",
                face_node_connectivity="vertex_of_cell",
                face_dimension="cell",
                edge_node_connectivity="edge_vertices",
                edge_dimension="edge",
                edge_coordinates="elon elat",
                face_coordinates="clon clat",
                face_edge_connectivity="edge_of_cell",
                face_face_connectivity="neighbor_cell_index",
                edge_face_connectivity="adjacent_cell_of_edge",
                node_dimension="vertex",
                # boundary_node_connectivity="",
            ),
        )



    def _patch_start_index(self, ds: xa.Dataset, with_zero_start_index:bool = False):
        for var in self.index_lists:
            if var in ds:
                if with_zero_start_index:
                    # work around for uxarray not supporting [start_index] = 1 properly
                    ds[var].data = xa.where(ds[var].data > 0, ds[var].data - 1, FILL_VALUE)
                    ds[var].attrs["start_index"] = 0
                else:
                    ds[var].attrs["start_index"] = 1
    def _set_fill_value(self, ds: xa.Dataset):
        for var in self.connectivities:
            if var in ds:
                ds[var].attrs["_FillValue"] = FILL_VALUE            
    
    def _transpose_index_lists(self, ds: xa.Dataset):
        """ Unify the dimension order of fields in ICON grid file.
        
        The ICON grid file contains some fields of order (sparse_dimension, horizontal_dimension) and others the other way around. We transpose them to have all the same ordering.
        """
        for name in self.connectivities:
            shp = ds[name].shape
            if len(shp) == 2 and (shp[0] < shp[1]):
                ds[name] = xa.DataArray(
                    data=ds[name].data.T,
                    dims=ds[name].dims[::-1],
                    coords=ds[name].coords,
                    attrs=ds[name].attrs,
                )
        
    def _validate(self, ds: xa.Dataset):
        grid = uxarray.open_grid(ds)
        try: 
            grid.validate()
        except RuntimeError as error:
            log.error(f"Validation of the ugrid failed with {error}>")
            raise ValidationError("Validation of the ugrid failed") from error


    def __call__(self, ds: xa.Dataset, validate: bool = False):
        self._patch_start_index(ds, with_zero_start_index=True)
        self._set_fill_value(ds)
        self._transpose_index_lists(ds)
        self._add_mesh_var(ds)
        if validate:
            self._validate(ds)
        return ds


# TODO (magdalena) encapsulate this thing somehow together with the opening of the file
# like this it could be called on an unpatched dataset
def dump_ugrid_file(ds: xa.Dataset, original_filename: Path, output_path: Path):
    stem = original_filename.stem
    filename = output_path.joinpath(stem + "_ugrid.nc")
    ds.to_netcdf(
        filename, format="NETCDF4", engine="netcdf4"
    )


class IconUGridWriter:
    def __init__(self, original_filename: Union[Path, str], output_path: Union[Path, str]):
        self.original_filename = original_filename
        self.output_path = output_path

    def __call__(self, validate: bool = False):
        patch = IconUGridPatch()
        with load_data_file(self.original_filename) as ds:
            patched_ds = patch(ds, validate)
            dump_ugrid_file(patched_ds, self.original_filename, self.output_path)


@contextlib.contextmanager
def load_data_file(filename: Union[Path|str]) -> xa.Dataset:
    ds = xa.open_dataset(filename)
    try:
        yield ds
    finally:
        ds.close()
