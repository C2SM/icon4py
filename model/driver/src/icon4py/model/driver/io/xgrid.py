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
from pathlib import Path

import uxarray
import xarray as xa
from gt4py.next import Dimension, DimensionKind

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, VertexDim


MESH = "mesh"


@contextlib.contextmanager
def load_data_file(filename: Path) -> xa.Dataset:
    ds = xa.open_dataset(filename)
    try:
        yield ds
    finally:
        ds.close()


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
    CellDim:"cell",
    KDim: "height",
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

def ugrid_attributes(dim:Dimension)->dict:
    if dim.kind == DimensionKind.HORIZONTAL:
        return dict(location=location_mapping[dim], coordinates=coordinates_mapping[dim], mesh=MESH)
    else:
        return {}
def extract_bounds(ds: xa.Dataset):
    """
    Extract the bounds from the ICON grid file.
    TODO (@halungge) does it  work for decomposed grids?
    """
    return dict(
        cell=(ds["clat_vertices"], ds["clon_vertices"]),
        vertex=(ds["vlat_vertices"], ds["vlon_vertices"]),
        edge=(ds["elat_vertices"], ds["elon_vertices"]),
    )

class IconUGridPatch:
    """
    Patch an ICON grid file with necessary information for UGRID.

    TODO: remove all the unused data.
    """

    def __init__(self):
        self.index_lists = (
            "edge_of_cell",  # E2C connectivity
            "vertex_of_cell",  # V2C connectivity
            "adjacent_cell_of_edge",  # C2E connectivity
            "edge_vertices",  # E2V connectivity
            "cells_of_vertex",  # C2V connectivity
            "edges_of_vertex",  # E2V connectivity
            "vertices_of_vertex",  # V2E2V connectivity
            "neighbor_cell_index",  # C2E2C connectivity
            "start_idx_c",  # start and end indices for refin_ctl_levels
            "end_idx_c",
            "start_idx_e",
            "end_idx_e",
            "start_idx_v",
            "end_idx_v",
            "edge_index",  # TODO (magdalena) do not exist on local grid?
            "vertex_index",
            "cell_index",
        )

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

    def _remap_index_lists(self, ds: xa.Dataset):
        for var in self.index_lists:
            if var in ds:
                ds[var].attrs["start_index"] = 1
                ds[var].attrs["_FillValue"] = -1

    def _validate(self, ds: xa.Dataset):
        grid = uxarray.open_grid(ds)
        assert grid.validate()

    def __call__(self, ds: xa.Dataset, validate: bool = False):
        self._remap_index_lists(ds)
        self._add_mesh_var(ds)
        if validate:
            self._validate(ds)
        return ds


# TODO (magdalena) encapsulate this thing somehow together witht the opening of the file
# like this it could be called on an unpatched dataset
def dump_ugrid_file(ds: xa.Dataset, original_filename: Path, output_path: Path):
    stem = original_filename.stem
    filename = output_path.joinpath(stem + "_ugrid.nc")
    ds.to_netcdf(
        filename,
    )
