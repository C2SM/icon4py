# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Doubly periodic (plane torus) triangular ICON grids for idealized test cases.

A thin adapter over the 'icon-grid-generator' package, kept so that the fork pin and the
argument convention live in one place.

The generator is parametrised by row and column counts rather than by domain extent. That is
what makes a convergence family well posed: doubling 'n_rows' and 'n_cols' while halving
'edge_length' bisects the mesh and leaves both extents bit-identical. MPI-M's GridGenerator
instead fits a requested extent to whole cells and adjusts the domain height while doing so, so
its grids of different resolution discretise slightly different domains.

'icon_periodicity' carries the shear of the y identification in the vertex numbering rather
than in the metric. Without it the stored coordinates span more than one period and only
reconstruct under a coupled two-lattice-vector minimum image, whereas ICON and icon4py apply a
per-axis one; see https://github.com/ofuhrer/icon-grid-generator/issues/1.
"""

import pathlib

import grid_generator


def generate_torus_grid(
    *,
    n_rows: int,
    n_cols: int,
    edge_length: float,
    out_file: pathlib.Path,
) -> pathlib.Path:
    """
    Write an ICON grid file for a doubly periodic mesh of equilateral triangles.

    The domain is 'n_cols * edge_length' wide and 'n_rows * edge_length * sqrt(3)/2' high, with
    its lower left corner at (0, 0).

    Args:
        n_rows: number of rows, even and at least 4
        n_cols: number of columns, at least 3
        edge_length: triangle side length in meters
        out_file: path of the NetCDF file to write

    Returns:
        The path that was written.
    """
    grid = grid_generator.generate_grid(
        grid_generator.TorusGridSpec(
            nx=n_cols,
            ny=n_rows,
            edge_length=edge_length,
            icon_periodicity=True,
        )
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    grid.to_netcdf(out_file)
    return out_file
