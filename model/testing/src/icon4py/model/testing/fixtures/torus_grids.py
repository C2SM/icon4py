# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Generated doubly periodic (plane torus) grids for idealized test cases."""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Protocol

import grid_generator
import pytest


class TorusGridFactory(Protocol):
    def __call__(self, *, n_rows: int, n_cols: int, edge_length: float) -> pathlib.Path: ...


def _write_torus_grid(
    *, n_rows: int, n_cols: int, edge_length: float, out_file: pathlib.Path
) -> pathlib.Path:
    # 'rectangular' wraps x and y independently, which is what icon4py does on a torus. It is
    # the default since 0.8.0 but is passed explicitly because the alternative, 'skew', stores
    # the coordinates on the coupled fundamental domain, where crossing y also shifts x, and
    # those only reconstruct under a coupled two lattice vector minimum image.
    grid = grid_generator.generate_grid(
        grid_generator.TorusGridSpec(
            nx=n_cols,
            ny=n_rows,
            edge_length=edge_length,
            periodic_layout="rectangular",
        )
    )
    grid.to_netcdf(out_file)
    return out_file


@pytest.fixture(scope="session")
def generate_torus_grid(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., pathlib.Path]:
    """
    Write an ICON grid file for a doubly periodic mesh of equilateral triangles.

    The domain is 'n_cols * edge_length' wide and 'n_rows * edge_length * sqrt(3)/2' high, with
    its lower left corner at (0, 0). Doubling 'n_rows' and 'n_cols' while halving 'edge_length'
    bisects the mesh and leaves both extents bit-identical, which is what makes a family of
    these grids usable for a convergence study. 'n_rows' must be even and at least 4, 'n_cols'
    at least 3.

    The files are written once per session and shared between tests asking for the same
    parameters, since generating the finer members takes a few seconds each.
    """
    directory = tmp_path_factory.mktemp("torus_grids")
    generated: dict[tuple[int, int, float], pathlib.Path] = {}

    def generate(*, n_rows: int, n_cols: int, edge_length: float) -> pathlib.Path:
        key = (n_rows, n_cols, edge_length)
        if key not in generated:
            # the stem labels the per-grid plots of the convergence studies
            out_file = directory / f"torus_{n_rows}x{n_cols}_res{edge_length:g}m.nc"
            generated[key] = _write_torus_grid(
                n_rows=n_rows, n_cols=n_cols, edge_length=edge_length, out_file=out_file
            )
        return generated[key]

    return generate
