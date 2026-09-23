# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.vertical_diffusion import (
    _assemble_vertical_diffusion_matrix_on_cell_half_levels,
    _assemble_vertical_diffusion_matrix_on_cells,
    _assemble_vertical_diffusion_matrix_on_edges,
    _solve_implicit_vertical_diffusion_on_cell_half_levels,
    _solve_implicit_vertical_diffusion_on_cells,
    _solve_implicit_vertical_diffusion_on_edges,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


def diffusion_matrix_numpy(interface_coeff: np.ndarray, inv_air_mass: np.ndarray) -> np.ndarray:
    """
    Matrix of minus the flux divergence on a column of rows, built column by column from unit
    vectors. interface_coeff[:, j] couples rows j and j + 1; no flux crosses the column ends.
    """
    num_rows = inv_air_mass.shape[1]
    unit_vectors = np.broadcast_to(np.eye(num_rows), (inv_air_mass.shape[0], num_rows, num_rows))
    downward_flux = interface_coeff[:, :, np.newaxis] * (unit_vectors[:, :-1] - unit_vectors[:, 1:])
    downward_flux = np.pad(downward_flux, ((0, 0), (1, 1), (0, 0)))
    return inv_air_mass[:, :, np.newaxis] * (downward_flux[:, 1:] - downward_flux[:, :-1])


def tridiagonal_matrix_numpy(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    matrix = np.zeros((*b.shape, b.shape[1]))
    rows = np.arange(b.shape[1])
    matrix[:, rows, rows] = b
    matrix[:, rows[1:], rows[:-1]] = a[:, 1:]
    matrix[:, rows[:-1], rows[1:]] = c[:, :-1]
    return matrix


def matrix_diagonals_on_rows(
    matrix: np.ndarray, shape: tuple[int, int], rows: slice
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, c = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    b[:, rows] = np.diagonal(matrix, axis1=1, axis2=2)
    a[:, rows][:, 1:] = np.diagonal(matrix, offset=-1, axis1=1, axis2=2)
    c[:, rows][:, :-1] = np.diagonal(matrix, offset=1, axis1=1, axis2=2)
    return a, b, c


def implicit_diffusion_tendency_numpy(
    *,
    var: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    rhs: np.ndarray,
    tend: np.ndarray,
    dtime: float,
    rows: slice,
) -> np.ndarray:
    matrix = tridiagonal_matrix_numpy(a[:, rows], b[:, rows], c[:, rows])
    matrix += np.eye(matrix.shape[1]) / dtime
    new_var = np.linalg.solve(matrix, (var[:, rows] / dtime + rhs[:, rows])[..., np.newaxis])
    out = np.zeros_like(var)
    out[:, rows] = tend[:, rows] + (new_var[..., 0] - var[:, rows]) / dtime
    return out


def vertical_rows(domain: dict[gtx.Dimension, tuple[int, int]], dim: gtx.Dimension) -> slice:
    return slice(*domain[dim])


class TestAssembleVerticalDiffusionMatrixOnCells(stencil_tests.StencilTest):
    PROGRAM = _assemble_vertical_diffusion_matrix_on_cells
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        diffusivity: np.ndarray,
        inv_dz: np.ndarray,
        inv_air_mass: np.ndarray,
        prefactor: float,
        domain: dict,
        **kwargs: Any,
    ) -> dict:
        rows = vertical_rows(domain, dims.KDim)
        interfaces = slice(rows.start + 1, rows.stop)
        matrix = diffusion_matrix_numpy(
            prefactor * diffusivity[:, interfaces] * inv_dz[:, interfaces],
            inv_air_mass[:, rows],
        )
        return dict(out=matrix_diagonals_on_rows(matrix, inv_air_mass.shape, rows))

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        vertical_start, vertical_end = 1, grid.num_levels - 1
        return dict(
            diffusivity=data_alloc.random_field(dims.CellDim, dims.KHalfDim, low=0.0),
            inv_dz=data_alloc.random_field(dims.CellDim, dims.KHalfDim, low=0.1),
            inv_air_mass=data_alloc.random_field(dims.CellDim, dims.KDim, low=0.1),
            prefactor=wpfloat(2.0),
            minlvl=gtx.int32(vertical_start),
            maxlvl=gtx.int32(vertical_end - 1),
            out=tuple(data_alloc.zero_field(dims.CellDim, dims.KDim) for _ in range(3)),
            domain={
                dims.CellDim: (0, gtx.int32(grid.num_cells)),
                dims.KDim: (gtx.int32(vertical_start), gtx.int32(vertical_end)),
            },
        )


class TestAssembleVerticalDiffusionMatrixOnCellHalfLevels(stencil_tests.StencilTest):
    PROGRAM = _assemble_vertical_diffusion_matrix_on_cell_half_levels
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        diffusivity: np.ndarray,
        inv_dz: np.ndarray,
        inv_air_mass: np.ndarray,
        prefactor: float,
        domain: dict,
        **kwargs: Any,
    ) -> dict:
        rows = vertical_rows(domain, dims.KHalfDim)
        interfaces = slice(rows.start, rows.stop - 1)
        matrix = diffusion_matrix_numpy(
            prefactor * diffusivity[:, interfaces] * inv_dz[:, interfaces],
            inv_air_mass[:, rows],
        )
        return dict(out=matrix_diagonals_on_rows(matrix, inv_air_mass.shape, rows))

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        vertical_start, vertical_end = 1, grid.num_levels
        return dict(
            diffusivity=data_alloc.random_field(dims.CellDim, dims.KDim, low=0.0),
            inv_dz=data_alloc.random_field(dims.CellDim, dims.KDim, low=0.1),
            inv_air_mass=data_alloc.random_field(dims.CellDim, dims.KHalfDim, low=0.1),
            prefactor=wpfloat(2.0),
            minlvl=gtx.int32(vertical_start),
            maxlvl=gtx.int32(vertical_end - 1),
            out=tuple(data_alloc.zero_field(dims.CellDim, dims.KHalfDim) for _ in range(3)),
            domain={
                dims.CellDim: (0, gtx.int32(grid.num_cells)),
                dims.KHalfDim: (gtx.int32(vertical_start), gtx.int32(vertical_end)),
            },
        )


class TestAssembleVerticalDiffusionMatrixOnEdges(stencil_tests.StencilTest):
    PROGRAM = _assemble_vertical_diffusion_matrix_on_edges
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        diffusivity: np.ndarray,
        inv_dz: np.ndarray,
        inv_air_mass: np.ndarray,
        prefactor: float,
        domain: dict,
        **kwargs: Any,
    ) -> dict:
        rows = vertical_rows(domain, dims.KDim)
        interfaces = slice(rows.start + 1, rows.stop)
        matrix = diffusion_matrix_numpy(
            prefactor * diffusivity[:, interfaces] * inv_dz[:, interfaces],
            inv_air_mass[:, rows],
        )
        return dict(out=matrix_diagonals_on_rows(matrix, inv_air_mass.shape, rows))

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        vertical_start, vertical_end = 0, grid.num_levels
        return dict(
            diffusivity=data_alloc.random_field(dims.EdgeDim, dims.KHalfDim, low=0.0),
            inv_dz=data_alloc.random_field(dims.EdgeDim, dims.KHalfDim, low=0.1),
            inv_air_mass=data_alloc.random_field(dims.EdgeDim, dims.KDim, low=0.1),
            prefactor=wpfloat(1.0),
            minlvl=gtx.int32(vertical_start),
            maxlvl=gtx.int32(vertical_end - 1),
            out=tuple(data_alloc.zero_field(dims.EdgeDim, dims.KDim) for _ in range(3)),
            domain={
                dims.EdgeDim: (0, gtx.int32(grid.num_edges)),
                dims.KDim: (gtx.int32(vertical_start), gtx.int32(vertical_end)),
            },
        )


def solve_input_data(
    data_alloc: stencil_tests.DataAllocationWrapper,
    *,
    horizontal_dim: gtx.Dimension,
    horizontal_size: int,
    vertical_dim: gtx.Dimension,
    vertical_start: int,
    vertical_end: int,
) -> dict:
    return dict(
        var=data_alloc.random_field(horizontal_dim, vertical_dim),
        a=data_alloc.random_field(horizontal_dim, vertical_dim, low=-1.0, high=0.0),
        b=data_alloc.random_field(horizontal_dim, vertical_dim, low=2.0, high=3.0),
        c=data_alloc.random_field(horizontal_dim, vertical_dim, low=-1.0, high=0.0),
        rhs=data_alloc.random_field(horizontal_dim, vertical_dim),
        tend=data_alloc.random_field(horizontal_dim, vertical_dim),
        dtime=wpfloat(0.5),
        out=data_alloc.zero_field(horizontal_dim, vertical_dim),
        domain={
            horizontal_dim: (0, gtx.int32(horizontal_size)),
            vertical_dim: (gtx.int32(vertical_start), gtx.int32(vertical_end)),
        },
    )


class TestSolveImplicitVerticalDiffusionOnCells(stencil_tests.StencilTest):
    PROGRAM = _solve_implicit_vertical_diffusion_on_cells
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(grid: base.Grid, *, domain: dict, out: np.ndarray, **kwargs: Any) -> dict:
        return dict(
            out=implicit_diffusion_tendency_numpy(**kwargs, rows=vertical_rows(domain, dims.KDim))
        )

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        return solve_input_data(
            data_alloc,
            horizontal_dim=dims.CellDim,
            horizontal_size=grid.num_cells,
            vertical_dim=dims.KDim,
            vertical_start=1,
            vertical_end=grid.num_levels - 1,
        )


class TestSolveImplicitVerticalDiffusionOnCellHalfLevels(stencil_tests.StencilTest):
    PROGRAM = _solve_implicit_vertical_diffusion_on_cell_half_levels
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(grid: base.Grid, *, domain: dict, out: np.ndarray, **kwargs: Any) -> dict:
        return dict(
            out=implicit_diffusion_tendency_numpy(
                **kwargs, rows=vertical_rows(domain, dims.KHalfDim)
            )
        )

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        return solve_input_data(
            data_alloc,
            horizontal_dim=dims.CellDim,
            horizontal_size=grid.num_cells,
            vertical_dim=dims.KHalfDim,
            vertical_start=1,
            vertical_end=grid.num_levels,
        )


class TestSolveImplicitVerticalDiffusionOnEdges(stencil_tests.StencilTest):
    PROGRAM = _solve_implicit_vertical_diffusion_on_edges
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(grid: base.Grid, *, domain: dict, out: np.ndarray, **kwargs: Any) -> dict:
        return dict(
            out=implicit_diffusion_tendency_numpy(**kwargs, rows=vertical_rows(domain, dims.KDim))
        )

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        return solve_input_data(
            data_alloc,
            horizontal_dim=dims.EdgeDim,
            horizontal_size=grid.num_edges,
            vertical_dim=dims.KDim,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
