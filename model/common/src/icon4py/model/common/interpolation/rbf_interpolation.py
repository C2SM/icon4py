# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import enum
import math
from types import ModuleType

import numpy as np
import scipy.linalg as sla

from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.grid import base as base_grid
from icon4py.model.common.utils import data_allocation as data_alloc


class RBFDimension(enum.Enum):
    CELL = "cell"
    EDGE = "edge"
    VERTEX = "vertex"
    GRADIENT = "midpoint_gradient"


RBF_STENCIL_SIZE: dict[RBFDimension, int] = {
    RBFDimension.CELL: 9,
    RBFDimension.EDGE: 4,
    RBFDimension.VERTEX: 6,
    RBFDimension.GRADIENT: 10,
}


class InterpolationKernel(enum.Enum):
    GAUSSIAN = 1
    INVERSE_MULTIQUADRATIC = 3


DEFAULT_RBF_KERNEL: dict[RBFDimension, InterpolationKernel] = {
    RBFDimension.CELL: InterpolationKernel.GAUSSIAN,
    RBFDimension.EDGE: InterpolationKernel.INVERSE_MULTIQUADRATIC,
    RBFDimension.VERTEX: InterpolationKernel.GAUSSIAN,
    RBFDimension.GRADIENT: InterpolationKernel.GAUSSIAN,
}


def compute_rbf_scale(mean_characteristic_length: ta.wpfloat, dim: RBFDimension):
    """Compute the default RBF scale factor. This assumes that the Gaussian
    kernel is used for vertices and cells, and that the inverse multiquadratic
    kernel is used for edges."""

    threshold = 2.5 if dim == RBFDimension.CELL else 2.0
    c1 = 0.4 if dim == RBFDimension.EDGE else 1.8
    if dim == RBFDimension.CELL:
        c2 = 3.75
        c3 = 0.9
    elif dim == RBFDimension.VERTEX:
        c2 = 3.0
        c3 = 0.96
    else:
        c2 = 2.0
        c3 = 0.325

    resol = mean_characteristic_length / 1000.0
    scale = 0.5 / (1.0 + c1 * math.log(threshold / resol) ** c2) if resol < threshold else 0.5
    return scale * (resol / 0.125) ** c3 if resol <= 0.125 else scale


def construct_rbf_matrix_offsets_tables_for_cells(
    grid: base_grid.BaseGrid,
) -> data_alloc.NDArray:
    """Compute the neighbor tables for the cell RBF matrix: rbf_vec_index_c"""
    offset = grid.connectivities[dims.C2E2C2EDim]
    assert offset.shape == (grid.num_cells, RBF_STENCIL_SIZE[RBFDimension.CELL])
    return offset


def construct_rbf_matrix_offsets_tables_for_edges(
    grid: base_grid.BaseGrid,
) -> data_alloc.NDArray:
    """Compute the neighbor tables for the edge RBF matrix: rbf_vec_index_e"""
    offset = grid.connectivities[dims.E2C2EDim]
    assert offset.shape == (grid.num_edges, RBF_STENCIL_SIZE[RBFDimension.EDGE])
    return offset


def construct_rbf_matrix_offsets_tables_for_vertices(
    grid: base_grid.BaseGrid,
) -> data_alloc.NDArray:
    """Compute the neighbor tables for the edge RBF matrix: rbf_vec_index_v"""
    offset = grid.connectivities[dims.V2EDim]
    assert offset.shape == (grid.num_vertices, RBF_STENCIL_SIZE[RBFDimension.VERTEX])
    return offset


def _dot_product(
    v1: data_alloc.NDArray, v2: data_alloc.NDArray, array_ns: ModuleType = np
) -> data_alloc.NDArray:
    # alias: array_ns.transpose(v2, axes=(0, 2, 1)) for 3d array
    v2_tilde = array_ns.moveaxis(v2, 1, -1)
    # use linalg.matmul (array API compatible)
    return array_ns.matmul(v1, v2_tilde)


# NOTE: this one computes the pairwise arc lengths between elements in v, the
# next version computes pairwise arc lengths between two different arrays
# TODO: Combine?
def _arc_length_pairwise(v: data_alloc.NDArray, array_ns: ModuleType = np) -> data_alloc.NDArray:
    # For pairs of points p1 and p2 compute:
    # arccos(dot(p1, p2) / (norm(p1) * norm(p2))) noqa: ERA001
    # Compute all pairs of dot products
    arc_lengths = _dot_product(v, v, array_ns=array_ns)
    # Use the dot product of the diagonals to get the norm of each point
    norms = array_ns.sqrt(array_ns.diagonal(arc_lengths, axis1=1, axis2=2))
    # Divide the dot products by the broadcasted norms
    # TODO: Check that these are broadcast correctly. Leaving them out has
    # almost no impact on result, since they're close to 1, but may affect
    # precision.
    arc_lengths = array_ns.divide(arc_lengths, norms[:, :, array_ns.newaxis])
    arc_lengths = array_ns.divide(arc_lengths, norms[:, array_ns.newaxis, :])
    # Ensure all points are within [-1.0, 1.0] (may be outside due to numerical
    # inaccuracies)
    arc_lengths = array_ns.clip(arc_lengths, -1.0, 1.0)
    # TODO: avoid intermediates?
    return array_ns.arccos(arc_lengths)


# TODO: This assumes v1 is 2d and v3 is 3d
# TODO: name?
# TODO: this is pretty much the same as above, except we don't get the squares
# of the norms directly from the first matmul
# TODO: this is used only in one place, it's probably not as generic as it looks
def _arc_length_2(
    v1: data_alloc.NDArray, v2: data_alloc.NDArray, array_ns: ModuleType = np
) -> data_alloc.NDArray:
    # For pairs of points p1 and p2 compute:
    # arccos(dot(p1, p2) / (norm(p1) * norm(p2))) noqa: ERA001
    # Compute all pairs of dot products
    arc_lengths = _dot_product(v1, v2, array_ns=array_ns)
    v1_norm = array_ns.linalg.norm(v1, axis=-1)
    v2_norm = array_ns.linalg.norm(v2, axis=-1)
    # Divide the dot products by the broadcasted norms
    arc_lengths = array_ns.divide(arc_lengths, v1_norm[:, :, array_ns.newaxis])
    arc_lengths = array_ns.divide(arc_lengths, v2_norm[:, array_ns.newaxis, :])
    # Ensure all points are within [-1.0, 1.0] (may be outside due to numerical
    # inaccuracies)
    arc_lengths = array_ns.clip(arc_lengths, -1.0, 1.0)
    return array_ns.squeeze(array_ns.arccos(arc_lengths), axis=1)


def _gaussian(
    lengths: data_alloc.NDArray, scale: ta.wpfloat, array_ns: ModuleType = np
) -> data_alloc.NDArray:
    val = lengths / scale
    return array_ns.exp(-1.0 * val * val)


def _inverse_multiquadratic(
    distance: data_alloc.NDArray,
    scale: ta.wpfloat,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    """

    Args:
        distance: radial distance
        scale: scaling parameter

    Returns:

    """
    val = distance / scale
    return 1.0 / array_ns.sqrt(1.0 + val * val)


def _kernel(
    kernel: InterpolationKernel,
    lengths: data_alloc.NDArray,
    scale: ta.wpfloat,
    array_ns: ModuleType = np,
):
    match kernel:
        case InterpolationKernel.GAUSSIAN:
            return _gaussian(lengths, scale, array_ns=array_ns)
        case InterpolationKernel.INVERSE_MULTIQUADRATIC:
            return _inverse_multiquadratic(lengths, scale, array_ns=array_ns)
        case _:
            raise ValueError(f"Unsupported kernel: {kernel}")


def _cartesian_coordinates_from_zonal_and_meridional_components(
    lat: data_alloc.NDArray,
    lon: data_alloc.NDArray,
    u: data_alloc.NDArray,
    v: data_alloc.NDArray,
    array_ns: ModuleType = np,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray, data_alloc.NDArray]:
    cos_lat = array_ns.cos(lat)
    sin_lat = array_ns.sin(lat)
    cos_lon = array_ns.cos(lon)
    sin_lon = array_ns.sin(lon)

    x = -u * sin_lon - v * sin_lat * cos_lon
    y = u * cos_lon - v * sin_lat * sin_lon
    z = cos_lat * v

    return x, y, z


def _compute_rbf_interpolation_matrix(
    element_center_lat,
    element_center_lon,
    element_center_x,
    element_center_y,
    element_center_z,
    edge_center_x,
    edge_center_y,
    edge_center_z,
    edge_normal_x,
    edge_normal_y,
    edge_normal_z,
    uv,
    rbf_offset,
    rbf_kernel: InterpolationKernel,
    scale_factor: ta.wpfloat,
    array_ns: ModuleType = np,
):
    num_elements = element_center_lat.shape[0]

    # Pad edge normals and centers with a dummy zero for easier vectorized
    # computation. This may produce nans (e.g. arc length between (0,0,0) and
    # another point on the sphere), but these don't hurt the computation.
    # TODO: Can nans be signaling? default is warn:
    # https://numpy.org/doc/stable/user/misc.html#how-numpy-handles-numerical-exceptions
    def pad(f):
        return array_ns.pad(f, (0, 1), mode="constant", constant_values=0.0)

    def index_offset(f):
        return f[rbf_offset]

    edge_normal = array_ns.stack(
        (
            index_offset(pad(edge_normal_x.ndarray)),
            index_offset(pad(edge_normal_y.ndarray)),
            index_offset(pad(edge_normal_z.ndarray)),
        ),
        axis=-1,
    )
    assert edge_normal.shape == (*rbf_offset.shape, 3)

    edge_center = array_ns.stack(
        (
            index_offset(pad(edge_center_x.ndarray)),
            index_offset(pad(edge_center_y.ndarray)),
            index_offset(pad(edge_center_z.ndarray)),
        ),
        axis=-1,
    )
    assert edge_center.shape == (*rbf_offset.shape, 3)

    # Compute distances for right hand side(s) of linear system
    element_center = array_ns.stack(
        (
            element_center_x.ndarray,
            element_center_y.ndarray,
            element_center_z.ndarray,
        ),
        axis=-1,
    )
    assert element_center.shape == (rbf_offset.shape[0], 3)
    vector_dist = _arc_length_2(
        element_center[:, array_ns.newaxis, :], edge_center, array_ns=array_ns
    )
    assert vector_dist.shape == rbf_offset.shape
    rbf_val = _kernel(rbf_kernel, vector_dist, scale_factor, array_ns=array_ns)
    assert rbf_val.shape == rbf_offset.shape

    # Set up right hand side(s) of linear system
    z_nx = []
    nxnx = []
    rhs = []
    num_zonal_meridional_components = len(uv)

    assert 1 <= num_zonal_meridional_components <= 2
    for i in range(num_zonal_meridional_components):
        z_nx_x, z_nx_y, z_nx_z = _cartesian_coordinates_from_zonal_and_meridional_components(
            element_center_lat.ndarray,
            element_center_lon.ndarray,
            uv[i][0],
            uv[i][1],
            array_ns=array_ns,
        )
        z_nx.append(array_ns.stack((z_nx_x, z_nx_y, z_nx_z), axis=-1))
        assert z_nx[i].shape == (rbf_offset.shape[0], 3)

        nxnx.append(
            array_ns.matmul(z_nx[i][:, array_ns.newaxis], edge_normal.transpose(0, 2, 1)).squeeze()
        )
        rhs.append(rbf_val * nxnx[i])
        assert rhs[i].shape == rbf_offset.shape

    # Compute dot product of normal vectors for RBF interpolation matrix
    z_nxprod = _dot_product(edge_normal, edge_normal, array_ns=array_ns)
    assert z_nxprod.shape == (
        rbf_offset.shape[0],
        rbf_offset.shape[1],
        rbf_offset.shape[1],
    )

    # Distance between edge midpoints for RBF interpolation matrix
    z_dist = _arc_length_pairwise(edge_center, array_ns=array_ns)
    assert z_dist.shape == (
        rbf_offset.shape[0],
        rbf_offset.shape[1],
        rbf_offset.shape[1],
    )

    # Set up RBF interpolation matrix
    z_rbfmat = z_nxprod * _kernel(rbf_kernel, z_dist, scale_factor, array_ns=array_ns)
    assert z_rbfmat.shape == (
        rbf_offset.shape[0],
        rbf_offset.shape[1],
        rbf_offset.shape[1],
    )

    # Solve linear system for coefficients
    #
    # Currently always on CPU. At the time of writing cupy does not have
    # cho_solve with the same interface as scipy, but one has been proposed:
    # https://github.com/cupy/cupy/pull/9116.
    rbf_vec_coeff_np = [
        np.zeros(rbf_offset.shape, dtype=ta.wpfloat) for j in range(num_zonal_meridional_components)
    ]
    rbf_offset_np = data_alloc.as_numpy(rbf_offset)
    z_rbfmat_np = data_alloc.as_numpy(z_rbfmat)
    rhs_np = [data_alloc.as_numpy(x) for x in rhs]
    for i in range(num_elements):
        invalid_neighbors = np.where(rbf_offset[i, :] < 0)[0]
        num_neighbors = rbf_offset_np.shape[1] - invalid_neighbors.size
        rbfmat_np = z_rbfmat_np[i, :num_neighbors, :num_neighbors]
        z_diag_np = sla.cho_factor(rbfmat_np)
        for j in range(num_zonal_meridional_components):
            rbf_vec_coeff_np[j][i, :num_neighbors] = sla.cho_solve(
                z_diag_np, np.nan_to_num(rhs_np[j][i, :num_neighbors])
            )
    rbf_vec_coeff = [array_ns.asarray(x) for x in rbf_vec_coeff_np]

    # Normalize coefficients
    for j in range(num_zonal_meridional_components):
        rbf_vec_coeff[j] /= array_ns.sum(nxnx[j] * rbf_vec_coeff[j], axis=1)[:, array_ns.newaxis]

    return rbf_vec_coeff


def compute_rbf_interpolation_matrix_cell(
    cell_center_lat: fa.CellField[ta.wpfloat],
    cell_center_lon: fa.CellField[ta.wpfloat],
    cell_center_x: fa.CellField[ta.wpfloat],
    cell_center_y: fa.CellField[ta.wpfloat],
    cell_center_z: fa.CellField[ta.wpfloat],
    edge_center_x: fa.EdgeField[ta.wpfloat],
    edge_center_y: fa.EdgeField[ta.wpfloat],
    edge_center_z: fa.EdgeField[ta.wpfloat],
    edge_normal_x: fa.EdgeField[ta.wpfloat],
    edge_normal_y: fa.EdgeField[ta.wpfloat],
    edge_normal_z: fa.EdgeField[ta.wpfloat],
    rbf_offset: fa.CellField[int],
    rbf_kernel: InterpolationKernel,
    scale_factor: ta.wpfloat,
    array_ns: ModuleType = np,
) -> tuple[fa.CellField, fa.CellField]:
    zeros = array_ns.zeros(rbf_offset.shape[0], dtype=ta.wpfloat)
    ones = array_ns.ones(rbf_offset.shape[0], dtype=ta.wpfloat)

    coeffs = _compute_rbf_interpolation_matrix(
        cell_center_lat,
        cell_center_lon,
        cell_center_x,
        cell_center_y,
        cell_center_z,
        edge_center_x,
        edge_center_y,
        edge_center_z,
        edge_normal_x,
        edge_normal_y,
        edge_normal_z,
        [(ones, zeros), (zeros, ones)],
        rbf_offset,
        rbf_kernel,
        scale_factor,
        array_ns=array_ns,
    )
    assert len(coeffs) == 2
    return coeffs


def compute_rbf_interpolation_matrix_edge(
    edge_center_lat: fa.EdgeField[ta.wpfloat],
    edge_center_lon: fa.EdgeField[ta.wpfloat],
    edge_center_x: fa.EdgeField[ta.wpfloat],
    edge_center_y: fa.EdgeField[ta.wpfloat],
    edge_center_z: fa.EdgeField[ta.wpfloat],
    edge_normal_x: fa.EdgeField[ta.wpfloat],
    edge_normal_y: fa.EdgeField[ta.wpfloat],
    edge_normal_z: fa.EdgeField[ta.wpfloat],
    edge_dual_normal_u: fa.EdgeField[ta.wpfloat],
    edge_dual_normal_v: fa.EdgeField[ta.wpfloat],
    rbf_offset: fa.EdgeField[int],
    rbf_kernel: InterpolationKernel,
    scale_factor: ta.wpfloat,
    array_ns: ModuleType = np,
):
    coeffs = _compute_rbf_interpolation_matrix(
        edge_center_lat,
        edge_center_lon,
        edge_center_x,
        edge_center_y,
        edge_center_z,
        edge_center_x,
        edge_center_y,
        edge_center_z,
        edge_normal_x,
        edge_normal_y,
        edge_normal_z,
        [(edge_dual_normal_u.ndarray, edge_dual_normal_v.ndarray)],
        rbf_offset,
        rbf_kernel,
        scale_factor,
        array_ns=array_ns,
    )
    assert len(coeffs) == 1
    return coeffs[0]


def compute_rbf_interpolation_matrix_vertex(
    vertex_center_lat: fa.VertexField[ta.wpfloat],
    vertex_center_lon: fa.VertexField[ta.wpfloat],
    vertex_center_x: fa.VertexField[ta.wpfloat],
    vertex_center_y: fa.VertexField[ta.wpfloat],
    vertex_center_z: fa.VertexField[ta.wpfloat],
    edge_center_x: fa.EdgeField[ta.wpfloat],
    edge_center_y: fa.EdgeField[ta.wpfloat],
    edge_center_z: fa.EdgeField[ta.wpfloat],
    edge_normal_x: fa.EdgeField[ta.wpfloat],
    edge_normal_y: fa.EdgeField[ta.wpfloat],
    edge_normal_z: fa.EdgeField[ta.wpfloat],
    rbf_offset: fa.EdgeField[int],
    rbf_kernel: InterpolationKernel,
    scale_factor: ta.wpfloat,
    array_ns: ModuleType = np,
) -> tuple[fa.VertexField, fa.VertexField]:
    zeros = array_ns.zeros(rbf_offset.shape[0], dtype=ta.wpfloat)
    ones = array_ns.ones(rbf_offset.shape[0], dtype=ta.wpfloat)

    coeffs = _compute_rbf_interpolation_matrix(
        vertex_center_lat,
        vertex_center_lon,
        vertex_center_x,
        vertex_center_y,
        vertex_center_z,
        edge_center_x,
        edge_center_y,
        edge_center_z,
        edge_normal_x,
        edge_normal_y,
        edge_normal_z,
        [(ones, zeros), (zeros, ones)],
        rbf_offset,
        rbf_kernel,
        scale_factor,
        array_ns=array_ns,
    )
    assert len(coeffs) == 2
    return coeffs
