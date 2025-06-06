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

import gt4py.next as gtx
import numpy as np
import scipy.linalg as sla

from icon4py.model.common import (
    dimension as dims,
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


def compute_default_rbf_scale(mean_characteristic_length: ta.wpfloat, dim: RBFDimension):
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


def _arc_length_pairwise(v: data_alloc.NDArray, array_ns: ModuleType = np) -> data_alloc.NDArray:
    """
    Compute the pairwise arc lengths between points in each row of v.

    Args:
        v: 3D array of shape (n, m, 3) where n is the number of elements,
           m is the number of points per row (RBF dimension), and 3 is the
           dimension of the points.
        array_ns: numpy or cupy module to use for computations.
    """
    # For pairs of points p1 and p2 compute:
    # arccos(dot(p1, p2) / (norm(p1) * norm(p2))) noqa: ERA001
    # Compute all pairs of dot products
    arc_lengths = _dot_product(v, v, array_ns=array_ns)
    # Use the dot product of the diagonals to get the norm of each point
    norms = array_ns.sqrt(array_ns.diagonal(arc_lengths, axis1=1, axis2=2))
    # Divide the dot products by the broadcasted norms
    array_ns.divide(arc_lengths, norms[:, :, array_ns.newaxis], out=arc_lengths)
    array_ns.divide(arc_lengths, norms[:, array_ns.newaxis, :], out=arc_lengths)
    # Ensure all points are within [-1.0, 1.0] (may be outside due to numerical
    # inaccuracies)
    array_ns.clip(arc_lengths, -1.0, 1.0, out=arc_lengths)
    return array_ns.arccos(arc_lengths)


def _arc_length_vector_matrix(
    v1: data_alloc.NDArray, v2: data_alloc.NDArray, array_ns: ModuleType = np
) -> data_alloc.NDArray:
    """
    Compute the arc lengths between each point in v1 and the points in v2 at the same row.

    Args:
        v1: 2D array of shape (n, 3) where n is the number of elements and 3 is
            the dimension of the points.
        v2: 3D array of shape (n, m, 3) where n is the number of elements,  m is
            the number of points per row (RBF dimension), and 3 is the dimension
            of the points.
        array_ns: numpy or cupy module to use for computations.
    """
    # For pairs of points p1 and p2 compute:
    # arccos(dot(p1, p2) / (norm(p1) * norm(p2))) noqa: ERA001
    # Compute all pairs of dot products
    arc_lengths = _dot_product(v1, v2, array_ns=array_ns)
    v1_norm = array_ns.linalg.norm(v1, axis=-1)
    v2_norm = array_ns.linalg.norm(v2, axis=-1)
    # Divide the dot products by the broadcasted norms
    array_ns.divide(arc_lengths, v1_norm[:, :, array_ns.newaxis], out=arc_lengths)
    array_ns.divide(arc_lengths, v2_norm[:, array_ns.newaxis, :], out=arc_lengths)
    # Ensure all points are within [-1.0, 1.0] (may be outside due to numerical
    # inaccuracies)
    array_ns.clip(arc_lengths, -1.0, 1.0, out=arc_lengths)
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


def _compute_rbf_interpolation_coeffs(
    element_center_lat: data_alloc.NDArray,
    element_center_lon: data_alloc.NDArray,
    element_center_x: data_alloc.NDArray,
    element_center_y: data_alloc.NDArray,
    element_center_z: data_alloc.NDArray,
    edge_center_x: data_alloc.NDArray,
    edge_center_y: data_alloc.NDArray,
    edge_center_z: data_alloc.NDArray,
    edge_normal_x: data_alloc.NDArray,
    edge_normal_y: data_alloc.NDArray,
    edge_normal_z: data_alloc.NDArray,
    uv: list[tuple[data_alloc.NDArray, data_alloc.NDArray]],
    rbf_offset: data_alloc.NDArray,
    rbf_kernel: InterpolationKernel,
    scale_factor: ta.wpfloat,
    horizontal_start: gtx.int32,
    array_ns: ModuleType = np,
):
    rbf_offset_shape_full = rbf_offset.shape
    rbf_offset = rbf_offset[horizontal_start:]
    num_elements = rbf_offset.shape[0]

    # Pad edge normals and centers with a dummy zero for easier vectorized
    # computation. This may produce nans (e.g. arc length between (0,0,0) and
    # another point on the sphere), but these don't hurt the computation.
    def pad(f):
        return array_ns.pad(f, (0, 1), mode="constant", constant_values=0.0)

    def index_offset(f):
        return f[rbf_offset]

    edge_normal = array_ns.stack(
        (
            index_offset(pad(edge_normal_x)),
            index_offset(pad(edge_normal_y)),
            index_offset(pad(edge_normal_z)),
        ),
        axis=-1,
    )
    assert edge_normal.shape == (*rbf_offset.shape, 3)

    edge_center = array_ns.stack(
        (
            index_offset(pad(edge_center_x)),
            index_offset(pad(edge_center_y)),
            index_offset(pad(edge_center_z)),
        ),
        axis=-1,
    )
    assert edge_center.shape == (*rbf_offset.shape, 3)

    # Compute distances for right hand side(s) of linear system
    element_center = array_ns.stack(
        (
            element_center_x[horizontal_start:],
            element_center_y[horizontal_start:],
            element_center_z[horizontal_start:],
        ),
        axis=-1,
    )
    assert element_center.shape == (rbf_offset.shape[0], 3)
    vector_dist = _arc_length_vector_matrix(
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
            element_center_lat[horizontal_start:],
            element_center_lon[horizontal_start:],
            uv[i][0][horizontal_start:],
            uv[i][1][horizontal_start:],
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
        np.zeros(rbf_offset_shape_full, dtype=ta.wpfloat)
        for j in range(num_zonal_meridional_components)
    ]
    rbf_offset_np = data_alloc.as_numpy(rbf_offset)
    z_rbfmat_np = data_alloc.as_numpy(z_rbfmat)
    rhs_np = [data_alloc.as_numpy(x) for x in rhs]
    for i in range(num_elements):
        valid_neighbors = np.where(rbf_offset_np[i, :] >= 0)[0]
        rbfmat_np = np.squeeze(z_rbfmat_np[np.ix_([i], valid_neighbors, valid_neighbors)])
        z_diag_np = sla.cho_factor(rbfmat_np)
        for j in range(num_zonal_meridional_components):
            rbf_vec_coeff_np[j][i + horizontal_start, valid_neighbors] = sla.cho_solve(
                z_diag_np, rhs_np[j][i, valid_neighbors]
            )
    rbf_vec_coeff = [array_ns.asarray(x) for x in rbf_vec_coeff_np]

    # Normalize coefficients
    for j in range(num_zonal_meridional_components):
        rbf_vec_coeff[j][horizontal_start:] /= array_ns.sum(
            nxnx[j] * rbf_vec_coeff[j][horizontal_start:], axis=1
        )[:, array_ns.newaxis]

    return rbf_vec_coeff


def compute_rbf_interpolation_coeffs_cell(
    cell_center_lat: data_alloc.NDArray,
    cell_center_lon: data_alloc.NDArray,
    cell_center_x: data_alloc.NDArray,
    cell_center_y: data_alloc.NDArray,
    cell_center_z: data_alloc.NDArray,
    edge_center_x: data_alloc.NDArray,
    edge_center_y: data_alloc.NDArray,
    edge_center_z: data_alloc.NDArray,
    edge_normal_x: data_alloc.NDArray,
    edge_normal_y: data_alloc.NDArray,
    edge_normal_z: data_alloc.NDArray,
    rbf_offset: data_alloc.NDArray,
    # TODO: Can't pass enum as "params" in NumpyFieldsProvider?
    rbf_kernel: int,
    scale_factor: ta.wpfloat,
    horizontal_start: gtx.int32,
    array_ns: ModuleType = np,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    zeros = array_ns.zeros(rbf_offset.shape[0], dtype=ta.wpfloat)
    ones = array_ns.ones(rbf_offset.shape[0], dtype=ta.wpfloat)

    coeffs = _compute_rbf_interpolation_coeffs(
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
        InterpolationKernel(rbf_kernel),
        scale_factor,
        horizontal_start,
        array_ns=array_ns,
    )
    assert len(coeffs) == 2
    return coeffs


def compute_rbf_interpolation_coeffs_edge(
    edge_lat: data_alloc.NDArray,
    edge_lon: data_alloc.NDArray,
    edge_center_x: data_alloc.NDArray,
    edge_center_y: data_alloc.NDArray,
    edge_center_z: data_alloc.NDArray,
    edge_normal_x: data_alloc.NDArray,
    edge_normal_y: data_alloc.NDArray,
    edge_normal_z: data_alloc.NDArray,
    edge_dual_normal_u: data_alloc.NDArray,
    edge_dual_normal_v: data_alloc.NDArray,
    rbf_offset: data_alloc.NDArray,
    rbf_kernel: int,
    scale_factor: ta.wpfloat,
    horizontal_start: gtx.int32,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    coeffs = _compute_rbf_interpolation_coeffs(
        edge_lat,
        edge_lon,
        edge_center_x,
        edge_center_y,
        edge_center_z,
        edge_center_x,
        edge_center_y,
        edge_center_z,
        edge_normal_x,
        edge_normal_y,
        edge_normal_z,
        [(edge_dual_normal_u, edge_dual_normal_v)],
        rbf_offset,
        InterpolationKernel(rbf_kernel),
        scale_factor,
        horizontal_start,
        array_ns=array_ns,
    )
    assert len(coeffs) == 1
    return coeffs[0]


def compute_rbf_interpolation_coeffs_vertex(
    vertex_lat: data_alloc.NDArray,
    vertex_lon: data_alloc.NDArray,
    vertex_x: data_alloc.NDArray,
    vertex_y: data_alloc.NDArray,
    vertex_z: data_alloc.NDArray,
    edge_center_x: data_alloc.NDArray,
    edge_center_y: data_alloc.NDArray,
    edge_center_z: data_alloc.NDArray,
    edge_normal_x: data_alloc.NDArray,
    edge_normal_y: data_alloc.NDArray,
    edge_normal_z: data_alloc.NDArray,
    rbf_offset: data_alloc.NDArray,
    rbf_kernel: int,
    scale_factor: ta.wpfloat,
    horizontal_start: gtx.int32,
    array_ns: ModuleType = np,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    zeros = array_ns.zeros(rbf_offset.shape[0], dtype=ta.wpfloat)
    ones = array_ns.ones(rbf_offset.shape[0], dtype=ta.wpfloat)

    coeffs = _compute_rbf_interpolation_coeffs(
        vertex_lat,
        vertex_lon,
        vertex_x,
        vertex_y,
        vertex_z,
        edge_center_x,
        edge_center_y,
        edge_center_z,
        edge_normal_x,
        edge_normal_y,
        edge_normal_z,
        [(ones, zeros), (zeros, ones)],
        rbf_offset,
        InterpolationKernel(rbf_kernel),
        scale_factor,
        horizontal_start,
        array_ns=array_ns,
    )
    assert len(coeffs) == 2
    return coeffs
