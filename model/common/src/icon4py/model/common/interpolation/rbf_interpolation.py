# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import enum
import math
from types import MappingProxyType

import gt4py.next as gtx
import numpy as np
import scipy.linalg as sla

from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.grid import base as base_grid
from icon4py.model.common.math.helpers import (
    cartesian_coordinates_from_zonal_and_meridional_components_on_cells,
    cartesian_coordinates_from_zonal_and_meridional_components_on_edges,
    cartesian_coordinates_from_zonal_and_meridional_components_on_vertices,
)
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


# TODO: Use these as default?
@dataclasses.dataclass(frozen=True)
class InterpolationConfig:
    # for nested setup this value is a vector of size num_domains
    # and the default value is resolution dependent, according to the namelist
    # documentation in ICON
    rbf_vector_scale: dict[RBFDimension, float] = MappingProxyType(
        {
            RBFDimension.CELL: 1.0,
            RBFDimension.EDGE: 1.0,
            RBFDimension.VERTEX: 1.0,
            RBFDimension.GRADIENT: 1.0,
        }
    )

    rbf_kernel: dict[RBFDimension, InterpolationKernel] = MappingProxyType(
        {
            RBFDimension.CELL: InterpolationKernel.GAUSSIAN,
            RBFDimension.EDGE: InterpolationKernel.INVERSE_MULTIQUADRATIC,
            RBFDimension.VERTEX: InterpolationKernel.GAUSSIAN,
            RBFDimension.GRADIENT: InterpolationKernel.GAUSSIAN,
        }
    )


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
    c2e2c = grid.connectivities[dims.C2E2CDim]
    c2e = grid.connectivities[dims.C2EDim]
    offset = c2e[c2e2c]
    assert len(offset.shape) == 3
    # flatten this offset to construct a (num_cells, RBFDimension.CELL) shape offset matrix
    flattened_offset = offset.reshape((offset.shape[0], offset.shape[1] * offset.shape[2]))
    assert flattened_offset.shape == (
        grid.num_cells,
        RBF_STENCIL_SIZE[RBFDimension.CELL],
    )
    return flattened_offset


def construct_rbf_matrix_offsets_tables_for_edges(
    grid: base_grid.BaseGrid,
) -> data_alloc.NDArray:
    """Compute the neighbor tables for the edge RBF matrix: rbf_vec_index_e"""
    e2c2e = grid.connectivities[dims.E2C2EDim]
    offset = e2c2e
    assert offset.shape == (grid.num_edges, RBF_STENCIL_SIZE[RBFDimension.EDGE])
    return offset


def construct_rbf_matrix_offsets_tables_for_vertices(
    grid: base_grid.BaseGrid,
) -> data_alloc.NDArray:
    """Compute the neighbor tables for the edge RBF matrix: rbf_vec_index_v"""
    offset = grid.connectivities[dims.V2EDim]
    assert offset.shape == (grid.num_vertices, RBF_STENCIL_SIZE[RBFDimension.VERTEX])
    return offset


def dot_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    # alias: np.transpose(v2, axes=(0, 2, 1)) for 3d array
    v2_tilde = np.moveaxis(v2, 1, -1)
    # use linalg.matmul (array API compatible)
    return np.matmul(v1, v2_tilde)


# NOTE: this one computes the pairwise arc lengths between elements in v, the
# next version computes pairwise arc lengths between two different arrays
# TODO: Combine?
def arc_length_pairwise(v: np.ndarray) -> np.ndarray:
    # For pairs of points p1 and p2 compute:
    # arccos(dot(p1, p2) / (norm(p1) * norm(p2))) noqa: ERA001
    # Compute all pairs of dot products
    arc_lengths = dot_product(v, v)  # np.matmul(v, np.transpose(v, axes=(0, 2, 1)))
    # Use the dot product of the diagonals to get the norm of each point
    norms = np.sqrt(np.diagonal(arc_lengths, axis1=1, axis2=2))
    # Divide the dot products by the broadcasted norms
    # TODO: Check that these are broadcast correctly. Leaving them out has
    # almost no impact on result, since they're close to 1, but may affect
    # precision.
    arc_lengths = np.divide(arc_lengths, norms[:, :, np.newaxis])
    arc_lengths = np.divide(arc_lengths, norms[:, np.newaxis, :])
    # Ensure all points are within [-1.0, 1.0] (may be outside due to numerical
    # inaccuracies)
    arc_lengths = np.clip(arc_lengths, -1.0, 1.0)
    # TODO: avoid intermediates?
    return np.arccos(arc_lengths)


# TODO: This assumes v1 is 2d and v3 is 3d
# TODO: name?
# TODO: this is pretty much the same as above, except we don't get the squares
# of the norms directly from the first matmul
# TODO: this is used only in one place, it's probably not as generic as it looks
def arc_length_2(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    # For pairs of points p1 and p2 compute:
    # arccos(dot(p1, p2) / (norm(p1) * norm(p2))) noqa: ERA001
    # Compute all pairs of dot products
    arc_lengths = dot_product(v1, v2)
    v1_norm = np.linalg.norm(v1, axis=-1)
    v2_norm = np.linalg.norm(v2, axis=-1)
    # Divide the dot products by the broadcasted norms
    arc_lengths = np.divide(arc_lengths, v1_norm[:, :, np.newaxis])
    arc_lengths = np.divide(arc_lengths, v2_norm[:, np.newaxis, :])
    # Ensure all points are within [-1.0, 1.0] (may be outside due to numerical
    # inaccuracies)
    arc_lengths = np.clip(arc_lengths, -1.0, 1.0)
    return np.squeeze(np.arccos(arc_lengths), axis=1)


def gaussian(lengths: np.ndarray, scale: float) -> np.ndarray:
    val = lengths / scale
    return np.exp(-1.0 * val * val)


def inverse_multiquadratic(distance: np.ndarray, scale: float) -> np.ndarray:
    """

    Args:
        distance: radial distance
        scale: scaling parameter

    Returns:

    """
    val = distance / scale
    return 1.0 / np.sqrt(1.0 + val * val)


def kernel(kernel: InterpolationKernel, lengths: np.ndarray, scale: float):
    match kernel:
        case InterpolationKernel.GAUSSIAN:
            return gaussian(lengths, scale)
        case InterpolationKernel.INVERSE_MULTIQUADRATIC:
            return inverse_multiquadratic(lengths, scale)
        case _:
            raise ValueError(f"Unsupported kernel: {kernel}")


def get_zonal_meridional_f(dim: gtx.Dimension):
    match dim:
        case gtx.Dimension("Cell"):
            return cartesian_coordinates_from_zonal_and_meridional_components_on_cells
        case gtx.Dimension("Edge"):
            return cartesian_coordinates_from_zonal_and_meridional_components_on_edges
        case gtx.Dimension("Vertex"):
            return cartesian_coordinates_from_zonal_and_meridional_components_on_vertices
        case _:
            raise ValueError(f"Unsupported dimension: {dim}")


def compute_rbf_interpolation_matrix(
    element_center_lat,  # fa.CellField[ta.wpfloat], TODO: any of CellField, EdgeField, VertexField
    element_center_lon,  # fa.CellField[ta.wpfloat],
    element_center_x,  # fa.CellField[ta.wpfloat],
    element_center_y,  # fa.CellField[ta.wpfloat],
    element_center_z,  # fa.CellField[ta.wpfloat],
    edge_center_x,  # fa.EdgeField[ta.wpfloat],
    edge_center_y,  # fa.EdgeField[ta.wpfloat],
    edge_center_z,  # fa.EdgeField[ta.wpfloat],
    edge_normal_x,  # fa.EdgeField[ta.wpfloat],
    edge_normal_y,  # fa.EdgeField[ta.wpfloat],
    edge_normal_z,  # fa.EdgeField[ta.wpfloat],
    u,
    v,
    rbf_offset,  # field_alloc.NDArray, [num_dim, RBFDimension(dim)]
    rbf_kernel: InterpolationKernel,
    scale_factor: float,
):
    # Pad edge normals and centers with a dummy zero for easier vectorized
    # computation. This may produce nans (e.g. arc length between (0,0,0) and
    # another point on the sphere), but these don't hurt the computation.
    # TODO: Can nans be signaling? default is warn:
    # https://numpy.org/doc/stable/user/misc.html#how-numpy-handles-numerical-exceptions
    def pad(f):
        return np.pad(f, (0, 1), mode="constant", constant_values=0.0)

    def index_offset(f):
        return f[rbf_offset]

    edge_normal = np.stack(
        (
            index_offset(pad(edge_normal_x.asnumpy())),
            index_offset(pad(edge_normal_y.asnumpy())),
            index_offset(pad(edge_normal_z.asnumpy())),
        ),
        axis=-1,
    )
    assert edge_normal.shape == (*rbf_offset.shape, 3)

    edge_center = np.stack(
        (
            index_offset(pad(edge_center_x.asnumpy())),
            index_offset(pad(edge_center_y.asnumpy())),
            index_offset(pad(edge_center_z.asnumpy())),
        ),
        axis=-1,
    )
    assert edge_center.shape == (*rbf_offset.shape, 3)

    # Compute distances for right hand side(s) of linear system
    element_center = np.stack(
        (
            element_center_x.asnumpy(),
            element_center_y.asnumpy(),
            element_center_z.asnumpy(),
        ),
        axis=-1,
    )
    assert element_center.shape == (rbf_offset.shape[0], 3)
    vector_dist = arc_length_2(element_center[:, np.newaxis, :], edge_center)
    assert vector_dist.shape == rbf_offset.shape
    rbf_val = kernel(rbf_kernel, vector_dist, scale_factor)
    assert rbf_val.shape == rbf_offset.shape

    # Set up right hand side(s) of linear system
    domain = element_center_lat.domain
    dim = domain[0].dim
    zonal_meridional_f = get_zonal_meridional_f(dim)

    z_nx = []
    nxnx = []
    rhs = []

    assert len(u) == len(v)
    assert 1 <= len(u) <= 2
    for i in range(len(u)):
        z_nx_x = gtx.zeros(domain, dtype=ta.wpfloat)
        z_nx_y = gtx.zeros(domain, dtype=ta.wpfloat)
        z_nx_z = gtx.zeros(domain, dtype=ta.wpfloat)

        zonal_meridional_f(
            element_center_lat,
            element_center_lon,
            u[i],
            v[i],
            out=(z_nx_x, z_nx_y, z_nx_z),
            offset_provider={},
        )
        z_nx.append(np.stack((z_nx_x.asnumpy(), z_nx_y.asnumpy(), z_nx_z.asnumpy()), axis=-1))
        assert z_nx[i].shape == (rbf_offset.shape[0], 3)

        nxnx.append(np.matmul(z_nx[i][:, np.newaxis], edge_normal.transpose(0, 2, 1)).squeeze())
        rhs.append(rbf_val * nxnx[i])
        assert rhs[i].shape == rbf_offset.shape

    # Compute dot product of normal vectors for RBF interpolation matrix
    z_nxprod = dot_product(edge_normal, edge_normal)
    assert z_nxprod.shape == (
        rbf_offset.shape[0],
        rbf_offset.shape[1],
        rbf_offset.shape[1],
    )

    # Distance between edge midpoints for RBF interpolation matrix
    z_dist = arc_length_pairwise(edge_center)
    assert z_dist.shape == (
        rbf_offset.shape[0],
        rbf_offset.shape[1],
        rbf_offset.shape[1],
    )

    # Set up RBF interpolation matrix
    z_rbfmat = z_nxprod * kernel(rbf_kernel, z_dist, scale_factor)
    assert z_rbfmat.shape == (
        rbf_offset.shape[0],
        rbf_offset.shape[1],
        rbf_offset.shape[1],
    )

    # Solve linear system for coefficients
    # TODO: vectorize this?
    rbf_vec_coeff = [np.zeros(rbf_offset.shape, dtype=ta.wpfloat) for j in range(len(u))]
    for i in range(z_rbfmat.shape[0]):
        invalid_neighbors = np.where(rbf_offset[i, :] < 0)[0]
        num_neighbors = rbf_offset.shape[1] - invalid_neighbors.size
        rbfmat = z_rbfmat[i, :num_neighbors, :num_neighbors]
        z_diag = sla.cho_factor(rbfmat)
        for j in range(len(u)):
            rbf_vec_coeff[j][i, :num_neighbors] = sla.cho_solve(
                z_diag, np.nan_to_num(rhs[j][i, :num_neighbors])
            )

    # Normalize coefficients
    for j in range(len(u)):
        rbf_vec_coeff[j] /= np.sum(nxnx[j] * rbf_vec_coeff[j], axis=1)[:, np.newaxis]

    dim2 = gtx.Dimension("What")  # TODO
    return tuple(gtx.as_field([dim, dim2], c) for c in rbf_vec_coeff)


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
) -> tuple[fa.CellField, fa.CellField]:
    zeros = gtx.zeros(cell_center_lat.domain, dtype=ta.wpfloat)
    ones = gtx.ones(cell_center_lat.domain, dtype=ta.wpfloat)

    coeffs = compute_rbf_interpolation_matrix(
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
        [ones, zeros],
        [zeros, ones],
        rbf_offset,
        rbf_kernel,
        scale_factor,
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
    scale_factor: float,
):
    # TODO: computing too much here
    coeffs = compute_rbf_interpolation_matrix(
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
        [edge_dual_normal_u],
        [edge_dual_normal_v],
        rbf_offset,
        rbf_kernel,
        scale_factor,
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
    scale_factor: float,
) -> tuple[fa.VertexField, fa.VertexField]:
    zeros = gtx.zeros(vertex_center_lat.domain, dtype=ta.wpfloat)
    ones = gtx.ones(vertex_center_lat.domain, dtype=ta.wpfloat)

    coeffs = compute_rbf_interpolation_matrix(
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
        [ones, zeros],
        [zeros, ones],
        rbf_offset,
        rbf_kernel,
        scale_factor,
    )
    assert len(coeffs) == 2
    return coeffs
