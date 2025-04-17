# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import enum
from types import MappingProxyType

import numpy as np
import scipy.linalg as sla

from icon4py.model.common import dimension as dims
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
    GAUSSIAN = (1,) # TODO: why tuple?
    INVERSE_MULTI_QUADRATIC = 3


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
            RBFDimension.EDGE: InterpolationKernel.INVERSE_MULTI_QUADRATIC,
            RBFDimension.VERTEX: InterpolationKernel.GAUSSIAN,
            RBFDimension.GRADIENT: InterpolationKernel.GAUSSIAN,
        }
    )


def construct_rbf_matrix_offsets_tables_for_cells(grid: base_grid.BaseGrid) -> data_alloc.NDArray:
    """Compute the neighbor tables for the cell RBF matrix: rbf_vec_index_c

    TODO: deal with Invalid C2E2C neighbors: either because of lateral boundaries or because of halos
    """
    c2e2c = grid.connectivities[dims.C2E2CDim]
    c2e = grid.connectivities[dims.C2EDim]
    offset = c2e[c2e2c]
    shp = offset.shape
    assert len(shp) == 3
    # flatten this offset to construct a (num_cells, RBFDimension.CELL) shape offset matrix
    new_shape = (shp[0], shp[1] * shp[2])
    flattened_offset = offset.reshape(new_shape)
    return flattened_offset


def construct_rbf_matrix_offsets_tables_for_edges(grid: base_grid.BaseGrid) -> data_alloc.NDArray:
    """Compute the neighbor tables for the edge RBF matrix: rbf_vec_index_e

    TODO: edge cases? -1 signals no neighbor
    """
    e2c2e = grid.connectivities[dims.E2C2EDim]
    offset = e2c2e
    shp = offset.shape
    assert len(shp) == 2
    return offset


def construct_rbf_matrix_offsets_tables_for_vertices(grid: base_grid.BaseGrid) -> data_alloc.NDArray:
    """Compute the neighbor tables for the edge RBF matrix: rbf_vec_index_v

    TODO: edge cases? -1 signals no neighbor
    """
    v2e = grid.connectivities[dims.V2EDim]
    offset = v2e
    shp = offset.shape
    assert len(shp) == 2
    return offset


def dot_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    # alias: np.transpose(v2, axes=(0, 2, 1)) for 3d array
    v2_tilde = np.moveaxis(v2, 1, -1)
    # use linalg.matmul (array API compatible)
    return np.matmul(v1, v2_tilde)


# NOTE: this one computes the pairwise arc lengths between elements in v, the
# next version computes pairwise arc lengths between two different arrays
# TODO: Combine?
def arc_length_matrix(v: np.ndarray) -> np.ndarray:
    # TODO: equivalent?
    # v_norm = _normalize_along_last_axis(v)
    # return _arc_length_of_normalized_input(v_norm, v_norm)

    # For pairs of points p1 and p2 compute:
    # arccos(dot(p1, p2) / (norm(p1) * norm(p2)))
    # Compute all pairs of dot products
    a = np.matmul(v, np.transpose(v, axes=(0, 2, 1)))
    # Use the dot product of the diagonals to get the norm of each point
    # TODO: Always 1? Almost 1? Points on the unit sphere?
    b = np.sqrt(np.diagonal(a, axis1=1, axis2=2))
    # Divide the dot products by the broadcasted norms
    # TODO: Use newaxis instead?
    c = np.divide(a, np.reshape(b, shape=(b.shape[0], b.shape[1], 1)))
    d = np.divide(c, np.reshape(b, shape=(b.shape[0], 1, b.shape[1])))
    # Ensure all points are within [-1.0, 1.0] (may be outside due to numerical
    # inaccuracies)
    e = np.clip(d, -1.0, 1.0)
    # TODO: avoid intermediates?
    return np.arccos(e)


# TODO: This assumes v1 is 2d and v3 is 3d
# TODO: name?
# TODO: this is pretty much the same as above, except we don't get the squares
# of the norms directly from the first matmul
# TODO: this is used only in one place, it's probably not as generic as it looks
def arc_length_2(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    # For pairs of points p1 and p2 compute:
    # arccos(dot(p1, p2) / (norm(p1) * norm(p2)))
    # Compute all pairs of dot products
    a = np.matmul(v1, np.transpose(v2, axes=(0, 2, 1)))
    # TODO: Always 1? Almost 1? Points on the unit sphere?
    b = np.linalg.norm(v1, axis=-1)
    c = np.linalg.norm(v2, axis=-1)
    # Divide the dot products by the broadcasted norms
    # TODO: Use newaxis instead?
    d = np.divide(a, np.reshape(b, shape=(b.shape[0], b.shape[1], 1)))
    e = np.divide(d, np.reshape(c, shape=(c.shape[0], 1, c.shape[1])))
    # Ensure all points are within [-1.0, 1.0] (may be outside due to numerical
    # inaccuracies)
    f = np.clip(e, -1.0, 1.0)
    return np.squeeze(np.arccos(f), axis=1)


# TODO: Use?
def arc_length(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    v1_norm = _normalize_along_last_axis(v1)
    v2_norm = _normalize_along_last_axis(v2)

    return _arc_length_of_normalized_input(v1_norm, v2_norm)


# TODO: Use?
def _arc_length_of_normalized_input(v1_norm, v2_norm):
    d = dot_product(v1_norm, v2_norm)
    d_clip = np.clip(d, -1.0, 1.0)
    r = np.arccos(d_clip)
    return r


def _normalize_along_last_axis(v: np.ndarray):
    norms = np.sqrt(np.sum(v * 1, axis=-1))
    # print("_normalize_along_last_axis")
    # print("norms")
    # print(norms)
    # print("v / norms[..., np.newaxis]")
    r = v / norms[..., np.newaxis]
    # print(r)
    return r


def gaussian(lengths: np.ndarray, scale: float) -> np.ndarray:
    val = lengths / scale
    return np.exp(-1.0 * val * val)


def multiquadratic(distance: np.ndarray, scale: float) -> np.ndarray:
    """

    Args:
        distance: radial distance
        scale: scaling parameter

    Returns:

    """
    val = distance * scale
    return 1.0 / np.sqrt(1.0 + val * val)


def kernel(kernel: InterpolationKernel, lengths: np.ndarray, scale: float):
    match kernel:
        case InterpolationKernel.GAUSSIAN:
            return gaussian(lengths, scale)
        case InterpolationKernel.INVERSE_MULTI_QUADRATIC:
            return multiquadratic(lengths, scale)
        case _:
            assert False # TODO: error?


# TODO proper name...
# TODO: Use function from helpers.py?
def zonal_meridional_component(
    thing_center_lat: data_alloc.NDArray,  # fa.CellField[ta.wpfloat], # TODO: type? not always CellField
    thing_center_lon: data_alloc.NDArray,  # fa.CellField[ta.wpfloat],
    u: data_alloc.NDArray,  # fa.CellField[ta.wpfloat],
    v: data_alloc.NDArray,  # fa.CellField[ta.wpfloat],
) -> data_alloc.NDArray:
    """compute z_nx1 and z_nx2"""
    sin_lat = np.sin(thing_center_lat)
    sin_lon = np.sin(thing_center_lon)
    cos_lat = np.cos(thing_center_lat)
    cos_lon = np.cos(thing_center_lon)

    x = -1.0 * (sin_lon * u + sin_lat * cos_lon * v)
    y = cos_lon * u - sin_lat * sin_lon * v
    z = cos_lat * v
    cartesian_v = np.stack((x, y, z), axis=-1)
    # TODO: Always 1 or almost 1?
    norms = np.linalg.norm(cartesian_v, axis=-1)
    return cartesian_v / norms[:, np.newaxis]


def compute_rbf_interpolation_matrix(
    # TODO: naming
    thing_center_lat: data_alloc.NDArray,  # fa.CellField[ta.wpfloat], TODO: any of CellField, EdgeField, VertexField
    thing_center_lon: data_alloc.NDArray,  # fa.CellField[ta.wpfloat],
    thing_center_x: data_alloc.NDArray,  # fa.CellField[ta.wpfloat],
    thing_center_y: data_alloc.NDArray,  # fa.CellField[ta.wpfloat],
    thing_center_z: data_alloc.NDArray,  # fa.CellField[ta.wpfloat],
    edge_center_x: data_alloc.NDArray,  # fa.EdgeField[ta.wpfloat],
    edge_center_y: data_alloc.NDArray,  # fa.EdgeField[ta.wpfloat],
    edge_center_z: data_alloc.NDArray,  # fa.EdgeField[ta.wpfloat],
    edge_normal_x: data_alloc.NDArray,  # fa.EdgeField[ta.wpfloat],
    edge_normal_y: data_alloc.NDArray,  # fa.EdgeField[ta.wpfloat],
    edge_normal_z: data_alloc.NDArray,  # fa.EdgeField[ta.wpfloat],
    rbf_offset: data_alloc.NDArray,  # field_alloc.NDArray, [num_dim, RBFDimension(dim)]
    rbf_kernel: InterpolationKernel,
    scale_factor: float,
    # TODO: Find another interface to handle edge field (only one set of
    # coefficients needed, different input for u and v)
    u: data_alloc.NDArray = None,
    v: data_alloc.NDArray = None,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    # compute neighbor list and create "cartesian coordinate" vectors (x,y,z) in last dimension
    # 1) get the rbf offset (neighbor list) - currently: input
    # Pad edge normals and centers with a dummy zero for easier vectorized
    # computation. This may produce nans (e.g. arc length between (0,0,0) and
    # another point on the sphere), but these don't hurt the computation.
    # TODO: Can nans be signaling? default is warn:
    # https://numpy.org/doc/stable/user/misc.html#how-numpy-handles-numerical-exceptions
    edge_normal_x = np.pad(edge_normal_x, (0, 1), mode="constant", constant_values=0.0)
    edge_normal_y = np.pad(edge_normal_y, (0, 1), mode="constant", constant_values=0.0)
    edge_normal_z = np.pad(edge_normal_z, (0, 1), mode="constant", constant_values=0.0)
    x_normal = edge_normal_x[rbf_offset]
    y_normal = edge_normal_y[rbf_offset]
    z_normal = edge_normal_z[rbf_offset]
    edge_normal = np.stack((x_normal, y_normal, z_normal), axis=-1)
    assert edge_normal.shape == (*rbf_offset.shape, 3)
    edge_center_x = np.pad(edge_center_x, (0, 1), mode="constant", constant_values=0.0)
    edge_center_y = np.pad(edge_center_y, (0, 1), mode="constant", constant_values=0.0)
    edge_center_z = np.pad(edge_center_z, (0, 1), mode="constant", constant_values=0.0)
    x_center = edge_center_x[rbf_offset]
    y_center = edge_center_y[rbf_offset]
    z_center = edge_center_z[rbf_offset]
    edge_centers = np.stack((x_center, y_center, z_center), axis=-1)
    assert edge_centers.shape == (*rbf_offset.shape, 3)

    z_nxprod = dot_product(edge_normal, edge_normal)
    assert z_nxprod.shape == (rbf_offset.shape[0], rbf_offset.shape[1], rbf_offset.shape[1])
    z_dist = arc_length_matrix(edge_centers)
    assert z_dist.shape == (rbf_offset.shape[0], rbf_offset.shape[1], rbf_offset.shape[1])

    z_rbfmat = z_nxprod * kernel(rbf_kernel, z_dist, scale_factor)
    assert z_rbfmat.shape == (rbf_offset.shape[0], rbf_offset.shape[1], rbf_offset.shape[1])

    # 3) z_nx2, z_nx1
    ones = np.ones(thing_center_lat.shape, dtype=float)
    zeros = np.zeros(thing_center_lat.shape, dtype=float)

    # TODO: This is dumb. For the edge field we only compute one array of
    # coefficients, with given u and v components. Right now this computes z_nx1
    # and z_nx2 identically for that case.
    assert (u is None and v is None ) or (u is not None and v is not None)
    if u is None:
        z_nx1 = zonal_meridional_component(
            thing_center_lat=thing_center_lat, thing_center_lon=thing_center_lon, u=ones, v=zeros
        )
        z_nx2 = zonal_meridional_component(
            thing_center_lat=thing_center_lat, thing_center_lon=thing_center_lon, u=zeros, v=ones
        )
    else:
        z_nx1 = zonal_meridional_component(
            thing_center_lat=thing_center_lat, thing_center_lon=thing_center_lon, u=u, v=v
        )
        z_nx2 = z_nx1

    assert z_nx1.shape == (rbf_offset.shape[0], 3)
    assert z_nx2.shape == (rbf_offset.shape[0], 3)
    z_nx3 = edge_normal
    assert z_nx3.shape == (*rbf_offset.shape, 3)

    # 4 right hand side
    thing_centers = np.stack((thing_center_x, thing_center_y, thing_center_z), axis=-1)
    assert thing_centers.shape == (rbf_offset.shape[0], 3)
    vector_dist = arc_length_2(thing_centers[:, np.newaxis, :], edge_centers)
    assert vector_dist.shape == rbf_offset.shape
    rbf_val = kernel(rbf_kernel, vector_dist, scale_factor)
    assert rbf_val.shape == rbf_offset.shape
    # projection
    # TODO: dot_product is not the same as the matmul below?
    # more memory? more compute? wrong result?
    # nx1nx3 = dot_product(z_nx1, z_nx3)
    # nx2nx3 = dot_product(z_nx2, z_nx3)
    nx1nx3 = np.matmul(z_nx1[:, np.newaxis], z_nx3.transpose(0, 2, 1)).squeeze()
    nx2nx3 = np.matmul(z_nx2[:, np.newaxis], z_nx3.transpose(0, 2, 1)).squeeze()
    assert nx1nx3.shape == rbf_offset.shape
    assert nx2nx3.shape == rbf_offset.shape
    rhs1 = rbf_val * nx1nx3
    rhs2 = rbf_val * nx2nx3
    assert rhs1.shape == rbf_offset.shape
    assert rhs2.shape == rbf_offset.shape

    # 2, 5) solve cholesky system
    rbf_vec_coeff_1 = np.zeros(rbf_offset.shape, dtype=float)
    rbf_vec_coeff_2 = np.zeros(rbf_offset.shape, dtype=float)

    # TODO: vectorize this?
    for i in range(z_rbfmat.shape[0]):
        # Require filling in potential nans on the diagonal (from invalid neighbors)
        # np.fill_diagonal(z_rbfmat[i, :], 1.0)
        # Alternative: only do cholesky decomposition on valid neighbors
        # (invalid always at the end?)
        invalid_neighbors = np.where(rbf_offset[i, :] < 0)[0]
        num_neighbors = rbf_offset.shape[1] - invalid_neighbors.size
        rbfmat = z_rbfmat[i, :num_neighbors, :num_neighbors]
        # z_diag = sla.cho_factor(np.nan_to_num(z_rbfmat[i, :]))
        z_diag = sla.cho_factor(rbfmat)
        # rbf_vec_coeff_1[i, :] = sla.cho_solve(z_diag, np.nan_to_num(rhs1[i, :]))
        # rbf_vec_coeff_2[i, :] = sla.cho_solve(z_diag, np.nan_to_num(rhs2[i, :]))
        rbf_vec_coeff_1[i, :num_neighbors] = sla.cho_solve(z_diag, np.nan_to_num(rhs1[i, :num_neighbors]))
        rbf_vec_coeff_2[i, :num_neighbors] = sla.cho_solve(z_diag, np.nan_to_num(rhs2[i, :num_neighbors]))
    assert nx1nx3.shape == rbf_vec_coeff_1.shape
    assert nx2nx3.shape == rbf_vec_coeff_2.shape

    # Normalize coefficients
    rbf_vec_coeff_1 /= np.sum(nx1nx3 * rbf_vec_coeff_1, axis=1)[:, np.newaxis]
    rbf_vec_coeff_2 /= np.sum(nx2nx3 * rbf_vec_coeff_2, axis=1)[:, np.newaxis]

    return rbf_vec_coeff_1, rbf_vec_coeff_2
