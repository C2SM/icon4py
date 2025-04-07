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
    GAUSSIAN = (1,)
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


def dot_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    v2_tilde = np.moveaxis(v2, 1, -1)
    # use linalg.matmul (array API compatible)
    return np.matmul(v1, v2_tilde)


def arc_length_matrix(v: np.ndarray) -> np.ndarray:
    v_norm = _normalize_along_last_axis(v)
    return _arc_length_of_normalized_input(v_norm, v_norm)


def arc_length(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    v1_norm = _normalize_along_last_axis(v1)
    v2_norm = _normalize_along_last_axis(v2)

    return _arc_length_of_normalized_input(v1_norm, v2_norm)


def _arc_length_of_normalized_input(v1_norm, v2_norm):
    d = dot_product(v1_norm, v2_norm)
    return np.arccos(d)


def _normalize_along_last_axis(v: np.ndarray):
    norms = np.sqrt(np.sum(v * 1, axis=-1))
    return v / norms[..., np.newaxis]


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
    return np.sqrt(1.0 + val * val)


def kernel(kernel: InterpolationKernel, lengths: np.ndarray, scale: float):
    return (
        gaussian(lengths, scale)
        if kernel == InterpolationKernel.GAUSSIAN
        else multiquadratic(lengths, scale)
    )


# TODO proper name...
def zonal_meridional_component(
    cell_center_lat: data_alloc.NDArray,  # fa.CellField[ta.wpfloat],
    cell_center_lon: data_alloc.NDArray,  # fa.CellField[ta.wpfloat],
    u: data_alloc.NDArray,  # fa.CellField[ta.wpfloat],
    v: data_alloc.NDArray,  # fa.CellField[ta.wpfloat],
) -> data_alloc.NDArray:
    """compute z_nx1 and z_nx2"""
    sin_lat = np.sin(cell_center_lat)
    sin_lon = np.sin(cell_center_lon)
    cos_lat = np.cos(cell_center_lat)
    cos_lon = np.cos(cell_center_lon)

    x = -1.0 * (sin_lon * u + sin_lat * sin_lon * v)
    y = cos_lon * u - sin_lat * sin_lon * v
    z = cos_lat * v
    cartesian_v = np.stack((x, y, z), axis=-1)
    norms = np.sqrt(np.sum(cartesian_v, axis=-1))
    return cartesian_v / norms[:, np.newaxis]


def compute_rbf_interpolation_matrix(
    cell_center_lat: data_alloc.NDArray,  # fa.CellField[ta.wpfloat],
    cell_center_lon: data_alloc.NDArray,  # fa.CellField[ta.wpfloat],
    cell_center_x: data_alloc.NDArray,  # fa.CellField[ta.wpfloat],
    cell_center_y: data_alloc.NDArray,  # fa.CellField[ta.wpfloat],
    cell_center_z: data_alloc.NDArray,  # fa.CellField[ta.wpfloat],
    edge_center_x: data_alloc.NDArray,  # fa.EdgeField[ta.wpfloat],
    edge_center_y: data_alloc.NDArray,  # fa.EdgeField[ta.wpfloat],
    edge_center_z: data_alloc.NDArray,  # fa.EdgeField[ta.wpfloat],
    edge_normal_x: data_alloc.NDArray,  # fa.EdgeField[ta.wpfloat],
    edge_normal_y: data_alloc.NDArray,  # fa.EdgeField[ta.wpfloat],
    edge_normal_z: data_alloc.NDArray,  # fa.EdgeField[ta.wpfloat],
    rbf_offset: data_alloc.NDArray,  # field_alloc.NDArray, [num_dim, RBFDimension(dim)]
    rbf_kernel: InterpolationKernel,
    scale_factor: float,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    # compute neighbor list and create "cartesian coordinate" vectors (x,y,z) in last dimension
    # 1) get the rbf offset (neighbor list) - currently: input
    x_normal = edge_normal_x[rbf_offset]
    y_normal = edge_normal_y[rbf_offset]
    z_normal = edge_normal_z[rbf_offset]
    edge_normal = np.stack((x_normal, y_normal, z_normal), axis=-1)
    x_center = edge_center_x[rbf_offset]
    y_center = edge_center_y[rbf_offset]
    z_center = edge_center_z[rbf_offset]
    edge_centers = np.stack((x_center, y_center, z_center), axis=-1)

    z_nxprod = dot_product(edge_normal, edge_normal)
    z_dist = arc_length_matrix(edge_centers)

    z_rbfmat = z_nxprod * kernel(rbf_kernel, z_dist, scale_factor)

    # 3) z_nx2, z_nx1
    ones = np.ones(cell_center_lat.shape, dtype=float)
    zeros = np.zeros(cell_center_lat.shape, dtype=float)

    z_nx1 = zonal_meridional_component(
        cell_center_lat=cell_center_lat, cell_center_lon=cell_center_lon, u=ones, v=zeros
    )
    z_nx2 = zonal_meridional_component(
        cell_center_lat=cell_center_lat, cell_center_lon=cell_center_lon, u=zeros, v=ones
    )
    z_nx3 = edge_normal

    # 4 right hand side
    cell_centers = np.stack((cell_center_x, cell_center_y, cell_center_z), axis=-1)
    vector_dist = arc_length(cell_centers, edge_centers)
    rbf_val = kernel(rbf_kernel, vector_dist, scale_factor)
    # projection
    rhs1 = rbf_val * dot_product(z_nx1, z_nx3)
    rhs2 = rbf_val * dot_product(z_nx2, z_nx3)

    # 2, 5) solve choleski system
    rbf_vec_coeff_1 = np.zeros(rbf_offset.shape, dtype=float)
    rbf_vec_coeff_2 = np.zeros(rbf_offset.shape, dtype=float)

    for i in range(z_rbfmat.shape[0]):
        mat = z_rbfmat[i, :]
        z_diag = np.linalg.cholesky(mat, upper=False)
        rbf_vec_coeff_1[i, :] = sla.solve_triangular(z_diag, rhs1)
        rbf_vec_coeff_2[i, :] = sla.solve_triangular(z_diag, rhs2)

    return rbf_vec_coeff_1, rbf_vec_coeff_2
