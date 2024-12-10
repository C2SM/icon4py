import dataclasses
import enum
from types import MappingProxyType

import numpy as np

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base as base_grid
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc


class RBFDimension(enum.Enum):
    CELL = "cell"
    EDGE = "edge"
    VERTEX = "vertex"
    GRADIENT = "midpoint_gradient"

RBF_STENCIL_SIZE:dict[RBFDimension, int] = {
    RBFDimension.CELL:9,
    RBFDimension.EDGE: 4,
    RBFDimension.VERTEX:6,
    RBFDimension.GRADIENT:10
}

class InterpolationKernel(enum.Enum):
    GAUSSIAN = 1,
    INVERSE_MULTI_QUADRATIC = 3
    

@dataclasses.dataclass(frozen=True)
class InterpolationConfig:

    # for nested setup this value is a vector of size num_domains
    # and the default value is resolution dependent, according to the namelist
    # documentation in ICON
    rbf_vector_scale: dict[RBFDimension, float] = MappingProxyType({
        RBFDimension.CELL: 1.0,
        RBFDimension.EDGE:1.0,
        RBFDimension.VERTEX:1.0,
        RBFDimension.GRADIENT:1.0
    })

    rbf_kernel: dict[RBFDimension, InterpolationKernel] = MappingProxyType({
        RBFDimension.CELL:InterpolationKernel.GAUSSIAN,
        RBFDimension.EDGE:InterpolationKernel.INVERSE_MULTI_QUADRATIC,
        RBFDimension.VERTEX: InterpolationKernel.GAUSSIAN,
        RBFDimension.GRADIENT: InterpolationKernel.GAUSSIAN
    })


def construct_rbf_matrix_offsets_tables_for_cells(grid:base_grid.BaseGrid)->field_alloc.NDArray:
    c2e2c = grid.connectivities[dims.C2E2CDim]
    c2e = grid.connectivities[dims.C2EDim]
    # flatten this things
    offset = c2e[c2e2c]
    shp = offset.shape
    assert len(shp) == 3
    new_shape = (shp[0], shp[1] * shp[2])
    flattened_offset = offset.reshape(new_shape)
    return flattened_offset
    
def dot_product(v:np.ndarray)->np.ndarray:
    v_tilde = np.moveaxis(v, 1, -1)
    # use linalg.matmul (array API compatible)
    return np.matmul(v, v_tilde)

def arc_length(v:np.ndarray)->np.ndarray:
    norms = np.sqrt(np.sum(v * v, axis=2))
    v = v / norms[:, :, np.newaxis]
    v_tilde = np.moveaxis(v, 1, -1)

    d = np.matmul(v, v_tilde)
    return np.arccos(d)

def gaussian(lengths:np.ndarray, scale:float)->np.ndarray:
    val = lengths / scale
    return np.exp( -1.0 * val * val)

def multiquadratic(distance: np.ndarray, scale:float) -> np.ndarray:
    """

    Args:
        distance: radial distance
        scale: scaling parameter

    Returns:

    """
    val = distance * scale
    return np.sqrt(1.0 + val*val)

def kernel(kernel: InterpolationKernel, lengths:np.ndarray, scale: float ):
    return gaussian(lengths, scale) if kernel == InterpolationKernel.GAUSSIAN else multiquadratic(lengths, scale)
  

def compute_rbf_interpolation_matrix(
    cell_center_x:field_alloc.NDArray,  # fa.EdgeField[ta.wpfloat],
    cell_center_y: field_alloc.NDArray, # fa.EdgeField[ta.wpfloat],
    cell_center_z: field_alloc.NDArray, # fa.EdgeField[ta.wpfloat],
    edge_center_x:field_alloc.NDArray,  # fa.EdgeField[ta.wpfloat],
    edge_center_y: field_alloc.NDArray, # fa.EdgeField[ta.wpfloat],
    edge_center_z: field_alloc.NDArray, # fa.EdgeField[ta.wpfloat],
    edge_normal_x: field_alloc.NDArray, #fa.EdgeField[ta.wpfloat],
    edge_normal_y: field_alloc.NDArray, #fa.EdgeField[ta.wpfloat],
    edge_normal_z: field_alloc.NDArray, #fa.EdgeField[ta.wpfloat],
    rbf_offset: field_alloc.NDArray, # field_alloc.NDArray, [num_dim, RBFDimension(dim)]
    rbf_kernel: InterpolationKernel,
    scale_factor: float,
):
    ...
    # 1) get the rbf offsets - currently: input

    # compute neighbor list and create "cartesian coordinate" vectors in last dimension
    x_normal = edge_normal_x[rbf_offset]
    y_normal = edge_normal_y[rbf_offset]
    z_normal = edge_normal_z[rbf_offset]
    normal =  np.stack((x_normal, y_normal, z_normal), axis = -1)
    x_center = edge_center_x[rbf_offset]
    y_center = edge_center_y[rbf_offset]
    z_center = edge_center_z[rbf_offset]
    edge_center = np.stack((x_center, y_center, z_center), axis=-1)

    z_nxprod = dot_product(normal)
    z_dist = arc_length(edge_center)

    z_rbfmat = z_nxprod * kernel(rbf_kernel, z_dist, scale_factor)

    # 2)  apply cholesky decomposition to z_rbfmat -> z_diag


    # right hand side
    cell_centers = np.stack(cell_center_x, cell_center_y, cell_center_z, axis=-1)
    arc_length(cell_centers, )




