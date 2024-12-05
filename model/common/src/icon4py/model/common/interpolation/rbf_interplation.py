import dataclasses
import enum
from types import MappingProxyType

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
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
    


def compute_rbf_interpolation_matrix(
    edge_center_x:field_alloc.NDArray,  # fa.EdgeField[ta.wpfloat],
    edge_center_y: fa.EdgeField[ta.wpfloat],
    edge_center_z: fa.EdgeField[ta.wpfloat],
    edge_normal_x: fa.EdgeField[ta.wpfloat],
    edge_normal_y: fa.EdgeField[ta.wpfloat],
    edge_normal_z: fa.EdgeField[ta.wpfloat],
    rbf_offset: field_alloc.NDArray,
):
    ...
    # get the rbf offsets
    # compute dot product
    # compute arc_length on cells
    # compute gauss for matrix (or the other kernel)


