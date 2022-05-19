from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import (
    Field,
    float32,
    neighbor_sum,
)

from src.icon4py.dimension import KDim, EdgeDim, CellDim, C2EDim, C2E


@field_operator
def _mo_nh_diffusion_stencil_14(
    z_nabla2_e: Field[[EdgeDim, KDim], float32],
    geofac_div: Field[[CellDim, C2EDim], float32],
) -> Field[[CellDim, KDim], float32]:
    return neighbor_sum(z_nabla2_e(C2E) * geofac_div, axis=C2EDim)


@program
def mo_nh_diffusion_stencil_14(
    z_nabla2_e: Field[[EdgeDim, KDim], float32],
    geofac_div: Field[[CellDim, C2EDim], float32],
    out: Field[[CellDim, KDim], float32],
):
    _mo_nh_diffusion_stencil_14(z_nabla2_e, geofac_div, out=out)
