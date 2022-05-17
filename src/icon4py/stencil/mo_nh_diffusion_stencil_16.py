from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, float32
from src.icon4py.dimension import KDim, CellDim, C2K


@field_operator
def _mo_nh_diffusion_stencil_16(
    rd_o_cvd: float32,
    z_temp: Field[[CellDim, KDim], float32],
    area: Field[[KDim], float32],
    theta_v: Field[[CellDim, KDim], float32],
    exner: Field[[CellDim, KDim], float32],
) -> Field[[CellDim, KDim], float32]:
    z_theta = theta_v
    theta_v = theta_v + (area(C2K) * z_temp)
    return exner * (float32(1.0) + rd_o_cvd * (theta_v / z_theta - float32(1.0)))


@program
def mo_nh_diffusion_stencil_16(
    rd_o_cvd: float32,
    z_temp: Field[[CellDim, KDim], float32],
    area: Field[[KDim], float32],
    theta_v: Field[[CellDim, KDim], float32],
    exner: Field[[CellDim, KDim], float32],
    out: Field[[CellDim, KDim], float32],
):
    _mo_nh_diffusion_stencil_16(rd_o_cvd, z_temp, area, theta_v, exner, out=out)
