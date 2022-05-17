import numpy as np

from src.icon4py.dimension import KDim, CellDim
from src.icon4py.stencil.mo_nh_diffusion_stencil_16 import mo_nh_diffusion_stencil_16
from .simple_mesh import SimpleMesh
from .utils import random_field, zero_field, get_cell_to_k_table


def mo_nh_diffusion_stencil_16_numpy(
    rd_o_cvd: float,
    z_temp: np.array,
    area: np.array,
    theta_v: np.array,
    exner: np.array,
) -> np.array:
    z_theta = theta_v
    theta_v = theta_v + (np.expand_dims(area, axis=-1) * z_temp)
    return exner * (1.0 + rd_o_cvd * (theta_v / z_theta - 1.0))


def test_mo_nh_diffusion_stencil_16():
    mesh = SimpleMesh()

    rd_o_cvd = np.float32(np.random.random_sample())
    z_temp = random_field(mesh, CellDim, KDim)
    area = random_field(mesh, CellDim)
    theta_v = random_field(mesh, CellDim, KDim)
    exner = random_field(mesh, CellDim, KDim)
    out = zero_field(mesh, CellDim, KDim)

    c2k = get_cell_to_k_table(area, mesh.k_level)

    ref = mo_nh_diffusion_stencil_16_numpy(
        rd_o_cvd,
        np.asarray(z_temp),
        np.asarray(area),
        np.asarray(theta_v),
        np.asarray(exner),
    )
    mo_nh_diffusion_stencil_16(
        rd_o_cvd,
        z_temp,
        area,
        theta_v,
        exner,
        out,
        offset_provider={"C2K": mesh.get_c2k_offset_provider(c2k)},
    )
    assert np.allclose(out, ref)
