import numpy as np

from src.icon4py.dimension import KDim, EdgeDim, CellDim, C2EDim
from src.icon4py.stencil.mo_nh_diffusion_stencil_14 import mo_nh_diffusion_stencil_14
from .simple_mesh import SimpleMesh
from .utils import random_field, zero_field


def mo_nh_diffusion_stencil_14_numpy(
    c2e: np.array, z_nabla2_e: np.array, geofac_div: np.array
) -> np.array:
    geofac_div = np.expand_dims(geofac_div, axis=-1)
    z_temp = np.sum(z_nabla2_e[c2e] * geofac_div, axis=1)  # sum along edge dimension
    return z_temp


def test_mo_nh_diffusion_stencil_14():
    mesh = SimpleMesh()

    z_nabla2_e = random_field(mesh, EdgeDim, KDim)
    geofac_div = random_field(mesh, CellDim, C2EDim)
    out = zero_field(mesh, CellDim, KDim)

    ref = mo_nh_diffusion_stencil_14_numpy(
        mesh.c2e, np.asarray(z_nabla2_e), np.asarray(geofac_div)
    )
    mo_nh_diffusion_stencil_14(
        z_nabla2_e,
        geofac_div,
        out,
        offset_provider={"C2E": mesh.get_c2e_offset_provider()},
    )
    assert np.allclose(out, ref)
