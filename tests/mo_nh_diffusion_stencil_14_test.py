import numpy as np
from functional.iterator.embedded import (
    np_as_located_field,
)

from src.icon4py.dimension import KDim, EdgeDim, CellDim, C2EDim
from src.icon4py.stencil.mo_nh_diffusion_stencil_14 import mo_nh_diffusion_stencil_14
from src.icon4py.utils import add_kdim
from .simple_mesh import SimpleMesh


def mo_nh_diffusion_stencil_14_numpy(
    c2e: np.array, z_nabla2_e: np.array, geofac_div: np.array
) -> np.array:
    geofac_div = np.expand_dims(geofac_div, axis=-1)
    z_temp = np.sum(z_nabla2_e[c2e] * geofac_div, axis=1)  # sum along edge dimension
    return z_temp


def test_mo_nh_diffusion_stencil_14():
    mesh = SimpleMesh()

    z_nabla2_e = add_kdim(np.random.randn(mesh.n_edges), mesh.k_level)
    geofac_div = np.random.randn(mesh.n_cells, mesh.n_c2e)
    out_arr = add_kdim(np.zeros(shape=(mesh.n_cells,)), mesh.k_level)

    z_nabla2_e_field = np_as_located_field(EdgeDim, KDim)(z_nabla2_e)
    geofac_div_field = np_as_located_field(CellDim, C2EDim)(geofac_div)
    out_field = np_as_located_field(CellDim, KDim)(out_arr)

    ref = mo_nh_diffusion_stencil_14_numpy(mesh.c2e, z_nabla2_e, geofac_div)
    mo_nh_diffusion_stencil_14(
        z_nabla2_e_field,
        geofac_div_field,
        out_field,
        offset_provider={"C2E": mesh.get_c2e_offset_provider()},
    )
    assert np.allclose(out_field, ref)
