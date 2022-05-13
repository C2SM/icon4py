import numpy as np

from src.icon4py.stencil.mo_nh_diffusion_stencil_14 import (
    mo_nh_diffusion_stencil_z_temp_numpy,
    mo_nh_diffusion_stencil_z_temp_gt4py,
)
from src.icon4py.utils import add_kdim
from .simple_mesh import cell_to_edge_table, n_edges

k = 10
c2e_shape = cell_to_edge_table.shape


def test_mo_nh_diffusion_stencil_z_temp_equality():
    z_nabla2_e = np.random.randn(n_edges)
    geofac_div = np.random.randn(*c2e_shape)
    out_arr = np.zeros(shape=(c2e_shape[0],))

    z_nabla2_e = add_kdim(z_nabla2_e, k)
    out_arr = add_kdim(out_arr, k)

    ref = mo_nh_diffusion_stencil_z_temp_numpy(
        cell_to_edge_table, z_nabla2_e, geofac_div
    )
    out = np.asarray(
        mo_nh_diffusion_stencil_z_temp_gt4py(
            cell_to_edge_table, z_nabla2_e, geofac_div, out_arr
        )
    )
    truth_arr = np.isclose(out, ref).flatten()
    assert all(truth_arr)
