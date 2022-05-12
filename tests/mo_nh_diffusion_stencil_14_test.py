import numpy as np

from src.mo_nh_diffusion_stencil_14 import (
    mo_nh_diffusion_stencil_z_temp_numpy,
    mo_nh_diffusion_stencil_z_temp_gt4py,
)
from .simple_mesh import cell_to_edge_table, n_edges


def test_mo_nh_diffusion_stencil_z_temp_equality():
    z_nabla2_e = np.random.randn(n_edges)
    geofac_div = np.random.randn(*cell_to_edge_table.shape)

    ref = mo_nh_diffusion_stencil_z_temp_numpy(
        cell_to_edge_table, z_nabla2_e, geofac_div
    )
    out = np.asarray(
        mo_nh_diffusion_stencil_z_temp_gt4py(cell_to_edge_table, z_nabla2_e, geofac_div)
    )
    assert all(np.isclose(out, ref))
