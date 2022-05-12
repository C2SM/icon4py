import numpy as np

from src.mo_nh_diffusion_stencil_02 import (
    mo_nh_diffusion_stencil_div_numpy,
    mo_nh_diffusion_stencil_div_gt4py,
)
from .simple_mesh import cell_to_edge_table, n_edges


def test_mo_nh_diffusion_stencil_div_equality():
    vn = np.random.randn(n_edges)
    geofac_div = np.random.randn(*cell_to_edge_table.shape)

    ref = mo_nh_diffusion_stencil_div_numpy(cell_to_edge_table, vn, geofac_div)
    out = mo_nh_diffusion_stencil_div_gt4py(cell_to_edge_table, vn, geofac_div)
    assert all(np.isclose(np.asarray(out), ref))


def test_mo_nh_diffusion_stencil_div_equality():
    # TODO
    pass
