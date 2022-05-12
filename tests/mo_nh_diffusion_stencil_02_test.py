import numpy as np

from src.mo_nh_diffusion_stencil_02 import (
    mo_nh_diffusion_stencil_div_numpy,
    mo_nh_diffusion_stencil_div_gt4py,
    mo_nh_diffusion_stencil_kh_c_numpy,
    mo_nh_diffusion_stencil_kh_c_gt4py
)
from .simple_mesh import cell_to_edge_table, n_edges, n_cells


def test_mo_nh_diffusion_stencil_div_equality():
    vn = np.random.randn(n_edges)
    geofac_div = np.random.randn(*cell_to_edge_table.shape)

    ref = mo_nh_diffusion_stencil_div_numpy(cell_to_edge_table, vn, geofac_div)
    out = np.asarray(mo_nh_diffusion_stencil_div_gt4py(cell_to_edge_table, vn, geofac_div))
    assert all(np.isclose(out, ref))


def test_mo_nh_diffusion_stencil_kh_c_equality():
    scalar_value = 2
    c2e_shape = cell_to_edge_table.shape
    kh_smag_ec = np.random.randn(n_edges)
    e_bln_c_s = np.random.randn(*c2e_shape)
    diff_multfac_smag = np.asarray([scalar_value]*c2e_shape[0])

    ref = mo_nh_diffusion_stencil_kh_c_numpy(cell_to_edge_table, kh_smag_ec, e_bln_c_s, diff_multfac_smag)
    out = np.asarray(mo_nh_diffusion_stencil_kh_c_gt4py(cell_to_edge_table, kh_smag_ec, e_bln_c_s, diff_multfac_smag))
    assert all(np.isclose(out, ref))