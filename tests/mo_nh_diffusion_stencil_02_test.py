import numpy as np

from src.stencil.mo_nh_diffusion_stencil_02 import (
    mo_nh_diffusion_stencil_div_numpy,
    mo_nh_diffusion_stencil_div_gt4py,
    mo_nh_diffusion_stencil_kh_c_numpy,
    mo_nh_diffusion_stencil_kh_c_gt4py,
)
from src.utils import add_kdim, get_cell_to_k_table
from .simple_mesh import cell_to_edge_table, n_edges

k = 10
c2e_shape = cell_to_edge_table.shape


def test_mo_nh_diffusion_stencil_div_equality():

    vn = np.random.randn(n_edges)
    geofac_div = np.random.randn(*c2e_shape)
    out_arr = np.zeros(shape=(c2e_shape[0],))

    vn = add_kdim(vn, k)
    out_arr = add_kdim(out_arr, k)

    ref = mo_nh_diffusion_stencil_div_numpy(cell_to_edge_table, vn, geofac_div)
    out = np.asarray(
        mo_nh_diffusion_stencil_div_gt4py(cell_to_edge_table, vn, geofac_div, out_arr)
    )
    truth_arr = np.isclose(out, ref).flatten()
    assert all(truth_arr)


def test_mo_nh_diffusion_stencil_kh_c_equality():

    kh_smag_ec = np.random.randn(n_edges)
    e_bln_c_s = np.random.randn(*c2e_shape)
    diff_multfac_smag = np.asarray([k] * c2e_shape[0])
    c2k = get_cell_to_k_table(diff_multfac_smag, k)
    out_arr = np.zeros(shape=(c2e_shape[0],))

    kh_smag_ec = add_kdim(kh_smag_ec, k)
    out_arr = add_kdim(out_arr, k)

    ref = mo_nh_diffusion_stencil_kh_c_numpy(
        cell_to_edge_table, kh_smag_ec, e_bln_c_s, diff_multfac_smag
    )

    out = np.asarray(
        mo_nh_diffusion_stencil_kh_c_gt4py(
            cell_to_edge_table,
            c2k,
            k,
            kh_smag_ec,
            e_bln_c_s,
            diff_multfac_smag,
            out_arr,
        )
    )
    truth_arr = np.isclose(out, ref).flatten()
    assert all(truth_arr)
