import numpy as np
from functional.iterator.embedded import np_as_located_field

from src.icon4py.dimension import KDim, EdgeDim, CellDim, C2EDim
from src.icon4py.stencil.mo_nh_diffusion_stencil_02 import (
    mo_nh_diffusion_stencil_02_div,
    mo_nh_diffusion_stencil_02_khc,
)
from src.icon4py.utils import add_kdim, get_cell_to_k_table
from .simple_mesh import SimpleMesh


def mo_nh_diffusion_stencil_02_div_numpy(
    c2e: np.array,
    vn: np.array,
    geofac_div: np.array,
) -> np.array:
    geofac_div = np.expand_dims(geofac_div, axis=-1)
    vn_geofac = vn[c2e] * geofac_div
    div = np.sum(vn_geofac, axis=1)  # sum along edge dimension
    return div


def mo_nh_diffusion_stencil_02_khc_numpy(
    c2e: np.array,
    kh_smag_ec: np.array,
    e_bln_c_s: np.array,
    diff_multfac_smag: np.array,
) -> np.array:
    e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
    diff_multfac_smag = np.expand_dims(diff_multfac_smag, axis=-1)
    mul = kh_smag_ec[c2e] * e_bln_c_s
    summed = np.sum(mul, axis=1)  # sum along edge dimension
    kh_c = summed / diff_multfac_smag
    return kh_c


def test_mo_nh_diffusion_stencil_02_div():
    mesh = SimpleMesh()

    vn = add_kdim(np.random.randn(mesh.n_edges), mesh.k_level)
    geofac_div = np.random.randn(mesh.n_cells, mesh.n_c2e)
    out_arr = add_kdim(np.zeros(shape=(mesh.n_cells,)), mesh.k_level)

    vn_field = np_as_located_field(EdgeDim, KDim)(vn)
    geofac_div_field = np_as_located_field(CellDim, C2EDim)(geofac_div)
    out_field = np_as_located_field(CellDim, KDim)(out_arr)

    ref = mo_nh_diffusion_stencil_02_div_numpy(mesh.c2e, vn, geofac_div)
    mo_nh_diffusion_stencil_02_div(
        vn_field,
        geofac_div_field,
        out_field,
        offset_provider={"C2E": mesh.get_c2e_offset_provider()},
    )
    assert np.allclose(out_field, ref)


def test_mo_nh_diffusion_stencil_02_khc():
    mesh = SimpleMesh()

    kh_smag_ec = add_kdim(np.random.randn(mesh.n_edges), mesh.k_level)
    e_bln_c_s = np.random.randn(mesh.n_cells, mesh.n_c2e)
    diff_multfac_smag = np.asarray([mesh.k_level] * mesh.n_cells)
    c2k = get_cell_to_k_table(diff_multfac_smag, mesh.k_level)
    out_arr = add_kdim(np.zeros(shape=(mesh.n_cells,)), mesh.k_level)

    kh_smag_ec_field = np_as_located_field(EdgeDim, KDim)(kh_smag_ec)
    e_bln_c_s_field = np_as_located_field(CellDim, C2EDim)(e_bln_c_s)
    diff_multfac_smag_field = np_as_located_field(KDim)(diff_multfac_smag)
    out_field = np_as_located_field(CellDim, KDim)(out_arr)

    ref = mo_nh_diffusion_stencil_02_khc_numpy(
        mesh.c2e, kh_smag_ec, e_bln_c_s, diff_multfac_smag
    )
    mo_nh_diffusion_stencil_02_khc(
        kh_smag_ec_field,
        e_bln_c_s_field,
        diff_multfac_smag_field,
        out_field,
        offset_provider={
            "C2E": mesh.get_c2e_offset_provider(),
            "C2K": mesh.get_c2k_offset_provider(c2k),
        },
    )

    assert np.allclose(out_field, ref)
