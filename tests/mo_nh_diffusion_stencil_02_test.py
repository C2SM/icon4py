import numpy as np
from functional.iterator.embedded import (
    np_as_located_field,
    NeighborTableOffsetProvider,
)

from src.icon4py.dimension import KDim, EdgeDim, CellDim, C2EDim
from src.icon4py.stencil.mo_nh_diffusion_stencil_02 import (
    mo_nh_diffusion_stencil_02_div,
    mo_nh_diffusion_stencil_02_khc,
)
from src.icon4py.utils import add_kdim, get_cell_to_k_table
from .simple_mesh import cell_to_edge_table, n_edges

K_VALUE = 10
C2E_SHAPE = cell_to_edge_table.shape


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


def test_mo_nh_diffusion_stencil_02_div(utils):
    vn = add_kdim(np.random.randn(n_edges), K_VALUE)
    geofac_div = np.random.randn(*C2E_SHAPE)
    out_arr = add_kdim(np.zeros(shape=(C2E_SHAPE[0],)), K_VALUE)

    vn_field = np_as_located_field(EdgeDim, KDim)(vn)
    geofac_div_field = np_as_located_field(CellDim, C2EDim)(geofac_div)
    out_field = np_as_located_field(CellDim, KDim)(out_arr)

    C2E_offset_provider = NeighborTableOffsetProvider(
        cell_to_edge_table, CellDim, EdgeDim, 3
    )

    ref = mo_nh_diffusion_stencil_02_div_numpy(cell_to_edge_table, vn, geofac_div)
    mo_nh_diffusion_stencil_02_div(
        vn_field,
        geofac_div_field,
        out_field,
        offset_provider={"C2E": C2E_offset_provider},
    )
    utils.assert_equality(out_field, ref)


def test_mo_nh_diffusion_stencil_02_khc(utils):
    kh_smag_ec = add_kdim(np.random.randn(n_edges), K_VALUE)
    e_bln_c_s = np.random.randn(*C2E_SHAPE)
    diff_multfac_smag = np.asarray([K_VALUE] * C2E_SHAPE[0])
    c2k = get_cell_to_k_table(diff_multfac_smag, K_VALUE)
    out_arr = add_kdim(np.zeros(shape=(C2E_SHAPE[0],)), K_VALUE)

    kh_smag_ec_field = np_as_located_field(EdgeDim, KDim)(kh_smag_ec)
    e_bln_c_s_field = np_as_located_field(CellDim, C2EDim)(e_bln_c_s)
    diff_multfac_smag_field = np_as_located_field(KDim)(diff_multfac_smag)
    out_field = np_as_located_field(CellDim, KDim)(out_arr)

    C2E_offset_provider = NeighborTableOffsetProvider(
        cell_to_edge_table, CellDim, EdgeDim, 3
    )
    C2K_offset_provider = NeighborTableOffsetProvider(c2k, CellDim, KDim, K_VALUE)

    ref = mo_nh_diffusion_stencil_02_khc_numpy(
        cell_to_edge_table, kh_smag_ec, e_bln_c_s, diff_multfac_smag
    )
    mo_nh_diffusion_stencil_02_khc(
        kh_smag_ec_field,
        e_bln_c_s_field,
        diff_multfac_smag_field,
        out_field,
        offset_provider={"C2E": C2E_offset_provider, "C2K": C2K_offset_provider},
    )

    utils.assert_equality(out_field, ref)
