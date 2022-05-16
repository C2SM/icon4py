import pytest
import numpy as np
from functional.iterator.embedded import (
    np_as_located_field,
    NeighborTableOffsetProvider,
)

from src.icon4py.dimension import KDim, EdgeDim, CellDim, C2EDim
from src.icon4py.stencil.mo_nh_diffusion_stencil_14 import mo_nh_diffusion_stencil_14
from src.icon4py.utils import add_kdim
from .simple_mesh import cell_to_edge_table, n_edges

K_LEVELS = list(range(1, 12, 3))
C2E_SHAPE = cell_to_edge_table.shape


def mo_nh_diffusion_stencil_14_numpy(
    c2e: np.array, z_nabla2_e: np.array, geofac_div: np.array
) -> np.array:
    geofac_div = np.expand_dims(geofac_div, axis=-1)
    z_temp = np.sum(z_nabla2_e[c2e] * geofac_div, axis=1)  # sum along edge dimension
    return z_temp


@pytest.mark.parametrize("k_level", K_LEVELS)
def test_mo_nh_diffusion_stencil_14(utils, k_level):
    z_nabla2_e = add_kdim(np.random.randn(n_edges), k_level)
    geofac_div = np.random.randn(*C2E_SHAPE)
    out_arr = add_kdim(np.zeros(shape=(C2E_SHAPE[0],)), k_level)

    z_nabla2_e_field = np_as_located_field(EdgeDim, KDim)(z_nabla2_e)
    geofac_div_field = np_as_located_field(CellDim, C2EDim)(geofac_div)
    out_field = np_as_located_field(CellDim, KDim)(out_arr)
    C2E_offset_provider = NeighborTableOffsetProvider(
        cell_to_edge_table, CellDim, EdgeDim, 3
    )

    ref = mo_nh_diffusion_stencil_14_numpy(cell_to_edge_table, z_nabla2_e, geofac_div)
    mo_nh_diffusion_stencil_14(
        z_nabla2_e_field,
        geofac_div_field,
        out_field,
        offset_provider={"C2E": C2E_offset_provider},
    )
    utils.assert_equality(out_field, ref)
