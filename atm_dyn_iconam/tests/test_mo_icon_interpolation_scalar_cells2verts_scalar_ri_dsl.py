import numpy as np

from icon4py.common.dimension import KDim, CellDim, VertexDim, V2CDim
from icon4py.atm_dyn_iconam.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy(
    v2c: np.array, p_cell_in: np.array, c_intp: np.array
) -> np.array:
    c_intp = np.expand_dims(c_intp, axis=-1)
    p_vert_out = np.sum(p_cell_in[v2c] * c_intp, axis=1)
    return p_vert_out


def test_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl():
    mesh = SimpleMesh()

    p_cell_in = random_field(mesh, CellDim, KDim)
    c_intp = random_field(mesh, VertexDim, V2CDim)
    p_vert_out = zero_field(mesh, VertexDim, KDim)

    ref = mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy(
        mesh.v2c, np.asarray(p_cell_in), np.asarray(c_intp)
    )
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
        p_cell_in,
        c_intp,
        p_vert_out,
        offset_provider={"V2C": mesh.get_v2c_offset_provider()},
    )

    assert np.allclose(p_vert_out, ref)
