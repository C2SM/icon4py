import numpy as np

from icon4py.common.dimension import KDim, EdgeDim, C2EDim, CellDim
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_09 import (
    mo_velocity_advection_stencil_09,
)

from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_velocity_advection_stencil_09_numpy(
    c2e: np.array, z_w_concorr_me: np.array, e_bln_c_s: np.array
):
    e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
    z_w_concorr_mc = np.sum(z_w_concorr_me[c2e] * e_bln_c_s, axis=1)
    return z_w_concorr_mc


def test_mo_velocity_advection_stencil_09():
    mesh = SimpleMesh()

    z_w_concorr_me = random_field(mesh, EdgeDim, KDim)
    e_bln_c_s = random_field(mesh, CellDim, C2EDim)
    z_w_concorr_mc = zero_field(mesh, CellDim, KDim)

    ref = mo_velocity_advection_stencil_09_numpy(
        mesh.c2e,
        np.asarray(z_w_concorr_me),
        np.asarray(e_bln_c_s),
    )
    mo_velocity_advection_stencil_09(
        z_w_concorr_me,
        e_bln_c_s,
        z_w_concorr_mc,
        offset_provider={"C2E": mesh.get_c2e_offset_provider()},
    )
    assert np.allclose(z_w_concorr_mc, ref)
