import numpy as np

from icon4py.common.dimension import KDim, EdgeDim
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_04 import (
    mo_velocity_advection_stencil_04,
)

from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_velocity_advection_stencil_04_numpy(
    vn: np.array, ddxn_z_full: np.array, ddxt_z_full: np.array, vt: np.array
):
    z_w_concorr_me = vn * ddxn_z_full + vt * ddxt_z_full
    return z_w_concorr_me


def test_mo_velocity_advection_stencil_04():
    mesh = SimpleMesh()

    vn = random_field(mesh, EdgeDim, KDim)
    ddxn_z_full = random_field(mesh, EdgeDim, KDim)
    ddxt_z_full = random_field(mesh, EdgeDim, KDim)
    vt = random_field(mesh, EdgeDim, KDim)
    z_w_concorr_me = zero_field(mesh, EdgeDim, KDim)

    ref = mo_velocity_advection_stencil_04_numpy(
        np.asarray(vn),
        np.asarray(ddxn_z_full),
        np.asarray(ddxt_z_full),
        np.asarray(vt),
    )
    mo_velocity_advection_stencil_04(
        vn, ddxn_z_full, ddxt_z_full, vt, z_w_concorr_me, offset_provider={}
    )
    assert np.allclose(z_w_concorr_me, ref)
