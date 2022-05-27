import numpy as np

from icon4py.common.dimension import KDim, EdgeDim, E2C2EDim
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_01 import (
    mo_velocity_advection_stencil_01,
)

from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field

def mo_velocity_advection_stencil_01_numpy(
    e2c2e: np.array, vn: np.array, rbf_vec_coeff_e: np.array
):
    rbf_vec_coeff_e = np.expand_dims(rbf_vec_coeff_e, axis=-1)
    vt = np.sum(vn[e2c2e] * rbf_vec_coeff_e, axis=1)
    return vt


def test_mo_velocity_advection_stencil_01():
    mesh = SimpleMesh()

    vn = random_field(mesh, EdgeDim, KDim)
    rbf_vec_coeff = random_field(mesh, EdgeDim, E2C2EDim)
    vt = zero_field(mesh, EdgeDim, KDim)

    ref = mo_velocity_advection_stencil_01_numpy(
        mesh.e2c2e,
        np.asarray(vn),
        np.asarray(rbf_vec_coeff),
    )
    mo_velocity_advection_stencil_01(
        vn,
        rbf_vec_coeff,
        vt,
        offset_provider={"E2C2E": mesh.get_e2c2e_offset_provider()},
    )
    assert np.allclose(vt, ref)
