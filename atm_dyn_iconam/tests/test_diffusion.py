import numpy as np

from icon4py.atm_dyn_iconam.diffusion import init_diff_multfac_vn
from icon4py.common.dimension import KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import zero_field


def test_init_diff_multifac_vn_const():
    mesh = SimpleMesh()
    diff_multfac_vn = zero_field(mesh, KDim)

    k4 = 1.0
    substeps = 5.0
    init_diff_multfac_vn(k4, substeps, diff_multfac_vn, offset_provider={})
    assert np.allclose(1.0/128.0 * np.ones(np.asarray(diff_multfac_vn).shape), diff_multfac_vn)

def test_init_diff_multifac_vn_k4_substeps():
    mesh = SimpleMesh()
    diff_multfac_vn = zero_field(mesh, KDim)

    k4 = 0.003
    substeps = 1.0
    init_diff_multfac_vn(k4, substeps, diff_multfac_vn, offset_provider={})
    assert np.allclose(k4*substeps / 3.0 * np.ones(np.asarray(diff_multfac_vn).shape), diff_multfac_vn)
