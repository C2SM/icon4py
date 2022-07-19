from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field
from icon4py.common.dimension import CellDim, KDim

from icon4py.atm_phy_schemes.mo_satad import satad


def test_mo_satad():
    mesh = SimpleMesh()

    qv = random_field(mesh, CellDim, KDim)
    qc = random_field(mesh, CellDim, KDim)
    t = random_field(mesh, CellDim, KDim)
    rho = random_field(mesh, CellDim, KDim)

    satad(
        qv,
        qc,
        t,
        rho,
        offset_provider={},
    )
