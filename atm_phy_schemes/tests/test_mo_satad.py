# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from icon4py.atm_phy_schemes.mo_satad import satad
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


# TODO: zero_field is used when a field is defined within a field_operator, if fields are re-assigned different values then there is no need to import that
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
