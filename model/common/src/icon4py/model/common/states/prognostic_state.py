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

from dataclasses import dataclass

from gt4py.next import as_field
from gt4py.next.common import Field
from model.common.tests import field_type_aliases as fa

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim


@dataclass
class PrognosticState:
    """Class that contains the prognostic state.

    Corresponds to ICON t_nh_prog
    """

    rho: fa.CKfloatField  # density, rho(nproma, nlev, nblks_c) [m/s]
    w: fa.CKfloatField  # vertical_wind field, w(nproma, nlevp1, nblks_c) [m/s]
    vn: Field[
        [EdgeDim, KDim], float
    ]  # horizontal wind normal to edges, vn(nproma, nlev, nblks_e)  [m/s]
    exner: fa.CKfloatField  # exner function, exner(nrpoma, nlev, nblks_c)
    theta_v: fa.CKfloatField  # virtual temperature, (nproma, nlev, nlbks_c) [K]

    @property
    def w_1(self) -> Field[[CellDim], float]:
        return as_field((CellDim,), self.w.ndarray[:, 0])
