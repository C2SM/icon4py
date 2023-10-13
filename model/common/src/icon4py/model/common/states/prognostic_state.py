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

import numpy as np
from gt4py.next.common import Field
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.iterator.embedded import np_as_located_field

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim


@dataclass
class PrognosticState:
    """Class that contains the prognostic state.

    Corresponds to ICON t_nh_prog
    """

    rho: Field[[CellDim, KDim], float]  # density, rho(nproma, nlev, nblks_c) [m/s]
    w: Field[[CellDim, KDim], float]  # vertical_wind field, w(nproma, nlevp1, nblks_c) [m/s]
    vn: Field[
        [EdgeDim, KDim], float
    ]  # horizontal wind normal to edges, vn(nproma, nlev, nblks_e)  [m/s]
    exner: Field[[CellDim, KDim], float]  # exner function, exner(nrpoma, nlev, nblks_c)
    theta_v: Field[[CellDim, KDim], float]  # virtual temperature, (nproma, nlev, nlbks_c) [K]

    @property
    def w_1(self) -> Field[[CellDim], float]:
        return np_as_located_field(CellDim)(np.asarray(self.w)[:, 0])


@field_operator
def copy_field_celldim_kdim(field: Field[[CellDim, KDim], float]) -> Field[[CellDim, KDim], float]:
    return field

@field_operator
def copy_field_edgedim_kdim(field: Field[[EdgeDim, KDim], float]) -> Field[[EdgeDim, KDim], float]:
    return field


@program
def copy_prognostics(
    rho: Field[[CellDim, KDim], float],
    rho_new: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    w_new: Field[[CellDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    vn_new: Field[[EdgeDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    exner_new: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    theta_v_new: Field[[CellDim, KDim], float]
):
    copy_field_celldim_kdim(rho, out=rho_new)
    copy_field_celldim_kdim(w, out=w_new)
    copy_field_edgedim_kdim(vn, out=vn_new)
    copy_field_celldim_kdim(exner, out=exner_new)
    copy_field_celldim_kdim(theta_v, out=theta_v_new)


