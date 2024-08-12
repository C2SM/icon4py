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

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


@dataclass
class PrognosticState:
    """Class that contains the prognostic state.

    Corresponds to ICON t_nh_prog
    """

    rho: fa.CellKField[ta.wpfloat]  # density, rho(nproma, nlev, nblks_c) [kg/m^3]
    w: fa.CellKField[ta.wpfloat]  # vertical_wind field, w(nproma, nlevp1, nblks_c) [m/s]
    vn: fa.EdgeKField[
        ta.wpfloat
    ]  # horizontal wind normal to edges, vn(nproma, nlev, nblks_e)  [m/s]
    exner: fa.CellKField[ta.wpfloat]  # exner function, exner(nrpoma, nlev, nblks_c)
    theta_v: fa.CellKField[ta.wpfloat]  # virtual temperature, (nproma, nlev, nlbks_c) [K]

    @property
    def w_1(self) -> fa.CellField[ta.wpfloat]:
        return as_field((dims.CellDim,), self.w.ndarray[:, 0])
