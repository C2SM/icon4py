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

from gt4py.next.common import Field

from icon4py.model.atmosphere.dycore.state_utils.utils import _allocate
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.grid.base import BaseGrid


@dataclass
class ZFields:
    z_gradh_exner: Field[[EdgeDim, KDim], float]
    z_alpha: Field[
        [EdgeDim, KDim], float
    ]  # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    z_beta: Field[[CellDim, KDim], float]
    z_w_expl: Field[
        [EdgeDim, KDim], float
    ]  # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    z_exner_expl: Field[[CellDim, KDim], float]
    z_q: Field[[CellDim, KDim], float]
    z_contr_w_fl_l: Field[
        [EdgeDim, KDim], float
    ]  # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    z_rho_e: Field[[EdgeDim, KDim], float]
    z_theta_v_e: Field[[EdgeDim, KDim], float]
    z_kin_hor_e: Field[[EdgeDim, KDim], float]
    z_vt_ie: Field[[EdgeDim, KDim], float]
    z_graddiv_vn: Field[[EdgeDim, KDim], float]
    z_rho_expl: Field[[CellDim, KDim], float]
    z_dwdz_dd: Field[[CellDim, KDim], float]


def _allocate_z_fields(grid: BaseGrid):
    return ZFields(
        z_gradh_exner=_allocate(EdgeDim, KDim, grid=grid),
        z_alpha=_allocate(CellDim, KDim, is_halfdim=True, grid=grid),
        z_beta=_allocate(
            CellDim, KDim, is_halfdim=True, grid=grid
        ),  # TODO (@halungge) overallocated with fake halfdim
        z_w_expl=_allocate(CellDim, KDim, is_halfdim=True, grid=grid),
        z_exner_expl=_allocate(CellDim, KDim, is_halfdim=True, grid=grid),
        z_q=_allocate(
            CellDim, KDim, is_halfdim=True, grid=grid
        ),  # TODO (@halungge) overallocated with fake halfdim
        z_contr_w_fl_l=_allocate(CellDim, KDim, is_halfdim=True, grid=grid),
        z_rho_e=_allocate(EdgeDim, KDim, grid=grid),
        z_theta_v_e=_allocate(EdgeDim, KDim, grid=grid),
        z_graddiv_vn=_allocate(EdgeDim, KDim, grid=grid),
        z_rho_expl=_allocate(
            CellDim, KDim, is_halfdim=True, grid=grid
        ),  # TODO (@halungge) overallocated with fake halfdim
        z_dwdz_dd=_allocate(CellDim, KDim, grid=grid),
        z_kin_hor_e=_allocate(
            EdgeDim, KDim, is_halfdim=True, grid=grid
        ),  # TODO (@halungge) overallocated with fake halfdim
        z_vt_ie=_allocate(
            EdgeDim, KDim, is_halfdim=True, grid=grid
        ),  # TODO (@halungge) overallocated with fake halfdim
    )
