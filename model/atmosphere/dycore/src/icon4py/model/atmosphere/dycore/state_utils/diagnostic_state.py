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

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim


@dataclass
class DiagnosticState:
    # fields for 3D elements in turbdiff
    hdef_ic: Field[
        [CellDim, KDim], float
    ]  # ! divergence at half levels(nproma,nlevp1,nblks_c)     [1/s]
    div_ic: Field[
        [CellDim, KDim], float
    ]  # ! horizontal wind field deformation (nproma,nlevp1,nblks_c)     [1/s^2]
    dwdx: Field[
        [CellDim, KDim], float
    ]  # zonal gradient of vertical wind speed (nproma,nlevp1,nblks_c)     [1/s]

    dwdy: Field[
        [CellDim, KDim], float
    ]  # meridional gradient of vertical wind speed (nproma,nlevp1,nblks_c)

    vt: Field[[EdgeDim, KDim], float]
    vn_ie: Field[
        [EdgeDim, KDim], float
    ]  # normal wind at half levels (nproma,nlevp1,nblks_e)   [m/s] # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    w_concorr_c: Field[
        [CellDim, KDim], float
    ]  # contravariant vert correction (nproma,nlevp1,nblks_c)[m/s] # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    ddt_w_adv_pc: Field[[CellDim, KDim], float]
    ddt_vn_apc_pc: Field[[EdgeDim, KDim], float]
    ntnd: float


@dataclass
class DiagnosticStateNonHydro:
    vt: Field[[EdgeDim, KDim], float]
    vn_ie: Field[
        [EdgeDim, KDim], float
    ]  # normal wind at half levels (nproma,nlevp1,nblks_e)   [m/s] # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    w_concorr_c: Field[
        [CellDim, KDim], float
    ]  # contravariant vert correction (nproma,nlevp1,nblks_c)[m/s] # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    theta_v_ic: Field[[CellDim, KDim], float]
    exner_pr: Field[[CellDim, KDim], float]
    rho_ic: Field[[CellDim, KDim], float]
    ddt_exner_phy: Field[[CellDim, KDim], float]
    grf_tend_rho: Field[[CellDim, KDim], float]
    grf_tend_thv: Field[[CellDim, KDim], float]
    grf_tend_w: Field[[CellDim, KDim], float]
    mass_fl_e: Field[[EdgeDim, KDim], float]
    ddt_vn_phy: Field[[EdgeDim, KDim], float]
    grf_tend_vn: Field[[EdgeDim, KDim], float]
    ddt_vn_apc_ntl1: Field[[EdgeDim, KDim], float]
    ddt_vn_apc_ntl2: Field[[EdgeDim, KDim], float]
    ddt_w_adv_ntl1: Field[[CellDim, KDim], float]
    ddt_w_adv_ntl2: Field[[CellDim, KDim], float]

    # Analysis increments
    rho_incr: Field[[EdgeDim, KDim], float]  # moist density increment [kg/m^3]
    vn_incr: Field[[EdgeDim, KDim], float]  # normal velocity increment [m/s]
    exner_incr: Field[[EdgeDim, KDim], float]  # exner increment [- ]

    @property
    def ddt_vn_apc_pc(
        self,
    ) -> tuple[Field[[EdgeDim, KDim], float], Field[[EdgeDim, KDim], float]]:
        return (self.ddt_vn_apc_ntl1, self.ddt_vn_apc_ntl2)

    @property
    def ddt_w_adv_pc(
        self,
    ) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
        return (self.ddt_w_adv_ntl1, self.ddt_w_adv_ntl2)
