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
from numpy import int32

from icon4py.model.common.dimension import CECDim, CellDim, ECDim, EdgeDim, KDim


@dataclass
class MetricState:
    mask_hdiff: Field[[CellDim, KDim], bool]
    theta_ref_mc: Field[[CellDim, KDim], float]
    wgtfac_c: Field[
        [CellDim, KDim], float
    ]  # weighting factor for interpolation from full to half levels (nproma,nlevp1,nblks_c)
    zd_vertidx: Field[[CECDim, KDim], int32]
    zd_diffcoef: Field[[CellDim, KDim], float]
    zd_intcoef: Field[[CECDim, KDim], float]

    coeff_gradekin: Field[[ECDim], float]
    ddqz_z_full_e: Field[[EdgeDim, KDim], float]
    wgtfac_e: Field[[EdgeDim, KDim], float]
    wgtfacq_e_dsl: Field[[EdgeDim, KDim], float]
    ddxn_z_full: Field[[EdgeDim, KDim], float]
    ddxt_z_full: Field[[EdgeDim, KDim], float]
    ddqz_z_half: Field[[CellDim, KDim], float]  # half KDim ?
    coeff1_dwdz: Field[[CellDim, KDim], float]
    coeff2_dwdz: Field[[CellDim, KDim], float]
    zd_vertoffset: Field[[CECDim, KDim], int32]


@dataclass
class MetricStateNonHydro:
    bdy_halo_c: Field[[CellDim], bool]
    # Finally, a mask field that excludes boundary halo points
    mask_prog_halo_c: Field[[CellDim, KDim], bool]
    rayleigh_w: Field[[KDim], float]

    wgtfac_c: Field[[CellDim, KDim], float]
    wgtfacq_c_dsl: Field[[CellDim, KDim], float]
    wgtfac_e: Field[[EdgeDim, KDim], float]
    wgtfacq_e_dsl: Field[[EdgeDim, KDim], float]

    exner_exfac: Field[[CellDim, KDim], float]
    exner_ref_mc: Field[[CellDim, KDim], float]
    rho_ref_mc: Field[[CellDim, KDim], float]
    theta_ref_mc: Field[[CellDim, KDim], float]
    rho_ref_me: Field[[EdgeDim, KDim], float]
    theta_ref_me: Field[[EdgeDim, KDim], float]
    theta_ref_ic: Field[[CellDim, KDim], float]

    d_exner_dz_ref_ic: Field[[CellDim, KDim], float]
    ddqz_z_half: Field[[CellDim, KDim], float]  # half KDim ?
    d2dexdz2_fac1_mc: Field[[CellDim, KDim], float]
    d2dexdz2_fac2_mc: Field[[CellDim, KDim], float]
    ddxn_z_full: Field[[EdgeDim, KDim], float]
    ddqz_z_full_e: Field[[EdgeDim, KDim], float]
    ddxt_z_full: Field[[EdgeDim, KDim], float]
    inv_ddqz_z_full: Field[[CellDim, KDim], float]

    vertoffset_gradp: Field[[ECDim, KDim], float]
    zdiff_gradp: Field[[ECDim, KDim], float]
    ipeidx_dsl: Field[[EdgeDim, KDim], bool]
    pg_exdist: Field[[EdgeDim, KDim], float]

    vwind_expl_wgt: Field[[CellDim], float]
    vwind_impl_wgt: Field[[CellDim], float]

    hmask_dd3d: Field[[EdgeDim], float]
    scalfac_dd3d: Field[[KDim], float]

    coeff1_dwdz: Field[[CellDim, KDim], float]
    coeff2_dwdz: Field[[CellDim, KDim], float]
    coeff_gradekin: Field[[ECDim], float]
