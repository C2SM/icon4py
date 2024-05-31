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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32
from model.common.tests import field_type_aliases as fa

from icon4py.model.atmosphere.dycore.interpolate_vn_to_ie_and_compute_ekin_on_edges import (
    _interpolate_vn_to_ie_and_compute_ekin_on_edges,
)
from icon4py.model.atmosphere.dycore.interpolate_vt_to_interface_edges import (
    _interpolate_vt_to_interface_edges,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.settings import backend


@field_operator
def _interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges(
    wgtfac_e: fa.EKvpField,
    vn: fa.EKwpField,
    vt: fa.EKvpField,
) -> tuple[
    fa.EKvpField,
    fa.EKvpField,
    fa.EKvpField,
]:
    """Formerly known as _mo_solve_nonhydro_stencil_36."""
    z_vt_ie = _interpolate_vt_to_interface_edges(wgtfac_e=wgtfac_e, vt=vt)
    vn_ie, z_kin_hor_e = _interpolate_vn_to_ie_and_compute_ekin_on_edges(
        wgtfac_e=wgtfac_e, vn=vn, vt=vt
    )
    return vn_ie, z_vt_ie, z_kin_hor_e


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges(
    wgtfac_e: fa.EKvpField,
    vn: fa.EKwpField,
    vt: fa.EKvpField,
    vn_ie: fa.EKvpField,
    z_vt_ie: fa.EKvpField,
    z_kin_hor_e: fa.EKvpField,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges(
        wgtfac_e,
        vn,
        vt,
        out=(vn_ie, z_vt_ie, z_kin_hor_e),
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
