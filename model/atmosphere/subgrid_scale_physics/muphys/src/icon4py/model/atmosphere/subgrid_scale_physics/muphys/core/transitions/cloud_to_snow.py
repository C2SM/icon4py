# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where, power

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common import constants as graupel_ct
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

gtx.field_operator
def _cloud_to_snow(
    t:       fa.CellField[ta.wpfloat],             # Temperature
    qc:      fa.CellField[ta.wpfloat],             # Cloud specific mass
    qs:      fa.CellField[ta.wpfloat],             # Snow specific mass
    ns:      fa.CellField[ta.wpfloat],             # Snow number
    lam:     fa.CellField[ta.wpfloat],             # Snow slope parameter (lambda)
) -> fa.CellField[ta.wpfloat]:                     # Return: Riming snow rate
    ECS = 0.9
    B_RIM = -(graupel_ct.v1s + 3.0)
    C_RIM = 2.61 * ECS * graupel_ct.v0s  # (with pi*gam(v1s+3)/4 = 2.610)
    return where( min(qc,qs) > graupel_ct.qmin and t > graupel_ct.tfrz_hom, C_RIM*ns*qc*lam**B_RIM, 0 )
#    return where( min(qc,qs) > graupel_ct.qmin and t > graupel_ct.tfrz_hom, C_RIM*ns*qc*power(lam, B_RIM), 0 )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def cloud_to_snow(
    t:       fa.CellField[ta.wpfloat],             # Temperature
    qc:      fa.CellField[ta.wpfloat],             # Cloud specific mass
    qs:      fa.CellField[ta.wpfloat],             # Snow specific mass
    ns:      fa.CellField[ta.wpfloat],             # Snow number
    lam:     fa.CellField[ta.wpfloat],             # Snow slope parameter
):
    _cloud_to_snow( t, qc, qs, ns, lam )
