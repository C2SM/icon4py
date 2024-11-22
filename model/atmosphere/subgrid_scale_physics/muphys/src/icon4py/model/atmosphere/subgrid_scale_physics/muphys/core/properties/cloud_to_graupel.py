# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.common import graupel_ct

gtx.field_operator
def _cloud_to_graupel(
    t:    gtx.Field[Dims[I,J,K], float],
    rho:  gtx.Field[Dims[I,J,K], float], 
    qc:   gtx.Field[Dims[I,J,K], float],
    qg:   gtx.Field[Dims[I,J,K], float],
) -> gtx.Field[Dims[I,J,K], float]:
    A_RIM = 4.43
    B_RIM = 0.94878
    return where( min(qc,qg) > graupel_ct.qmin and t > graupel_ct.tfrz_hom, A_RIM * qc * pow(qg * rho, B_RIM), 0 )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def cloud_to_graupel(
    t:    gtx.Field[Dims[I,J,K], float], 
    rho:  gtx.Field[Dims[I,J,K], float],
    qc:   gtx.Field[Dims[I,J,K], float],
    qg:   gtx.Field[Dims[I,J,K], float],
):
    _cloud_to_graupel( t, rho, gc, qg )
