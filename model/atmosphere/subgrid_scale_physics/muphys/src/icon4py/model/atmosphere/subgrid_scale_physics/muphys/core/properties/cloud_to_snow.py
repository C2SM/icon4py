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
def _cloud_to_snow(
    t:       gtx.Field[Dims[I,J,K], float],             # Temperature
    qc:      gtx.Field[Dims[I,J,K], float],             # Cloud specific mass
    qs:      gtx.Field[Dims[I,J,K], float],             # Snow specific mass
    ns:      gtx.Field[Dims[I,J,K], float],             # Snow number
    lam:     gtx.Field[Dims[I,J,K], float],             # Snow slope parameter (lambda)
) -> gtx.Field[Dims[I,J,K], float]:                     # Return: Riming snow rate
    ECS = 0.9
    B_RIM = -(graupel_ct.v1s + 3.0)
    C_RIM = 2.61 * ECS * graupel_ct.v0s  # (with pi*gam(v1s+3)/4 = 2.610)
    return where( min(qc,qs) > graupel_ct.qmin and t > graupel_ct.tfrz_hom, C_RIM*ns*qc*pow(lam, B_RIM), 0 )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def cloud_to_snow(
    t:       gtx.Field[Dims[I,J,K], float],             # Temperature
    qc:      gtx.Field[Dims[I,J,K], float],             # Cloud specific mass
    qs:      gtx.Field[Dims[I,J,K], float],             # Snow specific mass
    ns:      gtx.Field[Dims[I,J,K], float],             # Snow number
    lam:     gtx.Field[Dims[I,J,K], float],             # Snow slope parameter
):
    _cloud_to_snow( t, gc, qs, ns, lambda )
