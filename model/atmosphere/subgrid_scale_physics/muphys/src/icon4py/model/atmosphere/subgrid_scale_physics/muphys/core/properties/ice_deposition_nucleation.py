# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.common import graupel_ct

@gtx.field_operator
def _ice_deposition_nucleation(
    t:    gtx.Field[Dims[I,J,K], float], 
    qc:   gtx.Field[Dims[I,J,K], float], 
    qi:   gtx.Field[Dims[I,J,K], float],
    ni:   gtx.Field[Dims[I,J,K], float],
    dvsi: gtx.Field[Dims[I,J,K], float], 
    dt:   gtx.Field[Dims[I,J,K], float],
):
    return where( qi <= graupel_ct.qmin and ((t < graupel_ct.tfrz_het2 and dvsi > 0) or (t <= graupel_ct.tfrz_het1 and qc > graupel_ct.qmin)), min(graupel_ct.m0_ice * ni, max(0, dvsi)) / dt, 0 )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def ice_deposition_nucleation(
    t:    gtx.Field[Dims[I,J,K], float], 
    qc:   gtx.Field[Dims[I,J,K], float], 
    qi:   gtx.Field[Dims[I,J,K], float],
    ni:   gtx.Field[Dims[I,J,K], float],
    dvsi: gtx.Field[Dims[I,J,K], float], 
    dt:   gtx.Field[Dims[I,J,K], float],
):
    _ice_deposition_nucleation( t, qc, qi, ni, dvsi, dt )
