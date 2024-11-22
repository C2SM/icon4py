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
def _cloud_x_ice(
    t:       gtx.Field[Dims[I,J,K], float],             # Temperature
    qc:      gtx.Field[Dims[I,J,K], float],             # Cloud specific mass
    qi:      gtx.Field[Dims[I,J,K], float],             # Ice specific mass
    dt:      float,                                     # Time step
) -> gtx.Field[Dims[I,J,K], float]:                     # Return: Homogeneous freezing rate
    result = where( qc > graupel_ct.qmin and t < graupel_ct.tfrz_hom, qc / dt, 0 )
    result = where( qi > graupel_ct.qmin and t > thermodyn.tmelt, -qi/dt, result )
    return result

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def cloud_x_ice(
    t:       gtx.Field[Dims[I,J,K], float],             # Temperature
    qc:      gtx.Field[Dims[I,J,K], float],             # Cloud specific mass
    qi:      gtx.Field[Dims[I,J,K], float],             # Ice specific mass
    dt:      float,                                     # Time step
):
    _cloud_x_ice( t, gc, qi, dt )
