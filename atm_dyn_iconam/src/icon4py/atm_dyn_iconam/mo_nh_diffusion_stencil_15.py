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

from functional.iterator.builtins import (
    deref,
    if_,
    named_range,
    shift,
    unstructured_domain,
)
from functional.iterator.runtime import closure, fendef, fundef

from icon4py.common.dimension import C2E2C, C2E2CDim, CellDim, KDim, Koff


@fundef
def step(i, geofac_n2s_nbh, vcoef, theta_v, zd_vertidx):
    d_vcoef = deref(shift(i)(vcoef))
    s_theta_v = shift(C2E2C, i, Koff, deref(shift(i)(zd_vertidx)))(theta_v)
    return deref(shift(i)(geofac_n2s_nbh)) * (
        d_vcoef * deref(s_theta_v) + (1.0 - d_vcoef) * deref(shift(Koff, 1)(s_theta_v))
    )


@fundef
def _mo_nh_diffusion_stencil_15(
    mask, zd_vertidx, zd_diffcoef, geofac_n2s_c, geofac_n2s_nbh, vcoef, theta_v, z_temp
):
    summed = (
        step(0, geofac_n2s_nbh, vcoef, theta_v, zd_vertidx)
        + step(1, geofac_n2s_nbh, vcoef, theta_v, zd_vertidx)
        + step(2, geofac_n2s_nbh, vcoef, theta_v, zd_vertidx)
    )
    update = deref(z_temp) + deref(zd_diffcoef) * (
        deref(theta_v) * deref(geofac_n2s_c) + summed
    )

    return if_(deref(mask), update, deref(z_temp))


@fendef
def mo_nh_diffusion_stencil_15(
    mask,
    zd_vertidx,
    zd_diffcoef,
    geofac_n2s_c,
    geofac_n2s_nbh,
    vcoef,
    theta_v,
    z_temp,
    hstart,
    hend,
    kstart,
    kend,
):
    closure(
        unstructured_domain(
            named_range(CellDim, hstart, hend), named_range(KDim, kstart, kend)
        ),
        _mo_nh_diffusion_stencil_15,
        z_temp,
        [
            mask,
            zd_vertidx,
            zd_diffcoef,
            geofac_n2s_c,
            geofac_n2s_nbh,
            vcoef,
            theta_v,
            z_temp,
        ],
    )


_metadata = f"""{C2E2C.value}
mask           Field[[{CellDim.value}, {KDim.value}], dtype=bool]  in
zd_vertidx     Field[[{CellDim.value}, {C2E2CDim.value}, {KDim.value}], dtype=int32]  in
zd_diffcoef    Field[[{CellDim.value}, {KDim.value}], dtype=float64]  in
geofac_n2s_c   Field[[{CellDim.value}], dtype=float64]  in
geofac_n2s_nbh Field[[{CellDim.value}, {C2E2CDim.value}], dtype=float64]  in
vcoef          Field[[{CellDim.value}, {C2E2CDim.value}, {KDim.value}], dtype=float64]  in
theta_v        Field[[{CellDim.value}, {KDim.value}], dtype=float64]  in
z_temp         Field[[{CellDim.value}, {KDim.value}], dtype=float64]  inout"""

# patch the fendef with metainfo for icon4pygen
mo_nh_diffusion_stencil_15.__dict__["offsets"] = [
    Koff.value,
    C2E2C.value,
]  # could be done with a pass...
mo_nh_diffusion_stencil_15.__dict__["metadata"] = _metadata
