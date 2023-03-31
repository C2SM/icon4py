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

from gt4py.eve import SourceLocation
from gt4py.next.ffront import program_ast as past
from gt4py.next.iterator.builtins import (
    deref,
    if_,
    named_range,
    shift,
    unstructured_domain,
)
from gt4py.next.iterator.runtime import closure, fendef, fundef
from gt4py.next.type_system import type_specifications as ts

from icon4py.common.dimension import C2E2C, C2E2CDim, CellDim, KDim, Koff
from icon4py.pyutils.metadata import FieldInfo


@fundef
def step(i, geofac_n2s_nbh, vcoef, theta_v, zd_vertoffset):
    d_vcoef = deref(shift(i)(vcoef))
    s_theta_v = shift(C2E2C, i, Koff, deref(shift(i)(zd_vertoffset)))(theta_v)
    return deref(shift(i)(geofac_n2s_nbh)) * (
        d_vcoef * deref(s_theta_v) + (1.0 - d_vcoef) * deref(shift(Koff, 1)(s_theta_v))
    )


@fundef
def _mo_nh_diffusion_stencil_15(
    mask,
    zd_vertoffset,
    zd_diffcoef,
    geofac_n2s_c,
    geofac_n2s_nbh,
    vcoef,
    theta_v,
    z_temp,
):
    summed = (
        step(0, geofac_n2s_nbh, vcoef, theta_v, zd_vertoffset)
        + step(1, geofac_n2s_nbh, vcoef, theta_v, zd_vertoffset)
        + step(2, geofac_n2s_nbh, vcoef, theta_v, zd_vertoffset)
    )
    update = deref(z_temp) + deref(zd_diffcoef) * (
        deref(theta_v) * deref(geofac_n2s_c) + summed
    )

    return if_(deref(mask), update, deref(z_temp))


@fendef
def mo_nh_diffusion_stencil_15(
    mask,
    zd_vertoffset,
    zd_diffcoef,
    geofac_n2s_c,
    geofac_n2s_nbh,
    vcoef,
    theta_v,
    z_temp,
    horizontal_start,
    horizontal_end,
    vertical_start,
    vertical_end,
):
    closure(
        unstructured_domain(
            named_range(CellDim, horizontal_start, horizontal_end),
            named_range(KDim, vertical_start, vertical_end),
        ),
        _mo_nh_diffusion_stencil_15,
        z_temp,
        [
            mask,
            zd_vertoffset,
            zd_diffcoef,
            geofac_n2s_c,
            geofac_n2s_nbh,
            vcoef,
            theta_v,
            z_temp,
        ],
    )


_dummy_loc = SourceLocation(1, 1, "")
_metadata = {
    "mask": FieldInfo(
        field=past.FieldSymbol(
            id="mask",
            type=ts.FieldType(
                dims=[CellDim, KDim], dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL)
            ),
            location=_dummy_loc,
        ),
        inp=True,
        out=False,
    ),
    "zd_vertoffset": FieldInfo(
        field=past.FieldSymbol(
            id="zd_vertoffset",
            type=ts.FieldType(
                dims=[CellDim, C2E2CDim, KDim],
                dtype=ts.ScalarType(kind=ts.ScalarKind.INT32),
            ),
            location=_dummy_loc,
        ),
        inp=True,
        out=False,
    ),
    "zd_diffcoef": FieldInfo(
        field=past.FieldSymbol(
            id="zd_diffcoef",
            type=ts.FieldType(
                dims=[CellDim, KDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
            ),
            location=_dummy_loc,
        ),
        inp=True,
        out=False,
    ),
    "geofac_n2s_c": FieldInfo(
        field=past.FieldSymbol(
            id="geofac_n2s_c",
            type=ts.FieldType(
                dims=[CellDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
            ),
            location=_dummy_loc,
        ),
        inp=True,
        out=False,
    ),
    "geofac_n2s_nbh": FieldInfo(
        field=past.FieldSymbol(
            id="geofac_n2s_nbh",
            type=ts.FieldType(
                dims=[CellDim, C2E2CDim],
                dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
            ),
            location=_dummy_loc,
        ),
        inp=True,
        out=False,
    ),
    "vcoef": FieldInfo(
        field=past.FieldSymbol(
            id="vcoef",
            type=ts.FieldType(
                dims=[CellDim, C2E2CDim, KDim],
                dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
            ),
            location=_dummy_loc,
        ),
        inp=True,
        out=False,
    ),
    "theta_v": FieldInfo(
        field=past.FieldSymbol(
            id="theta_v",
            type=ts.FieldType(
                dims=[CellDim, KDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
            ),
            location=_dummy_loc,
        ),
        inp=True,
        out=False,
    ),
    "z_temp": FieldInfo(
        field=past.FieldSymbol(
            id="z_temp",
            type=ts.FieldType(
                dims=[CellDim, KDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
            ),
            location=_dummy_loc,
        ),
        inp=True,
        out=True,
    ),
}

# patch the fendef with metainfo for icon4pygen
mo_nh_diffusion_stencil_15.__dict__["offsets"] = [
    Koff.value,
    C2E2C.value,
]  # could be done with a pass...
mo_nh_diffusion_stencil_15.__dict__["metadata"] = _metadata
