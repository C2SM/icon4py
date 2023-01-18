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

from eve import SourceLocation
from functional.ffront import program_ast as past
from functional.ffront import type_specifications as ts
from functional.iterator.builtins import (
    deref,
    named_range,
    power,
    shift,
    unstructured_domain,
)
from functional.iterator.runtime import closure, fendef, fundef

from icon4py.common.dimension import E2C, CellDim, E2CDim, EdgeDim, KDim, Koff
from icon4py.pyutils.metadata import FieldInfo


@fundef
def step(i, theta_v, ikidx, zdiff_gradp, theta_v_ic, inv_ddqz_z_full):
    d_ikidx = deref(shift(i)(ikidx))

    d_theta_v = deref(shift(Koff, d_ikidx, E2C, i)(theta_v))
    s_theta_v_ic = shift(Koff, d_ikidx, E2C, i)(theta_v_ic)
    d_theta_v_ic = deref(s_theta_v_ic)
    d_theta_v_ic_p1 = deref(shift(Koff, 1)(s_theta_v_ic))
    d_inv_ddqz_z_full = deref(shift(Koff, d_ikidx, E2C, i)(inv_ddqz_z_full))
    d_zdiff_gradp = deref(shift(i)(zdiff_gradp))

    return (
        d_theta_v + d_zdiff_gradp * (d_theta_v_ic - d_theta_v_ic_p1) * d_inv_ddqz_z_full
    )


@fundef
def _mo_solve_nonhydro_stencil_21(
    theta_v,
    ikidx,
    zdiff_gradp,
    theta_v_ic,
    inv_ddqz_z_full,
    inv_dual_edge_length,
    grav_o_cpd,
):
    z_theta1 = step(0, theta_v, ikidx, zdiff_gradp, theta_v_ic, inv_ddqz_z_full)
    z_theta2 = step(1, theta_v, ikidx, zdiff_gradp, theta_v_ic, inv_ddqz_z_full)
    z_hydro_corr = (
        deref(grav_o_cpd)
        * deref(inv_dual_edge_length)
        * (z_theta2 - z_theta1)
        * 4.0
        / power((z_theta1 + z_theta2), 2.0)
    )
    return z_hydro_corr


@fendef
def mo_solve_nonhydro_stencil_21(
    theta_v,
    ikidx,
    zdiff_gradp,
    theta_v_ic,
    inv_ddqz_z_full,
    inv_dual_edge_length,
    grav_o_cpd,
    z_hydro_corr,
    hstart: int,
    hend: int,
    kstart: int,
    kend: int,
):
    closure(
        unstructured_domain(
            named_range(EdgeDim, hstart, hend), named_range(KDim, kstart, kend)
        ),
        _mo_solve_nonhydro_stencil_21,
        z_hydro_corr,
        [
            theta_v,
            ikidx,
            zdiff_gradp,
            theta_v_ic,
            inv_ddqz_z_full,
            inv_dual_edge_length,
            grav_o_cpd,
        ],
    )


_dummy_loc = SourceLocation(1, 1, "")
_metadata = {
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
    "ikidx": FieldInfo(
        field=past.FieldSymbol(
            id="ikidx",
            type=ts.FieldType(
                dims=[EdgeDim, E2CDim, KDim],
                dtype=ts.ScalarType(kind=ts.ScalarKind.INT32),
            ),
            location=_dummy_loc,
        ),
        inp=True,
        out=False,
    ),
    "zdiff_gradp": FieldInfo(
        field=past.FieldSymbol(
            id="zdiff_gradp",
            type=ts.FieldType(
                dims=[EdgeDim, E2CDim, KDim],
                dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
            ),
            location=_dummy_loc,
        ),
        inp=True,
        out=False,
    ),
    "theta_v_ic": FieldInfo(
        field=past.FieldSymbol(
            id="theta_v_ic",
            type=ts.FieldType(
                dims=[CellDim, KDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
            ),
            location=_dummy_loc,
        ),
        inp=True,
        out=False,
    ),
    "inv_ddqz_z_full": FieldInfo(
        field=past.FieldSymbol(
            id="inv_ddqz_z_full",
            type=ts.FieldType(
                dims=[CellDim, KDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
            ),
            location=_dummy_loc,
        ),
        inp=True,
        out=False,
    ),
    "inv_dual_edge_length": FieldInfo(
        field=past.FieldSymbol(
            id="inv_dual_edge_length",
            type=ts.FieldType(
                dims=[EdgeDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
            ),
            location=_dummy_loc,
        ),
        inp=True,
        out=False,
    ),
    "grav_o_cpd": FieldInfo(
        field=past.FieldSymbol(
            id="grav_o_cpd",
            type=ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)),
            location=_dummy_loc,
        ),
        inp=True,
        out=False,
    ),
    "z_hydro_corr": FieldInfo(
        field=past.FieldSymbol(
            id="z_hydro_corr",
            type=ts.FieldType(
                dims=[EdgeDim, KDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
            ),
            location=_dummy_loc,
        ),
        inp=False,
        out=True,
    ),
}

# patch the fendef with metainfo for icon4pygen
mo_solve_nonhydro_stencil_21.__dict__["offsets"] = [
    Koff.value,
    E2C.value,
]  # could be done with a pass...
mo_solve_nonhydro_stencil_21.__dict__["metadata"] = _metadata
