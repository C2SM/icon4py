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
    shift,
    unstructured_domain,
)
from functional.iterator.runtime import closure, fendef, fundef

from icon4py.common.dimension import E2C, CellDim, E2CDim, EdgeDim, KDim, Koff
from icon4py.pyutils.metadata import FieldInfo


@fundef
def step(
    i,
    z_exner_ex_pr,
    zdiff_gradp,
    ikidx,
    z_dexner_dz_c_1,
    z_dexner_dz_c_2,
):
    d_ikidx = deref(shift(i)(ikidx))
    d_z_exner_exp_pr = deref(shift(Koff, d_ikidx, E2C, i)(z_exner_ex_pr))
    d_z_dexner_dz_c_1 = deref(shift(Koff, d_ikidx, E2C, i)(z_dexner_dz_c_1))
    d_z_dexner_dz_c_2 = deref(shift(Koff, d_ikidx, E2C, i)(z_dexner_dz_c_2))
    d_zdiff_gradp = deref(shift(i)(zdiff_gradp))

    return d_z_exner_exp_pr + d_zdiff_gradp * (
        d_z_dexner_dz_c_1 + d_zdiff_gradp * d_z_dexner_dz_c_2
    )


@fundef
def _mo_solve_nonhydro_stencil_20(
    inv_dual_edge_length,
    z_exner_ex_pr,
    zdiff_gradp,
    ikidx,
    z_dexner_dz_c_1,
    z_dexner_dz_c_2,
):
    return deref(inv_dual_edge_length) * (
        step(1, z_exner_ex_pr, zdiff_gradp, ikidx, z_dexner_dz_c_1, z_dexner_dz_c_2)
        - step(0, z_exner_ex_pr, zdiff_gradp, ikidx, z_dexner_dz_c_1, z_dexner_dz_c_2)
    )


@fendef
def mo_solve_nonhydro_stencil_20(
    inv_dual_edge_length,
    z_exner_ex_pr,
    zdiff_gradp,
    ikidx,
    z_dexner_dz_c_1,
    z_dexner_dz_c_2,
    z_gradh_exner,
    hstart: int,
    hend: int,
    kstart: int,
    kend: int,
):
    closure(
        unstructured_domain(
            named_range(EdgeDim, hstart, hend), named_range(KDim, kstart, kend)
        ),
        _mo_solve_nonhydro_stencil_20,
        z_gradh_exner,
        [
            inv_dual_edge_length,
            z_exner_ex_pr,
            zdiff_gradp,
            ikidx,
            z_dexner_dz_c_1,
            z_dexner_dz_c_2,
        ],
    )


_dummy_loc = SourceLocation(1, 1, "")
_metadata = {
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
    "z_exner_ex_pr": FieldInfo(
        field=past.FieldSymbol(
            id="z_exner_ex_pr",
            type=ts.FieldType(
                dims=[CellDim, KDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
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
    "z_dexner_dz_c_1": FieldInfo(
        field=past.FieldSymbol(
            id="z_dexner_dz_c_1",
            type=ts.FieldType(
                dims=[CellDim, KDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
            ),
            location=_dummy_loc,
        ),
        inp=True,
        out=False,
    ),
    "z_dexner_dz_c_2": FieldInfo(
        field=past.FieldSymbol(
            id="z_dexner_dz_c_2",
            type=ts.FieldType(
                dims=[CellDim, KDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
            ),
            location=_dummy_loc,
        ),
        inp=True,
        out=False,
    ),
    "z_gradh_exner": FieldInfo(
        field=past.FieldSymbol(
            id="z_gradh_exner",
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
mo_solve_nonhydro_stencil_20.__dict__["offsets"] = [
    Koff.value,
    E2C.value,
]  # could be done with a pass...
mo_solve_nonhydro_stencil_20.__dict__["metadata"] = _metadata
