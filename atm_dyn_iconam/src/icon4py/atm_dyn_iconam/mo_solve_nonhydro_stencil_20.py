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
    list_get,
    named_range,
    shift,
    unstructured_domain,
)
from gt4py.next.iterator.runtime import closure, fendef, fundef
from gt4py.next.type_system import type_specifications as ts

from icon4py.common.dimension import E2C, CellDim, E2CDim, EdgeDim, KDim, Koff
from icon4py.icon4pygen.metadata import FieldInfo


@fundef
def step(
    i,
    z_exner_ex_pr,
    zdiff_gradp,
    ikoffset,
    z_dexner_dz_c_1,
    z_dexner_dz_c_2,
):
    d_ikoffset = list_get(i, deref(ikoffset))
    d_z_exner_exp_pr = deref(shift(Koff, d_ikoffset, E2C, i)(z_exner_ex_pr))
    d_z_dexner_dz_c_1 = deref(shift(Koff, d_ikoffset, E2C, i)(z_dexner_dz_c_1))
    d_z_dexner_dz_c_2 = deref(shift(Koff, d_ikoffset, E2C, i)(z_dexner_dz_c_2))
    d_zdiff_gradp = list_get(i, deref(zdiff_gradp))

    return d_z_exner_exp_pr + d_zdiff_gradp * (
        d_z_dexner_dz_c_1 + d_zdiff_gradp * d_z_dexner_dz_c_2
    )


@fundef
def _mo_solve_nonhydro_stencil_20(
    inv_dual_edge_length,
    z_exner_ex_pr,
    zdiff_gradp,
    ikoffset,
    z_dexner_dz_c_1,
    z_dexner_dz_c_2,
):
    return deref(inv_dual_edge_length) * (
        step(1, z_exner_ex_pr, zdiff_gradp, ikoffset, z_dexner_dz_c_1, z_dexner_dz_c_2)
        - step(
            0, z_exner_ex_pr, zdiff_gradp, ikoffset, z_dexner_dz_c_1, z_dexner_dz_c_2
        )
    )


@fendef
def mo_solve_nonhydro_stencil_20(
    inv_dual_edge_length,
    z_exner_ex_pr,
    zdiff_gradp,
    ikoffset,
    z_dexner_dz_c_1,
    z_dexner_dz_c_2,
    z_gradh_exner,
    horizontal_start,
    horizontal_end,
    vertical_start,
    vertical_end,
):
    closure(
        unstructured_domain(
            named_range(EdgeDim, horizontal_start, horizontal_end),
            named_range(KDim, vertical_start, vertical_end),
        ),
        _mo_solve_nonhydro_stencil_20,
        z_gradh_exner,
        [
            inv_dual_edge_length,
            z_exner_ex_pr,
            zdiff_gradp,
            ikoffset,
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
    "ikoffset": FieldInfo(
        field=past.FieldSymbol(
            id="ikoffset",
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
