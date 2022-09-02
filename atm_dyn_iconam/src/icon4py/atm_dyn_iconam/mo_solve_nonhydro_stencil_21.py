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
    named_range,
    shift,
    unstructured_domain,
)
from functional.iterator.runtime import closure, fendef, fundef

from icon4py.common.dimension import E2C, CellDim, E2CDim, EdgeDim, KDim, Koff


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


_metadata = f"""{E2C.value}
inv_dual_edge_length  Field[[{EdgeDim.value}], dtype=float64]  in
z_exner_ex_pr         Field[[{CellDim.value}, {KDim.value}], dtype=float64]  in
zdiff_gradp           Field[[{EdgeDim.value}, {E2CDim.value}, {KDim.value}], dtype=float64]  in
kidx                  Field[[{EdgeDim.value}, {E2CDim.value}, {KDim.value}], dtype=int32]  in
z_dexner_dz_c_1       Field[[{CellDim.value}, {KDim.value}], dtype=float64]  in
z_dexner_dz_c_2       Field[[{CellDim.value}, {KDim.value}], dtype=float64]  in
z_gradh_exner         Field[[{EdgeDim.value}, {KDim.value}], dtype=float64]  out"""

# patch the fendef with metainfo for icon4pygen
mo_solve_nonhydro_stencil_20.__dict__["offsets"] = [
    Koff.value,
    E2C.value,
]  # could be done with a pass...
mo_solve_nonhydro_stencil_20.__dict__["metadata"] = _metadata
