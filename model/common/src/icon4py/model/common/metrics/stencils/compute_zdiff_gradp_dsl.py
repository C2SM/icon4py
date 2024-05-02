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
import numpy as np


def compute_zdiff_gradp_dsl(
    e2c,
    c_lin_e: np.array,
    z_mc: np.array,
    zdiff_gradp: np.array,
    z_ifc: np.array,
    flat_idx: np.array,
    nlev: int,
    nedges: int,
) -> np.array:
    c_lin_e = np.expand_dims(c_lin_e, axis=-1)
    z_me = np.sum(z_mc[e2c] * c_lin_e, axis=1)
    zdiff_gradp[:, 0, :] = z_me - z_mc[e2c[:, 0]]
    zdiff_gradp[:, 1, :] = z_me - z_mc[e2c[:, 1]]

    for k in range(nlev):
        for e in range(
            nedges
        ):  # TODO: boundary here is not nedges, rather p_patch(jg)%edges%start_block(2)
            if (
                z_me[e, k] <= z_ifc[e2c[e, 0], k]
                and z_me[e, k] >= z_ifc[e2c[e, 0], k + 1]
                and z_me[e, k] <= z_ifc[e2c[e, 1], k]
                and z_me[e, k] >= z_ifc[e2c[e, 1], k + 1]
            ):
                flat_idx[e] = k

    for je in range(nedges):
        jk_start = flat_idx[je]
        for jk in range(flat_idx[je] + 1, nlev):
            for jk1 in range(jk_start, nlev):
                if (
                    jk1 == nlev
                    or z_me[je, jk] <= z_ifc[e2c[je, 0], jk1]
                    and z_me[je, jk] >= z_ifc[e2c[je, 0], jk1 + 1]
                ):
                    zdiff_gradp[je, 0, jk1] = z_me[je, jk] - z_mc[e2c[je, 0], jk1]
                    jk_start = jk1
                    break

    extrapol_dist = 5.0
    z_aux1 = np.maximum(z_ifc[e2c[:, 0], nlev], z_ifc[e2c[:, 1], nlev])
    z_aux2 = z_aux1 - extrapol_dist

    for je in range(nedges):
        jk_start = flat_idx[je]
        for jk in range(flat_idx[je] + 1, nlev):
            for jk1 in range(jk_start, nlev):
                if (
                    jk1 == nlev
                    or z_me[je, jk] <= z_ifc[e2c[je, 1], jk1]
                    and z_me[je, jk] >= z_ifc[e2c[je, 1], jk1 + 1]
                ):
                    zdiff_gradp[je, 1, jk1] = z_me[je, jk] - z_mc[e2c[je, 1], jk1]
                    jk_start = jk1
                    break

    for je in range(nedges):
        jk_start = flat_idx[je]
        for jk in range(flat_idx[je] + 1, nlev):
            if z_me[je, jk] < z_aux2[je]:
                for jk1 in range(jk_start, nlev):
                    if (
                        jk1 == nlev
                        or z_aux2[je] <= z_ifc[e2c[je, 0], jk1]
                        and z_aux2[je] >= z_ifc[e2c[je, 0], jk1 + 1]
                    ):
                        zdiff_gradp[je, 0, jk1] = z_aux2[je] - z_mc[e2c[je, 0], jk1]
                        jk_start = jk1
                        break

    for je in range(nedges):
        jk_start = flat_idx[je]
        for jk in range(flat_idx[je] + 1, nlev):
            if z_me[je, jk] < z_aux2[je]:
                for jk1 in range(jk_start, nlev):
                    if (
                        jk1 == nlev
                        or z_aux2[je] <= z_ifc[e2c[je, 1], jk1]
                        and z_aux2[je] >= z_ifc[e2c[je, 1], jk1 + 1]
                    ):
                        zdiff_gradp[je, 1, jk1] = z_aux2[je] - z_mc[e2c[je, 1], jk1]
                        jk_start = jk1
                        break

    return zdiff_gradp
