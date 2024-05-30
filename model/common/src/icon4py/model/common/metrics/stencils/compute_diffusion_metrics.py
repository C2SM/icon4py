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

import math

import numpy as np


def _compute_nbidx(
    k_start: list,
    k_end: list,
    z_mc: np.array,
    z_mc_off: np.array,
    nbidx: np.array,
    jc: int,
    ind: int,
    nlev: int,
) -> np.array:
    jk_start = nlev - 1
    for jk in reversed(range(k_start[jc], k_end[jc])):
        for jk1 in reversed(range(jk_start)):
            if (
                z_mc[jc, jk] <= z_mc_off[jc, ind, jk1]
                and z_mc[jc, jk] >= z_mc_off[jc, ind, jk1 + 1]
            ):
                nbidx[jc, ind, jk] = jk1
                jk_start = jk1 + 1
                break

    return nbidx[jc, ind, :]


def _compute_z_vintcoeff(
    k_start: list,
    k_end: list,
    z_mc: np.array,
    z_mc_off: np.array,
    z_vintcoeff: np.array,
    jc: int,
    ind: int,
    nlev: int,
) -> np.array:
    jk_start = nlev - 1
    for jk in reversed(range(k_start[jc], k_end[jc])):
        for jk1 in reversed(range(jk_start)):
            if (
                z_mc[jc, jk] <= z_mc_off[jc, ind, jk1]
                and z_mc[jc, jk] >= z_mc_off[jc, ind, jk1 + 1]
            ):
                z_vintcoeff[jc, ind, jk] = (z_mc[jc, jk] - z_mc_off[jc, ind, jk1 + 1]) / (
                    z_mc_off[jc, ind, jk1] - z_mc_off[jc, ind, jk1 + 1]
                )
                jk_start = jk1 + 1
                break

    return z_vintcoeff[jc, ind, :]


def _compute_i_params(
    k_start: list,
    k_end: list,
    z_maxslp_avg: np.array,
    z_maxhgtd_avg: np.array,
    c_owner_mask: np.array,
    thslp_zdiffu: float,
    thhgtd_zdiffu: float,
    cell_nudging: int,
    n_cells: int,
    nlev: int,
) -> tuple[list, int, int]:
    i_indlist = [0] * n_cells
    i_listreduce = 0
    ji = -1
    ji_ind = -1

    for jc in range(cell_nudging, n_cells):
        if (
            z_maxslp_avg[jc, nlev - 1] >= thslp_zdiffu
            or z_maxhgtd_avg[jc, nlev - 1] >= thhgtd_zdiffu
        ) and c_owner_mask[jc]:
            ji += 1
            i_indlist[ji] = jc

            if k_start[jc] is not None and k_end[jc] is not None and k_start[jc] > k_end[jc]:
                i_listreduce += 1
            else:
                ji_ind += 1
                i_indlist[ji_ind] = jc

    return i_indlist, i_listreduce, ji


def _compute_k_start_end(
    z_mc: np.array,
    max_nbhgt: np.array,
    z_maxslp_avg: np.array,
    z_maxhgtd_avg: np.array,
    c_owner_mask: np.array,
    thslp_zdiffu: float,
    thhgtd_zdiffu: float,
    cell_nudging: int,
    n_cells: int,
    nlev: int,
) -> tuple[list, list]:
    k_start = [None] * n_cells
    k_end = [None] * n_cells
    for jc in range(cell_nudging, n_cells):
        if (
            z_maxslp_avg[jc, nlev - 1] >= thslp_zdiffu
            or z_maxhgtd_avg[jc, nlev - 1] >= thhgtd_zdiffu
        ) and c_owner_mask[jc]:
            for jk in reversed(range(nlev)):
                if z_mc[jc, jk] >= max_nbhgt[jc]:
                    k_end[jc] = jk + 1
                    break

            for jk in range(nlev):
                if z_maxslp_avg[jc, jk] >= thslp_zdiffu or z_maxhgtd_avg[jc, jk] >= thhgtd_zdiffu:
                    k_start[jc] = jk
                    break

            if k_start[jc] is not None and k_end[jc] is not None and k_start[jc] > k_end[jc]:
                k_start[jc] = nlev - 1

    return k_start, k_end


def compute_diffusion_metrics(
    z_mc: np.array,
    z_mc_off: np.array,
    k_start: list,
    k_end: list,
    i_indlist: list,
    i_listdim: int,
    nbidx: np.array,
    z_vintcoeff: np.array,
    z_maxslp_avg: np.array,
    z_maxhgtd_avg: np.array,
    mask_hdiff: np.array,
    zd_diffcoef_dsl: np.array,
    zd_intcoef_dsl: np.array,
    zd_vertoffset_dsl: np.array,
    thslp_zdiffu: float,
    thhgtd_zdiffu: float,
    nlev: int,
) -> np.array:
    for ji in range(i_listdim):
        jc = i_indlist[ji]
        if k_start[jc] is not None and k_end[jc] is not None:
            nbidx[jc, 0, :] = _compute_nbidx(k_start, k_end, z_mc, z_mc_off, nbidx, jc, 0, nlev)
            nbidx[jc, 1, :] = _compute_nbidx(k_start, k_end, z_mc, z_mc_off, nbidx, jc, 1, nlev)
            nbidx[jc, 2, :] = _compute_nbidx(k_start, k_end, z_mc, z_mc_off, nbidx, jc, 2, nlev)

            z_vintcoeff[jc, 0, :] = _compute_z_vintcoeff(
                k_start, k_end, z_mc, z_mc_off, z_vintcoeff, jc, 0, nlev
            )
            z_vintcoeff[jc, 1, :] = _compute_z_vintcoeff(
                k_start, k_end, z_mc, z_mc_off, z_vintcoeff, jc, 1, nlev
            )
            z_vintcoeff[jc, 2, :] = _compute_z_vintcoeff(
                k_start, k_end, z_mc, z_mc_off, z_vintcoeff, jc, 2, nlev
            )
            for jk in range(k_start[jc], k_end[jc]):
                zd_intcoef_dsl[jc, 0, jk] = z_vintcoeff[jc, 0, jk]
                zd_intcoef_dsl[jc, 1, jk] = z_vintcoeff[jc, 1, jk]
                zd_intcoef_dsl[jc, 2, jk] = z_vintcoeff[jc, 2, jk]

                zd_vertoffset_dsl[jc, 0, jk] = nbidx[jc, 0, jk] - jk
                zd_vertoffset_dsl[jc, 1, jk] = nbidx[jc, 1, jk] - jk
                zd_vertoffset_dsl[jc, 2, jk] = nbidx[jc, 2, jk] - jk

                zd_diffcoef_dsl_var = max(
                    0.0,
                    math.sqrt(max(0.0, z_maxslp_avg[jc, jk] - thslp_zdiffu)) / 250.0,
                    2.0e-4 * math.sqrt(max(0.0, z_maxhgtd_avg[jc, jk] - thhgtd_zdiffu)),
                )
                zd_diffcoef_dsl_var = min(0.002, zd_diffcoef_dsl_var)
                zd_diffcoef_dsl[jc, jk] = zd_diffcoef_dsl_var

                mask_hdiff[jc, jk] = True

    return mask_hdiff, zd_diffcoef_dsl, zd_intcoef_dsl, zd_vertoffset_dsl
