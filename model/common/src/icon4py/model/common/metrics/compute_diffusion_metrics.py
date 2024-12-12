# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.common.settings import xp
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc


def compute_max_nbhgt_np(c2e2c: xp.ndarray, z_mc: xp.ndarray, nlev: int) -> xp.ndarray:
    z_mc_nlev = z_mc[:, nlev - 1]
    max_nbhgt_0_1 = xp.maximum(z_mc_nlev[c2e2c[:, 0]], z_mc_nlev[c2e2c[:, 1]])
    max_nbhgt = xp.maximum(max_nbhgt_0_1, z_mc_nlev[c2e2c[:, 2]])
    return max_nbhgt


def _compute_nbidx(
    k_range: range,
    z_mc: field_alloc.NDArray,
    z_mc_off: field_alloc.NDArray,
    nbidx: field_alloc.NDArray,
    jc: int,
    nlev: int,
) -> field_alloc.NDArray:
    for ind in range(3):
        jk_start = nlev - 1
        for jk in reversed(k_range):
            for jk1 in reversed(range(jk_start)):
                if (
                    z_mc[jc, jk] <= z_mc_off[jc, ind, jk1]
                    and z_mc[jc, jk] >= z_mc_off[jc, ind, jk1 + 1]
                ):
                    nbidx[jc, ind, jk] = jk1
                    jk_start = jk1 + 1
                    break

    return nbidx[jc, :, :]


def _compute_z_vintcoeff(
    k_range: range,
    z_mc: field_alloc.NDArray,
    z_mc_off: field_alloc.NDArray,
    z_vintcoeff: field_alloc.NDArray,
    jc: int,
    nlev: int,
) -> field_alloc.NDArray:
    for ind in range(3):
        jk_start = nlev - 1
        for jk in reversed(k_range):
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

    return z_vintcoeff[jc, :, :]


def _compute_ls_params(
    k_start: list,
    k_end: list,
    maxslp_avg: field_alloc.NDArray,
    maxhgtd_avg: field_alloc.NDArray,
    c_owner_mask: field_alloc.NDArray,
    thslp_zdiffu: float,
    thhgtd_zdiffu: float,
    cell_nudging: int,
    n_cells: int,
    nlev: int,
) -> tuple[list, int, int]:
    indlist = [0] * n_cells
    listreduce = 0
    ji = -1
    ji_ind = -1

    for jc in range(cell_nudging, n_cells):
        if (
            maxslp_avg[jc, nlev - 1] >= thslp_zdiffu or maxhgtd_avg[jc, nlev - 1] >= thhgtd_zdiffu
        ) and c_owner_mask[jc]:
            ji += 1
            indlist[ji] = jc

            if all((k_start[jc], k_end[jc])) and k_start[jc] > k_end[jc]:
                listreduce += 1
            else:
                ji_ind += 1
                indlist[ji_ind] = jc

    return indlist, listreduce, ji


def _compute_k_start_end(
    z_mc: field_alloc.NDArray,
    max_nbhgt: field_alloc.NDArray,
    maxslp_avg: field_alloc.NDArray,
    maxhgtd_avg: field_alloc.NDArray,
    c_owner_mask: field_alloc.NDArray,
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
            maxslp_avg[jc, nlev - 1] >= thslp_zdiffu or maxhgtd_avg[jc, nlev - 1] >= thhgtd_zdiffu
        ) and c_owner_mask[jc]:
            for jk in reversed(range(nlev)):
                if z_mc[jc, jk] >= max_nbhgt[jc]:
                    k_end[jc] = jk + 1
                    break

            for jk in range(nlev):
                if maxslp_avg[jc, jk] >= thslp_zdiffu or maxhgtd_avg[jc, jk] >= thhgtd_zdiffu:
                    k_start[jc] = jk
                    break

            if all((k_start[jc], k_end[jc])) and k_start[jc] > k_end[jc]:
                k_start[jc] = nlev - 1

    return k_start, k_end


def compute_diffusion_metrics(
    c2e2c: field_alloc.NDArray,
    z_mc: field_alloc.NDArray,
    max_nbhgt: field_alloc.NDArray,
    c_owner_mask: field_alloc.NDArray,
    maxslp_avg: field_alloc.NDArray,
    maxhgtd_avg: field_alloc.NDArray,
    thslp_zdiffu: float,
    thhgtd_zdiffu: float,
    n_c2e2c: int,
    cell_nudging: int,
    n_cells: int,
    nlev: int,
) -> tuple[field_alloc.NDArray, field_alloc.NDArray, field_alloc.NDArray, field_alloc.NDArray]:
    z_mc_off = z_mc[c2e2c]
    nbidx = xp.ones(shape=(n_cells, n_c2e2c, nlev), dtype=int)
    z_vintcoeff = xp.zeros(shape=(n_cells, n_c2e2c, nlev))
    mask_hdiff = xp.zeros(shape=(n_cells, nlev), dtype=bool)
    zd_vertoffset_dsl = xp.zeros(shape=(n_cells, n_c2e2c, nlev))
    zd_intcoef_dsl = xp.zeros(shape=(n_cells, n_c2e2c, nlev))
    zd_diffcoef_dsl = xp.zeros(shape=(n_cells, nlev))

    k_start, k_end = _compute_k_start_end(
        z_mc=z_mc,
        max_nbhgt=max_nbhgt,
        maxslp_avg=maxslp_avg,
        maxhgtd_avg=maxhgtd_avg,
        c_owner_mask=c_owner_mask,
        thslp_zdiffu=thslp_zdiffu,
        thhgtd_zdiffu=thhgtd_zdiffu,
        cell_nudging=cell_nudging,
        n_cells=n_cells,
        nlev=nlev,
    )

    indlist, listreduce, ji = _compute_ls_params(
        k_start=k_start,
        k_end=k_end,
        maxslp_avg=maxslp_avg,
        maxhgtd_avg=maxhgtd_avg,
        c_owner_mask=c_owner_mask,
        thslp_zdiffu=thslp_zdiffu,
        thhgtd_zdiffu=thhgtd_zdiffu,
        cell_nudging=cell_nudging,
        n_cells=n_cells,
        nlev=nlev,
    )

    listdim = ji - listreduce

    for ji in range(listdim):
        jc = indlist[ji]
        k_range = range(k_start[jc], k_end[jc])
        if all((k_range)):
            nbidx[jc, :, :] = _compute_nbidx(k_range, z_mc, z_mc_off, nbidx, jc, nlev)
            z_vintcoeff[jc, :, :] = _compute_z_vintcoeff(
                k_range, z_mc, z_mc_off, z_vintcoeff, jc, nlev
            )

            zd_intcoef_dsl[jc, :, k_range] = z_vintcoeff[jc, :, k_range]
            zd_vertoffset_dsl[jc, :, k_range] = nbidx[jc, :, k_range] - xp.transpose([k_range] * 3)
            mask_hdiff[jc, k_range] = True

            zd_diffcoef_dsl_var = xp.maximum(
                0.0,
                xp.maximum(
                    xp.sqrt(xp.maximum(0.0, maxslp_avg[jc, k_range] - thslp_zdiffu)) / 250.0,
                    2.0e-4 * xp.sqrt(xp.maximum(0.0, maxhgtd_avg[jc, k_range] - thhgtd_zdiffu)),
                ),
            )
            zd_diffcoef_dsl[jc, k_range] = xp.minimum(0.002, zd_diffcoef_dsl_var)

    # flatten first two dims:
    zd_intcoef_dsl = zd_intcoef_dsl.reshape(
        (zd_intcoef_dsl.shape[0] * zd_intcoef_dsl.shape[1],) + zd_intcoef_dsl.shape[2:]
    )
    zd_vertoffset_dsl = zd_vertoffset_dsl.reshape(
        (zd_vertoffset_dsl.shape[0] * zd_vertoffset_dsl.shape[1],) + zd_vertoffset_dsl.shape[2:]
    )

    return mask_hdiff, zd_diffcoef_dsl, zd_intcoef_dsl, zd_vertoffset_dsl
