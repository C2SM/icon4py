# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from types import ModuleType

import numpy as np

from icon4py.model.common.utils import data_allocation as data_alloc


def compute_max_nbhgt_array_ns(
    c2e2c: data_alloc.NDArray,
    z_mc: data_alloc.NDArray,
    nlev: int,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    z_mc_nlev = z_mc[:, nlev - 1]
    max_nbhgt_0_1 = array_ns.maximum(z_mc_nlev[c2e2c[:, 0]], z_mc_nlev[c2e2c[:, 1]])
    max_nbhgt = array_ns.maximum(max_nbhgt_0_1, z_mc_nlev[c2e2c[:, 2]])
    return max_nbhgt


def _compute_nbidx(
    k_range: range,
    z_mc: data_alloc.NDArray,
    z_mc_off: data_alloc.NDArray,
    nbidx: data_alloc.NDArray,
    jc: int,
    nlev: int,
) -> data_alloc.NDArray:
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
    z_mc: data_alloc.NDArray,
    z_mc_off: data_alloc.NDArray,
    z_vintcoeff: data_alloc.NDArray,
    jc: int,
    nlev: int,
) -> data_alloc.NDArray:
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


def _compute_k_start_end(
    z_mc: data_alloc.NDArray,
    max_nbhgt: data_alloc.NDArray,
    maxslp_avg: data_alloc.NDArray,
    maxhgtd_avg: data_alloc.NDArray,
    c_owner_mask: data_alloc.NDArray,
    thslp_zdiffu: float,
    thhgtd_zdiffu: float,
    nlev: int,
    array_ns: ModuleType = np,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray, data_alloc.NDArray]:
    
    condition1 = array_ns.logical_or(maxslp_avg >= thslp_zdiffu, maxhgtd_avg >= thhgtd_zdiffu)
    cell_mask = array_ns.tile(
        array_ns.where(condition1[:, nlev - 1], c_owner_mask, False), (nlev, 1)
    ).T
    threshold = array_ns.tile(max_nbhgt, (nlev, 1)).T
    owned_cell_above_threshold = array_ns.logical_and(cell_mask, z_mc >= threshold)
    last_true_indices = nlev - 1 - array_ns.argmax(owned_cell_above_threshold[:, ::-1], axis=1)
    # is a zero else value correct?
    kend = array_ns.where(array_ns.any(owned_cell_above_threshold, axis=1), last_true_indices + 1, 0)

    kstart = np.argmax(condition1, axis=1)
    # reset the values where start > end to be an empty range(start, end)
    kstart = array_ns.where(kstart > kend, nlev, kstart)
    cell_index_cell_mask = array_ns.where(kend > kstart, True, False)
    
    return kstart, kend, cell_index_cell_mask


# TODO (@halungge) refactor this to fully get rid of the lists
def compute_diffusion_metrics(
    c2e2c: data_alloc.NDArray,
    z_mc: data_alloc.NDArray,
    max_nbhgt: data_alloc.NDArray,
    c_owner_mask: data_alloc.NDArray,
    maxslp_avg: data_alloc.NDArray,
    maxhgtd_avg: data_alloc.NDArray,
    thslp_zdiffu: float,
    thhgtd_zdiffu: float,
    cell_nudging: int,
    nlev: int,
    array_ns: ModuleType = np,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray, data_alloc.NDArray, data_alloc.NDArray]:
    n_cells = c2e2c.shape[0]
    n_c2e2c = c2e2c.shape[1]
    z_mc_off = z_mc[c2e2c]
    nbidx = array_ns.ones(shape=(n_cells, n_c2e2c, nlev), dtype=int)
    z_vintcoeff = array_ns.zeros(shape=(n_cells, n_c2e2c, nlev))
    mask_hdiff = array_ns.zeros(shape=(n_cells, nlev), dtype=bool)
    mask_hdiff1 = array_ns.zeros(shape=(n_cells, nlev), dtype=bool)
    zd_vertoffset_dsl = array_ns.zeros(shape=(n_cells, n_c2e2c, nlev))
    zd_intcoef_dsl = array_ns.zeros(shape=(n_cells, n_c2e2c, nlev))
    zd_diffcoef_dsl = array_ns.zeros(shape=(n_cells, nlev))
    k_start, k_end, mask = _compute_k_start_end(
        z_mc=z_mc,
        max_nbhgt=max_nbhgt,
        maxslp_avg=maxslp_avg,
        maxhgtd_avg=maxhgtd_avg,
        c_owner_mask=c_owner_mask,
        thslp_zdiffu=thslp_zdiffu,
        thhgtd_zdiffu=thhgtd_zdiffu,
        nlev=nlev,
        array_ns=array_ns,
    )
    #k_index = array_ns.arange(nlev)
    
    #mask_hdiff1= array_ns.where(k_start <= k_index < k_end , True, False)

    # go back to loop for now... then fix _compute_nbidx, _compute_z_vintcoeff
    for jc  in range(cell_nudging, n_cells):
        if k_end[jc] > k_start[jc]:
            k_range = range(k_start[jc], k_end[jc])
            nbidx[jc, :, :] = _compute_nbidx(k_range, z_mc, z_mc_off, nbidx, jc, nlev)
            z_vintcoeff[jc, :, :] = _compute_z_vintcoeff(
                k_range, z_mc, z_mc_off, z_vintcoeff, jc, nlev
            )

            zd_intcoef_dsl[jc, :, k_range] = z_vintcoeff[jc, :, k_range]
            zd_vertoffset_dsl[jc, :, k_range] = (
                nbidx[jc, :, k_range] - array_ns.tile(array_ns.array(k_range), (3, 1)).T
            )
            mask_hdiff[jc, k_range] = True

            zd_diffcoef_dsl_var = array_ns.maximum(
                0.0,
                array_ns.maximum(
                    array_ns.sqrt(array_ns.maximum(0.0, maxslp_avg[jc, k_range] - thslp_zdiffu))
                    / 250.0,
                    2.0e-4
                    * array_ns.sqrt(
                        array_ns.maximum(0.0, maxhgtd_avg[jc, k_range] - thhgtd_zdiffu)
                    ),
                ),
            )
            zd_diffcoef_dsl[jc, k_range] = array_ns.minimum(0.002, zd_diffcoef_dsl_var)

    # flatten first two dims:
    zd_intcoef_dsl = zd_intcoef_dsl.reshape(
        (zd_intcoef_dsl.shape[0] * zd_intcoef_dsl.shape[1],) + zd_intcoef_dsl.shape[2:]
    )
    zd_vertoffset_dsl = zd_vertoffset_dsl.reshape(
        (zd_vertoffset_dsl.shape[0] * zd_vertoffset_dsl.shape[1],) + zd_vertoffset_dsl.shape[2:]
    )
    #assert array_ns.allclose(
    #    mask_hdiff[cell_nudging:], mask_hdiff1[cell_nudging:]
    #), "mask_hdiff does not match"
    return mask_hdiff, zd_diffcoef_dsl, zd_intcoef_dsl, zd_vertoffset_dsl
