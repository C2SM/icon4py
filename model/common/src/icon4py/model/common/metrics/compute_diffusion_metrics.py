# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from types import ModuleType

import gt4py.next as gtx
import numpy as np

from icon4py.model.common.utils import data_allocation as data_alloc


def compute_max_nbhgt_array_ns(
    c2e2c: data_alloc.NDArray,
    z_mc: data_alloc.NDArray,
    nlev: int,
    exchange: Callable[[data_alloc.NDArray], None],
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    z_mc_nlev = z_mc[:, nlev - 1]
    max_nbhgt_0_1 = array_ns.maximum(z_mc_nlev[c2e2c[:, 0]], z_mc_nlev[c2e2c[:, 1]])
    max_nbhgt = array_ns.maximum(max_nbhgt_0_1, z_mc_nlev[c2e2c[:, 2]])
    exchange(max_nbhgt)
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
    kend = array_ns.where(
        array_ns.any(owned_cell_above_threshold, axis=1), last_true_indices + 1, 0
    )
    kstart = np.argmax(condition1, axis=1)
    # reset the values where start > end to be an empty range(start, end)
    kstart = array_ns.where(kstart > kend, nlev, kstart)
    cell_index_mask = array_ns.where(kend > kstart, True, False)

    return kstart, kend, cell_index_mask


def compute_diffusion_mask_and_coef(
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
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    n_cells = c2e2c.shape[0]
    zd_diffcoef = array_ns.zeros(shape=(n_cells, nlev))
    k_start, k_end, _ = _compute_k_start_end(
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

    # go back to loop for now... fix _compute_nbidx, _compute_z_vintcoeff later
    for jc in range(cell_nudging, n_cells):
        kend = k_end[jc].item()
        kstart = k_start[jc].item()
        if kend > kstart:
            k_range = range(kstart, kend)

            zd_diffcoef_var = array_ns.maximum(
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
            zd_diffcoef[jc, k_range] = array_ns.minimum(0.002, zd_diffcoef_var)

    return zd_diffcoef


def compute_diffusion_intcoef_and_vertoffset(
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
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    n_cells = c2e2c.shape[0]
    n_c2e2c = c2e2c.shape[1]
    z_mc_off = z_mc[c2e2c]  # (n_cells, n_c2e2c, nlev)

    zd_vertoffset = array_ns.zeros(shape=(n_cells, n_c2e2c, nlev), dtype=gtx.int32)
    zd_intcoef = array_ns.zeros(shape=(n_cells, n_c2e2c, nlev))

    k_start, k_end, _ = _compute_k_start_end(
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

    # Identify active cells (those with non-empty k_range above nudging boundary)
    cell_indices = array_ns.arange(n_cells)
    active = array_ns.where((k_end > k_start) & (cell_indices >= cell_nudging))[0]
    n_active = len(active)

    if n_active > 0:
        z_mc_active = z_mc[active, :]  # (n_active, nlev)
        kstart_active = k_start[active]
        kend_active = k_end[active]

        # Level mask: True where level is in [kstart, kend) for each active cell
        level_idx = array_ns.arange(nlev)[array_ns.newaxis, :]  # (1, nlev)
        k_mask = (level_idx >= kstart_active[:, array_ns.newaxis]) & (
            level_idx < kend_active[:, array_ns.newaxis]
        )  # (n_active, nlev)

        for ind in range(n_c2e2c):
            profile = z_mc_off[active, ind, :]  # (n_active, nlev), monotonically decreasing

            # Row-wise searchsorted using offset trick:
            # Negate profiles (ascending) and add per-row offsets to separate rows
            neg_profile = -profile
            neg_query = -z_mc_active

            val_range = max(float(neg_profile.max() - neg_profile.min()), 1.0)
            row_offsets = array_ns.arange(n_active, dtype=array_ns.float64) * val_range

            flat_sorted = (neg_profile + row_offsets[:, array_ns.newaxis]).ravel()
            flat_query = (neg_query + row_offsets[:, array_ns.newaxis]).ravel()

            flat_idx = array_ns.searchsorted(flat_sorted, flat_query, side="right")
            raw_idx = (
                flat_idx.reshape(n_active, nlev)
                - array_ns.arange(n_active)[:, array_ns.newaxis] * nlev
                - 1
            )

            # Mark entries where no valid bracketing interval was found:
            # raw_idx < 0 means query is above the entire profile,
            # raw_idx > nlev-2 means query is below the entire profile.
            valid_bracket = (raw_idx >= 0) & (raw_idx <= nlev - 2)
            idx_2d = array_ns.clip(raw_idx, 0, nlev - 2)

            # Interpolation coefficients
            upper = array_ns.take_along_axis(profile, idx_2d, axis=1)
            lower = array_ns.take_along_axis(profile, idx_2d + 1, axis=1)
            denom = upper - lower
            denom = array_ns.where(denom == 0.0, 1.0, denom)
            vintcoeff = (z_mc_active - lower) / denom

            # For invalid brackets, use original defaults: vintcoeff=0, nbidx=1
            vintcoeff = array_ns.where(valid_bracket, vintcoeff, 0.0)
            idx_2d = array_ns.where(valid_bracket, idx_2d, 1)

            # Write results only for valid k_range levels
            zd_intcoef[active[:, array_ns.newaxis], ind, level_idx] = array_ns.where(
                k_mask, vintcoeff, 0.0
            )
            zd_vertoffset[active[:, array_ns.newaxis], ind, level_idx] = array_ns.where(
                k_mask, (idx_2d - level_idx).astype(gtx.int32), gtx.int32(0)
            )

    return zd_intcoef, zd_vertoffset
