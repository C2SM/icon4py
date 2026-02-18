# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy.linalg.lapack import lapack  # type: ignore[attr-defined]

from icon4py.model.common.grid import base as base_grid
from icon4py.model.common.math.projection import (
    gnomonic_proj_single_val,
    plane_torus_closest_coordinates,
)
from icon4py.model.common.utils import data_allocation as data_alloc


def compute_lsq_pseudoinv(
    cell_owner_mask: data_alloc.NDArray,
    lsq_pseudoinv: data_alloc.NDArray,
    z_lsq_mat_c: data_alloc.NDArray,
    lsq_weights_c: data_alloc.NDArray,
    start_idx: int,
    min_rlcell_int: int,
    lsq_dim_unk: int,
    lsq_dim_c: int,
) -> data_alloc.NDArray:
    for jjb in range(lsq_dim_c):
        for jjk in range(lsq_dim_unk):
            for jc in range(start_idx, min_rlcell_int):
                u, s, v_t, _ = lapack.dgesdd(z_lsq_mat_c[jc, :, :])
                if cell_owner_mask[jc]:
                    lsq_pseudoinv[jc, :lsq_dim_unk, jjb] = (
                        lsq_pseudoinv[jc, :lsq_dim_unk, jjb]
                        + v_t[jjk, :lsq_dim_unk] / s[jjk] * u[jjb, jjk] * lsq_weights_c[jc, jjb]
                    )
    return lsq_pseudoinv


def compute_lsq_weights_c(
    z_dist_g: data_alloc.NDArray,
    lsq_weights_c_jc: data_alloc.NDArray,
    lsq_dim_stencil: int,
    lsq_wgt_exp: int,
) -> data_alloc.NDArray:
    for js in range(lsq_dim_stencil):
        z_norm = np.sqrt(np.dot(z_dist_g[js, :], z_dist_g[js, :]))
        lsq_weights_c_jc[js] = 1.0 / (z_norm**lsq_wgt_exp)
    return lsq_weights_c_jc / np.max(lsq_weights_c_jc)


def compute_z_lsq_mat_c(
    cell_owner_mask: data_alloc.NDArray,
    z_lsq_mat_c: data_alloc.NDArray,
    lsq_weights_c: data_alloc.NDArray,
    z_dist_g: data_alloc.NDArray,
    jc: int,
    lsq_dim_unk: int,
    lsq_dim_c: int,
) -> data_alloc.NDArray:
    min_lsq_bound = min(lsq_dim_unk, lsq_dim_c)
    if cell_owner_mask[jc]:
        z_lsq_mat_c[jc, :min_lsq_bound, :min_lsq_bound] = 1.0

    for js in range(lsq_dim_c):
        z_lsq_mat_c[jc, js, :lsq_dim_unk] = lsq_weights_c[jc, js] * z_dist_g[js, :]

    return z_lsq_mat_c[jc, js, :lsq_dim_unk]


def lsq_compute_coeffs(
    cell_center_x: data_alloc.NDArray,
    cell_center_y: data_alloc.NDArray,
    cell_lat: data_alloc.NDArray,
    cell_lon: data_alloc.NDArray,
    c2e2c: data_alloc.NDArray,
    cell_owner_mask: data_alloc.NDArray,
    domain_length: float,
    domain_height: float,
    grid_sphere_radius: float,
    lsq_dim_unk: int,
    lsq_dim_c: int,
    lsq_wgt_exp: int,
    lsq_dim_stencil: int,
    start_idx: int,
    min_rlcell_int: int,
    geometry_type: base_grid.GeometryType,
) -> data_alloc.NDArray:
    lsq_weights_c = np.zeros((min_rlcell_int, lsq_dim_stencil))
    lsq_pseudoinv = np.zeros((min_rlcell_int, lsq_dim_unk, lsq_dim_c))
    z_lsq_mat_c = np.zeros((min_rlcell_int, lsq_dim_c, lsq_dim_c))

    for jc in range(start_idx, min_rlcell_int):
        match geometry_type:
            case base_grid.GeometryType.ICOSAHEDRON:
                z_dist_g = np.zeros((lsq_dim_c, 2))
                for js in range(lsq_dim_stencil):
                    z_dist_g[js, :] = gnomonic_proj_single_val(
                        cell_lon[jc], cell_lat[jc], cell_lon[c2e2c[jc, js]], cell_lat[c2e2c[jc, js]]
                    )
                z_dist_g *= grid_sphere_radius

                min_lsq_bound = min(lsq_dim_unk, lsq_dim_c)
                if cell_owner_mask[jc]:
                    z_lsq_mat_c[jc, :min_lsq_bound, :min_lsq_bound] = 1.0
            case base_grid.GeometryType.TORUS:
                ilc_s = c2e2c[jc, :lsq_dim_stencil]
                cc_cell = np.zeros((lsq_dim_stencil, 2))

                cc_cv = (cell_center_x[jc], cell_center_y[jc])
                for js in range(lsq_dim_stencil):
                    cc_cell[js, :] = plane_torus_closest_coordinates(
                        cell_center_x[jc],
                        cell_center_y[jc],
                        cell_center_x[ilc_s][js],
                        cell_center_y[ilc_s][js],
                        domain_length,
                        domain_height,
                    )
                z_dist_g = cc_cell - cc_cv

        lsq_weights_c[jc, :] = compute_lsq_weights_c(
            z_dist_g, lsq_weights_c[jc, :], lsq_dim_stencil, lsq_wgt_exp
        )
        z_lsq_mat_c[jc, js, :lsq_dim_unk] = compute_z_lsq_mat_c(
            cell_owner_mask, z_lsq_mat_c, lsq_weights_c, z_dist_g, jc, lsq_dim_unk, lsq_dim_c
        )

    lsq_pseudoinv = compute_lsq_pseudoinv(
        cell_owner_mask,
        lsq_pseudoinv,
        z_lsq_mat_c,
        lsq_weights_c,
        start_idx,
        min_rlcell_int,
        lsq_dim_unk,
        lsq_dim_c,
    )

    return lsq_pseudoinv
