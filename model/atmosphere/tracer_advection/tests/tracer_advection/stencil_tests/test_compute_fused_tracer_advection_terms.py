# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone numpy reference functions for stencils used in the fused tracer advection programs
that do not have their own individual stencil test files.

These functions are imported by the fused program test files.
"""

from collections.abc import Mapping

import gt4py.next as gtx
import numpy as np

from icon4py.model.common import dimension as dims


def apply_density_increment_numpy(  # noqa: PLR0917
    rhodz_in: np.ndarray,
    p_mflx_contra_v: np.ndarray,
    deepatmo_divzl: np.ndarray,
    deepatmo_divzu: np.ndarray,
    p_dtime: float,
    even_timestep: bool,
) -> np.ndarray:
    """Compute the updated air-mass column density after the vertical mass flux increment.

    p_mflx_contra_v has shape (ncells, nlev+1) on KHalfDim.
    At full level k:
      - lower interface (k+1 in KHalfDim) contributes with deepatmo_divzl
      - upper interface (k   in KHalfDim) contributes with deepatmo_divzu
    """
    rhodz_incr = p_dtime * (
        p_mflx_contra_v[:, 1:] * deepatmo_divzl - p_mflx_contra_v[:, :-1] * deepatmo_divzu
    )
    if even_timestep:
        rhodz_out = rhodz_in + rhodz_incr
    else:
        rhodz_out = np.maximum(0.1 * rhodz_in, rhodz_in) - rhodz_incr
    return rhodz_out


def apply_positive_definite_horizontal_multiplicative_flux_factor_numpy(
    r_m: np.ndarray,
    p_mflx_tracer_h: np.ndarray,
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
) -> np.ndarray:
    """Scale horizontal tracer fluxes by the positive-definite limiter factor r_m.

    Upwind fluxes (p_mflx_tracer_h >= 0) use the upwind cell's r_m (E2C[0]),
    downwind fluxes use the downwind cell's r_m (E2C[1]).
    """
    e2c = connectivities[dims.E2C]
    return np.where(
        p_mflx_tracer_h >= 0.0,
        p_mflx_tracer_h * r_m[e2c[:, 0]],
        p_mflx_tracer_h * r_m[e2c[:, 1]],
    )


def reconstruct_linear_coefficients_svd_numpy(
    p_cc: np.ndarray,
    lsq_pseudoinv_1: np.ndarray,
    lsq_pseudoinv_2: np.ndarray,
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct linear LSQ reconstruction coefficients via SVD pseudo-inverse.
    """
    c2e2c = connectivities[dims.C2E2C]
    lsq_pseudoinv_1_exp = np.expand_dims(lsq_pseudoinv_1, axis=-1)
    lsq_pseudoinv_2_exp = np.expand_dims(lsq_pseudoinv_2, axis=-1)
    # p_cc[c2e2c] has shape (ncells, 3, nlevels)
    p_cc_neighbors = p_cc[c2e2c]
    # diff: (ncells, 3, nlevels) = neighbor values minus center value
    diff = p_cc_neighbors - p_cc[:, np.newaxis, :]
    p_coeff_1 = p_cc
    p_coeff_2 = np.sum(lsq_pseudoinv_1_exp * diff, axis=1)
    p_coeff_3 = np.sum(lsq_pseudoinv_2_exp * diff, axis=1)
    return p_coeff_1, p_coeff_2, p_coeff_3


def compute_tangential_wind_numpy(
    vn: np.ndarray,
    rbf_vec_coeff_e: np.ndarray,
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
) -> np.ndarray:
    """Reconstruct tangential wind from normal components via RBF interpolation.

    Computes: z_real_vt[e, k] = sum_{e' in E2C2E} rbf_vec_coeff_e[e, e'] * vn[e', k]

    vn has shape (nedges, nlevels).
    rbf_vec_coeff_e has shape (nedges, 4) (E2C2EDim = 4 neighbors).
    Returns z_real_vt of shape (nedges, nlevels).
    """
    e2c2e = connectivities[dims.E2C2E]
    # rbf_vec_coeff_e_exp: (nedges, 4, 1) for broadcasting over K
    rbf_vec_coeff_e_exp = np.expand_dims(rbf_vec_coeff_e, axis=-1)
    return np.sum(rbf_vec_coeff_e_exp * vn[e2c2e], axis=1)
