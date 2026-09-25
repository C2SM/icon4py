# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping
from typing import Any

import gt4py.next as gtx
import numpy as np

from icon4py.model.common import constants, dimension as dims


def enhanced_smagorinski_factor_numpy(
    factor_in: tuple[float, ...], heigths_in: tuple[float, ...], a_vec: np.ndarray
) -> float:
    alin = (factor_in[1] - factor_in[0]) / (heigths_in[1] - heigths_in[0])
    df32 = factor_in[2] - factor_in[1]
    df42 = factor_in[3] - factor_in[1]
    dz32 = heigths_in[2] - heigths_in[1]
    dz42 = heigths_in[3] - heigths_in[1]
    bqdr = (df42 * dz32 - df32 * dz42) / (dz32 * dz42 * (dz42 - dz32))
    aqdr = df32 / dz32 - bqdr * dz32
    zf = 0.5 * (a_vec[:-1] + a_vec[1:])
    max0 = np.maximum(0.0, zf - heigths_in[0])
    dzlin = np.minimum(heigths_in[1] - heigths_in[0], max0)
    max1 = np.maximum(0.0, zf - heigths_in[1])
    dzqdr = np.minimum(heigths_in[3] - heigths_in[1], max1)
    return factor_in[0] + dzlin * alin + dzqdr * (aqdr + dzqdr * bqdr)


def nabla2_on_cell_numpy(
    connectivities: Mapping[type[gtx.NeighborConnectivity], np.ndarray],
    psi_c: np.ndarray,
    geofac_n2s: np.ndarray,
) -> np.ndarray:
    c2e2cO = connectivities[dims.C2E2CO]
    nabla2_psi_c = np.sum(np.where((c2e2cO != -1), psi_c[c2e2cO] * geofac_n2s, 0), axis=1)
    return nabla2_psi_c


def nabla2_on_cell_k_numpy(
    connectivities: Mapping[type[gtx.NeighborConnectivity], np.ndarray],
    psi_c: np.ndarray,
    geofac_n2s: np.ndarray,
) -> np.ndarray:
    c2e2cO = connectivities[dims.C2E2CO]
    geofac_n2s = np.expand_dims(geofac_n2s, axis=-1)
    nabla2_psi_c = np.sum(
        np.where((c2e2cO != -1)[:, :, np.newaxis], psi_c[c2e2cO] * geofac_n2s, 0), axis=1
    )
    return nabla2_psi_c


def compute_tangential_wind_numpy(
    connectivities: Mapping[type[gtx.NeighborConnectivity], np.ndarray],
    vn: np.ndarray,
    rbf_vec_coeff_e: np.ndarray,
) -> np.ndarray:
    """RBF interpolation of the normal wind to the edge-tangential direction."""
    rbf_vec_coeff_e = np.expand_dims(rbf_vec_coeff_e, axis=-1)
    e2c2e = connectivities[dims.E2C2E]
    return np.sum(np.where((e2c2e != -1)[:, :, np.newaxis], vn[e2c2e] * rbf_vec_coeff_e, 0), axis=1)


def interpolate_to_cell_center_numpy(
    connectivities: Mapping[type[gtx.NeighborConnectivity], np.ndarray],
    interpolant: np.ndarray,
    e_bln_c_s: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    """Interpolate an edge field to the cell centers with the bilinear C2E weights."""
    e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
    c2e = connectivities[dims.C2E]
    return np.sum(interpolant[c2e] * e_bln_c_s, axis=1)


def interpolate_cell_field_to_vertex_numpy(
    connectivities: Mapping[type[gtx.NeighborConnectivity], np.ndarray],
    cell_field: np.ndarray,
    c_intp: np.ndarray,
) -> np.ndarray:
    v2c = connectivities[dims.V2C]
    c_intp = np.expand_dims(c_intp, axis=-1)
    return np.sum(np.where((v2c != -1)[:, :, np.newaxis], cell_field[v2c] * c_intp, 0), axis=1)


def compute_curl_numpy(
    connectivities: Mapping[type[gtx.NeighborConnectivity], np.ndarray],
    edge_field: np.ndarray,
    geofac_rot: np.ndarray,
) -> np.ndarray:
    v2e = connectivities[dims.V2E]
    geofac_rot = np.expand_dims(geofac_rot, axis=-1)
    return np.sum(np.where((v2e != -1)[:, :, np.newaxis], edge_field[v2e] * geofac_rot, 0), axis=1)


def compute_dry_static_energy_numpy(
    temperature: np.ndarray,
    height_above_ground: np.ndarray,
    *,
    grav: float,
) -> np.ndarray:
    return constants.CPD * temperature + grav * height_above_ground


def compute_virtual_potential_temperature_numpy(
    virtual_temperature: np.ndarray,
    pressure: np.ndarray,
) -> np.ndarray:
    return virtual_temperature * (constants.P0REF / pressure) ** constants.RD_O_CPD


def compute_brunt_vaisala_frequency_numpy(
    theta_v: np.ndarray,
    wgtfac_c: np.ndarray,
    inv_ddqz_z_half: np.ndarray,
    *,
    grav: float,
) -> np.ndarray:
    nlev = theta_v.shape[1]
    bruvais = np.zeros((theta_v.shape[0], nlev + 1), dtype=theta_v.dtype)
    # Fortran jk = 2..nlev (1-based) -> k = 1..nlev-1 (0-based); the top and
    # bottom half levels (k = 0 and k = nlev) stay untouched (zero-initialized).
    theta_v_ic = (
        wgtfac_c[:, 1:nlev] * theta_v[:, 1:nlev]
        + (1.0 - wgtfac_c[:, 1:nlev]) * theta_v[:, 0 : nlev - 1]
    )
    bruvais[:, 1:nlev] = (
        grav
        * (theta_v[:, 0 : nlev - 1] - theta_v[:, 1:nlev])
        * inv_ddqz_z_half[:, 1:nlev]
        / theta_v_ic
    )
    return bruvais
