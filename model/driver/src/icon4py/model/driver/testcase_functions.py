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

from icon4py.model.common.constants import CPD_O_RD, CVD_O_RD, GRAV_O_RD, P0REF, RD
from icon4py.model.common.dimension import E2CDim
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.settings import xp


def mo_rbf_vec_interpol_cell_numpy(
    p_e_in: np.array,
    ptr_coeff_1: np.array,
    ptr_coeff_2: np.array,
    c2e2c2e: np.array,
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
) -> tuple[np.array, np.array]:
    expanded_ptr_coeff_1 = np.expand_dims(ptr_coeff_1[0:horizontal_end, :], axis=-1)
    expanded_ptr_coeff_2 = np.expand_dims(ptr_coeff_2[0:horizontal_end, :], axis=-1)
    mask = np.ones(c2e2c2e.shape[0], dtype=bool)
    mask[horizontal_end:] = False
    mask[0:horizontal_start] = False
    mask = np.repeat(np.expand_dims(mask, axis=-1), p_e_in.shape[1], axis=1)
    mask[:, vertical_end:] = False
    mask[:, 0:vertical_start] = False
    p_u_out = np.where(mask, np.sum(p_e_in[c2e2c2e] * expanded_ptr_coeff_1, axis=1), 0.0)
    p_v_out = np.where(mask, np.sum(p_e_in[c2e2c2e] * expanded_ptr_coeff_2, axis=1), 0.0)
    return p_u_out, p_v_out


def mo_cells2edges_scalar_numpy(
    grid: IconGrid,
    cells2edges_interpolation_coeff: np.array,
    cell_scalar: np.array,
    mask: np.array,
):
    e2c = grid.connectivities[E2CDim]
    cells2edges_interpolation_coeff = np.expand_dims(cells2edges_interpolation_coeff, axis=-1)
    mask = np.repeat(np.expand_dims(mask, axis=-1), cell_scalar.shape[1], axis=1)
    edge_scalar = np.where(
        mask, np.sum(cell_scalar[e2c] * cells2edges_interpolation_coeff, axis=1), 0.0
    )
    return edge_scalar


def mo_u2vn_jabw_numpy(
    jw_u0: float,
    jw_up: float,
    latC: float,
    lonC: float,
    edge_lat: np.array,
    edge_lon: np.array,
    primal_normal_x: np.array,
    eta_v_e: np.array,
    mask: np.array,
):
    mask = np.repeat(np.expand_dims(mask, axis=-1), eta_v_e.shape[1], axis=1)
    edge_lat = np.repeat(np.expand_dims(edge_lat, axis=-1), eta_v_e.shape[1], axis=1)
    edge_lon = np.repeat(np.expand_dims(edge_lon, axis=-1), eta_v_e.shape[1], axis=1)
    primal_normal_x = np.repeat(np.expand_dims(primal_normal_x, axis=-1), eta_v_e.shape[1], axis=1)
    u = np.where(mask, jw_u0 * (np.cos(eta_v_e) ** 1.5) * (np.sin(2.0 * edge_lat) ** 2), 0.0)
    if jw_up > 1.0e-20:
        u = np.where(
            mask,
            u
            + jw_up
            * np.exp(
                -10.0
                * np.arccos(
                    np.sin(latC) * np.sin(edge_lat)
                    + np.cos(latC) * np.cos(edge_lat) * np.cos(edge_lon - lonC)
                )
                ** 2
            ),
            u,
        )
    vn = u * primal_normal_x

    return vn


def hydrostatic_adjustment_ndarray(
    wgtfac_c: np.ndarray,
    ddqz_z_half: np.ndarray,
    exner_ref_mc: np.ndarray,
    d_exner_dz_ref_ic: np.ndarray,
    theta_ref_mc: np.ndarray,
    theta_ref_ic: np.ndarray,
    rho: np.ndarray,
    exner: np.ndarray,
    theta_v: np.ndarray,
    num_levels: int,
):
    # virtual temperature
    temp_v = theta_v * exner

    for k in range(num_levels - 2, -1, -1):
        fac1 = (
            wgtfac_c[:, k + 1] * (temp_v[:, k + 1] - theta_ref_mc[:, k + 1] * exner[:, k + 1])
            - (1.0 - wgtfac_c[:, k + 1]) * theta_ref_mc[:, k] * exner[:, k + 1]
        )
        fac2 = (1.0 - wgtfac_c[:, k + 1]) * temp_v[:, k] * exner[:, k + 1]
        fac3 = exner_ref_mc[:, k + 1] - exner_ref_mc[:, k] - exner[:, k + 1]

        quadratic_a = (theta_ref_ic[:, k + 1] * exner[:, k + 1] + fac1) / ddqz_z_half[:, k + 1]
        quadratic_b = -(
            quadratic_a * fac3 + fac2 / ddqz_z_half[:, k + 1] + fac1 * d_exner_dz_ref_ic[:, k + 1]
        )
        quadratic_c = -(fac2 * fac3 / ddqz_z_half[:, k + 1] + fac2 * d_exner_dz_ref_ic[:, k + 1])

        exner[:, k] = (quadratic_b + xp.sqrt(quadratic_b**2 + 4.0 * quadratic_a * quadratic_c)) / (
            2.0 * quadratic_a
        )
        theta_v[:, k] = temp_v[:, k] / exner[:, k]
        rho[:, k] = exner[:, k] ** CVD_O_RD * P0REF / (RD * theta_v[:, k])

    return rho, exner, theta_v


def mo_diagnose_temperature_numpy(
    theta_v: np.array,
    exner: np.array,
) -> np.array:
    temperature = theta_v * exner
    return temperature


def mo_diagnose_pressure_sfc_numpy(
    exner: np.array,
    temperature: np.array,
    ddqz_z_full: np.array,
    num_levels: int,
) -> np.array:
    pressure_sfc = P0REF * np.exp(
        CPD_O_RD * np.log(exner[:, num_levels - 3])
        + GRAV_O_RD
        * (
            ddqz_z_full[:, num_levels - 1] / temperature[:, num_levels - 1]
            + ddqz_z_full[:, num_levels - 2] / temperature[:, num_levels - 2]
            + 0.5 * ddqz_z_full[:, num_levels - 3] / temperature[:, num_levels - 3]
        )
    )
    return pressure_sfc


def mo_diagnose_pressure_numpy(
    pressure_sfc: np.array,
    temperature: np.array,
    ddqz_z_full: np.array,
    cell_size: int,
    num_levels: int,
) -> tuple[np.array, np.array]:
    pressure_ifc = np.zeros((cell_size, num_levels), dtype=float)
    pressure = np.zeros((cell_size, num_levels), dtype=float)
    pressure_ifc[:, num_levels - 1] = pressure_sfc * np.exp(
        -ddqz_z_full[:, num_levels - 1] / temperature[:, num_levels - 1]
    )
    pressure[:, num_levels - 1] = np.sqrt(pressure_ifc[:, num_levels - 1] * pressure_sfc)
    for k in range(num_levels - 2, -1, -1):
        pressure_ifc[:, k] = pressure_ifc[:, k + 1] * np.exp(-ddqz_z_full[:, k] / temperature[:, k])
        pressure[:, k] = np.sqrt(pressure_ifc[:, k] * pressure_ifc[:, k + 1])
    return pressure, pressure_ifc
