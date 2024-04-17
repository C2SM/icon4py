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
from icon4py.model.common.dimension import EdgeDim, E2CDim
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex


def interpolation_rbf_edges2cells_vector_numpy(
    p_e_in: np.array,
    ptr_coeff_1: np.array,
    ptr_coeff_2: np.array,
    c2e2c2e: np.array,
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
) -> tuple[np.array, np.array]:
    expanded_ptr_coeff_1 = np.expand_dims(ptr_coeff_1[horizontal_start:horizontal_end, :], axis=-1)
    expanded_ptr_coeff_2 = np.expand_dims(ptr_coeff_2[horizontal_start:horizontal_end, :], axis=-1)
    mask = np.zeros((c2e2c2e.shape[0], p_e_in.shape[1]), dtype=bool)
    mask[horizontal_start:horizontal_end, vertical_start:vertical_end] = True
    p_u_out = np.where(mask, np.sum(p_e_in[c2e2c2e] * expanded_ptr_coeff_1, axis=1), 0.0)
    p_v_out = np.where(mask, np.sum(p_e_in[c2e2c2e] * expanded_ptr_coeff_2, axis=1), 0.0)
    return p_u_out, p_v_out


def interpolation_cells2edges_scalar_numpy(
    icon_grid: IconGrid,
    cells2edges_interpolation_coeff: np.array,
    cell_scalar: np.array,
    horizontal_start_index: int,
    horizontal_end_index: int,
    vertical_start: int,
    vertical_end: int,
):
    """mask = np.repeat(np.expand_dims(mask, axis=-1), cell_scalar.shape[1], axis=1)"""
    """
    cells2edges_scalar in mo_icon_interpolation.f90
    """
    assert horizontal_start_index != HorizontalMarkerIndex.lateral_boundary(EdgeDim), "boundary edges cannot be obtained because there is only one neighboring cell"
    mask = np.zeros((icon_grid.num_edges, icon_grid.num_levels), dtype=bool)
    mask[horizontal_start_index:horizontal_end_index, vertical_start:vertical_end] = True
    e2c = icon_grid.connectivities[E2CDim]
    cells2edges_interpolation_coeff = np.expand_dims(cells2edges_interpolation_coeff, axis=-1)
    edge_scalar = np.where(
        mask, np.sum(cell_scalar[e2c] * cells2edges_interpolation_coeff, axis=1), 0.0
    )
    return edge_scalar


def zonal2normalwind_jabw_numpy(
    icon_grid: IconGrid,
    jw_u0: float,
    jw_up: float,
    lat_perturbation_center: float,
    lon_perturbation_center: float,
    edge_lat: np.array,
    edge_lon: np.array,
    primal_normal_x: np.array,
    eta_v_e: np.array,
):
    """mask = np.repeat(np.expand_dims(mask, axis=-1), eta_v_e.shape[1], axis=1)"""
    mask = np.ones((icon_grid.num_edges, icon_grid.num_levels), dtype=bool)
    mask[0:icon_grid.get_end_index(EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1), :] = False
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
                    np.sin(lat_perturbation_center) * np.sin(edge_lat)
                    + np.cos(lat_perturbation_center) * np.cos(edge_lat) * np.cos(edge_lon - lon_perturbation_center)
                )
                ** 2
            ),
            u,
        )
    vn = u * primal_normal_x

    return vn


def hydrostatic_adjustment_numpy(
    wgtfac_c: np.array,
    ddqz_z_half: np.array,
    exner_ref_mc: np.array,
    d_exner_dz_ref_ic: np.array,
    theta_ref_mc: np.array,
    theta_ref_ic: np.array,
    rho: np.array,
    exner: np.array,
    theta_v: np.array,
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

        exner[:, k] = (quadratic_b + np.sqrt(quadratic_b**2 + 4.0 * quadratic_a * quadratic_c)) / (
            2.0 * quadratic_a
        )
        theta_v[:, k] = temp_v[:, k] / exner[:, k]
        rho[:, k] = exner[:, k] ** CVD_O_RD * P0REF / (RD * theta_v[:, k])

    return rho, exner, theta_v


# TODO (Chia Rui): Construct a proper test for these diagnostic stencils
def diagnose_temperature_numpy(
    theta_v: np.array,
    exner: np.array,
) -> np.array:
    temperature = theta_v * exner
    return temperature


def diagnose_pressure_sfc_numpy(
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


def diagnose_pressure_numpy(
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
