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

from icon4py.model.common.constants import CVD_O_RD, P0REF, RD
from icon4py.model.common.dimension import E2CDim, EdgeDim
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex
from icon4py.model.common.grid.icon import IconGrid


def edge_2_cell_vector_rbf_interpolation_numpy(
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


def cell_2_edge_interpolation_numpy(
    icon_grid: IconGrid,
    cells2edges_interpolation_coeff: np.array,
    cell_scalar: np.array,
    horizontal_start_index: int,
    horizontal_end_index: int,
    vertical_start: int,
    vertical_end: int,
):
    """
    cells2edges_scalar in mo_icon_interpolation.f90
    """
    assert horizontal_start_index != HorizontalMarkerIndex.lateral_boundary(
        EdgeDim
    ), "boundary edges cannot be obtained because there is only one neighboring cell"
    mask = np.zeros((icon_grid.num_edges, icon_grid.num_levels), dtype=bool)
    mask[horizontal_start_index:horizontal_end_index, vertical_start:vertical_end] = True
    e2c = icon_grid.connectivities[E2CDim]
    cells2edges_interpolation_coeff = np.expand_dims(cells2edges_interpolation_coeff, axis=-1)
    edge_scalar = np.where(
        mask, np.sum(cell_scalar[e2c] * cells2edges_interpolation_coeff, axis=1), 0.0
    )
    return edge_scalar


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
