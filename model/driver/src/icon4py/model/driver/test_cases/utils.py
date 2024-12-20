# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np

from icon4py.model.common import (
    constants as phy_const,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid


def hydrostatic_adjustment_numpy(
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

        exner[:, k] = (quadratic_b + np.sqrt(quadratic_b**2 + 4.0 * quadratic_a * quadratic_c)) / (
            2.0 * quadratic_a
        )
        theta_v[:, k] = temp_v[:, k] / exner[:, k]
        rho[:, k] = (
            exner[:, k] ** phy_const.CVD_O_RD * phy_const.P0REF / (phy_const.RD * theta_v[:, k])
        )

    return rho, exner, theta_v


def hydrostatic_adjustment_constant_thetav_numpy(
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes a hydrostatically balanced profile. In constrast to the above
    hydrostatic_adjustment_numpy, the virtual temperature is kept (assumed)
    constant during the adjustment, leading to a simpler formula.
    """

    for k in range(num_levels - 2, -1, -1):
        theta_v_pr_ic = wgtfac_c[:, k + 1] * (theta_v[:, k + 1] - theta_ref_mc[:, k + 1]) + (
            1.0 - wgtfac_c[:, k + 1]
        ) * (theta_v[:, k] - theta_ref_mc[:, k])

        exner[:, k] = (
            exner[:, k + 1]
            + (exner_ref_mc[:, k] - exner_ref_mc[:, k + 1])
            - ddqz_z_half[:, k + 1]
            / (theta_v_pr_ic + theta_ref_ic[:, k + 1])
            * theta_v_pr_ic
            * d_exner_dz_ref_ic[:, k + 1]
        )

    for k in range(num_levels - 1, -1, -1):
        rho[:, k] = (
            exner[:, k] ** phy_const.CVD_O_RD * phy_const.P0REF / (phy_const.RD * theta_v[:, k])
        )

    return rho, exner


def zonalwind_2_normalwind_numpy(
    grid: icon_grid.IconGrid,
    jw_u0: float,
    jw_up: float,
    lat_perturbation_center: float,
    lon_perturbation_center: float,
    edge_lat: np.ndarray,
    edge_lon: np.ndarray,
    primal_normal_x: np.ndarray,
    eta_v_e: np.ndarray,
):
    """
    Compute normal wind at edge center from vertical eta coordinate (eta_v_e).

    Args:
        grid: IconGrid
        jw_u0: base zonal wind speed factor
        jw_up: perturbation amplitude
        lat_perturbation_center: perturbation center in latitude
        lon_perturbation_center: perturbation center in longitude
        edge_lat: edge center latitude
        edge_lon: edge center longitude
        primal_normal_x: zonal component of primal normal vector at edge center
        eta_v_e: vertical eta coordinate at edge center
    Returns: normal wind
    """
    # TODO (Chia Rui) this function needs a test

    mask = np.ones((grid.num_edges, grid.num_levels), dtype=bool)
    mask[
        0 : grid.end_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)),
        :,
    ] = False
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
                -(
                    (
                        10.0
                        * np.arccos(
                            np.sin(lat_perturbation_center) * np.sin(edge_lat)
                            + np.cos(lat_perturbation_center)
                            * np.cos(edge_lat)
                            * np.cos(edge_lon - lon_perturbation_center)
                        )
                    )
                    ** 2
                )
            ),
            u,
        )
    vn = u * primal_normal_x

    return vn


# TODO (Chia Rui): Can this kind of simple arithmetic operation be replaced by a more general stencil that does the same operation on a general field?
@gtx.field_operator
def _compute_perturbed_exner(
    exner: fa.CellKField[ta.wpfloat],
    exner_ref: fa.CellKField[ta.vpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the perturbed exner function (exner_pr).
        exner_pr = exner - exner_ref
    This stencil is copied from subroutine compute_exner_pert in mo_nh_init_utils in ICON. It should be called
    during the initialization to initialize exner_pr of DiagnosticStateHydro if the model does not restart from
    a restart file.

    Args:
        exner: exner function
        exner_ref: reference exner function
    Returns:
        Perturbed exner function
    """
    exner_pr = exner - exner_ref
    return exner_pr


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_perturbed_exner(
    exner: fa.CellKField[ta.wpfloat],
    exner_ref: fa.CellKField[ta.vpfloat],
    exner_pr: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_perturbed_exner(
        exner,
        exner_ref,
        out=exner_pr,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
