# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from types import ModuleType

import gt4py.next as gtx
import numpy as np

from icon4py.model.common import constants as phy_const, dimension as dims
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.math import helpers
from icon4py.model.common.utils import data_allocation as data_alloc


def apply_hydrostatic_adjustment_ndarray(
    rho: data_alloc.NDArray,
    exner: data_alloc.NDArray,
    theta_v: data_alloc.NDArray,
    exner_ref_mc: data_alloc.NDArray,
    d_exner_dz_ref_ic: data_alloc.NDArray,
    theta_ref_mc: data_alloc.NDArray,
    theta_ref_ic: data_alloc.NDArray,
    wgtfac_c: data_alloc.NDArray,
    ddqz_z_half: data_alloc.NDArray,
    num_levels: int,
    array_ns: ModuleType = np,
) -> None:
    """
    apply hydrostatic adjustment to update rho, exner, and theta_v arrays
    """
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

        exner[:, k] = (
            quadratic_b + array_ns.sqrt(quadratic_b**2 + 4.0 * quadratic_a * quadratic_c)
        ) / (2.0 * quadratic_a)
        theta_v[:, k] = temp_v[:, k] / exner[:, k]
        rho[:, k] = (
            exner[:, k] ** phy_const.CVD_O_RD * phy_const.P0REF / (phy_const.RD * theta_v[:, k])
        )


def hydrostatic_adjustment_constant_thetav_ndarray(
    wgtfac_c: data_alloc.NDArray,
    ddqz_z_half: data_alloc.NDArray,
    exner_ref_mc: data_alloc.NDArray,
    d_exner_dz_ref_ic: data_alloc.NDArray,
    theta_ref_mc: data_alloc.NDArray,
    theta_ref_ic: data_alloc.NDArray,
    rho: data_alloc.NDArray,
    exner: data_alloc.NDArray,
    theta_v: data_alloc.NDArray,
    num_levels: int,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    """
    Computes a hydrostatically balanced profile. In constrast to the above
    hydrostatic_adjustment_ndarray, the virtual temperature is kept (assumed)
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


def zonalwind_2_normalwind_ndarray(
    grid: icon_grid.IconGrid,
    jw_u0: float,
    jw_baroclinic_amplitude: float,
    lat_perturbation_center: float,
    lon_perturbation_center: float,
    edge_lat: data_alloc.NDArray,
    edge_lon: data_alloc.NDArray,
    primal_normal_x: data_alloc.NDArray,
    eta_v_at_edge: data_alloc.NDArray,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    """
    Compute normal wind at edge center from vertical eta coordinate (eta_v_at_edge).

    Args:
        grid: IconGrid
        jw_u0: base zonal wind speed factor, or maximum wind speed
        jw_baroclinic_amplitude: perturbation amplitude
        lat_perturbation_center: perturbation center in latitude
        lon_perturbation_center: perturbation center in longitude
        edge_lat: edge center latitude
        edge_lon: edge center longitude
        primal_normal_x: zonal component of primal normal vector at edge center
        eta_v_at_edge: vertical eta coordinate at edge center
    Returns: normal wind
    """
    # TODO(OngChia): this function needs a test
    ub = grid.end_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    mask = array_ns.ones((grid.num_edges, grid.num_levels), dtype=bool)
    mask[
        0:ub,
        :,
    ] = False
    edge_lat = array_ns.repeat(
        array_ns.expand_dims(edge_lat, axis=-1), eta_v_at_edge.shape[1], axis=1
    )
    edge_lon = array_ns.repeat(
        array_ns.expand_dims(edge_lon, axis=-1), eta_v_at_edge.shape[1], axis=1
    )
    primal_normal_x = array_ns.repeat(
        array_ns.expand_dims(primal_normal_x, axis=-1), eta_v_at_edge.shape[1], axis=1
    )
    u = array_ns.where(
        mask,
        jw_u0 * (array_ns.cos(eta_v_at_edge) ** 1.5) * (array_ns.sin(2.0 * edge_lat) ** 2),
        0.0,
    )
    if jw_baroclinic_amplitude > 1.0e-20:
        u = array_ns.where(
            mask,
            u
            + jw_baroclinic_amplitude
            * array_ns.exp(
                -(
                    (
                        10.0
                        * array_ns.arccos(
                            array_ns.sin(lat_perturbation_center) * array_ns.sin(edge_lat)
                            + array_ns.cos(lat_perturbation_center)
                            * array_ns.cos(edge_lat)
                            * array_ns.cos(edge_lon - lon_perturbation_center)
                        )
                    )
                    ** 2
                )
            ),
            u,
        )
    vn = u * primal_normal_x

    return vn


def init_w(
    grid,
    z_ifc,
    inv_dual_edge_length,
    edge_cell_length,
    primal_edge_length,
    cell_area,
    vn,
    vct_a,
    nlev,
):
    c2e = grid.get_connectivity("C2E")
    e2c = grid.get_connectivity("E2C")
    horizontal_start_e = grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    horizontal_end_e = grid.end_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.INTERIOR))

    horizontal_start_c = grid.start_index(
        h_grid.domain(dims.CellDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    horizontal_end_c = grid.end_index(h_grid.domain(dims.CellDim)(h_grid.Zone.INTERIOR))

    z_slope_e = gtx.as_field((dims.EdgeDim, dims.KDim), np.zeros((horizontal_end_e, nlev + 1)))
    z_wsfc_e = np.zeros((horizontal_end_e, 1))

    nlevp1 = nlev + 1
    helpers.grad_fd_norm(
        z_ifc,
        inv_dual_edge_length,
        out=z_slope_e,
        domain={
            dims.EdgeDim: (horizontal_start_e, horizontal_end_e),
            dims.KDim: (0, nlevp1),
        },
        offset_provider={"E2C": grid.get_connectivity("E2C")},
    )
    for je in range(horizontal_start_e, horizontal_end_e):
        z_wsfc_e[je, 0] = vn[je, nlev - 1] * z_slope_e.asnumpy()[je, nlevp1 - 1]

    e_inn_c = np.zeros((horizontal_end_c, 3))  # or 1
    for jc in range(horizontal_end_c):
        for je in range(3):
            idx_ce = 0 if e2c.asnumpy()[c2e.asnumpy()][jc, je, 0] == jc else 1
            e_inn_c[jc, je] = (
                edge_cell_length.asnumpy()[c2e.asnumpy()[jc, je], idx_ce]
                * primal_edge_length.asnumpy()[c2e.asnumpy()[jc, je]]
                / cell_area.asnumpy()[jc]
            )

    z_wsfc_c = np.sum(z_wsfc_e[c2e.asnumpy()] * e_inn_c[:, :, np.newaxis], axis=1)

    w = np.zeros((horizontal_end_c, nlevp1))
    for jc in range(horizontal_start_c, horizontal_end_c):
        w[jc, nlevp1 - 1] = z_wsfc_c[jc]

    for jk in reversed(range(1, nlev)):
        for jc in range(horizontal_start_c, horizontal_end_c):
            w[jc, jk] = z_wsfc_c[jc, 0] * np.exp(-vct_a.asnumpy()[jk] / 5000.0)

    return w
