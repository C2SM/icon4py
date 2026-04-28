# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from types import ModuleType

import numpy as np

from icon4py.model.common import constants as phy_const, dimension as dims
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.math.stencils import generic_math_operations_array_ns
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
    array_ns = data_alloc.array_namespace(rho)
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
    array_ns = data_alloc.array_namespace(edge_lat)
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
    grid: icon_grid.IconGrid,
    z_ifc: data_alloc.NDArray,
    inv_dual_edge_length: data_alloc.NDArray,
    edge_cell_distance: data_alloc.NDArray,
    primal_edge_length: data_alloc.NDArray,
    cell_area: data_alloc.NDArray,
    vn: data_alloc.NDArray,
    vct_b: data_alloc.NDArray,
    nlev: int,
    array_ns: ModuleType,
) -> data_alloc.NDArray:
    # The bounds need to include the first halo line because of the e2c -> c2e connectivity
    array_ns = data_alloc.array_namespace(c2e)
    lb_e = grid.start_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    ub_e = grid.end_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.END))
    lb_c = grid.start_index(h_grid.domain(dims.CellDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    ub_c = grid.end_index(h_grid.domain(dims.CellDim)(h_grid.Zone.END))

    c2e = grid.get_connectivity(dims.C2E).ndarray
    e2c = grid.get_connectivity(dims.E2C).ndarray

    z_grad_e = generic_math_operations_array_ns.compute_directional_derivative_on_edges(
        z_ifc[:, nlev], e2c, inv_dual_edge_length, lb_e, ub_e, grid.num_edges, array_ns
    )
    z_wsfc_e = vn[:, nlev - 1] * z_grad_e

    z_wsfc_c = generic_math_operations_array_ns.interpolate_edges_to_cell(
        z_wsfc_e,
        c2e,
        e2c,
        edge_cell_distance,
        primal_edge_length,
        cell_area,
        ub_c,
        grid.num_cells,
        array_ns,
    )

    w = array_ns.zeros((grid.num_cells, nlev + 1))
    w[lb_c:ub_c, nlev] = z_wsfc_c[lb_c:ub_c]
    w[lb_c:ub_c, 1:] = z_wsfc_c[lb_c:ub_c, array_ns.newaxis] * vct_b[array_ns.newaxis, 1:]

    return w
