# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import math
from types import ModuleType

import numpy as np

from icon4py.model.common import constants as phy_const, dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid, icon as icon_grid
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
    grid: icon_grid.IconGrid,
    c2e: data_alloc.NDArray,
    e2c: data_alloc.NDArray,
    z_ifc: data_alloc.NDArray,
    inv_dual_edge_length: data_alloc.NDArray,
    edge_cell_length: data_alloc.NDArray,
    primal_edge_length: data_alloc.NDArray,
    cell_area: data_alloc.NDArray,
    vn: data_alloc.NDArray,
    vct_b: data_alloc.NDArray,
    nlev: int,
    array_ns: ModuleType,
) -> data_alloc.NDArray:
    lb_e = grid.start_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    ub_e = grid.end_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.INTERIOR))

    lb_c = grid.start_index(h_grid.domain(dims.CellDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    ub_c = grid.end_index(h_grid.domain(dims.CellDim)(h_grid.Zone.INTERIOR))

    z_wsfc_e = array_ns.zeros((ub_e,))
    for je in range(lb_e, ub_e):
        z_wsfc_e[je] = (
            vn[je, nlev - 1]
            * ((z_ifc[e2c[:, 1]] - z_ifc[e2c[:, 0]])[je, :] * inv_dual_edge_length[je])[nlev]
        )

    e_inn_c = array_ns.zeros((ub_c, 3))  # or 1
    for jc in range(ub_c):
        for je in range(3):
            idx_ce = 0 if e2c[c2e][jc, je, 0] == jc else 1
            e_inn_c[jc, je] = (
                edge_cell_length[c2e[jc, je], idx_ce]
                * primal_edge_length[c2e[jc, je]]
                / cell_area[jc]
            )
    z_wsfc_c = array_ns.sum(z_wsfc_e[c2e] * e_inn_c, axis=1)

    w = array_ns.zeros((ub_c, nlev + 1))
    w[lb_c:, nlev] = z_wsfc_c[lb_c:ub_c]
    w[lb_c:, 1:] = z_wsfc_c[lb_c:ub_c, array_ns.newaxis] * vct_b[array_ns.newaxis, 1:]

    return w


def init_bubble(
    theta_v_ndarray: data_alloc.NDArray,
    rho_ndarray: data_alloc.NDArray,
    qv_ndarray: data_alloc.NDArray,
    exner_ndarray: data_alloc.NDArray,
    cell_cartesian_x: data_alloc.NDArray,
    cell_cartesian_y: data_alloc.NDArray,
    z_mc: data_alloc.NDArray,
    bubble_center_x: ta.wpfloat,
    bubble_center_y: ta.wpfloat,
    bubble_center_z: ta.wpfloat,
    bubble_width: ta.wpfloat,
    bubble_height: ta.wpfloat,
    bubble_amplitude: ta.wpfloat,
    geometry_type: base.GeometryType,
    domain_length: ta.wpfloat,
    domain_height: ta.wpfloat,
    array_ns: ModuleType = np,
) -> None:
    match geometry_type:
        case base.GeometryType.ICOSAHEDRON:
            raise NotImplementedError(
                "Bubble initialization not yet implemented on icosahedral grid."
            )
        case base.GeometryType.TORUS:
            norm_bubble_x = bubble_center_x / bubble_width
            norm_bubble_y = bubble_center_y / bubble_width
            norm_bubble_z = bubble_center_z / bubble_height
            norm_cell_cartesian_x = cell_cartesian_x / bubble_width
            norm_cell_cartesian_y = cell_cartesian_y / bubble_width
            norm_z_mc = z_mc / bubble_height
            bubble_distance = functools.partial(
                calculate_distance_to_a_point_on_cartesian_plane, array_ns=array_ns
            )(
                cartesian_x=norm_cell_cartesian_x,
                cartesian_y=norm_cell_cartesian_y,
                cartesian_z=norm_z_mc,
                point_x=norm_bubble_x,
                point_y=norm_bubble_y,
                point_z=norm_bubble_z,
                domain_length=domain_length,
                domain_height=domain_height,
            )
            mask = bubble_distance < ta.wpfloat(1.0)
            theta_v_ndarray[:, :] = array_ns.where(
                mask,
                theta_v_ndarray[:, :]
                + bubble_amplitude
                * array_ns.cos(bubble_distance * math.pi / 2.0) ** 2
                * (1.0 + phy_const.RV_O_RD_MINUS_1 * qv_ndarray[:, :]),
                theta_v_ndarray[:, :],
            )
            rho_ndarray[:, :] = array_ns.where(
                mask,
                exner_ndarray[:, :] ** phy_const.CVD_O_RD
                * phy_const.P0REF
                / (phy_const.RD * theta_v_ndarray[:, :]),
                rho_ndarray[:, :],
            )
        case _:
            raise AssertionError(f"Invalid geometry_type in bubble initialization: {geometry_type}")


def calculate_distance_to_a_point_on_cartesian_plane(
    cartesian_x: data_alloc.NDArray,
    cartesian_y: data_alloc.NDArray,
    cartesian_z: data_alloc.NDArray,
    point_x: ta.wpfloat,
    point_y: ta.wpfloat,
    point_z: ta.wpfloat,
    domain_length: ta.wpfloat,
    domain_height: ta.wpfloat,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    dx = array_ns.abs(cartesian_x - point_x)
    dy = array_ns.abs(cartesian_y - point_y)
    dx = array_ns.where(dx <= 0.5 * domain_length, dx, domain_length - dx)
    dy = array_ns.where(dy <= 0.5 * domain_length, dy, domain_height - dy)
    dz = array_ns.abs(cartesian_z - point_z)
    return array_ns.sqrt(dx**2 + dy**2 + dz**2)
