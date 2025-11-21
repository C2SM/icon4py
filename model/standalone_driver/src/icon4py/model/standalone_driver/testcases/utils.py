# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from types import ModuleType

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np

import icon4py.model.common.utils as common_utils
from icon4py.model.common import constants as phy_const, dimension as dims
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
)
from icon4py.model.common.utils import data_allocation as data_alloc


def hydrostatic_adjustment_ndarray(
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
    array_ns: ModuleType = np,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray, data_alloc.NDArray]:
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

    return rho, exner, theta_v


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
    jw_up: float,
    lat_perturbation_center: float,
    lon_perturbation_center: float,
    edge_lat: data_alloc.NDArray,
    edge_lon: data_alloc.NDArray,
    primal_normal_x: data_alloc.NDArray,
    eta_v_e: data_alloc.NDArray,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
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
    # TODO(OngChia): this function needs a test
    ub = grid.end_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    mask = array_ns.ones((grid.num_edges, grid.num_levels), dtype=bool)
    mask[
        0:ub,
        :,
    ] = False
    edge_lat = array_ns.repeat(array_ns.expand_dims(edge_lat, axis=-1), eta_v_e.shape[1], axis=1)
    edge_lon = array_ns.repeat(array_ns.expand_dims(edge_lon, axis=-1), eta_v_e.shape[1], axis=1)
    primal_normal_x = array_ns.repeat(
        array_ns.expand_dims(primal_normal_x, axis=-1), eta_v_e.shape[1], axis=1
    )
    u = array_ns.where(
        mask, jw_u0 * (array_ns.cos(eta_v_e) ** 1.5) * (array_ns.sin(2.0 * edge_lat) ** 2), 0.0
    )
    if jw_up > 1.0e-20:
        u = array_ns.where(
            mask,
            u
            + jw_up
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


def create_gt4py_field_for_prognostic_and_diagnostic_variables(
    vn_ndarray: data_alloc.NDArray,
    w_ndarray: data_alloc.NDArray,
    exner_ndarray: data_alloc.NDArray,
    rho_ndarray: data_alloc.NDArray,
    theta_v_ndarray: data_alloc.NDArray,
    temperature_ndarray: data_alloc.NDArray,
    pressure_ndarray: data_alloc.NDArray,
    pressure_ifc_ndarray: data_alloc.NDArray,
    grid: icon_grid.IconGrid,
    backend: gtx_typing.Backend | None,
) -> tuple[
    common_utils.TimeStepPair[prognostics.PrognosticState],
    diagnostics.DiagnosticState,
]:
    vn = gtx.as_field((dims.EdgeDim, dims.KDim), vn_ndarray, allocator=backend)
    w = gtx.as_field((dims.CellDim, dims.KDim), w_ndarray, allocator=backend)
    exner = gtx.as_field((dims.CellDim, dims.KDim), exner_ndarray, allocator=backend)
    rho = gtx.as_field((dims.CellDim, dims.KDim), rho_ndarray, allocator=backend)
    temperature = gtx.as_field((dims.CellDim, dims.KDim), temperature_ndarray, allocator=backend)
    virtual_temperature = gtx.as_field(
        (dims.CellDim, dims.KDim), temperature_ndarray, allocator=backend
    )
    pressure = gtx.as_field((dims.CellDim, dims.KDim), pressure_ndarray, allocator=backend)
    theta_v = gtx.as_field((dims.CellDim, dims.KDim), theta_v_ndarray, allocator=backend)
    pressure_ifc = gtx.as_field((dims.CellDim, dims.KDim), pressure_ifc_ndarray, allocator=backend)

    vn_next = gtx.as_field((dims.EdgeDim, dims.KDim), vn_ndarray, allocator=backend)
    w_next = gtx.as_field((dims.CellDim, dims.KDim), w_ndarray, allocator=backend)
    exner_next = gtx.as_field((dims.CellDim, dims.KDim), exner_ndarray, allocator=backend)
    rho_next = gtx.as_field((dims.CellDim, dims.KDim), rho_ndarray, allocator=backend)
    theta_v_next = gtx.as_field((dims.CellDim, dims.KDim), theta_v_ndarray, allocator=backend)

    u = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
    v = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)

    prognostic_state_now = prognostics.PrognosticState(
        w=w,
        vn=vn,
        theta_v=theta_v,
        rho=rho,
        exner=exner,
    )
    prognostic_state_next = prognostics.PrognosticState(
        w=w_next,
        vn=vn_next,
        theta_v=theta_v_next,
        rho=rho_next,
        exner=exner_next,
    )

    prognostics_states = common_utils.TimeStepPair(prognostic_state_now, prognostic_state_next)

    diagnostic_state = diagnostics.DiagnosticState(
        pressure=pressure,
        pressure_ifc=pressure_ifc,
        temperature=temperature,
        virtual_temperature=virtual_temperature,
        u=u,
        v=v,
    )

    return prognostics_states, diagnostic_state
