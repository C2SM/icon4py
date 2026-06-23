# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.advection import advection_states
from icon4py.model.atmosphere.diffusion import diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.common import constants as phy_const, dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import (
    geometry as grid_geometry,
    geometry_attributes as geometry_meta,
    horizontal as h_grid,
    icon as icon_grid,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.interpolation.stencils import edge_2_cell_vector_rbf_interpolation
from icon4py.model.common.math.stencils import (
    generic_math_operations as gt4py_math_op,
    generic_math_operations_array_ns,
)
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import driver_states


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.standalone_driver import config as driver_config


def apply_hydrostatic_adjustment_ndarray(
    *,
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
    *,
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
    *,
    grid: icon_grid.IconGrid,
    u0: float,
    baroclinic_amplitude: float,
    lat_perturbation_center: float,
    lon_perturbation_center: float,
    edge_lat: data_alloc.NDArray,
    edge_lon: data_alloc.NDArray,
    primal_normal_x: data_alloc.NDArray,
    eta_v_at_edge: data_alloc.NDArray,
) -> data_alloc.NDArray:
    """
    Compute normal wind at edge center from vertical eta coordinate (eta_v_at_edge).

    Args:
        grid: IconGrid
        u0: base zonal wind speed factor, or maximum wind speed
        baroclinic_amplitude: perturbation amplitude
        lat_perturbation_center: perturbation center in latitude
        lon_perturbation_center: perturbation center in longitude
        edge_lat: edge center latitude
        edge_lon: edge center longitude
        primal_normal_x: zonal component of primal normal vector at edge center
        eta_v_at_edge: vertical eta coordinate at edge center
    Returns: normal wind
    """
    # TODO(OngChia): this function needs a test
    array_ns = data_alloc.array_namespace(edge_lat)
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
        u0 * (array_ns.cos(eta_v_at_edge) ** 1.5) * (array_ns.sin(2.0 * edge_lat) ** 2),
        0.0,
    )
    if baroclinic_amplitude > 1.0e-20:
        u = array_ns.where(
            mask,
            u
            + baroclinic_amplitude
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
    *,
    grid: icon_grid.IconGrid,
    z_ifc: data_alloc.NDArray,
    inv_dual_edge_length: data_alloc.NDArray,
    edge_cell_distance: data_alloc.NDArray,
    primal_edge_length: data_alloc.NDArray,
    cell_area: data_alloc.NDArray,
    vn: data_alloc.NDArray,
    vct_b: data_alloc.NDArray,
    nlev: int,
) -> data_alloc.NDArray:
    array_ns = data_alloc.array_namespace(z_ifc)
    # The bounds need to include the first halo line because of the e2c -> c2e connectivity
    lb_e = grid.start_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    ub_e = grid.end_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.END))
    lb_c = grid.start_index(h_grid.domain(dims.CellDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    ub_c = grid.end_index(h_grid.domain(dims.CellDim)(h_grid.Zone.END))

    c2e = grid.get_connectivity(dims.C2E).ndarray
    e2c = grid.get_connectivity(dims.E2C).ndarray

    z_grad_e = generic_math_operations_array_ns.compute_directional_derivative_on_edges(
        cell_field=z_ifc[:, nlev],
        e2c=e2c,
        inv_dual_edge_length=inv_dual_edge_length,
        lb_e=lb_e,
        ub_e=ub_e,
        num_edges=grid.num_edges,
    )
    z_wsfc_e = vn[:, nlev - 1] * z_grad_e

    z_wsfc_c = generic_math_operations_array_ns.interpolate_edges_to_cell(
        edge_field=z_wsfc_e,
        c2e=c2e,
        e2c=e2c,
        edge_cell_length=edge_cell_distance,
        primal_edge_length=primal_edge_length,
        cell_area=cell_area,
        ub_c=ub_c,
        num_cells=grid.num_cells,
    )

    w = array_ns.zeros((grid.num_cells, nlev + 1))
    w[lb_c:ub_c, nlev] = z_wsfc_c[lb_c:ub_c]
    w[lb_c:ub_c, 1:] = z_wsfc_c[lb_c:ub_c, array_ns.newaxis] * vct_b[array_ns.newaxis, 1:]

    return w


# ---------------------------------------------------------------------------
# Shared field-extraction helpers (used by individual test-case modules)
# ---------------------------------------------------------------------------


def extract_metrics(
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
) -> dict[str, data_alloc.NDArray]:
    return {
        "wgtfac_c": metrics_field_source.get(metrics_attributes.WGTFAC_C).ndarray,
        "ddqz_z_half": metrics_field_source.get(metrics_attributes.DDQZ_Z_HALF).ndarray,
        "theta_ref_mc": metrics_field_source.get(metrics_attributes.THETA_REF_MC).ndarray,
        "theta_ref_ic": metrics_field_source.get(metrics_attributes.THETA_REF_IC).ndarray,
        "exner_ref_mc": metrics_field_source.get(metrics_attributes.EXNER_REF_MC).ndarray,
        "d_exner_dz_ref_ic": metrics_field_source.get(metrics_attributes.D_EXNER_DZ_REF_IC).ndarray,
        "geopot": phy_const.GRAV * metrics_field_source.get(metrics_attributes.Z_MC).ndarray,
        "z_ifc": metrics_field_source.get(metrics_attributes.CELL_HEIGHT_ON_HALF_LEVEL).ndarray,
    }


def extract_geometry(
    geometry_field_source: grid_geometry.GridGeometry,
) -> dict[str, data_alloc.NDArray]:
    return {
        "cell_lat": geometry_field_source.get(geometry_meta.CELL_LAT).ndarray,
        "edge_lat": geometry_field_source.get(geometry_meta.EDGE_LAT).ndarray,
        "edge_lon": geometry_field_source.get(geometry_meta.EDGE_LON).ndarray,
        "primal_normal_x": geometry_field_source.get(geometry_meta.EDGE_NORMAL_U).ndarray,
        "inv_dual_edge_length": geometry_field_source.get(
            f"inverse_of_{geometry_meta.DUAL_EDGE_LENGTH}"
        ).ndarray,
        "edge_cell_distance": geometry_field_source.get(geometry_meta.EDGE_CELL_DISTANCE).ndarray,
        "primal_edge_length": geometry_field_source.get(geometry_meta.EDGE_LENGTH).ndarray,
        "cell_area": geometry_field_source.get(geometry_meta.CELL_AREA).ndarray,
    }


def extract_interpolation(
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
) -> dict:
    return {
        "c_lin_e": interpolation_field_source.get(interpolation_attributes.C_LIN_E),
        "rbf_vec_coeff_c1": interpolation_field_source.get(
            interpolation_attributes.RBF_VEC_COEFF_C1
        ),
        "rbf_vec_coeff_c2": interpolation_field_source.get(
            interpolation_attributes.RBF_VEC_COEFF_C2
        ),
    }


def zone_indices(grid: icon_grid.IconGrid) -> dict[str, int]:
    edge_domain = h_grid.domain(dims.EdgeDim)
    cell_domain = h_grid.domain(dims.CellDim)
    return {
        "end_edge_lateral_boundary_level_2": grid.end_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        ),
        "end_edge_end": grid.end_index(edge_domain(h_grid.Zone.END)),
        "end_cell_lateral_boundary_level_2": grid.end_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        ),
        "end_cell_end": grid.end_index(cell_domain(h_grid.Zone.END)),
    }


def assemble_driver_states(
    *,
    grid: icon_grid.IconGrid,
    allocator: gtx_typing.Allocator,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
    interpolation: dict,
    zone_indices_map: dict[str, int],
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    prognostic_state_now: prognostics.PrognosticState,
    diagnostic_state: diagnostics.DiagnosticState,
    experiment_config: driver_config.ExperimentConfig,
) -> driver_states.DriverStates:
    prognostic_state_next = prognostics.PrognosticState(
        vn=data_alloc.as_field(prognostic_state_now.vn, allocator=allocator),
        w=data_alloc.as_field(prognostic_state_now.w, allocator=allocator),
        exner=data_alloc.as_field(prognostic_state_now.exner, allocator=allocator),
        rho=data_alloc.as_field(prognostic_state_now.rho, allocator=allocator),
        theta_v=data_alloc.as_field(prognostic_state_now.theta_v, allocator=allocator),
    )
    prognostic_states = common_utils.TimeStepPair(prognostic_state_now, prognostic_state_next)

    edge_2_cell_vector_rbf_interpolation.edge_2_cell_vector_rbf_interpolation.with_backend(backend)(
        p_e_in=prognostic_states.current.vn,
        ptr_coeff_1=interpolation["rbf_vec_coeff_c1"],
        ptr_coeff_2=interpolation["rbf_vec_coeff_c2"],
        p_u_out=diagnostic_state.u,
        p_v_out=diagnostic_state.v,
        horizontal_start=zone_indices_map["end_cell_lateral_boundary_level_2"],
        horizontal_end=zone_indices_map["end_cell_end"],
        vertical_start=0,
        vertical_end=grid.num_levels,
        offset_provider=grid.connectivities,
    )
    exchange.exchange(dims.CellDim, diagnostic_state.u, diagnostic_state.v)

    perturbed_exner = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator)
    gt4py_math_op.compute_difference_on_cell_k.with_backend(backend)(
        field_a=prognostic_states.current.exner,
        field_b=metrics_field_source.get(metrics_attributes.EXNER_REF_MC),
        output_field=perturbed_exner,
        horizontal_start=0,
        horizontal_end=grid.num_cells,
        vertical_start=0,
        vertical_end=grid.num_levels,
        offset_provider={},
    )

    solve_nonhydro_enabled = experiment_config.nonhydrostatic is not None
    diffusion_enabled = experiment_config.diffusion is not None
    tracer_advection_enabled = experiment_config.tracer_advection is not None

    solve_nonhydro_diagnostic_state = (
        dycore_states.initialize_solve_nonhydro_diagnostic_state(
            perturbed_exner_at_cells_on_model_levels=perturbed_exner,
            grid=grid,
            allocator=allocator,
        )
        if solve_nonhydro_enabled
        else None
    )
    prep_adv = (
        dycore_states.initialize_prep_advection(grid=grid, allocator=allocator)
        if solve_nonhydro_enabled
        else None
    )
    diffusion_diagnostic_state = (
        diffusion_states.initialize_diffusion_diagnostic_state(grid=grid, allocator=allocator)
        if diffusion_enabled
        else None
    )
    tracer_advection_diagnostic_state = (
        advection_states.initialize_advection_diagnostic_state(grid=grid, allocator=allocator)
        if tracer_advection_enabled
        else None
    )
    prep_tracer_adv = (
        advection_states.AdvectionPrepAdvState(
            vn_traj=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator),
            mass_flx_me=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator),
            mass_flx_ic=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator),
        )
        if tracer_advection_enabled
        else None
    )

    return driver_states.DriverStates(
        prep_advection_prognostic=prep_adv,
        solve_nonhydro_diagnostic=solve_nonhydro_diagnostic_state,
        prep_tracer_advection_prognostic=prep_tracer_adv,
        tracer_advection_diagnostic=tracer_advection_diagnostic_state,
        diffusion_diagnostic=diffusion_diagnostic_state,
        prognostics=prognostic_states,
        diagnostic=diagnostic_state,
    )
