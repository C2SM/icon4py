# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import logging
import math

from gt4py import next as gtx

import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.advection import advection_states
from icon4py.model.atmosphere.diffusion import diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.common import (
    constants as phy_const,
    dimension as dims,
    model_backends,
    type_alias as ta,
)
from icon4py.model.common.diagnostic_calculations.stencils import thermodynamic_functions
from icon4py.model.common.grid import (
    geometry as grid_geometry,
    geometry_attributes as geometry_meta,
    horizontal as h_grid,
    icon as icon_grid,
    vertical as v_grid,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.interpolation.stencils import (
    cell_2_edge_interpolation,
    edge_2_cell_vector_rbf_interpolation,
)
from icon4py.model.common.math.stencils import generic_math_operations as gt4py_math_op
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
    tracer_state as tracers,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import driver_states
from icon4py.model.standalone_driver.testcases import utils as testcases_utils


log = logging.getLogger(__name__)


def jablonowski_williamson(  # noqa: PLR0915 [too-many-statements]
    grid: icon_grid.IconGrid,
    geometry_field_source: grid_geometry.GridGeometry,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend: gtx.typing.Backend | None,
    lowest_layer_thickness: float,
    model_top_height: float,
    stretch_factor: float,
    damping_height: float,
) -> driver_states.DriverStates:
    """
    Initial condition of Jablonowski-Williamson test. Set jw_baroclinic_amplitude to values larger than 0.01 if
    you want to run baroclinic case.

    Args:
        grid: IconGrid
        geometry_field_source: geometric field factory
        interpolation_field_source: interpolation field factory
        metrics_field_source: metric field factory
        backend: GT4Py backend
    Returns: driver state

    The reference experiment config for this is in icon-exclaim/run/exp.exclaim_nh35_tri_jws_sb.
    """

    allocator = model_backends.get_allocator(backend)
    xp = data_alloc.import_array_ns(allocator)

    wgtfac_c = metrics_field_source.get(metrics_attributes.WGTFAC_C).ndarray
    ddqz_z_half = metrics_field_source.get(metrics_attributes.DDQZ_Z_HALF).ndarray
    theta_ref_mc = metrics_field_source.get(metrics_attributes.THETA_REF_MC).ndarray
    theta_ref_ic = metrics_field_source.get(metrics_attributes.THETA_REF_IC).ndarray
    exner_ref_mc = metrics_field_source.get(metrics_attributes.EXNER_REF_MC).ndarray
    d_exner_dz_ref_ic = metrics_field_source.get(metrics_attributes.D_EXNER_DZ_REF_IC).ndarray
    geopot = phy_const.GRAV * metrics_field_source.get(metrics_attributes.Z_MC).ndarray

    cell_lat = geometry_field_source.get(geometry_meta.CELL_LAT).ndarray
    edge_lat = geometry_field_source.get(geometry_meta.EDGE_LAT).ndarray
    edge_lon = geometry_field_source.get(geometry_meta.EDGE_LON).ndarray
    primal_normal_x = geometry_field_source.get(geometry_meta.EDGE_NORMAL_U).ndarray

    cell_2_edge_coeff = interpolation_field_source.get(interpolation_attributes.C_LIN_E)
    rbf_vec_coeff_c1 = interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_C1)
    rbf_vec_coeff_c2 = interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_C2)

    num_cells = grid.num_cells
    num_levels = grid.num_levels

    edge_domain = h_grid.domain(dims.EdgeDim)
    cell_domain = h_grid.domain(dims.CellDim)
    end_edge_lateral_boundary_level_2 = grid.end_index(
        edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    end_edge_end = grid.end_index(edge_domain(h_grid.Zone.END))
    end_cell_lateral_boundary_level_2 = grid.end_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    end_cell_end = grid.end_index(cell_domain(h_grid.Zone.END))

    # predefined constants used for Jablonowski-Williamson initial condition
    p_sfc = ta.wpfloat("100000.0")  # surface pressure (Pa)
    jw_baroclinic_amplitude = ta.wpfloat(
        "0.0"
    )  # if doing baroclinic wave test, please set it to a nonzero value
    jw_u0 = ta.wpfloat("35.0")  # maximum zonal wind speed (m/s)
    jw_temp0 = ta.wpfloat("288.0")
    eta_0 = ta.wpfloat("0.252")
    eta_t = ta.wpfloat("0.2")  # tropopause
    gamma = ta.wpfloat("0.005")  # temperature elapse rate (K/m)
    dtemp = ta.wpfloat("4.8e5")  # empirical temperature difference (K)
    lon_perturbation_center = math.pi / ta.wpfloat(
        "9.0"
    )  # longitude of the perturb centre in baroclinic wave test (jw_baroclinic_amplitude !=0)
    lat_perturbation_center = (
        ta.wpfloat("2.0") * lon_perturbation_center
    )  # latitude of the perturb centre in baroclinic wave test (jw_baroclinic_amplitude !=0)

    # Initialize prognostic state, diagnostic state and other local fields
    prognostic_state_now = prognostics.initialize_prognostic_state(
        grid=grid,
        allocator=allocator,
    )
    diagnostic_state = diagnostics.initialize_diagnostic_state(grid=grid, allocator=allocator)
    tracer_state_now = tracers.TracerState.zero_field(grid, allocator)
    eta_v = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        allocator=allocator,
        dtype=ta.wpfloat,
    )
    eta_v_at_edge = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator)

    exner_ndarray = prognostic_state_now.exner.ndarray
    rho_ndarray = prognostic_state_now.rho.ndarray
    theta_v_ndarray = prognostic_state_now.theta_v.ndarray
    temperature_ndarray = diagnostic_state.temperature.ndarray
    pressure_ndarray = diagnostic_state.pressure.ndarray
    eta_v_ndarray = eta_v.ndarray

    # set surface pressure
    diagnostic_state.pressure_ifc.ndarray[:, -1] = p_sfc

    sin_lat = xp.sin(cell_lat)
    cos_lat = xp.cos(cell_lat)
    fac1 = ta.wpfloat("1.0") / ta.wpfloat("6.3") - ta.wpfloat("2.0") * (sin_lat**6) * (
        cos_lat**2 + ta.wpfloat("1.0") / ta.wpfloat("3.0")
    )
    fac2 = (
        (
            ta.wpfloat("8.0")
            / ta.wpfloat("5.0")
            * (cos_lat**3)
            * (sin_lat**2 + ta.wpfloat("2.0") / ta.wpfloat("3.0"))
            - ta.wpfloat("0.25") * math.pi
        )
        * phy_const.EARTH_RADIUS
        * phy_const.EARTH_ANGULAR_VELOCITY
    )
    lapse_rate = phy_const.RD * gamma / phy_const.GRAV
    for k_index in range(num_levels - 1, -1, -1):
        eta_old = xp.full(num_cells, fill_value=ta.wpfloat("1.0e-7"), dtype=ta.wpfloat)
        log.info(f"In Newton iteration, k = {k_index}")
        # Newton iteration to determine zeta
        for _ in range(100):
            eta_v_ndarray[:, k_index] = (eta_old - eta_0) * math.pi * 0.5
            cos_etav = xp.cos(eta_v_ndarray[:, k_index])
            sin_etav = xp.sin(eta_v_ndarray[:, k_index])

            temperature_avg = jw_temp0 * (eta_old**lapse_rate)
            geopot_avg = (
                jw_temp0 * phy_const.GRAV / gamma * (ta.wpfloat("1.0") - eta_old**lapse_rate)
            )
            temperature_avg = xp.where(
                eta_old < eta_t, temperature_avg + dtemp * ((eta_t - eta_old) ** 5), temperature_avg
            )
            geopot_avg = xp.where(
                eta_old < eta_t,
                geopot_avg
                - phy_const.RD
                * dtemp
                * (
                    (xp.log(eta_old / eta_t) + ta.wpfloat("137.0") / ta.wpfloat("60.0"))
                    * (eta_t**5)
                    - ta.wpfloat("5.0") * (eta_t**4) * eta_old
                    + ta.wpfloat("5.0") * (eta_t**3) * (eta_old**2)
                    - ta.wpfloat("10.0") / ta.wpfloat("3.0") * (eta_t**2) * (eta_old**3)
                    + ta.wpfloat("1.25") * eta_t * (eta_old**4)
                    - ta.wpfloat("0.2") * (eta_old**5)
                ),
                geopot_avg,
            )

            geopot_jw = geopot_avg + jw_u0 * (cos_etav**1.5) * (
                fac1 * jw_u0 * (cos_etav**1.5) + fac2
            )
            temperature_jw = temperature_avg + ta.wpfloat(
                "0.75"
            ) * eta_old * math.pi * jw_u0 / phy_const.RD * sin_etav * xp.sqrt(cos_etav) * (
                ta.wpfloat("2.0") * jw_u0 * fac1 * (cos_etav**1.5) + fac2
            )
            newton_function = geopot_jw - geopot[:, k_index]
            newton_function_prime = -phy_const.RD / eta_old * temperature_jw
            eta_old = eta_old - newton_function / newton_function_prime

        # Final update for zeta_v
        eta_v_ndarray[:, k_index] = (eta_old - eta_0) * math.pi * 0.5
        # Use analytic expressions at all model level
        exner_ndarray[:, k_index] = (eta_old * p_sfc / phy_const.P0REF) ** phy_const.RD_O_CPD
        theta_v_ndarray[:, k_index] = temperature_jw / exner_ndarray[:, k_index]
        rho_ndarray[:, k_index] = (
            exner_ndarray[:, k_index] ** phy_const.CVD_O_RD
            * phy_const.P0REF
            / phy_const.RD
            / theta_v_ndarray[:, k_index]
        )
        # initialize diagnose pressure and temperature variables
        pressure_ndarray[:, k_index] = (
            phy_const.P0REF * exner_ndarray[:, k_index] ** phy_const.CPD_O_RD
        )
        temperature_ndarray[:, k_index] = temperature_jw
    log.info("Newton iteration completed.")

    cell_2_edge_interpolation.cell_2_edge_interpolation.with_backend(backend)(
        in_field=eta_v,
        coeff=cell_2_edge_coeff,
        out_field=eta_v_at_edge,
        horizontal_start=end_edge_lateral_boundary_level_2,
        horizontal_end=end_edge_end,
        vertical_start=0,
        vertical_end=num_levels,
        offset_provider=grid.connectivities,
    )
    log.info("Cell-to-edge eta_v computation completed.")

    prognostic_state_now.vn.ndarray[:, :] = functools.partial(
        testcases_utils.zonalwind_2_normalwind_ndarray, array_ns=xp
    )(
        grid=grid,
        jw_u0=jw_u0,
        jw_baroclinic_amplitude=jw_baroclinic_amplitude,
        lat_perturbation_center=lat_perturbation_center,
        lon_perturbation_center=lon_perturbation_center,
        edge_lat=edge_lat,
        edge_lon=edge_lon,
        primal_normal_x=primal_normal_x,
        eta_v_at_edge=eta_v_at_edge.ndarray,
    )
    vertical_config = v_grid.VerticalGridConfig(
        grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )

    _, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, model_backends.get_allocator(backend))

    prognostic_state_now.w.ndarray[:, :] = testcases_utils.init_w(
        grid,
        c2e=grid.get_connectivity(dims.C2E).ndarray,
        e2c=grid.get_connectivity(dims.E2C).ndarray,
        z_ifc=metrics_field_source.get(metrics_attributes.CELL_HEIGHT_ON_HALF_LEVEL).ndarray,
        inv_dual_edge_length=geometry_field_source.get(
            f"inverse_of_{geometry_meta.DUAL_EDGE_LENGTH}"
        ).ndarray,
        edge_cell_length=geometry_field_source.get(geometry_meta.EDGE_CELL_DISTANCE).ndarray,
        primal_edge_length=geometry_field_source.get(geometry_meta.EDGE_LENGTH).ndarray,
        cell_area=geometry_field_source.get(geometry_meta.CELL_AREA).ndarray,
        vn=prognostic_state_now.vn.ndarray,
        vct_b=vct_b.ndarray,
        nlev=num_levels,
        array_ns=xp,
    )
    log.info("U2vn computation completed.")

    functools.partial(testcases_utils.apply_hydrostatic_adjustment_ndarray, array_ns=xp)(
        rho=rho_ndarray,
        exner=exner_ndarray,
        theta_v=theta_v_ndarray,
        exner_ref_mc=exner_ref_mc,
        d_exner_dz_ref_ic=d_exner_dz_ref_ic,
        theta_ref_mc=theta_ref_mc,
        theta_ref_ic=theta_ref_ic,
        wgtfac_c=wgtfac_c,
        ddqz_z_half=ddqz_z_half,
        num_levels=num_levels,
    )
    log.info("Hydrostatic adjustment computation completed.")
    prognostic_state_next = prognostics.PrognosticState(
        vn=data_alloc.as_field(prognostic_state_now.vn, allocator=allocator),
        w=data_alloc.as_field(prognostic_state_now.w, allocator=allocator),
        exner=data_alloc.as_field(prognostic_state_now.exner, allocator=allocator),
        rho=data_alloc.as_field(prognostic_state_now.rho, allocator=allocator),
        theta_v=data_alloc.as_field(prognostic_state_now.theta_v, allocator=allocator),
    )
    prognostic_states = common_utils.TimeStepPair(prognostic_state_now, prognostic_state_next)

    tracer_state_next = tracers.TracerState.from_tracer_state(tracer_state_now, allocator)
    tracer_states = common_utils.TimeStepPair(tracer_state_now, tracer_state_next)

    edge_2_cell_vector_rbf_interpolation.edge_2_cell_vector_rbf_interpolation.with_backend(backend)(
        p_e_in=prognostic_states.current.vn,
        ptr_coeff_1=rbf_vec_coeff_c1,
        ptr_coeff_2=rbf_vec_coeff_c2,
        p_u_out=diagnostic_state.u,
        p_v_out=diagnostic_state.v,
        horizontal_start=end_cell_lateral_boundary_level_2,
        horizontal_end=end_cell_end,
        vertical_start=0,
        vertical_end=num_levels,
        offset_provider=grid.connectivities,
    )

    log.info("U, V computation completed.")

    perturbed_exner = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator)
    gt4py_math_op.compute_difference_on_cell_k.with_backend(backend)(
        field_a=prognostic_states.current.exner,
        field_b=metrics_field_source.get(metrics_attributes.EXNER_REF_MC),
        output_field=perturbed_exner,
        horizontal_start=0,
        horizontal_end=num_cells,
        vertical_start=0,
        vertical_end=num_levels,
        offset_provider={},
    )
    log.info("perturbed_exner initialization completed.")

    diffusion_diagnostic_state = diffusion_states.initialize_diffusion_diagnostic_state(
        grid=grid, allocator=allocator
    )
    solve_nonhydro_diagnostic_state = dycore_states.initialize_solve_nonhydro_diagnostic_state(
        perturbed_exner_at_cells_on_model_levels=perturbed_exner,
        grid=grid,
        allocator=allocator,
    )
    prep_adv = dycore_states.initialize_prep_advection(grid=grid, allocator=allocator)
    tracer_advection_diagnostic_state = advection_states.initialize_advection_diagnostic_state(
        grid=grid, allocator=allocator
    )
    prep_tracer_adv = advection_states.AdvectionPrepAdvState(
        vn_traj=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator),
        mass_flx_me=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator),
        mass_flx_ic=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator),
    )
    log.info("Initialization completed.")

    ds = driver_states.DriverStates(
        prep_advection_prognostic=prep_adv,
        solve_nonhydro_diagnostic=solve_nonhydro_diagnostic_state,
        prep_tracer_advection_prognostic=prep_tracer_adv,
        tracer_advection_diagnostic=tracer_advection_diagnostic_state,
        diffusion_diagnostic=diffusion_diagnostic_state,
        prognostics=prognostic_states,
        diagnostic=diagnostic_state,
        tracers=tracer_states,
    )

    return ds


def weisman_klemp(  # noqa: PLR0915 [too-many-statements]
    grid: icon_grid.IconGrid,
    vertical_grid: v_grid.VerticalGrid,
    geometry_field_source: grid_geometry.GridGeometry,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend: gtx.typing.Backend | None,
) -> driver_states.DriverStates:
    """
    Initial condition of Weisman-Klemp warm bubble test.

    Args:
        grid: IconGrid
        vertical_grid: VerticalGrid
        geometry_field_source: geometric field factory
        metrics_field_source: metric field factory
        backend: GT4Py backend
    Returns: driver state

    The reference experiment config for this is in icon-exclaim/run/exp.exclaim_nh_weisman_klemp_sb.
    """

    allocator = model_backends.get_allocator(backend)
    xp = data_alloc.import_array_ns(allocator)

    wgtfac_c = metrics_field_source.get(metrics_attributes.WGTFAC_C).ndarray
    ddqz_z_half = metrics_field_source.get(metrics_attributes.DDQZ_Z_HALF).ndarray
    theta_ref_mc = metrics_field_source.get(metrics_attributes.THETA_REF_MC).ndarray
    theta_ref_ic = metrics_field_source.get(metrics_attributes.THETA_REF_IC).ndarray
    exner_ref_mc = metrics_field_source.get(metrics_attributes.EXNER_REF_MC).ndarray
    d_exner_dz_ref_ic = metrics_field_source.get(metrics_attributes.D_EXNER_DZ_REF_IC).ndarray
    primal_normal_x = geometry_field_source.get(geometry_meta.EDGE_NORMAL_U).ndarray

    num_cells = grid.num_cells
    num_levels = grid.num_levels

    # predefined constants used for Weisman-Klemp initial condition
    HMIN = ta.wpfloat("0.0")  # base height of the profile
    H_TROPOPAUSE = ta.wpfloat("12000.0")  # tropopause height [m]
    THETA_SURFACE = ta.wpfloat("300.0")  # potential temperature at z=0 [K]
    THETA_TROPOPAUSE = ta.wpfloat("343.0")  # potential temperature at tropopause [K]
    EXP_FACTOR_THETA = ta.wpfloat("1.25")
    EXP_FACTOR_RH = ta.wpfloat("1.25")
    T_TROPOPAUSE = ta.wpfloat("213.0")
    RH_MIN = ta.wpfloat("0.1")
    RH_MAX = ta.wpfloat("0.95")
    HREF = ta.wpfloat("3000.0")
    U_INFTY = ta.wpfloat("15.0")  # maximum horizontal wind speed [m s-1]
    QV_MAX = ta.wpfloat("0.6")  # maximum moisture mixing ratio [kg kg-1]
    bubble_center_x = ta.wpfloat("0.0")
    bubble_center_y = ta.wpfloat("0.0")
    bubble_center_z = ta.wpfloat("1400.0")
    bubble_width = ta.wpfloat("5000.0")
    bubble_height = ta.wpfloat("1400.0")
    bubble_amplitude = ta.wpfloat("2.0")

    model_level_height = xp.repeat(
        xp.expand_dims(
            0.5
            * (
                vertical_grid.interface_physical_height.ndarray[1:]
                + vertical_grid.interface_physical_height.ndarray[:-1]
            ),
            axis=0,
        ),
        repeats=num_cells,
        axis=0,
    )
    # TODO (Yilu):
    above_tropopause_levels = xp.flatnonzero(model_level_height[0,:] > H_TROPOPAUSE)

    assert (
        above_tropopause_levels.size > 0
    ), f"model top height ({model_level_height[0]}) must be higher than the tropopaus height ({H_TROPOPAUSE}) in weisman klemp experiment."
    k_tropopause = above_tropopause_levels[-1]

    # Initialize prognostic state, diagnostic state, and tracer state
    prognostic_state_now = prognostics.initialize_prognostic_state(
        grid=grid,
        allocator=allocator,
    )
    diagnostic_state = diagnostics.initialize_diagnostic_state(grid=grid, allocator=allocator)
    tracer_state_now = tracers.TracerState.zero_field(grid=grid, allocator=allocator)

    exner_ndarray = prognostic_state_now.exner.ndarray
    rho_ndarray = prognostic_state_now.rho.ndarray
    theta_v_ndarray = prognostic_state_now.theta_v.ndarray
    temperature_ndarray = diagnostic_state.temperature.ndarray
    pressure_ndarray = diagnostic_state.pressure.ndarray
    qv_ndarray = tracer_state_now.qv.ndarray
    vn_ndarray = prognostic_state_now.vn
    theta_ndarray = xp.zeros_like(theta_v_ndarray)
    rh_ndarray = xp.zeros_like(qv_ndarray)

    # tropopause parameters
    exner_tropopause = T_TROPOPAUSE / THETA_TROPOPAUSE
    vapor_pres_tropopause = RH_MIN * thermodynamic_functions.calculate_saturation_presssure_water(
        T_TROPOPAUSE
    )
    pres_tropopause = phy_const.P0REF * exner_tropopause**phy_const.CPD_O_RD
    qv_tropopause = thermodynamic_functions.calculate_specific_humidity(
        pres_tropopause, vapor_pres_tropopause
    )
    theta_v_tropopause = THETA_TROPOPAUSE * (1.0 + phy_const.RV_O_RD_MINUS_1 * qv_tropopause)

    # above tropopause
    theta_ndarray[:, : k_tropopause + 1] = THETA_TROPOPAUSE * xp.exp(
        phy_const.GRAV_O_CPD
        / T_TROPOPAUSE
        * (model_level_height[:, : k_tropopause + 1] - H_TROPOPAUSE)
    )
    exner_ndarray[:, : k_tropopause + 1] = exner_tropopause * xp.exp(
        -phy_const.GRAV_O_CPD
        / T_TROPOPAUSE
        * (model_level_height[:, : k_tropopause + 1] - H_TROPOPAUSE)
    )
    pressure_ndarray[:, : k_tropopause + 1] = (
        phy_const.P0REF * exner_ndarray[:, : k_tropopause + 1] ** phy_const.CPD_O_RD
    )
    qv_ndarray[:, : k_tropopause + 1] = thermodynamic_functions.calculate_specific_humidity(
        pressure=pressure_ndarray[:, : k_tropopause + 1], vapor_pressure=vapor_pres_tropopause
    )
    theta_v_ndarray[:, : k_tropopause + 1] = theta_ndarray[:, : k_tropopause + 1] * (
        1.0 + phy_const.RV_O_RD_MINUS_1 * qv_ndarray[:, : k_tropopause + 1]
    )
    temperature_ndarray[:, : k_tropopause + 1] = T_TROPOPAUSE
    rh_ndarray[:, : k_tropopause + 1] = RH_MIN

    # below tropopause
    theta_ndarray[:, k_tropopause + 1 :] = (
        THETA_SURFACE
        + (THETA_TROPOPAUSE - THETA_SURFACE)
        * (model_level_height[:, k_tropopause + 1 :] / H_TROPOPAUSE) ** EXP_FACTOR_THETA
    )
    rh_ndarray[:, k_tropopause + 1 :] = xp.minimum(
        1.0 - 0.75 * (model_level_height[:, k_tropopause + 1 :] / H_TROPOPAUSE) ** EXP_FACTOR_RH,
        RH_MAX,
    )

    # piecewise vertical integration from the TP towards the bottom
    # 1st step: preliminary estimate
    theta_v_aux = theta_ndarray[:, k_tropopause + 1] * (
        1.0 + phy_const.RV_O_RD_MINUS_1 * qv_tropopause
    )
    exner_aux = exner_tropopause - phy_const.GRAV_O_CPD * (
        model_level_height[:, k_tropopause + 1] - H_TROPOPAUSE
    ) / (theta_v_aux - theta_v_tropopause) * xp.log(theta_v_aux / theta_v_tropopause)
    temperature_aux = theta_ndarray[:, k_tropopause + 1] * exner_aux
    vapor_pres_aux = rh_ndarray[
        :, k_tropopause + 1
    ] * thermodynamic_functions.calculate_saturation_presssure_water(temperature_aux)
    pressure_aux = phy_const.P0REF * exner_aux**phy_const.CPD_O_RD
    qv_aux = thermodynamic_functions.calculate_specific_humidity(
        pressure=pressure_aux, vapor_pressure=vapor_pres_aux
    )
    theta_v_aux = theta_ndarray[:, k_tropopause + 1] * (1.0 + phy_const.RV_O_RD_MINUS_1 * qv_aux)

    # 2nd step: final computation
    exner_ndarray[:, k_tropopause + 1] = exner_tropopause - phy_const.GRAV_O_CPD * (
        model_level_height[:, k_tropopause + 1] - H_TROPOPAUSE
    ) / (theta_v_aux - theta_v_tropopause) * xp.log(theta_v_aux / theta_v_tropopause)
    temperature_ndarray[:, k_tropopause + 1] = (
        theta_ndarray[:, k_tropopause + 1] * exner_ndarray[:, k_tropopause + 1]
    )
    vapor_pres_aux = rh_ndarray[
        :, k_tropopause + 1
    ] * thermodynamic_functions.calculate_saturation_presssure_water(
        temperature_ndarray[:, k_tropopause + 1]
    )
    pressure_ndarray[:, k_tropopause + 1] = (
        phy_const.P0REF * exner_ndarray[:, k_tropopause + 1] ** phy_const.CPD_O_RD
    )
    qv_ndarray[:, k_tropopause + 1] = thermodynamic_functions.calculate_specific_humidity(
        pressure=pressure_ndarray[:, k_tropopause + 1], vapor_pressure=vapor_pres_aux
    )
    theta_v_ndarray[:, k_tropopause + 1] = theta_ndarray[:, k_tropopause + 1] * (
        1.0 + phy_const.RV_O_RD_MINUS_1 * qv_ndarray[:, k_tropopause + 1]
    )

    # remaining model layers
    for k in range(k_tropopause + 2, num_levels):
        # 1st step: preliminary estimate
        qv_extrap = xp.minimum(
            QV_MAX,
            qv_ndarray[:, k - 1]
            + (qv_ndarray[:, k - 2] - qv_ndarray[:, k - 1])
            / (model_level_height[:, k - 2] - model_level_height[:, k - 1])
            * (model_level_height[:, k] - model_level_height[:, k - 1]),
        )
        theta_v_aux = theta_ndarray[:, k] * (1.0 + phy_const.RV_O_RD_MINUS_1 * qv_extrap)
        exner_aux = exner_ndarray[:, k - 1] - phy_const.GRAV_O_CPD * (
            model_level_height[:, k] - model_level_height[:, k - 1]
        ) / (theta_v_aux - theta_v_ndarray[:, k - 1]) * xp.log(
            theta_v_aux / theta_v_ndarray[:, k - 1]
        )
        temperature_aux = theta_ndarray[:, k] * exner_aux
        vapor_pres_aux = rh_ndarray[
            :, k
        ] * thermodynamic_functions.calculate_saturation_presssure_water(temperature_aux)
        pressure_aux = phy_const.P0REF * exner_aux**phy_const.CPD_O_RD
        qv_aux = xp.minimum(
            QV_MAX,
            thermodynamic_functions.calculate_specific_humidity(
                pressure=pressure_aux, vapor_pressure=vapor_pres_aux
            ),
        )
        theta_v_aux = theta_ndarray[:, k] * (1.0 + phy_const.RV_O_RD_MINUS_1 * qv_aux)

        # 2nd step: final computation
        exner_ndarray[:, k] = exner_ndarray[:, k - 1] - phy_const.GRAV_O_CPD * (
            model_level_height[:, k] - model_level_height[:, k - 1]
        ) / (theta_v_aux - theta_v_ndarray[:, k - 1]) * xp.log(
            theta_v_aux / theta_v_ndarray[:, k - 1]
        )
        temperature_ndarray[:, k] = theta_ndarray[:, k] * exner_ndarray[:, k]
        vapor_pres_aux = rh_ndarray[
            :, k
        ] * thermodynamic_functions.calculate_saturation_presssure_water(temperature_ndarray[:, k])
        pressure_ndarray[:, k] = phy_const.P0REF * exner_ndarray[:, k] ** phy_const.CPD_O_RD
        qv_ndarray[:, k] = xp.minimum(
            QV_MAX,
            thermodynamic_functions.calculate_specific_humidity(
                pressure=pressure_ndarray[:, k], vapor_pressure=vapor_pres_aux
            ),
        )
        theta_v_ndarray[:, k] = theta_ndarray[:, k] * (
            1.0 + phy_const.RV_O_RD_MINUS_1 * qv_ndarray[:, k]
        )

    rho_ndarray[:, :] = (
        exner_ndarray[:, :] ** phy_const.CVD_O_RD
        * phy_const.P0REF
        / phy_const.RD
        / theta_v_ndarray[:, :]
    )
    vn_ndarray[:, :] = (
        U_INFTY * (xp.tanh((model_level_height[0, :] - HMIN) / (HREF - HMIN)) - 0.45) * primal_normal_x[:, xp.newaxis]
    )
    log.info("Computations of Weisman-Klemp background fields completed.")

    prognostic_state_now.w.ndarray[:, :] = testcases_utils.init_w(
        grid,
        c2e=grid.get_connectivity(dims.C2E).ndarray,
        e2c=grid.get_connectivity(dims.E2C).ndarray,
        z_ifc=metrics_field_source.get(metrics_attributes.CELL_HEIGHT_ON_HALF_LEVEL).ndarray,
        inv_dual_edge_length=geometry_field_source.get(
            f"inverse_of_{geometry_meta.DUAL_EDGE_LENGTH}"
        ).ndarray,
        edge_cell_length=geometry_field_source.get(geometry_meta.EDGE_CELL_DISTANCE).ndarray,
        primal_edge_length=geometry_field_source.get(geometry_meta.EDGE_LENGTH).ndarray,
        cell_area=geometry_field_source.get(geometry_meta.CELL_AREA).ndarray,
        vn=prognostic_state_now.vn.ndarray,
        vct_b=vertical_grid._vct_b.ndarray,  # type: ignore[union-attr]
        nlev=num_levels,
        array_ns=xp,
    )
    log.info("W initialization completed.")

    functools.partial(testcases_utils.apply_hydrostatic_adjustment_ndarray, array_ns=xp)(
        rho=rho_ndarray,
        exner=exner_ndarray,
        theta_v=theta_v_ndarray,
        exner_ref_mc=exner_ref_mc,
        d_exner_dz_ref_ic=d_exner_dz_ref_ic,
        theta_ref_mc=theta_ref_mc,
        theta_ref_ic=theta_ref_ic,
        wgtfac_c=wgtfac_c,
        ddqz_z_half=ddqz_z_half,
        num_levels=num_levels,
    )
    log.info("Hydrostatic adjustment computation completed.")

    domain_length = grid.global_properties.domain_length
    domain_height = grid.global_properties.domain_height
    functools.partial(testcases_utils.init_bubble, array_ns=xp)(
        theta_v_ndarray=theta_v_ndarray,
        rho_ndarray=rho_ndarray,
        qv_ndarray=qv_ndarray,
        exner_ndarray=exner_ndarray,
        cell_cartesian_x=geometry_field_source.get(geometry_meta.CELL_CENTER_X).ndarray[:, xp.newaxis],
        cell_cartesian_y=geometry_field_source.get(geometry_meta.CELL_CENTER_Y).ndarray[:, xp.newaxis],
        z_mc=metrics_field_source.get(metrics_attributes.Z_MC).ndarray,
        bubble_center_x=bubble_center_x,
        bubble_center_y=bubble_center_y,
        bubble_center_z=bubble_center_z,
        bubble_width=bubble_width,
        bubble_height=bubble_height,
        bubble_amplitude=bubble_amplitude,
        geometry_type=grid.geometry_type,
        domain_length=domain_length,  # type: ignore[arg-type]
        domain_height=domain_height,  # type: ignore[arg-type]
    )
    log.info("Bubble initialization completed.")

    prognostic_state_next = prognostics.PrognosticState(
        vn=data_alloc.as_field(prognostic_state_now.vn, allocator=allocator),
        w=data_alloc.as_field(prognostic_state_now.w, allocator=allocator),
        exner=data_alloc.as_field(prognostic_state_now.exner, allocator=allocator),
        rho=data_alloc.as_field(prognostic_state_now.rho, allocator=allocator),
        theta_v=data_alloc.as_field(prognostic_state_now.theta_v, allocator=allocator),
    )
    prognostic_states = common_utils.TimeStepPair(prognostic_state_now, prognostic_state_next)

    tracer_state_next = tracers.TracerState.from_tracer_state(tracer_state_now, allocator)
    tracer_states = common_utils.TimeStepPair(tracer_state_now, tracer_state_next)

    perturbed_exner = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator)
    gt4py_math_op.compute_difference_on_cell_k.with_backend(backend)(
        field_a=prognostic_states.current.exner,
        field_b=metrics_field_source.get(metrics_attributes.EXNER_REF_MC),
        output_field=perturbed_exner,
        horizontal_start=0,
        horizontal_end=num_cells,
        vertical_start=0,
        vertical_end=num_levels,
        offset_provider={},
    )
    log.info("perturbed_exner initialization completed.")

    diffusion_diagnostic_state = diffusion_states.initialize_diffusion_diagnostic_state(
        grid=grid, allocator=allocator
    )
    solve_nonhydro_diagnostic_state = dycore_states.initialize_solve_nonhydro_diagnostic_state(
        perturbed_exner_at_cells_on_model_levels=perturbed_exner,
        grid=grid,
        allocator=allocator,
    )
    prep_adv = dycore_states.initialize_prep_advection(grid=grid, allocator=allocator)
    tracer_advection_diagnostic_state = advection_states.initialize_advection_diagnostic_state(
        grid=grid, allocator=allocator
    )
    prep_tracer_adv = advection_states.AdvectionPrepAdvState(
        vn_traj=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator),
        mass_flx_me=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator),
        mass_flx_ic=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator),
    )

    log.info("Weisman-Klemp initialization completed.")

    ds = driver_states.DriverStates(
        prep_advection_prognostic=prep_adv,
        solve_nonhydro_diagnostic=solve_nonhydro_diagnostic_state,
        prep_tracer_advection_prognostic=prep_tracer_adv,
        tracer_advection_diagnostic=tracer_advection_diagnostic_state,
        diffusion_diagnostic=diffusion_diagnostic_state,
        prognostics=prognostic_states,
        diagnostic=diagnostic_state,
        tracers=tracer_states,
    )

    return ds
