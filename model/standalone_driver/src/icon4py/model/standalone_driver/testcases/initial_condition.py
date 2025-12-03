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

import gt4py.next as gtx

from icon4py.model.atmosphere.diffusion import diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.common import (
    constants as phy_const,
    dimension as dims,
    model_backends,
    model_options,
    type_alias as ta,
)
from icon4py.model.common.grid import (
    geometry as grid_geometry,
    geometry_attributes as geometry_meta,
    horizontal as h_grid,
    icon as icon_grid,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.interpolation.stencils import (
    cell_2_edge_interpolation,
    edge_2_cell_vector_rbf_interpolation,
)
from icon4py.model.common.math.stencils import generic_math_operations as gt4py_math_op
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import driver_states
from icon4py.model.standalone_driver.testcases import utils as testcases_utils


log = logging.getLogger(__name__)


def jablonowski_williamson(  # noqa: PLR0915 [too-many-statements]
    grid: icon_grid.IconGrid,
    geometry_field_source: grid_geometry.GridGeometry,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend: model_backends.BackendLike,
) -> driver_states.DriverStates:
    """
    Initial condition of Jablonowski-Williamson test. Set jw_up to values larger than 0.01 if
    you want to run baroclinic case.

    Args:
        grid: IconGrid
        geometry_field_source: geometric field factory
        interpolation_field_source: interpolation field factory
        metrics_field_source: metric field factory
        backend: GT4Py backend
    Returns: driver state
    """
    allocator = model_backends.get_allocator(backend)
    concrete_backend = model_options.customize_backend(program=None, backend=backend)
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

    p_sfc = ta.wpfloat("100000.0")
    jw_up = ta.wpfloat("0.0")  # if doing baroclinic wave test, please set it to a nonzero value
    jw_u0 = ta.wpfloat("35.0")
    jw_temp0 = ta.wpfloat("288.0")
    # DEFINED PARAMETERS for jablonowski williamson:
    eta_0 = ta.wpfloat("0.252")
    eta_t = ta.wpfloat("0.2")  # tropopause
    gamma = ta.wpfloat("0.005")  # temperature elapse rate (K/m)
    dtemp = ta.wpfloat("4.8e5")  # empirical temperature difference (K)
    # for baroclinic wave test
    lon_perturbation_center = math.pi / 9.0  # longitude of the perturb centre
    lat_perturbation_center = 2.0 * lon_perturbation_center  # latitude of the perturb centre
    ps_o_p0ref = p_sfc / phy_const.P0REF

    w_ndarray = xp.zeros((num_cells, num_levels + 1), dtype=ta.wpfloat)
    exner_ndarray = xp.zeros((num_cells, num_levels), dtype=ta.wpfloat)
    rho_ndarray = xp.zeros((num_cells, num_levels), dtype=ta.wpfloat)
    temperature_ndarray = xp.zeros((num_cells, num_levels), dtype=ta.wpfloat)
    pressure_ndarray = xp.zeros((num_cells, num_levels), dtype=ta.wpfloat)
    theta_v_ndarray = xp.zeros((num_cells, num_levels), dtype=ta.wpfloat)
    eta_v_ndarray = xp.zeros((num_cells, num_levels), dtype=ta.wpfloat)

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
        exner_ndarray[:, k_index] = (eta_old * ps_o_p0ref) ** phy_const.RD_O_CPD
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

    eta_v = gtx.as_field((dims.CellDim, dims.KDim), eta_v_ndarray, allocator=allocator)
    eta_v_e = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator)
    cell_2_edge_interpolation.cell_2_edge_interpolation.with_backend(concrete_backend)(
        in_field=eta_v,
        coeff=cell_2_edge_coeff,
        out_field=eta_v_e,
        horizontal_start=end_edge_lateral_boundary_level_2,
        horizontal_end=end_edge_end,
        vertical_start=0,
        vertical_end=num_levels,
        offset_provider=grid.connectivities,
    )
    log.info("Cell-to-edge eta_v computation completed.")

    vn_ndarray = functools.partial(testcases_utils.zonalwind_2_normalwind_ndarray, array_ns=xp)(
        grid=grid,
        jw_u0=jw_u0,
        jw_up=jw_up,
        lat_perturbation_center=lat_perturbation_center,
        lon_perturbation_center=lon_perturbation_center,
        edge_lat=edge_lat,
        edge_lon=edge_lon,
        primal_normal_x=primal_normal_x,
        eta_v_e=eta_v_e.ndarray,
    )

    log.info("U2vn computation completed.")

    rho_ndarray, exner_ndarray, theta_v_ndarray = functools.partial(
        testcases_utils.hydrostatic_adjustment_ndarray, array_ns=xp
    )(
        wgtfac_c=wgtfac_c,
        ddqz_z_half=ddqz_z_half,
        exner_ref_mc=exner_ref_mc,
        d_exner_dz_ref_ic=d_exner_dz_ref_ic,
        theta_ref_mc=theta_ref_mc,
        theta_ref_ic=theta_ref_ic,
        rho=rho_ndarray,
        exner=exner_ndarray,
        theta_v=theta_v_ndarray,
        num_levels=num_levels,
    )
    log.info("Hydrostatic adjustment computation completed.")

    pressure_ifc_ndarray = xp.zeros((num_cells, num_levels + 1), dtype=ta.wpfloat)
    pressure_ifc_ndarray[:, -1] = p_sfc

    (prognostics_states, diagnostic_state) = (
        testcases_utils.create_gt4py_field_for_prognostic_and_diagnostic_variables(
            vn_ndarray=vn_ndarray,
            w_ndarray=w_ndarray,
            exner_ndarray=exner_ndarray,
            rho_ndarray=rho_ndarray,
            theta_v_ndarray=theta_v_ndarray,
            temperature_ndarray=temperature_ndarray,
            pressure_ndarray=pressure_ndarray,
            pressure_ifc_ndarray=pressure_ifc_ndarray,
            grid=grid,
            allocator=allocator,
        )
    )

    edge_2_cell_vector_rbf_interpolation.edge_2_cell_vector_rbf_interpolation.with_backend(
        concrete_backend
    )(
        p_e_in=prognostics_states.current.vn,
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
    gt4py_math_op.compute_difference_on_cell_k.with_backend(concrete_backend)(
        field_a=prognostics_states.current.exner,
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
    log.info("Initialization completed.")

    ds = driver_states.DriverStates(
        prep_advection_prognostic=prep_adv,
        solve_nonhydro_diagnostic=solve_nonhydro_diagnostic_state,
        diffusion_diagnostic=diffusion_diagnostic_state,
        prognostics=prognostics_states,
        diagnostic=diagnostic_state,
    )

    return ds
