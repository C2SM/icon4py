# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import logging
import math
from typing import TYPE_CHECKING, ClassVar

from icon4py.model.common import (
    constants as phy_const,
    dimension as dims,
    model_backends,
    type_alias as ta,
)
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import (
    geometry as grid_geometry,
    icon as icon_grid,
    vertical as v_grid,
)
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.interpolation.stencils import cell_2_edge_interpolation
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver.initial_condition.analytical import utils as testcases_utils


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.standalone_driver import driver_states

log = logging.getLogger(__name__)


@dataclasses.dataclass
class JablonowskiWilliamsonConfig:
    # jabw* and APE testcases share this initial condition and config
    # with the difference that jabw* has p_sfc hardcoded to 1e5, while APE
    # reads zp_ape from the nh_testcase_nml
    # The default values are from mo_nh_jabw_exp.f90 and mo_nh_testcases_nml.f90
    p_sfc: float = 100000.0
    baroclinic_amplitude: float = 0.0
    u0: float = 35.0
    temp0: float = 288.0
    eta_0: float = 0.252
    eta_t: float = 0.2
    gamma: float = 0.005
    dtemp: float = 4.8e5
    lon_perturbation_center: float = math.pi / 9.0
    lat_perturbation_center: float = 2.0 * math.pi / 9.0

    fortran_name_map: ClassVar[dict[str, str]] = {
        "jw_up": "baroclinic_amplitude",
        "jw_u0": "u0",
        "jw_temp0": "temp0",
        "zp_ape": "p_sfc",
    }


def jablonowski_williamson(  # noqa: PLR0915 [too-many-statements]
    *,
    config: JablonowskiWilliamsonConfig,
    vertical_config: v_grid.VerticalGridConfig,
    grid: icon_grid.IconGrid,
    geometry_field_source: grid_geometry.GridGeometry,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
) -> driver_states.DriverStates:
    """
    Initial condition for Jablonowski-Williamson test.
    Set jw_baroclinic_amplitude to values larger than 0.01 if you want to run
    baroclinic case.

    The reference experiment config for this is
    exp.exclaim_nh35_tri_jws_sb.
    """
    allocator = model_backends.get_allocator(backend)
    array_ns = data_alloc.import_array_ns(allocator)

    metrics = testcases_utils.extract_metrics(metrics_field_source)
    geometry = testcases_utils.extract_geometry(geometry_field_source)
    interp = testcases_utils.extract_interpolation(interpolation_field_source)
    zone_idx = testcases_utils.zone_indices(grid)

    p_sfc = config.p_sfc
    jw_baroclinic_amplitude = config.baroclinic_amplitude
    u0 = config.u0
    temp0 = config.temp0
    eta_0 = config.eta_0
    eta_t = config.eta_t
    gamma = config.gamma
    dtemp = config.dtemp
    lon_perturbation_center = config.lon_perturbation_center
    lat_perturbation_center = config.lat_perturbation_center

    num_cells = grid.num_cells
    num_levels = grid.num_levels

    prognostic_state_now = prognostics.initialize_prognostic_state(grid=grid, allocator=allocator)
    diagnostic_state = diagnostics.initialize_diagnostic_state(grid=grid, allocator=allocator)
    eta_v = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
    )
    eta_v_at_edge = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator)

    exner_ndarray = prognostic_state_now.exner.ndarray
    rho_ndarray = prognostic_state_now.rho.ndarray
    theta_v_ndarray = prognostic_state_now.theta_v.ndarray
    temperature_ndarray = diagnostic_state.temperature.ndarray
    pressure_ndarray = diagnostic_state.pressure.ndarray
    eta_v_ndarray = eta_v.ndarray

    diagnostic_state.pressure_ifc.ndarray[:, -1] = p_sfc

    sin_lat = array_ns.sin(geometry["cell_lat"])
    cos_lat = array_ns.cos(geometry["cell_lat"])
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
        eta_old = array_ns.full(num_cells, fill_value=ta.wpfloat("1.0e-7"), dtype=ta.wpfloat)
        log.info(f"In Newton iteration, k = {k_index}")
        for _ in range(100):
            eta_v_ndarray[:, k_index] = (eta_old - eta_0) * math.pi * 0.5
            cos_etav = array_ns.cos(eta_v_ndarray[:, k_index])
            sin_etav = array_ns.sin(eta_v_ndarray[:, k_index])

            temperature_avg = temp0 * (eta_old**lapse_rate)
            geopot_avg = temp0 * phy_const.GRAV / gamma * (ta.wpfloat("1.0") - eta_old**lapse_rate)
            temperature_avg = array_ns.where(
                eta_old < eta_t, temperature_avg + dtemp * ((eta_t - eta_old) ** 5), temperature_avg
            )
            geopot_avg = array_ns.where(
                eta_old < eta_t,
                geopot_avg
                - phy_const.RD
                * dtemp
                * (
                    (array_ns.log(eta_old / eta_t) + ta.wpfloat("137.0") / ta.wpfloat("60.0"))
                    * (eta_t**5)
                    - ta.wpfloat("5.0") * (eta_t**4) * eta_old
                    + ta.wpfloat("5.0") * (eta_t**3) * (eta_old**2)
                    - ta.wpfloat("10.0") / ta.wpfloat("3.0") * (eta_t**2) * (eta_old**3)
                    + ta.wpfloat("1.25") * eta_t * (eta_old**4)
                    - ta.wpfloat("0.2") * (eta_old**5)
                ),
                geopot_avg,
            )

            geopot_jw = geopot_avg + u0 * (cos_etav**1.5) * (fac1 * u0 * (cos_etav**1.5) + fac2)
            temperature_jw = temperature_avg + ta.wpfloat(
                "0.75"
            ) * eta_old * math.pi * u0 / phy_const.RD * sin_etav * array_ns.sqrt(cos_etav) * (
                ta.wpfloat("2.0") * u0 * fac1 * (cos_etav**1.5) + fac2
            )
            newton_function = geopot_jw - metrics["geopot"][:, k_index]
            newton_function_prime = -phy_const.RD / eta_old * temperature_jw
            eta_old = eta_old - newton_function / newton_function_prime

        eta_v_ndarray[:, k_index] = (eta_old - eta_0) * math.pi * 0.5
        exner_ndarray[:, k_index] = (eta_old * p_sfc / phy_const.P0REF) ** phy_const.RD_O_CPD
        theta_v_ndarray[:, k_index] = temperature_jw / exner_ndarray[:, k_index]
        rho_ndarray[:, k_index] = (
            exner_ndarray[:, k_index] ** phy_const.CVD_O_RD
            * phy_const.P0REF
            / phy_const.RD
            / theta_v_ndarray[:, k_index]
        )
        pressure_ndarray[:, k_index] = (
            phy_const.P0REF * exner_ndarray[:, k_index] ** phy_const.CPD_O_RD
        )
        temperature_ndarray[:, k_index] = temperature_jw
    log.info("Newton iteration completed.")

    cell_2_edge_interpolation.cell_2_edge_interpolation.with_backend(backend)(
        in_field=eta_v,
        coeff=interp["c_lin_e"],
        out_field=eta_v_at_edge,
        horizontal_start=zone_idx["end_edge_lateral_boundary_level_2"],
        horizontal_end=zone_idx["end_edge_end"],
        vertical_start=0,
        vertical_end=num_levels,
        offset_provider=grid.connectivities,
    )
    exchange.exchange(dims.EdgeDim, eta_v_at_edge)
    log.info("Cell-to-edge eta_v computation completed.")

    prognostic_state_now.vn.ndarray[:, :] = testcases_utils.zonalwind_2_normalwind_ndarray(
        grid=grid,
        u0=u0,
        baroclinic_amplitude=jw_baroclinic_amplitude,
        lat_perturbation_center=lat_perturbation_center,
        lon_perturbation_center=lon_perturbation_center,
        edge_lat=geometry["edge_lat"],
        edge_lon=geometry["edge_lon"],
        primal_normal_x=geometry["primal_normal_x"],
        eta_v_at_edge=eta_v_at_edge.ndarray,
    )
    log.info("U2vn computation completed.")

    _, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, allocator)

    prognostic_state_now.w.ndarray[:, :] = testcases_utils.init_w(
        grid=grid,
        z_ifc=metrics["z_ifc"],
        inv_dual_edge_length=geometry["inv_dual_edge_length"],
        edge_cell_distance=geometry["edge_cell_distance"],
        primal_edge_length=geometry["primal_edge_length"],
        cell_area=geometry["cell_area"],
        vn=prognostic_state_now.vn.ndarray,
        vct_b=vct_b.ndarray,
        nlev=num_levels,
    )
    exchange.exchange(dims.CellDim, prognostic_state_now.w)

    testcases_utils.apply_hydrostatic_adjustment_ndarray(
        rho=rho_ndarray,
        exner=exner_ndarray,
        theta_v=theta_v_ndarray,
        exner_ref_mc=metrics["exner_ref_mc"],
        d_exner_dz_ref_ic=metrics["d_exner_dz_ref_ic"],
        theta_ref_mc=metrics["theta_ref_mc"],
        theta_ref_ic=metrics["theta_ref_ic"],
        wgtfac_c=metrics["wgtfac_c"],
        ddqz_z_half=metrics["ddqz_z_half"],
        num_levels=num_levels,
    )
    log.info("Hydrostatic adjustment computation completed.")

    return testcases_utils.assemble_driver_states(
        grid=grid,
        allocator=allocator,
        backend=backend,
        exchange=exchange,
        interpolation=interp,
        zone_indices_map=zone_idx,
        metrics_field_source=metrics_field_source,
        prognostic_state_now=prognostic_state_now,
        diagnostic_state=diagnostic_state,
    )
