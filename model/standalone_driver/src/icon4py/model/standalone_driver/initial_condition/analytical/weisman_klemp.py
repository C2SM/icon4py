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
from typing import TYPE_CHECKING, ClassVar

from icon4py.model.common import (
    constants as phy_const,
    dimension as dims,
    model_backends,
    thermodynamic_functions as thermo,
)
from icon4py.model.common.grid import (
    geometry_attributes as geometry_meta,
    icon as icon_grid,
    vertical as v_grid,
)
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import driver_states
from icon4py.model.standalone_driver.initial_condition.analytical import utils as testcases_utils


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.decomposition import definitions as decomposition_defs

log = logging.getLogger(__name__)


@dataclasses.dataclass
class WeismanKlempConfig:
    # The default values are from mo_nh_wk_exp.f90 and mo_nh_testcases_nml.f90.
    #: base height of the profile [m] (hmin_wk)
    h_min: float = 0.0
    #: height of the tropopause [m] (h_tropo_wk)
    h_tropopause: float = 12000.0
    #: potential temperature at the surface [K] (theta_0_wk)
    theta_surface: float = 300.0
    #: potential temperature at the tropopause [K] (theta_tropo_wk)
    theta_tropopause: float = 343.0
    #: exponent of the potential-temperature profile below the tropopause (expo_theta_wk)
    exponent_theta: float = 1.25
    #: exponent of the relative-humidity profile below the tropopause (expo_relhum_wk)
    exponent_relative_humidity: float = 1.25
    #: temperature of the (isothermal) tropopause [K] (t_tropo_wk)
    t_tropopause: float = 213.0
    #: relative humidity above the tropopause (rh_min_wk)
    rh_min: float = 0.10
    #: maximum relative humidity below the tropopause (rh_max_wk)
    rh_max: float = 0.95
    #: scaling height of the wind profile (height of 70% wind speed) [m] (href_wk)
    wind_scale_height: float = 3000.0
    #: maximum horizontal wind speed [m/s] (u_infty_wk)
    max_wind_speed: float = 15.0
    #: maximum moisture content below the tropopause [kg/kg] (qv_max_wk)
    qv_max: float = 0.014
    #: x coordinate (torus) or longitude in degrees (sphere) of the bubble centre (bubctr_lon)
    bubble_center_x: float = 0.0
    #: y coordinate (torus) or latitude in degrees (sphere) of the bubble centre (bubctr_lat)
    bubble_center_y: float = 0.0
    #: altitude of the bubble centre [m] (bubctr_z)
    bubble_center_z: float = 1400.0
    #: horizontal size of the warm bubble [m] (bub_hor_width)
    bubble_horizontal_width: float = 5000.0
    #: vertical size of the warm bubble [m] (bub_ver_width)
    bubble_vertical_width: float = 1400.0
    #: temperature amplitude of the warm bubble [K] (bub_amp)
    bubble_amplitude: float = 2.0
    #: normalized radius below which the bubble perturbation is applied
    bubble_radius: float = 1.0

    fortran_name_map: ClassVar[dict[str, str]] = {
        "qv_max_wk": "qv_max",
        "u_infty_wk": "max_wind_speed",
        "bub_hor_width": "bubble_horizontal_width",
        "bub_ver_width": "bubble_vertical_width",
        "bubctr_lon": "bubble_center_x",
        "bubctr_lat": "bubble_center_y",
        "bubctr_z": "bubble_center_z",
        "bub_amp": "bubble_amplitude",
    }


def weisman_klemp(  # noqa: PLR0915 [too-many-statements]
    *,
    config: WeismanKlempConfig,
    vertical_config: v_grid.VerticalGridConfig,
    grid: icon_grid.IconGrid,
    static_fields: driver_states.StaticFieldFactories,
    prognostic_state_now: prognostics.PrognosticState,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
) -> None:
    """
    Initial condition for the Weisman-Klemp (WK82) idealized convection test case.

    A horizontally homogeneous, hydrostatically balanced moist base state is built by
    integrating downward from the tropopause, a sheared horizontal wind is prescribed,
    and a warm bubble is added to trigger convection. The moist base state requires the
    water-vapour tracer ``qv`` to be active.

    The reference experiment config for this is exp.exclaim_nh_weisman_klemp_sb.
    """
    if prognostic_state_now.tracer.qv is None:
        raise ValueError(
            "The Weisman-Klemp initial condition requires the 'qv' tracer to be active."
        )

    allocator = model_backends.get_allocator(backend)
    array_ns = data_alloc.import_array_ns(allocator)

    geometry = static_fields.geometry
    metrics = static_fields.metrics
    primal_normal_x = geometry.get(geometry_meta.EDGE_NORMAL_U).ndarray
    inv_dual_edge_length = geometry.get(f"inverse_of_{geometry_meta.DUAL_EDGE_LENGTH}").ndarray
    edge_cell_distance = geometry.get(geometry_meta.EDGE_CELL_DISTANCE).ndarray
    primal_edge_length = geometry.get(geometry_meta.EDGE_LENGTH).ndarray
    cell_area = geometry.get(geometry_meta.CELL_AREA).ndarray
    z_mc = metrics.get(metrics_attributes.Z_MC).ndarray
    z_ifc = metrics.get(metrics_attributes.CELL_HEIGHT_ON_HALF_LEVEL).ndarray
    exner_ref_mc = metrics.get(metrics_attributes.EXNER_REF_MC).ndarray
    d_exner_dz_ref_ic = metrics.get(metrics_attributes.D_EXNER_DZ_REF_IC).ndarray
    theta_ref_mc = metrics.get(metrics_attributes.THETA_REF_MC).ndarray
    theta_ref_ic = metrics.get(metrics_attributes.THETA_REF_IC).ndarray
    wgtfac_c = metrics.get(metrics_attributes.WGTFAC_C).ndarray
    ddqz_z_half = metrics.get(metrics_attributes.DDQZ_Z_HALF).ndarray
    zone_idx = testcases_utils.zone_indices(grid)

    num_edges = grid.num_edges
    num_levels = grid.num_levels

    grav_o_cpd = phy_const.GRAV_O_CPD
    vtmpc1 = phy_const.RV_O_RD_MINUS_1
    h_tropopause = config.h_tropopause
    t_tropopause = config.t_tropopause
    theta_tropopause = config.theta_tropopause
    qv_max = config.qv_max

    exner_ndarray = prognostic_state_now.exner.ndarray
    rho_ndarray = prognostic_state_now.rho.ndarray
    theta_v_ndarray = prognostic_state_now.theta_v.ndarray

    # With flat topography (wk82) the height of the model levels is horizontally
    # homogeneous, so the base-state profiles depend on the vertical level only.
    height = z_mc[0, :]

    # Level index right above the tropopause (levels are ordered top to bottom, so the
    # height decreases with the level index).
    k_tropopause = int(array_ns.sum(height >= h_tropopause)) - 1
    if not 0 <= k_tropopause < num_levels - 1:
        raise ValueError(f"The model top must be higher than the tropopause at {h_tropopause} m.")

    theta = array_ns.zeros((num_levels,))
    exner = array_ns.zeros((num_levels,))
    qv = array_ns.zeros((num_levels,))
    theta_v = array_ns.zeros((num_levels,))
    relative_humidity = array_ns.zeros((num_levels,))
    temperature = array_ns.zeros((num_levels,))

    # Tropopause reference values.
    exner_tropopause = t_tropopause / theta_tropopause
    vapor_pressure_tropopause = config.rh_min * thermo.sat_pres_water(
        array_ns.asarray(t_tropopause)
    )
    pressure_tropopause = phy_const.P0REF * exner_tropopause**phy_const.CPD_O_RD
    qv_tropopause = thermo.specific_humidity(vapor_pressure_tropopause, pressure_tropopause)
    theta_v_tropopause = theta_tropopause * (1.0 + vtmpc1 * qv_tropopause)

    # Above the tropopause the layer is isothermal (T = t_tropopause).
    above = slice(0, k_tropopause + 1)
    height_above = height[above]
    theta[above] = theta_tropopause * array_ns.exp(
        grav_o_cpd / t_tropopause * (height_above - h_tropopause)
    )
    exner[above] = exner_tropopause * array_ns.exp(
        -grav_o_cpd / t_tropopause * (height_above - h_tropopause)
    )
    pressure_above = phy_const.P0REF * exner[above] ** phy_const.CPD_O_RD
    qv[above] = thermo.specific_humidity(vapor_pressure_tropopause, pressure_above)
    theta_v[above] = theta[above] * (1.0 + vtmpc1 * qv[above])
    relative_humidity[above] = config.rh_min
    temperature[above] = t_tropopause

    # Below the tropopause the potential temperature and relative humidity profiles are
    # prescribed; exner, qv and theta_v then follow from a downward hydrostatic integration.
    below = slice(k_tropopause + 1, num_levels)
    height_below = height[below]
    theta[below] = (
        config.theta_surface
        + (theta_tropopause - config.theta_surface)
        * (height_below / h_tropopause) ** config.exponent_theta
    )
    relative_humidity[below] = array_ns.minimum(
        1.0 - 0.75 * (height_below / h_tropopause) ** config.exponent_relative_humidity,
        config.rh_max,
    )

    def _integrate_layer(
        k: int,
        exner_above: data_alloc.NDArray,
        theta_v_above: data_alloc.NDArray,
        qv_estimate: data_alloc.NDArray,
    ) -> None:
        # Two-pass piecewise hydrostatic integration of one layer (see mo_nh_wk_exp.f90):
        # a preliminary qv estimate gives theta_v, which sets exner, from which qv and
        # theta_v are finally recomputed.
        delta_height = (
            height[k] - height[k - 1] if k > k_tropopause + 1 else (height[k] - h_tropopause)
        )

        theta_v_aux = theta[k] * (1.0 + vtmpc1 * qv_estimate)
        exner_aux = exner_above - grav_o_cpd * delta_height / (
            theta_v_aux - theta_v_above
        ) * array_ns.log(theta_v_aux / theta_v_above)
        vapor_pressure_aux = relative_humidity[k] * thermo.sat_pres_water(theta[k] * exner_aux)
        pressure_aux = phy_const.P0REF * exner_aux**phy_const.CPD_O_RD
        qv_aux = thermo.specific_humidity(vapor_pressure_aux, pressure_aux)
        if k > k_tropopause + 1:
            qv_aux = array_ns.minimum(qv_max, qv_aux)
        theta_v_aux = theta[k] * (1.0 + vtmpc1 * qv_aux)

        exner[k] = exner_above - grav_o_cpd * delta_height / (
            theta_v_aux - theta_v_above
        ) * array_ns.log(theta_v_aux / theta_v_above)
        temperature[k] = theta[k] * exner[k]
        vapor_pressure = relative_humidity[k] * thermo.sat_pres_water(temperature[k])
        pressure = phy_const.P0REF * exner[k] ** phy_const.CPD_O_RD
        qv[k] = thermo.specific_humidity(vapor_pressure, pressure)
        if k > k_tropopause + 1:
            qv[k] = array_ns.minimum(qv_max, qv[k])
        theta_v[k] = theta[k] * (1.0 + vtmpc1 * qv[k])

    # First layer below the tropopause uses the tropopause reference values.
    _integrate_layer(k_tropopause + 1, exner_tropopause, theta_v_tropopause, qv_tropopause)
    # Remaining layers extrapolate qv from the two levels above as a first guess.
    for k in range(k_tropopause + 2, num_levels):
        qv_extrapolated = array_ns.minimum(
            qv_max,
            qv[k - 1]
            + (qv[k - 2] - qv[k - 1])
            / (height[k - 2] - height[k - 1])
            * (height[k] - height[k - 1]),
        )
        _integrate_layer(k, exner[k - 1], theta_v[k - 1], qv_extrapolated)

    # Broadcast the column profiles onto all cells.
    exner_ndarray[:, :] = exner[array_ns.newaxis, :]
    theta_v_ndarray[:, :] = theta_v[array_ns.newaxis, :]
    rho_ndarray[:, :] = (
        exner_ndarray**phy_const.CVD_O_RD * phy_const.P0REF / phy_const.RD / theta_v_ndarray
    )
    prognostic_state_now.tracer.qv.ndarray[:, :] = qv[array_ns.newaxis, :]
    log.info("Weisman-Klemp base-state profile completed.")

    # Sheared horizontal wind, projected onto the edge-normal direction.
    wind_speed = config.max_wind_speed * (
        array_ns.tanh((height - config.h_min) / (config.wind_scale_height - config.h_min)) - 0.45
    )
    interior_edge = array_ns.ones(num_edges, dtype=bool)
    interior_edge[0 : zone_idx["end_edge_lateral_boundary_level_2"]] = False
    prognostic_state_now.vn.ndarray[:, :] = (
        array_ns.where(interior_edge[:, array_ns.newaxis], wind_speed[array_ns.newaxis, :], 0.0)
        * primal_normal_x[:, array_ns.newaxis]
    )

    _, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, allocator)
    prognostic_state_now.w.ndarray[:, :] = testcases_utils.init_w(
        grid=grid,
        z_ifc=z_ifc,
        inv_dual_edge_length=inv_dual_edge_length,
        edge_cell_distance=edge_cell_distance,
        primal_edge_length=primal_edge_length,
        cell_area=cell_area,
        vn=prognostic_state_now.vn.ndarray,
        vct_b=vct_b.ndarray,
        nlev=num_levels,
    )
    exchange.exchange(dims.CellDim, prognostic_state_now.w)

    testcases_utils.apply_hydrostatic_adjustment_ndarray(
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

    testcases_utils.init_bubble(
        grid=grid,
        geometry=geometry,
        z_mc=z_mc,
        theta_v=theta_v_ndarray,
        rho=rho_ndarray,
        qv=prognostic_state_now.tracer.qv.ndarray,
        exner=exner_ndarray,
        center_x=config.bubble_center_x,
        center_y=config.bubble_center_y,
        center_z=config.bubble_center_z,
        horizontal_width=config.bubble_horizontal_width,
        vertical_width=config.bubble_vertical_width,
        amplitude=config.bubble_amplitude,
        radius=config.bubble_radius,
    )
    log.info("Warm-bubble perturbation completed.")
