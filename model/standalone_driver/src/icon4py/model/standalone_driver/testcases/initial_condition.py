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
from typing import TYPE_CHECKING, Any, ClassVar

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
from icon4py.model.common.decomposition import definitions as decomposition_defs
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
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import driver_states
from icon4py.model.standalone_driver.testcases import utils as testcases_utils


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

log = logging.getLogger(__name__)


def _params_from_dict(cls: type, source: dict[str, Any]):
    """Construct a dataclass from a namelist dict.

    Unknown keys are ignored (e.g. topography params mixed into the same nml
    block).  Missing keys fall back to the dataclass field defaults.
    Fortran→Python name translation is driven by the optional ``_fortran_name_map``
    class variable: ``{fortran_key: python_field_name}``.
    """
    name_map: dict[str, str] = getattr(cls, "_fortran_name_map", {})
    known_fields = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in source.items():
        python_name = name_map.get(key, key)
        if python_name in known_fields:
            kwargs[python_name] = value
    return cls(**kwargs)


@dataclasses.dataclass
class InitialConditionConfig:
    parameters: JablonowskiWilliamsonParameters | Gauss3DParameters

    @classmethod
    def from_fortran_dict(cls, atm_dict: dict[str, Any], input_dict: dict[str, Any]):
        if not atm_dict["run_nml"].get("ltestcase", False):
            return None

        testcase_nml = input_dict.get("nh_testcase_nml", {})
        match testcase_nml.get("nh_test_name"):
            case "jabw" | "jabw_s":
                parameters = _params_from_dict(JablonowskiWilliamsonParameters, testcase_nml)
            case "gauss3D":
                parameters = _params_from_dict(Gauss3DParameters, testcase_nml)
            case name:
                raise ValueError(f"Unknown or missing test case name: {name!r}")

        return cls(parameters=parameters)


@dataclasses.dataclass
class JablonowskiWilliamsonParameters:
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

    _fortran_name_map: ClassVar[dict[str, str]] = {
        "jw_up": "baroclinic_amplitude",
        "jw_u0": "u0",
        "jw_temp0": "temp0",
    }

    create = jablonowski_williamson

def jablonowski_williamson(  # noqa: PLR0915 [too-many-statements]
    parameters: JablonowskiWilliamsonParameters,
    grid: icon_grid.IconGrid,
    geometry_field_source: grid_geometry.GridGeometry,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend: gtx_typing.Backend | None,
    lowest_layer_thickness: float,
    model_top_height: float,
    stretch_factor: float,
    damping_height: float,
    exchange: decomposition_defs.ExchangeRuntime,
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

    metrics = _extract_metrics(metrics_field_source)
    geometry = _extract_geometry(geometry_field_source)
    interp = _extract_interpolation(interpolation_field_source)
    zone_idx = _zone_indices(grid)

    p_sfc = parameters.p_sfc
    jw_baroclinic_amplitude = parameters.baroclinic_amplitude
    u0 = parameters.u0
    temp0 = parameters.temp0
    eta_0 = parameters.eta_0
    eta_t = parameters.eta_t
    gamma = parameters.gamma
    dtemp = parameters.dtemp
    lon_perturbation_center = parameters.lon_perturbation_center
    lat_perturbation_center = parameters.lat_perturbation_center

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

    sin_lat = xp.sin(geometry["cell_lat"])
    cos_lat = xp.cos(geometry["cell_lat"])
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
        for _ in range(100):
            eta_v_ndarray[:, k_index] = (eta_old - eta_0) * math.pi * 0.5
            cos_etav = xp.cos(eta_v_ndarray[:, k_index])
            sin_etav = xp.sin(eta_v_ndarray[:, k_index])

            temperature_avg = temp0 * (eta_old**lapse_rate)
            geopot_avg = temp0 * phy_const.GRAV / gamma * (ta.wpfloat("1.0") - eta_old**lapse_rate)
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

            geopot_jw = geopot_avg + u0 * (cos_etav**1.5) * (fac1 * u0 * (cos_etav**1.5) + fac2)
            temperature_jw = temperature_avg + ta.wpfloat(
                "0.75"
            ) * eta_old * math.pi * u0 / phy_const.RD * sin_etav * xp.sqrt(cos_etav) * (
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
        jw_u0=u0,
        jw_baroclinic_amplitude=jw_baroclinic_amplitude,
        lat_perturbation_center=lat_perturbation_center,
        lon_perturbation_center=lon_perturbation_center,
        edge_lat=geometry["edge_lat"],
        edge_lon=geometry["edge_lon"],
        primal_normal_x=geometry["primal_normal_x"],
        eta_v_at_edge=eta_v_at_edge.ndarray,
    )
    log.info("U2vn computation completed.")

    vertical_config = v_grid.VerticalGridConfig(
        grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
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

    return _assemble_driver_states(
        grid=grid,
        allocator=allocator,
        backend=backend,
        exchange=exchange,
        interpolation=interp,
        zone_indices=zone_idx,
        metrics_field_source=metrics_field_source,
        prognostic_state_now=prognostic_state_now,
        diagnostic_state=diagnostic_state,
    )



@dataclasses.dataclass
class Gauss3DParameters:
    u0: float = 0.0
    t0: float = 300.0
    brunt_vais: float = 0.01

    _fortran_name_map: ClassVar[dict[str, str]] = {
        "nh_u0": "u0",
        "nh_t0": "t0",
        "nh_brunt_vais": "brunt_vais",
    }

    create = gauss3d

def gauss3d(
    parameters: Gauss3DParameters,
    grid: icon_grid.IconGrid,
    geometry_field_source: grid_geometry.GridGeometry,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend: gtx_typing.Backend | None,
    lowest_layer_thickness: float,
    model_top_height: float,
    stretch_factor: float,
    damping_height: float,
    exchange: decomposition_defs.ExchangeRuntime,
) -> driver_states.DriverStates:
    allocator = model_backends.get_allocator(backend)
    xp = data_alloc.import_array_ns(allocator)

    metrics = _extract_metrics(metrics_field_source)
    geometry = _extract_geometry(geometry_field_source)
    zone_idx = _zone_indices(grid)

    num_edges = grid.num_edges
    num_levels = grid.num_levels

    u0 = parameters.u0
    t0 = parameters.t0
    brunt_vais = parameters.brunt_vais

    prognostic_state_now = prognostics.initialize_prognostic_state(grid=grid, allocator=allocator)
    diagnostic_state = diagnostics.initialize_diagnostic_state(grid=grid, allocator=allocator)

    exner_ndarray = prognostic_state_now.exner.ndarray
    rho_ndarray = prognostic_state_now.rho.ndarray
    theta_v_ndarray = prognostic_state_now.theta_v.ndarray

    mask_array_edge_start_plus1_to_edge_end = xp.ones(num_edges, dtype=bool)
    mask_array_edge_start_plus1_to_edge_end[0:zone_idx["end_edge_lateral_boundary_level_2"]] = False
    mask = xp.repeat(
        xp.expand_dims(mask_array_edge_start_plus1_to_edge_end, axis=-1),
        num_levels,
        axis=1,
    )
    u = xp.where(mask, u0, 0.0)
    prognostic_state_now.vn.ndarray[:, :] = u * geometry["primal_normal_x"]

    for k_index in range(num_levels - 1, -1, -1):
        z_help = (brunt_vais / phy_const.GRAV) ** 2 * metrics["geopot"][:, k_index]
        theta_v_ndarray[:, k_index] = t0 * xp.exp(z_help)

    if brunt_vais != 0.0:
        z_help = (brunt_vais / phy_const.GRAV) ** 2 * metrics["geopot"][:, num_levels - 1]
        exner_ndarray[:, num_levels - 1] = (
            phy_const.GRAV / brunt_vais
        ) ** 2 / t0 / phy_const.CPD * (xp.exp(-z_help) - 1.0) + 1.0
    else:
        exner_ndarray[:, num_levels - 1] = (
            1.0 - metrics["geopot"][:, num_levels - 1] / phy_const.CPD / t0
        )

    testcases_utils.hydrostatic_adjustment_constant_thetav_ndarray(
        wgtfac_c=metrics["wgtfac_c"],
        ddqz_z_half=metrics["ddqz_z_half"],
        exner_ref_mc=metrics["exner_ref_mc"],
        d_exner_dz_ref_ic=metrics["d_exner_dz_ref_ic"],
        theta_ref_mc=metrics["theta_ref_mc"],
        theta_ref_ic=metrics["theta_ref_ic"],
        rho=rho_ndarray,
        exner=exner_ndarray,
        theta_v=theta_v_ndarray,
        num_levels=num_levels,
    )
    log.info("Hydrostatic adjustment (constant theta_v) computation completed.")

    vertical_config = v_grid.VerticalGridConfig(
        grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
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

    return _assemble_driver_states(
        grid=grid,
        allocator=allocator,
        backend=backend,
        exchange=exchange,
        interpolation=_extract_interpolation(interpolation_field_source),
        zone_indices=zone_idx,
        metrics_field_source=metrics_field_source,
        prognostic_state_now=prognostic_state_now,
        diagnostic_state=diagnostic_state,
    )

def _extract_metrics(
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


def _extract_geometry(
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


def _extract_interpolation(
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


def _zone_indices(grid: icon_grid.IconGrid) -> dict[str, int]:
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


def _assemble_driver_states(
    grid: icon_grid.IconGrid,
    allocator: gtx_typing.Allocator,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
    interpolation: dict,
    zone_indices: dict[str, int],
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    prognostic_state_now: prognostics.PrognosticState,
    diagnostic_state: diagnostics.DiagnosticState,
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
        horizontal_start=zone_indices["end_cell_lateral_boundary_level_2"],
        horizontal_end=zone_indices["end_cell_end"],
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

    return driver_states.DriverStates(
        prep_advection_prognostic=prep_adv,
        solve_nonhydro_diagnostic=solve_nonhydro_diagnostic_state,
        prep_tracer_advection_prognostic=prep_tracer_adv,
        tracer_advection_diagnostic=tracer_advection_diagnostic_state,
        diffusion_diagnostic=diffusion_diagnostic_state,
        prognostics=prognostic_states,
        diagnostic=diagnostic_state,
    )
