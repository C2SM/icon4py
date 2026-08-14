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
import types
from typing import TYPE_CHECKING, ClassVar

from icon4py.model.common import (
    constants as phy_const,
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    type_alias as ta,
)
from icon4py.model.common.grid import (
    geometry_attributes as geometry_meta,
    icon as icon_grid,
    vertical as v_grid,
)
from icon4py.model.common.initial_condition.analytical import utils as testcases_utils
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.states import prognostic_state as prognostics, tracer_states
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.decomposition import definitions as decomposition_defs
    from icon4py.model.common.states import static_fields

log = logging.getLogger(__name__)


@dataclasses.dataclass
class TracerBlobConfig:
    u0: float = 20.0
    t0: float = 300.0
    brunt_vais: float = 0.01
    # The blob parameters have no Fortran namelist counterpart and are dataclass-only.
    # None means: blob center at the domain center and blob radius equal to a quarter
    # of the smaller domain extent.
    blob_x: float | None = None
    blob_y: float | None = None
    blob_radius: float | None = None
    blob_amplitude: float = 1e-3

    fortran_name_map: ClassVar[dict[str, str]] = {
        "nh_u0": "u0",
        "nh_t0": "t0",
        "nh_brunt_vais": "brunt_vais",
    }


@dataclasses.dataclass
class TracerAdvectionPrescription:
    """Advection driving fields an IC prescribes when the dycore is disabled.

    The fields alias the driver's advection states (which are otherwise
    zero-allocated and only filled by a running dycore).
    """

    vn_traj: fa.EdgeKField[ta.wpfloat]
    mass_flx_me: fa.EdgeKField[ta.wpfloat]
    mass_flx_ic: fa.CellKField[ta.wpfloat]
    airmass_now: fa.CellKField[ta.wpfloat]
    airmass_new: fa.CellKField[ta.wpfloat]


def _init_balanced_base_state(
    *,
    config: TracerBlobConfig,
    vertical_config: v_grid.VerticalGridConfig,
    grid: icon_grid.IconGrid,
    static_fields: static_fields.StaticFieldFactories,
    prognostic_state_now: prognostics.PrognosticState,
    allocator: gtx_typing.Allocator,
    array_ns: types.ModuleType,
    exchange: decomposition_defs.ExchangeRuntime,
) -> None:
    """Base state as in gauss3d: constant zonal wind u0, Brunt-Vaisala theta_v
    profile, hydrostatic balance and the induced w."""
    geometry = static_fields.geometry
    metrics = static_fields.metrics
    primal_normal_x = geometry.get(geometry_meta.EDGE_NORMAL_U).ndarray
    inv_dual_edge_length = geometry.get(f"inverse_of_{geometry_meta.DUAL_EDGE_LENGTH}").ndarray
    edge_cell_distance = geometry.get(geometry_meta.EDGE_CELL_DISTANCE).ndarray
    primal_edge_length = geometry.get(geometry_meta.EDGE_LENGTH).ndarray
    cell_area = geometry.get(geometry_meta.CELL_AREA).ndarray
    geopot = phy_const.GRAV * metrics.get(metrics_attributes.Z_MC).ndarray
    z_ifc = metrics.get(metrics_attributes.CELL_HEIGHT_ON_HALF_LEVEL).ndarray

    num_levels = grid.num_levels

    t0 = config.t0
    brunt_vais = config.brunt_vais

    exner_ndarray = prognostic_state_now.exner.ndarray
    rho_ndarray = prognostic_state_now.rho.ndarray
    theta_v_ndarray = prognostic_state_now.theta_v.ndarray
    vn_ndarray = prognostic_state_now.vn.ndarray

    # constant zonal wind on all edges: the periodic torus has no lateral boundary
    vn_ndarray[:, :] = config.u0 * primal_normal_x[:, array_ns.newaxis]

    for k_index in range(num_levels - 1, -1, -1):
        z_help = (brunt_vais / phy_const.GRAV) ** 2 * geopot[:, k_index]
        theta_v_ndarray[:, k_index] = t0 * array_ns.exp(z_help)

    if brunt_vais != 0.0:
        z_help = (brunt_vais / phy_const.GRAV) ** 2 * geopot[:, num_levels - 1]
        exner_ndarray[:, num_levels - 1] = (
            phy_const.GRAV / brunt_vais
        ) ** 2 / t0 / phy_const.CPD * (array_ns.exp(-z_help) - 1.0) + 1.0
    else:
        exner_ndarray[:, num_levels - 1] = 1.0 - geopot[:, num_levels - 1] / phy_const.CPD / t0

    testcases_utils.hydrostatic_adjustment_constant_thetav_ndarray(
        wgtfac_c=metrics.get(metrics_attributes.WGTFAC_C).ndarray,
        ddqz_z_half=metrics.get(metrics_attributes.DDQZ_Z_HALF).ndarray,
        exner_ref_mc=metrics.get(metrics_attributes.EXNER_REF_MC).ndarray,
        d_exner_dz_ref_ic=metrics.get(metrics_attributes.D_EXNER_DZ_REF_IC).ndarray,
        theta_ref_mc=metrics.get(metrics_attributes.THETA_REF_MC).ndarray,
        theta_ref_ic=metrics.get(metrics_attributes.THETA_REF_IC).ndarray,
        rho=rho_ndarray,
        exner=exner_ndarray,
        theta_v=theta_v_ndarray,
        num_levels=num_levels,
    )
    log.info("Hydrostatic adjustment (constant theta_v) computation completed.")

    _, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, allocator)

    prognostic_state_now.w.ndarray[:, :] = testcases_utils.init_w(
        grid=grid,
        z_ifc=z_ifc,
        inv_dual_edge_length=inv_dual_edge_length,
        edge_cell_distance=edge_cell_distance,
        primal_edge_length=primal_edge_length,
        cell_area=cell_area,
        vn=vn_ndarray,
        vct_b=vct_b.ndarray,
        nlev=num_levels,
    )
    exchange.exchange(dims.CellDim, prognostic_state_now.w)


def tracer_blob(
    *,
    config: TracerBlobConfig,
    vertical_config: v_grid.VerticalGridConfig,
    grid: icon_grid.IconGrid,
    static_fields: static_fields.StaticFieldFactories,
    prognostic_state_now: prognostics.PrognosticState,
    tracer_state_now: tracer_states.TracerState,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
    prescription: TracerAdvectionPrescription,
) -> None:
    """
    Tracer-advection-only initial condition: constant zonal wind u0 in a
    hydrostatically balanced base state, plus a constant circular tracer disc
    (qv) that the prescribed mass fluxes advect around the periodic torus.
    """
    if grid.grid_params.geometry_type != icon_grid.GeometryType.TORUS:
        raise NotImplementedError(
            "The 'tracer_blob' initial condition is only implemented on a torus grid."
        )
    domain_length = grid.grid_params.domain_length
    domain_height = grid.grid_params.domain_height
    assert domain_length is not None and domain_height is not None

    qv = tracer_state_now.qv
    if qv is None:
        raise ValueError(
            "The 'tracer_blob' initial condition requires an active qv tracer (ntracer >= 1)."
        )

    allocator = model_backends.get_allocator(backend)
    array_ns = data_alloc.import_array_ns(allocator)

    _init_balanced_base_state(
        config=config,
        vertical_config=vertical_config,
        grid=grid,
        static_fields=static_fields,
        prognostic_state_now=prognostic_state_now,
        allocator=allocator,
        array_ns=array_ns,
        exchange=exchange,
    )

    geometry = static_fields.geometry
    metrics = static_fields.metrics
    cell_center_x = geometry.get(geometry_meta.CELL_CENTER_X).ndarray
    cell_center_y = geometry.get(geometry_meta.CELL_CENTER_Y).ndarray
    ddqz_z_full = metrics.get(metrics_attributes.DDQZ_Z_FULL).ndarray
    ddqz_z_full_e = metrics.get(metrics_attributes.DDQZ_Z_FULL_E).ndarray
    rho_ndarray = prognostic_state_now.rho.ndarray
    vn_ndarray = prognostic_state_now.vn.ndarray

    # constant circular blob: uniform amplitude inside the torus-periodic disc,
    # zero outside, constant in z
    blob_x = config.blob_x if config.blob_x is not None else 0.5 * domain_length
    blob_y = config.blob_y if config.blob_y is not None else 0.5 * domain_height
    blob_radius = (
        config.blob_radius
        if config.blob_radius is not None
        else 0.25 * min(domain_length, domain_height)
    )
    # minimal-image offsets (same closest-image logic as plane_torus_closest_coordinates)
    dx = (cell_center_x - blob_x + 0.5 * domain_length) % domain_length - 0.5 * domain_length
    dy = (cell_center_y - blob_y + 0.5 * domain_height) % domain_height - 0.5 * domain_height
    qv.ndarray[dx**2 + dy**2 <= blob_radius**2, :] = config.blob_amplitude

    # prescribe the advection driving fields the (disabled) dycore would provide
    e2c = grid.get_connectivity(dims.E2C).ndarray
    rho_at_edge = 0.5 * (rho_ndarray[e2c[:, 0], :] + rho_ndarray[e2c[:, 1], :])
    prescription.vn_traj.ndarray[:, :] = vn_ndarray
    prescription.mass_flx_me.ndarray[:, :] = rho_at_edge * vn_ndarray * ddqz_z_full_e
    prescription.mass_flx_ic.ndarray[:, :] = 0.0
    airmass = rho_ndarray * ddqz_z_full
    prescription.airmass_now.ndarray[:, :] = airmass
    prescription.airmass_new.ndarray[:, :] = airmass
