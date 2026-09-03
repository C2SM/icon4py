# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Hydrostatic pressure, and the saturation vapour pressures.

The surface pressure is extrapolated from the lowest three levels, then the
pressure is obtained by vertical integration of the virtual temperature;
``compute_surface_and_hydrostatic_pressure`` sequences the two stencils the way
ICON's ``diagnose_pres_temp`` does.

This is the *hydrostatic* pressure, deliberately not the Exner-function shortcut
``P0REF * exner ** CPD_O_RD``. The moist initial conditions rely on the
hydrostatic value because ICON re-diagnoses pressure via ``diagnose_pres_temp``
when initializing the water vapour (``init_nh_inwp_tracers`` with
``l_rediag=.TRUE.``), so the converged/serialized state matches this integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
from gt4py.next import exp, log, sqrt
from gt4py.next.experimental import concat_where

from icon4py.model.common import (
    constants as phy_const,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc


@gtx.field_operator
def _compute_surface_pressure(
    exner: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    ddqz_z_full: fa.CellKField[ta.wpfloat],
) -> fa.CellKHalfField[ta.wpfloat]:
    surface_pressure = PhysicsConstants.p0ref * exp(
        PhysicsConstants.cpd_o_rd * log(exner(dims.KHalfDim - 2.5))
        + PhysicsConstants.grav_o_rd
        * (
            ddqz_z_full(dims.KHalfDim - 0.5) / virtual_temperature(dims.KHalfDim - 0.5)
            + ddqz_z_full(dims.KHalfDim - 1.5) / virtual_temperature(dims.KHalfDim - 1.5)
            + 0.5 * ddqz_z_full(dims.KHalfDim - 2.5) / virtual_temperature(dims.KHalfDim - 2.5)
        )
    )
    return surface_pressure


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_pressure(  # noqa: PLR0917 [too-many-positional-arguments]
    exner: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    ddqz_z_full: fa.CellKField[ta.wpfloat],
    surface_pressure: fa.CellKHalfField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_surface_pressure(
        exner=exner,
        virtual_temperature=virtual_temperature,
        ddqz_z_full=ddqz_z_full,
        out=surface_pressure,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )


@gtx.scan_operator(axis=dims.KDim, forward=False, init=(0.0, 0.0, True))
def _scan_pressure(
    state: tuple[ta.wpfloat, ta.wpfloat, bool],
    ddqz_z_full: ta.wpfloat,
    virtual_temperature: ta.wpfloat,
    surface_pressure: ta.wpfloat,
):
    pressure_interface = (
        surface_pressure * exp(-PhysicsConstants.grav_o_rd * ddqz_z_full / virtual_temperature)
        if state[2]
        else state[1] * exp(-PhysicsConstants.grav_o_rd * ddqz_z_full / virtual_temperature)
    )
    pressure = (
        sqrt(surface_pressure * pressure_interface)
        if state[2]
        else sqrt(state[1] * pressure_interface)
    )
    return pressure, pressure_interface, False


@gtx.field_operator
def _compute_hydrostatic_pressure_on_model_levels(
    ddqz_z_full: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    surface_pressure: gtx.Field[gtx.Dims[dims.CellDim], ta.wpfloat],
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    """
    Update pressure by assuming hydrostatic balance (dp/dz = -rho g = p g / Rd / Tv).
    Note that virtual temperature is used in the equation.

    Args:
        ddqz_z_full: vertical grid spacing at full levels [m]
        virtual_temperature: air virtual temperature [K]
        surface_pressure: surface air pressure [Pa]
    Returns:
        pressure at full levels, pressure at the bounding upper interface of each level
    """
    pressure, pressure_ifc, _ = _scan_pressure(ddqz_z_full, virtual_temperature, surface_pressure)
    return pressure, pressure_ifc


@gtx.field_operator
def _pressure_on_half_levels(
    pressure_ifc_on_model_levels: fa.CellKField[ta.wpfloat],
    surface_pressure: gtx.Field[gtx.Dims[dims.CellDim], ta.wpfloat],
    nlev: gtx.int32,
) -> fa.CellKHalfField[ta.wpfloat]:
    # TODO(havogt): The range of the scan is deduced from the (unique) domain,
    # therefore multiple output domains are currently not possible with scans.
    return concat_where(
        dims.KHalfDim == nlev,
        surface_pressure,
        pressure_ifc_on_model_levels(dims.KHalfDim + 0.5),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_hydrostatic_pressure(  # noqa: PLR0917 [too-many-positional-arguments]
    ddqz_z_full: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    surface_pressure: fa.CellField[ta.wpfloat],
    pressure: fa.CellKField[ta.wpfloat],
    pressure_ifc_on_model_levels: fa.CellKField[ta.wpfloat],
    pressure_ifc: fa.CellKHalfField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_hydrostatic_pressure_on_model_levels(
        ddqz_z_full,
        virtual_temperature,
        surface_pressure,
        out=(pressure, pressure_ifc_on_model_levels),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _pressure_on_half_levels(
        pressure_ifc_on_model_levels,
        surface_pressure,
        vertical_end,
        out=pressure_ifc,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_start, vertical_end + 1),
        },
    )


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as grid_base


def compute_surface_and_hydrostatic_pressure(
    *,
    grid: grid_base.Grid,
    backend: gtx_typing.Backend | None,
    exner: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    ddqz_z_full: fa.CellKField[ta.wpfloat],
    surface_pressure: fa.CellField[ta.wpfloat],
    pressure: fa.CellKField[ta.wpfloat],
    pressure_on_cells_half_levels: fa.CellKHalfField[ta.wpfloat],
) -> None:
    """Diagnose the hydrostatic pressure into caller-provided buffers.

    Args:
        grid, backend: grid and gt4py backend.
        exner, virtual_temperature, ddqz_z_full: input cell-K fields.
        surface_pressure: cell field receiving the surface pressure.
        pressure: cell-K field receiving the full-level pressure.
        pressure_on_cells_half_levels: half-level (``nlev+1``) output buffer for
            pressure on cell half-levels; also receives the diagnosed surface pressure.
    """
    num_levels = grid.num_levels
    cell_domain = h_grid.domain(dims.CellDim)
    horizontal_end = grid.end_index(cell_domain(h_grid.Zone.END))

    compute_surface_pressure.with_backend(backend)(
        exner=exner,
        virtual_temperature=virtual_temperature,
        ddqz_z_full=ddqz_z_full,
        surface_pressure=pressure_on_cells_half_levels,
        horizontal_start=0,
        horizontal_end=horizontal_end,
        vertical_start=num_levels,
        vertical_end=num_levels + 1,
        offset_provider={},
    )
    # surface pressure lives at the bottom interface; extract it as a cell field
    surface_pressure.ndarray[:] = pressure_on_cells_half_levels.ndarray[:, num_levels]
    pressure_on_cells_half_levels.ndarray[:, -1] = surface_pressure.ndarray

    compute_hydrostatic_pressure.with_backend(backend)(
        ddqz_z_full=ddqz_z_full,
        virtual_temperature=virtual_temperature,
        surface_pressure=surface_pressure,
        pressure=pressure,
        pressure_ifc_on_model_levels=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=backend, dtype=ta.wpfloat
        ),
        pressure_ifc=pressure_on_cells_half_levels,
        horizontal_start=0,
        horizontal_end=horizontal_end,
        vertical_start=0,
        vertical_end=num_levels,
        offset_provider={},
    )


def compute_surface_and_hydrostatic_pressure_ndarray(
    *,
    grid: grid_base.Grid,
    backend: gtx_typing.Backend | None,
    allocator: gtx_typing.Allocator,
    exner: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    ddqz_z_full: fa.CellKField[ta.wpfloat],
) -> data_alloc.NDArray:
    """Allocate work buffers, diagnose the hydrostatic pressure, return its ndarray.

    Convenience wrapper around :func:`compute_surface_and_hydrostatic_pressure` for one-shot callers
    (e.g. initial-condition setup) that do not keep their own buffers.
    """
    surface_pressure = data_alloc.zero_field(
        grid, dims.CellDim, allocator=allocator, dtype=ta.wpfloat
    )
    pressure = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
    )
    pressure_on_cells_half_levels = data_alloc.zero_field(
        grid, dims.CellDim, dims.KHalfDim, allocator=allocator, dtype=ta.wpfloat
    )
    compute_surface_and_hydrostatic_pressure(
        grid=grid,
        backend=backend,
        exner=exner,
        virtual_temperature=virtual_temperature,
        ddqz_z_full=ddqz_z_full,
        surface_pressure=surface_pressure,
        pressure=pressure,
        pressure_on_cells_half_levels=pressure_on_cells_half_levels,
    )
    return pressure.ndarray


def sat_pres_water(temperature: data_alloc.NDArray) -> data_alloc.NDArray:
    """Saturation vapour pressure over liquid water [Pa] (Tetens formula)."""
    array_ns = data_alloc.array_namespace(temperature)
    return phy_const.TETENS_P0 * array_ns.exp(
        phy_const.TETENS_A_WATER
        * (temperature - phy_const.MELTING_TEMPERATURE)
        / (temperature - phy_const.TETENS_B_WATER)
    )


def sat_pres_ice(temperature: data_alloc.NDArray) -> data_alloc.NDArray:
    """Saturation vapour pressure over ice [Pa] (Tetens formula)."""
    array_ns = data_alloc.array_namespace(temperature)
    return phy_const.TETENS_P0 * array_ns.exp(
        phy_const.TETENS_A_ICE
        * (temperature - phy_const.MELTING_TEMPERATURE)
        / (temperature - phy_const.TETENS_B_ICE)
    )
