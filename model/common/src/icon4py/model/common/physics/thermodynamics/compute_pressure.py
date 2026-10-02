# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Hydrostatic pressure, and the saturation vapour pressures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
from gt4py.next import exp, log, sqrt

from icon4py.model.common import (
    constants as phy_const,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.math.vertical_operations import (
    _copy_model_level_below_to_half_levels_on_cells,
)
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as grid_base


@gtx.field_operator
def _compute_surface_pressure(
    exner: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    ddqz_z_full: fa.CellKField[ta.wpfloat],
) -> fa.CellKHalfField[ta.wpfloat]:
    """Extrapolate the surface pressure from the lowest three model levels."""
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
    pressure_ifc: fa.CellKHalfField[ta.wpfloat],
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    """
    Compute the hydrostatic pressure from the hydrostatic balance equation
    dp/dz = -rho g = -p g / (Rd Tv). This differs from the total pressure derived
    from the Exner function, ``P0REF * exner ** CPD_O_RD``, because the latter also
    contains the non-hydrostatic component.
    Note that virtual temperature is used in the equation.

    Args:
        ddqz_z_full: vertical grid spacing at full levels [m]
        virtual_temperature: air virtual temperature [K]
        pressure_ifc: pressure at half levels, of which only the surface row is used [Pa]
    Returns:
        pressure at full levels, pressure at the bounding upper interface of each level
    """
    pressure, pressure_ifc_on_model_levels, _ = _scan_pressure(
        ddqz_z_full, virtual_temperature, pressure_ifc(dims.KDim + 0.5)
    )
    return pressure, pressure_ifc_on_model_levels


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_and_hydrostatic_pressure(  # noqa: PLR0917 [too-many-positional-arguments]
    exner: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    ddqz_z_full: fa.CellKField[ta.wpfloat],
    pressure: fa.CellKField[ta.wpfloat],
    pressure_ifc_on_model_levels: fa.CellKField[ta.wpfloat],
    pressure_ifc: fa.CellKHalfField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    # The surface pressure goes into the bottom row of pressure_ifc first: the scan's
    # first iteration reads it from there, and the later calls must not overwrite it.
    _compute_surface_pressure(
        exner=exner,
        virtual_temperature=virtual_temperature,
        ddqz_z_full=ddqz_z_full,
        out=pressure_ifc,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_end, vertical_end + 1),
        },
    )
    _compute_hydrostatic_pressure_on_model_levels(
        ddqz_z_full,
        virtual_temperature,
        pressure_ifc,
        out=(pressure, pressure_ifc_on_model_levels),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    # TODO(havogt): The range of the scan is deduced from the (unique) domain,
    # therefore multiple output domains are currently not possible with scans.
    _copy_model_level_below_to_half_levels_on_cells(
        pressure_ifc_on_model_levels,
        out=pressure_ifc,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
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
    """Allocate work buffers, diagnose the pressure, return its ndarray.

    For one-shot callers (e.g. initial-condition setup) that keep no buffers of their own.
    """
    pressure = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
    )
    cell_domain = h_grid.domain(dims.CellDim)
    compute_surface_and_hydrostatic_pressure.with_backend(backend)(
        exner=exner,
        virtual_temperature=virtual_temperature,
        ddqz_z_full=ddqz_z_full,
        pressure=pressure,
        pressure_ifc_on_model_levels=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
        ),
        pressure_ifc=data_alloc.zero_field(
            grid, dims.CellDim, dims.KHalfDim, allocator=allocator, dtype=ta.wpfloat
        ),
        horizontal_start=0,
        horizontal_end=grid.end_index(cell_domain(h_grid.Zone.END)),
        vertical_start=0,
        vertical_end=grid.num_levels,
        offset_provider={},
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


@gtx.field_operator
def sat_pres_water_on_cells(temperature: fa.CellField[ta.wpfloat]) -> fa.CellField[ta.wpfloat]:
    """Saturation vapour pressure over liquid water [Pa] (Tetens formula), on a surface field."""
    return PhysicsConstants.tetens_p0 * exp(
        PhysicsConstants.tetens_a_water
        * (temperature - PhysicsConstants.tmelt)
        / (temperature - PhysicsConstants.tetens_b_water)
    )
