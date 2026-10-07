# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Providers of the surface fluxes that tmx takes as input."""

from __future__ import annotations

import dataclasses
import typing
from typing import Protocol

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils import (
    surface_fluxes as surface_flux_stencils,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.model_options import setup_program


if typing.TYPE_CHECKING:
    from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states
    from icon4py.model.common import field_type_aliases as fa, model_backends
    from icon4py.model.common.grid import base as base_grid


class SurfaceFluxProvider(Protocol):
    """Fills the surface-flux input of tmx, once per step before `Tmx.run`."""

    def compute(
        self,
        *,
        pressure_ifc: fa.CellKHalfField[ta.wpfloat],
        out: tmx_states.TmxSurfaceFluxState,
    ) -> None:
        """Write every field of `out` from the step's half-level pressure."""
        ...


class ZeroFluxProvider:
    """Zero surface fluxes."""

    def compute(
        self,
        *,
        pressure_ifc: fa.CellKHalfField[ta.wpfloat],
        out: tmx_states.TmxSurfaceFluxState,
    ) -> None:
        for field in dataclasses.fields(out):
            getattr(out, field.name).ndarray[...] = 0.0


class PrescribedFluxProvider:
    """
    Surface fluxes from fixed kinematic heat fluxes (`isrfc_type = 1` in ICON's
    `nh_testcase_nml`), as in `compute_sfc_fluxes` of `mo_tmx_surface.f90`.

    The heat fluxes are the kinematic fluxes `shflx` [K m/s] and `lhflx` [m/s] times the
    surface air density, which follows from the surface pressure of the step and the
    surface temperature; the wind stresses and `q_snocpymlt` are zero. The surface
    temperature is held fixed, as ICON's ocean tile does without a mixed-layer ocean
    (`mo_vdf_sfc.f90` zeroes its tendency).
    """

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        backend: model_backends.BackendLike,
        surface_temperature: fa.CellField[ta.wpfloat],
        shflx: float,
        lhflx: float,
    ) -> None:
        self._surface_level = dims.KHalfDim(grid.num_levels)
        self._surface_temperature = surface_temperature
        cell_domain = h_grid.domain(dims.CellDim)
        self._compute_prescribed_surface_fluxes = setup_program(
            backend=backend,
            program=surface_flux_stencils.compute_prescribed_surface_fluxes,
            constant_args={"shflx": ta.wpfloat(shflx), "lhflx": ta.wpfloat(lhflx)},
            horizontal_sizes={
                "horizontal_start": grid.start_index(cell_domain(h_grid.Zone.LOCAL)),
                "horizontal_end": grid.end_index(cell_domain(h_grid.Zone.END)),
            },
            offset_provider={},
        )

    def compute(
        self,
        *,
        pressure_ifc: fa.CellKHalfField[ta.wpfloat],
        out: tmx_states.TmxSurfaceFluxState,
    ) -> None:
        self._compute_prescribed_surface_fluxes(
            surface_temperature=self._surface_temperature,
            # a view of the lowest half level
            surface_pressure=pressure_ifc[self._surface_level],
            sensible_heat_flux=out.sensible_heat_flux,
            evapotranspiration=out.evapotranspiration,
        )
        out.u_stress.ndarray[...] = 0.0
        out.v_stress.ndarray[...] = 0.0
        out.q_snocpymlt.ndarray[...] = 0.0
