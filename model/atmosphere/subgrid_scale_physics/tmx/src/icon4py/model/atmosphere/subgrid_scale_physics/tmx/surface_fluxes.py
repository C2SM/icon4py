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

from gt4py.next import common as gtx_common

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils import (
    surface_fluxes as surface_flux_stencils,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.model_options import setup_program


if typing.TYPE_CHECKING:
    from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states
    from icon4py.model.common import field_type_aliases as fa, model_backends, type_alias as ta
    from icon4py.model.common.grid import base as base_grid


class SurfaceFluxProvider(Protocol):
    """Fills the surface-flux input of tmx, once per step before `Tmx.run`."""

    def compute(self, *, out: tmx_states.TmxSurfaceFluxState) -> None:
        """Write every field of `out`."""
        ...


class ZeroFluxProvider:
    """Zero surface fluxes."""

    def compute(self, *, out: tmx_states.TmxSurfaceFluxState) -> None:
        for field in dataclasses.fields(out):
            getattr(out, field.name).ndarray[...] = 0.0


class PrescribedFluxProvider:
    """
    Surface fluxes from fixed kinematic heat fluxes (`isrfc_type = 1` in ICON's
    `nh_testcase_nml`), as in `compute_sfc_fluxes` of `mo_tmx_surface.f90`.

    The heat fluxes are the kinematic fluxes `shflx` [K m/s] and `lhflx` [m/s] times the
    surface air density; the wind stresses and `q_snocpymlt` are zero. `pressure_ifc` and
    `surface_temperature` are read on every `compute`, so they must be the buffers the caller
    updates.
    """

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        backend: model_backends.BackendLike,
        pressure_ifc: fa.CellKHalfField[ta.wpfloat],
        surface_temperature: fa.CellField[ta.wpfloat],
        surface_type: int,
        shflx: float,
        lhflx: float,
    ) -> None:
        if surface_type != 1:
            raise NotImplementedError(
                "PrescribedFluxProvider implements fixed kinematic heat fluxes (isrfc_type = 1), "
                f"got isrfc_type = {surface_type}."
            )
        self._surface_temperature = surface_temperature
        # a view of the lowest half level, so that it follows the caller's updates
        self._surface_pressure = gtx_common._field(
            pressure_ifc.ndarray[:, grid.num_levels],
            domain={dims.CellDim: (0, grid.num_cells)},
        )
        cell_domain = h_grid.domain(dims.CellDim)
        self._compute_prescribed_surface_fluxes = setup_program(
            backend=backend,
            program=surface_flux_stencils.compute_prescribed_surface_fluxes,
            constant_args={"shflx": shflx, "lhflx": lhflx},
            horizontal_sizes={
                "horizontal_start": grid.start_index(cell_domain(h_grid.Zone.LOCAL)),
                "horizontal_end": grid.end_index(cell_domain(h_grid.Zone.END)),
            },
            offset_provider={},
        )

    def compute(self, *, out: tmx_states.TmxSurfaceFluxState) -> None:
        self._compute_prescribed_surface_fluxes(
            surface_temperature=self._surface_temperature,
            surface_pressure=self._surface_pressure,
            sensible_heat_flux=out.sensible_heat_flux,
            evapotranspiration=out.evapotranspiration,
        )
        out.u_stress.ndarray[...] = 0.0
        out.v_stress.ndarray[...] = 0.0
        out.q_snocpymlt.ndarray[...] = 0.0
