# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The energy update component of tmx.

`Update_energy_tendencies` in ICON's `mo_vdf.f90`, followed by the atmospheric part of
`Update_diagnostics` in `mo_vdf.f90` and `mo_vdf_atmo.f90`.
"""

from __future__ import annotations

import functools
import logging
import typing

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils import (
    energy_update as energy_stencils,
)
from icon4py.model.common import constants, dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import base as base_grid, horizontal as h_grid
from icon4py.model.common.math import vertical_operations
from icon4py.model.common.model_options import setup_program
from icon4py.model.common.utils import data_allocation as data_alloc


if typing.TYPE_CHECKING:
    from icon4py.model.common import field_type_aliases as fa, type_alias as ta


log = logging.getLogger(__name__)


class EnergyUpdate:
    """The dissipation heating and the end-of-step diagnostics of tmx."""

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        metric_state: tmx_states.TmxMetricState,
        backend: model_backends.BackendLike,
        exchange: decomposition.ExchangeRuntime,
        dissipation_factor: float,
        turb_prandtl: float,
        use_km_const: bool,
        km_const: float,
    ) -> None:
        self._exchange = exchange

        num_levels = grid.num_levels
        cell_domain = h_grid.domain(dims.CellDim)
        self._cell_start = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._cell_end = grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._surface_level = num_levels - 1

        zero_field = functools.partial(
            data_alloc.zero_field, grid, allocator=model_backends.get_allocator(backend)
        )
        # running sums from the top; `run` copies their last level to the 2D diagnostics on the
        # array side, since gt4py cannot write one K level into a field without K
        self._cptgz_vi: fa.CellKField[ta.wpfloat] = zero_field(dims.CellDim, dims.KDim)
        self._dissip_ke_vi: fa.CellKField[ta.wpfloat] = zero_field(dims.CellDim, dims.KDim)
        self._int_energy_vi: fa.CellKField[ta.wpfloat] = zero_field(dims.CellDim, dims.KDim)
        self._tend_int_energy_vi: fa.CellKField[ta.wpfloat] = zero_field(dims.CellDim, dims.KDim)

        horizontal_sizes = {
            "horizontal_start": self._cell_start,
            "horizontal_end": self._cell_end,
        }
        self._update_temperature_and_compute_end_of_step_diagnostics = setup_program(
            backend=backend,
            program=energy_stencils.update_temperature_and_compute_end_of_step_diagnostics,
            constant_args={
                "ddqz_z_full": metric_state.ddqz_z_full,
                "height_above_ground": metric_state.height_above_ground,
                "dissipation_factor": dissipation_factor,
                "grav": constants.GRAV,
            },
            horizontal_sizes=horizontal_sizes,
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        self._surface_exchange_coefficients: tuple[float, float] | None = None
        if use_km_const:
            self._surface_exchange_coefficients = (km_const, km_const * (1.0 / turb_prandtl))
            self._set_constant_on_surface_level = setup_program(
                backend=backend,
                program=vertical_operations.set_constant_on_model_levels_on_cells_wp,
                horizontal_sizes=horizontal_sizes,
                vertical_sizes={
                    "vertical_start": gtx.int32(self._surface_level),
                    "vertical_end": gtx.int32(num_levels),
                },
                offset_provider={},
            )

    def run(
        self,
        *,
        input_state: tmx_states.TmxInputState,
        surface_flux_state: tmx_states.TmxSurfaceFluxState,
        diagnostic_state: tmx_states.TmxDiagnosticState,
        tendency_state: tmx_states.TmxTendencyState,
        new_state: tmx_states.TmxNewState,
        dtime: float,
    ) -> None:
        """
        Add the dissipation heating to `tendency_state.tend_temperature`, write the updated
        temperature to `new_state` and the end-of-step diagnostics to `diagnostic_state`.

        Runs after the scalar and the wind diffusion: needs their tendencies and new states,
        and `km_ic` and `kh_ic` of `diagnostic_state`. Above the lowest model level, `km` and
        `kh` are copied from the half level below. At the lowest level, `use_km_const=True` sets
        `km = km_const` and `kh = km_const / turb_prandtl`; otherwise the caller must fill that
        level with the surface exchange coefficient.
        """
        log.debug("tmx energy update: start")

        self._update_temperature_and_compute_end_of_step_diagnostics(
            u=input_state.u,
            v=input_state.v,
            new_u=new_state.u,
            new_v=new_state.v,
            air_mass=input_state.air_mass,
            cv_air=input_state.cv_air,
            temperature=input_state.temperature,
            tend_temperature=tendency_state.tend_temperature,
            q_snocpymlt=surface_flux_state.q_snocpymlt,
            qv=input_state.qv,
            qc=input_state.qc,
            qi=input_state.qi,
            new_qv=new_state.qv,
            new_qc=new_state.qc,
            new_qi=new_state.qi,
            qr=input_state.qr,
            qs=input_state.qs,
            qg=input_state.qg,
            rho=input_state.rho,
            km_ic=diagnostic_state.km_ic,
            kh_ic=diagnostic_state.kh_ic,
            dissip_ke=diagnostic_state.dissip_ke,
            heating=diagnostic_state.heating,
            new_temperature=new_state.temperature,
            cptgz=diagnostic_state.cptgz,
            cptgz_vi=self._cptgz_vi,
            dissip_ke_vi=self._dissip_ke_vi,
            int_energy_vi=self._int_energy_vi,
            tend_int_energy_vi=self._tend_int_energy_vi,
            km=diagnostic_state.km,
            kh=diagnostic_state.kh,
            dtime=dtime,
        )
        temperature_exchange = self._exchange.start(
            dims.CellDim, new_state.temperature, tendency_state.tend_temperature
        )
        if self._surface_exchange_coefficients is not None:
            km_surface, kh_surface = self._surface_exchange_coefficients
            self._set_constant_on_surface_level(field=diagnostic_state.km, value=km_surface)
            self._set_constant_on_surface_level(field=diagnostic_state.kh, value=kh_surface)
        cells = slice(self._cell_start, self._cell_end)
        for running_sum, column_sum in (
            (self._cptgz_vi, diagnostic_state.cptgz_vi),
            (self._dissip_ke_vi, diagnostic_state.dissip_ke_vi),
            (self._int_energy_vi, diagnostic_state.int_energy_vi),
            (self._tend_int_energy_vi, diagnostic_state.tend_int_energy_vi),
        ):
            column_sum.ndarray[cells] = running_sum.ndarray[cells, self._surface_level]  # type: ignore[index]  # GT4Py NDArrayObject protocol limitation
        temperature_exchange.finish()

        log.debug("tmx energy update: end")
