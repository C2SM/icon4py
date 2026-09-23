# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The scalar diffusion component of tmx.

Port of ``Compute_diffusion_hydrometeors`` and ``Compute_diffusion_temperature`` in ICON's
``mo_vdf.f90``, with the implicit vertical solver.
"""

from __future__ import annotations

import functools
import logging
import typing

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import config as tmx_config, tmx_states
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils import (
    scalar_diffusion as scalar_stencils,
)
from icon4py.model.common import constants, dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import base as base_grid, horizontal as h_grid
from icon4py.model.common.model_options import setup_program
from icon4py.model.common.utils import data_allocation as data_alloc


if typing.TYPE_CHECKING:
    import icon4py.model.common.grid.states as grid_states
    from icon4py.model.common import field_type_aliases as fa, type_alias as ta


log = logging.getLogger(__name__)


class ScalarDiffusion:
    """The scalar (qv, qc, qi and temperature) diffusion stage of tmx."""

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        metric_state: tmx_states.TmxMetricState,
        interpolation_state: tmx_states.TmxInterpolationState,
        edge_params: grid_states.EdgeParams,
        backend: model_backends.BackendLike,
        exchange: decomposition.ExchangeRuntime,
        turb_prandtl: float,
        energy_type: tmx_config.EnergyType,
        use_scale_turb_energy_flux: bool,
        scale_turb_energy_flux: float,
    ) -> None:
        self._exchange = exchange
        # ``zfactor`` in Compute_diffusion_temperature
        energy_flux_factor = scale_turb_energy_flux if use_scale_turb_energy_flux else 1.0
        use_internal_energy = energy_type == tmx_config.EnergyType.INTERNAL

        cell_domain = h_grid.domain(dims.CellDim)
        cell_start_nudging = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        cell_end_local = grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        zero_field = functools.partial(
            data_alloc.zero_field, grid, allocator=model_backends.get_allocator(backend)
        )
        # the matrix is assembled once for the three tracers and once for the energy
        self._matrix_a: fa.CellKField[ta.wpfloat] = zero_field(dims.CellDim, dims.KDim)
        self._matrix_b: fa.CellKField[ta.wpfloat] = zero_field(dims.CellDim, dims.KDim)
        self._matrix_c: fa.CellKField[ta.wpfloat] = zero_field(dims.CellDim, dims.KDim)
        self._zero_surface_flux: fa.CellField[ta.wpfloat] = zero_field(dims.CellDim)
        self.energy: fa.CellKField[ta.wpfloat] = zero_field(dims.CellDim, dims.KDim)

        horizontal_sizes = {
            "horizontal_start": cell_start_nudging,
            "horizontal_end": cell_end_local,
        }
        vertical_sizes = {
            "vertical_start": gtx.int32(0),
            "vertical_end": gtx.int32(grid.num_levels),
        }
        assemble_matrix = functools.partial(
            setup_program,
            backend=backend,
            program=scalar_stencils.assemble_scalar_diffusion_matrix,
            horizontal_sizes=horizontal_sizes,
            vertical_sizes=vertical_sizes,
            offset_provider={},
        )
        self.assemble_tracer_diffusion_matrix = assemble_matrix(
            constant_args={"inv_dz": metric_state.inv_ddqz_z_half, "prefactor": 1.0}
        )
        self.assemble_energy_diffusion_matrix = assemble_matrix(
            constant_args={
                "inv_dz": metric_state.inv_ddqz_z_half,
                "prefactor": energy_flux_factor,
            }
        )
        horizontal_diffusion_args = {
            "inv_dual_edge_length": edge_params.inverse_dual_edge_lengths,
            "geofac_div": interpolation_state.geofac_div,
            "rturb_prandtl": 1.0 / turb_prandtl,
        }
        self.diffuse_tracer = setup_program(
            backend=backend,
            program=scalar_stencils.diffuse_tracer,
            constant_args={**horizontal_diffusion_args, "prefactor": 1.0},
            horizontal_sizes=horizontal_sizes,
            vertical_sizes=vertical_sizes,
            offset_provider=grid.connectivities,
        )
        self.compute_energy_from_temperature = setup_program(
            backend=backend,
            program=scalar_stencils.compute_energy_from_temperature,
            constant_args={
                "height_above_ground": metric_state.height_above_ground,
                "grav": constants.GRAV,
                "use_internal_energy": use_internal_energy,
            },
            horizontal_sizes=horizontal_sizes,
            vertical_sizes=vertical_sizes,
            offset_provider={},
        )
        self.diffuse_energy_and_update_temperature = setup_program(
            backend=backend,
            program=scalar_stencils.diffuse_energy_and_update_temperature,
            constant_args={
                **horizontal_diffusion_args,
                "height_above_ground": metric_state.height_above_ground,
                "prefactor": energy_flux_factor,
                "grav": constants.GRAV,
                "use_internal_energy": use_internal_energy,
            },
            horizontal_sizes=horizontal_sizes,
            vertical_sizes=vertical_sizes,
            offset_provider=grid.connectivities,
        )

    def run_hydrometeor_diffusion(
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
        Diffuse qv, qc and qi (``Compute_diffusion_hydrometeors`` in mo_vdf.f90, without CO2).

        Only qv has a surface flux, the evapotranspiration. Needs ``kh_ic`` and ``km_ie`` of
        ``diagnostic_state``.
        """
        log.debug("tmx hydrometeor diffusion (Compute_diffusion_hydrometeors): start")

        log.debug("communication of qv, qc, qi (cells): start")
        tracer_exchange = self._exchange.start(
            dims.CellDim, input_state.qv, input_state.qc, input_state.qi
        )

        self.assemble_tracer_diffusion_matrix(
            diffusivity=diagnostic_state.kh_ic,
            air_mass=input_state.air_mass,
            a=self._matrix_a,
            b=self._matrix_b,
            c=self._matrix_c,
        )

        tracer_exchange.finish()
        log.debug("communication of qv, qc, qi (cells): end")

        for var, surface_flux, new_var, tend in (
            (
                input_state.qv,
                surface_flux_state.evapotranspiration,
                new_state.qv,
                tendency_state.tend_qv,
            ),
            (input_state.qc, self._zero_surface_flux, new_state.qc, tendency_state.tend_qc),
            (input_state.qi, self._zero_surface_flux, new_state.qi, tendency_state.tend_qi),
        ):
            self.diffuse_tracer(
                var=var,
                a=self._matrix_a,
                b=self._matrix_b,
                c=self._matrix_c,
                surface_flux=surface_flux,
                air_mass=input_state.air_mass,
                rho=input_state.rho,
                km_ie=diagnostic_state.km_ie,
                new_var=new_var,
                tend=tend,
                dtime=dtime,
            )

        log.debug("tmx hydrometeor diffusion (Compute_diffusion_hydrometeors): end")

    def run_temperature_diffusion(
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
        Diffuse the temperature as dry static or internal energy (``Compute_diffusion_temperature``
        in mo_vdf.f90).

        The energy is computed with the input tracers and converted back with the new qv, qc and
        qi of ``new_state``, so this runs after :meth:`run_hydrometeor_diffusion`. Needs
        ``kh_ic`` and ``km_ie`` of ``diagnostic_state``.
        """
        log.debug("tmx temperature diffusion (Compute_diffusion_temperature): start")

        self.compute_energy_from_temperature(
            temperature=input_state.temperature,
            qv=input_state.qv,
            qc=input_state.qc,
            qi=input_state.qi,
            qr=input_state.qr,
            qs=input_state.qs,
            qg=input_state.qg,
            energy=self.energy,
        )

        log.debug("communication of energy (cells): start")
        energy_exchange = self._exchange.start(dims.CellDim, self.energy)

        self.assemble_energy_diffusion_matrix(
            diffusivity=diagnostic_state.kh_ic,
            air_mass=input_state.air_mass,
            a=self._matrix_a,
            b=self._matrix_b,
            c=self._matrix_c,
        )

        energy_exchange.finish()
        log.debug("communication of energy (cells): end")

        self.diffuse_energy_and_update_temperature(
            energy=self.energy,
            a=self._matrix_a,
            b=self._matrix_b,
            c=self._matrix_c,
            sensible_heat_flux=surface_flux_state.sensible_heat_flux,
            evapotranspiration=surface_flux_state.evapotranspiration,
            temperature=input_state.temperature,
            new_qv=new_state.qv,
            new_qc=new_state.qc,
            new_qi=new_state.qi,
            qr=input_state.qr,
            qs=input_state.qs,
            qg=input_state.qg,
            air_mass=input_state.air_mass,
            rho=input_state.rho,
            km_ie=diagnostic_state.km_ie,
            new_temperature=new_state.temperature,
            tend_temperature=tendency_state.tend_temperature,
            dtime=dtime,
        )

        log.debug("tmx temperature diffusion (Compute_diffusion_temperature): end")
