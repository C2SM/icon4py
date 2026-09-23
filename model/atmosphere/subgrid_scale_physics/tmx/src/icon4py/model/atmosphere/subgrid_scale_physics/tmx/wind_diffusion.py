# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The wind diffusion component of tmx."""

from __future__ import annotations

import functools
import logging
import typing

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import config as tmx_config, tmx_states
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils import (
    wind_diffusion as wind_stencils,
)
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import base as base_grid, horizontal as h_grid
from icon4py.model.common.model_options import setup_program
from icon4py.model.common.utils import data_allocation as data_alloc


if typing.TYPE_CHECKING:
    import icon4py.model.common.grid.states as grid_states
    from icon4py.model.common import field_type_aliases as fa, type_alias as ta


log = logging.getLogger(__name__)


class WindDiffusion:
    """The turbulent diffusion of the horizontal wind (u, v) and the vertical wind (w)."""

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        metric_state: tmx_states.TmxMetricState,
        interpolation_state: tmx_states.TmxInterpolationState,
        edge_params: grid_states.EdgeParams,
        backend: model_backends.BackendLike,
        exchange: decomposition.ExchangeRuntime,
        solver_type: tmx_config.SolverType,
    ) -> None:
        if solver_type != tmx_config.SolverType.IMPLICIT:
            raise NotImplementedError(
                "the wind diffusion only implements the implicit vertical diffusion solver."
            )
        assert edge_params.edge_cell_distances is not None

        self._exchange = exchange
        self._grid = grid

        zero_field = functools.partial(
            data_alloc.zero_field, grid, allocator=model_backends.get_allocator(backend)
        )
        # rows outside the computed domains stay zero and are read as such by the neighbour
        # gathers that consume these fields
        self._vn_tendency: fa.EdgeKField[ta.wpfloat] = zero_field(dims.EdgeDim, dims.KDim)
        self._w_horizontal_stress_tendency: fa.EdgeKHalfField[ta.wpfloat] = zero_field(
            dims.EdgeDim, dims.KHalfDim
        )

        num_levels = grid.num_levels
        cell_domain = h_grid.domain(dims.CellDim)
        edge_domain = h_grid.domain(dims.EdgeDim)
        cell_start_lateral_boundary_level_2 = grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        cell_start_nudging = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        cell_end_local = grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        edge_start_nudging = grid.start_index(edge_domain(h_grid.Zone.NUDGING))
        edge_start_nudging_level_2 = grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
        edge_end_local = grid.end_index(edge_domain(h_grid.Zone.LOCAL))
        edge_end_halo = grid.end_index(edge_domain(h_grid.Zone.HALO))

        self._compute_vn_diffusion_tendency = setup_program(
            backend=backend,
            program=wind_stencils.compute_vn_diffusion_tendency,
            constant_args={
                "inv_ddqz_z_full_e": metric_state.inv_ddqz_z_full_e,
                "inv_ddqz_z_half_e": metric_state.inv_ddqz_z_half_e,
                "c_lin_e": interpolation_state.c_lin_e,
                "primal_normal_cell_x": edge_params.primal_normal_cell[0],
                "primal_normal_cell_y": edge_params.primal_normal_cell[1],
                "primal_normal_vert_x": edge_params.primal_normal_vert[0],
                "primal_normal_vert_y": edge_params.primal_normal_vert[1],
                "dual_normal_vert_x": edge_params.dual_normal_vert[0],
                "dual_normal_vert_y": edge_params.dual_normal_vert[1],
                "tangent_orientation": edge_params.tangent_orientation,
                "inv_primal_edge_length": edge_params.inverse_primal_edge_lengths,
                "inv_vert_vert_length": edge_params.inverse_vertex_vertex_lengths,
                "inv_dual_edge_length": edge_params.inverse_dual_edge_lengths,
            },
            horizontal_sizes={
                "horizontal_start": edge_start_nudging_level_2,
                "horizontal_end": edge_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider=grid.connectivities,
        )
        self._interpolate_and_update_horizontal_wind = setup_program(
            backend=backend,
            program=wind_stencils.interpolate_and_update_horizontal_wind,
            constant_args={
                "rbf_coeff_c1": interpolation_state.rbf_coeff_c1,
                "rbf_coeff_c2": interpolation_state.rbf_coeff_c2,
            },
            horizontal_sizes={
                "tendency_horizontal_start": cell_start_lateral_boundary_level_2,
                "update_horizontal_start": cell_start_nudging,
                "horizontal_end": cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider=grid.connectivities,
        )
        # the edges include the first halo line: the cells of the update gather from them
        self._compute_w_diffusion_tendency_and_update = setup_program(
            backend=backend,
            program=wind_stencils.compute_w_diffusion_tendency_and_update,
            constant_args={
                "inv_ddqz_z_full": metric_state.inv_ddqz_z_full,
                "inv_ddqz_z_half": metric_state.inv_ddqz_z_half,
                "inv_ddqz_z_half_v": metric_state.inv_ddqz_z_half_v,
                "e_bln_c_s": interpolation_state.e_bln_c_s,
                "rbf_coeff_e": interpolation_state.rbf_coeff_e,
                "primal_normal_cell_x": edge_params.primal_normal_cell[0],
                "primal_normal_cell_y": edge_params.primal_normal_cell[1],
                "dual_normal_vert_x": edge_params.dual_normal_vert[0],
                "dual_normal_vert_y": edge_params.dual_normal_vert[1],
                "edge_cell_length": edge_params.edge_cell_distances,
                "tangent_orientation": edge_params.tangent_orientation,
                "inv_primal_edge_length": edge_params.inverse_primal_edge_lengths,
                "inv_vert_vert_length": edge_params.inverse_vertex_vertex_lengths,
                "inv_dual_edge_length": edge_params.inverse_dual_edge_lengths,
            },
            horizontal_sizes={
                "edge_start": edge_start_nudging,
                "edge_end": edge_end_halo,
                "cell_start": cell_start_nudging,
                "cell_end": cell_end_local,
            },
            # w = 0 on the top and bottom half levels, which are not diffused
            vertical_sizes={
                "vertical_start": gtx.int32(1),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider=grid.connectivities,
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
        Diffuse u, v and w: write their tendencies to tendency_state and the updated winds to
        new_state.

        Needs the diagnostics of diagnostic_state for the current input_state. new_state.w is
        zero on the top and bottom half levels. Otherwise only the rows the Fortran computes
        are written; the others keep their values.
        """
        log.debug("tmx wind diffusion: start")

        self._exchange.exchange(
            dims.CellDim, input_state.rho, surface_flux_state.u_stress, surface_flux_state.v_stress
        )
        self._compute_vn_diffusion_tendency(
            rho=input_state.rho,
            w=input_state.w,
            vn=diagnostic_state.vn,
            u_vert=diagnostic_state.u_vert,
            v_vert=diagnostic_state.v_vert,
            km_c=diagnostic_state.km_c,
            div_c=diagnostic_state.div_c,
            km_iv=diagnostic_state.km_iv,
            km_ie=diagnostic_state.km_ie,
            u_stress=surface_flux_state.u_stress,
            v_stress=surface_flux_state.v_stress,
            vn_tendency=self._vn_tendency,
            dtime=dtime,
        )
        self._exchange.exchange(dims.EdgeDim, self._vn_tendency)
        self._interpolate_and_update_horizontal_wind(
            vn_tendency=self._vn_tendency,
            u=input_state.u,
            v=input_state.v,
            tend_u=tendency_state.tend_u,
            tend_v=tendency_state.tend_v,
            new_u=new_state.u,
            new_v=new_state.v,
            dtime=dtime,
        )
        u_v_exchange = self._exchange.start(
            dims.CellDim,
            tendency_state.tend_u,
            tendency_state.tend_v,
            new_state.u,
            new_state.v,
        )

        self._compute_w_diffusion_tendency_and_update(
            w=input_state.w,
            u=input_state.u,
            v=input_state.v,
            vn=diagnostic_state.vn,
            rho_ic=diagnostic_state.rho_ic,
            u_vert=diagnostic_state.u_vert,
            v_vert=diagnostic_state.v_vert,
            w_vert=diagnostic_state.w_vert,
            w_ie=diagnostic_state.w_ie,
            km_c=diagnostic_state.km_c,
            km_ic=diagnostic_state.km_ic,
            km_iv=diagnostic_state.km_iv,
            div_c=diagnostic_state.div_c,
            horizontal_stress_tendency=self._w_horizontal_stress_tendency,
            tend_w=tendency_state.tend_w,
            new_w=new_state.w,
            dtime=dtime,
        )
        # one exchange in flight at a time: GHEX refuses a second on the same communicator
        u_v_exchange.finish()
        self._exchange.exchange(dims.CellDim, new_state.w)

        log.debug("tmx wind diffusion: end")
