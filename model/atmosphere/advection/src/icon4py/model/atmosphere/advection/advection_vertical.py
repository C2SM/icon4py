# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
import logging

from gt4py.next import backend

from icon4py.model.atmosphere.advection import advection_states

from icon4py.model.atmosphere.advection.stencils.copy_cell_kdim_field import copy_cell_kdim_field

from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, geometry
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc


"""
Advection components related to vertical transport.
"""

# flake8: noqa
log = logging.getLogger(__name__)


class VerticalAdvection(ABC):
    """Class that does one vertical advection step."""

    @abstractmethod
    def run(
        self,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO (dastrm): should be KHalfDim
        dtime: ta.wpfloat,
        even_timestep: bool = False,
    ):
        """
        Run a vertical advection step.

        Args:
            prep_adv: input argument, data class that contains precalculated advection fields
            p_tracer_now: input argument, field that contains current tracer mass fraction
            p_tracer_new: output argument, field that contains new tracer mass fraction
            rhodz_now: input argument, field that contains current air mass in each layer
            rhodz_new: input argument, field that contains new air mass in each layer
            p_mflx_tracer_v: output argument, field that contains new vertical tracer mass flux
            dtime: input argument, the time step
            even_timestep: input argument, determines whether halo points are included

        """
        ...


class NoAdvection(VerticalAdvection):
    """Class that implements disabled vertical advection."""

    def __init__(self, grid: icon_grid.IconGrid, backend: backend.Backend):
        log.debug("vertical advection class init - start")

        # input arguments
        self._grid = grid
        self._backend = backend

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_lateral_boundary_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._end_cell_end = self._grid.end_index(cell_domain(h_grid.Zone.END))

        # stencils
        self._copy_cell_kdim_field = copy_cell_kdim_field.with_backend(self._backend)

        log.debug("vertical advection class init - end")

    def run(
        self,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO (dastrm): should be KHalfDim
        dtime: ta.wpfloat,
        even_timestep: bool = False,
    ):
        log.debug("vertical advection run - start")

        horizontal_start = (
            self._start_cell_lateral_boundary_level_2 if even_timestep else self._start_cell_nudging
        )
        log.debug("running stencil copy_cell_kdim_field - start")
        self._copy_cell_kdim_field(
            field_in=p_tracer_now,
            field_out=p_tracer_new,
            horizontal_start=horizontal_start,
            horizontal_end=(self._end_cell_end if even_timestep else self._end_cell_local),
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil copy_cell_kdim_field - end")

        log.debug("vertical advection run - end")


class FiniteVolume(VerticalAdvection):
    """Class that defines a finite volume vertical advection scheme."""

    def run(
        self,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO (dastrm): should be KHalfDim
        dtime: ta.wpfloat,
        even_timestep: bool = False,
    ):
        log.debug("horizontal advection run - start")

        # TODO (dastrm): maybe change how the indices are handled here? originally:
        # if even step, vertical transport includes all halo points in order to avoid an additional synchronization step, i.e.
        #    if lstep_even: i_rlstart  = 2,                i_rlend = min_rlcell
        #    else:          i_rlstart  = grf_bdywidth_c+1, i_rlend = min_rlcell_int
        # note: horizontal advection is always called with the same indices, i.e. i_rlstart = grf_bdywidth_c+1, i_rlend = min_rlcell_int

        self._compute_numerical_flux(
            prep_adv=prep_adv,
            p_tracer_now=p_tracer_now,
            rhodz_now=rhodz_now,
            p_mflx_tracer_v=p_mflx_tracer_v,
            dtime=dtime,
        )

        self._update_unknowns(
            p_tracer_now=p_tracer_now,
            p_tracer_new=p_tracer_new,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            p_mflx_tracer_v=p_mflx_tracer_v,
            dtime=dtime,
        )

        log.debug("horizontal advection run - end")

    @abstractmethod
    def _compute_numerical_flux(
        self,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO (dastrm): should be KHalfDim
        dtime: ta.wpfloat,
    ):
        ...

    @abstractmethod
    def _update_unknowns(
        self,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO (dastrm): should be KHalfDim
        dtime: ta.wpfloat,
    ):
        ...


class SemiLagrangian(FiniteVolume):
    """Class that does one vertical semi-Lagrangian finite volume advection step."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        interpolation_state: advection_states.AdvectionInterpolationState,
        least_squares_state: advection_states.AdvectionLeastSquaresState,
        metric_state: advection_states.AdvectionMetricState,
        edge_params: geometry.EdgeParams,
        cell_params: geometry.CellParams,
        backend: backend.Backend,
        exchange: decomposition.ExchangeRuntime = decomposition.SingleNodeExchange(),
    ):
        log.debug("vertical advection class init - start")

        # input arguments
        self._grid = grid
        self._interpolation_state = interpolation_state
        self._least_squares_state = least_squares_state
        self._metric_state = metric_state
        self._edge_params = edge_params
        self._cell_params = cell_params
        self._backend = backend
        self._exchange = exchange

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_lateral_boundary_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._end_cell_end = self._grid.end_index(cell_domain(h_grid.Zone.END))

        log.debug("vertical advection class init - end")

    def _compute_numerical_flux(
        self,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO (dastrm): should be KHalfDim
        dtime: ta.wpfloat,
    ):
        # TODO (dastrm): implement this
        ...

    def _update_unknowns(
        self,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO (dastrm): should be KHalfDim
        dtime: ta.wpfloat,
    ):
        # TODO (dastrm): implement this
        ...
