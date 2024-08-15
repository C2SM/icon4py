# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import enum

from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.advection import advection_states
from icon4py.model.atmosphere.advection.stencils import step_advection_stencil_02

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, simple as simple_grid
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc
from icon4py.model.atmosphere.dycore.state_utils import states as solve_nh_states


"""
Advection module ported from ICON mo_advection_stepping.f90.
"""

# flake8: noqa
log = logging.getLogger(__name__)


class HorizontalAdvectionType(int, enum.Enum):
    """
    Horizontal operator scheme for advection.
    """

    NO_ADVECTION = 0  #: no horizontal advection
    LINEAR_2ND_ORDER = 2  #: 2nd order MIURA with linear reconstruction


class HorizontalAdvectionLimiterType(int, enum.Enum):
    """
    Limiter for horizontal advection operator.
    """

    NO_LIMITER = 0  #: no horizontal limiter
    POSITIVE_DEFINITE = 4  #: positive definite horizontal limiter


class AdvectionConfig:
    """
    Contains necessary parameters to configure an advection run.

    Default values match a basic implementation.
    """

    def __init__(
        self,
        horizontal_advection_type: HorizontalAdvectionType = HorizontalAdvectionType.LINEAR_2ND_ORDER,
        horizontal_advection_limiter: HorizontalAdvectionLimiterType = HorizontalAdvectionLimiterType.POSITIVE_DEFINITE,
    ):
        """Set the default values according to a basic implementation."""

        self.horizontal_advection_type: int = horizontal_advection_type
        self.horizontal_advection_limiter: int = horizontal_advection_limiter

        self._validate()

    def _validate(self):
        """Apply consistency checks and validation on configuration parameters."""
        if self.horizontal_advection_type != 2:
            raise NotImplementedError(
                "Only horizontal advection type 2 = `2nd order MIURA with linear reconstruction` is implemented"
            )
        if self.horizontal_advection_limiter != 4:
            raise NotImplementedError(
                "Only horizontal advection limiter 4 = `positive definite limiter` is implemented"
            )


class Advection:
    """Class that configures advection and does one advection step."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        config: AdvectionConfig,
        # params: AdvectionParams,
        # edge_params: h_grid.EdgeParams,
        # cell_params: h_grid.CellParams,
        exchange: decomposition.ExchangeRuntime = decomposition.SingleNodeExchange(),
    ):
        """
        Initialize advection granule with configuration.
        """
        log.debug("advection class init - start")

        self.grid: icon_grid.IconGrid = grid
        self.config: AdvectionConfig = config
        # self.params: AdvectionParams = params
        # self.edge_params = edge_params
        # self.cell_params = cell_params
        self._exchange = exchange

        self._allocate_temporary_fields()

        log.debug("advection class init - end")

    def _allocate_temporary_fields(self):
        self.rhodz_ast = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid
        )  # intermediate density times cell thickness [kg/m^2]  (nproma,nlev,nblks_c)
        self.rhodz_ast2 = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid
        )  # intermediate density times cell thickness, includes either the horizontal or vertical advective density increment [kg/m^2]  (nproma,nlev,nblks_c)

    def run(
        self,
        diagnostic_state: advection_states.AdvectionDiagnosticState,
        metric_state: advection_states.AdvectionMetricState,
        prep_adv: solve_nh_states.PrepAdvection,
        p_tracer_now: fa.CellKField[float],
        p_tracer_new: fa.CellKField[float],
        dtime: float,
    ):
        """
        Do one advection step with step size dtime.
        """
        log.debug("advection class run - start")

        self._do_advection_step(
            diagnostic_state=diagnostic_state,
            metric_state=metric_state,
            prep_adv=prep_adv,
            p_tracer_now=p_tracer_now,
            p_tracer_new=p_tracer_new,
            dtime=dtime,
        )

        log.debug("advection class run - end")

    def _do_advection_step(
        self,
        diagnostic_state: advection_states.AdvectionDiagnosticState,
        metric_state: advection_states.AdvectionMetricState,
        prep_adv: solve_nh_states.PrepAdvection,
        p_tracer_now: fa.CellKField[float],
        p_tracer_new: fa.CellKField[float],
        dtime: float,
    ):
        """
        Run an advection step.

        Args:
            diagnostic_state: output argument, data class that contains diagnostic variables
            prep_adv: input argument, data class that contains precalculated advection fields
            p_tracer_now: input argument, field that contains current tracer mass fraction
            p_tracer_new: output argument, field that contains new tracer mass fraction
            dtime: input argument, the time step

        """
        start_cell_lb = (
            0
            if isinstance(self.grid, simple_grid.SimpleGrid)
            else self.grid.get_start_index(
                CellDim, h_grid.HorizontalMarkerIndex.lateral_boundary(CellDim)
            )
        )
        end_cell_end = (
            int32(self.grid.num_cells)
            if isinstance(self.grid, simple_grid.SimpleGrid)
            else self.grid.get_end_index(CellDim, h_grid.HorizontalMarkerIndex.end(CellDim))
        )
        klevels = self.grid.num_levels

        # TODO (dastrm): is this necessary, and if so, correct?
        log.debug("communication of prep_adv cell field: mass_flx_ic - start")
        self._exchange.exchange_and_wait(CellDim, prep_adv.mass_flx_ic)
        log.debug("communication of prep_adv cell field: mass_flx_ic - end")

        # Godunov splitting
        if False:  # even timestep, vertical transport precedes horizontal transport
            # reintegrate density with vertical increment for conservation of mass

            # vertical transport

            # horizontal transport
            pass
        else:  # odd timestep, horizontal transport precedes vertical transport
            # reintegrate density with horizontal increment for conservation of mass
            log.debug("running stencil step_advection_stencil_02 - start")
            step_advection_stencil_02.step_advection_stencil_02(
                p_rhodz_new=diagnostic_state.airmass_new,
                p_mflx_contra_v=prep_adv.mass_flx_ic,
                deepatmo_divzl=metric_state.deepatmo_divzl,
                deepatmo_divzu=metric_state.deepatmo_divzu,
                p_dtime=dtime,
                rhodz_ast2=self.rhodz_ast2,
                horizontal_start=start_cell_lb,
                horizontal_end=end_cell_end,
                vertical_start=0,
                vertical_end=klevels - 1,
                offset_provider=self.grid.offset_providers,
            )
            log.debug("running stencil step_advection_stencil_02 - end")

            # horizontal transport

            # vertical transport
