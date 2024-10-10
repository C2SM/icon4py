# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
import dataclasses
import enum
import logging

from icon4py.model.atmosphere.advection import (
    advection_states,
    advection_horizontal,
    advection_vertical,
)
from icon4py.model.atmosphere.advection.stencils import (
    apply_density_increment,
    apply_interpolated_tracer_time_tendency,
    copy_cell_kdim_field,
)
from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, geometry
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc


"""
Advection module ported from ICON mo_advection_stepping.f90.
"""

# flake8: noqa
log = logging.getLogger(__name__)


class HorizontalAdvectionType(enum.Enum):
    """
    Horizontal operator scheme for advection.
    """

    #: no horizontal advection
    NO_ADVECTION = 0
    #: 2nd order MIURA with linear reconstruction
    LINEAR_2ND_ORDER = 2


class HorizontalAdvectionLimiter(enum.Enum):
    """
    Limiter for horizontal advection operator.
    """

    #: no horizontal limiter
    NO_LIMITER = 0
    #: positive definite horizontal limiter
    POSITIVE_DEFINITE = 4


class VerticalAdvectionType(enum.Enum):
    """
    Vertical operator scheme for advection.
    """

    #: no vertical advection
    NO_ADVECTION = 0


class VerticalAdvectionLimiter(enum.Enum):
    """
    Limiter for vertical advection operator.
    """

    #: no vertical limiter
    NO_LIMITER = 0


@dataclasses.dataclass(frozen=True)
class AdvectionConfig:
    horizontal_advection_type: HorizontalAdvectionType
    horizontal_advection_limiter: HorizontalAdvectionLimiter
    vertical_advection_type: VerticalAdvectionType
    vertical_advection_limiter: VerticalAdvectionLimiter

    """
    Contains necessary parameters to configure an advection run.
    """

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Apply consistency checks and validation on configuration parameters."""

        assert (
            self.horizontal_advection_type not in HorizontalAdvectionType.__members__
        ), f"Horizontal advection type {self.horizontal_advection_type} not implemented."
        assert (
            self.horizontal_advection_limiter not in HorizontalAdvectionLimiter.__members__
        ), f"Horizontal advection limiter {self.horizontal_advection_limiter} not implemented."
        assert (
            self.vertical_advection_type not in VerticalAdvectionType.__members__
        ), f"Vertical advection type {self.vertical_advection_type} not implemented."
        assert (
            self.vertical_advection_limiter not in VerticalAdvectionLimiter.__members__
        ), f"Vertical advection limiter {self.vertical_advection_limiter} not implemented."


class Advection(ABC):
    """Class that runs one three-dimensional advection step."""

    @abstractmethod
    def run(
        self,
        diagnostic_state: advection_states.AdvectionDiagnosticState,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
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
        ...


class NoAdvection(Advection):
    """Class that implements disabled three-dimensional advection."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        exchange: decomposition.ExchangeRuntime = decomposition.SingleNodeExchange(),
    ):
        log.debug("advection class init - start")

        # input arguments
        self._grid = grid
        self._exchange = exchange

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))

    def run(
        self,
        diagnostic_state: advection_states.AdvectionDiagnosticState,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ):
        log.debug("advection run - start")

        log.debug("communication of prep_adv cell field: mass_flx_ic - start")
        self._exchange.exchange_and_wait(dims.CellDim, prep_adv.mass_flx_ic)
        log.debug("communication of prep_adv cell field: mass_flx_ic - end")

        log.debug("running stencil copy_cell_kdim_field - start")
        copy_cell_kdim_field.copy_cell_kdim_field(
            field_in=p_tracer_now,
            field_out=p_tracer_new,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil copy_cell_kdim_field - end")

        log.debug("advection run - end")


class GodunovSplittingAdvection(Advection):
    """Class that implements three-dimensional advection based on Godunov splitting."""

    def __init__(
        self,
        horizontal_advection: advection_horizontal.HorizontalAdvection,
        vertical_advection: advection_vertical.VerticalAdvection,
        grid: icon_grid.IconGrid,
        metric_state: advection_states.AdvectionMetricState,
        exchange: decomposition.ExchangeRuntime = decomposition.SingleNodeExchange(),
        even_timestep: bool = False,
    ):
        log.debug("advection class init - start")

        # input arguments
        self._horizontal_advection = horizontal_advection
        self._vertical_advection = vertical_advection
        self._grid = grid
        self._metric_state = metric_state
        self._exchange = exchange
        self._even_timestep = even_timestep  # originally jstep_adv(:)%marchuk_order = 1

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_lateral_boundary = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY)
        )
        self._start_cell_lateral_boundary_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._start_cell_lateral_boundary_level_3 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        self._end_cell_lateral_boundary_level_4 = self._grid.end_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
        )
        self._end_cell_end = self._grid.end_index(cell_domain(h_grid.Zone.END))

        # density fields
        #: intermediate density times cell thickness, includes either the horizontal or vertical advective density increment [kg/m^2]
        self._rhodz_ast2 = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=self._grid)

        log.debug("advection class init - end")

    def run(
        self,
        diagnostic_state: advection_states.AdvectionDiagnosticState,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ):
        log.debug("advection run - start")

        log.debug("communication of prep_adv cell field: mass_flx_ic - start")
        self._exchange.exchange_and_wait(dims.CellDim, prep_adv.mass_flx_ic)
        log.debug("communication of prep_adv cell field: mass_flx_ic - end")

        # reintegrate density for conservation of mass
        log.debug("running stencil apply_density_increment - start")
        apply_density_increment.apply_density_increment(
            rhodz_in=diagnostic_state.airmass_now
            if self._even_timestep
            else diagnostic_state.airmass_new,
            p_mflx_contra_v=prep_adv.mass_flx_ic,
            deepatmo_divzl=self._metric_state.deepatmo_divzl,
            deepatmo_divzu=self._metric_state.deepatmo_divzu,
            p_dtime=dtime,
            even_timestep=self._even_timestep,
            rhodz_out=self._rhodz_ast2,
            horizontal_start=self._start_cell_lateral_boundary_level_2
            if self._even_timestep
            else self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_end,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil apply_density_increment - end")

        # Godunov splitting
        if self._even_timestep:
            # vertical transport
            self._vertical_advection.run(
                prep_adv=prep_adv,
                p_tracer_now=p_tracer_now,
                p_tracer_new=p_tracer_new,
                rhodz_now=diagnostic_state.airmass_now,
                rhodz_new=self._rhodz_ast2,
                p_mflx_tracer_v=diagnostic_state.vfl_tracer,
                dtime=dtime,
                even_timestep=self._even_timestep,
            )

            # horizontal transport
            self._horizontal_advection.run(
                prep_adv=prep_adv,
                p_tracer_now=p_tracer_new,
                p_tracer_new=p_tracer_new,
                rhodz_now=self._rhodz_ast2,
                rhodz_new=diagnostic_state.airmass_new,
                p_mflx_tracer_h=diagnostic_state.hfl_tracer,
                dtime=dtime,
            )

        else:
            # horizontal transport
            self._horizontal_advection.run(
                prep_adv=prep_adv,
                p_tracer_now=p_tracer_now,
                p_tracer_new=p_tracer_new,
                rhodz_now=diagnostic_state.airmass_now,
                rhodz_new=self._rhodz_ast2,
                p_mflx_tracer_h=diagnostic_state.hfl_tracer,
                dtime=dtime,
            )

            # vertical transport
            self._vertical_advection.run(
                prep_adv=prep_adv,
                p_tracer_now=p_tracer_new,
                p_tracer_new=p_tracer_new,
                rhodz_now=self._rhodz_ast2,
                rhodz_new=diagnostic_state.airmass_new,
                p_mflx_tracer_v=diagnostic_state.vfl_tracer,
                dtime=dtime,
                even_timestep=self._even_timestep,
            )

        # update lateral boundaries with interpolated time tendencies
        if self._grid.limited_area:
            log.debug("running stencil apply_interpolated_tracer_time_tendency - start")
            apply_interpolated_tracer_time_tendency.apply_interpolated_tracer_time_tendency(
                p_tracer_now=p_tracer_now,
                p_grf_tend_tracer=diagnostic_state.grf_tend_tracer,
                p_tracer_new=p_tracer_new,
                p_dtime=dtime,
                horizontal_start=self._start_cell_lateral_boundary,
                horizontal_end=self._end_cell_lateral_boundary_level_4,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
                offset_provider=self._grid.offset_providers,
            )
            log.debug("running stencil apply_interpolated_tracer_time_tendency - end")

        # exchange updated tracer values, originally happens only if iforcing /= inwp
        log.debug("communication of advection cell field: p_tracer_new - start")
        self._exchange.exchange_and_wait(dims.CellDim, p_tracer_new)
        log.debug("communication of advection cell field: p_tracer_new - end")

        # finalize step
        self._even_timestep = not self._even_timestep

        log.debug("advection run - end")


def convert_config_to_horizontal_vertical_advection(
    config: AdvectionConfig,
    grid: icon_grid.IconGrid,
    interpolation_state: advection_states.AdvectionInterpolationState,
    least_squares_state: advection_states.AdvectionLeastSquaresState,
    metric_state: advection_states.AdvectionMetricState,
    edge_params: geometry.EdgeParams,
    cell_params: geometry.CellParams,
    exchange: decomposition.ExchangeRuntime = decomposition.SingleNodeExchange(),
) -> tuple[advection_horizontal.HorizontalAdvection, advection_vertical.VerticalAdvection]:
    if config.horizontal_advection_type == HorizontalAdvectionType.NO_ADVECTION:
        horizontal_advection = advection_horizontal.NoHorizontalAdvection(grid=grid)
    else:
        if config.horizontal_advection_limiter == HorizontalAdvectionLimiter.POSITIVE_DEFINITE:
            horizontal_limiter = advection_horizontal.PositiveDefiniteLimiter(
                grid=grid, interpolation_state=interpolation_state, exchange=exchange
            )
        else:
            horizontal_limiter = advection_horizontal.HorizontalFluxLimiter()

        if config.horizontal_advection_type == HorizontalAdvectionType.LINEAR_2ND_ORDER:
            horizontal_flux = advection_horizontal.SecondOrderMiuraHorizontal(
                grid=grid,
                least_squares_state=least_squares_state,
                horizontal_limiter=horizontal_limiter,
            )

        horizontal_advection = advection_horizontal.SemiLagrangianHorizontalAdvection(
            horizontal_flux=horizontal_flux,
            grid=grid,
            interpolation_state=interpolation_state,
            least_squares_state=least_squares_state,
            metric_state=metric_state,
            edge_params=edge_params,
            cell_params=cell_params,
            exchange=exchange,
        )

    if config.vertical_advection_type == VerticalAdvectionType.NO_ADVECTION:
        vertical_advection = advection_vertical.NoVerticalAdvection(grid=grid)

    return horizontal_advection, vertical_advection


def convert_config_to_advection(
    config: AdvectionConfig,
    grid: icon_grid.IconGrid,
    interpolation_state: advection_states.AdvectionInterpolationState,
    least_squares_state: advection_states.AdvectionLeastSquaresState,
    metric_state: advection_states.AdvectionMetricState,
    edge_params: geometry.EdgeParams,
    cell_params: geometry.CellParams,
    exchange: decomposition.ExchangeRuntime = decomposition.SingleNodeExchange(),
    even_timestep: bool = False,
) -> Advection:
    if (
        config.horizontal_advection_type == HorizontalAdvectionType.NO_ADVECTION
        and config.vertical_advection_type == VerticalAdvectionType.NO_ADVECTION
    ):
        # advection is disabled for all tracers
        return NoAdvection(grid=grid)

    horizontal_advection, vertical_advection = convert_config_to_horizontal_vertical_advection(
        config=config,
        grid=grid,
        interpolation_state=interpolation_state,
        least_squares_state=least_squares_state,
        metric_state=metric_state,
        edge_params=edge_params,
        cell_params=cell_params,
        exchange=exchange,
    )

    advection = GodunovSplittingAdvection(
        horizontal_advection=horizontal_advection,
        vertical_advection=vertical_advection,
        grid=grid,
        metric_state=metric_state,
        exchange=exchange,
        even_timestep=even_timestep,
    )

    return advection