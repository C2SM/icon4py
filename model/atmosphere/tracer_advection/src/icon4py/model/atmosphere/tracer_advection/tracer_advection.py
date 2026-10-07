# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from enum import Enum

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.tracer_advection import (
    tracer_advection_horizontal,
    tracer_advection_states,
    tracer_advection_vertical,
)
from icon4py.model.atmosphere.tracer_advection.stencils.apply_density_increment import (
    apply_density_increment,
)
from icon4py.model.atmosphere.tracer_advection.stencils.apply_interpolated_tracer_time_tendency import (
    apply_interpolated_tracer_time_tendency,
)
from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    type_alias as ta,
)
from icon4py.model.common.components import framework as fw, quantities as qty, states
from icon4py.model.common.config import config_io
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.math.stencils import generic_math_operations
from icon4py.model.common.model_options import setup_program
from icon4py.model.common.utils import data_allocation as data_alloc


"""
Advection module ported from ICON mo_advection_stepping.f90.
"""

log = logging.getLogger(__name__)


@config_io.register_enum
class HorizontalAdvectionType(Enum):
    """
    Horizontal operator scheme for tracer advection (originally ihadv_tracer).
    """

    #: no horizontal tracer advection
    NO_ADVECTION = 0
    #: 1st order upwind
    FIRST_ORDER_UPWIND = 1
    #: 2nd order MIURA with linear reconstruction
    SECOND_ORDER_LINEAR_MIURA = 2


@config_io.register_enum
class HorizontalAdvectionLimiter(Enum):
    """
    Limiter for horizontal tracer advection operator (originally itype_hlimit).
    """

    #: no horizontal limiter
    NO_LIMITER = 0
    #: positive definite horizontal limiter
    POSITIVE_DEFINITE = 4


@config_io.register_enum
class VerticalAdvectionType(Enum):
    """
    Vertical operator scheme for tracer advection (originally ivadv_tracer).
    """

    #: no vertical tracer advection
    NO_ADVECTION = 0
    #: 1st order upwind
    FIRST_ORDER_UPWIND = 1
    #: 3rd order PPM
    THIRD_ORDER_PPM = 3


@config_io.register_enum
class VerticalAdvectionLimiter(Enum):
    """
    Limiter for vertical tracer advection operator (originally itype_vlimit).
    """

    #: no vertical limiter
    NO_LIMITER = 0
    #: semi-monotonic vertical limiter
    SEMI_MONOTONIC = 1


@dataclasses.dataclass(frozen=True)
class AdvectionConfig:
    """
    Contains necessary parameters to configure an tracer advection run.
    """

    horizontal_advection_type: HorizontalAdvectionType
    horizontal_advection_limiter: HorizontalAdvectionLimiter
    vertical_advection_type: VerticalAdvectionType
    vertical_advection_limiter: VerticalAdvectionLimiter


class Advection(fw.Component, ABC):
    """
    Runs one three-dimensional tracer advection step for every active tracer.

    The tracers are optional leaves: a tracer is advected when it is given at the current
    time level (input) and the next one (output). The Marchuk order of the horizontal and
    vertical transport alternates once per step, for all tracers (jstep_adv%marchuk_order
    in ICON).

    Missing tracer advection-specific features:
        -optional tendency output: depending on the physics package, opt_ddt_tracer_adv might be needed
        -maximum tracer advection height: tracer-specific control over which levels are used for tracer_advection
    """

    _exchange: decomposition.ExchangeRuntime

    class Input(fw.State):
        qv: fw.Field[qty.QvOnCellK] | None = None
        qc: fw.Field[qty.QcOnCellK] | None = None
        qi: fw.Field[qty.QiOnCellK] | None = None
        qr: fw.Field[qty.QrOnCellK] | None = None
        qs: fw.Field[qty.QsOnCellK] | None = None
        qg: fw.Field[qty.QgOnCellK] | None = None
        #: mass of air in the layer at the beginning and the end of the time step
        airmass_now: fw.Field[qty.AirMassOnCellK]
        airmass_new: fw.Field[qty.AirMassOnCellK]
        grf_tend_tracer: fw.Field[qty.GrfTendencyOfTracerOnCellK]
        #: the fluxes the dycore accumulated over its substeps
        vn_traj: fw.Field[qty.VnOnEdgeK]
        mass_flx_me: fw.Field[qty.MassFluxOnEdgeK]
        mass_flx_ic: fw.Field[qty.MassFluxOnCellKHalf]
        dtime: float

    class Output(fw.State):
        qv: fw.Field[qty.QvOnCellK] | None = None
        qc: fw.Field[qty.QcOnCellK] | None = None
        qi: fw.Field[qty.QiOnCellK] | None = None
        qr: fw.Field[qty.QrOnCellK] | None = None
        qs: fw.Field[qty.QsOnCellK] | None = None
        qg: fw.Field[qty.QgOnCellK] | None = None
        hfl_tracer: fw.Field[qty.HorizontalTracerFluxOnEdgeK]
        vfl_tracer: fw.Field[qty.VerticalTracerFluxOnCellKHalf]

    def run(self, inputs: Input, out: Output | None = None) -> Output:
        out = self.buffers(out)
        self._prepare(inputs)
        for name in states.TRACERS:
            tracer_now = getattr(inputs, name)
            if tracer_now is None:
                continue
            tracer_new = getattr(out, name)
            if tracer_new is None:
                raise ValueError(
                    f"Tracer '{name}' is given at the current time level but not at the next one."
                )
            self._advect(inputs, out, p_tracer_now=tracer_now.data, p_tracer_new=tracer_new.data)
        self._finalize()
        return out

    def _prepare(self, inputs: Input) -> None:
        """Before the tracers: the halo exchange of the vertical mass flux."""
        log.debug("communication of prep_adv cell field: mass_flx_ic - start")
        self._exchange.exchange(
            dims.CellDim, inputs.mass_flx_ic.data, stream=decomposition.DEFAULT_STREAM
        )
        log.debug("communication of prep_adv cell field: mass_flx_ic - end")

    @abstractmethod
    def _advect(
        self,
        inputs: Input,
        out: Output,
        *,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
    ) -> None:
        """Advect one tracer from `p_tracer_now` to `p_tracer_new`."""
        ...

    def _finalize(self) -> None:
        """After the tracers."""


class NoAdvection(Advection):
    """Disable three-dimensional tracer advection."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        backend: gtx_typing.Backend | None,
        exchange: decomposition.ExchangeRuntime,
    ):
        log.debug("tracer_advection class init - start")
        super().__init__(grid, model_backends.get_allocator(backend))

        # input arguments
        self._grid = grid
        self._backend = backend
        self._exchange = exchange

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        # stencils
        self._copy_field_on_cell_k = setup_program(
            backend=self._backend,
            program=generic_math_operations.copy_field_on_cell_k,
            horizontal_sizes={
                "horizontal_start": self._start_cell_nudging,
                "horizontal_end": self._end_cell_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": self._grid.num_levels,
            },
            offset_provider=self._grid.connectivities,
        )

    def _advect(
        self,
        inputs: Advection.Input,
        out: Advection.Output,
        *,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
    ) -> None:
        log.debug("tracer_advection run - start")
        log.debug("running stencil copy_field_on_cell_k - start")
        self._copy_field_on_cell_k(
            field=p_tracer_now,
            output_field=p_tracer_new,
        )
        log.debug("running stencil copy_field_on_cell_k - end")

        log.debug("tracer_advection run - end")


class GodunovSplittingAdvection(Advection):
    """Implements three-dimensional tracer advection based on Godunov splitting."""

    def __init__(
        self,
        *,
        horizontal_advection: tracer_advection_horizontal.HorizontalAdvection,
        vertical_advection: tracer_advection_vertical.VerticalAdvection,
        grid: icon_grid.IconGrid,
        metric_state: tracer_advection_states.AdvectionMetricState,
        backend: gtx_typing.Backend | None,
        exchange: decomposition.ExchangeRuntime,
        even_timestep: bool = False,
    ):
        log.debug("tracer_advection class init - start")
        super().__init__(grid, model_backends.get_allocator(backend))

        # input arguments
        self._horizontal_advection = horizontal_advection
        self._vertical_advection = vertical_advection
        self._grid = grid
        self._metric_state = metric_state
        self._backend = backend
        self._exchange = exchange
        self._even_timestep = even_timestep  # originally jstep_adv(:)%marchuk_order = 1

        # density fields
        #: intermediate density times cell thickness, includes either the horizontal or vertical advective density increment [kg/m^2]
        self._rhodz_ast2 = data_alloc.zero_field(
            self._grid,
            dims.CellDim,
            dims.KDim,
            allocator=model_backends.get_allocator(self._backend),
        )
        self._determine_local_domains()
        # stencils
        self._apply_density_increment = setup_program(
            backend=self._backend,
            program=apply_density_increment,
            constant_args={
                "deepatmo_divzl": self._metric_state.deepatmo_divzl,
                "deepatmo_divzu": self._metric_state.deepatmo_divzu,
            },
            horizontal_sizes={
                "horizontal_end": self._end_cell_end,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        self._apply_interpolated_tracer_time_tendency = setup_program(
            backend=self._backend,
            program=apply_interpolated_tracer_time_tendency,
            horizontal_sizes={
                "horizontal_start": self._start_cell_lateral_boundary,
                "horizontal_end": self._end_cell_lateral_boundary_level_4,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
        )

        log.debug("tracer_advection class init - end")

    def _determine_local_domains(self) -> None:
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

    def _advect(
        self,
        inputs: Advection.Input,
        out: Advection.Output,
        *,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
    ) -> None:
        log.debug("tracer_advection run - start")
        dtime = ta.wpfloat(inputs.dtime)

        # reintegrate density for conservation of mass
        rhodz_in, horizontal_start = (
            (inputs.airmass_now.data, self._start_cell_lateral_boundary_level_2)
            if self._even_timestep
            else (inputs.airmass_new.data, self._start_cell_lateral_boundary_level_3)
        )

        log.debug("running stencil apply_density_increment - start")
        self._apply_density_increment(
            rhodz_in=rhodz_in,
            p_mflx_contra_v=inputs.mass_flx_ic.data,
            rhodz_out=self._rhodz_ast2,
            p_dtime=dtime,
            even_timestep=self._even_timestep,
            horizontal_start=horizontal_start,
        )
        log.debug("running stencil apply_density_increment - end")

        # Godunov splitting
        if self._even_timestep:
            # vertical transport
            self._vertical_advection.run(
                mass_flx_ic=inputs.mass_flx_ic.data,
                p_tracer_now=p_tracer_now,
                p_tracer_new=p_tracer_new,
                rhodz_now=inputs.airmass_now.data,
                rhodz_new=self._rhodz_ast2,
                p_mflx_tracer_v=out.vfl_tracer.data,
                dtime=dtime,
                even_timestep=self._even_timestep,
            )

            # horizontal transport
            self._horizontal_advection.run(
                vn_traj=inputs.vn_traj.data,
                mass_flx_me=inputs.mass_flx_me.data,
                p_tracer_now=p_tracer_new,
                p_tracer_new=p_tracer_new,
                rhodz_now=self._rhodz_ast2,
                rhodz_new=inputs.airmass_new.data,
                p_mflx_tracer_h=out.hfl_tracer.data,
                dtime=dtime,
            )

        else:
            # horizontal transport
            self._horizontal_advection.run(
                vn_traj=inputs.vn_traj.data,
                mass_flx_me=inputs.mass_flx_me.data,
                p_tracer_now=p_tracer_now,
                p_tracer_new=p_tracer_new,
                rhodz_now=inputs.airmass_now.data,
                rhodz_new=self._rhodz_ast2,
                p_mflx_tracer_h=out.hfl_tracer.data,
                dtime=dtime,
            )

            # vertical transport
            self._vertical_advection.run(
                mass_flx_ic=inputs.mass_flx_ic.data,
                p_tracer_now=p_tracer_new,
                p_tracer_new=p_tracer_new,
                rhodz_now=self._rhodz_ast2,
                rhodz_new=inputs.airmass_new.data,
                p_mflx_tracer_v=out.vfl_tracer.data,
                dtime=dtime,
                even_timestep=self._even_timestep,
            )

        # update lateral boundaries with interpolated time tendencies
        if self._grid.limited_area:
            log.debug("running stencil apply_interpolated_tracer_time_tendency - start")
            self._apply_interpolated_tracer_time_tendency(
                p_tracer_now=p_tracer_now,
                p_grf_tend_tracer=inputs.grf_tend_tracer.data,
                p_tracer_new=p_tracer_new,
                p_dtime=dtime,
            )
            log.debug("running stencil apply_interpolated_tracer_time_tendency - end")

        # exchange updated tracer values, originally happens only if iforcing /= inwp
        log.debug("communication of tracer tracer_advection field: p_tracer_new - start")
        self._exchange.exchange(
            dims.CellDim,
            p_tracer_new,
            stream=decomposition.DEFAULT_STREAM,
        )
        log.debug("communication of tracer tracer_advection field: p_tracer_new - end")

        log.debug("tracer_advection run - end")

    def _finalize(self) -> None:
        # the Marchuk order alternates once per step, for all tracers
        self._even_timestep = not self._even_timestep


def convert_config_to_horizontal_vertical_advection(  # noqa: PLR0912 [too-many-branches]
    *,
    config: AdvectionConfig,
    grid: icon_grid.IconGrid,
    interpolation_state: tracer_advection_states.AdvectionInterpolationState,
    least_squares_state: tracer_advection_states.AdvectionLeastSquaresState,
    metric_state: tracer_advection_states.AdvectionMetricState,
    edge_params: grid_states.EdgeParams,
    cell_params: grid_states.CellParams,
    backend: gtx_typing.Backend | None,
    exchange: decomposition.ExchangeRuntime,
) -> tuple[
    tracer_advection_horizontal.HorizontalAdvection, tracer_advection_vertical.VerticalAdvection
]:
    assert exchange is not None, "Exchange runtime must not be None."
    horizontal_limiter: tracer_advection_horizontal.HorizontalFluxLimiter | None
    match config.horizontal_advection_limiter:
        case HorizontalAdvectionLimiter.NO_LIMITER:
            horizontal_limiter = tracer_advection_horizontal.NoLimiter()
        case HorizontalAdvectionLimiter.POSITIVE_DEFINITE:
            horizontal_limiter = tracer_advection_horizontal.PositiveDefinite(
                grid=grid,
                interpolation_state=interpolation_state,
                backend=backend,
                exchange=exchange,
            )
        case _:
            raise NotImplementedError("Unknown horizontal tracer advection limiter.")

    horizontal_advection: tracer_advection_horizontal.HorizontalAdvection
    match config.horizontal_advection_type:
        case HorizontalAdvectionType.NO_ADVECTION:
            horizontal_advection = tracer_advection_horizontal.NoAdvection(
                grid=grid, backend=backend
            )
        case HorizontalAdvectionType.FIRST_ORDER_UPWIND:
            horizontal_advection = tracer_advection_horizontal.FirstOrderUpwind(
                grid=grid,
                interpolation_state=interpolation_state,
                metric_state=metric_state,
                backend=backend,
            )
        case HorizontalAdvectionType.SECOND_ORDER_LINEAR_MIURA:
            tracer_flux = tracer_advection_horizontal.SecondOrderMiura(
                grid=grid,
                least_squares_state=least_squares_state,
                horizontal_limiter=horizontal_limiter,
                backend=backend,
            )
            horizontal_advection = tracer_advection_horizontal.SemiLagrangian(
                tracer_flux=tracer_flux,
                grid=grid,
                interpolation_state=interpolation_state,
                metric_state=metric_state,
                edge_params=edge_params,
                cell_params=cell_params,
                backend=backend,
            )
        case _:
            raise NotImplementedError("Unknown horizontal tracer_advection type.")

    vertical_limiter: tracer_advection_vertical.VerticalLimiter
    match config.vertical_advection_limiter:
        case VerticalAdvectionLimiter.NO_LIMITER:
            vertical_limiter = tracer_advection_vertical.NoLimiter(grid=grid, backend=backend)
        case VerticalAdvectionLimiter.SEMI_MONOTONIC:
            vertical_limiter = tracer_advection_vertical.SemiMonotonicLimiter(
                grid=grid, backend=backend
            )
        case _:
            raise NotImplementedError("Unknown vertical tracer_advection limiter.")

    vertical_advection: tracer_advection_vertical.VerticalAdvection
    match config.vertical_advection_type:
        case VerticalAdvectionType.NO_ADVECTION:
            vertical_advection = tracer_advection_vertical.NoAdvection(grid=grid, backend=backend)
        case VerticalAdvectionType.FIRST_ORDER_UPWIND:
            boundary_conditions = tracer_advection_vertical.NoFluxCondition(
                grid=grid, backend=backend
            )
            vertical_advection = tracer_advection_vertical.FirstOrderUpwind(
                boundary_conditions=boundary_conditions,
                grid=grid,
                metric_state=metric_state,
                backend=backend,
            )
        case VerticalAdvectionType.THIRD_ORDER_PPM:
            boundary_conditions = tracer_advection_vertical.NoFluxCondition(
                grid=grid, backend=backend
            )
            vertical_advection = tracer_advection_vertical.PiecewiseParabolicMethod(
                boundary_conditions=boundary_conditions,
                vertical_limiter=vertical_limiter,
                grid=grid,
                metric_state=metric_state,
                backend=backend,
            )
        case _:
            raise NotImplementedError("Unknown vertical tracer advection type.")

    return horizontal_advection, vertical_advection


def convert_config_to_advection(
    *,
    config: AdvectionConfig,
    grid: icon_grid.IconGrid,
    interpolation_state: tracer_advection_states.AdvectionInterpolationState,
    least_squares_state: tracer_advection_states.AdvectionLeastSquaresState,
    metric_state: tracer_advection_states.AdvectionMetricState,
    edge_params: grid_states.EdgeParams,
    cell_params: grid_states.CellParams,
    backend: gtx_typing.Backend | None,
    exchange: decomposition.ExchangeRuntime,
    even_timestep: bool = False,
) -> Advection:
    if (
        config.horizontal_advection_type == HorizontalAdvectionType.NO_ADVECTION
        and config.vertical_advection_type == VerticalAdvectionType.NO_ADVECTION
    ):
        # tracer advection is disabled for all tracers
        return NoAdvection(grid=grid, backend=backend, exchange=exchange)

    horizontal_advection, vertical_advection = convert_config_to_horizontal_vertical_advection(
        config=config,
        grid=grid,
        interpolation_state=interpolation_state,
        least_squares_state=least_squares_state,
        metric_state=metric_state,
        edge_params=edge_params,
        cell_params=cell_params,
        backend=backend,
        exchange=exchange,
    )

    advection = GodunovSplittingAdvection(
        horizontal_advection=horizontal_advection,
        vertical_advection=vertical_advection,
        grid=grid,
        metric_state=metric_state,
        backend=backend,
        exchange=exchange,
        even_timestep=even_timestep,
    )

    return advection
