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
from typing import Any

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.tracer_advection import tracer_advection_states
from icon4py.model.atmosphere.tracer_advection.stencils.apply_interpolated_tracer_time_tendency import (
    apply_interpolated_tracer_time_tendency,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_fused_tracer_advection import (
    compute_tracer_advection_after_horizontal_limiter,
    compute_tracer_advection_before_horizontal_limiter,
)
from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    type_alias as ta,
)
from icon4py.model.common.config import config_io
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.math.stencils import generic_math_operations
from icon4py.model.common.model_options import setup_program
from icon4py.model.common.states import tracer_prep_adv_states as prep_adv_states
from icon4py.model.common.utils import data_allocation as data_alloc, fortran_config


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

    @classmethod
    def from_fortran_dict(cls, atmo_dict: dict[str, Any], **overrides: Any) -> AdvectionConfig:
        transport_nml = atmo_dict["transport_nml"]
        return cls(
            horizontal_advection_type=HorizontalAdvectionType(
                fortran_config.list_to_value(transport_nml["ihadv_tracer"])
            ),
            horizontal_advection_limiter=HorizontalAdvectionLimiter(
                fortran_config.list_to_value(transport_nml["itype_hlimit"])
            ),
            vertical_advection_type=VerticalAdvectionType(
                fortran_config.list_to_value(transport_nml["ivadv_tracer"])
            ),
            vertical_advection_limiter=VerticalAdvectionLimiter(
                fortran_config.list_to_value(transport_nml["itype_vlimit"])
            ),
            **overrides,
        )


class Advection(ABC):
    """
    Runs one three-dimensional tracer advection step.

    Missing tracer advection-specific features:
        -tracer loops: currently the `run` method only advects one type of tracer at once
        -optional tendency output: depending on the physics package, opt_ddt_tracer_adv might be needed
        -maximum tracer advection height: tracer-specific control over which levels are used for tracer_advection
    """

    @abstractmethod
    def run(
        self,
        *,
        diagnostic_state: tracer_advection_states.AdvectionDiagnosticState,
        prep_adv: prep_adv_states.TracerPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        """
        Run an tracer advection step.

        Args:
            diagnostic_state: output argument, data class that contains diagnostic variables
            prep_adv: input argument, data class that contains precalculated fields for tracer advection
            p_tracer_now: input argument, field that contains current tracer mass fraction
            p_tracer_new: output argument, field that contains new tracer mass fraction
            dtime: input argument, the time step

        """
        ...


class NoAdvection(Advection):
    """Disable three-dimensional tracer advection."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        backend: gtx_typing.Backend | None,
        exchange: decomposition.ExchangeRuntime,
    ):
        log.debug("tracer_advection class init - start")

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

    def run(
        self,
        *,
        diagnostic_state: tracer_advection_states.AdvectionDiagnosticState,
        prep_adv: prep_adv_states.TracerPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        log.debug("tracer_advection run - start")
        log.debug("communication of prep_adv cell field: mass_flx_ic - start")
        self._exchange.exchange(
            dims.CellDim, prep_adv.mass_flx_ic, stream=decomposition.DEFAULT_STREAM
        )
        log.debug("communication of prep_adv cell field: mass_flx_ic - end")

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
        grid: icon_grid.IconGrid,
        metric_state: tracer_advection_states.AdvectionMetricState,
        interpolation_state: tracer_advection_states.AdvectionInterpolationState,
        least_squares_state: tracer_advection_states.AdvectionLeastSquaresState,
        edge_params: grid_states.EdgeParams,
        backend: gtx_typing.Backend | None,
        exchange: decomposition.ExchangeRuntime,
        even_timestep: bool = False,
        vertical_advection_type: VerticalAdvectionType = VerticalAdvectionType.THIRD_ORDER_PPM,
        horizontal_advection_type: HorizontalAdvectionType = HorizontalAdvectionType.SECOND_ORDER_LINEAR_MIURA,
        horizontal_advection_limiter: HorizontalAdvectionLimiter = HorizontalAdvectionLimiter.POSITIVE_DEFINITE,
        vertical_advection_limiter: VerticalAdvectionLimiter = VerticalAdvectionLimiter.SEMI_MONOTONIC,
    ):
        log.debug("tracer_advection class init - start")

        # input arguments
        self._grid = grid
        self._metric_state = metric_state
        self._backend = backend
        self._exchange = exchange
        self._even_timestep = even_timestep  # originally jstep_adv(:)%marchuk_order = 1

        self._determine_local_domains()

        allocator = model_backends.get_allocator(self._backend)

        self._rhodz_ast2 = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=allocator
        )
        self._r_m = data_alloc.zero_field(self._grid, dims.CellDim, dims.KDim, allocator=allocator)
        self._p_tracer_after_vertical = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=allocator
        )
        self._p_mflx_tracer_h_unlimited = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, allocator=allocator
        )
        self._k = data_alloc.index_field(
            self._grid,
            dims.KDim,
            dtype=gtx.int32,
            allocator=allocator,
        )
        self._k_half = data_alloc.index_field(
            self._grid,
            dims.KHalfDim,
            dtype=gtx.int32,
            allocator=allocator,
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

        metric_state = self._metric_state
        shared_horizontal_args: dict[str, gtx.Field | gtx_typing.Scalar] = {
            "rbf_vec_coeff_e": interpolation_state.rbf_vec_coeff_e,
            "pos_on_tplane_e_1": interpolation_state.pos_on_tplane_e_1,
            "pos_on_tplane_e_2": interpolation_state.pos_on_tplane_e_2,
            "primal_normal_cell_1": edge_params.primal_normal_cell[0],
            "dual_normal_cell_1": edge_params.dual_normal_cell[0],
            "primal_normal_cell_2": edge_params.primal_normal_cell[1],
            "dual_normal_cell_2": edge_params.dual_normal_cell[1],
            "lsq_pseudoinv_1": least_squares_state.lsq_pseudoinv_1,
            "lsq_pseudoinv_2": least_squares_state.lsq_pseudoinv_2,
            "geofac_div": interpolation_state.geofac_div,
            "dbl_eps": constants.DBL_EPS,
            "ihadv_tracer": gtx.int32(horizontal_advection_type.value),
            "itype_hlimit": gtx.int32(horizontal_advection_limiter.value),
        }
        shared_vertical_args: dict[str, gtx.Field | gtx_typing.Scalar] = {
            "p_cellhgt_mc_now": metric_state.ddqz_z_full,
            "deepatmo_divzl": metric_state.deepatmo_divzl,
            "deepatmo_divzu": metric_state.deepatmo_divzu,
            "k": self._k,
            "k_half": self._k_half,
            "slev": gtx.int32(0),
            "slevp1_ti": gtx.int32(1),
            "elev": gtx.int32(self._grid.num_levels - 1),
            "ivadv_tracer": gtx.int32(vertical_advection_type.value),
            "iadv_slev_jt": gtx.int32(0),
            "dbl_eps": constants.DBL_EPS,
            "itype_vlimit": gtx.int32(vertical_advection_limiter.value),
        }
        horizontal_domains: dict[str, gtx.int32] = {
            "start_cell_lateral_boundary_level_2": self._start_cell_lateral_boundary_level_2,
            "end_cell_local": self._end_cell_local,
            "end_cell_end": self._end_cell_end,
            "start_edge_lateral_boundary_level_5": self._start_edge_lateral_boundary_level_5,
            "end_edge_halo": self._end_edge_halo,
        }
        vertical_domains: dict[str, gtx.int32] = {"vertical_end": gtx.int32(self._grid.num_levels)}

        self._compute_after_horizontal_limiter = setup_program(
            backend=self._backend,
            program=compute_tracer_advection_after_horizontal_limiter,
            constant_args={
                **shared_vertical_args,
                "deepatmo_divh": metric_state.deepatmo_divh,
                "geofac_div": interpolation_state.geofac_div,
                "ihadv_tracer": gtx.int32(horizontal_advection_type.value),
                "itype_hlimit": gtx.int32(horizontal_advection_limiter.value),
            },
            horizontal_sizes={
                "start_cell_nudging": self._start_cell_nudging,
                "end_cell_local": self._end_cell_local,
                "start_edge_lateral_boundary_level_5": self._start_edge_lateral_boundary_level_5,
                "end_edge_halo": self._end_edge_halo,
            },
            vertical_sizes=vertical_domains,
            offset_provider=self._grid.connectivities,
        )
        self._compute_after_horizontal_limiter = setup_program(
            backend=self._backend,
            program=compute_tracer_advection_after_horizontal_limiter,
            constant_args={
                **shared_vertical_args,
                "deepatmo_divh": metric_state.deepatmo_divh,
                "geofac_div": interpolation_state.geofac_div,
                "ihadv_tracer": gtx.int32(horizontal_advection_type.value),
                "itype_hlimit": gtx.int32(horizontal_advection_limiter.value),
            },
            horizontal_sizes={
                "start_cell_nudging": self._start_cell_nudging,
                "end_cell_local": self._end_cell_local,
                "start_edge_lateral_boundary_level_5": self._start_edge_lateral_boundary_level_5,
                "end_edge_halo": self._end_edge_halo,
            },
            vertical_sizes=vertical_domains,
            offset_provider=self._grid.connectivities,
        )
        self._compute_before_horizontal_limiter = setup_program(
            backend=self._backend,
            program=compute_tracer_advection_before_horizontal_limiter,
            constant_args={**shared_vertical_args, **shared_horizontal_args},
            horizontal_sizes=horizontal_domains,
            vertical_sizes=vertical_domains,
            offset_provider=self._grid.connectivities,
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
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_lateral_boundary_level_4 = self._grid.end_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
        )
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._end_cell_end = self._grid.end_index(cell_domain(h_grid.Zone.END))

        edge_domain = h_grid.domain(dims.EdgeDim)
        self._start_edge_lateral_boundary_level_5 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._end_edge_halo = self._grid.end_index(edge_domain(h_grid.Zone.HALO))

    def run(
        self,
        *,
        diagnostic_state: tracer_advection_states.AdvectionDiagnosticState,
        prep_adv: prep_adv_states.TracerPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        log.debug("tracer_advection run - start")

        self._exchange.exchange(
            dims.CellDim,
            prep_adv.mass_flx_ic,
            stream=decomposition.DEFAULT_STREAM,
        )

        self._compute_before_horizontal_limiter(
            rhodz_ast2=self._rhodz_ast2,
            p_mflx_tracer_v=diagnostic_state.vfl_tracer,
            p_tracer_after_vertical=self._p_tracer_after_vertical,
            p_mflx_tracer_h_unlimited=self._p_mflx_tracer_h_unlimited,
            r_m=self._r_m,
            rhodz_now=diagnostic_state.airmass_now,
            rhodz_new=diagnostic_state.airmass_new,
            p_mflx_contra_v=prep_adv.mass_flx_ic,
            p_tracer_now=p_tracer_now,
            p_mass_flx_e=prep_adv.mass_flx_me,
            p_vn=prep_adv.vn_traj,
            p_dtime=dtime,
            even_timestep=self._even_timestep,
        )

        self._exchange.exchange(dims.CellDim, self._r_m, stream=decomposition.DEFAULT_STREAM)

        self._compute_after_horizontal_limiter(
            p_mflx_tracer_h=diagnostic_state.hfl_tracer,
            p_mflx_tracer_v=diagnostic_state.vfl_tracer,
            p_tracer_new=p_tracer_new,
            r_m=self._r_m,
            p_mflx_tracer_h_unlimited=self._p_mflx_tracer_h_unlimited,
            p_tracer_now=p_tracer_now,
            p_tracer_after_vertical=self._p_tracer_after_vertical,
            rhodz_ast2=self._rhodz_ast2,
            rhodz_now=diagnostic_state.airmass_now,
            rhodz_new=diagnostic_state.airmass_new,
            p_mflx_contra_v=prep_adv.mass_flx_ic,
            do_vertical_first=gtx.int32(1 if self._even_timestep else 0),
            p_dtime=dtime,
        )

        if self._grid.limited_area:
            self._apply_interpolated_tracer_time_tendency(
                p_tracer_now=p_tracer_now,
                p_grf_tend_tracer=diagnostic_state.grf_tend_tracer,
                p_tracer_new=p_tracer_new,
                p_dtime=dtime,
            )

        self._exchange.exchange(
            dims.CellDim,
            p_tracer_new,
            stream=decomposition.DEFAULT_STREAM,
        )
        self._even_timestep = not self._even_timestep

        log.debug("tracer_advection run - end")


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

    return GodunovSplittingAdvection(
        grid=grid,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        least_squares_state=least_squares_state,
        edge_params=edge_params,
        backend=backend,
        exchange=exchange,
        even_timestep=even_timestep,
        vertical_advection_type=config.vertical_advection_type,
        horizontal_advection_type=config.horizontal_advection_type,
        horizontal_advection_limiter=config.horizontal_advection_limiter,
        vertical_advection_limiter=config.vertical_advection_limiter,
    )
