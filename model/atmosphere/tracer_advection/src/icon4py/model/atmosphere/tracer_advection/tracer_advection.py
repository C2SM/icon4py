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
from icon4py.model.atmosphere.tracer_advection.stencils.compute_fused_tracer_advection import (
    compute_tracer_advection_even_timestep_after_horizontal_limiter,
    compute_tracer_advection_even_timestep_before_horizontal_limiter,
    compute_tracer_advection_odd_timestep_after_horizontal_limiter,
    compute_tracer_advection_odd_timestep_before_horizontal_limiter,
)
from icon4py.model.atmosphere.tracer_advection.stencils.copy_cell_kdim_field import (
    copy_cell_kdim_field,
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
from icon4py.model.common.model_options import setup_program
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
    #: 2nd order MIURA with linear reconstruction
    LINEAR_2ND_ORDER = 2


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
    UPWIND_1ST_ORDER = 1
    #: 3rd order PPM
    PPM_3RD_ORDER = 3


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
        prep_adv: tracer_advection_states.AdvectionPrepAdvState,
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
    """Class that implements disabled three-dimensional tracer advection."""

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
        self._copy_cell_kdim_field = setup_program(
            backend=self._backend,
            program=copy_cell_kdim_field,
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
        prep_adv: tracer_advection_states.AdvectionPrepAdvState,
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

        log.debug("running stencil copy_cell_kdim_field - start")
        self._copy_cell_kdim_field(
            field_in=p_tracer_now,
            field_out=p_tracer_new,
        )
        log.debug("running stencil copy_cell_kdim_field - end")

        log.debug("tracer_advection run - end")


class GodunovSplittingAdvection(Advection):
    """Class that implements three-dimensional tracer_advection based on Godunov splitting."""

    def __init__(
        self,
        *,
        horizontal_advection: tracer_advection_horizontal.HorizontalAdvection | None,
        vertical_advection: tracer_advection_vertical.VerticalAdvection | None,
        grid: icon_grid.IconGrid,
        metric_state: tracer_advection_states.AdvectionMetricState,
        backend: gtx_typing.Backend | None,
        exchange: decomposition.ExchangeRuntime,
        even_timestep: bool = False,
        production_state: tracer_advection_states.ProductionAdvectionState | None = None,
        production_interpolation_state: tracer_advection_states.AdvectionInterpolationState
        | None = None,
        production_least_squares_state: tracer_advection_states.AdvectionLeastSquaresState
        | None = None,
        production_edge_params: grid_states.EdgeParams | None = None,
    ):
        log.debug("tracer_advection class init - start")

        # input arguments
        self._horizontal_advection = horizontal_advection
        self._vertical_advection = vertical_advection
        self._grid = grid
        self._metric_state = metric_state
        self._backend = backend
        self._exchange = exchange
        self._even_timestep = even_timestep  # originally jstep_adv(:)%marchuk_order = 1
        self._production_state = production_state
        self._production_k: fa.KField[gtx.int32] | None = None

        self._determine_local_domains()
        if self._production_state is None:
            # intermediate density times cell thickness, includes either the horizontal or
            # vertical advective density increment [kg/m^2]
            self._rhodz_ast2 = data_alloc.zero_field(
                self._grid,
                dims.CellDim,
                dims.KDim,
                allocator=model_backends.get_allocator(self._backend),
            )
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
        if self._production_state is not None:
            assert production_interpolation_state is not None
            assert production_least_squares_state is not None
            assert production_edge_params is not None
            self._production_k = data_alloc.index_field(
                self._grid,
                dims.KDim,
                extend={dims.KDim: 1},
                dtype=gtx.int32,
                allocator=model_backends.get_allocator(self._backend),
            )
            self._setup_production_programs(
                interpolation_state=production_interpolation_state,
                least_squares_state=production_least_squares_state,
                edge_params=production_edge_params,
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

    def run(
        self,
        *,
        diagnostic_state: tracer_advection_states.AdvectionDiagnosticState,
        prep_adv: tracer_advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        log.debug("tracer_advection run - start")

        if self._production_state is not None:
            self._run_production_advection(
                diagnostic_state=diagnostic_state,
                prep_adv=prep_adv,
                p_tracer_now=p_tracer_now,
                p_tracer_new=p_tracer_new,
                dtime=dtime,
            )
            return

        log.debug("communication of prep_adv cell field: mass_flx_ic - start")
        self._exchange.exchange(
            dims.CellDim,
            prep_adv.mass_flx_ic,
            stream=decomposition.DEFAULT_STREAM,
        )
        log.debug("communication of prep_adv cell field: mass_flx_ic - end")

        assert self._horizontal_advection is not None
        assert self._vertical_advection is not None

        # reintegrate density for conservation of mass
        rhodz_in, horizontal_start = (
            (diagnostic_state.airmass_now, self._start_cell_lateral_boundary_level_2)
            if self._even_timestep
            else (diagnostic_state.airmass_new, self._start_cell_lateral_boundary_level_3)
        )

        log.debug("running stencil apply_density_increment - start")
        self._apply_density_increment(
            rhodz_in=rhodz_in,
            p_mflx_contra_v=prep_adv.mass_flx_ic,
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
            self._apply_interpolated_tracer_time_tendency(
                p_tracer_now=p_tracer_now,
                p_grf_tend_tracer=diagnostic_state.grf_tend_tracer,
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

        # finalize step
        self._even_timestep = not self._even_timestep

        log.debug("tracer_advection run - end")

    def _setup_production_programs(
        self,
        *,
        interpolation_state: tracer_advection_states.AdvectionInterpolationState,
        least_squares_state: tracer_advection_states.AdvectionLeastSquaresState,
        edge_params: grid_states.EdgeParams,
    ) -> None:
        assert self._production_state is not None
        assert self._production_k is not None
        metric_state = self._metric_state
        self._determine_production_local_domains()

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
        }
        shared_vertical_args: dict[str, gtx.Field | gtx_typing.Scalar] = {
            "p_cellhgt_mc_now": metric_state.ddqz_z_full,
            "deepatmo_divzl": metric_state.deepatmo_divzl,
            "deepatmo_divzu": metric_state.deepatmo_divzu,
            "k": self._production_k,
            "slev": gtx.int32(0),
            "slevp1_ti": gtx.int32(1),
            "elev": gtx.int32(self._grid.num_levels - 1),
            "ivadv_tracer": gtx.int32(1),
            "iadv_slev_jt": gtx.int32(0),
            "dbl_eps": constants.DBL_EPS,
        }
        horizontal_domains: dict[str, gtx.int32] = {
            "start_cell_lateral_boundary_level_2": self._start_cell_lateral_boundary_level_2,
            "end_cell_local": self._end_cell_local,
            "end_cell_end": self._end_cell_end,
            "start_edge_lateral_boundary_level_5": self._start_edge_lateral_boundary_level_5,
            "end_edge_halo": self._end_edge_halo,
        }
        vertical_domains: dict[str, gtx.int32] = {"vertical_end": gtx.int32(self._grid.num_levels)}

        self._compute_even_timestep_before_horizontal_limiter = setup_program(
            backend=self._backend,
            program=compute_tracer_advection_even_timestep_before_horizontal_limiter,
            constant_args={**shared_vertical_args, **shared_horizontal_args},
            horizontal_sizes=horizontal_domains,
            vertical_sizes=vertical_domains,
            offset_provider=self._grid.connectivities,
        )
        self._compute_even_timestep_after_horizontal_limiter = setup_program(
            backend=self._backend,
            program=compute_tracer_advection_even_timestep_after_horizontal_limiter,
            constant_args={
                "deepatmo_divh": metric_state.deepatmo_divh,
                "geofac_div": interpolation_state.geofac_div,
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
        self._compute_odd_timestep_before_horizontal_limiter = setup_program(
            backend=self._backend,
            program=compute_tracer_advection_odd_timestep_before_horizontal_limiter,
            constant_args={
                "deepatmo_divzl": metric_state.deepatmo_divzl,
                "deepatmo_divzu": metric_state.deepatmo_divzu,
                **shared_horizontal_args,
            },
            horizontal_sizes={
                **horizontal_domains,
                "start_cell_lateral_boundary_level_3": self._start_cell_lateral_boundary_level_3,
            },
            vertical_sizes=vertical_domains,
            offset_provider=self._grid.connectivities,
        )
        self._compute_odd_timestep_after_horizontal_limiter = setup_program(
            backend=self._backend,
            program=compute_tracer_advection_odd_timestep_after_horizontal_limiter,
            constant_args={
                **shared_vertical_args,
                "deepatmo_divh": metric_state.deepatmo_divh,
                "geofac_div": interpolation_state.geofac_div,
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

    def _determine_production_local_domains(self) -> None:
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

    def _run_production_advection(
        self,
        *,
        diagnostic_state: tracer_advection_states.AdvectionDiagnosticState,
        prep_adv: tracer_advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        log.debug("fused production tracer_advection run - start")
        assert self._production_state is not None
        production_state = self._production_state
        assert self._production_k is not None
        self._exchange.exchange(
            dims.CellDim,
            prep_adv.mass_flx_ic,
            stream=decomposition.DEFAULT_STREAM,
        )

        if self._even_timestep:
            self._compute_even_timestep_before_horizontal_limiter(
                rhodz_ast2=production_state.rhodz_ast2,
                p_mflx_tracer_v=diagnostic_state.vfl_tracer,
                p_tracer_after_vertical=production_state.p_tracer_after_vertical,
                p_mflx_tracer_h_unlimited=production_state.p_mflx_tracer_h_unlimited,
                r_m=production_state.r_m,
                rhodz_now=diagnostic_state.airmass_now,
                p_mflx_contra_v=prep_adv.mass_flx_ic,
                p_tracer_now=p_tracer_now,
                p_mass_flx_e=prep_adv.mass_flx_me,
                p_vn=prep_adv.vn_traj,
                p_dtime=dtime,
            )
        else:
            self._compute_odd_timestep_before_horizontal_limiter(
                rhodz_ast2=production_state.rhodz_ast2,
                p_mflx_tracer_h_unlimited=production_state.p_mflx_tracer_h_unlimited,
                r_m=production_state.r_m,
                rhodz_now=diagnostic_state.airmass_now,
                rhodz_new=diagnostic_state.airmass_new,
                p_mflx_contra_v=prep_adv.mass_flx_ic,
                p_tracer_now=p_tracer_now,
                p_mass_flx_e=prep_adv.mass_flx_me,
                p_vn=prep_adv.vn_traj,
                p_dtime=dtime,
            )

        self._exchange.exchange(
            dims.CellDim, production_state.r_m, stream=decomposition.DEFAULT_STREAM
        )

        if self._even_timestep:
            self._compute_even_timestep_after_horizontal_limiter(
                p_mflx_tracer_h=diagnostic_state.hfl_tracer,
                p_tracer_new=p_tracer_new,
                r_m=production_state.r_m,
                p_mflx_tracer_h_unlimited=production_state.p_mflx_tracer_h_unlimited,
                p_tracer_after_vertical=production_state.p_tracer_after_vertical,
                rhodz_ast2=production_state.rhodz_ast2,
                rhodz_new=diagnostic_state.airmass_new,
                p_dtime=dtime,
            )
        else:
            self._compute_odd_timestep_after_horizontal_limiter(
                p_mflx_tracer_h=diagnostic_state.hfl_tracer,
                p_mflx_tracer_v=diagnostic_state.vfl_tracer,
                p_tracer_new=p_tracer_new,
                r_m=production_state.r_m,
                p_mflx_tracer_h_unlimited=production_state.p_mflx_tracer_h_unlimited,
                p_tracer_now=p_tracer_now,
                rhodz_ast2=production_state.rhodz_ast2,
                rhodz_now=diagnostic_state.airmass_now,
                rhodz_new=diagnostic_state.airmass_new,
                p_mflx_contra_v=prep_adv.mass_flx_ic,
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
        log.debug("fused production tracer_advection run - end")


def _is_production_advection_config(config: AdvectionConfig) -> bool:
    return (
        config.horizontal_advection_type == HorizontalAdvectionType.LINEAR_2ND_ORDER
        and config.horizontal_advection_limiter == HorizontalAdvectionLimiter.POSITIVE_DEFINITE
        and config.vertical_advection_type == VerticalAdvectionType.PPM_3RD_ORDER
        and config.vertical_advection_limiter == VerticalAdvectionLimiter.SEMI_MONOTONIC
    )


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
        case HorizontalAdvectionType.LINEAR_2ND_ORDER:
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
        case VerticalAdvectionType.UPWIND_1ST_ORDER:
            boundary_conditions = tracer_advection_vertical.NoFluxCondition(
                grid=grid, backend=backend
            )
            vertical_advection = tracer_advection_vertical.FirstOrderUpwind(
                boundary_conditions=boundary_conditions,
                grid=grid,
                metric_state=metric_state,
                backend=backend,
            )
        case VerticalAdvectionType.PPM_3RD_ORDER:
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

    production_state: tracer_advection_states.ProductionAdvectionState | None
    horizontal_advection: tracer_advection_horizontal.HorizontalAdvection | None
    vertical_advection: tracer_advection_vertical.VerticalAdvection | None
    if _is_production_advection_config(config):
        production_state = tracer_advection_states.initialize_production_advection_state(
            grid=grid,
            allocator=model_backends.get_allocator(backend),
        )
        horizontal_advection = None
        vertical_advection = None
    else:
        production_state = None
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
        production_state=production_state,
        production_interpolation_state=interpolation_state,
        production_least_squares_state=least_squares_state,
        production_edge_params=edge_params,
    )

    return advection
