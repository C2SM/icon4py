# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import math
from abc import ABC, abstractmethod

import gt4py.next as gtx

import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.tracer_advection import tracer_advection_states, weno_least_squares
from icon4py.model.atmosphere.tracer_advection.stencils.accumulate_weno_candidate_flux_weights import (
    accumulate_weno_candidate_flux_weights,
)
from icon4py.model.atmosphere.tracer_advection.stencils.apply_monotone_horizontal_multiplicative_flux_factors import (
    apply_monotone_horizontal_multiplicative_flux_factors,
)
from icon4py.model.atmosphere.tracer_advection.stencils.apply_positive_definite_horizontal_multiplicative_flux_factor import (
    apply_positive_definite_horizontal_multiplicative_flux_factor,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_antidiffusive_cell_fluxes_and_min_max import (
    compute_antidiffusive_cell_fluxes_and_min_max,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_barycentric_backtrajectory_alt import (
    compute_barycentric_backtrajectory_alt,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_edge_tangential import (
    compute_edge_tangential,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_ffsl_backtrajectory import (
    compute_ffsl_backtrajectory,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_ffsl_backtrajectory_counterclockwise_indicator import (
    compute_ffsl_backtrajectory_counterclockwise_indicator,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_horizontal_tracer_flux_from_linear_coefficients_alt import (
    compute_horizontal_tracer_flux_from_linear_coefficients_alt,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_horizontal_tracer_flux_from_quadratic_coefficients import (
    compute_horizontal_tracer_flux_from_quadratic_coefficients,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_horizontal_tracer_flux_from_weno_coefficients import (
    compute_horizontal_tracer_flux_from_weno_coefficients,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_horizontal_tracer_flux_upwind import (
    compute_horizontal_tracer_flux_upwind,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_monotone_horizontal_multiplicative_flux_factors import (
    compute_monotone_horizontal_multiplicative_flux_factors,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_positive_definite_horizontal_multiplicative_flux_factor import (
    compute_positive_definite_horizontal_multiplicative_flux_factor,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_upwind_and_antidiffusive_flux import (
    compute_upwind_and_antidiffusive_flux,
)
from icon4py.model.atmosphere.tracer_advection.stencils.copy_cell_kdim_field import (
    copy_cell_kdim_field,
)
from icon4py.model.atmosphere.tracer_advection.stencils.init_constant_edge_kdim_field import (
    init_constant_edge_kdim_field,
)
from icon4py.model.atmosphere.tracer_advection.stencils.integrate_tracer_horizontally import (
    integrate_tracer_horizontally,
)
from icon4py.model.atmosphere.tracer_advection.stencils.postprocess_antidiffusive_cell_fluxes_and_min_max import (
    postprocess_antidiffusive_cell_fluxes_and_min_max,
)
from icon4py.model.atmosphere.tracer_advection.stencils.prepare_gauss_quadrature_quadratic_miura3 import (
    prepare_gauss_quadrature_quadratic_miura3,
)
from icon4py.model.atmosphere.tracer_advection.stencils.reconstruct_linear_coefficients_svd import (
    reconstruct_linear_coefficients_svd,
)
from icon4py.model.atmosphere.tracer_advection.stencils.reconstruct_linear_coefficients_weno_svd import (
    reconstruct_linear_coefficients_weno_svd,
)
from icon4py.model.atmosphere.tracer_advection.stencils.reconstruct_quadratic_coefficients_svd import (
    reconstruct_quadratic_coefficients_svd,
)
from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    model_options,
    type_alias as ta,
)
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.states import adv_states
from icon4py.model.common.utils import data_allocation as data_alloc


"""Advection components related to horizontal transport."""

log = logging.getLogger(__name__)


class HorizontalFluxLimiter(ABC):
    """Class that limits the horizontal finite volume numerical flux."""

    @abstractmethod
    def apply_flux_limiter(
        self,
        *,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None: ...


class NoLimiter(HorizontalFluxLimiter):
    """Do not apply any limiting."""

    def apply_flux_limiter(
        self,
        *,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None: ...


class PositiveDefinite(HorizontalFluxLimiter):
    """Class that implements a positive definite horizontal flux limiter."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        interpolation_state: tracer_advection_states.AdvectionInterpolationState,
        backend: gtx.typing.Backend | None,
        exchange: decomposition.ExchangeRuntime,
    ):
        self._grid = grid
        self._interpolation_state = interpolation_state
        self._backend = backend
        self._exchange = exchange

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_lateral_boundary_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        # edge indices
        edge_domain = h_grid.domain(dims.EdgeDim)

        self._start_edge_lateral_boundary_level_5 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._end_edge_halo = self._grid.end_index(edge_domain(h_grid.Zone.HALO))

        # limiter fields
        self._r_m = data_alloc.zero_field(
            self._grid,
            dims.CellDim,
            dims.KDim,
            allocator=model_backends.get_allocator(self._backend),
        )

        # stencils
        self._compute_positive_definite_horizontal_multiplicative_flux_factor = (
            model_options.setup_program(
                backend=self._backend,
                program=compute_positive_definite_horizontal_multiplicative_flux_factor,
                constant_args={
                    "geofac_div": self._interpolation_state.geofac_div,
                    "dbl_eps": constants.DBL_EPS,
                },
                horizontal_sizes={
                    "horizontal_start": self._start_cell_lateral_boundary_level_2,
                    "horizontal_end": self._end_cell_local,
                },
                vertical_sizes={
                    "vertical_start": gtx.int32(0),
                    "vertical_end": gtx.int32(self._grid.num_levels),
                },
                offset_provider=self._grid.connectivities,
            )
        )
        self._apply_positive_definite_horizontal_multiplicative_flux_factor = (
            model_options.setup_program(
                backend=self._backend,
                program=apply_positive_definite_horizontal_multiplicative_flux_factor,
                horizontal_sizes={
                    "horizontal_start": self._start_edge_lateral_boundary_level_5,
                    "horizontal_end": self._end_edge_halo,
                },
                vertical_sizes={
                    "vertical_start": gtx.int32(0),
                    "vertical_end": gtx.int32(self._grid.num_levels),
                },
                offset_provider=self._grid.connectivities,
            )
        )

    def apply_flux_limiter(
        self,
        *,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        # compute multiplicative flux factor to guarantee no undershoot
        log.debug(
            "running stencil compute_positive_definite_horizontal_multiplicative_flux_factor - start"
        )
        self._compute_positive_definite_horizontal_multiplicative_flux_factor(
            p_cc=p_tracer_now,
            p_rhodz_now=rhodz_now,
            p_mflx_tracer_h=p_mflx_tracer_h,
            r_m=self._r_m,
            p_dtime=dtime,
        )
        log.debug(
            "running stencil compute_positive_definite_horizontal_multiplicative_flux_factor - end"
        )

        log.debug("communication of tracer_advection cell field: r_m - start")
        self._exchange.exchange(dims.CellDim, self._r_m, stream=decomposition.DEFAULT_STREAM)
        log.debug("communication of tracer_advection cell field: r_m - end")

        # limit outward fluxes
        log.debug(
            "running stencil apply_positive_definite_horizontal_multiplicative_flux_factor - start"
        )
        self._apply_positive_definite_horizontal_multiplicative_flux_factor(
            r_m=self._r_m,
            p_mflx_tracer_h=p_mflx_tracer_h,
        )
        log.debug(
            "running stencil apply_positive_definite_horizontal_multiplicative_flux_factor - end"
        )


class Monotonic(HorizontalFluxLimiter):
    """Zalesak flux-corrected transport limiter, ported from hflx_limiter_mo.

    The high-order flux is split into a first-order upwind part plus an antidiffusive
    remainder, and the remainder is scaled down per edge until the low-order solution
    stays inside the local range of the neighbouring cells, widened by ``beta_fct``.
    """

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        interpolation_state: tracer_advection_states.AdvectionInterpolationState,
        backend: gtx.typing.Backend | None,
        exchange: decomposition.ExchangeRuntime,
        beta_fct: ta.wpfloat,
    ):
        self._grid = grid
        self._interpolation_state = interpolation_state
        self._backend = backend
        self._exchange = exchange
        self._beta_fct = beta_fct

        allocator = model_backends.get_allocator(self._backend)

        cell_domain = h_grid.domain(dims.CellDim)
        edge_domain = h_grid.domain(dims.EdgeDim)

        # f90 361-378 only repairs the boundary interpolation zone of a limited-area grid
        self._limited_area = self._grid.limited_area

        def _cell_field(dtype: type) -> fa.CellKField:
            return data_alloc.zero_field(
                self._grid, dims.CellDim, dims.KDim, dtype=dtype, allocator=allocator
            )

        def _edge_field(dtype: type) -> fa.EdgeKField:
            return data_alloc.zero_field(
                self._grid, dims.EdgeDim, dims.KDim, dtype=dtype, allocator=allocator
            )

        self._z_mflx_low = _edge_field(ta.wpfloat)
        self._z_anti = _edge_field(ta.wpfloat)
        self._z_mflx_anti_in = _cell_field(ta.vpfloat)
        self._z_mflx_anti_out = _cell_field(ta.vpfloat)
        self._z_tracer_new_low = _cell_field(ta.wpfloat)
        self._z_tracer_max = _cell_field(ta.vpfloat)
        self._z_tracer_min = _cell_field(ta.vpfloat)
        # zero-initialization is load-bearing: the r_p/r_m rows below the nudging zone are
        # never written by the stencil below, and f90 390-397 zeroes exactly those rows
        self._r_p = _cell_field(ta.wpfloat)
        self._r_m = _cell_field(ta.wpfloat)

        vertical_sizes = {
            "vertical_start": gtx.int32(0),
            "vertical_end": gtx.int32(self._grid.num_levels),
        }

        # f90 228-262: one halo row deeper than the other advection edge stencils, because
        # the cell stage below reads the antidiffusive flux of the cells' outermost edges
        self._compute_upwind_and_antidiffusive_flux = model_options.setup_program(
            backend=self._backend,
            program=compute_upwind_and_antidiffusive_flux,
            horizontal_sizes={
                "horizontal_start": self._grid.start_index(
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
                ),
                "horizontal_end": self._grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2)),
            },
            vertical_sizes=vertical_sizes,
            offset_provider=self._grid.connectivities,
        )

        # f90 279-350
        cell_stage_2_sizes = {
            "horizontal_start": self._grid.start_index(
                cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
            ),
            "horizontal_end": self._grid.end_index(cell_domain(h_grid.Zone.HALO)),
        }
        self._compute_antidiffusive_cell_fluxes_and_min_max = model_options.setup_program(
            backend=self._backend,
            program=compute_antidiffusive_cell_fluxes_and_min_max,
            constant_args={"geofac_div": self._interpolation_state.geofac_div},
            horizontal_sizes=cell_stage_2_sizes,
            vertical_sizes=vertical_sizes,
            offset_provider=self._grid.connectivities,
        )

        # f90 361-378
        self._postprocess_antidiffusive_cell_fluxes_and_min_max = (
            model_options.setup_program(
                backend=self._backend,
                program=postprocess_antidiffusive_cell_fluxes_and_min_max,
                constant_args={
                    "refin_ctrl": self._grid.refinement_control[dims.CellDim],
                    # the two boundary-interpolation rows, grf_bdywidth_c - 1 and grf_bdywidth_c
                    "lo_bound": gtx.int32(3),
                    "hi_bound": gtx.int32(4),
                },
                horizontal_sizes=cell_stage_2_sizes,
                vertical_sizes=vertical_sizes,
                offset_provider=self._grid.connectivities,
            )
            if self._limited_area
            else None
        )

        # f90 405-462
        self._compute_monotone_horizontal_multiplicative_flux_factors = model_options.setup_program(
            backend=self._backend,
            program=compute_monotone_horizontal_multiplicative_flux_factors,
            constant_args={
                "beta_fct": self._beta_fct,
                "r_beta_fct": 1.0 / self._beta_fct,
                "dbl_eps": constants.DBL_EPS,
            },
            horizontal_sizes={
                "horizontal_start": self._grid.start_index(
                    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
                ),
                "horizontal_end": self._grid.end_index(cell_domain(h_grid.Zone.LOCAL)),
            },
            vertical_sizes=vertical_sizes,
            offset_provider=self._grid.connectivities,
        )

        # f90 471-525
        self._apply_monotone_horizontal_multiplicative_flux_factors = model_options.setup_program(
            backend=self._backend,
            program=apply_monotone_horizontal_multiplicative_flux_factors,
            horizontal_sizes={
                "horizontal_start": self._grid.start_index(edge_domain(h_grid.Zone.NUDGING)),
                "horizontal_end": self._grid.end_index(edge_domain(h_grid.Zone.HALO)),
            },
            vertical_sizes=vertical_sizes,
            offset_provider=self._grid.connectivities,
        )

    def apply_flux_limiter(
        self,
        *,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        # split the high-order flux into a first-order upwind part and the remainder
        log.debug("running stencil compute_upwind_and_antidiffusive_flux - start")
        self._compute_upwind_and_antidiffusive_flux(
            p_mflx_tracer_h=p_mflx_tracer_h,
            p_mass_flx_e=p_mass_flx_e,
            p_cc=p_tracer_now,
            z_mflx_low=self._z_mflx_low,
            z_anti=self._z_anti,
        )
        log.debug("running stencil compute_upwind_and_antidiffusive_flux - end")

        # the low-order solution and the local range the limited solution must stay in
        log.debug("running stencil compute_antidiffusive_cell_fluxes_and_min_max - start")
        self._compute_antidiffusive_cell_fluxes_and_min_max(
            p_rhodz_now=rhodz_now,
            p_rhodz_new=rhodz_new,
            z_mflx_low=self._z_mflx_low,
            z_anti=self._z_anti,
            p_cc=p_tracer_now,
            z_mflx_anti_in=self._z_mflx_anti_in,
            z_mflx_anti_out=self._z_mflx_anti_out,
            z_tracer_new_low=self._z_tracer_new_low,
            z_tracer_max=self._z_tracer_max,
            z_tracer_min=self._z_tracer_min,
            p_dtime=dtime,
        )
        log.debug("running stencil compute_antidiffusive_cell_fluxes_and_min_max - end")

        if self._postprocess_antidiffusive_cell_fluxes_and_min_max is not None:
            log.debug("running stencil postprocess_antidiffusive_cell_fluxes_and_min_max - start")
            self._postprocess_antidiffusive_cell_fluxes_and_min_max(
                p_cc=p_tracer_now,
                z_tracer_new_low=self._z_tracer_new_low,
                z_tracer_max=self._z_tracer_max,
                z_tracer_min=self._z_tracer_min,
                z_tracer_new_low_out=self._z_tracer_new_low,
                z_tracer_max_out=self._z_tracer_max,
                z_tracer_min_out=self._z_tracer_min,
            )
            log.debug("running stencil postprocess_antidiffusive_cell_fluxes_and_min_max - end")

        # per-cell headroom for incoming and outgoing antidiffusive mass
        log.debug("running stencil compute_monotone_horizontal_multiplicative_flux_factors - start")
        self._compute_monotone_horizontal_multiplicative_flux_factors(
            z_tracer_max=self._z_tracer_max,
            z_tracer_min=self._z_tracer_min,
            z_mflx_anti_in=self._z_mflx_anti_in,
            z_mflx_anti_out=self._z_mflx_anti_out,
            z_tracer_new_low=self._z_tracer_new_low,
            r_p=self._r_p,
            r_m=self._r_m,
        )
        log.debug("running stencil compute_monotone_horizontal_multiplicative_flux_factors - end")

        log.debug("communication of tracer_advection cell fields: r_m, r_p - start")
        self._exchange.exchange(
            dims.CellDim, self._r_m, self._r_p, stream=decomposition.DEFAULT_STREAM
        )
        log.debug("communication of tracer_advection cell fields: r_m, r_p - end")

        # every edge takes the smaller of the two headrooms it connects
        log.debug("running stencil apply_monotone_horizontal_multiplicative_flux_factors - start")
        self._apply_monotone_horizontal_multiplicative_flux_factors(
            z_anti=self._z_anti,
            r_m=self._r_m,
            r_p=self._r_p,
            z_mflx_low=self._z_mflx_low,
            p_mflx_tracer_h=p_mflx_tracer_h,
        )
        log.debug("running stencil apply_monotone_horizontal_multiplicative_flux_factors - end")


class SemiLagrangianTracerFlux(ABC):
    """Class that defines the horizontal semi-Lagrangian tracer flux."""

    @abstractmethod
    def compute_tracer_flux(
        self,
        *,
        prep_adv: adv_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        p_distv_bary_1: fa.EdgeKField[ta.anyfloat],
        p_distv_bary_2: fa.EdgeKField[ta.anyfloat],
        p_vt: fa.EdgeKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        """Compute the tracer flux; p_vt is only consumed by the ffsl-based schemes."""
        ...


class SecondOrderMiura(SemiLagrangianTracerFlux):
    """Class that computes a Miura-based second-order accurate tracer flux."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        least_squares_state: tracer_advection_states.AdvectionLeastSquaresState,
        backend: gtx.typing.Backend | None,
        horizontal_limiter: HorizontalFluxLimiter | None = None,
    ):
        self._grid = grid
        self._least_squares_state = least_squares_state
        self._backend = backend
        self._horizontal_limiter = horizontal_limiter or NoLimiter()

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_lateral_boundary_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._end_cell_halo = self._grid.end_index(cell_domain(h_grid.Zone.HALO))

        # edge indices
        edge_domain = h_grid.domain(dims.EdgeDim)
        self._start_edge_lateral_boundary_level_5 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._end_edge_halo = self._grid.end_index(edge_domain(h_grid.Zone.HALO))

        # reconstruction fields
        allocator = model_backends.get_allocator(self._backend)
        self._p_coeff_1 = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=allocator
        )
        self._p_coeff_2 = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=allocator
        )
        self._p_coeff_3 = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=allocator
        )

        # stencils
        self._reconstruct_linear_coefficients_svd = model_options.setup_program(
            backend=self._backend,
            program=reconstruct_linear_coefficients_svd,
            constant_args={
                "lsq_pseudoinv_1": self._least_squares_state.lsq_pseudoinv_1,
                "lsq_pseudoinv_2": self._least_squares_state.lsq_pseudoinv_2,
            },
            horizontal_sizes={
                "horizontal_start": self._start_cell_lateral_boundary_level_2,
                "horizontal_end": self._end_cell_halo,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        self._compute_horizontal_tracer_flux_from_linear_coefficients_alt = (
            model_options.setup_program(
                backend=self._backend,
                program=compute_horizontal_tracer_flux_from_linear_coefficients_alt,
                horizontal_sizes={
                    "horizontal_start": self._start_edge_lateral_boundary_level_5,
                    "horizontal_end": self._end_edge_halo,
                },
                vertical_sizes={
                    "vertical_start": gtx.int32(0),
                    "vertical_end": gtx.int32(self._grid.num_levels),
                },
                offset_provider=self._grid.connectivities,
            )
        )

    def compute_tracer_flux(
        self,
        *,
        prep_adv: adv_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        p_distv_bary_1: fa.EdgeKField[ta.anyfloat],
        p_distv_bary_2: fa.EdgeKField[ta.anyfloat],
        p_vt: fa.EdgeKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        log.debug("horizontal tracer flux computation - start")

        # linear reconstruction using singular value decomposition
        log.debug("running stencil reconstruct_linear_coefficients_svd - start")
        self._reconstruct_linear_coefficients_svd(
            p_cc=p_tracer_now,
            p_coeff_1_dsl=self._p_coeff_1,
            p_coeff_2_dsl=self._p_coeff_2,
            p_coeff_3_dsl=self._p_coeff_3,
        )
        log.debug("running stencil reconstruct_linear_coefficients_svd - end")

        # compute reconstructed tracer value at each barycenter and corresponding flux at each edge
        log.debug(
            "running stencil compute_horizontal_tracer_flux_from_linear_coefficients_alt - start"
        )
        self._compute_horizontal_tracer_flux_from_linear_coefficients_alt(
            z_lsq_coeff_1=self._p_coeff_1,
            z_lsq_coeff_2=self._p_coeff_2,
            z_lsq_coeff_3=self._p_coeff_3,
            distv_bary_1=p_distv_bary_1,
            distv_bary_2=p_distv_bary_2,
            p_mass_flx_e=prep_adv.mass_flx_me,
            p_vn=prep_adv.vn_traj,
            p_out_e=p_mflx_tracer_h,
        )
        log.debug(
            "running stencil compute_horizontal_tracer_flux_from_linear_coefficients_alt - end"
        )

        self._horizontal_limiter.apply_flux_limiter(
            p_tracer_now=p_tracer_now,
            p_mflx_tracer_h=p_mflx_tracer_h,
            p_mass_flx_e=prep_adv.mass_flx_me,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            dtime=dtime,
        )

        log.debug("horizontal tracer flux computation - end")


class SecondOrderMiuraWeno(SemiLagrangianTracerFlux):
    """Class that computes a Miura-based second-order accurate tracer flux with linear WENO reconstruction (ihadv_tracer=102)."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        weno_linear_state: tracer_advection_states.AdvectionWenoLinearState,
        backend: gtx.typing.Backend | None,
        horizontal_limiter: HorizontalFluxLimiter | None = None,
    ):
        self._grid = grid
        self._weno_linear_state = weno_linear_state
        self._backend = backend
        self._horizontal_limiter = horizontal_limiter or NoLimiter()

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_lateral_boundary_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._end_cell_halo = self._grid.end_index(cell_domain(h_grid.Zone.HALO))

        # edge indices
        edge_domain = h_grid.domain(dims.EdgeDim)
        self._start_edge_lateral_boundary_level_5 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._end_edge_halo = self._grid.end_index(edge_domain(h_grid.Zone.HALO))

        # reconstruction fields
        allocator = model_backends.get_allocator(self._backend)
        self._p_coeff_1 = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=allocator
        )
        self._p_coeff_2 = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=allocator
        )
        self._p_coeff_3 = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=allocator
        )

        # stencils
        self._reconstruct_linear_coefficients_weno_svd = model_options.setup_program(
            backend=self._backend,
            program=reconstruct_linear_coefficients_weno_svd,
            horizontal_sizes={
                "horizontal_start": self._start_cell_lateral_boundary_level_2,
                "horizontal_end": self._end_cell_halo,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        self._compute_horizontal_tracer_flux_from_linear_coefficients_alt = (
            model_options.setup_program(
                backend=self._backend,
                program=compute_horizontal_tracer_flux_from_linear_coefficients_alt,
                horizontal_sizes={
                    "horizontal_start": self._start_edge_lateral_boundary_level_5,
                    "horizontal_end": self._end_edge_halo,
                },
                vertical_sizes={
                    "vertical_start": gtx.int32(0),
                    "vertical_end": gtx.int32(self._grid.num_levels),
                },
                offset_provider=self._grid.connectivities,
            )
        )

    def compute_tracer_flux(
        self,
        *,
        prep_adv: adv_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        p_distv_bary_1: fa.EdgeKField[ta.anyfloat],
        p_distv_bary_2: fa.EdgeKField[ta.anyfloat],
        p_vt: fa.EdgeKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        log.debug("horizontal tracer flux computation - start")

        # linear WENO reconstruction blending 3 least-squares candidates
        log.debug("running stencil reconstruct_linear_coefficients_weno_svd - start")
        self._reconstruct_linear_coefficients_weno_svd(
            p_cc=p_tracer_now,
            lsq_pseudoinv_zonal_c1=self._weno_linear_state.lsq_pseudoinv_zonal_c1,
            lsq_pseudoinv_zonal_c2=self._weno_linear_state.lsq_pseudoinv_zonal_c2,
            lsq_pseudoinv_zonal_c3=self._weno_linear_state.lsq_pseudoinv_zonal_c3,
            lsq_pseudoinv_meridional_c1=self._weno_linear_state.lsq_pseudoinv_meridional_c1,
            lsq_pseudoinv_meridional_c2=self._weno_linear_state.lsq_pseudoinv_meridional_c2,
            lsq_pseudoinv_meridional_c3=self._weno_linear_state.lsq_pseudoinv_meridional_c3,
            p_coeff_1_dsl=self._p_coeff_1,
            p_coeff_2_dsl=self._p_coeff_2,
            p_coeff_3_dsl=self._p_coeff_3,
        )
        log.debug("running stencil reconstruct_linear_coefficients_weno_svd - end")

        # compute reconstructed tracer value at each barycenter and corresponding flux at each edge
        log.debug(
            "running stencil compute_horizontal_tracer_flux_from_linear_coefficients_alt - start"
        )
        self._compute_horizontal_tracer_flux_from_linear_coefficients_alt(
            z_lsq_coeff_1=self._p_coeff_1,
            z_lsq_coeff_2=self._p_coeff_2,
            z_lsq_coeff_3=self._p_coeff_3,
            distv_bary_1=p_distv_bary_1,
            distv_bary_2=p_distv_bary_2,
            p_mass_flx_e=prep_adv.mass_flx_me,
            p_vn=prep_adv.vn_traj,
            p_out_e=p_mflx_tracer_h,
        )
        log.debug(
            "running stencil compute_horizontal_tracer_flux_from_linear_coefficients_alt - end"
        )

        self._horizontal_limiter.apply_flux_limiter(
            p_tracer_now=p_tracer_now,
            p_mflx_tracer_h=p_mflx_tracer_h,
            p_mass_flx_e=prep_adv.mass_flx_me,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            dtime=dtime,
        )

        log.debug("horizontal tracer flux computation - end")


def _gauss_legendre_o2_quadrature_args() -> dict:
    """Gauss-Legendre O2 shape functions and weights (init_2D_gauss_quad,
    mo_advection_config.f90 1084-1136), as constant arguments of the quadrature stencil."""
    gauss = 1.0 / math.sqrt(3.0)
    zeta = (-gauss, gauss, gauss, -gauss)
    eta = (-gauss, -gauss, gauss, gauss)
    args: dict = {
        "wgt_zeta_1": 1.0,
        "wgt_zeta_2": 1.0,
        "wgt_eta_1": 1.0,
        "wgt_eta_2": 1.0,
    }
    for jg in range(4):
        args[f"shape_func_1_{jg + 1}"] = 0.25 * (1.0 - zeta[jg]) * (1.0 - eta[jg])
        args[f"shape_func_2_{jg + 1}"] = 0.25 * (1.0 + zeta[jg]) * (1.0 - eta[jg])
        args[f"shape_func_3_{jg + 1}"] = 0.25 * (1.0 + zeta[jg]) * (1.0 + eta[jg])
        args[f"shape_func_4_{jg + 1}"] = 0.25 * (1.0 - zeta[jg]) * (1.0 + eta[jg])
    return args


class ThirdOrderMiura(SemiLagrangianTracerFlux):
    """Miura-based third-order tracer flux with a quadratic reconstruction (ihadv_tracer=3).

    Port of upwind_hflux_miura3 (mo_advection_hflux.f90 4500-4790) for lsq_high_ord=2,
    live path only (l_out_edgeval=.FALSE.). It is 'ThirdOrderMiuraWeno' with the
    27-candidate loop collapsed to the single full-stencil reconstruction, so it shares
    the geometry, the quadrature and the reconstruction stencil with it and differs only
    in the last step, which dots the coefficients of the upwind cell with the quadrature
    vector instead of a smoothness-weighted blend.

    The cubic variant (lsq_high_ord=3) is not implemented: it needs 10 unknowns, the
    third-order moments and its own quadrature.
    """

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        quadratic_state: tracer_advection_states.AdvectionQuadraticState,
        backend: gtx.typing.Backend | None,
        horizontal_limiter: HorizontalFluxLimiter | None = None,
    ):
        self._grid = grid
        self._quadratic_state = quadratic_state
        self._backend = backend
        self._horizontal_limiter = horizontal_limiter or NoLimiter()

        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_lateral_boundary_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._end_cell_halo = self._grid.end_index(cell_domain(h_grid.Zone.HALO))

        edge_domain = h_grid.domain(dims.EdgeDim)
        self._start_edge_lateral_boundary_level_5 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._end_edge_halo = self._grid.end_index(edge_domain(h_grid.Zone.HALO))

        allocator = model_backends.get_allocator(self._backend)
        self._lvn_sys_pos = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, dtype=bool, allocator=allocator
        )
        self._p_cell_idx = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, dtype=gtx.int32, allocator=allocator
        )
        self._p_cell_rel_idx_dsl = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, dtype=gtx.int32, allocator=allocator
        )
        self._p_cell_blk = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, dtype=gtx.int32, allocator=allocator
        )
        self._dreg_coords = {
            f"p_coords_dreg_v_{v}_{c}_dsl": data_alloc.zero_field(
                self._grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat, allocator=allocator
            )
            for v in (1, 2, 3, 4)
            for c in ("lon", "lat")
        }
        self._quad_vector_sums = {
            f"p_quad_vector_sum_{q}": data_alloc.zero_field(
                self._grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat, allocator=allocator
            )
            for q in (1, 2, 3, 4, 5, 6)
        }
        self._p_coeffs = {
            f"p_coeff_{c}_dsl": data_alloc.zero_field(
                self._grid, dims.CellDim, dims.KDim, allocator=allocator
            )
            for c in (1, 2, 3, 4, 5, 6)
        }

        e2c_table = self._grid.get_connectivity("E2C").asnumpy()
        cell_idx = gtx.as_field(
            (dims.EdgeDim, dims.E2CDim),
            e2c_table.astype(gtx.int32),  # type: ignore [arg-type] # type "ndarray[Any, Any] | NDArrayObject"; expected "NDArrayObject"
            allocator=allocator,
        )
        cell_blk = gtx.as_field(
            (dims.EdgeDim, dims.E2CDim),
            (0 * e2c_table).astype(gtx.int32),  # type: ignore [arg-type] # type "ndarray[Any, Any] | NDArrayObject"; expected "NDArrayObject"
            allocator=allocator,
        )

        edge_sizes = {
            "horizontal_start": self._start_edge_lateral_boundary_level_5,
            "horizontal_end": self._end_edge_halo,
        }
        vertical_sizes = {
            "vertical_start": gtx.int32(0),
            "vertical_end": gtx.int32(self._grid.num_levels),
        }
        self._compute_ffsl_backtrajectory_counterclockwise_indicator = model_options.setup_program(
            backend=self._backend,
            program=compute_ffsl_backtrajectory_counterclockwise_indicator,
            constant_args={
                "tangent_orientation": self._quadratic_state.tangent_orientation,
                # miura3 calls btraj_dreg with lcounterclock=.TRUE. (f90 4593)
                "lcounterclock": True,
            },
            horizontal_sizes=edge_sizes,
            vertical_sizes=vertical_sizes,
            offset_provider=self._grid.connectivities,
        )
        self._compute_ffsl_backtrajectory = model_options.setup_program(
            backend=self._backend,
            program=compute_ffsl_backtrajectory,
            constant_args={
                "cell_idx": cell_idx,
                "cell_blk": cell_blk,
                "edge_verts_1_x": self._quadratic_state.edge_verts_1_x,
                "edge_verts_2_x": self._quadratic_state.edge_verts_2_x,
                "edge_verts_1_y": self._quadratic_state.edge_verts_1_y,
                "edge_verts_2_y": self._quadratic_state.edge_verts_2_y,
                "pos_on_tplane_e_1_x": self._quadratic_state.pos_on_tplane_e_1_x,
                "pos_on_tplane_e_2_x": self._quadratic_state.pos_on_tplane_e_2_x,
                "pos_on_tplane_e_1_y": self._quadratic_state.pos_on_tplane_e_1_y,
                "pos_on_tplane_e_2_y": self._quadratic_state.pos_on_tplane_e_2_y,
                "primal_normal_cell_x": self._quadratic_state.primal_normal_cell_x,
                "primal_normal_cell_y": self._quadratic_state.primal_normal_cell_y,
                "dual_normal_cell_x": self._quadratic_state.dual_normal_cell_x,
                "dual_normal_cell_y": self._quadratic_state.dual_normal_cell_y,
            },
            horizontal_sizes=edge_sizes,
            vertical_sizes=vertical_sizes,
            offset_provider=self._grid.connectivities,
        )
        self._prepare_gauss_quadrature_quadratic_miura3 = model_options.setup_program(
            backend=self._backend,
            program=prepare_gauss_quadrature_quadratic_miura3,
            constant_args=_gauss_legendre_o2_quadrature_args(),
            horizontal_sizes=edge_sizes,
            vertical_sizes=vertical_sizes,
            offset_provider=self._grid.connectivities,
        )
        self._reconstruct_quadratic_coefficients_svd = model_options.setup_program(
            backend=self._backend,
            program=reconstruct_quadratic_coefficients_svd,
            constant_args={
                "lsq_moments_1": self._quadratic_state.lsq_moments_1,
                "lsq_moments_2": self._quadratic_state.lsq_moments_2,
                "lsq_moments_3": self._quadratic_state.lsq_moments_3,
                "lsq_moments_4": self._quadratic_state.lsq_moments_4,
                "lsq_moments_5": self._quadratic_state.lsq_moments_5,
                **{
                    f"lsq_pseudoinv_direct_{u + 1}": self._quadratic_state.lsq_pseudoinv_direct[u]
                    for u in range(5)
                },
                **{
                    f"lsq_pseudoinv_butterfly_{u + 1}": (
                        self._quadratic_state.lsq_pseudoinv_butterfly[u]
                    )
                    for u in range(5)
                },
            },
            horizontal_sizes={
                "horizontal_start": self._start_cell_lateral_boundary_level_2,
                "horizontal_end": self._end_cell_halo,
            },
            vertical_sizes=vertical_sizes,
            offset_provider=self._grid.connectivities,
        )
        self._compute_horizontal_tracer_flux_from_quadratic_coefficients = (
            model_options.setup_program(
                backend=self._backend,
                program=compute_horizontal_tracer_flux_from_quadratic_coefficients,
                horizontal_sizes=edge_sizes,
                vertical_sizes=vertical_sizes,
                offset_provider=self._grid.connectivities,
            )
        )

    def compute_tracer_flux(
        self,
        *,
        prep_adv: adv_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        p_distv_bary_1: fa.EdgeKField[ta.anyfloat],
        p_distv_bary_2: fa.EdgeKField[ta.anyfloat],
        p_vt: fa.EdgeKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        # p_distv_bary_* are unused: miura3 integrates over the full departure region
        log.debug("horizontal tracer flux computation - start")

        log.debug("running stencil compute_ffsl_backtrajectory_counterclockwise_indicator - start")
        self._compute_ffsl_backtrajectory_counterclockwise_indicator(
            p_vn=prep_adv.vn_traj,
            lvn_sys_pos=self._lvn_sys_pos,
        )
        log.debug("running stencil compute_ffsl_backtrajectory_counterclockwise_indicator - end")

        # departure regions swept over the full time step (btraj_dreg, f90 4593)
        log.debug("running stencil compute_ffsl_backtrajectory - start")
        self._compute_ffsl_backtrajectory(
            p_vn=prep_adv.vn_traj,
            p_vt=p_vt,
            lvn_sys_pos=self._lvn_sys_pos,
            p_cell_idx=self._p_cell_idx,
            p_cell_rel_idx_dsl=self._p_cell_rel_idx_dsl,
            p_cell_blk=self._p_cell_blk,
            **self._dreg_coords,
            p_dt=dtime,
        )
        log.debug("running stencil compute_ffsl_backtrajectory - end")

        # quadrature vector = departure region area averages of the monomials (f90 4607)
        log.debug("running stencil prepare_gauss_quadrature_quadratic_miura3 - start")
        self._prepare_gauss_quadrature_quadratic_miura3(
            **{
                f"p_coords_dreg_v_{v}_{xy}": self._dreg_coords[f"p_coords_dreg_v_{v}_{lonlat}_dsl"]
                for v in (1, 2, 3, 4)
                for xy, lonlat in (("x", "lon"), ("y", "lat"))
            },
            **self._quad_vector_sums,
        )
        log.debug("running stencil prepare_gauss_quadrature_quadratic_miura3 - end")

        # conservative quadratic fit per cell (recon_lsq_cell_q_svd, f90 4632)
        log.debug("running stencil reconstruct_quadratic_coefficients_svd - start")
        self._reconstruct_quadratic_coefficients_svd(
            p_cc=p_tracer_now,
            **self._p_coeffs,
        )
        log.debug("running stencil reconstruct_quadratic_coefficients_svd - end")

        log.debug(
            "running stencil compute_horizontal_tracer_flux_from_quadratic_coefficients - start"
        )
        self._compute_horizontal_tracer_flux_from_quadratic_coefficients(
            **{f"p_coeff_{c}": self._p_coeffs[f"p_coeff_{c}_dsl"] for c in (1, 2, 3, 4, 5, 6)},
            p_cell_rel_idx_dsl=self._p_cell_rel_idx_dsl,
            **self._quad_vector_sums,
            p_mass_flx_e=prep_adv.mass_flx_me,
            p_out_e=p_mflx_tracer_h,
        )
        log.debug(
            "running stencil compute_horizontal_tracer_flux_from_quadratic_coefficients - end"
        )

        self._horizontal_limiter.apply_flux_limiter(
            p_tracer_now=p_tracer_now,
            p_mflx_tracer_h=p_mflx_tracer_h,
            p_mass_flx_e=prep_adv.mass_flx_me,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            dtime=dtime,
        )

        log.debug("horizontal tracer flux computation - end")


class ThirdOrderMiuraWeno(SemiLagrangianTracerFlux):
    """Miura-based third-order tracer flux with quadratic 27-candidate WENO blending (ihadv_tracer=103).

    Port of upwind_hflux_miura3_weno (mo_advection_hflux.f90 2033-2620), live
    path only: quadratic reconstruction, SVD, l_out_edgeval=.FALSE. The
    departure regions and the quadrature vector are recomputed on every call;
    ICON shares them across tracers under ld_compute.
    TODO(jcanton): hoist the geometry to a tracer-independent step if more
    than one tracer is advected.
    """

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        weno_quadratic_state: tracer_advection_states.AdvectionWenoQuadraticState,
        backend: gtx.typing.Backend | None,
        horizontal_limiter: HorizontalFluxLimiter | None = None,
    ):
        self._grid = grid
        self._weno_quadratic_state = weno_quadratic_state
        self._backend = backend
        self._horizontal_limiter = horizontal_limiter or NoLimiter()

        # cell indices; the Fortran reconstructs from start_blk(3,1) to min_rlcell_int
        # (f90 2367-2368), here the SecondOrderMiuraWeno zones are kept: on the
        # boundary-free single-rank torus (the only supported configuration, see
        # the driver state construction) all cell zones coincide anyway
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_lateral_boundary_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._end_cell_halo = self._grid.end_index(cell_domain(h_grid.Zone.HALO))

        # edge indices (i_rlstart=5, f90 2194-2198)
        edge_domain = h_grid.domain(dims.EdgeDim)
        self._start_edge_lateral_boundary_level_5 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._end_edge_halo = self._grid.end_index(edge_domain(h_grid.Zone.HALO))

        # backtrajectory fields
        allocator = model_backends.get_allocator(self._backend)
        self._lvn_sys_pos = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, dtype=bool, allocator=allocator
        )
        self._p_cell_idx = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, dtype=gtx.int32, allocator=allocator
        )
        self._p_cell_rel_idx_dsl = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, dtype=gtx.int32, allocator=allocator
        )
        self._p_cell_blk = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, dtype=gtx.int32, allocator=allocator
        )
        self._dreg_coords = {
            f"p_coords_dreg_v_{v}_{c}_dsl": data_alloc.zero_field(
                self._grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat, allocator=allocator
            )
            for v in (1, 2, 3, 4)
            for c in ("lon", "lat")
        }

        # quadrature fields
        self._quad_vector_sums = {
            f"p_quad_vector_sum_{q}": data_alloc.zero_field(
                self._grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat, allocator=allocator
            )
            for q in (1, 2, 3, 4, 5, 6)
        }

        # candidate reconstruction fields
        self._p_coeffs = {
            f"p_coeff_{c}_dsl": data_alloc.zero_field(
                self._grid, dims.CellDim, dims.KDim, allocator=allocator
            )
            for c in (1, 2, 3, 4, 5, 6)
        }

        # WENO accumulator fields
        self._z_lsq_weighted = {
            f"z_lsq_weighted_{q}": data_alloc.zero_field(
                self._grid, dims.EdgeDim, dims.KDim, allocator=allocator
            )
            for q in (1, 2, 3, 4, 5, 6)
        }
        self._smooth_sum = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, allocator=allocator
        )

        # E2C line indices for the backtrajectory; blocks are zero in icon4py
        e2c_table = self._grid.get_connectivity("E2C").asnumpy()
        cell_idx = gtx.as_field(
            (dims.EdgeDim, dims.E2CDim),
            e2c_table.astype(gtx.int32),  # type: ignore [arg-type] # type "ndarray[Any, Any] | NDArrayObject"; expected "NDArrayObject"
            allocator=allocator,
        )
        cell_blk = gtx.as_field(
            (dims.EdgeDim, dims.E2CDim),
            (0 * e2c_table).astype(gtx.int32),  # type: ignore [arg-type] # type "ndarray[Any, Any] | NDArrayObject"; expected "NDArrayObject"
            allocator=allocator,
        )

        # stencils
        edge_sizes = {
            "horizontal_start": self._start_edge_lateral_boundary_level_5,
            "horizontal_end": self._end_edge_halo,
        }
        vertical_sizes = {
            "vertical_start": gtx.int32(0),
            "vertical_end": gtx.int32(self._grid.num_levels),
        }
        self._compute_ffsl_backtrajectory_counterclockwise_indicator = model_options.setup_program(
            backend=self._backend,
            program=compute_ffsl_backtrajectory_counterclockwise_indicator,
            constant_args={
                "tangent_orientation": self._weno_quadratic_state.tangent_orientation,
                # miura3 calls btraj_dreg with lcounterclock=.TRUE. (f90 2260-2265)
                "lcounterclock": True,
            },
            horizontal_sizes=edge_sizes,
            vertical_sizes=vertical_sizes,
            offset_provider=self._grid.connectivities,
        )
        self._compute_ffsl_backtrajectory = model_options.setup_program(
            backend=self._backend,
            program=compute_ffsl_backtrajectory,
            constant_args={
                "cell_idx": cell_idx,
                "cell_blk": cell_blk,
                "edge_verts_1_x": self._weno_quadratic_state.edge_verts_1_x,
                "edge_verts_2_x": self._weno_quadratic_state.edge_verts_2_x,
                "edge_verts_1_y": self._weno_quadratic_state.edge_verts_1_y,
                "edge_verts_2_y": self._weno_quadratic_state.edge_verts_2_y,
                "pos_on_tplane_e_1_x": self._weno_quadratic_state.pos_on_tplane_e_1_x,
                "pos_on_tplane_e_2_x": self._weno_quadratic_state.pos_on_tplane_e_2_x,
                "pos_on_tplane_e_1_y": self._weno_quadratic_state.pos_on_tplane_e_1_y,
                "pos_on_tplane_e_2_y": self._weno_quadratic_state.pos_on_tplane_e_2_y,
                "primal_normal_cell_x": self._weno_quadratic_state.primal_normal_cell_x,
                "primal_normal_cell_y": self._weno_quadratic_state.primal_normal_cell_y,
                "dual_normal_cell_x": self._weno_quadratic_state.dual_normal_cell_x,
                "dual_normal_cell_y": self._weno_quadratic_state.dual_normal_cell_y,
            },
            horizontal_sizes=edge_sizes,
            vertical_sizes=vertical_sizes,
            offset_provider=self._grid.connectivities,
        )
        self._prepare_gauss_quadrature_quadratic_miura3 = model_options.setup_program(
            backend=self._backend,
            program=prepare_gauss_quadrature_quadratic_miura3,
            constant_args=_gauss_legendre_o2_quadrature_args(),
            horizontal_sizes=edge_sizes,
            vertical_sizes=vertical_sizes,
            offset_provider=self._grid.connectivities,
        )
        self._init_constant_edge_kdim_field = model_options.setup_program(
            backend=self._backend,
            program=init_constant_edge_kdim_field,
            horizontal_sizes=edge_sizes,
            vertical_sizes=vertical_sizes,
            offset_provider=self._grid.connectivities,
        )
        self._reconstruct_quadratic_coefficients_svd = model_options.setup_program(
            backend=self._backend,
            program=reconstruct_quadratic_coefficients_svd,
            constant_args={
                "lsq_moments_1": self._weno_quadratic_state.lsq_moments_1,
                "lsq_moments_2": self._weno_quadratic_state.lsq_moments_2,
                "lsq_moments_3": self._weno_quadratic_state.lsq_moments_3,
                "lsq_moments_4": self._weno_quadratic_state.lsq_moments_4,
                "lsq_moments_5": self._weno_quadratic_state.lsq_moments_5,
            },
            horizontal_sizes={
                "horizontal_start": self._start_cell_lateral_boundary_level_2,
                "horizontal_end": self._end_cell_halo,
            },
            vertical_sizes=vertical_sizes,
            offset_provider=self._grid.connectivities,
        )
        self._accumulate_weno_candidate_flux_weights = model_options.setup_program(
            backend=self._backend,
            program=accumulate_weno_candidate_flux_weights,
            constant_args={
                "cell_area": self._weno_quadratic_state.cell_area,
            },
            horizontal_sizes=edge_sizes,
            vertical_sizes=vertical_sizes,
            offset_provider=self._grid.connectivities,
        )
        self._compute_horizontal_tracer_flux_from_weno_coefficients = model_options.setup_program(
            backend=self._backend,
            program=compute_horizontal_tracer_flux_from_weno_coefficients,
            horizontal_sizes=edge_sizes,
            vertical_sizes=vertical_sizes,
            offset_provider=self._grid.connectivities,
        )

    def compute_tracer_flux(
        self,
        *,
        prep_adv: adv_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        p_distv_bary_1: fa.EdgeKField[ta.anyfloat],
        p_distv_bary_2: fa.EdgeKField[ta.anyfloat],
        p_vt: fa.EdgeKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        # p_distv_bary_* are unused: miura3 integrates over the full departure region
        log.debug("horizontal tracer flux computation - start")

        # counterclockwise indicator (mo_advection_traj.f90 527-537)
        log.debug("running stencil compute_ffsl_backtrajectory_counterclockwise_indicator - start")
        self._compute_ffsl_backtrajectory_counterclockwise_indicator(
            p_vn=prep_adv.vn_traj,
            lvn_sys_pos=self._lvn_sys_pos,
        )
        log.debug("running stencil compute_ffsl_backtrajectory_counterclockwise_indicator - end")

        # departure regions swept over the full time step (btraj_dreg, f90 2260-2265)
        log.debug("running stencil compute_ffsl_backtrajectory - start")
        self._compute_ffsl_backtrajectory(
            p_vn=prep_adv.vn_traj,
            p_vt=p_vt,
            lvn_sys_pos=self._lvn_sys_pos,
            p_cell_idx=self._p_cell_idx,
            p_cell_rel_idx_dsl=self._p_cell_rel_idx_dsl,
            p_cell_blk=self._p_cell_blk,
            **self._dreg_coords,
            p_dt=dtime,
        )
        log.debug("running stencil compute_ffsl_backtrajectory - end")

        # quadrature vector = departure region area averages of the monomials
        log.debug("running stencil prepare_gauss_quadrature_quadratic_miura3 - start")
        self._prepare_gauss_quadrature_quadratic_miura3(
            **{
                f"p_coords_dreg_v_{v}_{xy}": self._dreg_coords[f"p_coords_dreg_v_{v}_{lonlat}_dsl"]
                for v in (1, 2, 3, 4)
                for xy, lonlat in (("x", "lon"), ("y", "lat"))
            },
            **self._quad_vector_sums,
        )
        log.debug("running stencil prepare_gauss_quadrature_quadratic_miura3 - end")

        # zero the WENO accumulators (f90 2450-2451)
        for accumulator in (*self._z_lsq_weighted.values(), self._smooth_sum):
            self._init_constant_edge_kdim_field(field=accumulator, value=0.0)

        # 27-candidate loop (f90 2458-2512); candidate reconstruction on cells,
        # then smoothness-weighted accumulation on edges
        for cand in range(27):
            direct = self._weno_quadratic_state.lsq_pseudoinv_direct[cand]
            butterfly = self._weno_quadratic_state.lsq_pseudoinv_butterfly[cand]
            self._reconstruct_quadratic_coefficients_svd(
                p_cc=p_tracer_now,
                **{f"lsq_pseudoinv_direct_{u + 1}": direct[u] for u in range(5)},
                **{f"lsq_pseudoinv_butterfly_{u + 1}": butterfly[u] for u in range(5)},
                **self._p_coeffs,
            )
            self._accumulate_weno_candidate_flux_weights(
                **{f"p_coeff_{c}": self._p_coeffs[f"p_coeff_{c}_dsl"] for c in (1, 2, 3, 4, 5, 6)},
                p_cell_rel_idx_dsl=self._p_cell_rel_idx_dsl,
                **{
                    f"z_quad_vector_sum_{q}": self._quad_vector_sums[f"p_quad_vector_sum_{q}"]
                    for q in (1, 2, 3, 4, 5, 6)
                },
                **self._z_lsq_weighted,
                smooth_sum=self._smooth_sum,
                l_weight_s=float(weno_least_squares.L_WEIGHTS_S[cand]),
            )

        # normalize and compute the flux (f90 2513-2521)
        log.debug("running stencil compute_horizontal_tracer_flux_from_weno_coefficients - start")
        self._compute_horizontal_tracer_flux_from_weno_coefficients(
            **self._z_lsq_weighted,
            smooth_sum=self._smooth_sum,
            **self._quad_vector_sums,
            p_mass_flx_e=prep_adv.mass_flx_me,
            p_out_e=p_mflx_tracer_h,
        )
        log.debug("running stencil compute_horizontal_tracer_flux_from_weno_coefficients - end")

        self._horizontal_limiter.apply_flux_limiter(
            p_tracer_now=p_tracer_now,
            p_mflx_tracer_h=p_mflx_tracer_h,
            p_mass_flx_e=prep_adv.mass_flx_me,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            dtime=dtime,
        )

        log.debug("horizontal tracer flux computation - end")


class HorizontalAdvection(ABC):
    """Class that does one horizontal tracer_advection step."""

    @abstractmethod
    def run(
        self,
        *,
        prep_adv: adv_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        """
        Run a horizontal tracer_advection step.

        Args:
            prep_adv: input argument, data class that contains precalculated tracer_advection fields
            p_tracer_now: input argument, field that contains current tracer mass fraction
            p_tracer_new: output argument, field that contains new tracer mass fraction
            rhodz_now: input argument, field that contains current air mass in each layer
            rhodz_new: input argument, field that contains new air mass in each layer
            p_mflx_tracer_h: output argument, field that contains new horizontal tracer mass flux
            dtime: input argument, the time step

        """
        ...


class NoAdvection(HorizontalAdvection):
    """Class that implements disabled horizontal tracer_advection."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        backend: gtx.typing.Backend | None,
    ):
        log.debug("horizontal tracer_advection class init - start")

        # input arguments
        self._backend = model_options.customize_backend(program=None, backend=backend)

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_nudging = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        # stencils
        self._copy_cell_kdim_field = model_options.setup_program(
            backend=self._backend,
            program=copy_cell_kdim_field,
            horizontal_sizes={
                "horizontal_start": self._start_cell_nudging,
                "horizontal_end": self._end_cell_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(grid.num_levels),
            },
            offset_provider=grid.connectivities,
        )

        log.debug("horizontal tracer_advection class init - end")

    def run(
        self,
        *,
        prep_adv: adv_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        log.debug("horizontal tracer_advection run - start")

        log.debug("running stencil copy_cell_kdim_field - start")
        self._copy_cell_kdim_field(
            field_in=p_tracer_now,
            field_out=p_tracer_new,
        )
        log.debug("running stencil copy_cell_kdim_field - end")
        log.debug("horizontal tracer_advection run - end")


class FiniteVolume(HorizontalAdvection):
    """Class that defines a finite volume horizontal tracer_advection scheme."""

    def run(
        self,
        *,
        prep_adv: adv_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        log.debug("horizontal tracer_advection run - start")

        self._compute_numerical_flux(
            prep_adv=prep_adv,
            p_tracer_now=p_tracer_now,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            p_mflx_tracer_h=p_mflx_tracer_h,
            dtime=dtime,
        )

        self._update_unknowns(
            p_tracer_now=p_tracer_now,
            p_tracer_new=p_tracer_new,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            p_mflx_tracer_h=p_mflx_tracer_h,
            dtime=dtime,
        )
        log.debug("horizontal tracer_advection run - end")

    @abstractmethod
    def _compute_numerical_flux(
        self,
        *,
        prep_adv: adv_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None: ...

    @abstractmethod
    def _update_unknowns(
        self,
        *,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None: ...


class FirstOrderUpwind(FiniteVolume):
    """Class that does one horizontal first-order accurate upwind finite volume advection step."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        interpolation_state: tracer_advection_states.AdvectionInterpolationState,
        metric_state: tracer_advection_states.AdvectionMetricState,
        backend: gtx.typing.Backend | None,
    ):
        log.debug("horizontal advection class init - start")

        self._grid = grid
        self._interpolation_state = interpolation_state
        self._metric_state = metric_state
        self._backend = backend

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        # edge indices
        edge_domain = h_grid.domain(dims.EdgeDim)
        self._start_edge_lateral_boundary_level_5 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._end_edge_halo = self._grid.end_index(edge_domain(h_grid.Zone.HALO))

        # stencils
        self._compute_horizontal_tracer_flux_upwind = model_options.setup_program(
            backend=self._backend,
            program=compute_horizontal_tracer_flux_upwind,
            horizontal_sizes={
                "horizontal_start": self._start_edge_lateral_boundary_level_5,
                "horizontal_end": self._end_edge_halo,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(grid.num_levels),
            },
            offset_provider=grid.connectivities,
        )
        self._integrate_tracer_horizontally = model_options.setup_program(
            backend=self._backend,
            program=integrate_tracer_horizontally,
            constant_args={
                "deepatmo_divh": self._metric_state.deepatmo_divh,
                "geofac_div": self._interpolation_state.geofac_div,
            },
            horizontal_sizes={
                "horizontal_start": self._start_cell_nudging,
                "horizontal_end": self._end_cell_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(grid.num_levels),
            },
            offset_provider=grid.connectivities,
        )

        log.debug("horizontal advection class init - end")

    def _compute_numerical_flux(
        self,
        *,
        prep_adv: adv_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        log.debug("horizontal numerical flux computation - start")

        log.debug("running stencil compute_horizontal_tracer_flux_upwind - start")
        self._compute_horizontal_tracer_flux_upwind(
            p_cc=p_tracer_now,
            p_mass_flx_e=prep_adv.mass_flx_me,
            p_vn=prep_adv.vn_traj,
            p_out_e=p_mflx_tracer_h,
        )
        log.debug("running stencil compute_horizontal_tracer_flux_upwind - end")

        log.debug("horizontal numerical flux computation - end")

    def _update_unknowns(
        self,
        *,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        log.debug("horizontal unknowns update - start")

        # update tracer mass fraction
        log.debug("running stencil integrate_tracer_horizontally - start")
        self._integrate_tracer_horizontally(
            p_mflx_tracer_h=p_mflx_tracer_h,
            tracer_now=p_tracer_now,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            tracer_new_hor=p_tracer_new,
            p_dtime=dtime,
        )
        log.debug("running stencil integrate_tracer_horizontally - end")

        log.debug("horizontal unknowns update - end")


class SemiLagrangian(FiniteVolume):
    """Class that does one horizontal semi-Lagrangian finite volume tracer_advection step."""

    def __init__(
        self,
        *,
        tracer_flux: SemiLagrangianTracerFlux,
        grid: icon_grid.IconGrid,
        interpolation_state: tracer_advection_states.AdvectionInterpolationState,
        metric_state: tracer_advection_states.AdvectionMetricState,
        edge_params: grid_states.EdgeParams,
        cell_params: grid_states.CellParams,
        backend: gtx.typing.Backend | None,
    ):
        log.debug("horizontal tracer_advection class init - start")

        # input arguments
        self._tracer_flux = tracer_flux
        self._grid = grid
        self._interpolation_state = interpolation_state
        self._metric_state = metric_state
        self._edge_params = edge_params
        self._backend = backend

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        # edge indices
        edge_domain = h_grid.domain(dims.EdgeDim)
        self._start_edge_lateral_boundary_level_2 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._start_edge_lateral_boundary_level_5 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._end_edge_halo = self._grid.end_index(edge_domain(h_grid.Zone.HALO))

        # backtrajectory fields
        allocator = model_backends.get_allocator(self._backend)
        self._z_real_vt = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, allocator=allocator
        )
        self._p_distv_bary_1 = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, allocator=allocator
        )
        self._p_distv_bary_2 = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, allocator=allocator
        )

        # stencils
        self._compute_edge_tangential = model_options.setup_program(
            backend=self._backend,
            program=compute_edge_tangential,
            constant_args={
                "ptr_coeff": self._interpolation_state.rbf_vec_coeff_e,
            },
            horizontal_sizes={
                "horizontal_start": self._start_edge_lateral_boundary_level_2,
                "horizontal_end": self._end_edge_halo,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )

        self._compute_barycentric_backtrajectory_alt = model_options.setup_program(
            backend=self._backend,
            program=compute_barycentric_backtrajectory_alt,
            constant_args={
                "pos_on_tplane_e_1": self._interpolation_state.pos_on_tplane_e_1,
                "pos_on_tplane_e_2": self._interpolation_state.pos_on_tplane_e_2,
                "primal_normal_cell_1": self._edge_params.primal_normal_cell[0],
                "dual_normal_cell_1": self._edge_params.dual_normal_cell[0],
                "primal_normal_cell_2": self._edge_params.primal_normal_cell[1],
                "dual_normal_cell_2": self._edge_params.dual_normal_cell[1],
            },
            horizontal_sizes={
                "horizontal_start": self._start_edge_lateral_boundary_level_5,
                "horizontal_end": self._end_edge_halo,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": self._grid.num_levels,
            },
            offset_provider=self._grid.connectivities,
        )
        self._integrate_tracer_horizontally = model_options.setup_program(
            backend=self._backend,
            program=integrate_tracer_horizontally,
            constant_args={
                "deepatmo_divh": self._metric_state.deepatmo_divh,
                "geofac_div": self._interpolation_state.geofac_div,
            },
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

        log.debug("horizontal tracer_advection class init - end")

    def _compute_numerical_flux(
        self,
        *,
        prep_adv: adv_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        log.debug("horizontal numerical flux computation - start")

        ## tracer-independent part

        # compute tangential velocity
        log.debug("running stencil compute_edge_tangential - start")
        self._compute_edge_tangential(
            p_vn_in=prep_adv.vn_traj,
            p_vt_out=self._z_real_vt,
        )
        log.debug("running stencil compute_edge_tangential - end")

        # backtrajectory calculation
        log.debug("running stencil compute_barycentric_backtrajectory_alt - start")
        self._compute_barycentric_backtrajectory_alt(
            p_vn=prep_adv.vn_traj,
            p_vt=self._z_real_vt,
            p_distv_bary_1=self._p_distv_bary_1,
            p_distv_bary_2=self._p_distv_bary_2,
            p_dthalf=0.5 * dtime,
        )
        log.debug("running stencil compute_barycentric_backtrajectory_alt - end")

        ## tracer-specific part

        self._tracer_flux.compute_tracer_flux(
            prep_adv=prep_adv,
            p_tracer_now=p_tracer_now,
            p_mflx_tracer_h=p_mflx_tracer_h,
            p_distv_bary_1=self._p_distv_bary_1,
            p_distv_bary_2=self._p_distv_bary_2,
            p_vt=self._z_real_vt,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            dtime=dtime,
        )

        log.debug("horizontal numerical flux computation - end")

    def _update_unknowns(
        self,
        *,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        log.debug("horizontal unknowns update - start")

        # update tracer mass fraction
        log.debug("running stencil integrate_tracer_horizontally - start")
        self._integrate_tracer_horizontally(
            p_mflx_tracer_h=p_mflx_tracer_h,
            tracer_now=p_tracer_now,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            tracer_new_hor=p_tracer_new,
            p_dtime=dtime,
        )
        log.debug("running stencil integrate_tracer_horizontally - end")

        log.debug("horizontal unknowns update - end")
