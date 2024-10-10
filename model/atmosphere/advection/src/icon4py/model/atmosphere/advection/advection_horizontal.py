# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
import logging

from icon4py.model.atmosphere.advection import advection_states
from icon4py.model.atmosphere.advection.stencils import (
    apply_positive_definite_horizontal_multiplicative_flux_factor,
    compute_barycentric_backtrajectory_alt,
    compute_edge_tangential,
    compute_horizontal_tracer_flux_from_linear_coefficients_alt,
    compute_positive_definite_horizontal_multiplicative_flux_factor,
    copy_cell_kdim_field,
    integrate_tracer_horizontally,
    reconstruct_linear_coefficients_svd,
)
from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, geometry
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc


"""
Advection components related to horizontal transport.
"""

# flake8: noqa
log = logging.getLogger(__name__)


class HorizontalFluxLimiter:
    """Class that limits the horizontal finite volume numerical flux."""

    def apply_flux_limiter(
        self,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ):
        ...


class PositiveDefiniteLimiter(HorizontalFluxLimiter):
    """Class that implements a positive definite horizontal flux limiter."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        interpolation_state: advection_states.AdvectionInterpolationState,
        exchange: decomposition.ExchangeRuntime = decomposition.SingleNodeExchange(),
    ):
        self._grid = grid
        self._interpolation_state = interpolation_state
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
        self._r_m = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=self._grid)

    def apply_flux_limiter(
        self,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ):
        # compute multiplicative flux factor to guarantee no undershoot
        log.debug(
            "running stencil compute_positive_definite_horizontal_multiplicative_flux_factor - start"
        )
        compute_positive_definite_horizontal_multiplicative_flux_factor.compute_positive_definite_horizontal_multiplicative_flux_factor(
            geofac_div=self._interpolation_state.geofac_div,
            p_cc=p_tracer_now,
            p_rhodz_now=rhodz_now,
            p_mflx_tracer_h=p_mflx_tracer_h,
            r_m=self._r_m,
            p_dtime=dtime,
            dbl_eps=constants.DBL_EPS,
            horizontal_start=self._start_cell_lateral_boundary_level_2,  # originally i_rlstart_c = get_startrow_c(startrow_e=5) = 2
            horizontal_end=self._end_cell_local,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug(
            "running stencil compute_positive_definite_horizontal_multiplicative_flux_factor - end"
        )

        log.debug("communication of advection cell field: r_m - start")
        self._exchange.exchange_and_wait(dims.CellDim, self._r_m)
        log.debug("communication of advection cell field: r_m - end")

        # limit outward fluxes
        log.debug(
            "running stencil apply_positive_definite_horizontal_multiplicative_flux_factor - start"
        )
        apply_positive_definite_horizontal_multiplicative_flux_factor.apply_positive_definite_horizontal_multiplicative_flux_factor(
            r_m=self._r_m,
            p_mflx_tracer_h=p_mflx_tracer_h,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug(
            "running stencil apply_positive_definite_horizontal_multiplicative_flux_factor - end"
        )


class HorizontalFlux(ABC):
    """Class that computes the horizontal finite volume numerical flux."""

    @abstractmethod
    def compute_numerical_flux(
        self,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        p_distv_bary_1: fa.EdgeKField[ta.vpfloat],
        p_distv_bary_2: fa.EdgeKField[ta.vpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ):
        ...


class SecondOrderMiuraHorizontal(HorizontalFlux):
    """Class that computes a Miura-based second-order accurate numerical flux."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        least_squares_state: advection_states.AdvectionLeastSquaresState,
        horizontal_limiter: HorizontalFluxLimiter = HorizontalFluxLimiter(),
    ):
        self._grid = grid
        self._least_squares_state = least_squares_state
        self._horizontal_limiter = horizontal_limiter

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
        self._p_coeff_1 = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=self._grid)
        self._p_coeff_2 = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=self._grid)
        self._p_coeff_3 = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=self._grid)

    def compute_numerical_flux(
        self,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        p_distv_bary_1: fa.EdgeKField[ta.vpfloat],
        p_distv_bary_2: fa.EdgeKField[ta.vpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ):
        # linear reconstruction using singular value decomposition
        log.debug("running stencil reconstruct_linear_coefficients_svd - start")
        reconstruct_linear_coefficients_svd.reconstruct_linear_coefficients_svd(
            p_cc=p_tracer_now,
            lsq_pseudoinv_1=self._least_squares_state.lsq_pseudoinv_1,
            lsq_pseudoinv_2=self._least_squares_state.lsq_pseudoinv_2,
            p_coeff_1_dsl=self._p_coeff_1,
            p_coeff_2_dsl=self._p_coeff_2,
            p_coeff_3_dsl=self._p_coeff_3,
            horizontal_start=self._start_cell_lateral_boundary_level_2,  # originally i_rlstart_c = get_startrow_c(startrow_e=5) = 2
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self._grid.num_levels,  # originally UBOUND(p_cc,2)
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil reconstruct_linear_coefficients_svd - end")

        # compute reconstructed tracer value at each barycenter and corresponding flux at each edge
        log.debug(
            "running stencil compute_horizontal_tracer_flux_from_linear_coefficients_alt - start"
        )
        compute_horizontal_tracer_flux_from_linear_coefficients_alt.compute_horizontal_tracer_flux_from_linear_coefficients_alt(
            z_lsq_coeff_1=self._p_coeff_1,
            z_lsq_coeff_2=self._p_coeff_2,
            z_lsq_coeff_3=self._p_coeff_3,
            distv_bary_1=p_distv_bary_1,
            distv_bary_2=p_distv_bary_2,
            p_mass_flx_e=prep_adv.mass_flx_me,
            p_vn=prep_adv.vn_traj,
            p_out_e=p_mflx_tracer_h,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug(
            "running stencil compute_horizontal_tracer_flux_from_linear_coefficients_alt - end"
        )

        self._horizontal_limiter.apply_flux_limiter(
            p_tracer_now=p_tracer_now,
            p_mflx_tracer_h=p_mflx_tracer_h,
            rhodz_now=rhodz_now,
            dtime=dtime,
        )


class HorizontalAdvection(ABC):
    """Class that does one horizontal advection step."""

    @abstractmethod
    def run(
        self,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ):
        """
        Run a horizontal advection step.

        Args:
            prep_adv: input argument, data class that contains precalculated advection fields
            p_tracer_now: input argument, field that contains current tracer mass fraction
            p_tracer_new: output argument, field that contains new tracer mass fraction
            rhodz_now: input argument, field that contains current air mass in each layer
            rhodz_new: input argument, field that contains new air mass in each layer
            p_mflx_tracer_h: output argument, field that contains new horizontal tracer mass flux
            dtime: input argument, the time step

        """
        ...


class NoHorizontalAdvection(HorizontalAdvection):
    """Class that implements disabled horizontal advection."""

    def __init__(self, grid: icon_grid.IconGrid):
        log.debug("horizontal advection class init - start")

        # input arguments
        self._grid = grid

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        log.debug("horizontal advection class init - end")

    def run(
        self,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ):
        log.debug("horizontal advection run - start")

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

        log.debug("horizontal advection run - end")


class SemiLagrangianHorizontalAdvection(HorizontalAdvection):
    """Class that does one semi-Lagrangian horizontal advection step."""

    def __init__(
        self,
        horizontal_flux: HorizontalFlux,
        grid: icon_grid.IconGrid,
        interpolation_state: advection_states.AdvectionInterpolationState,
        least_squares_state: advection_states.AdvectionLeastSquaresState,
        metric_state: advection_states.AdvectionMetricState,
        edge_params: geometry.EdgeParams,
        cell_params: geometry.CellParams,
        exchange: decomposition.ExchangeRuntime = decomposition.SingleNodeExchange(),
    ):
        log.debug("horizontal advection class init - start")

        # input arguments
        self._horizontal_flux = horizontal_flux
        self._grid = grid
        self._interpolation_state = interpolation_state
        self._least_squares_state = least_squares_state
        self._metric_state = metric_state
        self._edge_params = edge_params
        self._cell_params = cell_params
        self._exchange = exchange

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
        self._z_real_vt = field_alloc.allocate_zero_field(dims.EdgeDim, dims.KDim, grid=self._grid)
        self._p_distv_bary_1 = field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=self._grid, dtype=ta.vpfloat
        )
        self._p_distv_bary_2 = field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=self._grid, dtype=ta.vpfloat
        )

        log.debug("horizontal advection class init - end")

    def run(
        self,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ):
        log.debug("horizontal advection run - start")

        # get the horizontal numerical tracer flux
        self._compute_horizontal_tracer_flux(
            prep_adv=prep_adv,
            p_tracer_now=p_tracer_now,
            rhodz_now=rhodz_now,
            p_mflx_tracer_h=p_mflx_tracer_h,
            dtime=dtime,
        )

        # update tracer mass fraction
        log.debug("running stencil integrate_tracer_horizontally - start")
        integrate_tracer_horizontally.integrate_tracer_horizontally(
            p_mflx_tracer_h=p_mflx_tracer_h,
            deepatmo_divh=self._metric_state.deepatmo_divh,
            tracer_now=p_tracer_now,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            geofac_div=self._interpolation_state.geofac_div,
            tracer_new_hor=p_tracer_new,
            p_dtime=dtime,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil integrate_tracer_horizontally - end")

        log.debug("horizontal advection run - end")

    def _compute_horizontal_tracer_flux(
        self,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ):
        log.debug("horizontal tracer flux computation - start")

        ## tracer-independent part

        # compute tangential velocity
        log.debug("running stencil compute_edge_tangential - start")
        compute_edge_tangential.compute_edge_tangential(
            p_vn_in=prep_adv.vn_traj,
            ptr_coeff=self._interpolation_state.rbf_vec_coeff_e,
            p_vt_out=self._z_real_vt,
            horizontal_start=self._start_edge_lateral_boundary_level_2,
            horizontal_end=self._end_edge_halo,
            vertical_start=0,
            vertical_end=self._grid.num_levels,  # originally UBOUND(p_vn,2)
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil compute_edge_tangential - end")

        # backtrajectory calculation
        log.debug("running stencil compute_barycentric_backtrajectory_alt - start")
        compute_barycentric_backtrajectory_alt.compute_barycentric_backtrajectory_alt(
            p_vn=prep_adv.vn_traj,
            p_vt=self._z_real_vt,
            pos_on_tplane_e_1=self._interpolation_state.pos_on_tplane_e_1,
            pos_on_tplane_e_2=self._interpolation_state.pos_on_tplane_e_2,
            primal_normal_cell_1=self._edge_params.primal_normal_cell[0],
            dual_normal_cell_1=self._edge_params.dual_normal_cell[0],
            primal_normal_cell_2=self._edge_params.primal_normal_cell[1],
            dual_normal_cell_2=self._edge_params.dual_normal_cell[1],
            p_distv_bary_1=self._p_distv_bary_1,
            p_distv_bary_2=self._p_distv_bary_2,
            p_dthalf=0.5 * dtime,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil compute_barycentric_backtrajectory_alt - end")

        ## tracer-specific part

        self._horizontal_flux.compute_numerical_flux(
            prep_adv=prep_adv,
            p_tracer_now=p_tracer_now,
            p_mflx_tracer_h=p_mflx_tracer_h,
            p_distv_bary_1=self._p_distv_bary_1,
            p_distv_bary_2=self._p_distv_bary_2,
            rhodz_now=rhodz_now,
            dtime=dtime,
        )

        log.debug("horizontal tracer flux computation - end")