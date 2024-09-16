# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import enum
import logging
from icon4py.model.common.settings import xp

import gt4py.next as gtx

from icon4py.model.atmosphere.advection import advection_states
from icon4py.model.atmosphere.advection.stencils import (
    compute_positive_definite_horizontal_multiplicative_flux_factor,
    apply_positive_definite_horizontal_multiplicative_flux_factor,
    integrate_tracer_horizontally,
    compute_barycentric_backtrajectory,
    compute_edge_tangential,
    reconstruct_linear_coefficients_svd,
    apply_vertical_density_increment,
    apply_horizontal_density_increment,
    apply_interpolated_tracer_time_tendency,
    compute_horizontal_tracer_flux_from_linear_coefficients,
    copy_cell_kdim_field,
)
from icon4py.model.atmosphere.dycore import compute_tangential_wind
from icon4py.model.atmosphere.dycore.state_utils import states as solve_nh_states
from icon4py.model.common import constants, dimension as dims, field_type_aliases as fa
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, geometry
from icon4py.model.common.test_utils.helpers import (
    as_1D_sparse_field,
    constant_field,
    numpy_to_1D_sparse_field,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat
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

    NO_ADVECTION = 0  #: no horizontal advection
    LINEAR_2ND_ORDER = 2  #: 2nd order MIURA with linear reconstruction


class HorizontalAdvectionLimiterType(enum.Enum):
    """
    Limiter for horizontal advection operator.
    """

    #: no horizontal limiter
    NO_LIMITER = 0  
    POSITIVE_DEFINITE = 4  #: positive definite horizontal limiter


class VerticalAdvectionType(enum.Enum):
    """
    Vertical operator scheme for advection.
    """

    NO_ADVECTION = 0  #: no vertical advection


@dataclasses.dataclass(frozen = True) 
class AdvectionConfig:
    horizontal_advection_type: HorizontalAdvectionType
    horizontal_advection_limiter: HorizontalAdvectionLimiterType
    vertical_advection_type: VerticalAdvectionType
    
    
def __post_init__(self):
     self._validate()
    """
    Contains necessary parameters to configure an advection run.

    Default values match a basic implementation.
    """

    def __init__(
        self,
        horizontal_advection_type: HorizontalAdvectionType = HorizontalAdvectionType.LINEAR_2ND_ORDER,
        horizontal_advection_limiter: HorizontalAdvectionLimiterType = HorizontalAdvectionLimiterType.POSITIVE_DEFINITE,
        vertical_advection_type: VerticalAdvectionType = HorizontalAdvectionType.NO_ADVECTION,
    ):
        """Set the default values according to a basic implementation."""

        self.horizontal_advection_type: int = horizontal_advection_type
        self.horizontal_advection_limiter: int = horizontal_advection_limiter
        self.vertical_advection_type: int = vertical_advection_type

        self._validate()

    def _validate(self):
        """Apply consistency checks and validation on configuration parameters."""
        assert self.horizontal_advection_type not in HorizontalAdvectionType.__members__, f"vertical advection type {self.vertical_advection_type} not implemented"
              
            raise NotImplementedError(
                "Only horizontal advection type 2 = `2nd order MIURA with linear reconstruction` is implemented"
            )
        if self.horizontal_advection_limiter != 0 and self.horizontal_advection_limiter != 4:
            raise NotImplementedError(
                "Only horizontal advection limiter 4 = `positive definite limiter` is implemented"
            )
        if self.vertical_advection_type != 0:
            raise NotImplementedError(
                "Only vertical advection type 0 = `no vertical advection` is implemented"
            )


class Advection:
    """Class that configures advection and does one advection step."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        config: AdvectionConfig,
        interpolation_state: advection_states.AdvectionInterpolationState,
        least_squares_state: advection_states.AdvectionLeastSquaresState,
        metric_state: advection_states.AdvectionMetricState,
        edge_params: geometry.EdgeParams,
        cell_params: geometry.CellParams,
        exchange: decomposition.ExchangeRuntime = decomposition.SingleNodeExchange(),
    ):
        """
        Initialize advection granule with configuration.
        """
        log.debug("advection class init - start")

        # input arguments
        self.grid = grid
        self.config = config
        self.interpolation_state = interpolation_state
        self.least_squares_state = least_squares_state
        self.metric_state = metric_state
        self.edge_params = edge_params
        self.cell_params = cell_params
        self.exchange = exchange

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self.start_cell_lateral_boundary = self.grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY)
        )
        self.start_cell_lateral_boundary_level_2 = self.grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self.start_cell_lateral_boundary_level_3 = self.grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        self.end_cell_lateral_boundary_level_4 = self.grid.end_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
        )
        self.start_cell_nudging = self.grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self.end_cell_local = self.grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self.end_cell_halo = self.grid.end_index(cell_domain(h_grid.Zone.HALO))
        self.end_cell_end = self.grid.end_index(cell_domain(h_grid.Zone.END))

        # edge indices
        edge_domain = h_grid.domain(dims.EdgeDim)
        self.start_edge_lateral_boundary_level_2 = self.grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self.start_edge_lateral_boundary_level_5 = self.grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self.end_edge_halo = self.grid.end_index(edge_domain(h_grid.Zone.HALO))

        # fields
        self._allocate_temporary_fields()

        # misc
        self.even_timestep = False  # originally jstep_adv(:)%marchuk_order = 1 in integrate_nh

        log.debug("advection class init - end")

    def _allocate_temporary_fields(self):
        # density field
        self.rhodz_ast2 = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid
        )  # intermediate density times cell thickness, includes either the horizontal or vertical advective density increment [kg/m^2] (nproma,nlev,nblks_c)

        # backtrajectory fields
        self.z_real_vt = field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=self.grid
        )  # unweighted tangential velocity component at edges (nproma,nlev,nblks_e)
        self.cell_idx = numpy_to_1D_sparse_field(
            np.asarray(self.grid.connectivities[dims.E2CDim], dtype=int32), dims.ECDim
        )
        self.cell_blk = as_1D_sparse_field(
            constant_field(self.grid, 1, dims.EdgeDim, dims.E2CDim, dtype=int32), dims.ECDim
        )
        self.p_cell_idx = field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=self.grid, dtype=int32
        )
        self.p_cell_rel_idx = field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=self.grid, dtype=int32
        )
        self.p_cell_blk = field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=self.grid, dtype=int32
        )
        self.p_distv_bary_1 = field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=self.grid, dtype=vpfloat
        )
        self.p_distv_bary_2 = field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=self.grid, dtype=vpfloat
        )

        # reconstruction fields
        self.p_coeff_1 = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=self.grid)
        self.p_coeff_2 = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=self.grid)
        self.p_coeff_3 = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=self.grid)
        self.r_m = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=self.grid)

    def run(
        self,
        diagnostic_state: advection_states.AdvectionDiagnosticState,
        prep_adv: solve_nh_states.PrepAdvection,
        p_tracer_now: fa.CellKField[wpfloat],
        p_tracer_new: fa.CellKField[wpfloat],
        dtime: wpfloat,
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
        log.debug("advection class run - start")

        log.debug("communication of prep_adv cell field: mass_flx_ic - start")
        self.exchange.exchange_and_wait(dims.CellDim, prep_adv.mass_flx_ic)
        log.debug("communication of prep_adv cell field: mass_flx_ic - end")

        if (
            self.config.horizontal_advection_type == HorizontalAdvectionType.NO_ADVECTION
            and self.config.vertical_advection_type == VerticalAdvectionType.NO_ADVECTION
        ):
            log.debug("running stencil copy_cell_kdim_field - start")
            copy_cell_kdim_field.copy_cell_kdim_field(
                p_tracer_now,
                p_tracer_new,
                horizontal_start=self.start_cell_nudging,
                horizontal_end=self.end_cell_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )
            log.debug("running stencil copy_cell_kdim_field - end")

            log.debug("advection class run - early end")
            return

        # Godunov splitting
        if self.even_timestep:  # even timestep, vertical transport precedes horizontal transport
            # reintegrate density with vertical increment for conservation of mass
            log.debug("running stencil apply_vertical_density_increment - start")
            apply_vertical_density_increment.apply_vertical_density_increment(
                rhodz_ast=diagnostic_state.airmass_now,
                p_mflx_contra_v=prep_adv.mass_flx_ic,
                deepatmo_divzl=self.metric_state.deepatmo_divzl,
                deepatmo_divzu=self.metric_state.deepatmo_divzu,
                p_dtime=dtime,
                rhodz_ast2=self.rhodz_ast2,
                horizontal_start=self.start_cell_lateral_boundary_level_2,
                horizontal_end=self.end_cell_end,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )
            log.debug("running stencil apply_vertical_density_increment - end")

            # vertical transport
            self._run_vertical_advection(
                diagnostic_state=diagnostic_state,
                prep_adv=prep_adv,
                p_tracer_now=p_tracer_now,
                p_tracer_new=p_tracer_new,
                rhodz_now=diagnostic_state.airmass_now,
                rhodz_new=self.rhodz_ast2,
                dtime=dtime,
            )

            # horizontal transport
            self._run_horizontal_advection(
                prep_adv=prep_adv,
                p_tracer_now=p_tracer_new,
                p_tracer_new=p_tracer_new,
                rhodz_now=self.rhodz_ast2,
                rhodz_new=diagnostic_state.airmass_new,
                p_mflx_tracer_h=diagnostic_state.hfl_tracer,
                dtime=dtime,
            )

        else:  # odd timestep, horizontal transport precedes vertical transport
            # reintegrate density with horizontal increment for conservation of mass
            log.debug("running stencil apply_horizontal_density_increment - start")
            apply_horizontal_density_increment.apply_horizontal_density_increment(
                p_rhodz_new=diagnostic_state.airmass_new,
                p_mflx_contra_v=prep_adv.mass_flx_ic,
                deepatmo_divzl=self.metric_state.deepatmo_divzl,
                deepatmo_divzu=self.metric_state.deepatmo_divzu,
                p_dtime=dtime,
                rhodz_ast2=self.rhodz_ast2,
                horizontal_start=self.start_cell_lateral_boundary_level_3,
                horizontal_end=self.end_cell_end,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )
            log.debug("running stencil apply_horizontal_density_increment - end")

            # horizontal transport
            self._run_horizontal_advection(
                prep_adv=prep_adv,
                p_tracer_now=p_tracer_now,
                p_tracer_new=p_tracer_new,
                rhodz_now=diagnostic_state.airmass_now,
                rhodz_new=self.rhodz_ast2,
                p_mflx_tracer_h=diagnostic_state.hfl_tracer,
                dtime=dtime,
            )

            # vertical transport
            self._run_vertical_advection(
                diagnostic_state=diagnostic_state,
                prep_adv=prep_adv,
                p_tracer_now=p_tracer_new,
                p_tracer_new=p_tracer_new,
                rhodz_now=self.rhodz_ast2,
                rhodz_new=diagnostic_state.airmass_new,
                dtime=dtime,
            )

        # update lateral boundaries with interpolated time tendencies
        if self.grid.limited_area:
            log.debug("running stencil apply_interpolated_tracer_time_tendency - start")
            apply_interpolated_tracer_time_tendency.apply_interpolated_tracer_time_tendency(
                p_tracer_now=p_tracer_now,
                p_grf_tend_tracer=diagnostic_state.grf_tend_tracer,
                p_tracer_new=p_tracer_new,
                p_dtime=dtime,
                horizontal_start=self.start_cell_lateral_boundary,
                horizontal_end=self.end_cell_lateral_boundary_level_4,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )
            log.debug("running stencil apply_interpolated_tracer_time_tendency - end")

        # exchange updated tracer values, originally happens only if iforcing /= inwp
        log.debug("communication of advection cell field: p_tracer_new - start")
        self.exchange.exchange_and_wait(dims.CellDim, p_tracer_new)
        log.debug("communication of advection cell field: p_tracer_new - end")

        # finalize step
        self.even_timestep = not self.even_timestep

        log.debug("advection class run - end")

    def _run_horizontal_advection(
        self,
        prep_adv: solve_nh_states.PrepAdvection,
        p_tracer_now: fa.CellKField[wpfloat],
        p_tracer_new: fa.CellKField[wpfloat],
        rhodz_now: fa.CellKField[wpfloat],
        rhodz_new: fa.CellKField[wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[wpfloat],
        dtime: wpfloat,
    ):
        """
        Run a horizontal advection step.
        """
        log.debug("horizontal advection run - start")

        if self.config.horizontal_advection_type == HorizontalAdvectionType.NO_ADVECTION:
            log.debug("running stencil copy_cell_kdim_field - start")
            copy_cell_kdim_field.copy_cell_kdim_field(
                p_tracer_now,
                p_tracer_new,
                horizontal_start=self.start_cell_nudging,
                horizontal_end=self.end_cell_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )
            log.debug("running stencil copy_cell_kdim_field - end")

            log.debug("horizontal advection run - early end")
            return

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
            deepatmo_divh=self.metric_state.deepatmo_divh,
            tracer_now=p_tracer_now,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            geofac_div=self.interpolation_state.geofac_div,
            tracer_new_hor=p_tracer_new,
            p_dtime=dtime,
            horizontal_start=self.start_cell_nudging,
            horizontal_end=self.end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )
        log.debug("running stencil integrate_tracer_horizontally - end")

        log.debug("horizontal advection run - end")

    def _run_vertical_advection(
        self,
        diagnostic_state: advection_states.AdvectionDiagnosticState,
        prep_adv: solve_nh_states.PrepAdvection,
        p_tracer_now: fa.CellKField[wpfloat],
        p_tracer_new: fa.CellKField[wpfloat],
        rhodz_now: fa.CellKField[wpfloat],
        rhodz_new: fa.CellKField[wpfloat],
        dtime: wpfloat,
    ):
        """
        Run a vertical advection step.
        """
        log.debug("vertical advection run - start")

        # TODO (dastrm): maybe change how the indices are handled here? originally:
        # if even step, vertical transport includes all halo points in order to avoid an additional synchronization step, i.e.
        #    if lstep_even: i_rlstart  = 2,                i_rlend = min_rlcell
        #    else:          i_rlstart  = grf_bdywidth_c+1, i_rlend = min_rlcell_int
        # note: horizontal advection is always called with the same indices, i.e. i_rlstart = grf_bdywidth_c+1, i_rlend = min_rlcell_int

        if self.config.vertical_advection_type == VerticalAdvectionType.NO_ADVECTION:
            log.debug("running stencil copy_cell_kdim_field - start")
            copy_cell_kdim_field.copy_cell_kdim_field(
                p_tracer_now,
                p_tracer_new,
                horizontal_start=(
                    self.start_cell_lateral_boundary_level_2
                    if self.even_timestep
                    else self.start_cell_nudging
                ),
                horizontal_end=(self.end_cell_end if self.even_timestep else self.end_cell_local),
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )
            log.debug("running stencil copy_cell_kdim_field - end")

            log.debug("vertical advection run - early end")
            return

        # get the vertical numerical tracer flux

        # update tracer mass fraction

        log.debug("vertical advection run - end")

    def _compute_horizontal_tracer_flux(
        self,
        prep_adv: solve_nh_states.PrepAdvection,
        p_tracer_now: fa.CellKField[wpfloat],
        rhodz_now: fa.CellKField[wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[wpfloat],
        dtime: wpfloat,
    ):
        """
        Calculate the horizontal numerical tracer flux.
        """
        log.debug("horizontal tracer flux computation - start")

        ## tracer-independent part

        # compute tangential velocity
        log.debug("running stencil compute_edge_tangential - start")
        compute_edge_tangential.compute_edge_tangential(  # TODO (dastrm): duplicate stencil of compute_tangential_wind
            p_vn_in=prep_adv.vn_traj,
            ptr_coeff=self.interpolation_state.rbf_vec_coeff_e,
            p_vt_out=self.z_real_vt,
            horizontal_start=self.start_edge_lateral_boundary_level_2,
            horizontal_end=self.end_edge_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels,  # originally UBOUND(p_vn,2)
            offset_provider=self.grid.offset_providers,
        )
        log.debug("running stencil compute_edge_tangential - end")

        # backtrajectory calculation
        log.debug("running stencil compute_barycentric_backtrajectory - start")
        compute_barycentric_backtrajectory.compute_barycentric_backtrajectory(
            p_vn=prep_adv.vn_traj,
            p_vt=self.z_real_vt,
            cell_idx=self.cell_idx,
            cell_blk=self.cell_blk,
            pos_on_tplane_e_1=self.interpolation_state.pos_on_tplane_e_1,
            pos_on_tplane_e_2=self.interpolation_state.pos_on_tplane_e_2,
            primal_normal_cell_1=self.edge_params.primal_normal_cell[0],
            dual_normal_cell_1=self.edge_params.dual_normal_cell[0],
            primal_normal_cell_2=self.edge_params.primal_normal_cell[1],
            dual_normal_cell_2=self.edge_params.dual_normal_cell[1],
            p_cell_idx=self.p_cell_idx,
            p_cell_rel_idx_dsl=self.p_cell_rel_idx,
            p_cell_blk=self.p_cell_blk,
            p_distv_bary_1=self.p_distv_bary_1,
            p_distv_bary_2=self.p_distv_bary_2,
            p_dthalf=wpfloat(0.5) * dtime,
            horizontal_start=self.start_edge_lateral_boundary_level_5,
            horizontal_end=self.end_edge_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )
        log.debug("running stencil compute_barycentric_backtrajectory - end")

        ## tracer-specific part

        if self.config.horizontal_advection_type == HorizontalAdvectionType.LINEAR_2ND_ORDER:
            # linear reconstruction using singular value decomposition
            log.debug("running stencil reconstruct_linear_coefficients_svd - start")
            reconstruct_linear_coefficients_svd.reconstruct_linear_coefficients_svd(
                p_cc=p_tracer_now,
                lsq_pseudoinv_1=self.least_squares_state.lsq_pseudoinv_1,
                lsq_pseudoinv_2=self.least_squares_state.lsq_pseudoinv_2,
                p_coeff_1_dsl=self.p_coeff_1,
                p_coeff_2_dsl=self.p_coeff_2,
                p_coeff_3_dsl=self.p_coeff_3,
                horizontal_start=self.start_cell_lateral_boundary_level_2,  # originally i_rlstart_c = get_startrow_c(startrow_e=5) = 2
                horizontal_end=self.end_cell_halo,
                vertical_start=0,
                vertical_end=self.grid.num_levels,  # originally UBOUND(p_cc,2)
                offset_provider=self.grid.offset_providers,
            )
            log.debug("running stencil reconstruct_linear_coefficients_svd - end")

            # compute reconstructed tracer value at each barycenter and corresponding flux at each edge
            log.debug(
                "running stencil compute_horizontal_tracer_flux_from_linear_coefficients - start"
            )
            compute_horizontal_tracer_flux_from_linear_coefficients.compute_horizontal_tracer_flux_from_linear_coefficients(
                z_lsq_coeff_1=self.p_coeff_1,
                z_lsq_coeff_2=self.p_coeff_2,
                z_lsq_coeff_3=self.p_coeff_3,
                distv_bary_1=self.p_distv_bary_1,
                distv_bary_2=self.p_distv_bary_2,
                p_mass_flx_e=prep_adv.mass_flx_me,
                cell_rel_idx_dsl=self.p_cell_rel_idx,
                p_out_e=p_mflx_tracer_h,
                horizontal_start=self.start_edge_lateral_boundary_level_5,
                horizontal_end=self.end_edge_halo,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )
            log.debug(
                "running stencil compute_horizontal_tracer_flux_from_linear_coefficients - end"
            )

        # apply flux limiter
        if (
            self.config.horizontal_advection_limiter
            == HorizontalAdvectionLimiterType.POSITIVE_DEFINITE
        ):
            # compute multiplicative flux factor to guarantee no undershoot
            log.debug(
                "running stencil compute_positive_definite_horizontal_multiplicative_flux_factor - start"
            )
            compute_positive_definite_horizontal_multiplicative_flux_factor.compute_positive_definite_horizontal_multiplicative_flux_factor(
                geofac_div=self.interpolation_state.geofac_div,
                p_cc=p_tracer_now,
                p_rhodz_now=rhodz_now,
                p_mflx_tracer_h=p_mflx_tracer_h,
                r_m=self.r_m,
                p_dtime=dtime,
                dbl_eps=constants.DBL_EPS,
                horizontal_start=self.start_cell_lateral_boundary_level_2,  # originally i_rlstart_c = get_startrow_c(startrow_e=5) = 2
                horizontal_end=self.end_cell_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )
            log.debug(
                "running stencil compute_positive_definite_horizontal_multiplicative_flux_factor - end"
            )

            log.debug("communication of advection cell field: r_m - start")
            self.exchange.exchange_and_wait(dims.CellDim, self.r_m)
            log.debug("communication of advection cell field: r_m - end")

            # limit outward fluxes
            log.debug(
                "running stencil apply_positive_definite_horizontal_multiplicative_flux_factor - start"
            )
            apply_positive_definite_horizontal_multiplicative_flux_factor.apply_positive_definite_horizontal_multiplicative_flux_factor(
                r_m=self.r_m,
                p_mflx_tracer_h=p_mflx_tracer_h,
                horizontal_start=self.start_edge_lateral_boundary_level_5,
                horizontal_end=self.end_edge_halo,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )
            log.debug(
                "running stencil apply_positive_definite_horizontal_multiplicative_flux_factor - end"
            )

        log.debug("horizontal tracer flux computation - end")
