# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
import logging

import icon4py.model.common.grid.states as grid_states
import gt4py.next as gtx
from gt4py.next import backend

from icon4py.model.atmosphere.advection import advection_states

from icon4py.model.atmosphere.advection.stencils.compute_ppm_quadratic_face_values import (
    compute_ppm_quadratic_face_values,
)
from icon4py.model.atmosphere.advection.stencils.compute_ppm_quartic_face_values import (
    compute_ppm_quartic_face_values,
)
from icon4py.model.atmosphere.advection.stencils.compute_ppm_slope import compute_ppm_slope
from icon4py.model.atmosphere.advection.stencils.compute_ppm4gpu_courant_number import (
    compute_ppm4gpu_courant_number,
)
from icon4py.model.atmosphere.advection.stencils.compute_ppm4gpu_fractional_flux import (
    compute_ppm4gpu_fractional_flux,
)
from icon4py.model.atmosphere.advection.stencils.compute_ppm4gpu_integer_flux import (
    compute_ppm4gpu_integer_flux,
)
from icon4py.model.atmosphere.advection.stencils.compute_ppm4gpu_parabola_coefficients import (
    compute_ppm4gpu_parabola_coefficients,
)
from icon4py.model.atmosphere.advection.stencils.compute_vertical_parabola_limiter_condition import (
    compute_vertical_parabola_limiter_condition,
)
from icon4py.model.atmosphere.advection.stencils.compute_vertical_tracer_flux_upwind import (
    compute_vertical_tracer_flux_upwind,
)
from icon4py.model.atmosphere.advection.stencils.copy_cell_kdim_field import copy_cell_kdim_field
from icon4py.model.atmosphere.advection.stencils.copy_cell_kdim_field_koff_minus1 import (
    copy_cell_kdim_field_koff_minus1,
)
from icon4py.model.atmosphere.advection.stencils.copy_cell_kdim_field_koff_plus1 import (
    copy_cell_kdim_field_koff_plus1,
)
from icon4py.model.atmosphere.advection.stencils.init_constant_cell_kdim_field import (
    init_constant_cell_kdim_field,
)
from icon4py.model.atmosphere.advection.stencils.integrate_tracer_vertically import (
    integrate_tracer_vertically,
)
from icon4py.model.atmosphere.advection.stencils.limit_vertical_parabola_semi_monotonically import (
    limit_vertical_parabola_semi_monotonically,
)
from icon4py.model.atmosphere.advection.stencils.limit_vertical_slope_semi_monotonically import (
    limit_vertical_slope_semi_monotonically,
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
Advection components related to vertical transport.
"""

# flake8: noqa
log = logging.getLogger(__name__)


class BoundaryConditions(ABC):
    """Class that sets the upper and lower boundary conditions."""

    @abstractmethod
    def run(
        self,
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO (dastrm): should be KHalfDim
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ):
        """
        Set the vertical boundary conditions.

        Args:
            p_mflx_tracer_v: output argument, field that contains new vertical tracer mass flux
            horizontal_start: input argument, horizontal start index
            horizontal_end: input argument, horizontal end index

        """
        ...


class NoFluxCondition(BoundaryConditions):
    """Class that sets the upper and lower boundary fluxes to zero."""

    def __init__(self, grid: icon_grid.IconGrid, backend: backend.Backend):
        # input arguments
        self._grid = grid
        self._backend = backend

        # stencils
        self._init_constant_cell_kdim_field = init_constant_cell_kdim_field.with_backend(
            self._backend
        )

    def run(
        self,
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO (dastrm): should be KHalfDim
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ):
        log.debug("vertical boundary conditions computation - start")

        # set upper boundary conditions
        log.debug("running stencil init_constant_cell_kdim_field - start")
        self._init_constant_cell_kdim_field(
            field=p_mflx_tracer_v,
            value=0.0,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=1,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil init_constant_cell_kdim_field - end")

        # set lower boundary conditions
        log.debug("running stencil init_constant_cell_kdim_field - start")
        self._init_constant_cell_kdim_field(
            field=p_mflx_tracer_v,
            value=0.0,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=self._grid.num_levels,
            vertical_end=self._grid.num_levels + 1,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil init_constant_cell_kdim_field - end")

        log.debug("vertical boundary conditions computation - end")


class VerticalLimiter(ABC):
    """Class that limits the vertical reconstructed fields and the fluxes."""

    def limit_slope(
        self,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        z_slope: fa.CellKField[ta.wpfloat],
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ):
        ...

    def limit_parabola(
        self,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_face: fa.CellKField[ta.wpfloat],  # TODO (dastrm): should be KHalfDim
        p_face_up: fa.CellKField[ta.wpfloat],
        p_face_low: fa.CellKField[ta.wpfloat],
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ):
        ...

    def limit_fluxes(
        self,
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ):
        ...


class NoLimiter(VerticalLimiter):
    """Class that implements no vertical parabola limiter."""

    def __init__(self, grid: icon_grid.IconGrid, backend: backend.Backend):
        # input arguments
        self._grid = grid
        self._backend = backend

        # fields
        self._l_limit = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, backend=self._backend
        )

        # stencils
        self._copy_cell_kdim_field = copy_cell_kdim_field.with_backend(self._backend)
        self._copy_cell_kdim_field_koff_plus1 = copy_cell_kdim_field_koff_plus1.with_backend(
            self._backend
        )

    def limit_slope(
        self,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        z_slope: fa.CellKField[ta.wpfloat],
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ):
        ...

    def limit_parabola(
        self,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_face: fa.CellKField[ta.wpfloat],  # TODO (dastrm): should be KHalfDim
        p_face_up: fa.CellKField[ta.wpfloat],
        p_face_low: fa.CellKField[ta.wpfloat],
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ):
        # simply copy to up/low face values
        log.debug("running stencil copy_cell_kdim_field - start")
        self._copy_cell_kdim_field(
            field_in=p_face,
            field_out=p_face_up,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil copy_cell_kdim_field - end")

        log.debug("running stencil copy_cell_kdim_field_koff_plus1 - start")
        self._copy_cell_kdim_field_koff_plus1(
            field_in=p_face,
            field_out=p_face_low,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil copy_cell_kdim_field_koff_plus1 - end")

    def limit_fluxes(
        self,
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ):
        ...


class SemiMonotonicLimiter(VerticalLimiter):
    """Class that implements a semi-monotonic vertical parabola limiter."""

    def __init__(self, grid: icon_grid.IconGrid, backend: backend.Backend):
        # input arguments
        self._grid = grid
        self._backend = backend

        # fields
        self._k_field = field_alloc.allocate_indices(
            dims.KDim, grid=self._grid, is_halfdim=True, dtype=gtx.int32, backend=self._backend
        )  # TODO (dastrm): should be KHalfDim
        self._l_limit = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, dtype=gtx.int32, backend=self._backend
        )

        # stencils
        self._limit_vertical_slope_semi_monotonically = (
            limit_vertical_slope_semi_monotonically.with_backend(self._backend)
        )
        self._compute_vertical_parabola_limiter_condition = (
            compute_vertical_parabola_limiter_condition.with_backend(self._backend)
        )
        self._limit_vertical_parabola_semi_monotonically = (
            limit_vertical_parabola_semi_monotonically.with_backend(self._backend)
        )

    def limit_slope(
        self,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        z_slope: fa.CellKField[ta.wpfloat],
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ):
        log.debug("running stencil limit_vertical_slope_semi_monotonically - start")
        self._limit_vertical_slope_semi_monotonically(
            p_cc=p_tracer_now,
            z_slope=z_slope,
            k=self._k_field,
            elev=self._grid.num_levels - 1,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=1,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil limit_vertical_slope_semi_monotonically - end")

    def limit_parabola(
        self,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_face: fa.CellKField[ta.wpfloat],  # TODO (dastrm): should be KHalfDim
        p_face_up: fa.CellKField[ta.wpfloat],
        p_face_low: fa.CellKField[ta.wpfloat],
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ):
        # compute semi-monotonic limiter condition
        log.debug("running stencil compute_vertical_parabola_limiter_condition - start")
        self._compute_vertical_parabola_limiter_condition(
            p_face=p_face,
            p_cc=p_tracer_now,
            l_limit=self._l_limit,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil compute_vertical_parabola_limiter_condition - end")

        # apply semi-monotonic limiter condition and store to up/low face values
        log.debug("running stencil limit_vertical_parabola_semi_monotonically - start")
        self._limit_vertical_parabola_semi_monotonically(
            l_limit=self._l_limit,
            p_face=p_face,
            p_cc=p_tracer_now,
            p_face_up=p_face_up,
            p_face_low=p_face_low,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil limit_vertical_parabola_semi_monotonically - end")

    def limit_fluxes(
        self,
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ):
        ...


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

        Note:
            Originally, if even step, vertical transport includes all halo points in order to avoid an additional synchronization step, i.e.
                if lstep_even: i_rlstart  = 2,                i_rlend = min_rlcell
                else:          i_rlstart  = grf_bdywidth_c+1, i_rlend = min_rlcell_int
            Horizontal advection is always called with the same indices though, i.e.
                i_rlstart = grf_bdywidth_c+1, i_rlend = min_rlcell_int
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

    def _get_horizontal_start_end(self, even_timestep: bool):
        if even_timestep:
            horizontal_start = self._start_cell_lateral_boundary_level_2
            horizontal_end = self._end_cell_end
        else:
            horizontal_start = self._start_cell_nudging
            horizontal_end = self._end_cell_local

        return horizontal_start, horizontal_end

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

        horizontal_start, horizontal_end = self._get_horizontal_start_end(
            even_timestep=even_timestep
        )

        log.debug("running stencil copy_cell_kdim_field - start")
        self._copy_cell_kdim_field(
            field_in=p_tracer_now,
            field_out=p_tracer_new,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
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
        log.debug("vertical advection run - start")

        self._compute_numerical_flux(
            prep_adv=prep_adv,
            p_tracer_now=p_tracer_now,
            rhodz_now=rhodz_now,
            p_mflx_tracer_v=p_mflx_tracer_v,
            dtime=dtime,
            even_timestep=even_timestep,
        )

        self._update_unknowns(
            p_tracer_now=p_tracer_now,
            p_tracer_new=p_tracer_new,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            p_mflx_tracer_v=p_mflx_tracer_v,
            dtime=dtime,
            even_timestep=even_timestep,
        )

        log.debug("vertical advection run - end")

    @abstractmethod
    def _compute_numerical_flux(
        self,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO (dastrm): should be KHalfDim
        dtime: ta.wpfloat,
        even_timestep: bool,
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
        even_timestep: bool,
    ):
        ...


class FirstOrderUpwind(FiniteVolume):
    """Class that does one vertical first-order accurate upwind finite volume advection step."""

    def __init__(
        self,
        boundary_conditions: BoundaryConditions,
        grid: icon_grid.IconGrid,
        metric_state: advection_states.AdvectionMetricState,
        backend=backend,
    ):
        log.debug("vertical advection class init - start")

        # input arguments
        self._boundary_conditions = boundary_conditions
        self._grid = grid
        self._metric_state = metric_state
        self._backend = backend

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_lateral_boundary_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._end_cell_end = self._grid.end_index(cell_domain(h_grid.Zone.END))

        # fields
        self._k_field = field_alloc.allocate_indices(
            dims.KDim, grid=self._grid, is_halfdim=True, dtype=gtx.int32, backend=self._backend
        )  # TODO (dastrm): should be KHalfDim

        # stencils
        self._compute_vertical_tracer_flux_upwind = (
            compute_vertical_tracer_flux_upwind.with_backend(self._backend)
        )
        self._init_constant_cell_kdim_field = init_constant_cell_kdim_field.with_backend(
            self._backend
        )
        self._integrate_tracer_vertically = integrate_tracer_vertically.with_backend(self._backend)

        # misc
        self._ivadv_tracer = 1
        self._iadv_slev_jt = 0

        log.debug("vertical advection class init - end")

    def _get_horizontal_start_end(self, even_timestep: bool):
        if even_timestep:
            horizontal_start = self._start_cell_lateral_boundary_level_2
            horizontal_end = self._end_cell_end
        else:
            horizontal_start = self._start_cell_nudging
            horizontal_end = self._end_cell_local

        return horizontal_start, horizontal_end

    def _compute_numerical_flux(
        self,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO (dastrm): should be KHalfDim
        dtime: ta.wpfloat,
        even_timestep: bool,
    ):
        log.debug("vertical numerical flux computation - start")

        horizontal_start, horizontal_end = self._get_horizontal_start_end(
            even_timestep=even_timestep
        )

        log.debug("running stencil compute_vertical_tracer_flux_upwind - start")
        self._compute_vertical_tracer_flux_upwind(
            p_cc=p_tracer_now,
            p_mflx_contra_v=prep_adv.mass_flx_ic,
            p_upflux=p_mflx_tracer_v,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=1,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil compute_vertical_tracer_flux_upwind - end")

        # set boundary conditions
        self._boundary_conditions.run(
            p_mflx_tracer_v=p_mflx_tracer_v,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )

        log.debug("vertical numerical flux computation - end")

    def _update_unknowns(
        self,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO (dastrm): should be KHalfDim
        dtime: ta.wpfloat,
        even_timestep: bool,
    ):
        log.debug("vertical unknowns update - start")

        horizontal_start, horizontal_end = self._get_horizontal_start_end(
            even_timestep=even_timestep
        )

        # update tracer mass fraction
        log.debug("running stencil integrate_tracer_vertically - start")
        self._integrate_tracer_vertically(
            tracer_now=p_tracer_now,
            rhodz_now=rhodz_now,
            p_mflx_tracer_v=p_mflx_tracer_v,
            deepatmo_divzl=self._metric_state.deepatmo_divzl,
            deepatmo_divzu=self._metric_state.deepatmo_divzu,
            rhodz_new=rhodz_new,
            tracer_new=p_tracer_new,
            k=self._k_field,
            p_dtime=dtime,
            ivadv_tracer=self._ivadv_tracer,
            iadv_slev_jt=self._iadv_slev_jt,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil integrate_tracer_vertically - end")

        log.debug("vertical unknowns update - end")


class PiecewiseParabolicMethod(FiniteVolume):
    """Class that does one vertical PPM finite volume advection step."""

    def __init__(
        self,
        boundary_conditions: BoundaryConditions,
        vertical_limiter: VerticalLimiter,
        grid: icon_grid.IconGrid,
        metric_state: advection_states.AdvectionMetricState,
        backend=backend,
    ):
        log.debug("vertical advection class init - start")

        # input arguments
        self._boundary_conditions = boundary_conditions
        self._vertical_limiter = vertical_limiter
        self._grid = grid
        self._metric_state = metric_state
        self._backend = backend

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_lateral_boundary_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._end_cell_end = self._grid.end_index(cell_domain(h_grid.Zone.END))

        # fields
        self._k_field = field_alloc.allocate_indices(
            dims.KDim, grid=self._grid, is_halfdim=True, dtype=gtx.int32, backend=self._backend
        )  # TODO (dastrm): should be KHalfDim
        self._z_cfl = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, is_halfdim=True, backend=self._backend
        )  # TODO (dastrm): should be KHalfDim
        self._z_slope = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self._z_face = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, is_halfdim=True, backend=self._backend
        )  # TODO (dastrm): should be KHalfDim
        self._z_face_up = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self._z_face_low = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self._z_delta_q = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self._z_a1 = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, backend=self._backend
        )

        # stencils
        self._init_constant_cell_kdim_field = init_constant_cell_kdim_field.with_backend(
            self._backend
        )
        self._compute_ppm4gpu_courant_number = compute_ppm4gpu_courant_number.with_backend(
            self._backend
        )
        self._compute_ppm_slope = compute_ppm_slope.with_backend(self._backend)
        self._compute_ppm_quadratic_face_values = compute_ppm_quadratic_face_values.with_backend(
            self._backend
        )
        self._compute_ppm_quartic_face_values = compute_ppm_quartic_face_values.with_backend(
            self._backend
        )
        self._copy_cell_kdim_field = copy_cell_kdim_field.with_backend(self._backend)
        self._copy_cell_kdim_field_koff_minus1 = copy_cell_kdim_field_koff_minus1.with_backend(
            self._backend
        )
        self._compute_ppm4gpu_parabola_coefficients = (
            compute_ppm4gpu_parabola_coefficients.with_backend(self._backend)
        )
        self._compute_ppm4gpu_fractional_flux = compute_ppm4gpu_fractional_flux.with_backend(
            self._backend
        )
        self._compute_ppm4gpu_integer_flux = compute_ppm4gpu_integer_flux.with_backend(
            self._backend
        )
        self._integrate_tracer_vertically = integrate_tracer_vertically.with_backend(self._backend)

        # misc
        self._slev = 0
        self._slevp1_ti = 1
        self._elev = self._grid.num_levels - 1
        self._nlev = self._grid.num_levels - 1
        self._ivadv_tracer = 1
        self._iadv_slev_jt = 0

        log.debug("vertical advection class init - end")

    def _get_horizontal_start_end(self, even_timestep: bool):
        if even_timestep:
            horizontal_start = self._start_cell_lateral_boundary_level_2
            horizontal_end = self._end_cell_end
        else:
            horizontal_start = self._start_cell_nudging
            horizontal_end = self._end_cell_local

        return horizontal_start, horizontal_end

    def _compute_numerical_flux(
        self,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO (dastrm): should be KHalfDim
        dtime: ta.wpfloat,
        even_timestep: bool,
    ):
        log.debug("vertical numerical flux computation - start")

        horizontal_start, horizontal_end = self._get_horizontal_start_end(
            even_timestep=even_timestep
        )

        ## compute density-weighted Courant number

        log.debug("running stencil init_constant_cell_kdim_field - start")
        self._init_constant_cell_kdim_field(
            field=self._z_cfl,
            value=0.0,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=self._grid.num_levels + 1,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil init_constant_cell_kdim_field - end")

        log.debug("running stencil compute_ppm4gpu_courant_number - start")
        self._compute_ppm4gpu_courant_number(
            p_mflx_contra_v=prep_adv.mass_flx_ic,
            p_cellmass_now=rhodz_now,
            z_cfl=self._z_cfl,
            k=self._k_field,
            slevp1_ti=self._slevp1_ti,
            nlev=self._nlev,
            dbl_eps=constants.DBL_EPS,
            p_dtime=dtime,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=1,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil compute_ppm4gpu_courant_number - end")

        ## reconstruct face values

        # compute slope
        log.debug("running stencil compute_ppm_slope - start")
        self._compute_ppm_slope(
            p_cc=p_tracer_now,
            p_cellhgt_mc_now=self._metric_state.ddqz_z_full,
            k=self._k_field,
            z_slope=self._z_slope,
            elev=self._elev,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=1,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil compute_ppm_slope - end")

        # limit slope
        self._vertical_limiter.limit_slope(
            p_tracer_now=p_tracer_now,
            z_slope=self._z_slope,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )

        # compute second highest face value
        log.debug("running stencil compute_ppm_quadratic_face_values - start")
        self._compute_ppm_quadratic_face_values(
            p_cc=p_tracer_now,
            p_cellhgt_mc_now=self._metric_state.ddqz_z_full,
            p_face=self._z_face,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=1,
            vertical_end=2,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil compute_ppm_quadratic_face_values - end")

        # compute second lowest face value
        log.debug("running stencil compute_ppm_quadratic_face_values - start")
        self._compute_ppm_quadratic_face_values(
            p_cc=p_tracer_now,
            p_cellhgt_mc_now=self._metric_state.ddqz_z_full,
            p_face=self._z_face,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=self._grid.num_levels - 1,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil compute_ppm_quadratic_face_values - end")

        # compute highest face value
        log.debug("running stencil copy_cell_kdim_field - start")
        self._copy_cell_kdim_field(
            field_in=p_tracer_now,
            field_out=self._z_face,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=1,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil copy_cell_kdim_field - end")

        # compute lowest face value
        log.debug("running stencil copy_cell_kdim_field_koff_minus1 - start")
        self._copy_cell_kdim_field_koff_minus1(
            field_in=p_tracer_now,
            field_out=self._z_face,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=self._grid.num_levels,
            vertical_end=self._grid.num_levels + 1,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil copy_cell_kdim_field_koff_minus1 - end")

        # compute all other face values
        log.debug("running stencil compute_ppm_quartic_face_values - start")
        self._compute_ppm_quartic_face_values(
            p_cc=p_tracer_now,
            p_cellhgt_mc_now=self._metric_state.ddqz_z_full,
            z_slope=self._z_slope,
            p_face=self._z_face,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=2,
            vertical_end=self._grid.num_levels - 1,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil compute_ppm_quartic_face_values - end")

        ## limit reconstruction

        self._vertical_limiter.limit_parabola(
            p_tracer_now=p_tracer_now,
            p_face=self._z_face,
            p_face_up=self._z_face_up,
            p_face_low=self._z_face_low,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )

        ## compute fractional numerical flux

        log.debug("running stencil compute_ppm4gpu_parabola_coefficients - start")
        self._compute_ppm4gpu_parabola_coefficients(
            z_face_up=self._z_face_up,
            z_face_low=self._z_face_low,
            p_cc=p_tracer_now,
            z_delta_q=self._z_delta_q,
            z_a1=self._z_a1,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil compute_ppm4gpu_parabola_coefficients - end")

        log.debug("running stencil compute_ppm4gpu_fractional_flux - start")
        self._compute_ppm4gpu_fractional_flux(
            p_cc=p_tracer_now,
            p_cellmass_now=rhodz_now,
            z_cfl=self._z_cfl,
            z_delta_q=self._z_delta_q,
            z_a1=self._z_a1,
            p_upflux=p_mflx_tracer_v,
            k=self._k_field,
            slev=self._slev,
            p_dtime=dtime,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=1,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil compute_ppm4gpu_fractional_flux - end")

        ## compute integer numerical flux

        log.debug("running stencil compute_ppm4gpu_integer_flux - start")
        self._compute_ppm4gpu_integer_flux(
            p_cc=p_tracer_now,
            p_cellmass_now=rhodz_now,
            z_cfl=self._z_cfl,
            p_upflux=p_mflx_tracer_v,
            k=self._k_field,
            slev=self._slev,
            p_dtime=dtime,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=1,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil compute_ppm4gpu_integer_flux - end")

        ## set boundary conditions

        self._boundary_conditions.run(
            p_mflx_tracer_v=p_mflx_tracer_v,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )

        ## apply flux limiter

        self._vertical_limiter.limit_fluxes(
            horizontal_start=horizontal_start, horizontal_end=horizontal_end
        )

        log.debug("vertical numerical flux computation - end")

    def _update_unknowns(
        self,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO (dastrm): should be KHalfDim
        dtime: ta.wpfloat,
        even_timestep: bool,
    ):
        log.debug("vertical unknowns update - start")

        horizontal_start, horizontal_end = self._get_horizontal_start_end(
            even_timestep=even_timestep
        )

        # update tracer mass fraction
        log.debug("running stencil integrate_tracer_vertically - start")
        self._integrate_tracer_vertically(
            tracer_now=p_tracer_now,
            rhodz_now=rhodz_now,
            p_mflx_tracer_v=p_mflx_tracer_v,
            deepatmo_divzl=self._metric_state.deepatmo_divzl,
            deepatmo_divzu=self._metric_state.deepatmo_divzu,
            rhodz_new=rhodz_new,
            tracer_new=p_tracer_new,
            k=self._k_field,
            p_dtime=dtime,
            ivadv_tracer=self._ivadv_tracer,
            iadv_slev_jt=self._iadv_slev_jt,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil integrate_tracer_vertically - end")

        log.debug("vertical unknowns update - end")
