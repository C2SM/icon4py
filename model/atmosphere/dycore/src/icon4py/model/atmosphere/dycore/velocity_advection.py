# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import backend

import icon4py.model.atmosphere.dycore.velocity_advection_stencils as velocity_stencils
from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.atmosphere.dycore.stencils.add_extra_diffusion_for_normal_wind_tendency_approaching_cfl import (
    add_extra_diffusion_for_normal_wind_tendency_approaching_cfl,
)
from icon4py.model.atmosphere.dycore.stencils.add_extra_diffusion_for_w_con_approaching_cfl import (
    add_extra_diffusion_for_w_con_approaching_cfl,
)
from icon4py.model.atmosphere.dycore.stencils.compute_advective_normal_wind_tendency import (
    compute_advective_normal_wind_tendency,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_advection_term_for_vertical_velocity import (
    compute_horizontal_advection_term_for_vertical_velocity,
)
from icon4py.model.atmosphere.dycore.stencils.compute_tangential_wind import compute_tangential_wind
from icon4py.model.atmosphere.dycore.stencils.interpolate_contravariant_vertical_velocity_to_full_levels import (
    interpolate_contravariant_vertical_velocity_to_full_levels,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_to_cell_center import (
    interpolate_to_cell_center,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_vn_to_ie_and_compute_ekin_on_edges import (
    interpolate_vn_to_ie_and_compute_ekin_on_edges,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_vt_to_interface_edges import (
    interpolate_vt_to_interface_edges,
)
from icon4py.model.atmosphere.dycore.stencils.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.atmosphere.dycore.stencils.mo_math_divrot_rot_vertex_ri_dsl import (
    mo_math_divrot_rot_vertex_ri_dsl,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.grid import (
    horizontal as h_grid,
    icon as icon_grid,
    states as grid_states,
    vertical as v_grid,
)
from icon4py.model.common.settings import xp
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc


class VelocityAdvection:
    def __init__(
        self,
        grid: icon_grid.IconGrid,
        metric_state: dycore_states.MetricStateNonHydro,
        interpolation_state: dycore_states.InterpolationState,
        vertical_params: v_grid.VerticalGrid,
        edge_params: grid_states.EdgeParams,
        owner_mask: fa.CellField[bool],
        backend: backend.Backend,
    ):
        self.grid: icon_grid.IconGrid = grid
        self._backend = backend
        self.metric_state: dycore_states.MetricStateNonHydro = metric_state
        self.interpolation_state: dycore_states.InterpolationState = interpolation_state
        self.vertical_params = vertical_params
        self.edge_params = edge_params
        self.c_owner_mask = owner_mask

        self.cfl_w_limit: float = 0.65
        self.scalfac_exdiff: float = 0.05
        self._allocate_local_fields()
        self._determine_local_domains()

        self._mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl = (
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl.with_backend(self._backend)
        )
        self._mo_math_divrot_rot_vertex_ri_dsl = mo_math_divrot_rot_vertex_ri_dsl.with_backend(
            self._backend
        )
        self._compute_tangential_wind = compute_tangential_wind.with_backend(self._backend)
        self._interpolate_vn_to_ie_and_compute_ekin_on_edges = (
            interpolate_vn_to_ie_and_compute_ekin_on_edges.with_backend(self._backend)
        )
        self._interpolate_vt_to_interface_edges = interpolate_vt_to_interface_edges.with_backend(
            self._backend
        )
        self._fused_stencils_4_5 = velocity_stencils.fused_stencils_4_5.with_backend(self._backend)
        self._extrapolate_at_top = velocity_stencils.extrapolate_at_top.with_backend(self._backend)
        self._compute_horizontal_advection_term_for_vertical_velocity = (
            compute_horizontal_advection_term_for_vertical_velocity.with_backend(self._backend)
        )
        self._interpolate_to_cell_center = interpolate_to_cell_center.with_backend(self._backend)
        self._fused_stencils_9_10 = velocity_stencils.fused_stencils_9_10.with_backend(
            self._backend
        )
        self._fused_stencils_11_to_13 = velocity_stencils.fused_stencils_11_to_13.with_backend(
            self._backend
        )
        self._fused_stencil_14 = velocity_stencils.fused_stencil_14.with_backend(self._backend)
        self._interpolate_contravariant_vertical_velocity_to_full_levels = (
            interpolate_contravariant_vertical_velocity_to_full_levels.with_backend(self._backend)
        )
        self._fused_stencils_16_to_17 = velocity_stencils.fused_stencils_16_to_17.with_backend(
            self._backend
        )
        self._add_extra_diffusion_for_w_con_approaching_cfl = (
            add_extra_diffusion_for_w_con_approaching_cfl.with_backend(self._backend)
        )
        self._compute_advective_normal_wind_tendency = (
            compute_advective_normal_wind_tendency.with_backend(self._backend)
        )
        self._add_extra_diffusion_for_normal_wind_tendency_approaching_cfl = (
            add_extra_diffusion_for_normal_wind_tendency_approaching_cfl.with_backend(self._backend)
        )

    def _allocate_local_fields(self):
        self.z_w_v = field_alloc.allocate_zero_field(
            dims.VertexDim, dims.KDim, is_halfdim=True, grid=self.grid, backend=self._backend
        )
        self.z_v_grad_w = field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_ekinh = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_w_concorr_mc = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_w_con_c = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, is_halfdim=True, grid=self.grid, backend=self._backend
        )
        self.zeta = field_alloc.allocate_zero_field(
            dims.VertexDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_w_con_c_full = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.cfl_clipping = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid, dtype=bool, backend=self._backend
        )
        self.levmask = field_alloc.allocate_zero_field(
            dims.KDim, grid=self.grid, dtype=bool, backend=self._backend
        )
        self.vcfl_dsl = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.k_field = field_alloc.allocate_indices(
            dims.KDim, grid=self.grid, is_halfdim=True, backend=self._backend
        )

    def _determine_local_domains(self):
        vertex_domain = h_grid.domain(dims.VertexDim)
        edge_domain = h_grid.domain(dims.EdgeDim)
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_vertex_lateral_boundary_level_2 = self.grid.start_index(
            vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._end_vertex_halo = self.grid.end_index(vertex_domain(h_grid.Zone.HALO))

        self._start_edge_lateral_boundary_level_5 = self.grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._start_edge_lateral_boundary_level_7 = self.grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7)
        )
        self._start_edge_nudging_level_2 = self.grid.start_index(
            edge_domain(h_grid.Zone.NUDGING_LEVEL_2)
        )

        self._end_edge_local = self.grid.end_index(edge_domain(h_grid.Zone.LOCAL))
        self._end_edge_halo = self.grid.end_index(edge_domain(h_grid.Zone.HALO))
        self._end_edge_halo_level_2 = self.grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))

        self._start_cell_lateral_boundary_level_3 = self.grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        self._start_cell_lateral_boundary_level_4 = self.grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
        )
        self._start_cell_nudging = self.grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self.grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._end_cell_halo = self.grid.end_index(cell_domain(h_grid.Zone.HALO))

    def run_predictor_step(
        self,
        vn_only: bool,
        diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_state: prognostics.PrognosticState,
        z_w_concorr_me: fa.EdgeKField[float],
        z_kin_hor_e: fa.EdgeKField[float],
        z_vt_ie: fa.EdgeKField[float],
        dtime: float,
        ntnd: int,
        cell_areas: fa.CellField[float],
    ):
        cfl_w_limit, scalfac_exdiff = self._scale_factors_by_dtime(dtime)

        if not vn_only:
            self._mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
                p_cell_in=prognostic_state.w,
                c_intp=self.interpolation_state.c_intp,
                p_vert_out=self.z_w_v,
                horizontal_start=self._start_vertex_lateral_boundary_level_2,
                horizontal_end=self._end_vertex_halo,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        # scidoc:
        # Outputs:
        #  - zeta :
        #     $$
        #     \vortvert{\n}{\v}{\k} = \sum_{\offProv{v2e}} \Crot \vn{\n}{\e}{\k}
        #     $$
        #     Compute the vorticity on vertices using the discrete Stokes
        #     theorem (eq. 5 in |BonaventuraRingler2005|).
        #
        # Inputs:
        #  - $\Crot$ : geofac_rot
        #  - $\vn{\n}{\e}{\k}$ : vn
        #
        self._mo_math_divrot_rot_vertex_ri_dsl(
            vec_e=prognostic_state.vn,
            geofac_rot=self.interpolation_state.geofac_rot,
            rot_vec=self.zeta,
            horizontal_start=self._start_vertex_lateral_boundary_level_2,
            horizontal_end=self._end_vertex_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        # scidoc:
        # Outputs:
        #  - vt :
        #     $$
        #     \vt{\n}{\e}{\k} = \sum_{\offProv{e2c2e}} \Wrbf \vn{\n}{\e}{\k}
        #     $$
        #     Compute the tangential velocity by RBF interpolation from four neighboring
        #     edges (diamond shape) projected along the tangential direction.
        #
        # Inputs:
        #  - $\Wrbf$ : rbf_vec_coeff_e
        #  - $\vn{\n}{\e}{\k}$ : vn
        #
        self._compute_tangential_wind(
            vn=prognostic_state.vn,
            rbf_vec_coeff_e=self.interpolation_state.rbf_vec_coeff_e,
            vt=diagnostic_state.vt,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        # scidoc:
        # Outputs:
        #  - vn_ie :
        #     $$
        #     \vn{\n}{\e}{\k-1/2} = \Wlev \vn{\n}{\e}{\k} + (1 - \Wlev) \vn{\n}{\e}{\k-1}, \quad \k \in [1, \nlev)
        #     $$
        #     Interpolate the normal velocity from full to half levels.
        #  - z_kin_hor_e :
        #     $$
        #     \kinehori{\n}{\e}{\k} = \frac{1}{2} \left( \vn{\n}{\e}{\k}^2 + \vt{\n}{\e}{\k}^2 \right), \quad \k \in [1, \nlev)
        #     $$
        #     Compute the horizontal kinetic energy. Exclude the first full level.
        #
        # Inputs:
        #  - $\Wlev$ : wgtfac_e
        #  - $\vn{\n}{\e}{\k}$ : vn
        #  - $\vt{\n}{\e}{\k}$ : vt
        #
        self._interpolate_vn_to_ie_and_compute_ekin_on_edges(
            wgtfac_e=self.metric_state.wgtfac_e,
            vn=prognostic_state.vn,
            vt=diagnostic_state.vt,
            vn_ie=diagnostic_state.vn_ie,
            z_kin_hor_e=z_kin_hor_e,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        if not vn_only:
            self._interpolate_vt_to_interface_edges(
                wgtfac_e=self.metric_state.wgtfac_e,
                vt=diagnostic_state.vt,
                z_vt_ie=z_vt_ie,
                horizontal_start=self._start_edge_lateral_boundary_level_5,
                horizontal_end=self._end_edge_halo_level_2,
                vertical_start=1,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        # scidoc:
        # Outputs:
        #  - z_w_concorr_me :
        #     $$
        #     \wcc{\n}{\e}{\k} = \vn{\n}{\e}{\k} \pdxn{z} + \vt{\n}{\e}{\k} \pdxt{z}, \quad \k \in [\nflatlev, \nlev)
        #     $$
        #     Compute the contravariant correction to the vertical wind due to
        #     terrain-following coordinate. $\pdxn{}$ and $\pdxt{}$ are the
        #     horizontal derivatives along the normal and tangent directions
        #     respectively (eq. 17 in |ICONdycorePaper|).
        #  - vn_ie :
        #     $$
        #     \vn{\n}{\e}{-1/2} = \vn{\n}{\e}{0}
        #     $$
        #     Set the normal wind at model top equal to the normal wind at the
        #     first full level.
        #  - z_vt_ie :
        #     $$
        #     \vt{\n}{\e}{-1/2} = \vt{\n}{\e}{0}
        #     $$
        #     Set the tangential wind at model top equal to the tangential wind
        #     at the first full level.
        #  - z_kin_hor_e :
        #     $$
        #     \kinehori{\n}{\e}{0} = \frac{1}{2} \left( \vn{\n}{\e}{0}^2 + \vt{\n}{\e}{0}^2 \right)
        #     $$
        #     Compute the horizontal kinetic energy on the first full level.
        #
        # Inputs:
        #  - $\vn{\n}{\e}{\k}$ : vn
        #  - $\vt{\n}{\e}{\k}$ : vt
        #  - $\pdxn{z}$ : ddxn_z_full
        #  - $\pdxt{z}$ : ddxt_z_full
        #
        self._fused_stencils_4_5(
            vn=prognostic_state.vn,
            vt=diagnostic_state.vt,
            vn_ie=diagnostic_state.vn_ie,
            z_vt_ie=z_vt_ie,
            z_kin_hor_e=z_kin_hor_e,
            ddxn_z_full=self.metric_state.ddxn_z_full,
            ddxt_z_full=self.metric_state.ddxt_z_full,
            z_w_concorr_me=z_w_concorr_me,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.num_levels,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )
        self._extrapolate_at_top(
            wgtfacq_e=self.metric_state.wgtfacq_e,
            vn=prognostic_state.vn,
            vn_ie=diagnostic_state.vn_ie,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=self.grid.num_levels,
            vertical_end=self.grid.num_levels + 1,
            offset_provider=self.grid.offset_providers,
        )

        if not vn_only:
            self._compute_horizontal_advection_term_for_vertical_velocity(
                vn_ie=diagnostic_state.vn_ie,
                inv_dual_edge_length=self.edge_params.inverse_dual_edge_lengths,
                w=prognostic_state.w,
                z_vt_ie=z_vt_ie,
                inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
                tangent_orientation=self.edge_params.tangent_orientation,
                z_w_v=self.z_w_v,
                z_v_grad_w=self.z_v_grad_w,
                horizontal_start=self._start_edge_lateral_boundary_level_7,
                horizontal_end=self._end_edge_halo,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        # scidoc:
        # Outputs:
        #  - z_ekinh :
        #     $$
        #     \kinehori{\n}{\c}{\k} = \sum_{\offProv{c2e}} \Whor \kinehori{\n}{\e}{\k}
        #     $$
        #     Interpolate the horizonal kinetic energy from edge to cell center.
        #
        # Inputs:
        #  - $\Whor$ : e_bln_c_s
        #  - $\kinehori{\n}{\e}{\k}$ : z_kin_hor_e
        #
        self._interpolate_to_cell_center(
            interpolant=z_kin_hor_e,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            interpolation=self.z_ekinh,
            horizontal_start=self._start_cell_lateral_boundary_level_4,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        # scidoc:
        # Outputs:
        #  - z_w_concorr_mc :
        #     $$
        #     \wcc{\n}{\c}{\k} = \sum_{\offProv{c2e}} \Whor \wcc{\n}{\e}{\k}
        #     $$
        #     Interpolate the contravariant correction from edge to cell center.
        #  - w_concorr_c :
        #     $$
        #     \wcc{\n}{\c}{\k-1/2} = \Wlev \wcc{\n}{\c}{\k} + (1 - \Wlev) \wcc{\n}{\c}{\k-1}, \quad \k \in [\nflatlev+1, \nlev)
        #     $$
        #     Interpolate the contravariant correction from full to half levels.
        #
        # Inputs:
        #  - $\wcc{\n}{\e}{\k}$ : z_w_concorr_me
        #  - $\Whor$ : e_bln_c_s
        #  - $\Wlev$ : wgtfac_c
        #
        self._fused_stencils_9_10(
            z_w_concorr_me=z_w_concorr_me,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            local_z_w_concorr_mc=self.z_w_concorr_mc,
            wgtfac_c=self.metric_state.wgtfac_c,
            w_concorr_c=diagnostic_state.w_concorr_c,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.num_levels,
            horizontal_start=self._start_cell_lateral_boundary_level_4,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        # scidoc:
        # Outputs:
        #  - z_w_con_c :
        #     $$
        #     (\w{\n}{\c}{\k-1/2} - \wcc{\n}{\c}{\k-1/2}) = 
        #     \begin{cases}
        #         \w{\n}{\c}{\k-1/2},                        & \k \in [0, \nflatlev+1)     \\
        #         \w{\n}{\c}{\k-1/2} - \wcc{\n}{\c}{\k-1/2}, & \k \in [\nflatlev+1, \nlev) \\
        #         0,                                         & \k = \nlev
        #     \end{cases}
        #     $$
        #     Subtract the contravariant correction $\wcc{}{}{}$ from the
        #     vertical wind $\w{}{}{}$ in the terrain-following levels. This is
        #     done for convevnience here, instead of directly in the advection
        #     tendency update, because the result needs to be interpolated to
        #     edge centers and full levels for later use.
        #     The papers do not use a new symbol for this variable, and the code
        #     ambiguosly mixes the variable names used for
        #     $\wcc{}{}{}$ and $(\w{}{}{} - \wcc{}{}{})$.
        #     
        # Inputs:
        #  - $\w{\n}{\c}{\k\pm1/2}$ : w
        #  - $\wcc{\n}{\c}{\k\pm1/2}$ : w_concorr_c
        #
        self._fused_stencils_11_to_13(
            w=prognostic_state.w,
            w_concorr_c=diagnostic_state.w_concorr_c,
            local_z_w_con_c=self.z_w_con_c,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.num_levels,
            horizontal_start=self._start_cell_lateral_boundary_level_4,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider={},
        )

        self._fused_stencil_14(
            ddqz_z_half=self.metric_state.ddqz_z_half,
            local_z_w_con_c=self.z_w_con_c,
            local_cfl_clipping=self.cfl_clipping,
            local_vcfl=self.vcfl_dsl,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=self._start_cell_lateral_boundary_level_4,
            horizontal_end=self._end_cell_halo,
            vertical_start=gtx.int32(
                max(3, self.vertical_params.end_index_of_damping_layer - 2) - 1
            ),
            vertical_end=gtx.int32(self.grid.num_levels - 3),
            offset_provider={},
        )

        self._update_levmask_from_cfl_clipping()

        # scidoc:
        # Outputs:
        #  - z_w_con_c_full :
        #     $$
        #     (\w{\n}{\c}{\k} - \wcc{\n}{\c}{\k}) = \frac{1}{2} [ (\w{\n}{\c}{\k-1/2} - \wcc{\n}{\c}{\k-1/2})
        #                                                       + (\w{\n}{\c}{\k+1/2} - \wcc{\n}{\c}{\k+1/2}) ]
        #     $$
        #     Interpolate the vertical wind with contravariant correction from
        #     half to full levels.
        #
        # Inputs:
        #  - $(\w{\n}{\c}{\k\pm1/2} - \wcc{\n}{\c}{\k\pm1/2})$ : z_w_con_c
        #
        self._interpolate_contravariant_vertical_velocity_to_full_levels(
            z_w_con_c=self.z_w_con_c,
            z_w_con_c_full=self.z_w_con_c_full,
            horizontal_start=self._start_cell_lateral_boundary_level_4,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        if not vn_only:
            self._fused_stencils_16_to_17(
                w=prognostic_state.w,
                local_z_v_grad_w=self.z_v_grad_w,
                e_bln_c_s=self.interpolation_state.e_bln_c_s,
                local_z_w_con_c=self.z_w_con_c,
                coeff1_dwdz=self.metric_state.coeff1_dwdz,
                coeff2_dwdz=self.metric_state.coeff2_dwdz,
                ddt_w_adv=diagnostic_state.ddt_w_adv_pc[ntnd],
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=1,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

            self._add_extra_diffusion_for_w_con_approaching_cfl(
                levmask=self.levmask,
                cfl_clipping=self.cfl_clipping,
                owner_mask=self.c_owner_mask,
                z_w_con_c=self.z_w_con_c,
                ddqz_z_half=self.metric_state.ddqz_z_half,
                area=cell_areas,
                geofac_n2s=self.interpolation_state.geofac_n2s,
                w=prognostic_state.w,
                ddt_w_adv=diagnostic_state.ddt_w_adv_pc[ntnd],
                scalfac_exdiff=scalfac_exdiff,
                cfl_w_limit=cfl_w_limit,
                dtime=dtime,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=gtx.int32(
                    max(3, self.vertical_params.end_index_of_damping_layer - 2) - 1
                ),
                vertical_end=gtx.int32(self.grid.num_levels - 3),
                offset_provider=self.grid.offset_providers,
            )

        self.levelmask = self.levmask

        # scidoc:
        # Outputs:
        #  - ddt_vn_apc_pc[ntnd] :
        #     $$
        #     \advvn{\n}{\e}{\k} &&= \pdxn{\kinehori{}{}{}} + \vt{}{}{} (\vortvert{}{}{} + \coriolis{}) + \pdz{\vn{}{}{}} (\w{}{}{} - \wcc{}{}{}) \\
        #                        &&= \Gradn_{\offProv{e2c}} \Cgrad \kinehori{\n}{c}{\k} + \kinehori{\n}{\e}{\k} \Gradn_{\offProv{e2c}} \Cgrad     \\
        #                        &&+ \vt{\n}{\e}{\k} (\coriolis{\e} + 1/2 \sum_{\offProv{e2v}} \vortvert{\n}{\v}{\k})                             \\
        #                        &&+ \frac{\vn{\n}{\e}{\k-1/2} - \vn{\n}{\e}{\k+1/2}}{\Dz{k}}
        #                            \sum_{\offProv{e2c}} \Whor (\w{\n}{\c}{\k} - \wcc{\n}{\c}{\k})
        #     $$
        #     Compute the advective tendency of the normal wind (eq. 13 in
        #     |ICONdycorePaper|).
        #     The edge-normal derivative of the kinetic energy is computed by
        #     combining the first order approximation across adiacent cell
        #     centres (eq. 7 in |BonaventuraRingler2005|) with the edge value of
        #     the kinetic energy (TODO: this needs explaining and a reference).
        #
        # Inputs:
        #  - $\Cgrad$ : coeff_gradekin
        #  - $\kinehori{\n}{\e}{\k}$ : z_kin_hor_e
        #  - $\kinehori{\n}{\c}{\k}$ : z_ekinh
        #  - $\vt{\n}{\e}{\k}$ : vt
        #  - $\coriolis{\e}$ : f_e
        #  - $\vortvert{\n}{\v}{\k}$ : zeta
        #  - $\Whor$ : c_lin_e
        #  - $(\w{\n}{\c}{\k} - \wcc{\n}{\c}{\k})$ : z_w_con_c_full
        #  - $\vn{\n}{\e}{\k\pm1/2}$ : vn_ie
        #  - $\Dz{\k}$ : ddqz_z_full_e
        #
        self._compute_advective_normal_wind_tendency(
            z_kin_hor_e=z_kin_hor_e,
            coeff_gradekin=self.metric_state.coeff_gradekin,
            z_ekinh=self.z_ekinh,
            zeta=self.zeta,
            vt=diagnostic_state.vt,
            f_e=self.edge_params.f_e,
            c_lin_e=self.interpolation_state.c_lin_e,
            z_w_con_c_full=self.z_w_con_c_full,
            vn_ie=diagnostic_state.vn_ie,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            ddt_vn_apc=diagnostic_state.ddt_vn_apc_pc[ntnd],
            horizontal_start=self._start_edge_nudging_level_2,
            horizontal_end=self._end_edge_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        self._add_extra_diffusion_for_normal_wind_tendency_approaching_cfl(
            levelmask=self.levelmask,
            c_lin_e=self.interpolation_state.c_lin_e,
            z_w_con_c_full=self.z_w_con_c_full,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            area_edge=self.edge_params.edge_areas,
            tangent_orientation=self.edge_params.tangent_orientation,
            inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
            zeta=self.zeta,
            geofac_grdiv=self.interpolation_state.geofac_grdiv,
            vn=prognostic_state.vn,
            ddt_vn_apc=diagnostic_state.ddt_vn_apc_pc[ntnd],
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            dtime=dtime,
            horizontal_start=self._start_edge_nudging_level_2,
            horizontal_end=self._end_edge_local,
            vertical_start=gtx.int32(
                max(3, self.vertical_params.end_index_of_damping_layer - 2) - 1
            ),
            vertical_end=gtx.int32(self.grid.num_levels - 4),
            offset_provider=self.grid.offset_providers,
        )

    def _update_levmask_from_cfl_clipping(self):
        self.levmask = gtx.as_field(
            domain=(dims.KDim,), data=(xp.any(self.cfl_clipping.ndarray, 0)), dtype=bool
        )

    def _scale_factors_by_dtime(self, dtime):
        scaled_cfl_w_limit = self.cfl_w_limit / dtime
        scalfac_exdiff = self.scalfac_exdiff / (dtime * (0.85 - scaled_cfl_w_limit * dtime))
        return scaled_cfl_w_limit, scalfac_exdiff

    def run_corrector_step(
        self,
        vn_only: bool,
        diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_state: prognostics.PrognosticState,
        z_kin_hor_e: fa.EdgeKField[float],
        z_vt_ie: fa.EdgeKField[float],
        dtime: float,
        ntnd: int,
        cell_areas: fa.CellField[float],
    ):
        cfl_w_limit, scalfac_exdiff = self._scale_factors_by_dtime(dtime)

        if not vn_only:
            self._mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
                p_cell_in=prognostic_state.w,
                c_intp=self.interpolation_state.c_intp,
                p_vert_out=self.z_w_v,
                horizontal_start=self._start_vertex_lateral_boundary_level_2,
                horizontal_end=self._end_vertex_halo,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        self._mo_math_divrot_rot_vertex_ri_dsl(
            vec_e=prognostic_state.vn,
            geofac_rot=self.interpolation_state.geofac_rot,
            rot_vec=self.zeta,
            horizontal_start=self._start_vertex_lateral_boundary_level_2,
            horizontal_end=self._end_vertex_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        if not vn_only:
            self._compute_horizontal_advection_term_for_vertical_velocity(
                vn_ie=diagnostic_state.vn_ie,
                inv_dual_edge_length=self.edge_params.inverse_dual_edge_lengths,
                w=prognostic_state.w,
                z_vt_ie=z_vt_ie,
                inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
                tangent_orientation=self.edge_params.tangent_orientation,
                z_w_v=self.z_w_v,
                z_v_grad_w=self.z_v_grad_w,
                horizontal_start=self._start_edge_lateral_boundary_level_7,
                horizontal_end=self._end_edge_halo,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        self._interpolate_to_cell_center(
            interpolant=z_kin_hor_e,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            interpolation=self.z_ekinh,
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        self._fused_stencils_11_to_13(
            w=prognostic_state.w,
            w_concorr_c=diagnostic_state.w_concorr_c,
            local_z_w_con_c=self.z_w_con_c,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.num_levels,
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        self._fused_stencil_14(
            ddqz_z_half=self.metric_state.ddqz_z_half,
            local_z_w_con_c=self.z_w_con_c,
            local_cfl_clipping=self.cfl_clipping,
            local_vcfl=self.vcfl_dsl,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_halo,
            vertical_start=gtx.int32(max(3, self.vertical_params.end_index_of_damping_layer - 2)),
            vertical_end=gtx.int32(self.grid.num_levels - 3),
            offset_provider={},
        )

        self._update_levmask_from_cfl_clipping()

        self._interpolate_contravariant_vertical_velocity_to_full_levels(
            z_w_con_c=self.z_w_con_c,
            z_w_con_c_full=self.z_w_con_c_full,
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        self._fused_stencils_16_to_17(
            w=prognostic_state.w,
            local_z_v_grad_w=self.z_v_grad_w,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            local_z_w_con_c=self.z_w_con_c,
            coeff1_dwdz=self.metric_state.coeff1_dwdz,
            coeff2_dwdz=self.metric_state.coeff2_dwdz,
            ddt_w_adv=diagnostic_state.ddt_w_adv_pc[ntnd],
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        self._add_extra_diffusion_for_w_con_approaching_cfl(
            levmask=self.levmask,
            cfl_clipping=self.cfl_clipping,
            owner_mask=self.c_owner_mask,
            z_w_con_c=self.z_w_con_c,
            ddqz_z_half=self.metric_state.ddqz_z_half,
            area=cell_areas,
            geofac_n2s=self.interpolation_state.geofac_n2s,
            w=prognostic_state.w,
            ddt_w_adv=diagnostic_state.ddt_w_adv_pc[ntnd],
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=gtx.int32(max(3, self.vertical_params.end_index_of_damping_layer - 2)),
            vertical_end=gtx.int32(self.grid.num_levels - 4),
            offset_provider=self.grid.offset_providers,
        )

        # This behaviour needs to change for multiple blocks
        self.levelmask = self.levmask

        self._compute_advective_normal_wind_tendency(
            z_kin_hor_e=z_kin_hor_e,
            coeff_gradekin=self.metric_state.coeff_gradekin,
            z_ekinh=self.z_ekinh,
            zeta=self.zeta,
            vt=diagnostic_state.vt,
            f_e=self.edge_params.f_e,
            c_lin_e=self.interpolation_state.c_lin_e,
            z_w_con_c_full=self.z_w_con_c_full,
            vn_ie=diagnostic_state.vn_ie,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            ddt_vn_apc=diagnostic_state.ddt_vn_apc_pc[ntnd],
            horizontal_start=self._start_edge_nudging_level_2,
            horizontal_end=self._end_edge_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        self._add_extra_diffusion_for_normal_wind_tendency_approaching_cfl(
            levelmask=self.levelmask,
            c_lin_e=self.interpolation_state.c_lin_e,
            z_w_con_c_full=self.z_w_con_c_full,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            area_edge=self.edge_params.edge_areas,
            tangent_orientation=self.edge_params.tangent_orientation,
            inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
            zeta=self.zeta,
            geofac_grdiv=self.interpolation_state.geofac_grdiv,
            vn=prognostic_state.vn,
            ddt_vn_apc=diagnostic_state.ddt_vn_apc_pc[ntnd],
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            dtime=dtime,
            horizontal_start=self._start_edge_nudging_level_2,
            horizontal_end=self._end_edge_local,
            vertical_start=gtx.int32(max(3, self.vertical_params.end_index_of_damping_layer - 2)),
            vertical_end=gtx.int32(self.grid.num_levels - 4),
            offset_provider=self.grid.offset_providers,
        )
