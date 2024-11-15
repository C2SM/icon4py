# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import backend

import icon4py.model.atmosphere.dycore.velocity.velocity_advection_program as velocity_prog
import icon4py.model.common.grid.geometry as geometry
from icon4py.model.atmosphere.dycore.add_extra_diffusion_for_normal_wind_tendency_approaching_cfl import (
    add_extra_diffusion_for_normal_wind_tendency_approaching_cfl,
)
from icon4py.model.atmosphere.dycore.add_extra_diffusion_for_w_con_approaching_cfl import (
    add_extra_diffusion_for_w_con_approaching_cfl,
)
from icon4py.model.atmosphere.dycore.compute_advective_normal_wind_tendency import (
    compute_advective_normal_wind_tendency,
)
from icon4py.model.atmosphere.dycore.compute_horizontal_advection_term_for_vertical_velocity import (
    compute_horizontal_advection_term_for_vertical_velocity,
)
from icon4py.model.atmosphere.dycore.compute_tangential_wind import compute_tangential_wind
from icon4py.model.atmosphere.dycore.interpolate_contravariant_vertical_velocity_to_full_levels import (
    interpolate_contravariant_vertical_velocity_to_full_levels,
)
from icon4py.model.atmosphere.dycore.interpolate_to_cell_center import (
    interpolate_to_cell_center,
)
from icon4py.model.atmosphere.dycore.interpolate_vn_to_ie_and_compute_ekin_on_edges import (
    interpolate_vn_to_ie_and_compute_ekin_on_edges,
)
from icon4py.model.atmosphere.dycore.interpolate_vt_to_interface_edges import (
    interpolate_vt_to_interface_edges,
)
from icon4py.model.atmosphere.dycore.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.atmosphere.dycore.mo_math_divrot_rot_vertex_ri_dsl import (
    mo_math_divrot_rot_vertex_ri_dsl,
)
from icon4py.model.atmosphere.dycore.state_utils import states as solve_nh_states
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, vertical as v_grid
from icon4py.model.common.settings import xp
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc


class VelocityAdvection:
    def __init__(
        self,
        grid: icon_grid.IconGrid,
        metric_state: solve_nh_states.MetricStateNonHydro,
        interpolation_state: solve_nh_states.InterpolationState,
        vertical_params: v_grid.VerticalGrid,
        edge_params: geometry.EdgeParams,
        owner_mask: fa.CellField[bool],
        backend: backend.Backend,
    ):
        self.grid: icon_grid.IconGrid = grid
        self._backend = backend
        self.metric_state: solve_nh_states.MetricStateNonHydro = metric_state
        self.interpolation_state: solve_nh_states.InterpolationState = interpolation_state
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
        self._fused_stencils_4_5 = velocity_prog.fused_stencils_4_5.with_backend(self._backend)
        self._extrapolate_at_top = velocity_prog.extrapolate_at_top.with_backend(self._backend)
        self._compute_horizontal_advection_term_for_vertical_velocity = (
            compute_horizontal_advection_term_for_vertical_velocity.with_backend(self._backend)
        )
        self._interpolate_to_cell_center = interpolate_to_cell_center.with_backend(self._backend)
        self._fused_stencils_9_10 = velocity_prog.fused_stencils_9_10.with_backend(self._backend)
        self._fused_stencils_11_to_13 = velocity_prog.fused_stencils_11_to_13.with_backend(
            self._backend
        )
        self._fused_stencil_14 = velocity_prog.fused_stencil_14.with_backend(self._backend)
        self._interpolate_contravariant_vertical_velocity_to_full_levels = (
            interpolate_contravariant_vertical_velocity_to_full_levels.with_backend(self._backend)
        )
        self._fused_stencils_16_to_17 = velocity_prog.fused_stencils_16_to_17.with_backend(
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
        diagnostic_state: solve_nh_states.DiagnosticStateNonHydro,
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
            """
            z_w_v (0:nlev-1):
                Compute the vertical wind at cell vertices at half levels by simple area-weighted interpolation.
                When itime_scheme = 4, we just use its value computed at the velocity advection corrector step in the previous substep.
            """
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

        """
        zeta (0:nlev-1):
            Compute the vorticity at cell vertices at full levels using discrete Stokes theorem.
            zeta = integral_circulation dotproduct(V, tangent) / dual_cell_area = sum V_i dual_edge_length_i / dual_cell_area
        """
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
        #     \vt{\n}{\e}{\k} = \sum_{\offProv{e2c2e}} \Wrbf \vn{\n}{\e}{\k}, \qquad \k \in [0, \nlev)
        #     $$
        #     Compute the tangential velocity by RBF interpolation from four neighboring
        #     edges (diamond shape) and projected to tangential direction.
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
        #     \vn{\n}{\e}{\k-1/2} = \Wlev \vn{\n}{\e}{\k} + (1 - \Wlev) \vn{\n}{\e}{\k-1}, \qquad \k \in [1, \nlev)
        #     $$
        #     Linearly interpolate the normal velocity from full levels to half levels.
        #  - z_kin_hor_e :
        #     $$
        #     \kinehori{\n}{\e}{\k} = \frac{1}{2} \left( \vn{\n}{\e}{\k}^2 + \vt{\n}{\e}{\k}^2 \right), \qquad \k \in [1, \nlev)
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
            """
            z_vt_ie (1:nlev-1):
                Compute tangential velocity at half levels (edge center) simply by interpolating two neighboring
                tangential velocity at full levels.
                When itime_scheme = 4, we just use its value computed at the solve nonhydro predictor step after the intermediate velocity is obtained in the previous substep.
            """
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
        #     \wcc{\n}{\e}{\k} = \vn{\n}{\e}{\k} \pdxn{z} + \vt{\n}{\e}{\k} \pdxt{z}, \qquad \k \in [\nflatlev, \nlev)
        #     $$
        #     Compute the contravariant correction (due to terrain-following
        #     coordinates) to vertical wind. Note that here $\pdxt{}$ is the
        #     horizontal derivative along the tangent direction (see eq. 17 in
        #     |ICONdycorePaper|).
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
        """
        vn_ie (nlev):
            Compute normal wind at ground level (edge center) by quadratic extrapolation.
            ---------------  z4
                   z3'
            ---------------  z3
                   z2'
            ---------------  z2
                   z1'
            ---------------  z1 (surface)
            ///////////////
            The three reference points for extrapolation are at z2, z2', and z3'. Value at z1 is
            then obtained by quadratic interpolation polynomial based on these three points.
        """
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
            """
            z_v_grad_w (0:nlev-1):
                Compute horizontal advection of vertical wind at half levels (edge center) by first order
                discretization.
                z_v_grad_w = vn dw/dn + vt dw/dt, t is tangent. The vertical wind at half levels at vertices is
                z_w_v which is computed at the very beginning. It also requires vn and vt at half levels at edge center
                (vn_ie and z_vt_ie, respectively) computed above.
            """
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
        #     \kinehori{\n}{\c}{\k} = \sum_{\offProv{c2e}} \Whor \kinehori{\n}{\e}{\k}, \qquad \k \in [0, \nlev)
        #     $$
        #     Interpolate the horizonal kinetic energy from edge center to cell
        #     center (three neighboring edges).
        #
        # Inputs:
        #  - $\Whor$ : wgtfac_c_s
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

        """
        z_w_concorr_mc (flat_lev:nlev-1):
            Interpolate the contravariant correction (vn dz/dn + vt dz/dt, where t is tangent) at full levels from
            edge center (three neighboring edges), which is z_w_concorr_me, to cell center based on
            bilinear interpolation in a triangle.
        w_concorr_c (flat_lev+1:nlev-1):
            Interpolate contravariant correction at cell center from full levels, which is
            z_w_concorr_mc computed above, to half levels using simple linear interpolation.
        """
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

        """
        z_w_con_c (0:nlev):
            This is the vertical wind with contravariant correction at half levels (cell center).
            It is first simply set equal to vertical wind, w, from level 0 to nlev-1.
            At level nlev, it is first set to zero.
            From flat_lev+1 to nlev-1, it is subtracted from contravariant correction, which is w_concorr_c computed
            previously, at half levels at cell center. TODO (Chia Rui): Check why is it subtraction.
        """
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

        """
        cfl_clipping (max(3,damp_lev-2)-1:nlev-4):
            A boolean that detects whether clf condition in vertical direction is broken at half levels (cell center).
            When w > clf_w_limit * dz, dz = cell_center_height_k+1 - cell_center_height and w is vertical
            wind with contravariant correction (z_w_con_c), it is set to True, else False.
        vcfl_dsl (max(3,damp_lev-2)-1:nlev-4):
            This is w * dt / dz when cfl_clipping is True, where dt is time step,
            dz = cell_center_height_k+1 - cell_center_height, and w is vertical wind with
            contravariant correction (z_w_con_c). Else, it is zero.
        z_w_con_c (max(3,damp_lev-2)-1:nlev-4):
            Modified vertical wind with contravariant correction at half levels (cell center) according to
            whether cfl condition is broken (cfl_clipping) and vertical wind is too large (|vcfl_dsl| > 0.85).
            When both cfl_clipping is True and vcfl_dsl > 0.85 (vcfl_dsl < -0.85), it is modified to
            0.85 * dz / dt (-0.85 * dz / dt).
        """
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

        """
        z_w_con_c_full (0:nlev-1):
            Compute the vertical wind with contravariant correction at full levels (cell center) by
            taking average from values at neighboring half levels.
            z_w_con_c_full[k] = 0.5 (z_w_con_c[k] + z_w_con_c[k+1])
        """
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
            """
            ddt_w_adv_pc[ntnd] (1:nlev-1):
                Compute the advection of vertical wind (vn dw/dn + vt dw/dt + w dw/dz, t is tangent) at half levels (cell center).
                The vertical derivative is obtained by first order discretization.
                vn dw/dn + vt dw/dt at cell center is linearly interpolated from values at neighboring edge centers (z_v_grad_w).
            """
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

            """
            ddt_w_adv_pc[ntnd] (max(3,damp_lev-2)-1:nlev-4):
                Add diffusion to the vertical advection of vertical wind (vn dw/dn + vt dw/dt + w dw/dz, t is tangent) at half levels (cell center) according
                to whether [condition_1_2_3] 1) its cell is owned by this process, 2) levmask is True, and, 3) cfl_clipping is
                True are all satisfied.

                levmask at half levels is True when there is a cell whose cfl_clipping is True at a particular height.

                diffusion_coeff = scalfac_exdiff * minimum(0.85 - cfl_w_limit * dt, z_w_con_c * dt / dz - cfl_w_limit * dt)
                if condition_1_2_3 is True, else zero.

                ddt_w_adv_pc[ntnd] = ddt_w_adv_pc[ntnd] + cell_area Laplacian(diffusion_coeff) if [condition_1_2_3] is
                True.
            """
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
        #     \advvn{\n}{\e}{\k} &&= \pdxn{\kinehori{\n}{}{}} + \vt{\n}{}{} (\vortvert{\n}{}{} + \coriolis{}) + \pdz{\vn{\n}{}{}} \w{\n}{}{}, \qquad \k \in [0, \nlev) \\
        #                        &&= \kinehori{\n}{\e}{\k} (\Cgrad_0 - \Cgrad_1) + \kinehori{\n}{\e2\c\ 1}{\k} \Cgrad_1 - \kinehori{\n}{\e2\c\ 0}{\k} \Cgrad_0 \\
        #                        &&+ \vt{\n}{\e}{\k} (\coriolis{\e} + 0.5 \sum_{\offProv{e2v}} \vortvert{\n}{\v}{\k}) \\
        #                        &&+ \frac{\vn{\n}{\e}{\k-1/2} - \vn{\n}{\e}{\k+1/2}}{\Dz{k}}
        #                            \sum_{\offProv{e2c}} \Whor \wcc{\n}{\c}{\k}
        #     $$
        #     Compute the advective tendency of the normal wind.
        #     $\vortvert{}{}{}$ is the vorticity, its value at edge center is computed as average of the values on the neighboring vertices.
        #     $\wcc{}{}{}$ is the vertical wind with contravariant correction, its value at edge center is computed as linear interpolation from the neighboring cells.
        #     The vertical derivative of normal wind is computed as first order approximation from the values on half levels.
        #     TODO: understand the coefficient coeff_gradekin and come up with better symbol and expression.
        #
        # Inputs:
        #  - $\Cgrad$ : coeff_gradekin
        #  - $\kinehori{\n}{\e}{\k}$ : z_kin_hor_e
        #  - $\vt{\n}{\e}{\k}$ : vt
        #  - $\coriolis{\e}$ : f_e
        #  - $\vortvert{\n}{\v}{\k}$ : zeta
        #  - $\Whor$ : c_lin_e
        #  - $\wcc{\n}{\c}{\k}$ : z_w_con_c_full
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

        """
        ddt_vn_apc_pc[ntnd] (max(3,damp_lev-2)-1:nlev-5):
            Add diffusion to the advection of normal wind at full levels (edge center) according
            to whether [condition_1_2] 1) levmask at half level up or down is True, and, 2) absolute vertical wind with
            contravariant correction at full levels (edge center) is larger than a limit are all satisfied. The limit
            is set to cfl_w_limit * ddqz_z_full_e, where ddqz_z_full_e is layer thickness at edge center obtained by
            lienar interpolation from neighboring cells.

            levmask at half levels is True when there is a cell whose cfl_clipping is True at a particular height.

            The vertical wind with contravariant correction at full levels (edge center), dentoed by w_con_e, is obtained by linear
            interpolation from neighboring cells (from z_w_con_c_full).

            diffusion_coeff = scalfac_exdiff * minimum(0.85 - cfl_w_limit * dt, w_con_e * dt / dz - cfl_w_limit * dt)
            if condition_1_2 is True, else zero.

            ddt_vn_apc_pc[ntnd] = ddt_vn_apc_pc[ntnd] + diffusion_coeff * edge_area * dotproduct(Laplacian(v), normal_direction) if
            [condition_1_2] is True.
            dotproduct(Laplacian(v), normal_direction) = Del(normal_direction) div(v) + Del(tangent_direction) vorticity, see eq B.48 in Hui Wan's thesis.
            Del(tangent_direction) vorticity is obtained by first order discretization of derivative from zeta.
        """
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
        diagnostic_state: solve_nh_states.DiagnosticStateNonHydro,
        prognostic_state: prognostics.PrognosticState,
        z_kin_hor_e: fa.EdgeKField[float],
        z_vt_ie: fa.EdgeKField[float],
        dtime: float,
        ntnd: int,
        cell_areas: fa.CellField[float],
    ):
        cfl_w_limit, scalfac_exdiff = self._scale_factors_by_dtime(dtime)

        if not vn_only:
            """
            z_w_v (0:nlev-1):
                Compute the vertical wind at cell vertices at half levels by simple area-weighted interpolation.
            """
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

        """
        zeta (0:nlev-1):
            Compute the vorticity at cell vertices at full levels using discrete Stokes theorem.
            zeta = integral_circulation dotproduct(V, tangent) / dual_cell_area = sum V_i dual_edge_length_i / dual_cell_area
        """
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
            """
            z_v_grad_w (0:nlev-1):
                Compute horizontal advection of vertical wind at half levels (edge center) by first order
                discretization.
                z_v_grad_w = vn dw/dn + vt dw/dt, t is tangent. The vertical wind at half levels at vertices is
                z_w_v which is computed at the very beginning. It also requires vn and vt at half levels at edge center
                (vn_ie and z_vt_ie, respectively) computed above.
            """
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

        """
        z_ekinh (0:nlev-1):
            Interpolate the horizonal kinetic energy (vn^2 + vt^2)/2 at full levels from
            edge center (three neighboring edges) to cell center.
        """
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

        """
        z_w_con_c (0:nlev):
            This is the vertical wind with contravariant correction at half levels (cell center).
            It is first simply set equal to vertical wind, w, from level 0 to nlev-1.
            At level nlev, it is first set to zero.
            From flat_lev+1 to nlev-1, it is subtracted from contravariant correction, which is w_concorr_c computed
            previously, at half levels at cell center. TODO (Chia Rui): Check why is it subtraction.
        """
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

        """
        cfl_clipping (max(3,damp_lev-2)-1:nlev-4):
            A boolean that detects whether clf condition in vertical direction is broken at half levels (cell center).
            When w > clf_w_limit * dz, dz = cell_center_height_k+1 - cell_center_height and w is vertical
            wind with contravariant correction (z_w_con_c), it is set to True, else False.
        vcfl_dsl (max(3,damp_lev-2)-1:nlev-4):
            This is w * dt / dz when cfl_clipping is True, where dt is time step,
            dz = cell_center_height_k+1 - cell_center_height, and w is vertical wind with
            contravariant correction (z_w_con_c). Else, it is zero.
        z_w_con_c (max(3,damp_lev-2)-1:nlev-4):
            Modified vertical wind with contravariant correction at half levels (cell center) according to
            whether cfl condition is broken (cfl_clipping) and vertical wind is too large (|vcfl_dsl| > 0.85).
            When both cfl_clipping is True and vcfl_dsl > 0.85 (vcfl_dsl < -0.85), it is modified to
            0.85 * dz / dt (-0.85 * dz / dt).
        """
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

        """
        z_w_con_c_full (0:nlev-1):
            Compute the vertical wind with contravariant correction at full levels (cell center) by
            taking average of from it values at neighboring half levels.
            z_w_con_c_full[k] = 0.5 (z_w_con_c[k] + z_w_con_c[k+1])
        """
        self._interpolate_contravariant_vertical_velocity_to_full_levels(
            z_w_con_c=self.z_w_con_c,
            z_w_con_c_full=self.z_w_con_c_full,
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        """
        ddt_w_adv_pc[ntnd] (1:nlev-1):
            Compute the advection of vertical wind (vn dw/dn + vt dw/dt + w dw/dz, t is tangent) at half levels (cell center).
            The vertical derivative is obtained by first order discretization.
            vn dw/dn + vt dw/dt at cell center is linearly interpolated from values at neighboring edge centers (z_v_grad_w).
        """
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

        """
        ddt_w_adv_pc[ntnd] (max(3,damp_lev-2)-1:nlev-4):
            Add diffusion to the advection of vertical wind (vn dw/dn + vt dw/dt + w dw/dz) at half levels (cell center) according
            to whether [condition_1_2_3] 1) its cell is owned by this process, 2) levmask is True, and, 3) cfl_clipping is
            True are all satisfied.

            levmask at half levels is True when there is a cell whose cfl_clipping is True at a particular height.

            diffusion_coeff = scalfac_exdiff * minimum(0.85 - cfl_w_limit * dt, z_w_con_c * dt / dz - cfl_w_limit * dt)
            if condition_1_2_3 is True, else zero.

            ddt_w_adv_pc[ntnd] = ddt_w_adv_pc[ntnd] + cell_area Laplacian(diffusion_coeff) if [condition_1_2_3] is
            True.
        """
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

        """
        ddt_vn_apc_pc[ntnd] (0:nlev-1):
            Compute the advection of normal wind at full levels (edge center).
            ddt_vn_apc_pc[ntnd] = dKEH/dn + vt * (vorticity + coriolis) + wdvn/dz,
            where vn is normal wind, vt is tangential wind, KEH is horizontal kinetic energy.
            z_kin_hor_e is KEH at full levels (edge center).
            zeta is vorticity at full levels (vertex). Its value at edge center is simply taken as average of vorticity at neighboring vertices.
            w, which is vertical wind with contravariant correction, at edge center is obtained by linearly interpolating z_w_con_c_full at neighboring cells.
            dvn/dz, the vertical derivative of normal wind, is computed by first order discretization from vn_ie.
            TODO (Chia Rui): understand the coefficient coeff_gradekin
        """
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

        """
        ddt_vn_apc_pc[ntnd] (max(3,damp_lev-2)-1:nlev-5):
            Add diffusion to the advection of normal wind at full levels (edge center) according
            to whether [condition_1_2] 1) levmask at half level up or down is True, and, 2) absolute vertical wind with
            contravariant correction at full levels (edge center) is larger than a limit are all satisfied. The limit
            is set to cfl_w_limit * ddqz_z_full_e, where ddqz_z_full_e is layer thickness at edge center obtained by
            lienar interpolation from neighboring cells.

            levmask at half levels is True when there is a cell whose cfl_clipping is True at a particular height.

            The vertical wind with contravariant correction at full levels (edge center), dentoed by w_con_e, is obtained by linear
            interpolation from neighboring cells (from z_w_con_c_full).

            diffusion_coeff = scalfac_exdiff * minimum(0.85 - cfl_w_limit * dt, w_con_e * dt / dz - cfl_w_limit * dt)
            if condition_1_2 is True, else zero.

            ddt_vn_apc_pc[ntnd] = ddt_vn_apc_pc[ntnd] + diffusion_coeff * edge_area * dotproduct(Laplacian(v), normal_direction) if
            [condition_1_2] is True.
            dotproduct(Laplacian(v), normal_direction) = Del(normal_direction) div(v) + Del(tangent_direction) vorticity, see eq B.48 in Hui Wan's thesis.
            Del(tangent_direction) vorticity is obtained by first order discretization of derivative from zeta.
        """
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
