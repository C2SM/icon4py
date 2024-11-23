# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py.next import as_field
from gt4py.next.common import Field
from gt4py.next.iterator.builtins import int32

from icon4py.model.atmosphere.dycore.nh_solve.helpers import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.atmosphere.dycore.state_utils.states import (
    DiagnosticStateNonHydro,
    InterpolationState,
    MetricStateNonHydro,
    OutputIntermediateFields,
)
from icon4py.model.atmosphere.dycore.state_utils.utils import _allocate, _allocate_indices
from icon4py.model.atmosphere.dycore.velocity.helpers import (
    add_extra_diffusion_for_normal_wind_tendency_approaching_cfl,
    add_extra_diffusion_for_w_con_approaching_cfl,
    compute_advective_normal_wind_tendency,
    compute_horizontal_advection_term_for_vertical_velocity,
    compute_tangential_wind,
    extrapolate_at_top,
    fused_stencil_14,
    fused_stencils_4_5,
    fused_stencils_9_10,
    fused_stencils_11_to_13,
    fused_stencils_16_to_17,
    interpolate_contravariant_vertical_velocity_to_full_levels,
    interpolate_to_cell_center,
    interpolate_vn_to_ie_and_compute_ekin_on_edges,
    interpolate_vt_to_interface_edges,
    mo_math_divrot_rot_vertex_ri_dsl,
)
from icon4py.model.atmosphere.dycore.nh_solve.helpers import (
    copy_cell_kdim_field_to_vp,
    copy_edge_kdim_field_to_vp,
)
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, VertexDim
from icon4py.model.common.grid.horizontal import EdgeParams, HorizontalMarkerIndex
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.settings import xp
from icon4py.model.common.states.prognostic_state import PrognosticState


class VelocityAdvection:
    def __init__(
        self,
        grid: IconGrid,
        metric_state: MetricStateNonHydro,
        interpolation_state: InterpolationState,
        vertical_params: VerticalModelParams,
        edge_params: EdgeParams,
        owner_mask: Field[[CellDim], bool],
    ):
        self._initialized = False
        self.grid: IconGrid = grid
        self.metric_state: MetricStateNonHydro = metric_state
        self.interpolation_state: InterpolationState = interpolation_state
        self.vertical_params = vertical_params
        self.edge_params = edge_params
        self.c_owner_mask = owner_mask

        self.cfl_w_limit: float = 0.65
        self.scalfac_exdiff: float = 0.05
        self._allocate_local_fields()

        self._initialized = True

    @property
    def initialized(self):
        return self._initialized

    def _allocate_local_fields(self):
        self.z_w_v = _allocate(VertexDim, KDim, is_halfdim=True, grid=self.grid)
        self.z_v_grad_w = _allocate(EdgeDim, KDim, grid=self.grid)
        self.z_ekinh = _allocate(CellDim, KDim, grid=self.grid)
        self.z_w_concorr_mc = _allocate(CellDim, KDim, grid=self.grid)
        self.z_w_con_c = _allocate(CellDim, KDim, is_halfdim=True, grid=self.grid)
        self.zeta = _allocate(VertexDim, KDim, grid=self.grid)
        self.z_w_con_c_full = _allocate(CellDim, KDim, grid=self.grid)
        self.cfl_clipping = _allocate(CellDim, KDim, grid=self.grid, dtype=bool)
        self.levmask = _allocate(KDim, grid=self.grid, dtype=bool)
        self.vcfl_dsl = _allocate(CellDim, KDim, grid=self.grid)
        self.k_field = _allocate_indices(KDim, grid=self.grid, is_halfdim=True)
        self.output_hgrad_kinetic_e = _allocate(EdgeDim, KDim, grid=self.grid)
        self.output_total_vorticity_e = _allocate(EdgeDim, KDim, grid=self.grid)
        self.output_vertical_wind_e = _allocate(EdgeDim, KDim, grid=self.grid)
        self.output_vgrad_vn_e = _allocate(EdgeDim, KDim, grid=self.grid)

    def run_predictor_step(
        self,
        vn_only: bool,
        diagnostic_state: DiagnosticStateNonHydro,
        prognostic_state: PrognosticState,
        z_w_concorr_me: Field[[EdgeDim, KDim], float],
        z_kin_hor_e: Field[[EdgeDim, KDim], float],
        z_vt_ie: Field[[EdgeDim, KDim], float],
        dtime: float,
        ntnd: int,
        cell_areas: Field[[CellDim], float],
        output_intermediate_fields: OutputIntermediateFields,
    ):
        cfl_w_limit, scalfac_exdiff = self._scale_factors_by_dtime(dtime)

        start_vertex_lb_plus1 = self.grid.get_start_index(
            VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1
        )
        end_vertex_local_minus1 = self.grid.get_end_index(
            VertexDim, HorizontalMarkerIndex.local(VertexDim) - 1
        )

        start_edge_lb_plus4 = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4
        )
        start_edge_lb_plus6 = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6
        )
        start_edge_nudging_plus1 = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim) + 1
        )
        end_edge_local = self.grid.get_end_index(EdgeDim, HorizontalMarkerIndex.local(EdgeDim))

        end_edge_local_minus1 = self.grid.get_end_index(
            EdgeDim, HorizontalMarkerIndex.local(EdgeDim) - 1
        )
        end_edge_local_minus2 = self.grid.get_end_index(
            EdgeDim, HorizontalMarkerIndex.local(EdgeDim) - 2
        )

        start_cell_lb_plus3 = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 3
        )
        start_cell_nudging = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.nudging(CellDim)
        )
        end_cell_local = self.grid.get_end_index(CellDim, HorizontalMarkerIndex.local(CellDim))
        end_cell_local_minus1 = self.grid.get_end_index(
            CellDim, HorizontalMarkerIndex.local(CellDim) - 1
        )

        if not vn_only:
            """
            z_w_v (0:nlev-1):
                Compute the vertical wind at cell vertices at half levels by simple area-weighted interpolation.
                When itime_scheme = 4, we just use its value computed at the velocity advection corrector step in the previous substep.
            """
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
                p_cell_in=prognostic_state.w,
                c_intp=self.interpolation_state.c_intp,
                p_vert_out=self.z_w_v,
                horizontal_start=start_vertex_lb_plus1,
                horizontal_end=end_vertex_local_minus1,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        """
        zeta (0:nlev-1):
            Compute the vorticity at cell vertices at full levels using discrete Stokes theorem.
            zeta = integral_circulation dotproduct(V, tangent) / dual_cell_area = sum V_i dual_edge_length_i / dual_cell_area
        """
        mo_math_divrot_rot_vertex_ri_dsl(
            vec_e=prognostic_state.vn,
            geofac_rot=self.interpolation_state.geofac_rot,
            rot_vec=self.zeta,
            horizontal_start=start_vertex_lb_plus1,
            horizontal_end=end_vertex_local_minus1,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        """
        vt (0:nlev-1):
            Compute tangential velocity at full levels (edge center) by RBF interpolation from four neighboring
            edges (diamond shape) and projected to tangential direction.
        """
        compute_tangential_wind(
            vn=prognostic_state.vn,
            rbf_vec_coeff_e=self.interpolation_state.rbf_vec_coeff_e,
            vt=diagnostic_state.vt,
            horizontal_start=start_edge_lb_plus4,
            horizontal_end=end_edge_local_minus2,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        """
        vn_ie (1:nlev-1):
            Compute normal velocity at half levels (edge center) simply by interpolating two neighboring
            normal velocity at full levels.
        z_kin_hor_e (1:nlev-1):
            Compute the horizontal kinetic energy (vn^2 + vt^2)/2 at full levels (edge center).
        """
        interpolate_vn_to_ie_and_compute_ekin_on_edges(
            wgtfac_e=self.metric_state.wgtfac_e,
            vn=prognostic_state.vn,
            vt=diagnostic_state.vt,
            vn_ie=diagnostic_state.vn_ie,
            z_kin_hor_e=z_kin_hor_e,
            horizontal_start=start_edge_lb_plus4,
            horizontal_end=end_edge_local_minus2,
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
            interpolate_vt_to_interface_edges(
                wgtfac_e=self.metric_state.wgtfac_e,
                vt=diagnostic_state.vt,
                z_vt_ie=z_vt_ie,
                horizontal_start=start_edge_lb_plus4,
                horizontal_end=end_edge_local_minus2,
                vertical_start=1,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        """
        z_w_concorr_me (flat_lev:nlev-1):
            Compute contravariant correction (due to terrain-following coordinates) to vertical wind at
            full levels (edge center). The correction is equal to vn dz/dn + vt dz/dt, where t is tangent.
        vn_ie (0):
            Compute normal wind at model top (edge center). It is simply set equal to normal wind.
        z_vt_ie (0):
            Compute tangential wind at model top (edge center). It is simply set equal to tangential wind.
        z_kin_hor_e (0):
            Compute the horizontal kinetic energy (vn^2 + vt^2)/2 at first full level (edge center).
        """
        fused_stencils_4_5(
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
            horizontal_start=start_edge_lb_plus4,
            horizontal_end=end_edge_local_minus2,
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
        extrapolate_at_top(
            wgtfacq_e=self.metric_state.wgtfacq_e,
            vn=prognostic_state.vn,
            vn_ie=diagnostic_state.vn_ie,
            horizontal_start=start_edge_lb_plus4,
            horizontal_end=end_edge_local_minus2,
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
            compute_horizontal_advection_term_for_vertical_velocity(
                vn_ie=diagnostic_state.vn_ie,
                inv_dual_edge_length=self.edge_params.inverse_dual_edge_lengths,
                w=prognostic_state.w,
                z_vt_ie=z_vt_ie,
                inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
                tangent_orientation=self.edge_params.tangent_orientation,
                z_w_v=self.z_w_v,
                z_v_grad_w=self.z_v_grad_w,
                horizontal_start=start_edge_lb_plus6,
                horizontal_end=end_edge_local_minus1,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        """
        z_ekinh (0:nlev-1):
            Interpolate the horizonal kinetic energy (vn^2 + vt^2)/2 at full levels from
            edge center (three neighboring edges) to cell center.
        """
        interpolate_to_cell_center(
            interpolant=z_kin_hor_e,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            interpolation=self.z_ekinh,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_local_minus1,
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
        fused_stencils_9_10(
            z_w_concorr_me=z_w_concorr_me,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            local_z_w_concorr_mc=self.z_w_concorr_mc,
            wgtfac_c=self.metric_state.wgtfac_c,
            w_concorr_c=diagnostic_state.w_concorr_c,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.num_levels,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_local_minus1,
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
        fused_stencils_11_to_13(
            w=prognostic_state.w,
            w_concorr_c=diagnostic_state.w_concorr_c,
            local_z_w_con_c=self.z_w_con_c,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.num_levels,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_local_minus1,
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
        fused_stencil_14(
            local_z_w_con_c=self.z_w_con_c,
            ddqz_z_half=self.metric_state.ddqz_z_half,
            local_cfl_clipping=self.cfl_clipping,
            local_vcfl=self.vcfl_dsl,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_local_minus1,
            vertical_start=int32(max(3, self.vertical_params.index_of_damping_layer - 2) - 1),
            vertical_end=int32(self.grid.num_levels - 3),
            offset_provider={},
        )

        self._update_levmask_from_cfl_clipping()

        """
        z_w_con_c_full (0:nlev-1):
            Compute the vertical wind with contravariant correction at full levels (cell center) by
            taking average from values at neighboring half levels.
            z_w_con_c_full[k] = 0.5 (z_w_con_c[k] + z_w_con_c[k+1])
        """
        interpolate_contravariant_vertical_velocity_to_full_levels(
            z_w_con_c=self.z_w_con_c,
            z_w_con_c_full=self.z_w_con_c_full,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_local_minus1,
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
            fused_stencils_16_to_17(
                w=prognostic_state.w,
                local_z_v_grad_w=self.z_v_grad_w,
                e_bln_c_s=self.interpolation_state.e_bln_c_s,
                local_z_w_con_c=self.z_w_con_c,
                coeff1_dwdz=self.metric_state.coeff1_dwdz,
                coeff2_dwdz=self.metric_state.coeff2_dwdz,
                ddt_w_adv=diagnostic_state.ddt_w_adv_pc[ntnd],
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
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
            add_extra_diffusion_for_w_con_approaching_cfl(
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
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=int32(max(3, self.vertical_params.index_of_damping_layer - 2) - 1),
                vertical_end=int32(self.grid.num_levels - 3),
                offset_provider=self.grid.offset_providers,
            )

        self.levelmask = self.levmask

        """
        ddt_vn_apc_pc[ntnd] (0:nlev-1):
            Compute the advection of normal wind at full levels (edge center).
            ddt_vn_apc_pc[ntnd] = dKEH/dn + vt * (vorticity + coriolis) + w dvn/dz,
            where vn is normal wind, vt is tangential wind, KEH is horizontal kinetic energy.
            z_kin_hor_e is KEH at full levels (edge center).
            zeta is vorticity at full levels (vertex). Its value at edge center is simply taken as average of vorticity at neighboring vertices.
            w, which is vertical wind with contravariant correction, at edge center is obtained by linearly interpolating z_w_con_c_full at neighboring cells.
            dvn/dz, the vertical derivative of normal wind, is computed by first order discretization from vn_ie.
            TODO (Chia Rui): understand the coefficient coeff_gradekin
        """
        compute_advective_normal_wind_tendency(
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
            output_hgrad_kinetic_e=self.output_hgrad_kinetic_e,
            output_total_vorticity_e=self.output_total_vorticity_e,
            output_vertical_wind_e=self.output_vertical_wind_e,
            output_vgrad_vn_e=self.output_vgrad_vn_e,
            ddt_vn_apc=diagnostic_state.ddt_vn_apc_pc[ntnd],
            horizontal_start=start_edge_nudging_plus1,
            horizontal_end=end_edge_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )
        if output_intermediate_fields is not None:
            copy_edge_kdim_field_to_vp(
                field=self.output_hgrad_kinetic_e,
                field_copy=output_intermediate_fields.output_velocity_predictor_hgrad_kinetic_e,
                horizontal_start=int32(0),
                horizontal_end=end_edge_local,
                vertical_start=int32(0),
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
            copy_edge_kdim_field_to_vp(
                field=diagnostic_state.vt,
                field_copy=output_intermediate_fields.output_velocity_predictor_tangent_wind,
                horizontal_start=int32(0),
                horizontal_end=end_edge_local,
                vertical_start=int32(0),
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
            copy_edge_kdim_field_to_vp(
                field=self.output_total_vorticity_e,
                field_copy=output_intermediate_fields.output_velocity_predictor_total_vorticity_e,
                horizontal_start=int32(0),
                horizontal_end=end_edge_local,
                vertical_start=int32(0),
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
            copy_edge_kdim_field_to_vp(
                field=self.output_vertical_wind_e,
                field_copy=output_intermediate_fields.output_velocity_predictor_vertical_wind_e,
                horizontal_start=int32(0),
                horizontal_end=end_edge_local,
                vertical_start=int32(0),
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
            copy_edge_kdim_field_to_vp(
                field=self.output_vgrad_vn_e,
                field_copy=output_intermediate_fields.output_velocity_predictor_vgrad_vn_e,
                horizontal_start=int32(0),
                horizontal_end=end_edge_local,
                vertical_start=int32(0),
                vertical_end=self.grid.num_levels,
                offset_provider={},
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
        add_extra_diffusion_for_normal_wind_tendency_approaching_cfl(
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
            horizontal_start=start_edge_nudging_plus1,
            horizontal_end=end_edge_local,
            vertical_start=int32(max(3, self.vertical_params.index_of_damping_layer - 2) - 1),
            vertical_end=int32(self.grid.num_levels - 4),
            offset_provider=self.grid.offset_providers,
        )

    def _update_levmask_from_cfl_clipping(self):
        self.levmask = as_field(
            domain=(KDim,), data=(xp.any(self.cfl_clipping.ndarray, 0)), dtype=bool
        )

    def _scale_factors_by_dtime(self, dtime):
        scaled_cfl_w_limit = self.cfl_w_limit / dtime
        scalfac_exdiff = self.scalfac_exdiff / (dtime * (0.85 - scaled_cfl_w_limit * dtime))
        return scaled_cfl_w_limit, scalfac_exdiff

    def run_corrector_step(
        self,
        vn_only: bool,
        diagnostic_state: DiagnosticStateNonHydro,
        prognostic_state: PrognosticState,
        z_kin_hor_e: Field[[EdgeDim, KDim], float],
        z_vt_ie: Field[[EdgeDim, KDim], float],
        dtime: float,
        ntnd: int,
        cell_areas: Field[[CellDim], float],
        output_intermediate_fields: OutputIntermediateFields,
    ):
        cfl_w_limit, scalfac_exdiff = self._scale_factors_by_dtime(dtime)

        start_vertex_lb_plus1 = self.grid.get_start_index(
            VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1
        )
        end_vertex_local_minus1 = self.grid.get_end_index(
            VertexDim, HorizontalMarkerIndex.local(VertexDim) - 1
        )

        start_edge_lb_plus6 = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6
        )
        start_edge_nudging_plus1 = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim) + 1
        )
        end_edge_local = self.grid.get_end_index(EdgeDim, HorizontalMarkerIndex.local(EdgeDim))
        end_edge_local_minus1 = self.grid.get_end_index(
            EdgeDim, HorizontalMarkerIndex.local(EdgeDim) - 1
        )

        start_cell_lb_plus3 = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 3
        )
        start_cell_nudging = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.nudging(CellDim)
        )
        end_cell_local = self.grid.get_end_index(CellDim, HorizontalMarkerIndex.local(CellDim))
        end_cell_lb_minus1 = self.grid.get_end_index(
            CellDim, HorizontalMarkerIndex.local(CellDim) - 1
        )

        if not vn_only:
            """
            z_w_v (0:nlev-1):
                Compute the vertical wind at cell vertices at half levels by simple area-weighted interpolation.
            """
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
                p_cell_in=prognostic_state.w,
                c_intp=self.interpolation_state.c_intp,
                p_vert_out=self.z_w_v,
                horizontal_start=start_vertex_lb_plus1,
                horizontal_end=end_vertex_local_minus1,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        """
        zeta (0:nlev-1):
            Compute the vorticity at cell vertices at full levels using discrete Stokes theorem.
            zeta = integral_circulation dotproduct(V, tangent) / dual_cell_area = sum V_i dual_edge_length_i / dual_cell_area
        """
        mo_math_divrot_rot_vertex_ri_dsl(
            vec_e=prognostic_state.vn,
            geofac_rot=self.interpolation_state.geofac_rot,
            rot_vec=self.zeta,
            horizontal_start=start_vertex_lb_plus1,
            horizontal_end=end_vertex_local_minus1,
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
            compute_horizontal_advection_term_for_vertical_velocity(
                vn_ie=diagnostic_state.vn_ie,
                inv_dual_edge_length=self.edge_params.inverse_dual_edge_lengths,
                w=prognostic_state.w,
                z_vt_ie=z_vt_ie,
                inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
                tangent_orientation=self.edge_params.tangent_orientation,
                z_w_v=self.z_w_v,
                z_v_grad_w=self.z_v_grad_w,
                horizontal_start=start_edge_lb_plus6,
                horizontal_end=end_edge_local_minus1,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        """
        z_ekinh (0:nlev-1):
            Interpolate the horizonal kinetic energy (vn^2 + vt^2)/2 at full levels from
            edge center (three neighboring edges) to cell center.
        """
        interpolate_to_cell_center(
            interpolant=z_kin_hor_e,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            interpolation=self.z_ekinh,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_lb_minus1,
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
        fused_stencils_11_to_13(
            w=prognostic_state.w,
            w_concorr_c=diagnostic_state.w_concorr_c,
            local_z_w_con_c=self.z_w_con_c,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.num_levels,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_lb_minus1,
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
        fused_stencil_14(
            local_z_w_con_c=self.z_w_con_c,
            ddqz_z_half=self.metric_state.ddqz_z_half,
            local_cfl_clipping=self.cfl_clipping,
            local_vcfl=self.vcfl_dsl,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_lb_minus1,
            vertical_start=int32(max(3, self.vertical_params.index_of_damping_layer - 2)),
            vertical_end=int32(self.grid.num_levels - 3),
            offset_provider={},
        )

        self._update_levmask_from_cfl_clipping()

        """
        z_w_con_c_full (0:nlev-1):
            Compute the vertical wind with contravariant correction at full levels (cell center) by
            taking average of from it values at neighboring half levels.
            z_w_con_c_full[k] = 0.5 (z_w_con_c[k] + z_w_con_c[k+1])
        """
        interpolate_contravariant_vertical_velocity_to_full_levels(
            z_w_con_c=self.z_w_con_c,
            z_w_con_c_full=self.z_w_con_c_full,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_lb_minus1,
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
        fused_stencils_16_to_17(
            w=prognostic_state.w,
            local_z_v_grad_w=self.z_v_grad_w,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            local_z_w_con_c=self.z_w_con_c,
            coeff1_dwdz=self.metric_state.coeff1_dwdz,
            coeff2_dwdz=self.metric_state.coeff2_dwdz,
            ddt_w_adv=diagnostic_state.ddt_w_adv_pc[ntnd],
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
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
        add_extra_diffusion_for_w_con_approaching_cfl(
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
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=int32(max(3, self.vertical_params.index_of_damping_layer - 2)),
            vertical_end=int32(self.grid.num_levels - 4),
            offset_provider=self.grid.offset_providers,
        )

        # This behaviour needs to change for multiple blocks
        self.levelmask = self.levmask

        """
        ddt_vn_apc_pc[ntnd] (0:nlev-1):
            Compute the advection of normal wind at full levels (edge center).
            ddt_vn_apc_pc[ntnd] = dKEH/dn + vt * (vorticity + coriolis) + w dvn/dz,
            where vn is normal wind, vt is tangential wind, KEH is horizontal kinetic energy.
            z_kin_hor_e is KEH at full levels (edge center).
            zeta is vorticity at full levels (vertex). Its value at edge center is simply taken as average of vorticity at neighboring vertices.
            w, which is vertical wind with contravariant correction, at edge center is obtained by linearly interpolating z_w_con_c_full at neighboring cells.
            dvn/dz, the vertical derivative of normal wind, is computed by first order discretization from vn_ie.
            TODO (Chia Rui): understand the coefficient coeff_gradekin
        """
        compute_advective_normal_wind_tendency(
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
            output_hgrad_kinetic_e=self.output_hgrad_kinetic_e,
            output_total_vorticity_e=self.output_total_vorticity_e,
            output_vertical_wind_e=self.output_vertical_wind_e,
            output_vgrad_vn_e=self.output_vgrad_vn_e,
            ddt_vn_apc=diagnostic_state.ddt_vn_apc_pc[ntnd],
            horizontal_start=start_edge_nudging_plus1,
            horizontal_end=end_edge_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )
        if output_intermediate_fields is not None:
            copy_edge_kdim_field_to_vp(
                field=self.output_hgrad_kinetic_e,
                field_copy=output_intermediate_fields.output_velocity_corrector_hgrad_kinetic_e,
                horizontal_start=int32(0),
                horizontal_end=end_edge_local,
                vertical_start=int32(0),
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
            copy_edge_kdim_field_to_vp(
                field=diagnostic_state.vt,
                field_copy=output_intermediate_fields.output_velocity_corrector_tangent_wind,
                horizontal_start=int32(0),
                horizontal_end=end_edge_local,
                vertical_start=int32(0),
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
            copy_edge_kdim_field_to_vp(
                field=self.output_total_vorticity_e,
                field_copy=output_intermediate_fields.output_velocity_corrector_total_vorticity_e,
                horizontal_start=int32(0),
                horizontal_end=end_edge_local,
                vertical_start=int32(0),
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
            copy_edge_kdim_field_to_vp(
                field=self.output_vertical_wind_e,
                field_copy=output_intermediate_fields.output_velocity_corrector_vertical_wind_e,
                horizontal_start=int32(0),
                horizontal_end=end_edge_local,
                vertical_start=int32(0),
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
            copy_edge_kdim_field_to_vp(
                field=self.output_vgrad_vn_e,
                field_copy=output_intermediate_fields.output_velocity_corrector_vgrad_vn_e,
                horizontal_start=int32(0),
                horizontal_end=end_edge_local,
                vertical_start=int32(0),
                vertical_end=self.grid.num_levels,
                offset_provider={},
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
        add_extra_diffusion_for_normal_wind_tendency_approaching_cfl(
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
            horizontal_start=start_edge_nudging_plus1,
            horizontal_end=end_edge_local,
            vertical_start=int32(max(3, self.vertical_params.index_of_damping_layer - 2)),
            vertical_end=int32(self.grid.num_levels - 4),
            offset_provider=self.grid.offset_providers,
        )
