# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest
from gt4py.next import gtx

from icon4py.model.atmosphere.dycore.fused_velocity_advection_stencil_1_to_7 import (
    fused_velocity_advection_stencil_1_to_7,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc

from .test_compute_contravariant_correction import compute_contravariant_correction_numpy
from .test_compute_horizontal_advection_term_for_vertical_velocity import (
    compute_horizontal_advection_term_for_vertical_velocity_numpy,
)
from .test_compute_horizontal_kinetic_energy import compute_horizontal_kinetic_energy_numpy
from .test_compute_tangential_wind import compute_tangential_wind_numpy
from .test_extrapolate_at_top import extrapolate_at_top_numpy
from .test_interpolate_vn_to_ie_and_compute_ekin_on_edges import (
    interpolate_vn_to_ie_and_compute_ekin_on_edges_numpy,
)
from .test_interpolate_vt_to_interface_edges import interpolate_vt_to_interface_edges_numpy
from .test_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy,
)


class TestFusedVelocityAdvectionStencil1To7(StencilTest):
    PROGRAM = fused_velocity_advection_stencil_1_to_7
    OUTPUTS = (
        "vt",
        "vn_ie",
        "z_kin_hor_e",
        "z_w_concorr_me",
        "z_v_grad_w",
    )

    @staticmethod
    def _fused_velocity_advection_stencil_1_to_6_numpy(
        grid,
        vn,
        rbf_vec_coeff_e,
        wgtfac_e,
        ddxn_z_full,
        ddxt_z_full,
        z_w_concorr_me,
        wgtfacq_e,
        nflatlev,
        z_vt_ie,
        vt,
        vn_ie,
        z_kin_hor_e,
        k,
        nlev,
        lvn_only,
    ):
        k = k[np.newaxis, :]
        k_nlev = k[:, :-1]

        condition1 = k_nlev < nlev
        vt = np.where(
            condition1,
            compute_tangential_wind_numpy(grid, vn, rbf_vec_coeff_e),
            vt,
        )

        condition2 = (1 <= k_nlev) & (k_nlev < nlev)
        vn_ie[:, :-1], z_kin_hor_e = np.where(
            condition2,
            interpolate_vn_to_ie_and_compute_ekin_on_edges_numpy(grid, wgtfac_e, vn, vt),
            (vn_ie[:, :nlev], z_kin_hor_e),
        )

        if not lvn_only:
            z_vt_ie = np.where(
                condition2,
                interpolate_vt_to_interface_edges_numpy(grid, wgtfac_e, vt),
                z_vt_ie,
            )

        condition3 = k_nlev == 0
        vn_ie[:, :nlev], z_vt_ie, z_kin_hor_e = np.where(
            condition3,
            compute_horizontal_kinetic_energy_numpy(vn, vt),
            (vn_ie[:, :nlev], z_vt_ie, z_kin_hor_e),
        )

        condition4 = k == nlev
        vn_ie = np.where(
            condition4,
            extrapolate_at_top_numpy(grid, wgtfacq_e, vn),
            vn_ie,
        )

        condition5 = (nflatlev <= k_nlev) & (k_nlev < nlev)
        z_w_concorr_me = np.where(
            condition5,
            compute_contravariant_correction_numpy(vn, ddxn_z_full, ddxt_z_full, vt),
            z_w_concorr_me,
        )

        return vt, vn_ie, z_kin_hor_e, z_w_concorr_me

    @classmethod
    def reference(
        cls,
        grid,
        vn,
        rbf_vec_coeff_e,
        wgtfac_e,
        ddxn_z_full,
        ddxt_z_full,
        z_w_concorr_me,
        wgtfacq_e,
        nflatlev,
        c_intp,
        w,
        inv_dual_edge_length,
        inv_primal_edge_length,
        tangent_orientation,
        z_vt_ie,
        vt,
        vn_ie,
        z_kin_hor_e,
        z_v_grad_w,
        k,
        istep,
        nlev,
        lvn_only,
        edge,
        lateral_boundary_7,
        halo_1,
        **kwargs,
    ):
        k_nlev = k[:-1]

        if istep == 1:
            (
                vt,
                vn_ie,
                z_kin_hor_e,
                z_w_concorr_me,
            ) = cls._fused_velocity_advection_stencil_1_to_6_numpy(
                grid,
                vn,
                rbf_vec_coeff_e,
                wgtfac_e,
                ddxn_z_full,
                ddxt_z_full,
                z_w_concorr_me,
                wgtfacq_e,
                nflatlev,
                z_vt_ie,
                vt,
                vn_ie,
                z_kin_hor_e,
                k,
                nlev,
                lvn_only,
            )

        edge = edge[:, np.newaxis]

        condition_mask = (lateral_boundary_7 <= edge) & (edge < halo_1) & (k_nlev < nlev)

        z_v_w = mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy(grid, w, c_intp)

        if not lvn_only:
            z_v_grad_w = np.where(
                condition_mask,
                compute_horizontal_advection_term_for_vertical_velocity_numpy(
                    grid,
                    vn_ie[:, :-1],
                    inv_dual_edge_length,
                    w,
                    z_vt_ie,
                    inv_primal_edge_length,
                    tangent_orientation,
                    z_v_w,
                ),
                z_v_grad_w,
            )

        return dict(
            vt=vt,
            vn_ie=vn_ie,
            z_kin_hor_e=z_kin_hor_e,
            z_w_concorr_me=z_w_concorr_me,
            z_v_grad_w=z_v_grad_w,
        )

    @pytest.fixture
    def input_data(self, grid):
        pytest.xfail(
            "Verification of z_v_grad_w currently not working, because numpy version incorrect."
        )
        if isinstance(grid, IconGrid) and grid.limited_area:
            pytest.xfail(
                "Execution domain needs to be restricted or boundary taken into account in stencil."
            )

        c_intp = random_field(grid, dims.VertexDim, dims.V2CDim)
        vn = random_field(grid, dims.EdgeDim, dims.KDim)
        rbf_vec_coeff_e = random_field(grid, dims.EdgeDim, dims.E2C2EDim)
        vt = zero_field(grid, dims.EdgeDim, dims.KDim)
        wgtfac_e = random_field(grid, dims.EdgeDim, dims.KDim)
        vn_ie = zero_field(grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1})
        z_kin_hor_e = zero_field(grid, dims.EdgeDim, dims.KDim)
        z_vt_ie = zero_field(grid, dims.EdgeDim, dims.KDim)
        ddxn_z_full = random_field(grid, dims.EdgeDim, dims.KDim)
        ddxt_z_full = random_field(grid, dims.EdgeDim, dims.KDim)
        z_w_concorr_me = zero_field(grid, dims.EdgeDim, dims.KDim)
        inv_dual_edge_length = random_field(grid, dims.EdgeDim)
        w = random_field(grid, dims.CellDim, dims.KDim)
        inv_primal_edge_length = random_field(grid, dims.EdgeDim)
        tangent_orientation = random_field(grid, dims.EdgeDim)
        z_v_grad_w = zero_field(grid, dims.EdgeDim, dims.KDim)
        wgtfacq_e = random_field(grid, dims.EdgeDim, dims.KDim)

        k = field_alloc.allocate_indices(dims.KDim, grid=grid, is_halfdim=True)

        edge = zero_field(grid, dims.EdgeDim, dtype=gtx.int32)
        for e in range(grid.num_edges):
            edge[e] = e

        nlev = grid.num_levels
        nflatlev = 13

        istep = 1
        lvn_only = False

        lateral_boundary_7 = 0
        halo_1 = grid.num_edges

        horizontal_start = 0
        horizontal_end = grid.num_edges
        vertical_start = 0
        vertical_end = nlev + 1

        return dict(
            vn=vn,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            wgtfac_e=wgtfac_e,
            ddxn_z_full=ddxn_z_full,
            ddxt_z_full=ddxt_z_full,
            z_w_concorr_me=z_w_concorr_me,
            wgtfacq_e=wgtfacq_e,
            nflatlev=nflatlev,
            c_intp=c_intp,
            w=w,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
            z_vt_ie=z_vt_ie,
            vt=vt,
            vn_ie=vn_ie,
            z_kin_hor_e=z_kin_hor_e,
            z_v_grad_w=z_v_grad_w,
            k=k,
            istep=istep,
            nlev=nlev,
            lvn_only=lvn_only,
            edge=edge,
            lateral_boundary_7=lateral_boundary_7,
            halo_1=halo_1,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
        )
