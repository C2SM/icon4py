# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the fused tracer advection program (before horizontal limiter).

Two test classes cover the two modes controlled by even_timestep:
  - Even timestep: vertical flux and integration run first; r_m is computed from
    the vertically-integrated tracer and rhodz_ast2.
  - Odd timestep: no vertical integration before the limiter; r_m is computed from
    the current tracer and rhodz_now.
"""

from typing import Any

import gt4py.next as gtx
import numpy as np

from icon4py.model.atmosphere.tracer_advection.stencils.compute_fused_tracer_advection import (
    compute_tracer_advection_before_horizontal_limiter,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.testing import stencil_tests

from .test_compute_barycentric_backtrajectory_alt import (
    compute_barycentric_backtrajectory_alt_numpy,
)
from .test_compute_fused_tracer_advection_terms import (
    apply_density_increment_numpy,
    compute_tangential_wind_numpy,
    reconstruct_linear_coefficients_svd_numpy,
)
from .test_compute_horizontal_tracer_flux_from_linear_coefficients_alt import (
    compute_horizontal_tracer_flux_from_linear_coefficients_alt_numpy,
)
from .test_compute_positive_definite_horizontal_multiplicative_flux_factor import (
    compute_positive_definite_horizontal_multiplicative_flux_factor_numpy,
)
from .test_integrate_tracer_vertically import integrate_tracer_vertically_numpy


class TestComputeTracerAdvectionBeforeHorizontalLimiterEven(stencil_tests.StencilTest):
    """Even-timestep mode: vertical flux and integration precede the horizontal limiter."""

    PROGRAM = compute_tracer_advection_before_horizontal_limiter
    OUTPUTS = (
        "rhodz_ast2",
        "p_mflx_tracer_v",
        "p_tracer_after_vertical",
        "p_mflx_tracer_h_unlimited",
        "r_m",
    )

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        rhodz_now: np.ndarray,
        p_mflx_contra_v: np.ndarray,
        p_tracer_now: np.ndarray,
        p_mass_flx_e: np.ndarray,
        p_vn: np.ndarray,
        deepatmo_divzl: np.ndarray,
        deepatmo_divzu: np.ndarray,
        geofac_div: np.ndarray,
        rbf_vec_coeff_e: np.ndarray,
        pos_on_tplane_e_1: np.ndarray,
        pos_on_tplane_e_2: np.ndarray,
        primal_normal_cell_1: np.ndarray,
        dual_normal_cell_1: np.ndarray,
        primal_normal_cell_2: np.ndarray,
        dual_normal_cell_2: np.ndarray,
        lsq_pseudoinv_1: np.ndarray,
        lsq_pseudoinv_2: np.ndarray,
        ivadv_tracer: int,
        ihadv_tracer: int,
        itype_hlimit: int,
        k: np.ndarray,
        iadv_slev_jt: int,
        p_dtime: float,
        dbl_eps: float,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)

        # Step 1: apply density increment (even timestep uses rhodz_now)
        rhodz_ast2 = apply_density_increment_numpy(
            rhodz_now, p_mflx_contra_v, deepatmo_divzl, deepatmo_divzu, p_dtime, True
        )

        p_mflx_tracer_v = np.zeros_like(p_mflx_contra_v)

        p_tracer_after_vertical = integrate_tracer_vertically_numpy(
            p_tracer_now,
            rhodz_now,
            p_mflx_tracer_v,
            deepatmo_divzl,
            deepatmo_divzu,
            rhodz_ast2,
            k,
            p_dtime,
            ivadv_tracer,
            iadv_slev_jt,
        )

        # Step 4: 2nd-order Miura horizontal flux from vertically-integrated tracer
        if ihadv_tracer == 2:
            z_real_vt = compute_tangential_wind_numpy(p_vn, rbf_vec_coeff_e, connectivities)
            p_distv_bary_1, p_distv_bary_2 = compute_barycentric_backtrajectory_alt_numpy(
                p_vn,
                z_real_vt,
                pos_on_tplane_e_1,
                pos_on_tplane_e_2,
                primal_normal_cell_1,
                dual_normal_cell_1,
                primal_normal_cell_2,
                dual_normal_cell_2,
                0.5 * p_dtime,
            )
            p_coeff_1, p_coeff_2, p_coeff_3 = reconstruct_linear_coefficients_svd_numpy(
                p_tracer_after_vertical, lsq_pseudoinv_1, lsq_pseudoinv_2, connectivities
            )
            p_mflx_tracer_h_unlimited = (
                compute_horizontal_tracer_flux_from_linear_coefficients_alt_numpy(
                    connectivities,
                    p_coeff_1,
                    p_coeff_2,
                    p_coeff_3,
                    p_distv_bary_1,
                    p_distv_bary_2,
                    p_mass_flx_e,
                    p_vn,
                )
            )
        else:
            p_mflx_tracer_h_unlimited = np.zeros(
                (grid.num_edges, grid.num_levels), dtype=rhodz_now.dtype
            )

        # Step 5: positive definite limiter factor uses vertically-integrated tracer and rhodz_ast2
        if itype_hlimit == 4:
            r_m = compute_positive_definite_horizontal_multiplicative_flux_factor_numpy(
                connectivities,
                geofac_div,
                p_tracer_after_vertical,
                rhodz_ast2,
                p_mflx_tracer_h_unlimited,
                p_dtime,
                dbl_eps,
            )
        else:
            r_m = np.ones((grid.num_cells, grid.num_levels), dtype=rhodz_now.dtype)

        return dict(
            rhodz_ast2=rhodz_ast2,
            p_mflx_tracer_v=p_mflx_tracer_v,
            p_tracer_after_vertical=p_tracer_after_vertical,
            p_mflx_tracer_h_unlimited=p_mflx_tracer_h_unlimited,
            r_m=r_m,
        )

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        rhodz_now = data_alloc.random_field(dims.CellDim, dims.KDim)
        rhodz_new = data_alloc.random_field(dims.CellDim, dims.KDim)
        p_mflx_contra_v = data_alloc.random_field(dims.CellDim, dims.KHalfDim)
        p_tracer_now = data_alloc.random_field(dims.CellDim, dims.KDim)
        p_mass_flx_e = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        p_vn = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        p_cellhgt_mc_now = data_alloc.random_field(dims.CellDim, dims.KDim)
        deepatmo_divzl = data_alloc.random_field(dims.KDim)
        deepatmo_divzu = data_alloc.random_field(dims.KDim)
        rbf_vec_coeff_e = data_alloc.random_field(dims.EdgeDim, dims.E2C2EDim)
        pos_on_tplane_e_1 = data_alloc.random_field(dims.EdgeDim, dims.E2CDim)
        pos_on_tplane_e_2 = data_alloc.random_field(dims.EdgeDim, dims.E2CDim)
        primal_normal_cell_1 = data_alloc.random_field(dims.EdgeDim, dims.E2CDim)
        dual_normal_cell_1 = data_alloc.random_field(dims.EdgeDim, dims.E2CDim)
        primal_normal_cell_2 = data_alloc.random_field(dims.EdgeDim, dims.E2CDim)
        dual_normal_cell_2 = data_alloc.random_field(dims.EdgeDim, dims.E2CDim)
        lsq_pseudoinv_1 = data_alloc.random_field(dims.CellDim, dims.C2E2CDim)
        lsq_pseudoinv_2 = data_alloc.random_field(dims.CellDim, dims.C2E2CDim)
        geofac_div = data_alloc.random_field(dims.CellDim, dims.C2EDim)
        k = data_alloc.index_field(dims.KDim)
        k_half = data_alloc.index_field(dims.KHalfDim)

        rhodz_ast2 = data_alloc.zero_field(dims.CellDim, dims.KDim)
        p_mflx_tracer_v = data_alloc.zero_field(dims.CellDim, dims.KHalfDim)
        p_tracer_after_vertical = data_alloc.zero_field(dims.CellDim, dims.KDim)
        p_mflx_tracer_h_unlimited = data_alloc.zero_field(dims.EdgeDim, dims.KDim)
        r_m = data_alloc.zero_field(dims.CellDim, dims.KDim)

        p_dtime = np.float64(5.0)
        dbl_eps = np.float64(1e-9)
        ivadv_tracer = gtx.int32(0)
        ihadv_tracer = gtx.int32(2)
        itype_hlimit = gtx.int32(4)
        itype_vlimit = gtx.int32(1)
        iadv_slev_jt = gtx.int32(0)
        slev = gtx.int32(0)
        slevp1_ti = gtx.int32(1)
        elev = gtx.int32(grid.num_levels - 1)

        return dict(
            rhodz_ast2=rhodz_ast2,
            p_mflx_tracer_v=p_mflx_tracer_v,
            p_tracer_after_vertical=p_tracer_after_vertical,
            p_mflx_tracer_h_unlimited=p_mflx_tracer_h_unlimited,
            r_m=r_m,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            p_mflx_contra_v=p_mflx_contra_v,
            p_tracer_now=p_tracer_now,
            p_mass_flx_e=p_mass_flx_e,
            p_vn=p_vn,
            p_cellhgt_mc_now=p_cellhgt_mc_now,
            deepatmo_divzl=deepatmo_divzl,
            deepatmo_divzu=deepatmo_divzu,
            k=k,
            k_half=k_half,
            slev=slev,
            slevp1_ti=slevp1_ti,
            elev=elev,
            ivadv_tracer=ivadv_tracer,
            ihadv_tracer=ihadv_tracer,
            itype_hlimit=itype_hlimit,
            itype_vlimit=itype_vlimit,
            iadv_slev_jt=iadv_slev_jt,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            pos_on_tplane_e_1=pos_on_tplane_e_1,
            pos_on_tplane_e_2=pos_on_tplane_e_2,
            primal_normal_cell_1=primal_normal_cell_1,
            dual_normal_cell_1=dual_normal_cell_1,
            primal_normal_cell_2=primal_normal_cell_2,
            dual_normal_cell_2=dual_normal_cell_2,
            lsq_pseudoinv_1=lsq_pseudoinv_1,
            lsq_pseudoinv_2=lsq_pseudoinv_2,
            geofac_div=geofac_div,
            dbl_eps=dbl_eps,
            p_dtime=p_dtime,
            even_timestep=True,
            start_cell_lateral_boundary_level_2=gtx.int32(0),
            end_cell_local=gtx.int32(grid.num_cells),
            end_cell_end=gtx.int32(grid.num_cells),
            start_edge_lateral_boundary_level_5=gtx.int32(0),
            end_edge_halo=gtx.int32(grid.num_edges),
            vertical_end=gtx.int32(grid.num_levels),
        )


class TestComputeTracerAdvectionBeforeHorizontalLimiterOdd(stencil_tests.StencilTest):
    """Odd-timestep mode: no vertical integration before the horizontal limiter."""

    PROGRAM = compute_tracer_advection_before_horizontal_limiter
    OUTPUTS = (
        "rhodz_ast2",
        "p_mflx_tracer_v",
        "p_tracer_after_vertical",
        "p_mflx_tracer_h_unlimited",
        "r_m",
    )

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        rhodz_now: np.ndarray,
        rhodz_new: np.ndarray,
        p_mflx_contra_v: np.ndarray,
        p_tracer_now: np.ndarray,
        p_mass_flx_e: np.ndarray,
        p_vn: np.ndarray,
        deepatmo_divzl: np.ndarray,
        deepatmo_divzu: np.ndarray,
        geofac_div: np.ndarray,
        rbf_vec_coeff_e: np.ndarray,
        pos_on_tplane_e_1: np.ndarray,
        pos_on_tplane_e_2: np.ndarray,
        primal_normal_cell_1: np.ndarray,
        dual_normal_cell_1: np.ndarray,
        primal_normal_cell_2: np.ndarray,
        dual_normal_cell_2: np.ndarray,
        lsq_pseudoinv_1: np.ndarray,
        lsq_pseudoinv_2: np.ndarray,
        ihadv_tracer: int,
        itype_hlimit: int,
        p_dtime: float,
        dbl_eps: float,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)

        rhodz_ast2 = apply_density_increment_numpy(
            rhodz_new, p_mflx_contra_v, deepatmo_divzl, deepatmo_divzu, p_dtime, False
        )

        p_mflx_tracer_v = np.zeros_like(p_mflx_contra_v)
        p_tracer_after_vertical = p_tracer_now + np.zeros_like(p_tracer_now)

        if ihadv_tracer == 2:
            z_real_vt = compute_tangential_wind_numpy(p_vn, rbf_vec_coeff_e, connectivities)
            p_distv_bary_1, p_distv_bary_2 = compute_barycentric_backtrajectory_alt_numpy(
                p_vn,
                z_real_vt,
                pos_on_tplane_e_1,
                pos_on_tplane_e_2,
                primal_normal_cell_1,
                dual_normal_cell_1,
                primal_normal_cell_2,
                dual_normal_cell_2,
                0.5 * p_dtime,
            )
            p_coeff_1, p_coeff_2, p_coeff_3 = reconstruct_linear_coefficients_svd_numpy(
                p_tracer_now, lsq_pseudoinv_1, lsq_pseudoinv_2, connectivities
            )
            p_mflx_tracer_h_unlimited = (
                compute_horizontal_tracer_flux_from_linear_coefficients_alt_numpy(
                    connectivities,
                    p_coeff_1,
                    p_coeff_2,
                    p_coeff_3,
                    p_distv_bary_1,
                    p_distv_bary_2,
                    p_mass_flx_e,
                    p_vn,
                )
            )
        else:
            p_mflx_tracer_h_unlimited = np.zeros(
                (grid.num_edges, grid.num_levels), dtype=rhodz_now.dtype
            )

        if itype_hlimit == 4:
            r_m = compute_positive_definite_horizontal_multiplicative_flux_factor_numpy(
                connectivities,
                geofac_div,
                p_tracer_now,
                rhodz_now,
                p_mflx_tracer_h_unlimited,
                p_dtime,
                dbl_eps,
            )
        else:
            r_m = np.ones((grid.num_cells, grid.num_levels), dtype=rhodz_now.dtype)

        return dict(
            rhodz_ast2=rhodz_ast2,
            p_mflx_tracer_v=p_mflx_tracer_v,
            p_tracer_after_vertical=p_tracer_after_vertical,
            p_mflx_tracer_h_unlimited=p_mflx_tracer_h_unlimited,
            r_m=r_m,
        )

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        rhodz_now = data_alloc.random_field(dims.CellDim, dims.KDim)
        rhodz_new = data_alloc.random_field(dims.CellDim, dims.KDim)
        p_mflx_contra_v = data_alloc.random_field(dims.CellDim, dims.KHalfDim)
        p_tracer_now = data_alloc.random_field(dims.CellDim, dims.KDim)
        p_mass_flx_e = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        p_vn = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        p_cellhgt_mc_now = data_alloc.random_field(dims.CellDim, dims.KDim)
        deepatmo_divzl = data_alloc.random_field(dims.KDim)
        deepatmo_divzu = data_alloc.random_field(dims.KDim)
        rbf_vec_coeff_e = data_alloc.random_field(dims.EdgeDim, dims.E2C2EDim)
        pos_on_tplane_e_1 = data_alloc.random_field(dims.EdgeDim, dims.E2CDim)
        pos_on_tplane_e_2 = data_alloc.random_field(dims.EdgeDim, dims.E2CDim)
        primal_normal_cell_1 = data_alloc.random_field(dims.EdgeDim, dims.E2CDim)
        dual_normal_cell_1 = data_alloc.random_field(dims.EdgeDim, dims.E2CDim)
        primal_normal_cell_2 = data_alloc.random_field(dims.EdgeDim, dims.E2CDim)
        dual_normal_cell_2 = data_alloc.random_field(dims.EdgeDim, dims.E2CDim)
        lsq_pseudoinv_1 = data_alloc.random_field(dims.CellDim, dims.C2E2CDim)
        lsq_pseudoinv_2 = data_alloc.random_field(dims.CellDim, dims.C2E2CDim)
        geofac_div = data_alloc.random_field(dims.CellDim, dims.C2EDim)
        k = data_alloc.index_field(dims.KDim)
        k_half = data_alloc.index_field(dims.KHalfDim)

        rhodz_ast2 = data_alloc.zero_field(dims.CellDim, dims.KDim)
        p_mflx_tracer_v = data_alloc.zero_field(dims.CellDim, dims.KHalfDim)
        p_tracer_after_vertical = data_alloc.zero_field(dims.CellDim, dims.KDim)
        p_mflx_tracer_h_unlimited = data_alloc.zero_field(dims.EdgeDim, dims.KDim)
        r_m = data_alloc.zero_field(dims.CellDim, dims.KDim)

        p_dtime = np.float64(5.0)
        dbl_eps = np.float64(1e-9)
        ivadv_tracer = gtx.int32(0)
        ihadv_tracer = gtx.int32(2)
        itype_hlimit = gtx.int32(4)
        itype_vlimit = gtx.int32(1)
        iadv_slev_jt = gtx.int32(0)
        slev = gtx.int32(0)
        slevp1_ti = gtx.int32(1)
        elev = gtx.int32(grid.num_levels - 1)

        return dict(
            rhodz_ast2=rhodz_ast2,
            p_mflx_tracer_v=p_mflx_tracer_v,
            p_tracer_after_vertical=p_tracer_after_vertical,
            p_mflx_tracer_h_unlimited=p_mflx_tracer_h_unlimited,
            r_m=r_m,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            p_mflx_contra_v=p_mflx_contra_v,
            p_tracer_now=p_tracer_now,
            p_mass_flx_e=p_mass_flx_e,
            p_vn=p_vn,
            p_cellhgt_mc_now=p_cellhgt_mc_now,
            deepatmo_divzl=deepatmo_divzl,
            deepatmo_divzu=deepatmo_divzu,
            k=k,
            k_half=k_half,
            slev=slev,
            slevp1_ti=slevp1_ti,
            elev=elev,
            ivadv_tracer=ivadv_tracer,
            ihadv_tracer=ihadv_tracer,
            itype_hlimit=itype_hlimit,
            itype_vlimit=itype_vlimit,
            iadv_slev_jt=iadv_slev_jt,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            pos_on_tplane_e_1=pos_on_tplane_e_1,
            pos_on_tplane_e_2=pos_on_tplane_e_2,
            primal_normal_cell_1=primal_normal_cell_1,
            dual_normal_cell_1=dual_normal_cell_1,
            primal_normal_cell_2=primal_normal_cell_2,
            dual_normal_cell_2=dual_normal_cell_2,
            lsq_pseudoinv_1=lsq_pseudoinv_1,
            lsq_pseudoinv_2=lsq_pseudoinv_2,
            geofac_div=geofac_div,
            dbl_eps=dbl_eps,
            p_dtime=p_dtime,
            even_timestep=False,
            start_cell_lateral_boundary_level_2=gtx.int32(0),
            end_cell_local=gtx.int32(grid.num_cells),
            end_cell_end=gtx.int32(grid.num_cells),
            start_edge_lateral_boundary_level_5=gtx.int32(0),
            end_edge_halo=gtx.int32(grid.num_edges),
            vertical_end=gtx.int32(grid.num_levels),
        )
