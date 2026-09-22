# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the fused tracer advection program (after horizontal limiter)."""

from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.tracer_advection.stencils.compute_fused_tracer_advection import (
    compute_tracer_advection_after_horizontal_limiter,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.testing import stencil_tests

from .test_compute_fused_tracer_advection_terms import (
    apply_positive_definite_horizontal_multiplicative_flux_factor_numpy,
)
from .test_integrate_tracer_horizontally import integrate_tracer_horizontally_numpy


@pytest.mark.uses_concat_where
class TestComputeTracerAdvectionAfterHorizontalLimiter(stencil_tests.StencilTest):
    PROGRAM = compute_tracer_advection_after_horizontal_limiter
    OUTPUTS = (
        "p_mflx_tracer_h",
        "p_mflx_tracer_v",
        "p_tracer_new",
    )

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        r_m: np.ndarray,
        p_mflx_tracer_h_unlimited: np.ndarray,
        p_tracer_now: np.ndarray,
        p_tracer_after_vertical: np.ndarray,
        p_mflx_tracer_v: np.ndarray,
        rhodz_ast2: np.ndarray,
        rhodz_now: np.ndarray,
        rhodz_new: np.ndarray,
        deepatmo_divh: np.ndarray,
        geofac_div: np.ndarray,
        do_vertical_first: int,
        ihadv_tracer: int,
        itype_hlimit: int,
        p_dtime: float,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)

        if do_vertical_first == 1:
            tracer_now_for_h = p_tracer_after_vertical
            rhodz_for_h_now = rhodz_ast2
            rhodz_for_h_new = rhodz_new
        else:
            tracer_now_for_h = p_tracer_now
            rhodz_for_h_now = rhodz_now
            rhodz_for_h_new = rhodz_ast2

        if itype_hlimit == 4:
            p_mflx_tracer_h = apply_positive_definite_horizontal_multiplicative_flux_factor_numpy(
                r_m, p_mflx_tracer_h_unlimited, connectivities
            )
        else:
            p_mflx_tracer_h = p_mflx_tracer_h_unlimited.copy()

        if ihadv_tracer != 0:
            p_tracer_new = integrate_tracer_horizontally_numpy(
                connectivities,
                p_mflx_tracer_h,
                deepatmo_divh,
                tracer_now_for_h,
                rhodz_for_h_now,
                rhodz_for_h_new,
                geofac_div,
                p_dtime,
            )
        else:
            p_tracer_new = tracer_now_for_h.copy()

        return dict(
            p_mflx_tracer_h=p_mflx_tracer_h,
            p_mflx_tracer_v=p_mflx_tracer_v,
            p_tracer_new=p_tracer_new,
        )

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        r_m = data_alloc.random_field(dims.CellDim, dims.KDim)
        p_mflx_tracer_h_unlimited = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        p_tracer_now = data_alloc.random_field(dims.CellDim, dims.KDim)
        p_tracer_after_vertical = data_alloc.random_field(dims.CellDim, dims.KDim)
        p_mflx_tracer_v = data_alloc.random_field(dims.CellDim, dims.KHalfDim)
        rhodz_ast2 = data_alloc.random_field(dims.CellDim, dims.KDim)
        rhodz_now = data_alloc.random_field(dims.CellDim, dims.KDim)
        rhodz_new = data_alloc.random_field(dims.CellDim, dims.KDim)
        p_mflx_contra_v = data_alloc.random_field(dims.CellDim, dims.KHalfDim)
        p_cellhgt_mc_now = data_alloc.random_field(dims.CellDim, dims.KDim)
        deepatmo_divh = data_alloc.random_field(dims.KDim)
        deepatmo_divzl = data_alloc.random_field(dims.KDim)
        deepatmo_divzu = data_alloc.random_field(dims.KDim)
        k = data_alloc.index_field(dims.KDim)
        k_half = data_alloc.index_field(dims.KHalfDim)
        geofac_div = data_alloc.random_field(dims.CellDim, dims.C2EDim)

        # Output fields (initialized to zero)
        p_mflx_tracer_h = data_alloc.zero_field(dims.EdgeDim, dims.KDim)
        # p_mflx_tracer_v is both an input (read for even-timestep pass-through) and an output;
        # it is already declared above as the input field.
        p_tracer_new = data_alloc.zero_field(dims.CellDim, dims.KDim)

        # Scalar parameters
        p_dtime = np.float64(5.0)
        dbl_eps = np.float64(1e-9)
        # Use even timestep (do_vertical_first=1): vertical integration done before this program
        do_vertical_first = gtx.int32(1)
        ivadv_tracer = gtx.int32(0)
        ihadv_tracer = gtx.int32(2)
        itype_hlimit = gtx.int32(4)
        itype_vlimit = gtx.int32(1)
        iadv_slev_jt = gtx.int32(0)
        slev = gtx.int32(0)
        slevp1_ti = gtx.int32(1)
        elev = gtx.int32(grid.num_levels - 1)

        return dict(
            p_mflx_tracer_h=p_mflx_tracer_h,
            p_mflx_tracer_v=p_mflx_tracer_v,
            p_tracer_new=p_tracer_new,
            r_m=r_m,
            p_mflx_tracer_h_unlimited=p_mflx_tracer_h_unlimited,
            p_tracer_now=p_tracer_now,
            p_tracer_after_vertical=p_tracer_after_vertical,
            rhodz_ast2=rhodz_ast2,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            p_mflx_contra_v=p_mflx_contra_v,
            p_cellhgt_mc_now=p_cellhgt_mc_now,
            deepatmo_divh=deepatmo_divh,
            deepatmo_divzl=deepatmo_divzl,
            deepatmo_divzu=deepatmo_divzu,
            k=k,
            k_half=k_half,
            slev=slev,
            slevp1_ti=slevp1_ti,
            elev=elev,
            do_vertical_first=do_vertical_first,
            ivadv_tracer=ivadv_tracer,
            ihadv_tracer=ihadv_tracer,
            itype_hlimit=itype_hlimit,
            itype_vlimit=itype_vlimit,
            iadv_slev_jt=iadv_slev_jt,
            geofac_div=geofac_div,
            dbl_eps=dbl_eps,
            p_dtime=p_dtime,
            start_cell_nudging=gtx.int32(0),
            end_cell_local=gtx.int32(grid.num_cells),
            start_edge_lateral_boundary_level_5=gtx.int32(0),
            end_edge_halo=gtx.int32(grid.num_edges),
            vertical_end=gtx.int32(grid.num_levels),
        )
