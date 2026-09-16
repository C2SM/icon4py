# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.diffusion.stencils.diffusion_for_w_and_horizontal_gradients_for_turbulence import (
    diffusion_for_w_and_horizontal_gradients_for_turbulence,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.testing import stencil_tests

from .test_apply_nabla2_to_w import apply_nabla2_to_w_numpy
from .test_horizontal_gradients_for_turbulence import horizontal_gradients_for_turbulence_numpy
from .test_nabla2_for_w import nabla2_for_w_numpy
from .test_nabla2_for_w_in_upper_damping_layer import nabla2_for_w_in_upper_damping_layer_numpy


@pytest.mark.embedded_remap_error
@pytest.mark.continuous_benchmarking
class TestDiffusionForWAndHorizontalGradientsForTurbulence(stencil_tests.StencilTest):
    PROGRAM = diffusion_for_w_and_horizontal_gradients_for_turbulence
    OUTPUTS = ("w", "dwdx", "dwdy")
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "halo_idx",
            "interior_idx",
            "vertical_start",
            "vertical_end",
            "nrdmax",
            "type_shear",
        ),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
            "nrdmax",
            "type_shear",
        ),
    }

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        area,
        geofac_n2s,
        geofac_grg_x,
        geofac_grg_y,
        w_old,
        type_shear,
        dwdx,
        dwdy,
        diff_multfac_w,
        diff_multfac_n2w,
        nrdmax,
        interior_idx,
        halo_idx,
        horizontal_start,
        horizontal_end,
        vertical_start,
        vertical_end,
        **kwargs,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        k = np.arange(w_old.shape[1])
        cell = np.arange(w_old.shape[0])
        reshaped_k = k[np.newaxis, :]
        reshaped_cell = cell[:, np.newaxis]
        # Initialize outputs with the pass-through values that the per-output
        # domains no longer write back.
        out_w = w_old.copy()
        out_dwdx = dwdx.copy()
        out_dwdy = dwdy.copy()

        if type_shear == 2:
            grad_dwdx, grad_dwdy = horizontal_gradients_for_turbulence_numpy(
                connectivities, w_old, geofac_grg_x, geofac_grg_y
            )
            grad_slice = (
                slice(horizontal_start, horizontal_end),
                slice(vertical_start + 1, vertical_end),
            )
            out_dwdx[grad_slice] = grad_dwdx[grad_slice]
            out_dwdy[grad_slice] = grad_dwdy[grad_slice]

        z_nabla2_c = nabla2_for_w_numpy(connectivities, w_old, geofac_n2s)

        w = apply_nabla2_to_w_numpy(
            connectivities=connectivities,
            area=area,
            z_nabla2_c=z_nabla2_c,
            geofac_n2s=geofac_n2s,
            w=w_old,
            diff_multfac_w=diff_multfac_w,
        )

        w = np.where(
            (reshaped_k > 0)
            & (reshaped_k < nrdmax)
            & (interior_idx <= reshaped_cell)
            & (reshaped_cell < halo_idx),
            nabla2_for_w_in_upper_damping_layer_numpy(w, diff_multfac_n2w, area, z_nabla2_c),
            w,
        )
        w_slice = (slice(interior_idx, halo_idx), slice(vertical_start, vertical_end))
        out_w[w_slice] = w[w_slice]
        return dict(w=out_w, dwdx=out_dwdx, dwdy=out_dwdy)

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        nrdmax = 13
        cell_domain = h_grid.domain(dims.CellDim)
        interior_idx = grid.start_index(cell_domain(h_grid.Zone.INTERIOR))  # 0 for simple grid
        halo_idx = grid.end_index(
            cell_domain(h_grid.Zone.LOCAL)
        )  # same as horizontal_end for simple grid
        type_shear = 2

        def _get_start_index_for_w_diffusion() -> int32:
            return (
                grid.start_index(cell_domain(h_grid.Zone.NUDGING))
                if grid.limited_area
                else grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
            )

        horizontal_start = _get_start_index_for_w_diffusion()
        horizontal_end = grid.end_index(cell_domain(h_grid.Zone.HALO))

        geofac_grg_x = data_alloc.random_field(dims.CellDim, dims.C2E2CODim)
        geofac_grg_y = data_alloc.random_field(dims.CellDim, dims.C2E2CODim)
        diff_multfac_n2w = data_alloc.random_field(dims.KHalfDim)
        area = data_alloc.random_field(dims.CellDim)
        geofac_n2s = data_alloc.random_field(dims.CellDim, dims.C2E2CODim)
        w_old = data_alloc.random_field(dims.CellDim, dims.KHalfDim)
        diff_multfac_w = 5.0

        w = data_alloc.as_field(w_old.asnumpy().copy(), dims.CellDim, dims.KHalfDim)
        dwdx = data_alloc.random_field(dims.CellDim, dims.KHalfDim)
        dwdy = data_alloc.random_field(dims.CellDim, dims.KHalfDim)

        return dict(
            area=area,
            geofac_n2s=geofac_n2s,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            w_old=w_old,
            type_shear=type_shear,
            diff_multfac_w=diff_multfac_w,
            diff_multfac_n2w=diff_multfac_n2w,
            nrdmax=nrdmax,
            interior_idx=interior_idx,
            halo_idx=halo_idx,
            w=w,
            dwdx=dwdx,
            dwdy=dwdy,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=grid.num_levels + 1,
        )
