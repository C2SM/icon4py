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

from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence import (
    apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StandardStaticVariants, StencilTest

from .test_apply_nabla2_to_w import apply_nabla2_to_w_numpy
from .test_apply_nabla2_to_w_in_upper_damping_layer import (
    apply_nabla2_to_w_in_upper_damping_layer_numpy,
)
from .test_calculate_horizontal_gradients_for_turbulence import (
    calculate_horizontal_gradients_for_turbulence_numpy,
)
from .test_calculate_nabla2_for_w import calculate_nabla2_for_w_numpy


@pytest.mark.embedded_remap_error
class TestApplyDiffusionToWAndComputeHorizontalGradientsForTurbulence(StencilTest):
    PROGRAM = apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence
    OUTPUTS = ("w", "dwdx", "dwdy")
    STATIC_PARAMS = {
        StandardStaticVariants.NONE: (),
        StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "halo_idx",
            "interior_idx",
            "vertical_start",
            "vertical_end",
            "nrdmax",
            "type_shear",
        ),
        StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
            "nrdmax",
            "type_shear",
        ),
    }

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
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
        **kwargs,
    ) -> dict:
        k = np.arange(w_old.shape[1])
        cell = np.arange(w_old.shape[0])
        reshaped_k = k[np.newaxis, :]
        reshaped_cell = cell[:, np.newaxis]
        if type_shear == 2:
            dwdx, dwdy = np.where(
                reshaped_k > 0,
                calculate_horizontal_gradients_for_turbulence_numpy(
                    connectivities, w_old, geofac_grg_x, geofac_grg_y
                ),
                (dwdx, dwdy),
            )

        z_nabla2_c = calculate_nabla2_for_w_numpy(connectivities, w_old, geofac_n2s)

        w = np.where(
            (interior_idx <= reshaped_cell) & (reshaped_cell < halo_idx),
            apply_nabla2_to_w_numpy(
                connectivities, area, z_nabla2_c, geofac_n2s, w_old, diff_multfac_w
            ),
            w_old,
        )

        w = np.where(
            (reshaped_k > 0)
            & (reshaped_k < nrdmax)
            & (interior_idx <= reshaped_cell)
            & (reshaped_cell < halo_idx),
            apply_nabla2_to_w_in_upper_damping_layer_numpy(w, diff_multfac_n2w, area, z_nabla2_c),
            w,
        )
        return dict(w=w, dwdx=dwdx, dwdy=dwdy)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        nrdmax = 13
        interior_idx = 1
        halo_idx = 5
        type_shear = 2

        geofac_grg_x = random_field(grid, dims.CellDim, dims.C2E2CODim)
        geofac_grg_y = random_field(grid, dims.CellDim, dims.C2E2CODim)
        diff_multfac_n2w = random_field(grid, dims.KDim)
        area = random_field(grid, dims.CellDim)
        geofac_n2s = random_field(grid, dims.CellDim, dims.C2E2CODim)
        w_old = random_field(grid, dims.CellDim, dims.KDim)
        diff_multfac_w = 5.0

        w = zero_field(grid, dims.CellDim, dims.KDim)
        dwdx = random_field(grid, dims.CellDim, dims.KDim)
        dwdy = random_field(grid, dims.CellDim, dims.KDim)

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
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )


@pytest.mark.continuous_benchmarking
class TestApplyDiffusionToWAndComputeHorizontalGradientsForTurbulenceContinuousBenchmarking(
    TestApplyDiffusionToWAndComputeHorizontalGradientsForTurbulence
):
    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        assert (
            grid.id == "01f00602-c07e-cd84-b894-bd17fffd2720"
        ), "This test only works with the icon_benchmark grid."
        # Use the parent class's fixture indirectly by calling its method, not the fixture itself
        base_data = (
            TestApplyDiffusionToWAndComputeHorizontalGradientsForTurbulence.input_data.__wrapped__(
                self, grid
            )
        )
        cell_domain = h_grid.domain(dims.CellDim)
        base_data["interior_idx"] = grid.start_index(cell_domain(h_grid.Zone.INTERIOR))
        base_data["halo_idx"] = grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        def _get_start_index_for_w_diffusion() -> int32:
            return (
                grid.start_index(cell_domain(h_grid.Zone.NUDGING))
                if grid.limited_area
                else grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
            )

        base_data["horizontal_start"] = _get_start_index_for_w_diffusion()
        base_data["horizontal_end"] = grid.end_index(cell_domain(h_grid.Zone.HALO))
        return base_data
