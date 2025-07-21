# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_theta_and_exner import (
    apply_diffusion_to_theta_and_exner,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils.data_allocation import (
    flatten_first_two_dims,
    random_field,
    random_mask,
    unflatten_first_two_dims,
    zero_field,
)
from icon4py.model.testing.helpers import StencilTest

from .test_calculate_nabla2_for_z import calculate_nabla2_for_z_numpy
from .test_calculate_nabla2_of_theta import calculate_nabla2_of_theta_numpy
from .test_truly_horizontal_diffusion_nabla_of_theta_over_steep_points import (
    truly_horizontal_diffusion_nabla_of_theta_over_steep_points_numpy,
)
from .test_update_theta_and_exner import update_theta_and_exner_numpy


class TestApplyDiffusionToThetaAndExner(StencilTest):
    PROGRAM = apply_diffusion_to_theta_and_exner
    OUTPUTS = ("theta_v", "exner")
    MARKERS = (
        pytest.mark.embedded_remap_error,
        pytest.mark.uses_as_offset,
        pytest.mark.skip_value_error,
    )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        kh_smag_e: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        theta_v_in: np.ndarray,
        geofac_div: np.ndarray,
        mask: np.ndarray,
        zd_vertoffset: np.ndarray,
        zd_diffcoef: np.ndarray,
        geofac_n2s_c: np.ndarray,
        geofac_n2s_nbh: np.ndarray,
        vcoef: np.ndarray,
        area: np.ndarray,
        exner: np.ndarray,
        rd_o_cvd: float,
        **kwargs: Any,
    ) -> dict:
        kwargs_2 = {k: v for k, v in kwargs.items() if k != "theta_v"}  # remove unused kwargs

        z_nabla2_e = np.zeros_like(kh_smag_e)
        z_nabla2_e = calculate_nabla2_for_z_numpy(
            connectivities, kh_smag_e, inv_dual_edge_length, theta_v_in, z_nabla2_e, **kwargs_2
        )
        z_temp = calculate_nabla2_of_theta_numpy(connectivities, z_nabla2_e, geofac_div)

        geofac_n2s_nbh = unflatten_first_two_dims(geofac_n2s_nbh)
        zd_vertoffset = unflatten_first_two_dims(zd_vertoffset)

        z_temp = truly_horizontal_diffusion_nabla_of_theta_over_steep_points_numpy(
            connectivities,
            mask,
            zd_vertoffset,
            zd_diffcoef,
            geofac_n2s_c,
            geofac_n2s_nbh,
            vcoef,
            theta_v_in,
            z_temp,
        )
        theta_v, exner = update_theta_and_exner_numpy(z_temp, area, theta_v_in, exner, rd_o_cvd)

        return dict(theta_v=theta_v, exner=exner)

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid):
        pytest.xfail(
            "stencil segfaults with GTFN and it is not used in diffusion: it is missing an if condition"
        )
        kh_smag_e = random_field(grid, dims.EdgeDim, dims.KDim)
        inv_dual_edge_length = random_field(grid, dims.EdgeDim)
        theta_v_in = random_field(grid, dims.CellDim, dims.KDim)
        geofac_div = random_field(grid, dims.CEDim)
        mask = random_mask(grid, dims.CellDim, dims.KDim)
        zd_vertoffset = zero_field(grid, dims.CellDim, dims.C2E2CDim, dims.KDim, dtype=gtx.int32)
        rng = np.random.default_rng()
        for k in range(grid.num_levels):
            # construct offsets that reach all k-levels except the last (because we are using the entries of this field with `+1`)
            zd_vertoffset[:, :, k] = rng.integers(
                low=0 - k,
                high=grid.num_levels - k - 1,
                size=(zd_vertoffset.shape[0], zd_vertoffset.shape[1]),
            )
        zd_diffcoef = random_field(grid, dims.CellDim, dims.KDim)
        geofac_n2s_c = random_field(grid, dims.CellDim)
        geofac_n2s_nbh = random_field(grid, dims.CellDim, dims.C2E2CDim)
        vcoef = random_field(grid, dims.CellDim, dims.C2E2CDim, dims.KDim)
        area = random_field(grid, dims.CellDim)
        theta_v = random_field(grid, dims.CellDim, dims.KDim)
        exner = random_field(grid, dims.CellDim, dims.KDim)
        rd_o_cvd = 5.0

        vcoef_new = flatten_first_two_dims(dims.CECDim, dims.KDim, field=vcoef)
        zd_vertoffset_new = flatten_first_two_dims(dims.CECDim, dims.KDim, field=zd_vertoffset)
        geofac_n2s_nbh_new = flatten_first_two_dims(dims.CECDim, field=geofac_n2s_nbh)
        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))

        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.LOCAL))

        return dict(
            kh_smag_e=kh_smag_e,
            inv_dual_edge_length=inv_dual_edge_length,
            theta_v_in=theta_v_in,
            geofac_div=geofac_div,
            mask=mask,
            zd_vertoffset=zd_vertoffset_new,
            zd_diffcoef=zd_diffcoef,
            geofac_n2s_c=geofac_n2s_c,
            geofac_n2s_nbh=geofac_n2s_nbh_new,
            vcoef=vcoef_new,
            area=area,
            theta_v=theta_v,
            exner=exner,
            rd_o_cvd=rd_o_cvd,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
