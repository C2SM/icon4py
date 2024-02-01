import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common.dimension import KDim, CellDim
from icon4py.model.common.metrics.metric_fields import compute_z_mc
from icon4py.model.common.test_utils.helpers import StencilTest, zero_field, random_field


class TestComputeZMc(StencilTest):
    PROGRAM = compute_z_mc
    OUTPUTS = ("z_mc",)

    @staticmethod
    def reference(
        grid,
        z_ifc: np.array,
        **kwargs,
    ):
        shp = z_ifc.shape
        z_mc = 0.5 * (z_ifc + np.roll(z_ifc, shift=-1, axis=1))[:, : shp[1] - 1]
        return dict(z_mc=z_mc)

    @pytest.fixture
    def input_data(self, grid):
        z_mc = zero_field(grid, CellDim, KDim)
        z_if = random_field(grid, CellDim, KDim, extend={KDim: 1})
        horizontal_start = int32(0)
        horizontal_end = grid.num_cells
        vertical_start = int32(0)
        vertical_end = grid.num_levels

        return dict(
            z_mc=z_mc,
            z_ifc=z_if,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
