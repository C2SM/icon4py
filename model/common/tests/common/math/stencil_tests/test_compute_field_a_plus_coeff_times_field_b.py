from typing import Any

import gt4py.next as gtx
import numpy as np

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.math.stencils import generic_math_operations
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


class TestComputeFieldAPlusCoeffTimesFieldBOnCellKHalf(stencil_tests.StencilTest):
    PROGRAM = generic_math_operations.compute_field_a_plus_coeff_times_field_b_on_cell_khalf
    OUTPUTS = ("output_field",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        field_a: np.ndarray,
        coeff: float,
        field_b: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        return dict(output_field=field_a + coeff * field_b)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        return dict(
            field_a=data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=wpfloat),
            coeff=wpfloat(300.0),
            field_b=data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=wpfloat),
            output_field=data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=wpfloat),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels + 1),
        )
