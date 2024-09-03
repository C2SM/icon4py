# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import math

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.test_utils.datatest_utils import (
    GLOBAL_EXPERIMENT,
    REGIONAL_EXPERIMENT,
)
from icon4py.model.common.test_utils.helpers import dallclose


NUM_LEVELS = 65


@pytest.mark.parametrize(
    "max_h,damping_height,delta",
    [(60000, 34000, 612), (12000, 10000, 100), (109050, 45000, 123)],
)
def test_damping_layer_calculation(max_h, damping_height, delta, flat_height):
    vct_a = np.arange(0, max_h, delta)
    vct_a_field = gtx.as_field((dims.KDim,), data=vct_a[::-1])
    vertical_config = v_grid.VerticalGridConfig(
        num_levels=NUM_LEVELS,
        flat_height=flat_height,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a_field,
        vct_b=None,
        _min_index_flat_horizontal_grad_pressure=10,
    )
    assert (
        vertical_params.end_index_of_damping_layer
        == vct_a.shape[0] - math.ceil(damping_height / delta) - 1
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [REGIONAL_EXPERIMENT, GLOBAL_EXPERIMENT])
def test_damping_layer_calculation_from_icon_input(
    grid_savepoint, experiment, damping_height, flat_height
):
    a = grid_savepoint.vct_a()
    b = grid_savepoint.vct_b()
    nrdmax = grid_savepoint.nrdmax()
    vertical_config = v_grid.VerticalGridConfig(
        num_levels=grid_savepoint.num(dims.KDim),
        flat_height=flat_height,
        rayleigh_damping_height=damping_height,
    )
    vertical_grid = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=a,
        vct_b=b,
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp,
    )
    assert nrdmax == vertical_grid.end_index_of_damping_layer
    a_array = a.asnumpy()
    assert a_array[nrdmax] > damping_height
    assert a_array[nrdmax + 1] < damping_height
    assert vertical_grid.index(v_grid.Domain(dims.KDim, v_grid.Zone.DAMPING)) == nrdmax


@pytest.mark.datatest
def test_grid_size(grid_savepoint):
    config = v_grid.VerticalGridConfig(num_levels=grid_savepoint.num(dims.KDim))
    vertical_grid = v_grid.VerticalGrid(
        config=config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp,
    )

    assert NUM_LEVELS == vertical_grid.size(dims.KDim)
    assert NUM_LEVELS + 1 == vertical_grid.size(dims.KHalfDim)


@pytest.mark.parametrize(
    "dim", (dims.CellDim, dims.VertexDim, dims.EdgeDim, dims.C2EDim, dims.C2VDim, dims.E2VDim)
)
@pytest.mark.datatest
def test_grid_size_raises_for_non_vertical_dim(grid_savepoint, dim):
    vertical_grid = configure_vertical_grid(grid_savepoint)
    with pytest.raises(AssertionError):
        vertical_grid.size(dim)


@pytest.mark.datatest
def test_grid_size_raises_for_unknown_vertical_dim(grid_savepoint):
    vertical_grid = configure_vertical_grid(grid_savepoint)
    j_dim = gtx.Dimension("J", kind=gtx.DimensionKind.VERTICAL)
    with pytest.raises(ValueError):
        vertical_grid.size(j_dim)


def configure_vertical_grid(grid_savepoint, top_moist_threshold=22500.0):
    config = v_grid.VerticalGridConfig(
        num_levels=grid_savepoint.num(dims.KDim), htop_moist_proc=top_moist_threshold
    )
    vertical_grid = v_grid.VerticalGrid(
        config=config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp,
    )

    return vertical_grid


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, expected_moist_level", [(REGIONAL_EXPERIMENT, 0), (GLOBAL_EXPERIMENT, 25)]
)
def test_moist_level_calculation(grid_savepoint, experiment, expected_moist_level):
    threshold = 22500.0
    vertical_grid = configure_vertical_grid(grid_savepoint, top_moist_threshold=threshold)
    assert expected_moist_level == vertical_grid.kstart_moist
    assert expected_moist_level == vertical_grid.index(v_grid.Domain(dims.KDim, v_grid.Zone.MOIST))


@pytest.mark.datatest
def test_interface_physical_height(grid_savepoint):
    vertical_grid = configure_vertical_grid(grid_savepoint)
    assert dallclose(
        grid_savepoint.vct_a().asnumpy(), vertical_grid.interface_physical_height.asnumpy()
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [REGIONAL_EXPERIMENT, GLOBAL_EXPERIMENT])
def test_flat_level_calculation(grid_savepoint, experiment, flat_height):
    vertical_grid = configure_vertical_grid(grid_savepoint)

    assert grid_savepoint.nflatlev() == vertical_grid.nflatlev
    assert grid_savepoint.nflatlev() == vertical_grid.index(
        v_grid.Domain(dims.KDim, v_grid.Zone.FLAT)
    )


@pytest.mark.parametrize("experiment, levels", [(REGIONAL_EXPERIMENT, 65), (GLOBAL_EXPERIMENT, 60)])
def test_grid_index_top(grid_savepoint, experiment, levels):
    vertical_grid = configure_vertical_grid(grid_savepoint)
    assert 0 == vertical_grid.index(v_grid.Domain(dims.KDim, v_grid.Zone.TOP))
    assert levels == vertical_grid.index(v_grid.Domain(dims.KDim, v_grid.Zone.BOTTOM))
    assert levels + 1 == vertical_grid.index(v_grid.Domain(dims.KHalfDim, v_grid.Zone.BOTTOM))
    assert 0 == vertical_grid.index(v_grid.Domain(dims.KHalfDim, v_grid.Zone.TOP))


@pytest.mark.parametrize("experiment, levels", [(REGIONAL_EXPERIMENT, 65), (GLOBAL_EXPERIMENT, 60)])
def test_grid_index_bottom(grid_savepoint, experiment, levels):
    vertical_grid = configure_vertical_grid(grid_savepoint)
    assert levels == vertical_grid.index(v_grid.Domain(dims.KDim, v_grid.Zone.BOTTOM))
    assert levels + 1 == vertical_grid.index(v_grid.Domain(dims.KHalfDim, v_grid.Zone.BOTTOM))


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [REGIONAL_EXPERIMENT, GLOBAL_EXPERIMENT])
def test_vct_a_vct_b_calculation_from_icon_input(
    grid_savepoint,
    experiment,
    maximal_layer_thickness,
    top_height_limit_for_maximal_layer_thickness,
    lowest_layer_thickness,
    model_top_height,
    flat_height,
    stretch_factor,
    damping_height,
    htop_moist_proc,
):
    vertical_config = v_grid.VerticalGridConfig(
        num_levels=grid_savepoint.num(dims.KDim),
        maximal_layer_thickness=maximal_layer_thickness,
        top_height_limit_for_maximal_layer_thickness=top_height_limit_for_maximal_layer_thickness,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        flat_height=flat_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
        htop_moist_proc=htop_moist_proc,
    )
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config)

    assert dallclose(vct_a.asnumpy(), grid_savepoint.vct_a().asnumpy())
    assert dallclose(vct_b.asnumpy(), grid_savepoint.vct_b().asnumpy())
