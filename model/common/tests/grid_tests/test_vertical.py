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


@pytest.mark.parametrize(
    "max_h,damping_height,delta",
    [(60000, 34000, 612), (12000, 10000, 100), (109050, 45000, 123)],
)
def test_nrdmax_calculation(max_h, damping_height, delta, flat_height, grid_savepoint):
    vct_a = np.arange(0, max_h, delta)
    vct_a_field = gtx.as_field((dims.KDim,), data=vct_a[::-1])
    vertical_config = v_grid.VerticalGridConfig(
        num_levels=grid_savepoint.num(dims.KDim),
        flat_height=flat_height,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = v_grid.VerticalGridParams(
        vertical_config=vertical_config,
        vct_a=vct_a_field,
        vct_b=None,
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp,
    )
    assert (
        vertical_params.end_index_of_damping_layer
        == vct_a.shape[0] - math.ceil(damping_height / delta) - 1
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [REGIONAL_EXPERIMENT, GLOBAL_EXPERIMENT])
def test_nrdmax_calculation_from_icon_input(
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
    vertical_params = v_grid.VerticalGridParams(
        vertical_config=vertical_config,
        vct_a=a,
        vct_b=b,
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp,
    )
    assert nrdmax == vertical_params.end_index_of_damping_layer
    a_array = a.asnumpy()
    assert a_array[nrdmax] > damping_height
    assert a_array[nrdmax + 1] < damping_height


@pytest.mark.datatest
def test_grid_size(grid_savepoint):
    assert 65 == grid_savepoint.num(dims.KDim)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, kmoist_level", [(REGIONAL_EXPERIMENT, 0), (GLOBAL_EXPERIMENT, 25)]
)
def test_kmoist_calculation(grid_savepoint, experiment, kmoist_level):
    threshold = 22500.0
    vct_a = grid_savepoint.vct_a().asnumpy()
    assert kmoist_level == v_grid.VerticalGridParams._determine_kstart_moist(
        vct_a, threshold, nshift_total=0
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [REGIONAL_EXPERIMENT, GLOBAL_EXPERIMENT])
def test_kflat_calculation(grid_savepoint, experiment, flat_height):
    vct_a = grid_savepoint.vct_a().asnumpy()
    assert grid_savepoint.nflatlev() == v_grid.VerticalGridParams._determine_kstart_flat(
        vct_a, flat_height
    )


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
