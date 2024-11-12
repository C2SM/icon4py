# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import math
from icon4py.model.common.settings import xp

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.test_utils import datatest_utils as dt_utils, helpers
from icon4py.model.common.grid import geometry, topography as topo
from icon4py.model.driver.test_cases import artificial_topography


NUM_LEVELS = 65


@pytest.mark.parametrize(
    "max_h,damping_height,delta",
    [(60000, 34000, 612), (12000, 10000, 100), (109050, 45000, 123)],
)
def test_damping_layer_calculation(max_h, damping_height, delta, flat_height):
    vct_a = np.arange(0, max_h, delta)
    vct_a_field = gtx.as_field((dims.KDim,), data=vct_a[::-1])
    vertical_config = v_grid.VerticalGridConfig(
        num_levels=1000,
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
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
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
    "experiment, expected_moist_level",
    [(dt_utils.REGIONAL_EXPERIMENT, 0), (dt_utils.GLOBAL_EXPERIMENT, 25)],
)
def test_moist_level_calculation(grid_savepoint, experiment, expected_moist_level):
    threshold = 22500.0
    vertical_grid = configure_vertical_grid(grid_savepoint, top_moist_threshold=threshold)
    assert expected_moist_level == vertical_grid.kstart_moist
    assert expected_moist_level == vertical_grid.index(v_grid.Domain(dims.KDim, v_grid.Zone.MOIST))


@pytest.mark.datatest
def test_interface_physical_height(grid_savepoint):
    vertical_grid = configure_vertical_grid(grid_savepoint)
    assert helpers.dallclose(
        grid_savepoint.vct_a().asnumpy(), vertical_grid.interface_physical_height.asnumpy()
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_flat_level_calculation(grid_savepoint, experiment, flat_height):
    vertical_grid = configure_vertical_grid(grid_savepoint)

    assert grid_savepoint.nflatlev() == vertical_grid.nflatlev
    assert grid_savepoint.nflatlev() == vertical_grid.index(
        v_grid.Domain(dims.KDim, v_grid.Zone.FLAT)
    )


def offsets():
    for i in range(5):
        yield i


def vertical_zones():
    for z in v_grid.Zone.__members__.values():
        yield z


@pytest.mark.parametrize("zone", vertical_zones())
@pytest.mark.parametrize("kind", (gtx.DimensionKind.LOCAL, gtx.DimensionKind.HORIZONTAL))
def test_domain_raises_for_non_vertical_dim(zone, kind):
    dim = gtx.Dimension("I", kind=kind)
    with pytest.raises(AssertionError):
        v_grid.Domain(dim, zone)


@pytest.mark.datatest
@pytest.mark.parametrize("dim", [dims.KDim, dims.KHalfDim])
@pytest.mark.parametrize("offset", offsets())
def test_grid_index_top(grid_savepoint, dim, offset):
    vertical_grid = configure_vertical_grid(grid_savepoint)
    assert offset == vertical_grid.index(v_grid.Domain(dim, v_grid.Zone.TOP, offset))


@pytest.mark.datatest
@pytest.mark.parametrize("experiment, levels", [(dt_utils.GLOBAL_EXPERIMENT, 60)])
@pytest.mark.parametrize("dim", [dims.KDim, dims.KHalfDim])
@pytest.mark.parametrize("offset", offsets())
def test_grid_index_damping(grid_savepoint, experiment, levels, dim, offset):
    vertical_grid = configure_vertical_grid(grid_savepoint)
    upwards = -offset
    downwards = offset
    zone = v_grid.Zone.DAMPING
    domain = v_grid.Domain(dim, zone, upwards)
    assert vertical_grid.end_index_of_damping_layer + upwards == vertical_grid.index(domain)
    domain = v_grid.Domain(dim, zone, downwards)
    assert vertical_grid.end_index_of_damping_layer + downwards == vertical_grid.index(domain)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment, levels", [(dt_utils.GLOBAL_EXPERIMENT, 60)])
@pytest.mark.parametrize("dim", [dims.KDim, dims.KHalfDim])
@pytest.mark.parametrize("offset", offsets())
def test_grid_index_moist(grid_savepoint, experiment, levels, dim, offset):
    vertical_grid = configure_vertical_grid(grid_savepoint)
    upwards = -offset
    downwards = offset
    zone = v_grid.Zone.MOIST
    domain = v_grid.Domain(dim, zone, upwards)
    assert vertical_grid.kstart_moist + upwards == vertical_grid.index(domain)
    domain = v_grid.Domain(dim, zone, downwards)
    assert vertical_grid.kstart_moist + downwards == vertical_grid.index(domain)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment, levels", [(dt_utils.GLOBAL_EXPERIMENT, 60)])
@pytest.mark.parametrize("dim", [dims.KDim, dims.KHalfDim])
@pytest.mark.parametrize("offset", offsets())
def test_grid_index_flat(grid_savepoint, experiment, levels, dim, offset):
    vertical_grid = configure_vertical_grid(grid_savepoint)
    upwards = -offset
    downwards = offset
    zone = v_grid.Zone.FLAT
    domain = v_grid.Domain(dim, zone, upwards)
    assert vertical_grid.nflatlev + upwards == vertical_grid.index(domain)
    domain = v_grid.Domain(dim, zone, downwards)
    assert vertical_grid.nflatlev + downwards == vertical_grid.index(domain)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, levels",
    [(dt_utils.REGIONAL_EXPERIMENT, NUM_LEVELS), (dt_utils.GLOBAL_EXPERIMENT, 60)],
)
@pytest.mark.parametrize("dim", [dims.KDim, dims.KHalfDim])
@pytest.mark.parametrize("offset", offsets())
def test_grid_index_bottom(grid_savepoint, experiment, levels, dim, offset):
    valid_offset = -offset
    vertical_grid = configure_vertical_grid(grid_savepoint)
    num_levels = levels if dim == dims.KDim else levels + 1
    domain = v_grid.Domain(dim, v_grid.Zone.BOTTOM, valid_offset)
    assert num_levels + valid_offset == vertical_grid.index(domain)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment, levels", [(dt_utils.GLOBAL_EXPERIMENT, 60)])
@pytest.mark.parametrize("zone", vertical_zones())
@pytest.mark.parametrize("dim", [dims.KDim, dims.KHalfDim])
@pytest.mark.parametrize("offset", offsets())
def test_grid_index_raises_if_index_above_num_levels(
    grid_savepoint, experiment, levels, zone, dim, offset
):
    vertical_size = levels if dim == dims.KDim else levels + 1
    invalid_offset = vertical_size + 1 + offset
    vertical_grid = configure_vertical_grid(grid_savepoint)
    domain = v_grid.Domain(dim, zone, invalid_offset)
    with pytest.raises(expected_exception=AssertionError):
        vertical_grid.index(domain)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment, levels", [(dt_utils.GLOBAL_EXPERIMENT, 60)])
@pytest.mark.parametrize("zone", vertical_zones())
@pytest.mark.parametrize("dim", [dims.KDim, dims.KHalfDim])
@pytest.mark.parametrize("offset", offsets())
def test_grid_index_raises_if_index_below_zero(
    grid_savepoint, experiment, levels, zone, dim, offset
):
    vertical_size = levels if dim == dims.KDim else levels + 1
    invalid_offset = -(vertical_size + 1 + offset)
    vertical_grid = configure_vertical_grid(grid_savepoint)
    with pytest.raises(expected_exception=AssertionError):
        domain = v_grid.Domain(dim, zone, invalid_offset)
        vertical_grid.index(domain)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", (dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT))
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

    assert helpers.dallclose(vct_a.asnumpy(), grid_savepoint.vct_a().asnumpy())
    assert helpers.dallclose(vct_b.asnumpy(), grid_savepoint.vct_b().asnumpy())



@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.GAUSS3D_EXPERIMENT])
def test_init_vert_coord(
    grid_savepoint,
    metrics_savepoint,
    constant_fields_savepoint,
    experiment,
    icon_grid,
):

    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_config = v_grid.VerticalGridConfig(
        num_levels=grid_savepoint.num(dims.KDim),
    )
    vertical_geometry = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )
    topography = constant_fields_savepoint.topo_c()
    topography_smoothed = constant_fields_savepoint.topo_smt_c()

    z_ifc = v_grid.init_vert_coord(
        vct_a=vct_a,
        topography=topography,
        topography_smoothed=topography_smoothed,
        grid=icon_grid,
        vertical_config=vertical_config,
        vertical_geometry=vertical_geometry,
    )

    assert helpers.dallclose(z_ifc.ndarray, metrics_savepoint.z_ifc().ndarray, atol=1e-13)
