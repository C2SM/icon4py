# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import logging

import pytest

import icon4py.model
import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.horizontal as h_grid
from icon4py.model.testing import datatest_utils as dt_utils
import gt4py.next as gtx

from .. import utils


log = logging.getLogger(__name__)


@pytest.mark.parametrize("dim", utils.non_horizontal_dims())
def test_domain_raises_for_non_horizontal_dim(dim):
    with pytest.raises(AssertionError) as e:
        h_grid.domain(dim)
    e.match("horizontal dimensions")


def zones():
    for zone in h_grid.Zone.__members__.values():
        yield zone


@pytest.mark.parametrize("dim", utils.horizontal_dims())
@pytest.mark.parametrize("zone", zones())
def test_domain_raises_for_invalid_zones(dim, zone, caplog):
    caplog.set_level(logging.DEBUG)
    log.debug(f"dim={dim}, zone={zone},")
    if dim == dims.CellDim or dim == dims.VertexDim:
        if zone in (
            h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5,
            h_grid.Zone.LATERAL_BOUNDARY_LEVEL_6,
            h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7,
        ):
            with pytest.raises(AssertionError) as e:
                h_grid.domain(dim)(zone)
            e.match("not a valid zone")


@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_zone_and_domain_index(dim, caplog):
    """test mostly used for documentation purposes"""
    caplog.set_level(logging.INFO)
    for zone in zones():
        try:
            domain = h_grid.domain(dim)(zone)
            log.info(f"dim={dim}: zone={zone:16}: index={domain():3}")
            assert domain() <= h_grid._BOUNDS[dim][1]
        except AssertionError:
            log.info(f"dim={dim}: zone={zone:16}: invalid")


@pytest.mark.parametrize("zone", zones())
def test_halo_zones(zone):
    if zone in (h_grid.Zone.HALO, h_grid.Zone.HALO_LEVEL_2):
        assert zone.is_halo()
    else:
        assert not zone.is_halo()


@pytest.mark.datatest
@pytest.mark.parametrize("grid_file", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.R02B04_GLOBAL])
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_map_domain_bounds_start_index(grid_file, dim):
    grid = utils.run_grid_manager(dt_utils.R02B04_GLOBAL, keep_skip_values=True, backend=None).grid

    start_index_array = grid._start_indices[dim]
    _map_and_assert_array(dim, start_index_array)


@pytest.mark.datatest
@pytest.mark.parametrize("grid_file", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.R02B04_GLOBAL])
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_map_domain_bounds_end_index(grid_file, dim):
    grid = utils.run_grid_manager(dt_utils.R02B04_GLOBAL, keep_skip_values=True, backend=None).grid

    end_index_array = grid._end_indices[dim]
    _map_and_assert_array(dim, end_index_array)


def _map_and_assert_array(dim, index_array):
    index_map = icon4py.model.common.grid.horizontal.map_domain_bounds(dim, index_array)
    for d, index in index_map.items():
        assert index == index_array[d._index]
        assert isinstance(index, gtx.int32)
