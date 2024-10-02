# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import logging

import pytest

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.horizontal as h_grid

from . import utils


log = logging.getLogger(__name__)


@pytest.mark.parametrize("dim", [dims.C2EDim, dims.C2E2C2EDim, dims.E2VDim, dims.V2EDim, dims.KDim])
def test_domain_raises_for_non_horizontal_dim(dim):
    with pytest.raises(AssertionError) as e:
        h_grid.domain(dim)
    e.match("horizontal dimensions")


def zones():
    for zone in h_grid.Zone.__members__.values():
        yield zone


@pytest.mark.parametrize("dim", utils.horizontal_dim())
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


@pytest.mark.parametrize("dim", utils.horizontal_dim())
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
