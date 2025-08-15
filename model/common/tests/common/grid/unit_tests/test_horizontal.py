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
            e.match("Invalid zone")


@pytest.mark.parametrize("zone", zones())
def test_halo_zones(zone):
    if zone in (h_grid.Zone.HALO, h_grid.Zone.HALO_LEVEL_2):
        assert zone.is_halo()
    else:
        assert not zone.is_halo()

@pytest.mark.parametrize("zone", zones())
@pytest.mark.parametrize("dim", utils.horizontal_dims())
def test_get_refinement_control(zone, dim):
    zone = h_grid.Zone.LATERAL_BOUNDARY
    ref_ctrl = h_grid.get_refinement_control(dim, zone)
    assert 1 == ref_ctrl

