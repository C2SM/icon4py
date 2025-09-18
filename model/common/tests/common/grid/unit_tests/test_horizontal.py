# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import gt4py.next as gtx
import pytest

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.horizontal as h_grid

from .. import utils
from ..fixtures import *  # noqa: F401, F403


if TYPE_CHECKING:
    from collections.abc import Iterator

    import gt4py.next as gtx

log = logging.getLogger(__name__)


@pytest.mark.parametrize("dim", utils.non_horizontal_dims())
def test_domain_raises_for_non_horizontal_dim(dim: gtx.Dimension) -> None:
    with pytest.raises(AssertionError) as e:
        h_grid.domain(dim)
    e.match("horizontal dimensions")


def zones() -> Iterator[h_grid.Zone]:
    yield from h_grid.Zone.__members__.values()


@pytest.mark.parametrize("dim", utils.horizontal_dims())
@pytest.mark.parametrize("zone", zones())
def test_domain_raises_for_invalid_zones(dim: gtx.Dimension, zone: h_grid.Zone) -> None:
    if dim in (dims.CellDim, dims.VertexDim):
        if zone in (
            h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5,
            h_grid.Zone.LATERAL_BOUNDARY_LEVEL_6,
            h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7,
        ):
            with pytest.raises(AssertionError) as e:
                h_grid.domain(dim)(zone)
            e.match("Invalid zone")


@pytest.mark.parametrize("zone", zones())
def test_halo_zones(zone: h_grid.Zone) -> None:
    if zone in (h_grid.Zone.HALO, h_grid.Zone.HALO_LEVEL_2):
        assert zone.is_halo()
    else:
        assert not zone.is_halo()
