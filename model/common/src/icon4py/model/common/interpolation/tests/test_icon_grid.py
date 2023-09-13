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
import pytest

from icon4py.model.common.dimension import CellDim, EdgeDim, VertexDim
from icon4py.grid.horizontal import HorizontalMarkerIndex


@pytest.mark.datatest
# TODO(Magdalena) HorizontalMarkerIndex.local(dim) does not yield equvalent results form grid file
#  and serialized data, why?. Serialized data has those strange -1 values
@pytest.mark.parametrize(
    "dim, marker, index",
    [
        (CellDim, HorizontalMarkerIndex.interior(CellDim), 20896),
        (CellDim, HorizontalMarkerIndex.interior(CellDim) + 1, 850),
        (CellDim, HorizontalMarkerIndex.local(CellDim) - 2, 20896),
        (CellDim, HorizontalMarkerIndex.local(CellDim) - 1, 20896),
        (CellDim, HorizontalMarkerIndex.local(CellDim), 20896),
        (CellDim, HorizontalMarkerIndex.nudging(CellDim), 4104),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 3, 3316),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 2, 2511),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 1, 1688),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 0, 850),
        (EdgeDim, HorizontalMarkerIndex.interior(EdgeDim), 31558),
        (EdgeDim, HorizontalMarkerIndex.local(EdgeDim) - 2, 31558),
        (EdgeDim, HorizontalMarkerIndex.local(EdgeDim) - 1, 31558),
        (EdgeDim, HorizontalMarkerIndex.local(EdgeDim), 31558),
        (EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim) + 1, 6176),
        (EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim), 5387),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 7, 4989),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6, 4184),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 5, 3777),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4, 2954),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 3, 2538),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 2, 1700),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1, 1278),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 0, 428),
        (VertexDim, HorizontalMarkerIndex.interior(VertexDim), 10663),
        (VertexDim, HorizontalMarkerIndex.local(VertexDim) - 2, 10663),
        (VertexDim, HorizontalMarkerIndex.local(VertexDim) - 1, 10663),
        (VertexDim, HorizontalMarkerIndex.local(VertexDim), 10663),
        (VertexDim, HorizontalMarkerIndex.nudging(VertexDim) + 1, 10663),
        (VertexDim, HorizontalMarkerIndex.nudging(VertexDim), 10663),
        (VertexDim, HorizontalMarkerIndex.end(VertexDim), 10663),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 4, 2071),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 3, 1673),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 2, 1266),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1, 850),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 0, 428),
    ],
)
def test_horizontal_end_index(icon_grid, dim, marker, index):
    assert index == icon_grid.get_end_index(dim, marker)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "dim, marker, index",
    [
        (CellDim, HorizontalMarkerIndex.interior(CellDim), 4104),
        (CellDim, HorizontalMarkerIndex.interior(CellDim) + 1, 0),
        (CellDim, HorizontalMarkerIndex.local(CellDim) - 1, 20896),
        (CellDim, HorizontalMarkerIndex.local(CellDim), -1),
        (CellDim, HorizontalMarkerIndex.halo(CellDim), 20896),
        (CellDim, HorizontalMarkerIndex.nudging(CellDim), 3316),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 3, 2511),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 2, 1688),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 1, 850),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 0, 0),
        (EdgeDim, HorizontalMarkerIndex.interior(EdgeDim), 6176),
        (EdgeDim, HorizontalMarkerIndex.local(EdgeDim) - 2, 31558),
        (EdgeDim, HorizontalMarkerIndex.local(EdgeDim) - 1, 31558),
        (EdgeDim, HorizontalMarkerIndex.local(EdgeDim), -1),  # ????
        (EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim) + 1, 5387),
        (EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim), 4989),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 7, 4184),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6, 3777),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 5, 2954),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4, 2538),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 3, 1700),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 2, 1278),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1, 428),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 0, 0),
        (VertexDim, HorizontalMarkerIndex.interior(VertexDim), 2071),
        (VertexDim, HorizontalMarkerIndex.local(VertexDim) - 1, 10663),
        (VertexDim, HorizontalMarkerIndex.local(VertexDim), -1),  # ???
        (VertexDim, HorizontalMarkerIndex.nudging(VertexDim) + 1, 10663),
        (VertexDim, HorizontalMarkerIndex.nudging(VertexDim), 10663),
        (VertexDim, HorizontalMarkerIndex.end(VertexDim), 10663),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 4, 1673),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 3, 1266),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 2, 850),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1, 428),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 0, 0),
    ],
)
def test_horizontal_start_index(icon_grid, dim, marker, index):
    assert index == icon_grid.get_start_index(dim, marker)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "start_marker, end_marker, expected_bounds",
    [
        (
            HorizontalMarkerIndex.lateral_boundary(CellDim),
            HorizontalMarkerIndex.lateral_boundary(CellDim),
            (0, 850),
        ),
        (
            HorizontalMarkerIndex.lateral_boundary(CellDim) + 1,
            HorizontalMarkerIndex.lateral_boundary(CellDim) + 1,
            (850, 1688),
        ),
        (
            HorizontalMarkerIndex.lateral_boundary(CellDim) + 2,
            HorizontalMarkerIndex.lateral_boundary(CellDim) + 2,
            (1688, 2511),
        ),
        (
            HorizontalMarkerIndex.lateral_boundary(CellDim) + 3,
            HorizontalMarkerIndex.lateral_boundary(CellDim) + 3,
            (2511, 3316),
        ),
        (
            HorizontalMarkerIndex.interior(CellDim),
            HorizontalMarkerIndex.interior(CellDim),
            (4104, 20896),
        ),
        (
            HorizontalMarkerIndex.interior(CellDim) + 1,
            HorizontalMarkerIndex.interior(CellDim) + 1,
            (0, 850),
        ),
        (
            HorizontalMarkerIndex.nudging(CellDim),
            HorizontalMarkerIndex.nudging(CellDim),
            (
                3316,
                4104,
            ),
        ),
        (
            HorizontalMarkerIndex.end(CellDim),
            HorizontalMarkerIndex.end(CellDim),
            (
                20896,
                20896,
            ),
        ),
        (
            HorizontalMarkerIndex.halo(CellDim),
            HorizontalMarkerIndex.halo(CellDim),
            (
                20896,
                20896,
            ),
        ),
        (
            HorizontalMarkerIndex.local(CellDim),
            HorizontalMarkerIndex.local(CellDim),
            (-1, 20896),
        ),
    ],
)
def test_horizontal_cell_markers(icon_grid, start_marker, end_marker, expected_bounds):
    assert (
        icon_grid.get_indices_from_to(
            CellDim,
            start_marker,
            end_marker,
        )
        == expected_bounds
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "start_marker, end_marker, expected_bounds",
    [
        (
            HorizontalMarkerIndex.lateral_boundary(EdgeDim),
            HorizontalMarkerIndex.lateral_boundary(EdgeDim),
            (0, 428),
        ),
        (
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
            (428, 1278),
        ),
        (
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 2,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 2,
            (1278, 1700),
        ),
        (
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 3,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 3,
            (1700, 2538),
        ),
        (
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4,
            (2538, 2954),
        ),
        (
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 5,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 5,
            (2954, 3777),
        ),
        (
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6,
            (3777, 4184),
        ),
        (
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 7,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 7,
            (4184, 4989),
        ),
        (
            HorizontalMarkerIndex.interior(EdgeDim),
            HorizontalMarkerIndex.interior(EdgeDim),
            (6176, 31558),
        ),
        (
            HorizontalMarkerIndex.nudging(EdgeDim),
            HorizontalMarkerIndex.nudging(EdgeDim),
            (
                4989,
                5387,
            ),
        ),
        (
            HorizontalMarkerIndex.nudging(EdgeDim) + 1,
            HorizontalMarkerIndex.nudging(EdgeDim) + 1,
            (5387, 6176),
        ),
        (
            HorizontalMarkerIndex.end(EdgeDim),
            HorizontalMarkerIndex.end(EdgeDim),
            (
                31558,
                31558,
            ),
        ),
        (
            HorizontalMarkerIndex.halo(EdgeDim),
            HorizontalMarkerIndex.halo(EdgeDim),
            (
                31558,
                31558,
            ),
        ),
        (
            HorizontalMarkerIndex.local(EdgeDim),
            HorizontalMarkerIndex.local(EdgeDim),
            (-1, 31558),
        ),
    ],
)
def test_horizontal_edge_markers(icon_grid, start_marker, end_marker, expected_bounds):
    assert (
        icon_grid.get_indices_from_to(
            EdgeDim,
            start_marker,
            end_marker,
        )
        == expected_bounds
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "start_marker, end_marker, expected_bounds",
    [
        (
            HorizontalMarkerIndex.lateral_boundary(VertexDim),
            HorizontalMarkerIndex.lateral_boundary(VertexDim),
            (0, 428),
        ),
        (
            HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1,
            HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1,
            (428, 850),
        ),
        (
            HorizontalMarkerIndex.lateral_boundary(VertexDim) + 2,
            HorizontalMarkerIndex.lateral_boundary(VertexDim) + 2,
            (850, 1266),
        ),
        (
            HorizontalMarkerIndex.lateral_boundary(VertexDim) + 3,
            HorizontalMarkerIndex.lateral_boundary(VertexDim) + 3,
            (1266, 1673),
        ),
        (
            HorizontalMarkerIndex.interior(VertexDim),
            HorizontalMarkerIndex.interior(VertexDim),
            (2071, 10663),
        ),
        (
            HorizontalMarkerIndex.interior(VertexDim) + 1,
            HorizontalMarkerIndex.interior(VertexDim) + 1,
            (0, 428),
        ),
        (
            HorizontalMarkerIndex.end(CellDim),
            HorizontalMarkerIndex.end(CellDim),
            (
                10663,
                10663,
            ),
        ),
        (
            HorizontalMarkerIndex.halo(VertexDim),
            HorizontalMarkerIndex.halo(VertexDim),
            (
                10663,
                10663,
            ),
        ),
        (
            HorizontalMarkerIndex.local(VertexDim),
            HorizontalMarkerIndex.local(VertexDim),
            (-1, 10663),
        ),
    ],
)
def test_horizontal_vertex_markers(
    icon_grid, start_marker, end_marker, expected_bounds
):
    assert (
        icon_grid.get_indices_from_to(
            VertexDim,
            start_marker,
            end_marker,
        )
        == expected_bounds
    )


@pytest.mark.datatest
def test_cross_check_marker_equivalences(icon_grid):
    """Check actual equivalences of calculated markers."""
    # TODO(Magdalena): This should go away once we refactor these markers in a good way, such that no calculation need to be done with them anymore.

    assert icon_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.local(CellDim) - 1,
        HorizontalMarkerIndex.local(CellDim) - 1,
    ) == icon_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.halo(CellDim),
        HorizontalMarkerIndex.halo(CellDim),
    )
    assert icon_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.nudging(CellDim) - 1,
        HorizontalMarkerIndex.nudging(CellDim) - 1,
    ) == icon_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) + 3,
        HorizontalMarkerIndex.lateral_boundary(CellDim) + 3,
    )
    assert icon_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.local(EdgeDim) - 1,
        HorizontalMarkerIndex.local(EdgeDim) - 1,
    ) == icon_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.halo(EdgeDim),
        HorizontalMarkerIndex.halo(EdgeDim),
    )

    assert icon_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 8,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 8,
    ) == icon_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.nudging(EdgeDim),
        HorizontalMarkerIndex.nudging(EdgeDim),
    )
