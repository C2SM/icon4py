# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.common.math import distance_array_ns


def test_horizontal_distance_to_point_no_wrap() -> None:
    # Default: plain Euclidean distance, ignoring any periodicity (ICON behaviour).
    x = np.array([0.0, 3.0, 9.0])
    y = np.array([0.0, 4.0, 0.0])
    dist = distance_array_ns.horizontal_distance_to_point(x=x, y=y, point_x=0.0, point_y=0.0)
    np.testing.assert_allclose(dist, [0.0, 5.0, 9.0])


def test_horizontal_distance_to_point_wrap_uses_nearest_image() -> None:
    # With wrapping a point near the far edge is closest to the target via the
    # periodic image, so its distance is small.
    x = np.array([1.0, 9.0])
    y = np.array([0.0, 0.0])
    dist = distance_array_ns.horizontal_distance_to_point(
        x=x, y=y, point_x=0.0, point_y=0.0, domain_length=10.0, domain_height=10.0, wrap=True
    )
    np.testing.assert_allclose(dist, [1.0, 1.0])


def test_horizontal_distance_to_point_wrap_requires_domain_extents() -> None:
    x = np.array([1.0])
    y = np.array([1.0])
    with pytest.raises(ValueError, match="domain_length"):
        distance_array_ns.horizontal_distance_to_point(
            x=x, y=y, point_x=0.0, point_y=0.0, wrap=True
        )
