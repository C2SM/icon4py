# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import NamedTuple, TypeAlias, TypeVar

import pytest

from icon4py.model.common.grid import base
from icon4py.model.common.utils import device_utils
from icon4py.model.testing import test_utils


T = TypeVar("T")
Predicate: TypeAlias = Callable[[T], bool]


class ItemFilter(NamedTuple):
    """Test item filter definition."""

    condition: Predicate[pytest.Item]
    action: Callable[[], None]


item_marker_filters: dict[str, ItemFilter] = {
    "cpu_only": ItemFilter(
        condition=lambda item: device_utils.is_cupy_device(
            test_utils.get_fixture_value("backend", item)
        ),
        action=functools.partial(pytest.skip, "currently only runs on CPU"),
    ),
    "embedded_only": ItemFilter(
        condition=lambda item: not test_utils.is_embedded(
            test_utils.get_fixture_value("backend", item)
        ),
        action=functools.partial(pytest.skip, "stencil runs only on embedded backend"),
    ),
    "embedded_remap_error": ItemFilter(
        condition=lambda item: test_utils.is_embedded(
            test_utils.get_fixture_value("backend", item)
        ),
        action=functools.partial(
            pytest.xfail, "Embedded backend currently fails in remap function."
        ),
    ),
    "embedded_static_args": ItemFilter(
        condition=lambda item: test_utils.is_embedded(
            test_utils.get_fixture_value("backend", item)
        ),
        action=functools.partial(
            pytest.xfail, " gt4py _compiled_programs returns error when backend is None."
        ),
    ),
    "uses_as_offset": ItemFilter(
        condition=lambda item: test_utils.is_embedded(
            test_utils.get_fixture_value("backend", item)
        ),
        action=functools.partial(pytest.xfail, "Embedded backend does not support as_offset."),
    ),
    "uses_concat_where": ItemFilter(
        condition=lambda item: test_utils.is_embedded(
            test_utils.get_fixture_value("backend", item)
        ),
        action=functools.partial(pytest.xfail, "Embedded backend does not support concat_where."),
    ),
    "gtfn_too_slow": ItemFilter(
        condition=lambda item: test_utils.is_gtfn_backend(
            test_utils.get_fixture_value("backend", item)
        ),
        action=functools.partial(pytest.skip, "GTFN compilation is too slow for this test."),
    ),
    "skip_value_error": ItemFilter(
        condition=lambda item: (grid := test_utils.get_fixture_value("grid", item)).limited_area
        or grid.geometry_type == base.GeometryType.ICOSAHEDRON,
        action=functools.partial(
            pytest.skip,
            "Stencil does not support domain containing skip values. Consider shrinking domain.",
        ),
    ),
}
