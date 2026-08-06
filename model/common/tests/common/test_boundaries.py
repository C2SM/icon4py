# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import textwrap

import pytest

from icon4py.model.common import boundaries
from icon4py.model.common.config import config_io
from icon4py.model.common.initial_condition.analytical import jablonowski_williamson as jw_ic
from icon4py.model.common.topography.analytical import (
    flat_topography as flat_topo,
    jablonowski_williamson as jw_topo,
)


def test_read_minimal() -> None:
    testee = config_io.read_yaml_str(
        textwrap.dedent(
            """
            initial_condition:
                type: jablonowski_williamson
            topography:
                type: flat
            """
        ),
        config_cls=boundaries.BoundariesConfig,
    )
    assert isinstance(testee.initial_condition, jw_ic.JablonowskiWilliamsonConfig)
    assert isinstance(testee.topography, flat_topo.FlatTopographyConfig)


def test_read_common_params() -> None:
    testee = config_io.read_yaml_str(
        textwrap.dedent(
            """
            params:
                u0: 42
                eta_0: 117
            initial_condition:
                type: jablonowski_williamson
            topography:
                type: jablonowski_williamson
            """,
        ),
        config_cls=boundaries.BoundariesConfig,
    )
    assert isinstance(testee.initial_condition, jw_ic.JablonowskiWilliamsonConfig)
    assert isinstance(testee.topography, jw_topo.JablonowskiWilliamsonConfig)
    assert testee.topography.u0 == testee.initial_condition.u0
    assert testee.topography.eta_0 == testee.initial_condition.eta_0


def test_read_rejects_unused_params() -> None:
    with pytest.raises(TypeError, match="'bar', 'foo'"):
        _ = config_io.read_yaml_str(
            textwrap.dedent(
                """
                params:
                    u0: 42
                    foo: not
                    bar: used
                initial_condition:
                    type: jablonowski_williamson
                topography:
                    type: jablonowski_williamson
                """
            ),
            config_cls=boundaries.BoundariesConfig,
        )
