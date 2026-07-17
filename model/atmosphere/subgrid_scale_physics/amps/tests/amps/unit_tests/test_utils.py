# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import sys

from icon4py.model.atmosphere.subgrid_scale_physics.amps import utils


class TestRecursionLimit:
    def test_raises_limit_within_context(self):
        original = sys.getrecursionlimit()
        with utils.recursion_limit(original + 12345):
            assert sys.getrecursionlimit() == original + 12345
        assert sys.getrecursionlimit() == original

    def test_restores_limit_on_exception(self):
        original = sys.getrecursionlimit()
        try:
            with utils.recursion_limit(original + 999):
                raise ValueError("boom")
        except ValueError:
            pass
        assert sys.getrecursionlimit() == original
