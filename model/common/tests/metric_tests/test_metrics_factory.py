# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import icon4py.model.common.settings as settings
from icon4py.model.common.metrics import metrics_factory


def test_factory(icon_grid):
    factory = metrics_factory.fields_factory
    factory.with_grid(icon_grid).with_allocator(settings.backend)
    factory.get("height_on_interface_levels", metrics_factory.RetrievalType.FIELD)
