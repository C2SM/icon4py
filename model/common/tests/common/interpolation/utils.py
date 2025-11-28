# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import gt4py.next as gtx

from icon4py.model.common.utils import data_allocation as data_alloc


def dummy_exchange_buffer(dim: Sequence[gtx.Dimension], *field: data_alloc.NDArray) -> None:
    return None
