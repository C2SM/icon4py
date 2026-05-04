# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition.definitions import single_node_default

# Noop exchange for single-node tests, created from SingleNodeExchange.exchange
# with a bound dimension. SingleNodeExchange.exchange is inherently a noop,
# so the specific dimension does not matter.
noop_exchange = functools.partial(single_node_default.exchange, dims.CellDim)
