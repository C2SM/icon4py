# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from collections.abc import Mapping

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common.grid import base


@dataclasses.dataclass
class _ConnectivityConceptFixer:
    """
    This works around a misuse of dimensions as an identifier for connectivities.
    Since GT4Py might change the way the mesh is represented, we could
    keep this for a while, otherwise we need to touch all StencilTests.
    """

    _grid: base.Grid

    def __getitem__(self, dim: gtx.Dimension | str) -> np.ndarray:
        if isinstance(dim, gtx.Dimension):
            dim = dim.value
        return self._grid.get_connectivity(dim).asnumpy()


@pytest.fixture(scope="session")
def connectivities_as_numpy(grid: base.Grid) -> Mapping[gtx.Dimension, np.ndarray]:
    return _ConnectivityConceptFixer(grid)  # type: ignore[return-value] # it's not exactly a Mapping but doesn't make sense to use a Protocol here
