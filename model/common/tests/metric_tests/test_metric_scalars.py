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

import numpy as np
import pytest
from gt4py.next import as_field
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common.dimension import KDim
from icon4py.model.common.metrics.metric_fields import compute_scalfac_dd3d
from icon4py.model.common.metrics.metric_scalars import compute_kstart_dd3d
from icon4py.model.common.test_utils.helpers import (
    dallclose,
    zero_field,
)


@pytest.mark.datatest
def test_compute_dd3d(icon_grid, metrics_savepoint, grid_savepoint, backend):
    scalfac_dd3d_ref = metrics_savepoint.scalfac_dd3d()
    kstart_dd3d_ref = 2  # metrics_savepoint.kstart_dd3d_ref # TODO: ref value not serialized
    scalfac_dd3d_full = zero_field(icon_grid, KDim)
    k_index = as_field((KDim,), np.arange(icon_grid.num_levels, dtype=int32))
    divdamp_trans_start = 12500.0
    divdamp_trans_end = 17500.0
    divdamp_type = 3

    compute_scalfac_dd3d.with_backend(backend=backend)(
        vct_a=grid_savepoint.vct_a(),
        scalfac_dd3d=scalfac_dd3d_full,
        divdamp_trans_start=divdamp_trans_start,
        divdamp_trans_end=divdamp_trans_end,
        divdamp_type=divdamp_type,
        vertical_start=int32(0),
        vertical_end=icon_grid.num_levels,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )
    kstart_dd3d = compute_kstart_dd3d(
        vct_a=grid_savepoint.vct_a().asnumpy(),
        k_levels=k_index.asnumpy(),
        divdamp_trans_end=divdamp_trans_end,
        divdamp_type=divdamp_type,
    )

    assert dallclose(scalfac_dd3d_ref.asnumpy(), scalfac_dd3d_full.asnumpy())
    assert dallclose(kstart_dd3d, kstart_dd3d_ref)
