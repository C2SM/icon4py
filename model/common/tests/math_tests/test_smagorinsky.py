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

from icon4py.model.common import dimension as dims
from icon4py.model.common.math.smagorinsky import en_smag_fac_for_zero_nshift
from icon4py.model.common.test_utils.helpers import random_field, zero_field
from icon4py.model.common.test_utils.reference_funcs import (
    enhanced_smagorinski_factor_numpy,
)


# TODO (halungge) stencil does not run on embedded backend, broadcast(0.0, (dims.KDim,)) return scalar?
def test_init_enh_smag_fac(backend, grid):
    if backend is None:
        pytest.skip("test does not run on embedded backend")
    enh_smag_fac = zero_field(grid, dims.KDim)
    a_vec = random_field(grid, dims.KDim, low=1.0, high=10.0, extend={dims.KDim: 1})
    fac = (0.67, 0.5, 1.3, 0.8)
    z = (0.1, 0.2, 0.3, 0.4)

    enhanced_smag_fac_np = enhanced_smagorinski_factor_numpy(fac, z, a_vec.asnumpy())
    en_smag_fac_for_zero_nshift.with_backend(backend)(
        a_vec,
        *fac,
        *z,
        enh_smag_fac,
        offset_provider={"Koff": dims.KDim},
    )
    assert np.allclose(enhanced_smag_fac_np, enh_smag_fac.asnumpy())
