# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys import saturation_adjustment
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.test_utils.helpers import dallclose, zero_field


@pytest.mark.parametrize(
    "t, qc, qi, ni, dvsi, dt",
    [
        (272.731, 0.0, 2.02422e-23, 5.05089, -0.000618828, 30.0),
    ],
)

def test_ice_deposition_nucleation(
    t,
    qc,
    qi,
    ni,
    dvsi,
    dt,
):
    ice_deposition_nucleation.run(
        t=t,
        qc=qc,
        qi=qi,
        ni=ni,
        dvsi=dvsi,
        dt=dt,
    )

    assert dallclose(
        updated_qc,
        gscp_satad_exit_savepoint.qc().ndarray,
        atol=1.0e-13,
    )
