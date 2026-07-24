# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.common.config import reader


def test_read():
    config = reader.read(
        (pathlib.Path(__file__).parent / "data" / "test_config.yml").read_text(),
        diffusion.DiffusionConfig,
    )
    assert (
        config.shear_type
        is diffusion.TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND
    )
    assert config.ndyn_substeps == 4
    assert not config.apply_zdiffusion_t
