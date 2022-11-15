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

from collections import namedtuple
from typing import Final

import numpy as np
from functional.common import Field
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import broadcast, minimum
from functional.iterator.embedded import np_as_located_field

from icon4py.atm_dyn_iconam.grid import GridConfig
from icon4py.common.dimension import KDim


DiffusionTupleVT = namedtuple("DiffusionParamVT", "v t")

# TODO
# def _setup_initial_diff_multfac_vn


@field_operator
def _setup_runtime_diff_multfac_vn(k4: float, dyn_substeps: float):
    con = 1.0 / 128.0
    dyn = k4 * dyn_substeps / 3.0
    return broadcast(minimum(con, dyn), (KDim,))


@field_operator
def _setup_smag_limit(diff_multfac_vn: Field[[KDim], float]):
    return 0.125 - 4.0 * diff_multfac_vn


@program
def init_diffusion_local_fields(
    k4: float,
    dyn_substeps: float,
    diff_multfac_vn: Field[[KDim], float],
    smag_limit: Field[[KDim], float],
) -> tuple[Field[[KDim], float], Field[[KDim], float]]:
    _setup_runtime_diff_multfac_vn(k4, dyn_substeps, out=diff_multfac_vn)
    _setup_smag_limit(diff_multfac_vn, out=smag_limit)


class DiffusionConfig:
    """contains necessary parameter to configure a diffusion run.

    - encapsulates namelist parameters
    """

    grid = GridConfig()
    ndyn_substeps = 5  # namelist mo_nonhydro_nml
    horizontal_diffusion_order = 5  # namelist
    lhdiff_rcf = True  # namelist, remove if always true
    hdiff_efdt_ratio = 24.0  # namelist
    lateral_boundary_denominator = DiffusionTupleVT(v=200.0, t=135.0)

    # TODO [ml] keep those derived params in config or move to diffustion.__init__
    K2: Final[float] = 1.0 / (hdiff_efdt_ratio * 8.0)
    K4: Final[float] = K2 / 8.0
    K8: Final[float] = K4 / 8.0

    def substep_as_float(self):
        return float(self.ndyn_substeps)


class Diffusion:
    """class that configures diffusion and does one diffusion step."""

    def __init__(self, config: DiffusionConfig):
        """
        TODO [ml]: handle initial run: linit = .TRUE.:  smag_offset and diff_multfac_vn are defined
                differently.
        """
        # different for init call smag_offset = 0
        self.smag_offset = 0.25 * config.K4 * config.substep_as_float()

        # different for initial run!, through diff_multfac_vn
        self.diff_multfac_vn = np_as_located_field(KDim)(
            np.zeros(config.grid.get_k_size())
        )
        self.smag_limit = np_as_located_field(KDim)(np.zeros(config.grid.get_k_size()))

        init_diffusion_local_fields(
            config.K4,
            config.substep_as_float(),
            self.diff_multfac_vn,
            self.smag_limit,
            offset_provider={},
        )
