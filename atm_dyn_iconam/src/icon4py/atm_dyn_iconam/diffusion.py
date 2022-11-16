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

# TODO [ml] initial RUN linit = TRUE
# def _setup_initial_diff_multfac_vn


@field_operator
def _setup_runtime_diff_multfac_vn(
    k4: float, dyn_substeps: float
) -> Field[[KDim], float]:
    con = 1.0 / 128.0
    dyn = k4 * dyn_substeps / 3.0
    return broadcast(minimum(con, dyn), (KDim,))


@field_operator
def _setup_smag_limit(diff_multfac_vn: Field[[KDim], float]) -> Field[[KDim], float]:
    return 0.125 - 4.0 * diff_multfac_vn


@program
def init_diffusion_local_fields(
    k4: float,
    dyn_substeps: float,
    diff_multfac_vn: Field[[KDim], float],
    smag_limit: Field[[KDim], float],
):
    _setup_runtime_diff_multfac_vn(k4, dyn_substeps, out=diff_multfac_vn)
    _setup_smag_limit(diff_multfac_vn, out=smag_limit)

#@field_operator
def _enhanced_smagorinski_factor():
    # aka diff_multfac_smag
    pass

#@field_operator
def _diff_multfac_n2w(shift: int):#  -> Field[[KDim], float]:
    pass

class DiffusionConfig:
    """contains necessary parameter to configure a diffusion run.

    - encapsulates namelist parameters and derived parameters (for now)
    """

    grid = GridConfig()
    ndyn_substeps = 5  # namelist mo_nonhydro_nml
    horizontal_diffusion_order = 5  # namelist
    lhdiff_rcf = True  # namelist, remove if always true
    hdiff_efdt_ratio = 24.0  # namelist
    lateral_boundary_denominator = DiffusionTupleVT(v=200.0, t=135.0)
    hdiff_smag_fac = 0.025 # namelist

    def substep_as_float(self):
        return float(self.ndyn_substeps)



class DiffusionParams:
    def __init__(self, config:DiffusionConfig):
        #TODO [ml] logging for case KX == 0
        #TODO [ml] calculation for x_dom (jg) > 2..n_dom

        self.K2: Final[float] = 1.0 / (config.hdiff_efdt_ratio * 8.0) \
            if config.hdiff_efdt_ratio > 0.0 else 0.0
        self.K4: Final[float] = self.K2 / 8.0
        self.K8: Final[float] = self.K2 / 64.0
        self.K4W: Final[float] = self.K2 / 4.0


class Diffusion:
    """Class that configures diffusion and does one diffusion step."""

    def __init__(self, config: DiffusionConfig, params: DiffusionParams):
        """
        Initialize Diffusion granule.

        TODO [ml]: initial run: linit = .TRUE.:  smag_offset and diff_multfac_vn are defined
        differently.
        """
        self.params: params

        # different for init call: smag_offset = 0
        self.smag_offset = 0.25 * params.K4 * config.substep_as_float()
        self.diff_multfac_w = np.minimum(
            1.0 / 48.0, params.K4W * config.substep_as_float()
        )

        # different for initial run!, through diff_multfac_vn
        self.diff_multfac_vn = np_as_located_field(KDim)(
            np.zeros(config.grid.get_k_size())
        )
        self.smag_limit = np_as_located_field(KDim)(np.zeros(config.grid.get_k_size()))

        init_diffusion_local_fields(
            params.K4,
            config.substep_as_float(),
            self.diff_multfac_vn,
            self.smag_limit,
            offset_provider={},
        )

        self.diff_multfac_n2w = np_as_located_field(KDim)(
            np.zeros(config.grid.get_k_size())
        )
        # TODO [ml] init diff_multfac_n2w

        self.enh_smag_fac = np_as_located_field(KDim)(
            np.zeros(config.grid.get_k_size())
        )
        # TODO [ml] init enh_smag_fac

    def do_step(
        self,
        diagnostic_state,
        prognostic_state,
        metric_state,
        interpolation_state,
        dtime,
    ):
        """
        Run a diffusion step.

            inputs are
            - output fields:
            - input fields that have changed from one time step to another:
            - simulation parameters: dtime - timestep
        """
        # -------
        # OUTLINE
        # -------
        # 1.  CALL rbf_vec_interpol_vertex
        # 2.  HALO EXCHANGE -- CALL sync_patch_array_mult
        # 3.  mo_nh_diffusion_stencil_01, mo_nh_diffusion_stencil_02, mo_nh_diffusion_stencil_03
        # 4.  IF (discr_vn > 1) THEN CALL sync_patch_array -> false for MCH
        # 5.  CALL rbf_vec_interpol_vertex_wp
        # 6.  HALO EXCHANGE -- CALL sync_patch_array_mult
        # 7.  mo_nh_diffusion_stencil_04, mo_nh_diffusion_stencil_05
        # 7a. IF (l_limited_area .OR. jg > 1) mo_nh_diffusion_stencil_06
        # 7b. mo_nh_diffusion_stencil_07, mo_nh_diffusion_stencil_08,
        #     mo_nh_diffusion_stencil_09, mo_nh_diffusion_stencil_10
        # 8.  HALO EXCHANGE: CALL sync_patch_array
        # 9.  mo_nh_diffusion_stencil_11, mo_nh_diffusion_stencil_12, mo_nh_diffusion_stencil_13,
        #     mo_nh_diffusion_stencil_14, mo_nh_diffusion_stencil_15, mo_nh_diffusion_stencil_16
        # 10. HALO EXCHANGE sync_patch_array
