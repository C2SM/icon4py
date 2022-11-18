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
import math
from collections import namedtuple
from typing import Final

import numpy as np
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, broadcast, maximum, minimum
from functional.iterator.embedded import np_as_located_field

from icon4py.atm_dyn_iconam.vertical import (
    VerticalGridConfig,
    VerticalModelParams,
)
from icon4py.common.dimension import KDim, Koff


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


@field_operator
def _init_diffusion_local_fields(
    k4: float, dyn_substeps: float
) -> tuple[Field[[KDim], float], Field[[KDim], float]]:
    diff_multfac_vn = _setup_runtime_diff_multfac_vn(k4, dyn_substeps)
    smag_limit = _setup_smag_limit(diff_multfac_vn)
    return diff_multfac_vn, smag_limit


@program
def init_diffusion_local_fields(
    k4: float,
    dyn_substeps: float,
    diff_multfac_vn: Field[[KDim], float],
    smag_limit: Field[[KDim], float],
):
    _init_diffusion_local_fields(k4, dyn_substeps, out=(diff_multfac_vn, smag_limit))


@field_operator
def _en_smag_fac_for_zero_nshift(
    hdiff_smag_fac: float,
    hdiff_smag_fac2: float,
    hdiff_smag_fac3: float,
    hdiff_smag_fac4: float,
    hdiff_smag_z: float,
    hdiff_smag_z2: float,
    hdiff_smag_z3: float,
    hdiff_smag_z4: float,
    vect_a: Field[[KDim], float],
) -> Field[[KDim], float]:
    dz21 = hdiff_smag_z2 - hdiff_smag_z
    alin = (hdiff_smag_fac2 - hdiff_smag_fac) / dz21
    df32 = hdiff_smag_fac3 - hdiff_smag_fac2
    df42 = hdiff_smag_fac4 - hdiff_smag_fac2
    dz32 = hdiff_smag_z3 - hdiff_smag_z2
    dz42 = hdiff_smag_z4 - hdiff_smag_z2

    bqdr = (df42 * dz32 - df32 * dz42) / (dz32 * dz42 * (dz42 - dz32))
    aqdr = df32 / dz32 - bqdr * dz32
    zf = 0.5 * (vect_a + vect_a(Koff[1]))

    dzlin = minimum(dz21, maximum(0.0, zf - hdiff_smag_z))
    dzqdr = minimum(dz42, maximum(0.0, zf - hdiff_smag_z2))
    enh_smag_fac = hdiff_smag_fac + (dzlin * alin) + dzqdr * (aqdr + dzqdr * bqdr)
    return enh_smag_fac


@program
def enhanced_smagorinski_factor(
    hdiff_smag_fac: float,
    hdiff_smag_fac2: float,
    hdiff_smag_fac3: float,
    hdiff_smag_fac4: float,
    hdiff_smag_z: float,
    hdiff_smag_z2: float,
    hdiff_smag_z3: float,
    hdiff_smag_z4: float,
    vect_a: Field[[KDim], float],
    enh_smag_fac: Field[[KDim], float],
):
    _en_smag_fac_for_zero_nshift(
        hdiff_smag_fac,
        hdiff_smag_fac2,
        hdiff_smag_fac3,
        hdiff_smag_fac4,
        hdiff_smag_z,
        hdiff_smag_z2,
        hdiff_smag_z3,
        hdiff_smag_z4,
        vect_a,
        out=enh_smag_fac,
    )


@field_operator
def _set_zero_k():
    return broadcast(0.0, (KDim,))


class DiffusionConfig:
    """contains necessary parameter to configure a diffusion run.

    - encapsulates namelist parameters and derived parameters (for now)

    currently we use the MCH r04b09_dsl experiment as constants here. These should
    be read from config and the default from mo_diffusion_nml.f90 set as defaults.



    TODO: [ml] read from config
    TODO: [ml] handle dependencies on other namelists (below...)
    """

    # TODO [ml]: external stuff grid related, p_patch,
    grid = VerticalGridConfig()
    vertical_params = VerticalModelParams(
        rayleigh_damping_height=12500, vct_a=np.zeros(grid.get_k_size())
    )

    # from namelist diffusion_nml
    diffusion_type = 5  # hdiff_order ! order of nabla operator for diffusion
    lhdiff_vn = True  # ! diffusion on the horizontal wind field
    lhdiff_temp = True  # ! diffusion on the temperature field
    lhdiff_w = True  # ! diffusion on the vertical wind field
    lhdiff_rcf = True  # namelist, remove if always true
    itype_vn_diffu = 1  # ! reconstruction method used for Smagorinsky diffusion
    itype_t_diffu = 2  # ! discretization of temperature diffusion
    hdiff_efdt_ratio = 24.0  # ! ratio of e-folding time to time step
    hdiff_smag_fac = 0.025  # ! scaling factor for Smagorinsky diffusion
    # defaults:

    #
    # from parent namelist nonhydrostatic_nml
    l_zdiffu_t = (
        True  # ! l_zdiffu_t: specifies computation of Smagorinsky temperature diffusion
    )
    damp_height = 12500
    ndyn_substeps = 5

    # from other namelists
    # namelist gridref_nml
    # denom_diffu_v = 150   ! denominator for lateral boundary diffusion of velocity
    lateral_boundary_denominator = DiffusionTupleVT(v=200.0, t=135.0)

    # namelist grid_nml -> TODO [ml] could go to grid config?
    l_limited_area = True

    def substep_as_float(self):
        return float(self.ndyn_substeps)


class DiffusionParams:
    """Calculates derived quantities depending on the diffusion config."""

    def __init__(self, config: DiffusionConfig):
        # TODO [ml] logging for case KX == 0
        # TODO [ml] generrally calculation for x_dom (jg) > 2..n_dom, why is jg special

        self.K2: Final[float] = (
            1.0 / (config.hdiff_efdt_ratio * 8.0)
            if config.hdiff_efdt_ratio > 0.0
            else 0.0
        )
        self.K4: Final[float] = self.K2 / 8.0
        self.K8: Final[float] = self.K2 / 64.0
        self.K4W: Final[float] = self.K2 / 4.0
        (
            self.smagorinski_factor,
            self.smagorinski_height,
        ) = self.determine_smagorinski_factor(config)

    def determine_smagorinski_factor(self, config: DiffusionConfig):
        """Enhanced Smagorinsky diffusion factor.

        Smagorinsky diffusion factor is defined as a profile in height
        above sea level with 4 height sections.

        It is calculated/used only in the case of diffusion_type 3 or 5
        """
        match config.diffusion_type:
            case 5:
                (
                    smagorinski_factor,
                    smagorinski_height,
                ) = self.diffusion_type_5_smagorinski_factor(config)
            case 4:
                # according to mo_nh_diffusion.f90 this isn't used anywhere the factor is only
                # used for diffusion_type (3,5) but the defaults are only defined for iequations=3
                smagorinski_factor = (
                    config.hdiff_smag_fac if config.hdiff_smag_fac else 0.15,
                )
                smagorinski_height = None
            case _:
                pass
        return smagorinski_factor, smagorinski_height

    @staticmethod
    def diffusion_type_5_smagorinski_factor(config: DiffusionConfig):
        # initial values from mo_diffusion_nml.f90
        magic_sqrt = math.sqrt(1600.0 * (1600 + 50000.0))
        magic_fac2_value = 2e-6 * (1600.0 + 25000.0 + magic_sqrt)
        magic_z2 = 1600.0 + 50000.0 + magic_sqrt
        factor = (config.hdiff_smag_fac, magic_fac2_value, 0.0, 1.0)
        heights = (32500.0, magic_z2, 50000.0, 90000.0)
        return factor, heights


class Diffusion:
    """Class that configures diffusion and does one diffusion step."""

    def __init__(
        self,
        config: DiffusionConfig,
        params: DiffusionParams,
        a_vect: Field[[KDim], float],
    ):
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

        self.enh_smag_fac = np_as_located_field(KDim)(
            np.zeros(config.grid.get_k_size())
        )
        enhanced_smagorinski_factor(
            *params.smagorinski_factor,
            *params.smagorinski_height,
            a_vect,
            self.enh_smag_fac,
            offset_provider={"Koff", KDim},
        )

        self.diff_multfac_n2w = (
            config.vertical_params.init_nabla2_factor_in_upper_damping_zone(
                k_size=config.grid.get_k_size()
            )
        )

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
        # Oa logging
        # 0b call timer start
        # 0c. apply dtime to enh_smag_factor
        # TODO does not work because the self.enh_smag_fact is a field: do this where the factor is used, ie inside stencil

        timestep_scaled_smagorinski_factor = self.enh_smag_fac * dtime

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
