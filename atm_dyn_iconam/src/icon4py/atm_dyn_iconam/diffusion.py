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

from icon4py.atm_dyn_iconam.horizontal import HorizontalMeshConfig
from icon4py.atm_dyn_iconam.icon_mesh import MeshConfig, VerticalModelParams
from icon4py.atm_dyn_iconam.interpolation_state import InterpolationState
from icon4py.atm_dyn_iconam.mo_intp_rbf_rbf_vec_interpol_vertex import (
    mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_01 import (
    _mo_nh_diffusion_stencil_01,
)
from icon4py.atm_dyn_iconam.prognostic import PrognosticState
from icon4py.common.dimension import ECVDim, EdgeDim, KDim, Koff, VertexDim


DiffusionTupleVT = namedtuple("DiffusionParamVT", "v t")
CartesianVectorTuple = namedtuple("CartesianVectorTuple", "x y")


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


# TODO [ml] rename!
@field_operator
def _mo_nh_diffusion_stencil_01_scale_dtime(
    enh_smag_fac: Field[[KDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    inv_vert_vert_length: Field[[EdgeDim], float],
    u_vert: Field[[VertexDim, KDim], float],
    v_vert: Field[[VertexDim, KDim], float],
    primal_normal_vert_x: Field[[ECVDim], float],
    primal_normal_vert_y: Field[[ECVDim], float],
    dual_normal_vert_x: Field[[ECVDim], float],
    dual_normal_vert_y: Field[[ECVDim], float],
    vn: Field[[EdgeDim, KDim], float],
    smag_limit: Field[[KDim], float],
    smag_offset: float,
    dtime: float,
) -> tuple[
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
]:
    diff_multfac_smag = enh_smag_fac * dtime
    return _mo_nh_diffusion_stencil_01(
        diff_multfac_smag,
        tangent_orientation,
        inv_primal_edge_length,
        inv_vert_vert_length,
        u_vert,
        v_vert,
        primal_normal_vert_x,
        primal_normal_vert_y,
        dual_normal_vert_x,
        dual_normal_vert_y,
        vn,
        smag_limit,
        smag_offset,
    )


@program
def mo_nh_diffusion_stencil_01_scaled_dtime(
    enh_smag_fac: Field[[KDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    inv_vert_vert_length: Field[[EdgeDim], float],
    u_vert: Field[[VertexDim, KDim], float],
    v_vert: Field[[VertexDim, KDim], float],
    primal_normal_vert_x: Field[[ECVDim], float],
    primal_normal_vert_y: Field[[ECVDim], float],
    dual_normal_vert_x: Field[[ECVDim], float],
    dual_normal_vert_y: Field[[ECVDim], float],
    vn: Field[[EdgeDim, KDim], float],
    smag_limit: Field[[KDim], float],
    kh_smag_e: Field[[EdgeDim, KDim], float],
    kh_smag_ec: Field[[EdgeDim, KDim], float],
    z_nabla2_e: Field[[EdgeDim, KDim], float],
    smag_offset: float,
    dtime: float,
):
    _mo_nh_diffusion_stencil_01_scale_dtime(
        enh_smag_fac,
        tangent_orientation,
        inv_primal_edge_length,
        inv_vert_vert_length,
        u_vert,
        v_vert,
        primal_normal_vert_x,
        primal_normal_vert_y,
        dual_normal_vert_x,
        dual_normal_vert_y,
        vn,
        smag_limit,
        smag_offset,
        dtime,
        out=(kh_smag_e, kh_smag_ec, z_nabla2_e),
    )


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
def _set_zero_k() -> Field[[KDim], float]:
    return broadcast(0.0, (KDim,))


@field_operator
def _set_zero_v_k() -> Field[[VertexDim, KDim], float]:
    return broadcast(0.0, (VertexDim, KDim))


@program
def set_zero_v_k(field: Field[[VertexDim, KDim], float]):
    _set_zero_v_k(out=field)


class DiffusionConfig:
    """contains necessary parameter to configure a diffusion run.

    - encapsulates namelist parameters and derived parameters (for now)

    currently we use the MCH r04b09_dsl experiment as constants here. These should
    be read from config and the default from mo_diffusion_nml.f90 set as defaults.



    TODO: [ml] read from config
    TODO: [ml] handle dependencies on other namelists (below...)
    """

    @classmethod
    def create_with_defaults(cls):
        """
        Create DiffusionConfig.

        initialize with values from exp.mch_ch_r04b09_dsl namelist
        """
        horizontal = HorizontalMeshConfig(
            num_vertices=50000, num_cells=50000, num_edges=50000
        )
        gridConfig = MeshConfig(horizontalMesh=horizontal)
        verticalParams = VerticalModelParams(
            rayleigh_damping_height=12500, vct_a=np.zeros(gridConfig.get_num_k_levels())
        )
        return cls(grid_config=gridConfig, vertical_params=verticalParams)

    def __init__(self, grid_config: MeshConfig, vertical_params: VerticalModelParams):
        # TODO [ml]: external stuff grid related, p_patch,
        self.grid = grid_config
        self.vertical_params = vertical_params
        # from namelist diffusion_nml
        self.diffusion_type = 5  # hdiff_order ! order of nabla operator for diffusion
        self.lhdiff_vn = True  # ! diffusion on the horizontal wind field
        self.lhdiff_temp = True  # ! diffusion on the temperature field
        self.lhdiff_w = True  # ! diffusion on the vertical wind field
        self.lhdiff_rcf = True  # namelist, remove if always true
        self.itype_vn_diffu = (
            1  # ! reconstruction method used for Smagorinsky diffusion
        )
        self.itype_t_diffu = 2  # ! discretization of temperature diffusion
        self.hdiff_efdt_ratio = 24.0  # ! ratio of e-folding time to time step
        self.hdiff_smag_fac = 0.025  # ! scaling factor for Smagorinsky diffusion

        # from other namelists
        # from parent namelist nonhydrostatic_nml
        self.l_zdiffu_t = True  # ! l_zdiffu_t: specifies computation of Smagorinsky temperature diffusion
        self.ndyn_substeps = 5

        # namelist gridref_nml
        # denom_diffu_v = 150   ! denominator for lateral boundary diffusion of velocity
        self.lateral_boundary_denominator = DiffusionTupleVT(v=200.0, t=135.0)

        # namelist grid_nml -> TODO [ml] should go to grid config?
        self.l_limited_area = True

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
            np.zeros(config.grid.get_num_k_levels())
        )
        self.smag_limit = np_as_located_field(KDim)(
            np.zeros(config.grid.get_num_k_levels())
        )

        init_diffusion_local_fields(
            params.K4,
            config.substep_as_float(),
            self.diff_multfac_vn,
            self.smag_limit,
            offset_provider={},
        )

        self.enh_smag_fac = np_as_located_field(KDim)(
            np.zeros(config.grid.get_num_k_levels())
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
                k_size=config.grid.get_num_k_levels()
            )
        )
        self.u_vert = np_as_located_field(VertexDim, KDim)(
            np.zeros(config.grid.get_num_vertices(), config.grid.get_num_k_levels())
        )
        self.v_vert = np_as_located_field(VertexDim, KDim)(
            np.zeros(config.grid.get_num_vertices(), config.grid.get_num_k_levels())
        )
        allocate_ek = np_as_located_field(VertexDim, KDim)(
            np.zeros(config.grid.get_num_edges(), config.grid.get_num_k_levels())
        )
        self.kh_smag_e = allocate_ek
        self.kh_smag_ec = allocate_ek
        self.z_nabla2_e = allocate_ek

    def run(
        self,
        diagnostic_state,
        prognostic_state: PrognosticState,
        metric_state,
        interpolation_state: InterpolationState,
        dtime,
        tangent_orientation: Field[[EdgeDim], float],
        inverse_primal_edge_lengths: Field[[EdgeDim], float],
        inverse_vertical_vertex_lengths: Field[[EdgeDim], float],
        primal_normal_vert: CartesianVectorTuple[
            Field[[ECVDim], float], Field[[ECVDim], float]
        ],
        dual_normal_vert: CartesianVectorTuple[
            Field[[ECVDim], float], Field[[ECVDim], float]
        ],
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
        # ~~0c. apply dtime to enh_smag_factor~~ done inside stencil

        # TODO is this needed?
        set_zero_v_k(self.u_vert)
        set_zero_v_k(self.v_vert)

        # 1.  CALL rbf_vec_interpol_vertex

        mo_intp_rbf_rbf_vec_interpol_vertex(
            prognostic_state.normal_wind,
            interpolation_state.rbf_coeff_1,
            interpolation_state.rbf_coeff_2,
            self.u_vert,
            self.v_vert,
        )
        # 2.  HALO EXCHANGE -- CALL sync_patch_array_mult
        # 3.  mo_nh_diffusion_stencil_01, mo_nh_diffusion_stencil_02, mo_nh_diffusion_stencil_03

        # tangent_orientation = p_patch % edges % tangent_orientation(:, 1)
        # inv_primal_edge_length=p_patch%edges%inv_primal_edge_length(:,1)
        # inv_vert_vert_length = p_patch % edges % inv_vert_vert_length(:, 1),
        # primal_normal_vert_x = p_patch % edges % primal_normal_vert_x(:,:, 1)
        # primal_normal_vert_y = p_patch % edges % primal_normal_vert_y(:,:, 1)
        # dual_normal_vert_x=p_patch%edges%dual_normal_vert_x(:,:,1)
        # dual_normal_vert_x = p_patch % edges % dual_normal_vert_x(:,:, 1)

        mo_nh_diffusion_stencil_01_scaled_dtime(
            self.enh_smag_fac,
            tangent_orientation,
            inverse_primal_edge_lengths,
            inverse_vertical_vertex_lengths,
            self.u_vert,
            self.v_vert,
            primal_normal_vert.x,
            primal_normal_vert.y,
            dual_normal_vert.x,
            dual_normal_vert.y,
            prognostic_state.normal_wind,
            self.smag_limit,
            self.kh_smag_e,
            self.kh_smag_ec,
            self.z_nabla2_e,
            self.smag_offset,
            dtime,
        )

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
