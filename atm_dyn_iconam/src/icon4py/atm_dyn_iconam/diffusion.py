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
import sys
from collections import namedtuple
from typing import Final

import numpy as np
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, broadcast, maximum, minimum
from functional.iterator.embedded import np_as_located_field
from functional.program_processors.runners import gtfn_cpu

from icon4py.atm_dyn_iconam.constants import CPD, GAS_CONSTANT_DRY_AIR
from icon4py.atm_dyn_iconam.diagnostic import DiagnosticState
from icon4py.atm_dyn_iconam.fused_mo_nh_diffusion_stencil_02_03 import (
     _fused_mo_nh_diffusion_stencil_02_03,
)
from icon4py.atm_dyn_iconam.fused_mo_nh_diffusion_stencil_04_05_06 import (
    _fused_mo_nh_diffusion_stencil_04_05_06,
)
from icon4py.atm_dyn_iconam.fused_mo_nh_diffusion_stencil_07_08_09_10 import (
    _fused_mo_nh_diffusion_stencil_07_08_09_10,
)
from icon4py.atm_dyn_iconam.fused_mo_nh_diffusion_stencil_11_12 import (
    fused_mo_nh_diffusion_stencil_11_12,
)
from icon4py.atm_dyn_iconam.fused_mo_nh_diffusion_stencil_13_14 import (
    _fused_mo_nh_diffusion_stencil_13_14,
)
from icon4py.atm_dyn_iconam.horizontal import (
    HorizontalMarkerIndex,
    HorizontalMeshConfig,
)
from icon4py.atm_dyn_iconam.icon_grid import (
    IconGrid,
    MeshConfig,
    VerticalModelParams,
)
from icon4py.atm_dyn_iconam.interpolation_state import InterpolationState
from icon4py.atm_dyn_iconam.metric_state import MetricState
from icon4py.atm_dyn_iconam.mo_intp_rbf_rbf_vec_interpol_vertex import (
    _mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_01 import (
    _mo_nh_diffusion_stencil_01,
)
from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_16 import (
    _mo_nh_diffusion_stencil_16,
)
from icon4py.atm_dyn_iconam.prognostic import PrognosticState
from icon4py.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    CellDim,
    ECVDim,
    EdgeDim,
    KDim,
    Koff,
    VertexDim,
)


DiffusionTupleVT = namedtuple("DiffusionParamVT", "v t")
VectorTuple = namedtuple("CartesianVectorTuple", "x y")


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
def _scale_k(field: Field[[KDim], float], factor: float) -> Field[[KDim], float]:
    return field * factor


@program
def scale_k(
    field: Field[[KDim], float], factor: float, scaled_field: Field[[KDim], float]
):
    _scale_k(field, factor, out=scaled_field)


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
    diff_multfac_smag = _scale_k(enh_smag_fac, dtime)
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

def init_nabla2_factor_in_upper_damping_zone(
    k_size: int,
    nrdmax:int,
    nshift: int,
    physical_heights: np.ndarray
) -> Field[[KDim], float]:
    """
    calculated diff_multfac_n2w

    numpy version gt4py does not allow non-constant indexing into fields

    Args
        k_size: number of vertical levels
        nrdmax: index of the level where rayleigh dampint starts
        nshift:
        physcial_heights: vector of physical heights [m] of the height levels
    """
    buffer = np.zeros(k_size)
    buffer[2: nrdmax + 1] = (
        1.0
        / 12.0
        * ((
            physical_heights[2 + nshift: nrdmax + 1 + nshift]
            - physical_heights[nshift + nrdmax + 1]
        )
        / (physical_heights[2] - physical_heights[nshift + nrdmax + 1]))**4
    )
    return np_as_located_field(KDim)(buffer)


class DiffusionConfig:
    """contains necessary parameter to configure a diffusion run.

    - encapsulates namelist parameters and derived parameters (for now)

    currently we use the MCH r04b09_dsl experiment as constants here. These should
    be read from config and the default from mo_diffusion_nml.f90 set as defaults.

    TODO: [ml] read from config
    TODO: [ml] handle dependencies on other namelists (see below...)
    """
    def __init__(self,
                 grid: IconGrid,
                 vertical_params: VerticalModelParams,
                 diffusion_type:int = 5,
                 apply_to_horizontal_wind: bool = True,
                 apply_to_vertical_wind: bool = True,
                 apply_to_temperature: bool = True,
                 reconstruction_type_smag: int = 1,
                 compute_3d_smag_coeff:bool = False,
                 temperature_discretization: int=2,
                 horizontal_efdt_ratio: float = 24.0,
                 smag_scaling_factor: float = 0.025
                 ):
        # TODO [ml]: move external stuff out: grid related stuff, other than diffusion namelists (see below
        self.grid = grid
        self.vertical_params = vertical_params
        # from namelist diffusion_nml
        self.diffusion_type = diffusion_type  # hdiff_order ! order of nabla operator for diffusion
        self.lhdiff_vn = apply_to_horizontal_wind  # ! diffusion on the horizontal wind field
        self.lhdiff_temp = apply_to_temperature  # ! diffusion on the temperature field
        self.lhdiff_w = apply_to_vertical_wind  # ! diffusion on the vertical wind field
        self.lhdiff_rcf = True  # namelist, remove if always true
        self.itype_vn_diffu = (
            reconstruction_type_smag  # ! reconstruction method used for Smagorinsky diffusion
        )
        self.l_smag_d = compute_3d_smag_coeff  # namelist lsmag_d,  if `true`, compute 3D Smagorinsky diffusion coefficient.

        self.itype_t_diffu = temperature_discretization  # ! discretization of temperature diffusion
        self.hdiff_efdt_ratio = horizontal_efdt_ratio  # ! ratio of e-folding time to time step
        self.hdiff_smag_fac = smag_scaling_factor  # ! scaling factor for Smagorinsky diffusion

        # from other namelists
        # from parent namelist nonhydrostatic_nml
        self.l_zdiffu_t = True  # ! l_zdiffu_t: specifies computation of Smagorinsky temperature diffusion
        self.ndyn_substeps = 5

        # namelist gridref_nml
        # denom_diffu_v = 150   ! denominator for lateral boundary diffusion of velocity
        self.lateral_boundary_denominator = DiffusionTupleVT(v=200.0, t=135.0)

        # namelist grid_nml
        self.l_limited_area = True

        # name list: interpol_nml
        self.nudge_max_coeff = 0.075

    def substep_as_float(self):
        return float(self.ndyn_substeps)


class DiffusionParams:
    """Calculates derived quantities depending on the diffusion config."""

    def __init__(self, config: DiffusionConfig):
        # TODO [ml] logging for case KX == 0
        # TODO [ml] generrally calculation for x_dom (jg) > 2..n_dom, why is jg special
        self.boundary_diffusion_start_index_edges = (
            5  # mo_nh_diffusion.start_bdydiff_e - 1 = 5 -1
        )

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
                print("not implemented")
                smagorinski_factor = None
                smagorinski_height = None
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


def mo_nh_diffusion_stencil_15_numpy(
    mask_hdiff: Field[[CellDim, KDim], int],
    zd_vertidx: Field[[CellDim, C2E2CDim, KDim], int],
    zd_diffcoef: Field[[CellDim, KDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    vcoef: Field[[C2E2CDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    z_temp: Field[[CellDim, KDim], float],
    domain,
    offset_provider,
):
    pass


class Diffusion:
    """Class that configures diffusion and does one diffusion step."""

    def __init__(
        self,
        config: DiffusionConfig,
        params: DiffusionParams,
        vct_a: Field[[KDim], float],
    ):
        """
        Initialize Diffusion granule.

        TODO [ml]: initial run: linit = .TRUE.:  smag_offset and diff_multfac_vn are defined
        differently.
        """
        self.params = params
        self.config = config
        self.grid = config.grid
        self.rd_o_cvd: float = GAS_CONSTANT_DRY_AIR / (CPD - GAS_CONSTANT_DRY_AIR)
        self.nudgezone_diff = 0.04 / (config.nudge_max_coeff + sys.float_info.epsilon)
        self.bdy_diff = 0.015 / (config.nudge_max_coeff + sys.float_info.epsilon)
        self.fac_bdydiff_v = (
            math.sqrt(config.substep_as_float()) / config.lateral_boundary_denominator.v
            if config.lhdiff_rcf
            else 1.0 / config.lateral_boundary_denominator.v
        )
        self.thresh_tdiff = (
            -5.0
        )  # threshold temperature deviation from neighboring grid points hat activates extra diffusion against runaway cooling

        # different for init call: smag_offset = 0
        self.smag_offset: float = 0.25 * params.K4 * config.substep_as_float()
        self.diff_multfac_w: float = min(
            1.0 / 48.0, params.K4W * config.substep_as_float()
        )

        # different for initial run!, through diff_multfac_vn
        self.diff_multfac_vn = np_as_located_field(KDim)(
            np.zeros(config.grid.k_levels())
        )
        self.smag_limit = np_as_located_field(KDim)(np.zeros(config.grid.k_levels()))

        init_diffusion_local_fields(
            params.K4,
            config.substep_as_float(),
            self.diff_multfac_vn,
            self.smag_limit,
            offset_provider={},
        )

        self.enh_smag_fac = np_as_located_field(KDim)(
            np.zeros(config.grid.k_levels(), float)
        )
        enhanced_smagorinski_factor(
            *params.smagorinski_factor,
            *params.smagorinski_height,
            vct_a,
            self.enh_smag_fac,
            offset_provider={"Koff": KDim},
        )

        self.diff_multfac_n2w = init_nabla2_factor_in_upper_damping_zone(
                k_size=config.grid.k_levels(), nshift = 0, physical_heights=np.asarray(vct_a),
            nrdmax=self.config.vertical_params.index_of_damping_height

        )
        self.diff_multfac_smag = np_as_located_field(KDim)(
            np.zeros(config.grid.k_levels())
        )
        shape_vk = (config.grid.num_vertices(), config.grid.k_levels())
        shape_ck = (config.grid.num_cells(), config.grid.k_levels())
        self.u_vert = np_as_located_field(VertexDim, KDim)(np.zeros(shape_vk, float))
        self.v_vert = np_as_located_field(VertexDim, KDim)(np.zeros(shape_vk, float))
        shape_ek = (config.grid.num_edges(), config.grid.k_levels())
        allocate_ek = np_as_located_field(EdgeDim, KDim)(np.zeros(shape_ek, float))
        self.kh_smag_e = allocate_ek
        self.kh_smag_ec = allocate_ek
        self.z_nabla2_e = allocate_ek
        self.z_temp = np_as_located_field(CellDim, KDim)(np.zeros(shape_ck, float))
        self.vertical_index = np_as_located_field(KDim)(
            np.arange(self.grid.k_levels() + 1)
        )
        self.horizontal_cell_index = np_as_located_field(CellDim)(
            np.arange((shape_ck[0]))
        )
        self.horizontal_edge_index = np_as_located_field(EdgeDim)(np.arange((shape_ek[0])))



    def run(
        self,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
        metric_state: MetricState,
        interpolation_state: InterpolationState,
        dtime: float,
        tangent_orientation: Field[[EdgeDim], float],
        inverse_primal_edge_lengths: Field[[EdgeDim], float],
        inverse_dual_edge_length: Field[[EdgeDim], float],
        inverse_vertical_vertex_lengths: Field[[EdgeDim], float],
        primal_normal_vert: VectorTuple[Field[[ECVDim], float], Field[[ECVDim], float]],
        dual_normal_vert: VectorTuple[Field[[ECVDim], float], Field[[ECVDim], float]],
        edge_areas: Field[[EdgeDim], float],
        cell_areas: Field[[CellDim], float],
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
        # 0c. dtime dependent stuff: enh_smag_factor, r_dtimensubsteps
        r_dtimensubsteps = (
            1.0 / dtime
            if self.config.lhdiff_rcf
            else 1.0 / (dtime * self.config.substep_as_float())
        )
        klevels = self.grid.k_levels()
        # TODO is this needed?
        set_zero_v_k(self.u_vert, offset_provider={})
        set_zero_v_k(self.v_vert, offset_provider={})

        # 1.  CALL rbf_vec_interpol_vertex

        _mo_intp_rbf_rbf_vec_interpol_vertex(
            prognostic_state.normal_wind,
            interpolation_state.rbf_coeff_1,
            interpolation_state.rbf_coeff_2,
            out=(self.u_vert, self.v_vert),
            domain={
                KDim: (0, klevels),
                VertexDim: self.grid.get_indices_from_to(
                    VertexDim,
                    HorizontalMarkerIndex.local_boundary(VertexDim) + 1,
                    HorizontalMarkerIndex.halo(VertexDim) - 1,
                ),
            },
            offset_provider={"V2E": self.grid.get_v2e_offset_provider()},

        )

        # 2.  HALO EXCHANGE -- CALL sync_patch_array_mult
        # 3.  mo_nh_diffusion_stencil_01, mo_nh_diffusion_stencil_02, mo_nh_diffusion_stencil_03
        # 0c. dtime dependent stuff: enh_smag_factor, ~~r_dtimensubsteps~~
        scale_k(self.enh_smag_fac, dtime, self.diff_multfac_smag, offset_provider={})

        _mo_nh_diffusion_stencil_01(
            self.diff_multfac_smag,
            tangent_orientation,
            inverse_primal_edge_lengths,
            inverse_vertical_vertex_lengths,
            self.u_vert,
            self.v_vert,
            primal_normal_vert[0],
            primal_normal_vert[1],
            dual_normal_vert[0],
            dual_normal_vert[1],
            prognostic_state.normal_wind,
            self.smag_limit,
            self.smag_offset,
            domain={
                KDim: (0, klevels),
                EdgeDim: (
                    self.grid.get_indices_from_to(
                        EdgeDim,
                        self.params.boundary_diffusion_start_index_edges,
                        HorizontalMarkerIndex.halo(EdgeDim) - 2,
                    )
                ),
            },
            out=(self.kh_smag_e, self.kh_smag_ec, self.z_nabla2_e),
            offset_provider={"E2C2V": self.grid.get_e2c2v_connectivity()},
            backend = gtfn_cpu.run_gtfn
         )

        _fused_mo_nh_diffusion_stencil_02_03(
            self.kh_smag_ec,
            prognostic_state.normal_wind,
            interpolation_state.e_bln_c_s,
            interpolation_state.geofac_div,
            self.diff_multfac_smag,
            metric_state.wgtfac_c,
            out=(
                diagnostic_state.div_ic,
                diagnostic_state.hdef_ic,
            ),
            domain={
                KDim: (0, klevels),
                CellDim: (
                    self.grid.get_indices_from_to(CellDim,
                        HorizontalMarkerIndex.nudging(CellDim),
                        HorizontalMarkerIndex.halo(CellDim)
                    )
                ),
            },
            offset_provider={"C2E": self.grid.get_c2e_connectivity()},
            backend=gtfn_cpu.run_gtfn
        )

        # 4.  IF (discr_vn > 1) THEN CALL sync_patch_array -> false for MCH

        # 5.  CALL rbf_vec_interpol_vertex_wp
        _mo_intp_rbf_rbf_vec_interpol_vertex(
            self.z_nabla2_e,
            interpolation_state.rbf_coeff_1,
            interpolation_state.rbf_coeff_2,
            out=(self.u_vert, self.v_vert),
            # domain={
            #     KDim: (0, klevels),
            #     VertexDim: self.grid.get_indices_from_to(
            #         VertexDim,
            #         HorizontalMarkerIndex.local_boundary(VertexDim) + 3,
            #         HorizontalMarkerIndex.halo(VertexDim),
            #     ),
            # },
            offset_provider={"V2E": self.grid.get_e2v_connectivity()},
        )
        # 6.  HALO EXCHANGE -- CALL sync_patch_array_mult

        # 7.  mo_nh_diffusion_stencil_04, mo_nh_diffusion_stencil_05
        # 7a. IF (l_limited_area .OR. jg > 1) mo_nh_diffusion_stencil_06

        start_2nd_nudge_line = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim)-1,
            HorizontalMarkerIndex.nudging(EdgeDim)-1)[0]
        _fused_mo_nh_diffusion_stencil_04_05_06(
            self.u_vert,
            self.v_vert,
            primal_normal_vert.x,
            primal_normal_vert.y,
            self.z_nabla2_e,
            inverse_vertical_vertex_lengths,
            inverse_primal_edge_lengths,
            edge_areas,
            self.kh_smag_e,
            self.diff_multfac_vn,
            interpolation_state.nudgecoeff_e,
            prognostic_state.normal_wind,
            self.horizontal_edge_index,
            self.nudgezone_diff,
            self.fac_bdydiff_v,
            start_2nd_nudge_line,
            out=prognostic_state.normal_wind,
            domain={
                KDim: (0, klevels),
                EdgeDim: self.grid.get_indices_from_to(
                    EdgeDim,
                    HorizontalMarkerIndex.nudging(EdgeDim) + 1,
                    HorizontalMarkerIndex.halo(EdgeDim),
                ),
            },
            offset_provider={"E2C2V": self.grid.get_e2c2v_connectivity()},
        )
        # 7b. mo_nh_diffusion_stencil_07, mo_nh_diffusion_stencil_08,
        #     mo_nh_diffusion_stencil_09, mo_nh_diffusion_stencil_10
        interior_start_index, halo_endindex = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.interior(CellDim),
            HorizontalMarkerIndex.halo(CellDim))

        _fused_mo_nh_diffusion_stencil_07_08_09_10(
            cell_areas,
            interpolation_state.geofac_n2s,
            interpolation_state.geofac_grg_x,
            interpolation_state.geofac_grg_y,
            prognostic_state.vertical_wind,
            prognostic_state.vertical_wind,
            diagnostic_state.dwdx,
            diagnostic_state.dwdy,
            self.diff_multfac_w,
            self.diff_multfac_n2w,
            self.vertical_index,
            self.horizontal_cell_index,
            self.config.vertical_params.index_of_damping_height,
            interior_start_index,
            halo_endindex,
            out=(prognostic_state.vertical_wind, diagnostic_state.dwdx, diagnostic_state.dwdy),
            domain={
                KDim: (0, klevels),
                CellDim: self.grid.get_indices_from_to(
                    CellDim,
                    # TODO: global mode is from NUDGING - 1
                    HorizontalMarkerIndex.nudging(CellDim),
                    HorizontalMarkerIndex.halo(CellDim) + 1,
                ),
            },
            offset_provider={"C2E2CO": self.grid.get_c2e2c0_connectivity()},
        )
        # 8.  HALO EXCHANGE: CALL sync_patch_array
        # 9.  mo_nh_diffusion_stencil_11, mo_nh_diffusion_stencil_12, mo_nh_diffusion_stencil_13,
        #     mo_nh_diffusion_stencil_14, mo_nh_diffusion_stencil_15, mo_nh_diffusion_stencil_16

        # TODO check: kh_smag_e is an out field, should  not be calculated in init?
        fused_mo_nh_diffusion_stencil_11_12(
            prognostic_state.theta_v,
            metric_state.theta_ref_mc,
            self.thresh_tdiff,
            out=self.kh_smag_e,
            domain={
                KDim: (klevels - 2, klevels),
                CellDim: self.grid.get_indices_from_to(
                    CellDim,
                    HorizontalMarkerIndex.nudging(CellDim) - 1,
                    HorizontalMarkerIndex.halo(CellDim) + 1,
                ),
                EdgeDim: self.grid.get_indices_from_to(
                    EdgeDim,
                    HorizontalMarkerIndex.nudging(EdgeDim) + 1,
                    HorizontalMarkerIndex.halo(EdgeDim),
                ),
            },
            offset_provider={
                "E2C": self.grid.get_e2c_connectivity(),
                "C2E2C": self.grid.get_c2e2c_connectivity(),
            },
        )

        _fused_mo_nh_diffusion_stencil_13_14(
            self.kh_smag_e,
            inverse_dual_edge_length,
            prognostic_state.theta_v,
            interpolation_state.geofac_div,
            out = self.z_temp,
            domain={
                KDim: (0, klevels),
                CellDim: self.grid.get_indices_from_to(
                    CellDim,
                    HorizontalMarkerIndex.nudging(CellDim),
                    HorizontalMarkerIndex.halo(CellDim),
                ),
                EdgeDim: self.grid.get_indices_from_to(
                    EdgeDim,
                    HorizontalMarkerIndex.nudging(EdgeDim),
                    HorizontalMarkerIndex.halo(EdgeDim) - 1,
                ),
            },
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "E2C": self.grid.get_e2c_connectivity(),
            },
        )

        mo_nh_diffusion_stencil_15_numpy(
            mask_hdiff=metric_state.mask_hdiff,
            zd_vertidx=metric_state.zd_vertidx,
            vcoef=metric_state.zd_diffcoef,
            zd_diffcoef=metric_state.zd_diffcoef,
            geofac_n2s=interpolation_state.geofac_n2s,
            theta_v=prognostic_state.theta_v,
            z_temp=self.z_temp,
            domain={
                KDim: (0, klevels),
                CellDim: self.grid.get_indices_from_to(
                    CellDim,
                    HorizontalMarkerIndex.nudging(CellDim),
                    HorizontalMarkerIndex.halo(CellDim),
                ),
            },
            offset_provider={},
        )

        _mo_nh_diffusion_stencil_16(
            self.z_temp,
            cell_areas,
            prognostic_state.theta_v,
            prognostic_state.exner_pressure,
            self.rd_o_cvd,
            out=(prognostic_state.theta_v, prognostic_state.exner_pressure),
            domain={
                KDim: (0, klevels),
                CellDim: self.grid.get_indices_from_to(
                    CellDim,
                    HorizontalMarkerIndex.nudging(CellDim),
                    HorizontalMarkerIndex.halo(CellDim),
                ),
            },
            offset_provider={},
        )
        # 10. HALO EXCHANGE sync_patch_array
