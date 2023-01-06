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

# flake8: noqa F841, E800
import math
import sys
from collections import namedtuple
from typing import Final, Tuple

import numpy as np
from functional.common import Dimension
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, broadcast, int32, maximum, minimum
from functional.iterator.embedded import (
    StridedNeighborOffsetProvider,
    np_as_located_field,
)

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
    _fused_mo_nh_diffusion_stencil_11_12,
)
from icon4py.atm_dyn_iconam.fused_mo_nh_diffusion_stencil_13_14 import (
    _fused_mo_nh_diffusion_stencil_13_14,
)
from icon4py.atm_dyn_iconam.mo_intp_rbf_rbf_vec_interpol_vertex import (
    mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_01 import (
    _mo_nh_diffusion_stencil_01,
)
from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_16 import (
    _mo_nh_diffusion_stencil_16,
)
from icon4py.common.constants import CPD, GAS_CONSTANT_DRY_AIR
from icon4py.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CellDim,
    ECVDim,
    EdgeDim,
    KDim,
    Koff,
    V2EDim,
    VertexDim,
)
from icon4py.diffusion.diagnostic import DiagnosticState
from icon4py.diffusion.diffusion_program import diffusion_run
from icon4py.diffusion.horizontal import HorizontalMarkerIndex
from icon4py.diffusion.icon_grid import IconGrid, VerticalModelParams
from icon4py.diffusion.interpolation_state import InterpolationState
from icon4py.diffusion.metric_state import MetricState
from icon4py.diffusion.prognostic import PrognosticState
from icon4py.diffusion.utils import scale_k, set_zero_v_k, zero_field


VectorTuple = namedtuple("VectorTuple", "x y")


@field_operator
def _setup_smag_limit(diff_multfac_vn: Field[[KDim], float]) -> Field[[KDim], float]:
    return 0.125 - 4.0 * diff_multfac_vn


@field_operator
def _setup_runtime_diff_multfac_vn(
    k4: float, dyn_substeps: float
) -> Field[[KDim], float]:
    con = 1.0 / 128.0
    dyn = k4 * dyn_substeps / 3.0
    return broadcast(minimum(con, dyn), (KDim,))


# @field_operator(backend=run_gtfn)
@field_operator
def _setup_initial_diff_multfac_vn(
    k4: float, hdiff_efdt_ratio: float
) -> Field[[KDim], float]:
    return broadcast(k4 / 3.0 * hdiff_efdt_ratio, (KDim,))


@field_operator
def _setup_fields_for_initial_step(k4: float, hdiff_efdt_ratio: float):
    diff_multfac_vn = _setup_initial_diff_multfac_vn(k4, hdiff_efdt_ratio)
    smag_limit = _setup_smag_limit(diff_multfac_vn)
    return diff_multfac_vn, smag_limit


@field_operator
def _en_smag_fac_for_zero_nshift(
    vect_a: Field[[KDim], float],
    hdiff_smag_fac: float,
    hdiff_smag_fac2: float,
    hdiff_smag_fac3: float,
    hdiff_smag_fac4: float,
    hdiff_smag_z: float,
    hdiff_smag_z2: float,
    hdiff_smag_z3: float,
    hdiff_smag_z4: float,
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


@field_operator
def _init_diffusion_local_fields_for_regular_timestemp(
    k4: float,
    dyn_substeps: float,
    hdiff_smag_fac: float,
    hdiff_smag_fac2: float,
    hdiff_smag_fac3: float,
    hdiff_smag_fac4: float,
    hdiff_smag_z: float,
    hdiff_smag_z2: float,
    hdiff_smag_z3: float,
    hdiff_smag_z4: float,
    vect_a: Field[[KDim], float],
) -> tuple[Field[[KDim], float], Field[[KDim], float], Field[[KDim], float]]:
    diff_multfac_vn = _setup_runtime_diff_multfac_vn(k4, dyn_substeps)
    smag_limit = _setup_smag_limit(diff_multfac_vn)
    enh_smag_fac = _en_smag_fac_for_zero_nshift(
        vect_a,
        hdiff_smag_fac,
        hdiff_smag_fac2,
        hdiff_smag_fac3,
        hdiff_smag_fac4,
        hdiff_smag_z,
        hdiff_smag_z2,
        hdiff_smag_z3,
        hdiff_smag_z4,
    )
    return (
        diff_multfac_vn,
        smag_limit,
        enh_smag_fac,
    )


@program
def init_diffusion_local_fields_for_regular_timestep(
    k4: float,
    dyn_substeps: float,
    hdiff_smag_fac: float,
    hdiff_smag_fac2: float,
    hdiff_smag_fac3: float,
    hdiff_smag_fac4: float,
    hdiff_smag_z: float,
    hdiff_smag_z2: float,
    hdiff_smag_z3: float,
    hdiff_smag_z4: float,
    vect_a: Field[[KDim], float],
    diff_multfac_vn: Field[[KDim], float],
    smag_limit: Field[[KDim], float],
    enh_smag_fac: Field[[KDim], float],
):
    _init_diffusion_local_fields_for_regular_timestemp(
        k4,
        dyn_substeps,
        hdiff_smag_fac,
        hdiff_smag_fac2,
        hdiff_smag_fac3,
        hdiff_smag_fac4,
        hdiff_smag_z,
        hdiff_smag_z2,
        hdiff_smag_z3,
        hdiff_smag_z4,
        vect_a,
        out=(
            diff_multfac_vn,
            smag_limit,
            enh_smag_fac,
        ),
    )


def init_nabla2_factor_in_upper_damping_zone(
    k_size: int, nrdmax: int, nshift: int, physical_heights: np.ndarray
) -> Field[[KDim], float]:
    """
    Calculate diff_multfac_n2w.

    numpy version, since gt4py does not allow non-constant indexing into fields
    TODO: [ml] fix this once IndexedFields are implemented

    Args
        k_size: number of vertical levels
        nrdmax: index of the level where rayleigh dampint starts
        nshift:
        physcial_heights: vector of physical heights [m] of the height levels
    """
    buffer = np.zeros(k_size)
    buffer[1 : nrdmax + 1] = (
        1.0
        / 12.0
        * (
            (
                physical_heights[1 + nshift : nrdmax + 1 + nshift]
                - physical_heights[nshift + nrdmax + 1]
            )
            / (physical_heights[1] - physical_heights[nshift + nrdmax + 1])
        )
        ** 4
    )
    return np_as_located_field(KDim)(buffer)


class DiffusionConfig:
    """
    Contains necessary parameter to configure a diffusion run.

    Encapsulates namelist parameters and derived parameters.
    Values should be read from configuration.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    TODO: [ml] to be read from config
    TODO: [ml] handle dependencies on other namelists (see below...)
    """

    def __init__(
        self,
        grid: IconGrid,
        vertical_params: VerticalModelParams,
        diffusion_type: int = 5,
        hdiff_w=True,
        hdiff_vn=True,
        hdiff_temp=True,
        type_vn_diffu: int = 1,
        smag_3d: bool = False,
        type_t_diffu: int = 2,
        hdiff_efdt_ratio: float = 36.0,
        hdiff_w_efdt_ratio: float = 15.0,
        smagorinski_scaling_factor: float = 0.015,
        n_substeps: int = 5,
        zdiffu_t: bool = True,
        hdiff_rcf: bool = True,
        velocity_boundary_diffusion_denom: float = 200.0,
        temperature_boundary_diffusion_denom: float = 135.0,
        max_nudging_coeff: float = 0.02,
        nudging_decay_rate: float = 2.0,
    ):
        self.grid = grid
        self.vertical_params = vertical_params

        # parameters from namelist diffusion_nml
        self.diffusion_type: int = diffusion_type
        """
        Order of nabla operator for diffusion.

        Called `hdiff_order` in mo_diffusion_nml.f90.
        Possible values are:
        - -1: no diffusion
        - 2: 2nd order linear diffusion on all vertical levels
        - 3: Smagorinsky diffusion without background diffusion
        - 4: 4th order linear diffusion on all vertical levels
        - 5: Smagorinsky diffusion with fourth-order background diffusion

        We only support type 5.
        TODO: [ml] use enum
        """

        self.apply_to_vertical_wind: bool = hdiff_w
        """Ff True, apply diffusion on the vertical wind field

        Called `lhdiff_w` in in mo_diffusion_nml.f90
        """

        self.apply_to_horizontal_wind = hdiff_vn
        """
        If True apply diffusion on the horizontal wind field, is ONLY used in mo_nh_stepping.f90

        Called `lhdiff_vn` in in mo_diffusion_nml.f90
        """

        self.apply_to_temperature: bool = hdiff_temp
        """
        If True, apply horizontal diffusion to temperature field

        Called `lhdiff_temp` in in mo_diffusion_nml.f90
        """

        self.compute_3d_smag_coeff: bool = smag_3d
        """
        If True, compute 3D Smagorinsky diffusion coefficient.

        Called `lsmag_3d` in in mo_diffusion_nml.f90
        """

        self.type_vn_diffu: int = type_vn_diffu
        """
        Options for discretizing the Smagorinsky momentum diffusion.

        Called `itype_vn_diffu` in in mo_diffusion_nml.f90
        """

        self.type_t_diffu = type_t_diffu
        """
        Options for discretizing the Smagorinsky temperature diffusion.

        Called `itype_t_diffu` in in mo_diffusion_nml.f90
        """

        self.hdiff_efdt_ratio: float = hdiff_efdt_ratio
        """
        Ratio of e-folding time to (2*)time step.

        Called `hdiff_efdt_ratio` in in mo_diffusion_nml.f90.
        """

        self.hdiff_w_efdt_ratio: float = hdiff_w_efdt_ratio
        """
        Ratio of e-folding time to time step for w diffusion (NH only).

        Called `hdiff_w_efdt_ratio` in in mo_diffusion_nml.f90.
        """

        self.smagorinski_scaling_factor: float = smagorinski_scaling_factor
        """
        Scaling factor for Smagorinsky diffusion at height hdiff_smag_z and below.

        Called `hdiff_smag_fac` in in mo_diffusion_nml.f90.
        """

        self.apply_zdiffusion_t: bool = zdiffu_t
        """
        If True, apply truly horizontal temperature diffusion over steep slopes.

        From parent namelist mo_nonhydrostatic_nml.f90, but is only used in diffusion,
        and in mo_vertical_grid.f90>prepare_zdiffu.
        Called 'l_zdiffu_t' in mo_nonhydrostatic_nml.f90.
        """

        # from other namelists

        # from parent namelist mo_nonhydrostatic_nml
        self.ndyn_substeps: int = n_substeps
        """
        Number of dynamics substeps per fast-physics step.

        Called 'ndyn_substeps' in mo_nonhydrostatic_nml.f90.
        """

        self.lhdiff_rcf: bool = hdiff_rcf
        """
        If True, compute horizontal diffusion only at the large time step.

        Called 'lhdiff_rcf' in mo_nonhydrostatic_nml.f90.
        """

        # namelist mo_gridref_nml.f90
        self.temperature_boundary_diffusion_denominator: float = (
            temperature_boundary_diffusion_denom
        )
        """
        Denominator for temperature boundary diffusion.

        Called 'denom_diffu_t' in mo_gridref_nml.f90.
        """

        self.velocity_boundary_diffusion_denominator: float = (
            velocity_boundary_diffusion_denom
        )
        """
        Denominator for velocity boundary diffusion.

        Called 'denom_diffu_v' in mo_gridref_nml.f90.
        """

        # namelist: mo_interpol_nml.f90
        self.nudge_max_coeff: float = max_nudging_coeff
        """
        Parameter describing the lateral boundary nudging in limited area mode.

        Maximal value of the nudging coefficients used cell row bordering the boundary
        interpolation zone, from there nudging coefficients decay exponentially with
        `nudge_efold_width` in units of cell rows.

        Called `nudge_max_coeff` in mo_interpol_nml.f90
        """

        self.nudge_efold_width: float = nudging_decay_rate
        """
        Exponential decay rate (in units of cell rows) of the lateral boundary nudging coefficients.

        Called `nudge_efold_width` in mo_interpol_nml.f90
        """

        self._validate()

    def _validate(self):
        """
        Apply consistency checks and validation on configuration parameters.
        """
        if self.diffusion_type != 5:
            raise NotImplementedError(
                "Only diffusion type 5 = `Smagorinsky diffusion with fourth-order background diffusion` is implemented"
            )

        if not self.grid.limited_area:
            raise NotImplementedError("Only limited area mode is implemented")

        if self.diffusion_type < 0:
            self.apply_to_temperature = False
            self.apply_to_horizontal_wind = False
            self.apply_to_vertical_wind = False
        else:
            self.apply_to_temperature = True
            self.apply_to_horizontal_wind = True

        if not self.apply_zdiffusion_t:
            raise NotImplementedError(
                "zdiffu_t = False is not implemented (leaves out stencil_15)"
            )

    def substep_as_float(self):
        return float(self.ndyn_substeps)


class DiffusionParams:
    """Calculates derived quantities depending on the diffusion config."""

    def __init__(self, config: DiffusionConfig):
        self.boundary_diffusion_start_index_edges = int32(
            5  # mo_nh_diffusion.start_bdydiff_e - 1 = 5 -1
        )

        self.K2: Final[float] = (
            1.0 / (config.hdiff_efdt_ratio * 8.0)
            if config.hdiff_efdt_ratio > 0.0
            else 0.0
        )
        self.K4: Final[float] = self.K2 / 8.0
        self.K6: Final[float] = self.K2 / 64.0

        self.K4W: Final[float] = (
            1.0 / (config.hdiff_w_efdt_ratio * 36.0)
            if config.hdiff_w_efdt_ratio > 0
            else 0.0
        )

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
                ) = self._diffusion_type_5_smagorinski_factor(config)
            case 4:
                # according to mo_nh_diffusion.f90 this isn't used anywhere the factor is only
                # used for diffusion_type (3,5) but the defaults are only defined for iequations=3
                smagorinski_factor = (
                    config.smagorinski_scaling_factor
                    if config.smagorinski_scaling_factor
                    else 0.15,
                )
                smagorinski_height = None
            case _:
                raise NotImplementedError("Only implemented for diffusion type 4 and 5")
                smagorinski_factor = None
                smagorinski_height = None
                pass
        return smagorinski_factor, smagorinski_height

    @staticmethod
    def _diffusion_type_5_smagorinski_factor(config: DiffusionConfig):
        """
        Initialize smagorinski factors used in diffusion type 5.

        The calculation and magic numbers are taken from mo_diffusion_nml.f90
        """
        magic_sqrt = math.sqrt(1600.0 * (1600 + 50000.0))
        magic_fac2_value = 2e-6 * (1600.0 + 25000.0 + magic_sqrt)
        magic_z2 = 1600.0 + 50000.0 + magic_sqrt
        factor = (config.smagorinski_scaling_factor, magic_fac2_value, 0.0, 1.0)
        heights = (32500.0, magic_z2, 50000.0, 90000.0)
        return factor, heights


def mo_nh_diffusion_stencil_15_numpy(
    c2e2c,
    mask_hdiff: Field[[CellDim, KDim], int],
    zd_vertidx: Field[[C2E2CDim, KDim], int],
    zd_diffcoef: Field[[CellDim, KDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    vcoef: Field[[C2E2CDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    z_temp: Field[[CellDim, KDim], float],
):
    geofac_n2s = np.asarray(geofac_n2s)
    geofac_n2s_nbh = geofac_n2s[:, 1:]
    geofac_n2s_c = geofac_n2s[:, 0]
    mask_hdiff = np.asarray(mask_hdiff)
    zd_vertidx = np.asarray(zd_vertidx)
    zd_diffcoef = np.asarray(zd_diffcoef)
    vcoef = np.asarray(vcoef)
    second = (1 - vcoef) * theta_v[c2e2c][zd_vertidx + 1]

    z_temp = np.asarray(z_temp)
    vertidx_ = vcoef * theta_v[c2e2c][zd_vertidx]
    first = geofac_n2s_nbh * vertidx_
    summed = np.sum(first + second, axis=0)

    z_temp = np.where(
        mask_hdiff, z_temp + zd_diffcoef * (theta_v * geofac_n2s_c + summed), z_temp
    )
    return z_temp


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

        calculates all local fields that are used in diffusion within the time loop
        """
        self.params: DiffusionParams = params
        self.config: DiffusionConfig = config
        self.grid = config.grid
        self.rd_o_cvd: float = GAS_CONSTANT_DRY_AIR / (CPD - GAS_CONSTANT_DRY_AIR)
        self.nudgezone_diff: float = 0.04 / (
            config.nudge_max_coeff + sys.float_info.epsilon
        )
        self.bdy_diff: float = 0.015 / (config.nudge_max_coeff + sys.float_info.epsilon)
        self.fac_bdydiff_v: float = (
            math.sqrt(config.substep_as_float())
            / config.velocity_boundary_diffusion_denominator
            if config.lhdiff_rcf
            else 1.0 / config.velocity_boundary_diffusion_denominator
        )
        self.thresh_tdiff: float = (
            -5.0
        )  # threshold temperature deviation from neighboring grid points hat activates extra diffusion against runaway cooling

        self._smag_offset: float = 0.25 * params.K4 * config.substep_as_float()
        self.diff_multfac_w: float = min(
            1.0 / 48.0, params.K4W * config.substep_as_float()
        )

        self._allocate_local_fields()

        init_diffusion_local_fields_for_regular_timestep(
            params.K4,
            config.substep_as_float(),
            *params.smagorinski_factor,
            *params.smagorinski_height,
            vct_a,
            self._diff_multfac_vn,
            self._smag_limit,
            self.enh_smag_fac,
            offset_provider={"Koff": KDim},
        )

        self.diff_multfac_n2w = init_nabla2_factor_in_upper_damping_zone(
            k_size=config.grid.n_lev(),
            nshift=0,
            physical_heights=np.asarray(vct_a),
            nrdmax=self.config.vertical_params.index_of_damping_height,
        )

    def _allocate_local_fields(self):
        def _allocate(*dims: Dimension):
            return zero_field(self.grid, *dims)

        def _index_field(dim: Dimension, size=None):
            size = size if size else self.grid.size[dim]
            return np_as_located_field(dim)(np.arange(size, dtype=int32))

        self._diff_multfac_vn = _allocate(KDim)

        self._smag_limit = _allocate(KDim)
        self.enh_smag_fac = _allocate(KDim)
        self.u_vert = _allocate(VertexDim, KDim)
        self.v_vert = _allocate(VertexDim, KDim)
        self.kh_smag_e = _allocate(EdgeDim, KDim)
        self.kh_smag_ec = _allocate(EdgeDim, KDim)
        self.z_nabla2_e = _allocate(EdgeDim, KDim)
        self.z_temp = _allocate(CellDim, KDim)
        self.diff_multfac_smag = _allocate(KDim)
        self.vertical_index = _index_field(KDim, self.grid.n_lev() + 1)
        self.horizontal_cell_index = _index_field(CellDim)
        self.horizontal_edge_index = _index_field(EdgeDim)

    def initial_step(
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
        Calculate initial diffusion step.

        In ICON at the start of the simulation diffusion is run with a parameter linit = True:

        'For real-data runs, perform an extra diffusion call before the first time
        step because no other filtering of the interpolated velocity field is done'

        This run uses special values for diff_multfac_vn, smag_limit and smag_offset

        """
        diff_multfac_vn = zero_field(self.grid, KDim)
        smag_limit = zero_field(self.grid, KDim)

        _setup_fields_for_initial_step(
            self.params.K4,
            self.config.hdiff_efdt_ratio,
            out=(diff_multfac_vn, smag_limit),
        )
        self._do_diffusion_step(
            diagnostic_state,
            prognostic_state,
            metric_state,
            interpolation_state,
            dtime,
            tangent_orientation,
            inverse_primal_edge_lengths,
            inverse_dual_edge_length,
            inverse_vertical_vertex_lengths,
            primal_normal_vert,
            dual_normal_vert,
            edge_areas,
            cell_areas,
            diff_multfac_vn,
            smag_limit,
            0.0,
        )

    def time_step(
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
        Do one diffusion step within regular time loop.

        runs a diffusion step for the parameter linit=False, within regular time loop.
        """
        # self._do_diffusion_step(
        #     diagnostic_state,
        #     prognostic_state,
        #     metric_state,
        #     interpolation_state,
        #     dtime,
        #     tangent_orientation,
        #     inverse_primal_edge_lengths,
        #     inverse_dual_edge_length,
        #     inverse_vertical_vertex_lengths,
        #     primal_normal_vert,
        #     dual_normal_vert,
        #     edge_areas,
        #     cell_areas,
        #     self._diff_multfac_vn,
        #     self._smag_limit,
        #     self._smag_offset,
        # )

        cell_startindex_nudging, cell_endindex_local = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.nudging(CellDim),
            HorizontalMarkerIndex.local(CellDim),
        )

        (
            cell_startindex_interior,
            cell_endindex_local_plus1,
        ) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.interior(CellDim),
            HorizontalMarkerIndex.local(CellDim) + 1,
        )

        (
            edge_startindex_nudging_plus1,
            edge_endindex_local,
        ) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim) + 1,
            HorizontalMarkerIndex.local(EdgeDim),
        )

        (
            edge_startindex_nudging_minus1,
            edge_endindex_local_minus2,
        ) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim) - 1,
            HorizontalMarkerIndex.local(EdgeDim) - 2,
        )

        (
            vertex_startindex_lb_plus3,
            vertex_endindex_local,
        ) = self.grid.get_indices_from_to(
            VertexDim,
            HorizontalMarkerIndex.local_boundary(VertexDim) + 3,
            HorizontalMarkerIndex.local(VertexDim),
        )

        (
            vertex_startindex_lb_plus1,
            vertex_endindex_local_minus1,
        ) = self.grid.get_indices_from_to(
            VertexDim,
            HorizontalMarkerIndex.local_boundary(VertexDim) + 1,
            HorizontalMarkerIndex.local(VertexDim) - 1,
        )

        diffusion_run(
            diagnostic_state.hdef_ic,
            diagnostic_state.div_ic,
            diagnostic_state.dwdx,
            diagnostic_state.dwdy,
            prognostic_state.vertical_wind,
            prognostic_state.normal_wind,
            prognostic_state.exner_pressure,
            prognostic_state.theta_v,
            metric_state.theta_ref_mc,
            metric_state.wgtfac_c,
            metric_state.mask_hdiff,
            metric_state.zd_vertidx,
            metric_state.zd_diffcoef,
            metric_state.zd_intcoef,
            interpolation_state.e_bln_c_s,
            interpolation_state.rbf_coeff_1,
            interpolation_state.rbf_coeff_2,
            interpolation_state.geofac_div,
            interpolation_state.geofac_grg_x,
            interpolation_state.geofac_grg_y,
            interpolation_state.nudgecoeff_e,
            interpolation_state.geofac_n2s,
            tangent_orientation,
            inverse_primal_edge_lengths,
            inverse_dual_edge_length,
            inverse_vertical_vertex_lengths,
            primal_normal_vert[0],
            primal_normal_vert[1],
            dual_normal_vert[0],
            dual_normal_vert[1],
            edge_areas,
            cell_areas,
            self._diff_multfac_vn,
            dtime,
            self.rd_o_cvd,
            self.thresh_tdiff,
            self._smag_limit,
            self.u_vert,
            self.v_vert,
            self.enh_smag_fac,
            self.kh_smag_e,
            self.kh_smag_ec,
            self.z_nabla2_e,
            self.z_temp,
            self.diff_multfac_smag,
            self.diff_multfac_n2w,
            self._smag_offset,
            self.nudgezone_diff,
            self.fac_bdydiff_v,
            self.diff_multfac_w,
            self.vertical_index,
            self.horizontal_cell_index,
            self.horizontal_edge_index,
            cell_startindex_interior,
            cell_startindex_nudging,
            cell_endindex_local_plus1,
            cell_endindex_local,
            edge_startindex_nudging_plus1,
            edge_startindex_nudging_minus1,
            edge_endindex_local,
            edge_endindex_local_minus2,
            vertex_startindex_lb_plus3,
            vertex_startindex_lb_plus1,
            vertex_endindex_local,
            vertex_endindex_local_minus1,
            self.config.vertical_params.index_of_damping_height,
            self.grid.n_lev(),
            self.params.boundary_diffusion_start_index_edges,
            offset_provider={
                "V2E": self.grid.get_v2e_connectivity(),
                "V2EDim": V2EDim,
                "E2C2V": self.grid.get_e2c2v_connectivity(),
                "E2ECV": StridedNeighborOffsetProvider(EdgeDim, ECVDim, 4),
                "C2E": self.grid.get_c2e_connectivity(),
                "C2EDim": C2EDim,
                "E2C": self.grid.get_e2c_connectivity(),
                "C2E2C": self.grid.get_c2e2c_connectivity(),
                "C2E2CDim": C2E2CDim,
                "ECVDim": ECVDim,
                "C2E2CODim": C2E2CODim,
                "C2E2CO": self.grid.get_c2e2co_connectivity(),
                "Koff": KDim,
            },
        )

    def _do_diffusion_step(
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
        primal_normal_vert: Tuple[Field[[ECVDim], float], Field[[ECVDim], float]],
        dual_normal_vert: Tuple[Field[[ECVDim], float], Field[[ECVDim], float]],
        edge_areas: Field[[EdgeDim], float],
        cell_areas: Field[[CellDim], float],
        diff_multfac_vn: Field[[KDim], float],
        smag_limit: Field[[KDim], float],
        smag_offset: float,
    ):
        """
        Run a diffusion step.

        Args:
            diagnostic_state: output argument, data class that contains diagnostic variables
            prognostic_state: output argument, data class that contains prognostic variables
            metric_state:
            interpolation_state:
            dtime: the time step,
            tangent_orientation:
            inverse_primal_edge_lengths:
            inverse_dual_edge_length:
            inverse_vertical_vertex_lengths:
            primal_normal_vert:
            dual_normal_vert:
            edge_areas:
            cell_areas:
            diff_multfac_vn:
            smag_limit:
            smag_offset:

        """
        # -------
        # OUTLINE
        # -------
        klevels = self.grid.n_lev()
        k_start_end_minus2 = klevels - 2

        cell_start_nudging_minus1, cell_end_local_plus1 = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.nudging(CellDim) - 1,
            HorizontalMarkerIndex.local(CellDim) + 1,
        )

        cell_start_interior, cell_end_halo = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.interior(CellDim),
            HorizontalMarkerIndex.local(CellDim),
        )

        cell_start_nudging, cell_end_local = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.nudging(CellDim),
            HorizontalMarkerIndex.local(CellDim),
        )

        edge_start_nudging_plus_one, edge_end_local = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim) + 1,
            HorizontalMarkerIndex.local(EdgeDim),
        )

        (
            edge_start_nudging_minus1,
            edge_end_local_minus2,
        ) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim) - 1,
            HorizontalMarkerIndex.local(EdgeDim) - 2,
        )

        (
            vertex_start_local_boundary_plus3,
            vertex_end_local,
        ) = self.grid.get_indices_from_to(
            VertexDim,
            HorizontalMarkerIndex.local_boundary(VertexDim) + 3,
            HorizontalMarkerIndex.local(VertexDim),
        )
        (
            vertex_start_local_boundary_plus1,
            vertex_end_local_minus1,
        ) = self.grid.get_indices_from_to(
            VertexDim,
            HorizontalMarkerIndex.local_boundary(VertexDim) + 1,
            HorizontalMarkerIndex.local(VertexDim) - 1,
        )

        # Oa TODO: logging
        # 0b TODO: call timer start
        #
        # 0c. dtime dependent stuff: enh_smag_factor,
        scale_k(self.enh_smag_fac, dtime, self.diff_multfac_smag, offset_provider={})

        # TODO: is this needed?, if not remove
        set_zero_v_k(self.u_vert, offset_provider={})
        set_zero_v_k(self.v_vert, offset_provider={})

        # # 1.  CALL rbf_vec_interpol_vertex

        v_start = vertex_start_local_boundary_plus1
        v_end = vertex_end_local_minus1
        k_start = 0
        k_end = klevels
        mo_intp_rbf_rbf_vec_interpol_vertex(
            prognostic_state.normal_wind,
            interpolation_state.rbf_coeff_1,
            interpolation_state.rbf_coeff_2,
            self.u_vert,
            self.v_vert,
            v_start,
            v_end,
            k_start,
            k_end,
            offset_provider={"V2E": self.grid.get_v2e_connectivity(), "V2EDim": V2EDim},
        )

        # 2.  HALO EXCHANGE -- CALL sync_patch_array_mult
        # 3.  mo_nh_diffusion_stencil_01, mo_nh_diffusion_stencil_02, mo_nh_diffusion_stencil_03

        e_start = self.params.boundary_diffusion_start_index_edges
        e_end = edge_end_local_minus2
        k_start = 0
        k_end = klevels
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
            smag_limit,
            smag_offset,
            out=(self.kh_smag_e, self.kh_smag_ec, self.z_nabla2_e),
            offset_provider={
                "E2C2V": self.grid.get_e2c2v_connectivity(),
                "E2ECV": StridedNeighborOffsetProvider(EdgeDim, ECVDim, 4),
            },
        )

        c_start = (cell_start_nudging,)
        c_end = (cell_end_local,)
        k_start = 0
        k_end = klevels
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
            offset_provider={"C2E": self.grid.get_c2e_connectivity(), "C2EDim": C2EDim},
        )
        #
        # # 4.  IF (discr_vn > 1) THEN CALL sync_patch_array -> false for MCH
        #
        # # 5.  CALL rbf_vec_interpol_vertex_wp

        k_start = 0
        k_end = klevels
        v_start = vertex_start_local_boundary_plus3
        v_end = vertex_end_local
        mo_intp_rbf_rbf_vec_interpol_vertex(
            self.z_nabla2_e,
            interpolation_state.rbf_coeff_1,
            interpolation_state.rbf_coeff_2,
            self.u_vert,
            self.v_vert,
            v_start,
            v_end,
            k_start,
            k_end,
            offset_provider={"V2E": self.grid.get_v2e_connectivity(), "V2EDim": V2EDim},
        )
        # # 6.  HALO EXCHANGE -- CALL sync_patch_array_mult
        #
        # # 7.  mo_nh_diffusion_stencil_04, mo_nh_diffusion_stencil_05
        # # 7a. IF (l_limited_area .OR. jg > 1) mo_nh_diffusion_stencil_06
        #

        e_start = edge_start_nudging_plus_one
        e_end = edge_end_local
        k_start = 0
        k_end = klevels
        _fused_mo_nh_diffusion_stencil_04_05_06(
            self.u_vert,
            self.v_vert,
            primal_normal_vert[0],
            primal_normal_vert[1],
            self.z_nabla2_e,
            inverse_vertical_vertex_lengths,
            inverse_primal_edge_lengths,
            edge_areas,
            self.kh_smag_e,
            diff_multfac_vn,
            interpolation_state.nudgecoeff_e,
            prognostic_state.normal_wind,
            self.horizontal_edge_index,
            self.nudgezone_diff,
            self.fac_bdydiff_v,
            edge_start_nudging_minus1,
            out=prognostic_state.normal_wind,
            offset_provider={
                "E2C2V": self.grid.get_e2c2v_connectivity(),
                "E2ECV": StridedNeighborOffsetProvider(EdgeDim, ECVDim, 4),
                "ECVDim": ECVDim,
            },
        )
        # # 7b. mo_nh_diffusion_stencil_07, mo_nh_diffusion_stencil_08,
        # #     mo_nh_diffusion_stencil_09, mo_nh_diffusion_stencil_10

        c_start = cell_start_nudging
        c_end = cell_end_local_plus1
        k_start = 0
        k_end = klevels
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
            cell_start_interior,
            cell_end_halo,
            out=(
                prognostic_state.vertical_wind,
                diagnostic_state.dwdx,
                diagnostic_state.dwdy,
            ),
            offset_provider={
                "C2E2CO": self.grid.get_c2e2co_connectivity(),
                "C2E2CODim": C2E2CODim,
            },
        )
        # # 8.  HALO EXCHANGE: CALL sync_patch_array
        # # 9.  mo_nh_diffusion_stencil_11, mo_nh_diffusion_stencil_12, mo_nh_diffusion_stencil_13,
        # #     mo_nh_diffusion_stencil_14, mo_nh_diffusion_stencil_15, mo_nh_diffusion_stencil_16
        #
        # # TODO check: kh_smag_e is an out field, should  not be calculated in init?
        #

        e_start = edge_start_nudging_plus_one
        e_end = edge_end_local
        c_start = cell_start_nudging_minus1
        c_end = cell_end_local_plus1
        k_start = k_start_end_minus2
        k_end = klevels

        _fused_mo_nh_diffusion_stencil_11_12(
            prognostic_state.theta_v,
            metric_state.theta_ref_mc,
            self.thresh_tdiff,
            self.kh_smag_e,
            out=self.kh_smag_e,
            offset_provider={
                "E2C": self.grid.get_e2c_connectivity(),
                "C2E2C": self.grid.get_c2e2c_connectivity(),
                "C2E2CDim": C2E2CDim,
            },
        )

        c_start = (cell_start_nudging,)
        c_end = cell_end_local
        _fused_mo_nh_diffusion_stencil_13_14(
            self.kh_smag_e,
            inverse_dual_edge_length,
            prognostic_state.theta_v,
            interpolation_state.geofac_div,
            out=self.z_temp,
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "E2C": self.grid.get_e2c_connectivity(),
                "C2EDim": C2EDim,
            },
        )

        # mo_nh_diffusion_stencil_15_numpy(
        #     mask_hdiff=metric_state.mask_hdiff,
        #     zd_vertidx=metric_state.zd_vertidx,
        #     vcoef=metric_state.zd_diffcoef,
        #     zd_diffcoef=metric_state.zd_diffcoef,
        #     geofac_n2s=interpolation_state.geofac_n2s,
        #     theta_v=prognostic_state.theta_v,
        #     z_temp=self.z_temp,
        #     domain={
        #         CellDim: self.grid.get_indices_from_to(
        #             CellDim,
        #             HorizontalMarkerIndex.nudging(CellDim),
        #             HorizontalMarkerIndex.halo(CellDim),
        #         ),
        #     },
        #     offset_provider={},
        # )

        c_start = cell_start_nudging
        c_end = cell_end_local
        _mo_nh_diffusion_stencil_16(
            self.z_temp,
            cell_areas,
            prognostic_state.theta_v,
            prognostic_state.exner_pressure,
            self.rd_o_cvd,
            offset_provider={},
        )
        # 10. HALO EXCHANGE sync_patch_array
