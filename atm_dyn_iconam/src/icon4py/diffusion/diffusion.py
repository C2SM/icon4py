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
import logging
import math
import sys
from collections import namedtuple
from typing import Final, Optional, Tuple

import numpy as np
from gt4py.next.common import Dimension
from gt4py.next.ffront.fbuiltins import Field, int32
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next.program_processors.runners.gtfn_cpu import (
    run_gtfn,
    run_gtfn_cached,
)

from icon4py.atm_dyn_iconam.apply_nabla2_to_w import apply_nabla2_to_w
from icon4py.atm_dyn_iconam.apply_nabla2_to_w_in_upper_damping_layer import (
    apply_nabla2_to_w_in_upper_damping_layer,
)
from icon4py.atm_dyn_iconam.calculate_horizontal_gradients_for_turbulence import (
    calculate_horizontal_gradients_for_turbulence,
)
from icon4py.atm_dyn_iconam.calculate_nabla2_and_smag_coefficients_for_vn import (
    calculate_nabla2_and_smag_coefficients_for_vn,
)
from icon4py.atm_dyn_iconam.calculate_nabla2_for_w import calculate_nabla2_for_w
from icon4py.atm_dyn_iconam.fused_mo_nh_diffusion_stencil_02_03 import (
    fused_mo_nh_diffusion_stencil_02_03,
)
from icon4py.atm_dyn_iconam.fused_mo_nh_diffusion_stencil_04_05_06 import (
    fused_mo_nh_diffusion_stencil_04_05_06,
)
from icon4py.atm_dyn_iconam.fused_mo_nh_diffusion_stencil_07_08_09_10 import (
    fused_mo_nh_diffusion_stencil_07_08_09_10,
)
from icon4py.atm_dyn_iconam.fused_mo_nh_diffusion_stencil_11_12 import (
    fused_mo_nh_diffusion_stencil_11_12,
)
from icon4py.atm_dyn_iconam.fused_mo_nh_diffusion_stencil_13_14 import (
    fused_mo_nh_diffusion_stencil_13_14,
)
from icon4py.atm_dyn_iconam.mo_intp_rbf_rbf_vec_interpol_vertex import (
    mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_15 import (
    mo_nh_diffusion_stencil_15,
)
from icon4py.atm_dyn_iconam.update_theta_and_exner import update_theta_and_exner
from icon4py.common.constants import (
    CPD,
    DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO,
    GAS_CONSTANT_DRY_AIR,
)
from icon4py.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CellDim,
    E2C2VDim,
    E2CDim,
    ECVDim,
    EdgeDim,
    KDim,
    V2EDim,
    VertexDim,
)
from icon4py.diffusion.diagnostic_state import DiagnosticState
from icon4py.diffusion.horizontal import HorizontalMarkerIndex
from icon4py.diffusion.icon_grid import IconGrid, VerticalModelParams
from icon4py.diffusion.interpolation_state import InterpolationState
from icon4py.diffusion.metric_state import MetricState
from icon4py.diffusion.prognostic_state import PrognosticState
from icon4py.diffusion.utils import (
    init_diffusion_local_fields_for_regular_timestep,
    init_nabla2_factor_in_upper_damping_zone,
    scale_k,
    set_zero_v_k,
    setup_fields_for_initial_step,
    zero_field,
)


# flake8: noqa
log = logging.getLogger(__name__)

VectorTuple = namedtuple("VectorTuple", "x y")

cached_backend = run_gtfn_cached
compiled_backend = run_gtfn
backend = compiled_backend  #


class DiffusionConfig:
    """
    Contains necessary parameter to configure a diffusion run.

    Encapsulates namelist parameters and derived parameters.
    Values should be read from configuration.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    TODO: @magdalena to be read from config
    TODO: @magdalena handle dependencies on other namelists (see below...)
    """

    def __init__(
        self,
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
        """
        If True, apply diffusion on the vertical wind field

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

        # parameters from namelist: mo_interpol_nml.f90
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
        """Apply consistency checks and validation on configuration parameters."""
        if self.diffusion_type != 5:
            raise NotImplementedError(
                "Only diffusion type 5 = `Smagorinsky diffusion with fourth-order background "
                "diffusion` is implemented"
            )

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
        # see mo_interpol_nml.f90:
        self.scaled_nudge_max_coeff = (
            config.nudge_max_coeff * DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO
        )

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


class Diffusion:
    """Class that configures diffusion and does one diffusion step."""

    def __init__(self, run_program=True):

        self._initialized = False
        self.rd_o_cvd: float = GAS_CONSTANT_DRY_AIR / (CPD - GAS_CONSTANT_DRY_AIR)
        self.thresh_tdiff: float = (
            -5.0
        )  # threshold temperature deviation from neighboring grid points hat activates extra diffusion against runaway cooling
        self.grid: Optional[IconGrid] = None
        self.config: Optional[DiffusionConfig] = None
        self.params: Optional[DiffusionParams] = None
        self.vertical_params: Optional[VerticalModelParams] = None
        self.interpolation_state: InterpolationState = None
        self.metric_state: MetricState = None
        self.diff_multfac_w: Optional[float] = None
        self.diff_multfac_n2w: Field[[KDim], float] = None
        self.smag_offset: Optional[float] = None
        self.fac_bdydiff_v: Optional[float] = None
        self.bdy_diff: Optional[float] = None
        self.nudgezone_diff: Optional[float] = None

    def init(
        self,
        grid: IconGrid,
        config: DiffusionConfig,
        params: DiffusionParams,
        vertical_params: VerticalModelParams,
        metric_state: MetricState,
        interpolation_state: InterpolationState,
    ):
        """
        Initialize Diffusion granule with configuration.

        calculates all local fields that are used in diffusion within the time loop
        """
        self.config: DiffusionConfig = config
        self.params: DiffusionParams = params
        self.grid = grid
        self.vertical_params = vertical_params
        self.metric_state: MetricState = metric_state
        self.interpolation_state: InterpolationState = interpolation_state

        self._allocate_local_fields()

        self.nudgezone_diff: float = 0.04 / (
            params.scaled_nudge_max_coeff + sys.float_info.epsilon
        )
        self.bdy_diff: float = 0.015 / (
            params.scaled_nudge_max_coeff + sys.float_info.epsilon
        )
        self.fac_bdydiff_v: float = (
            math.sqrt(config.substep_as_float())
            / config.velocity_boundary_diffusion_denominator
            if config.lhdiff_rcf
            else 1.0 / config.velocity_boundary_diffusion_denominator
        )

        self.smag_offset: float = 0.25 * params.K4 * config.substep_as_float()
        self.diff_multfac_w: float = min(
            1.0 / 48.0, params.K4W * config.substep_as_float()
        )

        init_diffusion_local_fields_for_regular_timestep.with_backend(backend)(
            params.K4,
            config.substep_as_float(),
            *params.smagorinski_factor,
            *params.smagorinski_height,
            self.vertical_params.physical_heights,
            self.diff_multfac_vn,
            self.smag_limit,
            self.enh_smag_fac,
            offset_provider={"Koff": KDim},
        )

        self.diff_multfac_n2w = init_nabla2_factor_in_upper_damping_zone(
            k_size=self.grid.n_lev(),
            nshift=0,
            physical_heights=np.asarray(self.vertical_params.physical_heights),
            nrdmax=self.vertical_params.index_of_damping_layer,
        )
        self._initialized = True

    @property
    def initialized(self):
        return self._initialized

    def _allocate_local_fields(self):
        def _allocate(*dims: Dimension):
            return zero_field(self.grid, *dims)

        def _index_field(dim: Dimension, size=None):
            size = size if size else self.grid.size[dim]
            return np_as_located_field(dim)(np.arange(size, dtype=int32))

        self.diff_multfac_vn = _allocate(KDim)

        self.smag_limit = _allocate(KDim)
        self.enh_smag_fac = _allocate(KDim)
        self.u_vert = _allocate(VertexDim, KDim)
        self.v_vert = _allocate(VertexDim, KDim)
        self.kh_smag_e = _allocate(EdgeDim, KDim)
        self.kh_smag_ec = _allocate(EdgeDim, KDim)
        self.z_nabla2_e = _allocate(EdgeDim, KDim)
        self.z_temp = _allocate(CellDim, KDim)
        self.diff_multfac_smag = _allocate(KDim)
        # TODO @magdalena this is KHalfDim
        self.vertical_index = _index_field(KDim, self.grid.n_lev() + 1)
        self.horizontal_cell_index = _index_field(CellDim)
        self.horizontal_edge_index = _index_field(EdgeDim)

    def initial_step(
        self,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
        dtime: float,
        tangent_orientation: Field[[EdgeDim], float],
        inverse_primal_edge_lengths: Field[[EdgeDim], float],
        inverse_dual_edge_length: Field[[EdgeDim], float],
        inverse_vert_vert_lengths: Field[[EdgeDim], float],
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

        setup_fields_for_initial_step.with_backend(backend)(
            self.params.K4,
            self.config.hdiff_efdt_ratio,
            diff_multfac_vn,
            smag_limit,
            offset_provider={},
        )
        self._do_diffusion_step(
            diagnostic_state,
            prognostic_state,
            dtime,
            tangent_orientation,
            inverse_primal_edge_lengths,
            inverse_dual_edge_length,
            inverse_vert_vert_lengths,
            primal_normal_vert,
            dual_normal_vert,
            edge_areas,
            cell_areas,
            diff_multfac_vn,
            smag_limit,
            0.0,
        )

    def run(
        self,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
        dtime: float,
        tangent_orientation: Field[[EdgeDim], float],
        inverse_primal_edge_lengths: Field[[EdgeDim], float],
        inverse_dual_edge_length: Field[[EdgeDim], float],
        inverse_vert_vert_lengths: Field[[EdgeDim], float],
        primal_normal_vert: VectorTuple[Field[[ECVDim], float], Field[[ECVDim], float]],
        dual_normal_vert: VectorTuple[Field[[ECVDim], float], Field[[ECVDim], float]],
        edge_areas: Field[[EdgeDim], float],
        cell_areas: Field[[CellDim], float],
    ):
        """
        Do one diffusion step within regular time loop.

        runs a diffusion step for the parameter linit=False, within regular time loop.
        """

        self._do_diffusion_step(
            diagnostic_state=diagnostic_state,
            prognostic_state=prognostic_state,
            dtime=dtime,
            tangent_orientation=tangent_orientation,
            inverse_primal_edge_lengths=inverse_primal_edge_lengths,
            inverse_dual_edge_length=inverse_dual_edge_length,
            inverse_vertex_vertex_lengths=inverse_vert_vert_lengths,
            primal_normal_vert=primal_normal_vert,
            dual_normal_vert=dual_normal_vert,
            edge_areas=edge_areas,
            cell_areas=cell_areas,
            diff_multfac_vn=self.diff_multfac_vn,
            smag_limit=self.smag_limit,
            smag_offset=self.smag_offset,
        )
        log.info("diffusion program: end")

    def _do_diffusion_step(
        self,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
        dtime: float,
        tangent_orientation: Field[[EdgeDim], float],
        inverse_primal_edge_lengths: Field[[EdgeDim], float],
        inverse_dual_edge_length: Field[[EdgeDim], float],
        inverse_vertex_vertex_lengths: Field[[EdgeDim], float],
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
            dtime: the time step,
            tangent_orientation:
            inverse_primal_edge_lengths:
            inverse_dual_edge_length:
            inverse_vertex_vertex_lengths:
            primal_normal_vert:
            dual_normal_vert:
            edge_areas:
            cell_areas:
            diff_multfac_vn:
            smag_limit:
            smag_offset:

        """
        klevels = self.grid.n_lev()
        cell_start_interior, cell_end_local = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.interior(CellDim),
            HorizontalMarkerIndex.local(CellDim),
        )

        cell_start_nudging, cell_end_halo = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.nudging(CellDim),
            HorizontalMarkerIndex.halo(CellDim),
        )

        edge_start_nudging_plus_one, edge_end_local = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim) + 1,
            HorizontalMarkerIndex.local(EdgeDim),
        )

        edge_start_nudging, edge_end_halo = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim),
            HorizontalMarkerIndex.halo(EdgeDim),
        )

        edge_start_lb_plus4, _ = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4,
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
            HorizontalMarkerIndex.lateral_boundary(VertexDim) + 3,
            HorizontalMarkerIndex.local(VertexDim),
        )
        (
            vertex_start_lb_plus1,
            vertex_end_local_minus1,
        ) = self.grid.get_indices_from_to(
            VertexDim,
            HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1,
            HorizontalMarkerIndex.local(VertexDim) - 1,
        )

        # 0b call timer start
        #
        # 0c. dtime dependent stuff: enh_smag_factor,
        scale_k.with_backend(backend)(
            self.enh_smag_fac, dtime, self.diff_multfac_smag, offset_provider={}
        )

        # TODO: @magdalena is this needed?, if not remove
        set_zero_v_k.with_backend(backend)(self.u_vert, offset_provider={})
        set_zero_v_k.with_backend(backend)(self.v_vert, offset_provider={})
        log.debug("rbf interpolation: start")
        # # 1.  CALL rbf_vec_interpol_vertex
        mo_intp_rbf_rbf_vec_interpol_vertex.with_backend(backend)(
            p_e_in=prognostic_state.vn,
            ptr_coeff_1=self.interpolation_state.rbf_coeff_1,
            ptr_coeff_2=self.interpolation_state.rbf_coeff_2,
            p_u_out=self.u_vert,
            p_v_out=self.v_vert,
            horizontal_start=vertex_start_lb_plus1,
            horizontal_end=vertex_end_local,
            vertical_start=0,
            vertical_end=klevels,
            offset_provider={"V2E": self.grid.get_v2e_connectivity()},
        )
        log.debug("rbf interpolation: end")
        # 2.  HALO EXCHANGE -- CALL sync_patch_array_mult
        # 3.  mo_nh_diffusion_stencil_01, mo_nh_diffusion_stencil_02, mo_nh_diffusion_stencil_03

        log.debug("running calculate_nabla2_and_smag_coefficients_for_vn: start")
        calculate_nabla2_and_smag_coefficients_for_vn.with_backend(backend)(
            diff_multfac_smag=self.diff_multfac_smag,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inverse_primal_edge_lengths,
            inv_vert_vert_length=inverse_vertex_vertex_lengths,
            u_vert=self.u_vert,
            v_vert=self.v_vert,
            primal_normal_vert_x=primal_normal_vert[0],
            primal_normal_vert_y=primal_normal_vert[1],
            dual_normal_vert_x=dual_normal_vert[0],
            dual_normal_vert_y=dual_normal_vert[1],
            vn=prognostic_state.vn,
            smag_limit=smag_limit,
            kh_smag_e=self.kh_smag_e,
            kh_smag_ec=self.kh_smag_ec,
            z_nabla2_e=self.z_nabla2_e,
            smag_offset=smag_offset,
            horizontal_start=edge_start_lb_plus4,
            horizontal_end=edge_end_local_minus2,
            vertical_start=0,
            vertical_end=klevels,
            offset_provider={
                "E2C2V": self.grid.get_e2c2v_connectivity(),
                "E2ECV": self.grid.get_e2ecv_connectivity(),
            },
        )
        log.debug("running calculate_nabla2_and_smag_coefficients_for_vn: end")
        log.debug("running fused stencil fused stencil 02_03: start")
        fused_mo_nh_diffusion_stencil_02_03.with_backend(backend)(
            kh_smag_ec=self.kh_smag_ec,
            vn=prognostic_state.vn,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            geofac_div=self.interpolation_state.geofac_div,
            diff_multfac_smag=self.diff_multfac_smag,
            wgtfac_c=self.metric_state.wgtfac_c,
            div_ic=diagnostic_state.div_ic,
            hdef_ic=diagnostic_state.hdef_ic,
            horizontal_start=cell_start_nudging,
            horizontal_end=cell_end_local,
            vertical_start=1,
            vertical_end=klevels,
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "Koff": KDim,
            },
        )
        log.debug("running fused stencil fused stencil 02_03: end")
        #
        # # 4.  IF (discr_vn > 1) THEN CALL sync_patch_array -> false for MCH
        #
        # # 5.  CALL rbf_vec_interpol_vertex_wp
        log.debug("rbf interpolation: start")
        mo_intp_rbf_rbf_vec_interpol_vertex.with_backend(backend)(
            p_e_in=self.z_nabla2_e,
            ptr_coeff_1=self.interpolation_state.rbf_coeff_1,
            ptr_coeff_2=self.interpolation_state.rbf_coeff_2,
            p_u_out=self.u_vert,
            p_v_out=self.v_vert,
            horizontal_start=vertex_start_lb_plus1,
            horizontal_end=vertex_end_local,
            vertical_start=0,
            vertical_end=klevels,
            offset_provider={"V2E": self.grid.get_v2e_connectivity()},
        )
        log.debug("rbf interpolation: end")
        # # # 6.  HALO EXCHANGE -- CALL sync_patch_array_mult
        # #
        # # # 7.  mo_nh_diffusion_stencil_04, mo_nh_diffusion_stencil_05
        # # # 7a. IF (l_limited_area .OR. jg > 1) mo_nh_diffusion_stencil_06
        # #
        #
        log.debug("running fused stencil 04_05_06: start")
        fused_mo_nh_diffusion_stencil_04_05_06.with_backend(backend)(
            u_vert=self.u_vert,
            v_vert=self.v_vert,
            primal_normal_vert_v1=primal_normal_vert[0],
            primal_normal_vert_v2=primal_normal_vert[1],
            z_nabla2_e=self.z_nabla2_e,
            inv_vert_vert_length=inverse_vertex_vertex_lengths,
            inv_primal_edge_length=inverse_primal_edge_lengths,
            area_edge=edge_areas,
            kh_smag_e=self.kh_smag_e,
            diff_multfac_vn=diff_multfac_vn,
            nudgecoeff_e=self.interpolation_state.nudgecoeff_e,
            vn=prognostic_state.vn,
            horz_idx=self.horizontal_edge_index,
            nudgezone_diff=self.nudgezone_diff,
            fac_bdydiff_v=self.fac_bdydiff_v,
            start_2nd_nudge_line_idx_e=int32(edge_start_nudging_plus_one),
            horizontal_start=edge_start_lb_plus4,
            horizontal_end=edge_end_local,
            vertical_start=0,
            vertical_end=klevels,
            offset_provider={
                "E2C2V": self.grid.get_e2c2v_connectivity(),
                "E2ECV": self.grid.get_e2ecv_connectivity(),
            },
        )
        log.debug("running fused stencil 04_05_06: end")
        # # 7b. mo_nh_diffusion_stencil_07, mo_nh_diffusion_stencil_08,
        # #     mo_nh_diffusion_stencil_09, mo_nh_diffusion_stencil_10

        log.debug("running stencils 07_08_09_10: start")
        calculate_horizontal_gradients_for_turbulence.with_backend(backend)(
            w=prognostic_state.w,
            geofac_grg_x=self.interpolation_state.geofac_grg_x,
            geofac_grg_y=self.interpolation_state.geofac_grg_y,
            dwdx=diagnostic_state.dwdx,
            dwdy=diagnostic_state.dwdy,
            vertical_start=1,
            vertical_end=klevels,
            horizontal_start=cell_start_nudging,
            horizontal_end=cell_end_halo,
            offset_provider={"C2E2CO": self.grid.get_c2e2co_connectivity()},
        )

        z_nabla2_c = zero_field(self.grid, CellDim, KDim, dtype=float)

        calculate_nabla2_for_w.with_backend(backend)(
            w=prognostic_state.w,
            geofac_n2s=self.interpolation_state.geofac_n2s,
            z_nabla2_c=z_nabla2_c,
            vertical_start=0,
            vertical_end=klevels,
            horizontal_start=cell_start_nudging,
            horizontal_end=cell_end_halo,
            offset_provider={"C2E2CO": self.grid.get_c2e2co_connectivity()},
        )
        apply_nabla2_to_w.with_backend(backend)(
            area=cell_areas,
            z_nabla2_c=z_nabla2_c,
            w=prognostic_state.w,
            diff_multfac_w=self.diff_multfac_w,
            geofac_n2s=self.interpolation_state.geofac_n2s,
            vertical_start=0,
            vertical_end=klevels,
            horizontal_start=cell_start_interior,
            horizontal_end=cell_end_local,
            offset_provider={"C2E2CO": self.grid.get_c2e2co_connectivity()},
        )

        # TODO @magdalena: fix this python offset problem (+1): int(self.vertical_params.index_of_damping_layer + 1)
        apply_nabla2_to_w_in_upper_damping_layer.with_backend(backend)(
            w=prognostic_state.w,
            diff_multfac_n2w=self.diff_multfac_n2w,
            cell_area=cell_areas,
            z_nabla2_c=z_nabla2_c,
            vertical_start=1,
            vertical_end=int(self.vertical_params.index_of_damping_layer + 1),
            horizontal_start=int(cell_start_interior),
            horizontal_end=int(cell_end_local),
            offset_provider={},
        )
        # w_old = prognostic_state.w
        # fused_mo_nh_diffusion_stencil_07_08_09_10.with_backend(backend)(
        #     area=cell_areas,
        #     geofac_n2s=self.interpolation_state.geofac_n2s,
        #     geofac_grg_x=self.interpolation_state.geofac_grg_x,
        #     geofac_grg_y=self.interpolation_state.geofac_grg_y,
        #     w_old=w_old,
        #     w=prognostic_state.w,
        #     dwdx=diagnostic_state.dwdx,
        #     dwdy=diagnostic_state.dwdy,
        #     diff_multfac_w=self.diff_multfac_w,
        #     diff_multfac_n2w=self.diff_multfac_n2w,
        #     vert_idx=self.vertical_index,
        #     horz_idx=self.horizontal_cell_index,
        #     nrdmax=int32(self.vertical_params.index_of_damping_layer +1) ,
        #     interior_idx=int32(
        #         cell_start_interior -1
        #     ),  # h end index for stencil_09 and stencil_10 # TODO: in ICON: start_interior_idx_c -1 ??
        #     halo_idx=int32(
        #         cell_end_local
        #     ),  # h end index for stencil_09 and stencil_10,
        #     horizontal_start=cell_start_nudging,  # h start index for stencil_07 and stencil_08
        #     horizontal_end=cell_end_halo,  # h end index for stencil_07 and stencil_08
        #     vertical_start=0,
        #     vertical_end=klevels,
        #     offset_provider={
        #         "C2E2CO": self.grid.get_c2e2co_connectivity(),
        #     },
        # )
        log.debug("running fused stencil 07_08_09_10: end")
        # # 8.  HALO EXCHANGE: CALL sync_patch_array
        # # 9.  mo_nh_diffusion_stencil_11, mo_nh_diffusion_stencil_12, mo_nh_diffusion_stencil_13,
        # #     mo_nh_diffusion_stencil_14, mo_nh_diffusion_stencil_15, mo_nh_diffusion_stencil_16
        #
        # # TODO @magdalena check: kh_smag_e is an out field, should  not be calculated in init?
        #
        log.debug("running fused stencil 11_12: start")
        fused_mo_nh_diffusion_stencil_11_12.with_backend(backend)(
            theta_v=prognostic_state.theta_v,
            theta_ref_mc=self.metric_state.theta_ref_mc,
            thresh_tdiff=self.thresh_tdiff,
            kh_smag_e=self.kh_smag_e,
            horizontal_start=edge_start_nudging,
            horizontal_end=edge_end_halo,
            vertical_start=(klevels - 2),
            vertical_end=klevels,
            offset_provider={
                "E2C": self.grid.get_e2c_connectivity(),
                "C2E2C": self.grid.get_c2e2c_connectivity(),
            },
        )
        log.debug("running fused stencil 11_12: end")
        log.debug("running fused stencil 13_14: start")
        fused_mo_nh_diffusion_stencil_13_14.with_backend(backend)(
            kh_smag_e=self.kh_smag_e,
            inv_dual_edge_length=inverse_dual_edge_length,
            theta_v=prognostic_state.theta_v,
            geofac_div=self.interpolation_state.geofac_div,
            z_temp=self.z_temp,
            horizontal_start=cell_start_nudging,
            horizontal_end=cell_end_local,
            vertical_start=0,
            vertical_end=klevels,
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "E2C": self.grid.get_e2c_connectivity(),
            },
        )
        log.debug("running fused stencil 13_14: end")
        log.debug("running fused stencil 15: start")
        mo_nh_diffusion_stencil_15.with_backend(backend)(
            mask=self.metric_state.mask_hdiff,
            zd_vertoffset=self.metric_state.zd_vertoffset,
            zd_diffcoef=self.metric_state.zd_diffcoef,
            geofac_n2s_c=self.interpolation_state.geofac_n2s_c,
            geofac_n2s_nbh=self.interpolation_state.geofac_n2s_nbh,
            vcoef=self.metric_state.zd_intcoef,
            theta_v=prognostic_state.theta_v,
            z_temp=self.z_temp,
            horizontal_start=cell_start_nudging,
            horizontal_end=cell_end_local,
            vertical_start=0,
            vertical_end=klevels,
            offset_provider={
                "C2CEC": self.grid.get_c2cec_connectivity(),
                "C2E2C": self.grid.get_c2e2c_connectivity(),
                "Koff": KDim,
            },
        )

        log.debug("running fused stencil 15: end")
        log.debug("running fused stencil update_theta_and_exner: start")
        update_theta_and_exner.with_backend(backend)(
            z_temp=self.z_temp,
            area=cell_areas,
            theta_v=prognostic_state.theta_v,
            exner=prognostic_state.exner_pressure,
            rd_o_cvd=self.rd_o_cvd,
            horizontal_start=cell_start_nudging,
            horizontal_end=cell_end_local,
            vertical_start=0,
            vertical_end=klevels,
            offset_provider={},
        )

        log.debug("running fused stencil update_theta_and_exner: end")
        # 10. HALO EXCHANGE sync_patch_array
