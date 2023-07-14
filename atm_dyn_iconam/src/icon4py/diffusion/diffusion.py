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
import functools
import logging
import math
import sys
from collections import namedtuple
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Final, Optional, Tuple

import numpy as np
from gt4py.next.common import Dimension
from gt4py.next.ffront.fbuiltins import Field, int32
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next.program_processors.runners.gtfn_cpu import (
    run_gtfn,
    run_gtfn_cached,
)

from icon4py.atm_dyn_iconam.apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulance import (
    apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulance,
)
from icon4py.atm_dyn_iconam.calculate_diagnostic_quantities_for_turbulence import (
    calculate_diagnostic_quantities_for_turbulence,
)
from icon4py.atm_dyn_iconam.calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools import (
    calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools,
)
from icon4py.atm_dyn_iconam.calculate_nabla2_and_smag_coefficients_for_vn import (
    calculate_nabla2_and_smag_coefficients_for_vn,
)
from icon4py.atm_dyn_iconam.calculate_nabla2_for_theta import (
    calculate_nabla2_for_theta,
)
from icon4py.atm_dyn_iconam.fused_mo_nh_diffusion_stencil_04_05_06 import (
    fused_mo_nh_diffusion_stencil_04_05_06,
)
from icon4py.atm_dyn_iconam.mo_intp_rbf_rbf_vec_interpol_vertex import (
    mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.atm_dyn_iconam.truly_horizontal_diffusion_nabla_of_theta_over_steep_points import (
    truly_horizontal_diffusion_nabla_of_theta_over_steep_points,
)
from icon4py.atm_dyn_iconam.update_theta_and_exner import update_theta_and_exner
from icon4py.common.constants import (
    CPD,
    DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO,
    GAS_CONSTANT_DRY_AIR,
)
from icon4py.common.dimension import CellDim, EdgeDim, KDim, VertexDim
from icon4py.diffusion.diffusion_utils import (
    copy_field,
    init_diffusion_local_fields_for_regular_timestep,
    init_nabla2_factor_in_upper_damping_zone,
    scale_k,
    setup_fields_for_initial_step,
    zero_field,
)
from icon4py.diffusion.state_utils import (
    DiagnosticState,
    InterpolationState,
    MetricState,
    PrognosticState,
)
from icon4py.grid.horizontal import CellParams, EdgeParams, HorizontalMarkerIndex
from icon4py.grid.icon_grid import IconGrid, VerticalModelParams


# flake8: noqa
log = logging.getLogger(__name__)

VectorTuple = namedtuple("VectorTuple", "x y")

cached_backend = run_gtfn_cached
compiled_backend = run_gtfn
backend = compiled_backend


class DiffusionType(int, Enum):
    """
    Order of nabla operator for diffusion.
    Note: Called `hdiff_order` in `mo_diffusion_nml.f90`.
    Note: We currently only support type 5.
    """

    NO_DIFFUSION = -1  #: no diffusion
    LINEAR_2ND_ORDER = 2  #: 2nd order linear diffusion on all vertical levels
    SMAGORINSKY_NO_BACKGROUND = 3  #: Smagorinsky diffusion without background diffusion
    LINEAR_4TH_ORDER = 4  #: 4th order linear diffusion on all vertical levels
    SMAGORINSKY_4TH_ORDER = (
        5  #: Smagorinsky diffusion with fourth-order background diffusion
    )


class DiffusionConfig:
    """
    Contains necessary parameter to configure a diffusion run.

    Encapsulates namelist parameters and derived parameters.
    Values should be read from configuration.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    """

    # TODO(Magdalena): to be read from config
    # TODO(Magdalena):  handle dependencies on other namelists (see below...)

    def __init__(
        self,
        diffusion_type: DiffusionType = DiffusionType.SMAGORINSKY_4TH_ORDER,
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
        """Set the diffusion configuration parameters with the ICON default values."""
        # parameters from namelist diffusion_nml

        self.diffusion_type: int = diffusion_type

        #: If True, apply diffusion on the vertical wind field
        #: Called `lhdiff_w` in mo_diffusion_nml.f90
        self.apply_to_vertical_wind: bool = hdiff_w

        #: True apply diffusion on the horizontal wind field, is ONLY used in mo_nh_stepping.f90
        #: Called `lhdiff_vn` in mo_diffusion_nml.f90
        self.apply_to_horizontal_wind = hdiff_vn

        #:  If True, apply horizontal diffusion to temperature field
        #: Called `lhdiff_temp` in mo_diffusion_nml.f90
        self.apply_to_temperature: bool = hdiff_temp

        #: If True, compute 3D Smagorinsky diffusion coefficient
        #: Called `lsmag_3d` in mo_diffusion_nml.f90
        self.compute_3d_smag_coeff: bool = smag_3d

        #: Options for discretizing the Smagorinsky momentum diffusion
        #: Called `itype_vn_diffu` in mo_diffusion_nml.f90
        self.type_vn_diffu: int = type_vn_diffu

        #: Options for discretizing the Smagorinsky temperature diffusion
        #: Called `itype_t_diffu` inmo_diffusion_nml.f90
        self.type_t_diffu = type_t_diffu

        #: Ratio of e-folding time to (2*)time step
        #: Called `hdiff_efdt_ratio` inmo_diffusion_nml.f90
        self.hdiff_efdt_ratio: float = hdiff_efdt_ratio

        #: Ratio of e-folding time to time step for w diffusion (NH only)
        #: Called `hdiff_w_efdt_ratio` inmo_diffusion_nml.f90.
        self.hdiff_w_efdt_ratio: float = hdiff_w_efdt_ratio

        #: Scaling factor for Smagorinsky diffusion at height hdiff_smag_z and below
        #: Called `hdiff_smag_fac` inmo_diffusion_nml.f90
        self.smagorinski_scaling_factor: float = smagorinski_scaling_factor

        #: If True, apply truly horizontal temperature diffusion over steep slopes
        #: Called 'l_zdiffu_t' in mo_nonhydrostatic_nml.f90
        self.apply_zdiffusion_t: bool = zdiffu_t

        # from other namelists:
        # from parent namelist mo_nonhydrostatic_nml

        #: Number of dynamics substeps per fast-physics step
        #: Called 'ndyn_substeps' in mo_nonhydrostatic_nml.f90
        self.ndyn_substeps: int = n_substeps

        #: If True, compute horizontal diffusion only at the large time step
        #: Called 'lhdiff_rcf' in mo_nonhydrostatic_nml.f90
        self.lhdiff_rcf: bool = hdiff_rcf

        # namelist mo_gridref_nml.f90

        #: Denominator for temperature boundary diffusion
        #: Called 'denom_diffu_t' in mo_gridref_nml.f90
        self.temperature_boundary_diffusion_denominator: float = (
            temperature_boundary_diffusion_denom
        )

        #: Denominator for velocity boundary diffusion
        #: Called 'denom_diffu_v' in mo_gridref_nml.f90
        self.velocity_boundary_diffusion_denominator: float = (
            velocity_boundary_diffusion_denom
        )

        # parameters from namelist: mo_interpol_nml.f90

        #: Parameter describing the lateral boundary nudging in limited area mode.
        #:
        #: Maximal value of the nudging coefficients used cell row bordering the boundary interpolation zone,
        #: from there nudging coefficients decay exponentially with `nudge_efold_width` in units of cell rows.
        #: Called `nudge_max_coeff` in mo_interpol_nml.f90
        self.nudge_max_coeff: float = max_nudging_coeff

        #: Exponential decay rate (in units of cell rows) of the lateral boundary nudging coefficients
        #: Called `nudge_efold_width` in mo_interpol_nml.f90
        self.nudge_efold_width: float = nudging_decay_rate

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

    @functools.cached_property
    def substep_as_float(self):
        return float(self.ndyn_substeps)


@dataclass(frozen=True)
class DiffusionParams:
    """Calculates derived quantities depending on the diffusion config."""

    config: InitVar[DiffusionConfig]
    K2: Final[float] = field(init=False)
    K4: Final[float] = field(init=False)
    K6: Final[float] = field(init=False)
    K4W: Final[float] = field(init=False)
    smagorinski_factor: Final[float] = field(init=False)
    smagorinski_height: Final[float] = field(init=False)
    scaled_nudge_max_coeff: Final[float] = field(init=False)

    def __post_init__(self, config):
        object.__setattr__(
            self,
            "K2",
            (
                1.0 / (config.hdiff_efdt_ratio * 8.0)
                if config.hdiff_efdt_ratio > 0.0
                else 0.0
            ),
        )
        object.__setattr__(self, "K4", self.K2 / 8.0)
        object.__setattr__(self, "K6", self.K2 / 64.0)
        object.__setattr__(
            self,
            "K4W",
            (
                1.0 / (config.hdiff_w_efdt_ratio * 36.0)
                if config.hdiff_w_efdt_ratio > 0
                else 0.0
            ),
        )

        (
            smagorinski_factor,
            smagorinski_height,
        ) = self._determine_smagorinski_factor(config)
        object.__setattr__(self, "smagorinski_factor", smagorinski_factor)
        object.__setattr__(self, "smagorinski_height", smagorinski_height)
        # see mo_interpol_nml.f90:
        object.__setattr__(
            self,
            "scaled_nudge_max_coeff",
            config.nudge_max_coeff * DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO,
        )

    def _determine_smagorinski_factor(self, config: DiffusionConfig):
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
                ) = diffusion_type_5_smagorinski_factor(config)
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


def diffusion_type_5_smagorinski_factor(config: DiffusionConfig):
    """
    Initialize Smagorinski factors used in diffusion type 5.

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

    def __init__(self):

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
        self.edge_params: Optional[EdgeParams] = None
        self.cell_params: Optional[CellParams] = None

    def init(
        self,
        grid: IconGrid,
        config: DiffusionConfig,
        params: DiffusionParams,
        vertical_params: VerticalModelParams,
        metric_state: MetricState,
        interpolation_state: InterpolationState,
        edge_params: EdgeParams,
        cell_params: CellParams,
    ):
        """
        Initialize Diffusion granule with configuration.

        calculates all local fields that are used in diffusion within the time loop.

        Args:
            grid:
            config:
            params:
            vertical_params:
            metric_state:
            interpolation_state:
            edge_params:
            cell_params:
        """
        self.config: DiffusionConfig = config
        self.params: DiffusionParams = params
        self.grid = grid
        self.vertical_params = vertical_params
        self.metric_state: MetricState = metric_state
        self.interpolation_state: InterpolationState = interpolation_state
        self.edge_params = edge_params
        self.cell_params = cell_params

        self._allocate_temporary_fields()

        self.nudgezone_diff: float = 0.04 / (
            params.scaled_nudge_max_coeff + sys.float_info.epsilon
        )
        self.bdy_diff: float = 0.015 / (
            params.scaled_nudge_max_coeff + sys.float_info.epsilon
        )
        self.fac_bdydiff_v: float = (
            math.sqrt(config.substep_as_float)
            / config.velocity_boundary_diffusion_denominator
            if config.lhdiff_rcf
            else 1.0 / config.velocity_boundary_diffusion_denominator
        )

        self.smag_offset: float = 0.25 * params.K4 * config.substep_as_float
        self.diff_multfac_w: float = min(
            1.0 / 48.0, params.K4W * config.substep_as_float
        )

        init_diffusion_local_fields_for_regular_timestep.with_backend(backend)(
            params.K4,
            config.substep_as_float,
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

    def _allocate_temporary_fields(self):
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
        # TODO(Magdalena): this is KHalfDim
        self.vertical_index = _index_field(KDim, self.grid.n_lev() + 1)
        self.horizontal_cell_index = _index_field(CellDim)
        self.horizontal_edge_index = _index_field(EdgeDim)
        self.w_tmp = np_as_located_field(CellDim, KDim)(
            np.zeros((self.grid.num_cells(), self.grid.n_lev() + 1), dtype=float)
        )

    def initial_run(
        self,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
        dtime: float,
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
            diff_multfac_vn,
            smag_limit,
            0.0,
        )

    def run(
        self,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
        dtime: float,
    ):
        """
        Do one diffusion step within regular time loop.

        runs a diffusion step for the parameter linit=False, within regular time loop.
        """

        self._do_diffusion_step(
            diagnostic_state=diagnostic_state,
            prognostic_state=prognostic_state,
            dtime=dtime,
            diff_multfac_vn=self.diff_multfac_vn,
            smag_limit=self.smag_limit,
            smag_offset=self.smag_offset,
        )

    def _do_diffusion_step(
        self,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
        dtime: float,
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

        # dtime dependent: enh_smag_factor,
        scale_k.with_backend(backend)(
            self.enh_smag_fac, dtime, self.diff_multfac_smag, offset_provider={}
        )

        log.debug("rbf interpolation: start")
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

        # HALO EXCHANGE -- CALL sync_patch_array_mult

        log.debug(
            "running stencil 01(calculate_nabla2_and_smag_coefficients_for_vn): start"
        )
        calculate_nabla2_and_smag_coefficients_for_vn.with_backend(backend)(
            diff_multfac_smag=self.diff_multfac_smag,
            tangent_orientation=self.edge_params.tangent_orientation,
            inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
            inv_vert_vert_length=self.edge_params.inverse_vertex_vertex_lengths,
            u_vert=self.u_vert,
            v_vert=self.v_vert,
            primal_normal_vert_x=self.edge_params.primal_normal_vert[0],
            primal_normal_vert_y=self.edge_params.primal_normal_vert[1],
            dual_normal_vert_x=self.edge_params.dual_normal_vert[0],
            dual_normal_vert_y=self.edge_params.dual_normal_vert[1],
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
        log.debug(
            "running stencil 01 (calculate_nabla2_and_smag_coefficients_for_vn): end"
        )
        log.debug(
            "running stencils 02 03 (calculate_diagnostic_quantities_for_turbulence): start"
        )
        calculate_diagnostic_quantities_for_turbulence.with_backend(backend)(
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
                "C2CE": self.grid.get_c2ce_connectivity(),
                "Koff": KDim,
            },
        )
        log.debug(
            "running stencils 02 03 (calculate_diagnostic_quantities_for_turbulence): end"
        )

        # HALO EXCHANGE  IF (discr_vn > 1) THEN CALL sync_patch_array -> false for MCH

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

        # 6.  HALO EXCHANGE -- CALL sync_patch_array_mult

        log.debug("running stencil 04 05 06: start")
        fused_mo_nh_diffusion_stencil_04_05_06.with_backend(backend)(
            u_vert=self.u_vert,
            v_vert=self.v_vert,
            primal_normal_vert_v1=self.edge_params.primal_normal_vert[0],
            primal_normal_vert_v2=self.edge_params.primal_normal_vert[1],
            z_nabla2_e=self.z_nabla2_e,
            inv_vert_vert_length=self.edge_params.inverse_vertex_vertex_lengths,
            inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
            area_edge=self.edge_params.edge_areas,
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
        log.debug("runningstencils 04 05 06: end")

        log.debug(
            "running stencils 07 08 09 10 (apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulance): start"
        )
        copy_field.with_backend(backend)(
            prognostic_state.w, self.w_tmp, offset_provider={}
        )
        apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulance.with_backend(
            backend
        )(
            area=self.cell_params.area,
            geofac_n2s=self.interpolation_state.geofac_n2s,
            geofac_grg_x=self.interpolation_state.geofac_grg_x,
            geofac_grg_y=self.interpolation_state.geofac_grg_y,
            w_old=self.w_tmp,
            w=prognostic_state.w,
            dwdx=diagnostic_state.dwdx,
            dwdy=diagnostic_state.dwdy,
            diff_multfac_w=self.diff_multfac_w,
            diff_multfac_n2w=self.diff_multfac_n2w,
            vert_idx=self.vertical_index,
            horz_idx=self.horizontal_cell_index,
            nrdmax=int32(
                self.vertical_params.index_of_damping_layer + 1
            ),  # +1 since Fortran includes boundaries
            interior_idx=int32(cell_start_interior),
            halo_idx=int32(cell_end_local),
            horizontal_start=cell_start_nudging,
            horizontal_end=cell_end_halo,
            vertical_start=0,
            vertical_end=klevels,
            offset_provider={
                "C2E2CO": self.grid.get_c2e2co_connectivity(),
            },
        )
        log.debug(
            "running stencils 07 08 09 10 (apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulance): end"
        )
        # HALO EXCHANGE: CALL sync_patch_array

        log.debug(
            "running fused stencils 11 12 (calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools): start"
        )
        calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools.with_backend(
            backend
        )(
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
        log.debug(
            "running stencils 11 12 (calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools): end"
        )
        log.debug("running stencils 13 14 (calculate_nabla2_for_theta): start")
        calculate_nabla2_for_theta.with_backend(backend)(
            kh_smag_e=self.kh_smag_e,
            inv_dual_edge_length=self.edge_params.inverse_dual_edge_lengths,
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
                "C2CE": self.grid.get_c2ce_connectivity(),
            },
        )
        log.debug("running stencils 13_14 (calculate_nabla2_for_theta): end")
        log.debug(
            "running stencil 15 (truly_horizontal_diffusion_nabla_of_theta_over_steep_points): start"
        )
        truly_horizontal_diffusion_nabla_of_theta_over_steep_points.with_backend(
            backend
        )(
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

        log.debug(
            "running fused stencil 15 (truly_horizontal_diffusion_nabla_of_theta_over_steep_points): end"
        )
        log.debug("running fused stencil 16 (update_theta_and_exner): start")
        update_theta_and_exner.with_backend(backend)(
            z_temp=self.z_temp,
            area=self.cell_params.area,
            theta_v=prognostic_state.theta_v,
            exner=prognostic_state.exner_pressure,
            rd_o_cvd=self.rd_o_cvd,
            horizontal_start=cell_start_nudging,
            horizontal_end=cell_end_local,
            vertical_start=0,
            vertical_end=klevels,
            offset_provider={},
        )
        log.debug("running stencil 16 (update_theta_and_exner): end")
        # 10. HALO EXCHANGE sync_patch_array
