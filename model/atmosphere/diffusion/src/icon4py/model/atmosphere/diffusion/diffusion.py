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
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Final, Optional

from gt4py.next import as_field
from gt4py.next.common import Dimension
from gt4py.next.ffront.fbuiltins import Field, int32

from icon4py.model.atmosphere.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
)
from icon4py.model.atmosphere.diffusion.diffusion_utils import (
    copy_field,
    init_diffusion_local_fields_for_regular_timestep,
    init_nabla2_factor_in_upper_damping_zone,
    scale_k,
    setup_fields_for_initial_step,
    zero_field,
)
from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_vn import apply_diffusion_to_vn
from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence import (
    apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_diagnostic_quantities_for_turbulence import (
    calculate_diagnostic_quantities_for_turbulence,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools import (
    calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_and_smag_coefficients_for_vn import (
    calculate_nabla2_and_smag_coefficients_for_vn,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_theta import (
    calculate_nabla2_for_theta,
)
from icon4py.model.atmosphere.diffusion.stencils.truly_horizontal_diffusion_nabla_of_theta_over_steep_points import (
    truly_horizontal_diffusion_nabla_of_theta_over_steep_points,
)
from icon4py.model.atmosphere.diffusion.stencils.update_theta_and_exner import (
    update_theta_and_exner,
)
from icon4py.model.common.constants import (
    CPD,
    DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO,
    GAS_CONSTANT_DRY_AIR,
    dbl_eps,
)
from icon4py.model.common.decomposition.definitions import ExchangeRuntime, SingleNodeExchange
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, VertexDim, DimensionKind
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams, HorizontalMarkerIndex
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.interpolation.stencils.mo_intp_rbf_rbf_vec_interpol_vertex import (
    mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.model.common.states.prognostic_state import PrognosticState

import dace
from dace import hooks, dtypes
from dace.config import Config
from dace.memlet import Memlet
from dace.properties import CodeBlock
from gt4py.next.program_processors.runners.dace import run_dace_cpu_noopt
from icon4py.model.common.decomposition.mpi_decomposition import GHexMultiNodeExchange
from icon4py.model.common.decomposition import definitions, mpi_decomposition
from icon4py.model.common.decomposition.definitions import DecompositionInfo as di
try:
    import ghex
    import mpi4py
    from dace.sdfg.utils import distributed_compile
except ImportError:
    ghex = None
    mpi4py = None

from icon4py.model.common.settings import backend
from icon4py.model.common.settings import xp

"""
Diffusion module ported from ICON mo_nh_diffusion.f90.

Supports only diffusion_type (=hdiff_order) 5 from the diffusion namelist.
"""

# flake8: noqa
log = logging.getLogger(__name__)


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
    SMAGORINSKY_4TH_ORDER = 5  #: Smagorinsky diffusion with fourth-order background diffusion


class TurbulenceShearForcingType(int, Enum):
    """
    Type of shear forcing used in turbulance.

    Note: called `itype_sher` in `mo_turbdiff_nml.f90`
    """

    VERTICAL_OF_HORIZONTAL_WIND = 0  #: only vertical shear of horizontal wind
    VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND = (
        1  #: as `VERTICAL_ONLY` plus horizontal shar correction
    )
    VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND = (
        2  #: as `VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND` plus shear form vertical velocity
    )
    VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND_LTHESH = 3  #: same as `VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND` but scaling of coarse-grid horizontal shear production term with 1/sqrt(Ri) (if LTKESH = TRUE)


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
        thslp_zdiffu: float = 0.025,
        thhgtd_zdiffu: float = 200.0,
        velocity_boundary_diffusion_denom: float = 200.0,
        temperature_boundary_diffusion_denom: float = 135.0,
        max_nudging_coeff: float = 0.02,
        nudging_decay_rate: float = 2.0,
        shear_type: TurbulenceShearForcingType = TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND,
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
        #: Called `hdiff_smag_fac` in mo_diffusion_nml.f90
        self.smagorinski_scaling_factor: float = smagorinski_scaling_factor

        #: If True, apply truly horizontal temperature diffusion over steep slopes
        #: Called 'l_zdiffu_t' in mo_nonhydrostatic_nml.f90
        self.apply_zdiffusion_t: bool = zdiffu_t

        #:slope threshold (temperature diffusion): is used to build up an index list for application of truly horizontal diffusion in mo_vertical_grid.f89
        self.thslp_zdiffu = thslp_zdiffu
        #: threshold [m] for height difference between adjacent grid points, defaults to 200m (temperature diffusion)
        self.thhgtd_zdiffu = thhgtd_zdiffu

        # from other namelists:
        # from parent namelist mo_nonhydrostatic_nml

        #: Number of dynamics substeps per fast-physics step
        #: Called 'ndyn_substeps' in mo_nonhydrostatic_nml.f90

        # TODO (magdalena) ndyn_substeps may dynamically increase during a model run in order to
        #       reduce instabilities. Need to figure out whether the parameter is the configured
        #       (constant!) one or the dynamical one. In the latter case it should be removed from
        #       DiffusionConfig and init()
        self.ndyn_substeps: int = n_substeps

        # namelist mo_gridref_nml.f90

        #: Denominator for temperature boundary diffusion
        #: Called 'denom_diffu_t' in mo_gridref_nml.f90
        self.temperature_boundary_diffusion_denominator: float = (
            temperature_boundary_diffusion_denom
        )

        #: Denominator for velocity boundary diffusion
        #: Called 'denom_diffu_v' in mo_gridref_nml.f90
        self.velocity_boundary_diffusion_denominator: float = velocity_boundary_diffusion_denom

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

        #: Type of shear forcing used in turbulence
        #: Called itype_shear in `mo_turbdiff_nml.f90
        self.shear_type = shear_type

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

        if self.shear_type not in (
            TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND,
            TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND,
        ):
            raise NotImplementedError(
                f"Turbulence Shear only {TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND} "
                f"and {TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND} "
                f"implemented"
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
            (1.0 / (config.hdiff_efdt_ratio * 8.0) if config.hdiff_efdt_ratio > 0.0 else 0.0),
        )
        object.__setattr__(self, "K4", self.K2 / 8.0)
        object.__setattr__(self, "K6", self.K2 / 64.0)
        object.__setattr__(
            self,
            "K4W",
            (1.0 / (config.hdiff_w_efdt_ratio * 36.0) if config.hdiff_w_efdt_ratio > 0 else 0.0),
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

    def __init__(self, exchange: ExchangeRuntime = SingleNodeExchange()):
        self._exchange = exchange
        self._initialized = False
        self.rd_o_cvd: float = GAS_CONSTANT_DRY_AIR / (CPD - GAS_CONSTANT_DRY_AIR)
        #: threshold temperature deviation from neighboring grid points hat activates extra diffusion against runaway cooling
        self.thresh_tdiff: float = -5.0
        self.grid: Optional[IconGrid] = None
        self.config: Optional[DiffusionConfig] = None
        self.params: Optional[DiffusionParams] = None
        self.vertical_params: Optional[VerticalModelParams] = None
        self.interpolation_state: DiffusionInterpolationState = None
        self.metric_state: DiffusionMetricState = None
        self.diff_multfac_w: Optional[float] = None
        self.diff_multfac_n2w: Field[[KDim], float] = None
        self.smag_offset: Optional[float] = None
        self.fac_bdydiff_v: Optional[float] = None
        self.bdy_diff: Optional[float] = None
        self.nudgezone_diff: Optional[float] = None
        self.edge_params: Optional[EdgeParams] = None
        self.cell_params: Optional[CellParams] = None
        self._horizontal_start_index_w_diffusion: int32 = 0

    def init(
        self,
        grid: IconGrid,
        config: DiffusionConfig,
        params: DiffusionParams,
        vertical_params: VerticalModelParams,
        metric_state: DiffusionMetricState,
        interpolation_state: DiffusionInterpolationState,
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
        self.metric_state: DiffusionMetricState = metric_state
        self.interpolation_state: DiffusionInterpolationState = interpolation_state
        self.edge_params = edge_params
        self.cell_params = cell_params

        self._allocate_temporary_fields()

        def _get_start_index_for_w_diffusion() -> int32:
            return self.grid.get_start_index(
                CellDim,
                (
                    HorizontalMarkerIndex.nudging(CellDim)
                    if self.grid.limited_area
                    else HorizontalMarkerIndex.interior(CellDim)
                ),
            )

        self.nudgezone_diff: float = 0.04 / (params.scaled_nudge_max_coeff + sys.float_info.epsilon)
        self.bdy_diff: float = 0.015 / (params.scaled_nudge_max_coeff + sys.float_info.epsilon)
        self.fac_bdydiff_v: float = (
            math.sqrt(config.substep_as_float) / config.velocity_boundary_diffusion_denominator
        )

        self.smag_offset: float = 0.25 * params.K4 * config.substep_as_float
        self.diff_multfac_w: float = min(1.0 / 48.0, params.K4W * config.substep_as_float)

        init_diffusion_local_fields_for_regular_timestep(
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

        # TODO (magdalena) port to gt4py?
        self.diff_multfac_n2w = init_nabla2_factor_in_upper_damping_zone(
            k_size=self.grid.num_levels,
            nshift=0,
            physical_heights=self.vertical_params.physical_heights,
            nrdmax=self.vertical_params.index_of_damping_layer,
        )
        self._horizontal_start_index_w_diffusion = _get_start_index_for_w_diffusion()
        self._initialized = True

    @property
    def initialized(self):
        return self._initialized

    def _allocate_temporary_fields(self):
        def _allocate(*dims: Dimension):
            return zero_field(self.grid, *dims)

        def _index_field(dim: Dimension, size=None):
            size = size if size else self.grid.size[dim]
            return as_field((dim,), xp.arange(size, dtype=int32))

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
        self.vertical_index = _index_field(KDim, self.grid.num_levels + 1)
        self.horizontal_cell_index = _index_field(CellDim)
        self.horizontal_edge_index = _index_field(EdgeDim)
        self.w_tmp = as_field(
            (CellDim, KDim), xp.zeros((self.grid.num_cells, self.grid.num_levels + 1), dtype=float)
        )

    def initial_run(
        self,
        diagnostic_state: DiffusionDiagnosticState,
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

        setup_fields_for_initial_step(
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
        self._sync_cell_fields(prognostic_state)

    def run(
        self,
        diagnostic_state: DiffusionDiagnosticState,
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

    def _sync_cell_fields(self, prognostic_state):
        """
        Communicate theta_v, exner and w.

        communication only done in original code if the following condition applies:
        IF ( linit .OR. (iforcing /= inwp .AND. iforcing /= iaes) ) THEN
        """
        log.debug("communication of prognostic cell fields: theta, w, exner - start")
        self._exchange.exchange_and_wait(
            CellDim,
            prognostic_state.w,
            prognostic_state.theta_v,
            prognostic_state.exner,
        )
        log.debug("communication of prognostic cell fields: theta, w, exner - done")

    def _do_diffusion_step(
        self,
        diagnostic_state: DiffusionDiagnosticState,
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
        klevels = self.grid.num_levels
        cell_start_interior = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.interior(CellDim)
        )
        cell_start_nudging = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.nudging(CellDim)
        )
        cell_end_local = self.grid.get_end_index(CellDim, HorizontalMarkerIndex.local(CellDim))
        cell_end_halo = self.grid.get_end_index(CellDim, HorizontalMarkerIndex.halo(CellDim))

        edge_start_nudging_plus_one = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim) + 1
        )
        edge_start_nudging = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim)
        )
        edge_start_lb_plus4 = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4
        )
        edge_end_local = self.grid.get_end_index(EdgeDim, HorizontalMarkerIndex.local(EdgeDim))
        edge_end_local_minus2 = self.grid.get_end_index(
            EdgeDim, HorizontalMarkerIndex.local(EdgeDim) - 2
        )
        edge_end_halo = self.grid.get_end_index(EdgeDim, HorizontalMarkerIndex.halo(EdgeDim))

        vertex_start_lb_plus1 = self.grid.get_start_index(
            VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1
        )
        vertex_end_local = self.grid.get_end_index(
            VertexDim, HorizontalMarkerIndex.local(VertexDim)
        )

        wait_on_comm_handle = mpi_decomposition.WaitOnComm(self._exchange) if isinstance(self._exchange, GHexMultiNodeExchange) else definitions.WaitOnComm()

        @dace_jit(self)
        def fuse():
            # dtime dependent: enh_smag_factor,
            scale_k(self.enh_smag_fac, dtime, self.diff_multfac_smag, offset_provider={})

            log.debug("rbf interpolation 1: start")
            mo_intp_rbf_rbf_vec_interpol_vertex(
                p_e_in=prognostic_state.vn,
                ptr_coeff_1=self.interpolation_state.rbf_coeff_1,
                ptr_coeff_2=self.interpolation_state.rbf_coeff_2,
                p_u_out=self.u_vert,
                p_v_out=self.v_vert,
                horizontal_start=vertex_start_lb_plus1,
                horizontal_end=vertex_end_local,
                vertical_start=0,
                vertical_end=klevels,
                offset_provider=self.grid.offset_providers,
            )
            log.debug("rbf interpolation 1: end")

            # 2.  HALO EXCHANGE -- CALL sync_patch_array_mult u_vert and v_vert
            log.debug("communication rbf extrapolation of vn - start")
            #self._exchange(self.u_vert, self.v_vert, dim=VertexDim, wait=True)
            log.debug("communication rbf extrapolation of vn - end")

            log.debug("running stencil 01(calculate_nabla2_and_smag_coefficients_for_vn): start")
            calculate_nabla2_and_smag_coefficients_for_vn(
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
                offset_provider=self.grid.offset_providers,
            )
            log.debug("running stencil 01 (calculate_nabla2_and_smag_coefficients_for_vn): end")

            if (
                self.config.shear_type
                >= TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND
            ):
                log.debug(
                    "running stencils 02 03 (calculate_diagnostic_quantities_for_turbulence): start"
                )
                calculate_diagnostic_quantities_for_turbulence(
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
                    offset_provider=self.grid.offset_providers,
                )
                log.debug(
                    "running stencils 02 03 (calculate_diagnostic_quantities_for_turbulence): end"
                )

            # HALO EXCHANGE  IF (discr_vn > 1) THEN CALL sync_patch_array
            # TODO (magdalena) move this up and do asynchronous exchange
            if self.config.type_vn_diffu > 1:
                log.debug("communication rbf extrapolation of z_nable2_e - start")
                #self._exchange(self.z_nabla2_e, dim=EdgeDim, wait=True)
                log.debug("communication rbf extrapolation of z_nable2_e - end")

            log.debug("2nd rbf interpolation: start")
            mo_intp_rbf_rbf_vec_interpol_vertex(
                p_e_in=self.z_nabla2_e,
                ptr_coeff_1=self.interpolation_state.rbf_coeff_1,
                ptr_coeff_2=self.interpolation_state.rbf_coeff_2,
                p_u_out=self.u_vert,
                p_v_out=self.v_vert,
                horizontal_start=vertex_start_lb_plus1,
                horizontal_end=vertex_end_local,
                vertical_start=0,
                vertical_end=klevels,
                offset_provider=self.grid.offset_providers,
            )
            log.debug("2nd rbf interpolation: end")

            # 6.  HALO EXCHANGE -- CALL sync_patch_array_mult (Vertex Fields)
            log.debug("communication rbf extrapolation of z_nable2_e - start")
            #self._exchange(self.u_vert, self.v_vert, dim=VertexDim, wait=True)
            log.debug("communication rbf extrapolation of z_nable2_e - end")

            log.debug("running stencils 04 05 06 (apply_diffusion_to_vn): start")
            apply_diffusion_to_vn(
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
                edge=self.horizontal_edge_index,
                nudgezone_diff=self.nudgezone_diff,
                fac_bdydiff_v=self.fac_bdydiff_v,
                start_2nd_nudge_line_idx_e=int32(edge_start_nudging_plus_one),
                limited_area=self.grid.limited_area,
                horizontal_start=edge_start_lb_plus4,
                horizontal_end=edge_end_local,
                vertical_start=0,
                vertical_end=klevels,
                offset_provider=self.grid.offset_providers,
            )
            log.debug("running stencils 04 05 06 (apply_diffusion_to_vn): end")

            log.debug("communication of prognistic.vn : start")
            #handle_edge_comm = self._exchange(prognostic_state.vn, dim=EdgeDim, wait=False)

            log.debug(
                "running stencils 07 08 09 10 (apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence): start"
            )
            # TODO (magdalena) get rid of this copying. So far passing an empty buffer instead did not verify?
            copy_field(prognostic_state.w, self.w_tmp, offset_provider={})
            apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence(
                area=self.cell_params.area,
                geofac_n2s=self.interpolation_state.geofac_n2s,
                geofac_grg_x=self.interpolation_state.geofac_grg_x,
                geofac_grg_y=self.interpolation_state.geofac_grg_y,
                w_old=self.w_tmp,
                w=prognostic_state.w,
                type_shear=int32(self.config.shear_type.value),
                dwdx=diagnostic_state.dwdx,
                dwdy=diagnostic_state.dwdy,
                diff_multfac_w=self.diff_multfac_w,
                diff_multfac_n2w=self.diff_multfac_n2w,
                k=self.vertical_index,
                cell=self.horizontal_cell_index,
                nrdmax=int32(
                    self.vertical_params.index_of_damping_layer + 1
                ),  # +1 since Fortran includes boundaries
                interior_idx=int32(cell_start_interior),
                halo_idx=int32(cell_end_local),
                horizontal_start=self._horizontal_start_index_w_diffusion,
                horizontal_end=cell_end_halo,
                vertical_start=0,
                vertical_end=klevels,
                offset_provider=self.grid.offset_providers,
            )
            log.debug(
                "running stencils 07 08 09 10 (apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence): end"
            )

            log.debug(
                "running fused stencils 11 12 (calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools): start"
            )
            calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools(
                theta_v=prognostic_state.theta_v,
                theta_ref_mc=self.metric_state.theta_ref_mc,
                thresh_tdiff=self.thresh_tdiff,
                smallest_vpfloat=dbl_eps,
                kh_smag_e=self.kh_smag_e,
                horizontal_start=edge_start_nudging,
                horizontal_end=edge_end_halo,
                vertical_start=(klevels - 2),
                vertical_end=klevels,
                offset_provider=self.grid.offset_providers,
            )
            log.debug(
                "running stencils 11 12 (calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools): end"
            )

            log.debug("running stencils 13 14 (calculate_nabla2_for_theta): start")
            calculate_nabla2_for_theta(
                kh_smag_e=self.kh_smag_e,
                inv_dual_edge_length=self.edge_params.inverse_dual_edge_lengths,
                theta_v=prognostic_state.theta_v,
                geofac_div=self.interpolation_state.geofac_div,
                z_temp=self.z_temp,
                horizontal_start=cell_start_nudging,
                horizontal_end=cell_end_local,
                vertical_start=0,
                vertical_end=klevels,
                offset_provider=self.grid.offset_providers,
            )
            log.debug("running stencils 13_14 (calculate_nabla2_for_theta): end")

            log.debug(
                "running stencil 15 (truly_horizontal_diffusion_nabla_of_theta_over_steep_points): start"
            )
            if self.config.apply_zdiffusion_t:
                truly_horizontal_diffusion_nabla_of_theta_over_steep_points(
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
                    offset_provider=self.grid.offset_providers,
                )

                log.debug(
                    "running fused stencil 15 (truly_horizontal_diffusion_nabla_of_theta_over_steep_points): end"
                )
            
            log.debug("running stencil 16 (update_theta_and_exner): start")
            update_theta_and_exner(
                z_temp=self.z_temp,
                area=self.cell_params.area,
                theta_v=prognostic_state.theta_v,
                exner=prognostic_state.exner,
                rd_o_cvd=self.rd_o_cvd,
                horizontal_start=cell_start_nudging,
                horizontal_end=cell_end_local,
                vertical_start=0,
                vertical_end=klevels,
                offset_provider={},
            )
            log.debug("running stencil 16 (update_theta_and_exner): end")
            
            #dummy = wait_on_comm_handle(handle_edge_comm)  # need to do this here, since we currently only use 1 communication object.
            log.debug("communication of prognogistic.vn - end")
        
            #return dummy

        fuse()

def dace_jit(self):
    def decorator(fuse_func):
        def wrapper(*args, **kwargs):
            if backend == run_dace_cpu_noopt:
                
                kwargs.update({
                                # GHEX C++ ptrs
                                "__context_ptr":ghex.expose_cpp_ptr(self._exchange._context) if isinstance(self._exchange, GHexMultiNodeExchange) else 0,
                                "__comm_ptr":ghex.expose_cpp_ptr(self._exchange._comm) if isinstance(self._exchange, GHexMultiNodeExchange) else 0,
                                **{f"__pattern_{dim.value}Dim_ptr":ghex.expose_cpp_ptr(self._exchange._patterns[dim]) if isinstance(self._exchange, GHexMultiNodeExchange) else 0 for dim in (CellDim, VertexDim, EdgeDim)},
                                **{f"__domain_descriptor_{dim.value}Dim_ptr":ghex.expose_cpp_ptr(self._exchange._domain_descriptors[dim]) if isinstance(self._exchange, GHexMultiNodeExchange) else 0 for dim in (CellDim, VertexDim, EdgeDim)},
                                # offset providers
                                **{f"__connectivity_{k}":v.table for k,v in self.grid.offset_providers.items() if hasattr(v, "table")},
                                #
                                **{f"__gids_{ind.name}_{dim.value}":self._exchange._decomposition_info.global_index(dim, ind) if isinstance(self._exchange, GHexMultiNodeExchange) else np.empty(1, dtype=np.int64) for ind in (di.EntryType.ALL, di.EntryType.OWNED, di.EntryType.HALO) for dim in (CellDim, VertexDim, EdgeDim)},
                              })
                
                with dace.config.temporary_config():
                    dace.config.Config.set("compiler", "build_type", value="RelWithDebInfo")
                    dace.config.Config.set("compiler", "allow_view_arguments", value=True)
                    dace.config.Config.set("frontend", "check_args", value=True)
                    compiler_args ="-std=c++17 -fPIC -Wall -Wextra -O3 -march=native -ffast-math -Wno-unused-parameter -Wno-unused-label -fno-finite-math-only"
                    on_gpu = False
                    dace.config.Config.set("compiler", "cuda" if on_gpu else "cpu", "args", value=compiler_args)

                    dace.config.Config.set("optimizer", "automatic_simplification", value=False)
                    dace.config.Config.set("cache", value="unique")

                    daceP = dace.program(recreate_sdfg=False, regenerate_code=False, recompile=False, distributed_compilation=False)(fuse_func)

                    ################################################################################
                    # Copy of the __call__ function of DaceProgram class
                    # Expose the generated SDFG and modify it
                    ################################################################################

                    # Update global variables with current closure
                    daceP.global_vars = dace.frontend.python.parser._get_locals_and_globals(daceP.f)

                    # Move "self" from an argument into the closure
                    if daceP.methodobj is not None:
                        daceP.global_vars[daceP.objname] = daceP.methodobj

                    argtypes, arg_mapping, constant_args, specified = daceP._get_type_annotations(args, kwargs)

                    # Add constant arguments to globals for caching
                    daceP.global_vars.update(constant_args)

                    # Cache key
                    cachekey = daceP._cache.make_key(argtypes, specified, daceP.closure_array_keys, daceP.closure_constant_keys,
                                                    constant_args)

                    if daceP._cache.has(cachekey):
                        entry = daceP._cache.get(cachekey)
                        # If the cache does not just contain a parsed SDFG
                        if entry.compiled_sdfg is not None:
                            kwargs.update(arg_mapping)
                            entry.compiled_sdfg.clear_return_values()
                            return entry.compiled_sdfg(**daceP._create_sdfg_args(entry.sdfg, args, kwargs))

                    # Clear cache to enforce deletion and closure of compiled program
                    # daceP._cache.pop()

                    # Parse SDFG
                    sdfg = daceP._parse(args, kwargs)

                    if isinstance(self._exchange, GHexMultiNodeExchange):
                        counter = 0
                        for nested_sdfg in sdfg.all_sdfgs_recursive():
                            if not hasattr(nested_sdfg, "GT4Py_Program_output_fields"):
                                continue
                            
                            field_dims = set()
                            for buffer_name, dims in nested_sdfg.GT4Py_Program_output_fields.items():
                                for dim in dims:
                                    if dim.kind == DimensionKind.HORIZONTAL:
                                        field_dims.add(dim)

                            if len(field_dims) > 1:
                                raise ValueError("The output fields to be communicated are not defined on the same dimension kind.")

                            dim = {'Cell':CellDim, 'Edge':EdgeDim, 'Vertex':VertexDim}[list(field_dims)[0].value] if len(field_dims) == 1 else None
                            if not dim:
                                continue
                            wait = True

                            for sdfg_state in sdfg.states():
                                if sdfg_state.label == nested_sdfg.parent.label:
                                    break
                            state = sdfg.add_state_after(sdfg_state, label='_halo_exchange_')

                            if counter == 0:
                                for buffer_name in kwargs:
                                    if '_ptr' in buffer_name:
                                        sdfg.add_scalar(buffer_name, dtype=dace.uintp)
                                
                            tasklet = dace.sdfg.nodes.Tasklet('_halo_exchange_',
                                                            inputs=None,
                                                            outputs=None,
                                                            code='',
                                                            language=dace.dtypes.Language.CPP,
                                                            side_effects=True,)
                            state.add_node(tasklet)

                            in_connectors = {}
                            out_connectors = {}

                            global_buffer_descriptor = []
                            for i, buffer_name in enumerate(nested_sdfg.GT4Py_Program_output_fields):
                                data_descriptor = nested_sdfg.arrays[buffer_name]

                                global_buffer_name = None
                                for edge in sdfg_state.all_edges_recursive():
                                    if hasattr(edge[0], "src_conn") and (edge[0].src_conn == buffer_name):
                                        global_buffer_name = edge[0].dst.label
                                        break

                                if not global_buffer_name:
                                    raise ValueError("Could not link the local buffer_name to the global one (coming from the closure).")

                                global_buffer_descriptor.append(sdfg.arrays[global_buffer_name])

                                buffer = state.add_read(global_buffer_name)
                                in_connectors['IN_' + f'field_{i}'] = dtypes.pointer(data_descriptor.dtype)
                                state.add_edge(buffer, None, tasklet, 'IN_' + f'field_{i}', Memlet.from_array(global_buffer_name, data_descriptor))

                                update = state.add_write(global_buffer_name)
                                out_connectors['OUT_' + f'field_{i}'] = dtypes.pointer(data_descriptor.dtype)
                                state.add_edge(tasklet, 'OUT_' + f'field_{i}', update, None, Memlet.from_array(global_buffer_name, data_descriptor))

                            if counter == 0:
                                for buffer_name in kwargs:
                                    if '_ptr' in buffer_name:
                                        buffer = state.add_read(buffer_name)
                                        data_descriptor = dace.uintp
                                        in_connectors['IN_' + buffer_name] = data_descriptor.dtype
                                        memlet_ =  Memlet(buffer_name, subset='0')
                                        state.add_edge(buffer, None, tasklet, 'IN_' + buffer_name, memlet_)


                            tasklet.in_connectors = in_connectors
                            tasklet.out_connectors = out_connectors
                            tasklet.environments = ['icon4py.model.common.decomposition.mpi_decomposition.DaceGHEX']

                            pattern_type = self._exchange._patterns[dim].__cpp_type__
                            domain_descriptor_type = self._exchange._domain_descriptors[dim].__cpp_type__
                            communication_object_type = self._exchange._comm.__cpp_type__
                            communication_handle_type = communication_object_type[communication_object_type.find('<')+1:communication_object_type.rfind('>')]

                            fields_desc_glob_vars = '\n'
                            fields_desc = '\n'
                            descr_unique_names = []
                            for i, arg in enumerate(global_buffer_descriptor):                                
                                # https://github.com/ghex-org/GHEX/blob/master/bindings/python/src/_pyghex/unstructured/field_descriptor.cpp
                                if len(arg.shape) > 2:
                                    raise ValueError("field has too many dimensions")
                                if arg.shape[0] != self._exchange._domain_descriptors[dim].size():
                                    raise ValueError("field's first dimension must match the size of the domain")
                                
                                levels_first = True
                                outer_strides = 0
                                # DaCe strides: number of elements to jump
                                # GHEX/NumPy strides: number of bytes to jump
                                if len(arg.shape) == 2 and arg.strides[1] != 1:
                                    levels_first = False
                                    if arg.strides[0] != 1:
                                        raise ValueError("field's strides are not compatible with GHEX")
                                    outer_strides = arg.strides[1]
                                elif len(arg.shape) == 2:
                                    if arg.strides[1] != 1:
                                        raise ValueError("field's strides are not compatible with GHEX")
                                    outer_strides = arg.strides[0]
                                else:
                                    if arg.strides[0] != 1:
                                        raise ValueError("field's strides are not compatible with GHEX")

                                levels = 1 if len(arg.shape) == 1 else arg.shape[1]

                                device = 'cpu' if arg.storage.value <= 5 else 'gpu'
                                field_dtype = arg.dtype.ctype
                                
                                descr_unique_name = f'field_desc_{i}_{counter}_{id(self)}'
                                descr_unique_names.append(descr_unique_name)
                                descr_type_ = f"ghex::unstructured::data_descriptor<ghex::{device}, int, int, {field_dtype}>"
                                if wait:
                                    # de-allocated descriptors once out-of-scope, no need for storing them in global vars
                                    fields_desc += f"{descr_type_} {descr_unique_name}{{*domain_descriptor, IN_field_{i}, {levels}, {'true' if levels_first else 'false'}, {outer_strides}}};\n"
                                else:
                                    # for async exchange, we need to keep the descriptors alive, until the wait
                                    fields_desc += f"{descr_unique_name} = {descr_type_}{{*domain_descriptor, IN_field_{i}, {levels}, {'true' if levels_first else 'false'}, {outer_strides}}};\n"
                                    fields_desc_glob_vars += f"{descr_type_} {descr_unique_name};\n"

                            code = ''
                            if counter == 0:
                                __pattern = ''
                                __domain_descriptor = ''
                                for dim_ in (CellDim, VertexDim, EdgeDim):
                                    __pattern += f"__pattern_{dim_.value}Dim_ptr_{id(self)} = IN___pattern_{dim_.value}Dim_ptr;\n"
                                    __domain_descriptor += f"__domain_descriptor_{dim_.value}Dim_ptr_{id(self)} = IN___domain_descriptor_{dim_.value}Dim_ptr;\n"
                                    
                                code = f'''
                                    __context_ptr_{id(self)} = IN___context_ptr;
                                    __comm_ptr_{id(self)} = IN___comm_ptr;
                                    {__pattern}
                                    {__domain_descriptor}
                                    '''

                            code += f'''
                                    ghex::context* m = reinterpret_cast<ghex::context*>(__context_ptr_{id(self)});
                                    
                                    {pattern_type}* pattern = reinterpret_cast<{pattern_type}*>(__pattern_{dim.value}Dim_ptr_{id(self)});
                                    {domain_descriptor_type}* domain_descriptor = reinterpret_cast<{domain_descriptor_type}*>(__domain_descriptor_{dim.value}Dim_ptr_{id(self)});
                                    {communication_object_type}* communication_object = reinterpret_cast<{communication_object_type}*>(__comm_ptr_{id(self)});

                                    {fields_desc}

                                    h_{id(self)} = communication_object->exchange({", ".join([f'(*pattern)({descr_unique_names[i]})' for i in range(len(global_buffer_descriptor))])});
                                    { 'h_'+str(id(self))+'.wait();' if wait else ''}
                                    '''

                            tasklet.code = CodeBlock(code=code, language=dace.dtypes.Language.CPP)
                            if counter == 0:
                                __pattern = ''
                                __domain_descriptor = ''
                                for dim_ in (CellDim, VertexDim, EdgeDim):
                                    __pattern += f"{dace.uintp.dtype} __pattern_{dim_.value}Dim_ptr_{id(self)};\n"
                                    __domain_descriptor += f"{dace.uintp.dtype} __domain_descriptor_{dim_.value}Dim_ptr_{id(self)};\n"
                                
                                code = f'''
                                        {dace.uintp.dtype} __context_ptr_{id(self)};
                                        {dace.uintp.dtype} __comm_ptr_{id(self)};
                                        {__pattern}
                                        {__domain_descriptor}
                                        {fields_desc_glob_vars}
                                        ghex::communication_handle<{communication_handle_type}> h_{id(self)};
                                        '''
                            else:
                                code = fields_desc_glob_vars
                            tasklet.code_global = CodeBlock(code=code, language=dace.dtypes.Language.CPP)

                            counter += 1

                    # Add named arguments to the call
                    kwargs.update(arg_mapping)
                    sdfg_args = daceP._create_sdfg_args(sdfg, args, kwargs)

                    if daceP.recreate_sdfg:
                        # Invoke auto-optimization as necessary
                        if Config.get_bool('optimizer', 'autooptimize') or daceP.autoopt:
                            sdfg = daceP.auto_optimize(sdfg, symbols=sdfg_args)
                            sdfg.simplify()

                    with hooks.invoke_sdfg_call_hooks(sdfg) as sdfg:
                        if daceP.distributed_compilation and mpi4py:
                            binaryobj = distributed_compile(sdfg, mpi4py.MPI.COMM_WORLD, validate=daceP.validate)
                        else:
                            # Compile SDFG (note: this is done after symbol inference due to shape
                            # altering transformations such as Vectorization)
                            binaryobj = sdfg.compile(validate=daceP.validate)

                        # Recreate key and add to cache
                        cachekey = daceP._cache.make_key(argtypes, specified, daceP.closure_array_keys, daceP.closure_constant_keys,
                                                        constant_args)
                        daceP._cache.add(cachekey, sdfg, binaryobj)

                        # Call SDFG
                        result = binaryobj(**sdfg_args)

                    return result
            else:
                fuse_func(*args, **kwargs)
        return wrapper
    return decorator
