# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import enum
import functools
import logging
import math
import sys
from typing import Final, Optional

import gt4py.next as gtx

import icon4py.model.common.states.prognostic_state as prognostics
from icon4py.model.atmosphere.diffusion import diffusion_utils, diffusion_states
from icon4py.model.atmosphere.diffusion.diffusion_utils import (
    copy_field,
    init_diffusion_local_fields_for_regular_timestep,
    scale_k,
    setup_fields_for_initial_step,
)
from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_vn import (
    apply_diffusion_to_vn,
)
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
from icon4py.model.common import field_type_aliases as fa, constants, dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import (
    horizontal as h_grid,
    vertical as v_grid,
    icon as icon_grid,
    geometry,
)
from icon4py.model.common.interpolation.stencils.mo_intp_rbf_rbf_vec_interpol_vertex import (
    mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.model.common.settings import xp
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc

"""
Diffusion module ported from ICON mo_nh_diffusion.f90.

Supports only diffusion_type (=hdiff_order) 5 from the diffusion namelist.
"""

# flake8: noqa
log = logging.getLogger(__name__)


class DiffusionType(enum.Enum):
    """
    Order of nabla operator for diffusion.

    Note: Called `hdiff_order` in `mo_diffusion_nml.f90`.
    Note: We currently only support type 5.
    """

    #: no diffusion
    NO_DIFFUSION = -1

    #: 2nd order linear diffusion on all vertical levels
    LINEAR_2ND_ORDER = 2

    #: Smagorinsky diffusion without background diffusion
    SMAGORINSKY_NO_BACKGROUND = 3

    #: 4th order linear diffusion on all vertical levels
    LINEAR_4TH_ORDER = 4

    #: Smagorinsky diffusion with fourth-order background diffusion
    SMAGORINSKY_4TH_ORDER = 5


class TurbulenceShearForcingType(enum.Enum):
    """
    Type of shear forcing used in turbulance.

    Note: called `itype_sher` in `mo_turbdiff_nml.f90`
    """

    #: only vertical shear of horizontal wind
    VERTICAL_OF_HORIZONTAL_WIND = 0

    #: as `VERTICAL_ONLY` plus horizontal shar correction
    VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND = 1

    #: as `VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND` plus shear form vertical velocity
    VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND = 2

    #: same as `VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND` but scaling of coarse-grid horizontal shear production term with 1/sqrt(Ri) (if LTKESH = TRUE)
    VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND_LTHESH = 3


@dataclasses.dataclass(frozen=True)
class DiffusionConfig:
    """
    Contains necessary parameter to configure a diffusion run.

    Encapsulates namelist parameters and derived parameters.
    Values should be read from configuration.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.

     # TODO(Magdalena):  handle dependencies on other namelists (see below...)
    """

    diffusion_type: DiffusionType = DiffusionType.SMAGORINSKY_4TH_ORDER

    #: If True, apply diffusion on the vertical wind field
    #: Called `lhdiff_w` in mo_diffusion_nml.f90
    apply_to_vertical_wind: bool = True

    #: True apply diffusion on the horizontal wind field, is ONLY used in mo_nh_stepping.f90
    #: Called `lhdiff_vn` in mo_diffusion_nml.f90
    apply_to_horizontal_wind: bool = True

    #:  If True, apply horizontal diffusion to temperature field
    #: Called `lhdiff_temp` in mo_diffusion_nml.f90
    apply_to_temperature: bool = True

    #: Options for discretizing the Smagorinsky momentum diffusion
    #: Called `itype_vn_diffu` in mo_diffusion_nml.f90
    type_vn_diffu: int = 1

    #: If True, compute 3D Smagorinsky diffusion coefficient
    #: Called `lsmag_3d` in mo_diffusion_nml.f90
    compute_3d_smag_coeff: bool = False

    #: Options for discretizing the Smagorinsky temperature diffusion
    #: Called `itype_t_diffu` inmo_diffusion_nml.f90
    type_t_diffu: int = 2

    #: Ratio of e-folding time to (2*)time step
    #: Called `hdiff_efdt_ratio` inmo_diffusion_nml.f90
    hdiff_efdt_ratio: float = 36.0

    #: Ratio of e-folding time to time step for w diffusion (NH only)
    #: Called `hdiff_w_efdt_ratio` inmo_diffusion_nml.f90.
    hdiff_w_efdt_ratio: float = 15.0

    #: Scaling factor for Smagorinsky diffusion at height hdiff_smag_z and below
    #: Called `hdiff_smag_fac` in mo_diffusion_nml.f90
    smagorinski_scaling_factor: float = 0.015

    #: Number of dynamics substeps per fast-physics step
    #: Called 'ndyn_substeps' in mo_nonhydrostatic_nml.f90
    # TODO (magdalena) ndyn_substeps may dynamically increase during a model run in order to
    #       reduce instabilities. Need to figure out whether the parameter is the configured
    #       (constant!) one or the dynamical one. In the latter case it should be removed from
    #       DiffusionConfig and init()
    ndyn_substeps: int = 5

    #: If True, apply truly horizontal temperature diffusion over steep slopes
    #: Called 'l_zdiffu_t' in mo_nonhydrostatic_nml.f90
    apply_zdiffusion_t: bool = True

    #:slope threshold (temperature diffusion): is used to build up an index list for application of truly horizontal diffusion in mo_vertical_grid.f90
    thslp_zdiffu: float = 0.025

    #: threshold [m] for height difference between adjacent grid points, defaults to 200m (temperature diffusion)
    thhgtd_zdiffu: float = 200.0

    #: Denominator for velocity boundary diffusion
    #: Called 'denom_diffu_v' in mo_gridref_nml.f90
    velocity_boundary_diffusion_denominator: float = 200.0

    #: Denominator for temperature boundary diffusion
    #: Called 'denom_diffu_t' in mo_gridref_nml.f90
    temperature_boundary_diffusion_denominator: float = 135.0

    #: Parameter describing the lateral boundary nudging in limited area mode.
    #:
    #: Maximal value of the nudging coefficients used cell row bordering the boundary interpolation zone,
    #: from there nudging coefficients decay exponentially with `nudge_efold_width` in units of cell rows.
    #: Called `nudge_max_coeff` in mo_interpol_nml.f90
    max_nudging_coefficient: float = 0.02

    #: Exponential decay rate (in units of cell rows) of the lateral boundary nudging coefficients
    #: Called `nudge_efold_width` in mo_interpol_nml.f90
    nudge_efold_width: float = 2.0

    #: Type of shear forcing used in turbulence
    #: Called itype_shear in `mo_turbdiff_nml.f90
    shear_type: TurbulenceShearForcingType = TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Apply consistency checks and validation on configuration parameters."""
        if self.diffusion_type != DiffusionType.SMAGORINSKY_4TH_ORDER:
            raise NotImplementedError(
                "Only diffusion type 5 = `Smagorinsky diffusion with fourth-order background "
                "diffusion` is implemented"
            )
        if self.diffusion_type == DiffusionType.NO_DIFFUSION:
            assert (
                not self.apply_to_temperature
                and not self.apply_to_horizontal_wind
                and not self.apply_to_vertical_wind
            ), f"Inconsistent configuration: DiffusionType = {self.diffusion_type} - but application flags are set to:  horizontal wind = '{self.apply_to_horizontal_wind}', vertical wind = '{self.apply_to_vertical_wind}', temperature = '{self.apply_to_temperature}' "

        if self.shear_type not in (
            TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND,
            TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND,
        ):
            raise NotImplementedError(
                f"Turbulence Shear: only {TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND} "
                f"and {TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND} "
                f"implemented"
            )

    @functools.cached_property
    def substep_as_float(self):
        return float(self.ndyn_substeps)


@dataclasses.dataclass(frozen=True)
class DiffusionParams:
    """Calculates derived quantities depending on the diffusion config."""

    config: dataclasses.InitVar[DiffusionConfig]
    K2: Final[float] = dataclasses.field(init=False)
    K4: Final[float] = dataclasses.field(init=False)
    K6: Final[float] = dataclasses.field(init=False)
    K4W: Final[float] = dataclasses.field(init=False)
    smagorinski_factor: Final[float] = dataclasses.field(init=False)
    smagorinski_height: Final[float] = dataclasses.field(init=False)
    scaled_nudge_max_coeff: Final[float] = dataclasses.field(init=False)

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
            config.max_nudging_coefficient * constants.DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO,
        )

    def _determine_smagorinski_factor(self, config: DiffusionConfig):
        """Enhanced Smagorinsky diffusion factor.

        Smagorinsky diffusion factor is defined as a profile in height
        above sea level with 4 height sections.

        It is calculated/used only in the case of diffusion_type 3 or 5
        """
        match config.diffusion_type:
            case DiffusionType.SMAGORINSKY_4TH_ORDER:
                (
                    smagorinski_factor,
                    smagorinski_height,
                ) = diffusion_type_5_smagorinski_factor(config)
            case DiffusionType.LINEAR_4TH_ORDER:
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

    def __init__(
        self, exchange: decomposition.ExchangeRuntime = decomposition.SingleNodeExchange()
    ):
        self._exchange = exchange
        self._initialized = False
        self.rd_o_cvd: float = constants.GAS_CONSTANT_DRY_AIR / (
            constants.CPD - constants.GAS_CONSTANT_DRY_AIR
        )
        #: threshold temperature deviation from neighboring grid points hat activates extra diffusion against runaway cooling
        self.thresh_tdiff: float = -5.0
        self.grid: Optional[icon_grid.IconGrid] = None
        self.config: Optional[DiffusionConfig] = None
        self.params: Optional[DiffusionParams] = None
        self.vertical_grid: Optional[v_grid.VerticalGrid] = None
        self.interpolation_state: diffusion_states.DiffusionInterpolationState = None
        self.metric_state: diffusion_states.DiffusionMetricState = None
        self.diff_multfac_w: Optional[float] = None
        self.diff_multfac_n2w: fa.KField[float] = None
        self.smag_offset: Optional[float] = None
        self.fac_bdydiff_v: Optional[float] = None
        self.bdy_diff: Optional[float] = None
        self.nudgezone_diff: Optional[float] = None
        self.edge_params: Optional[geometry.EdgeParams] = None
        self.cell_params: Optional[geometry.CellParams] = None
        self._horizontal_start_index_w_diffusion: gtx.int32 = gtx.int32(0)

    def init(
        self,
        grid: icon_grid.IconGrid,
        config: DiffusionConfig,
        params: DiffusionParams,
        vertical_grid: v_grid.VerticalGrid,
        metric_state: diffusion_states.DiffusionMetricState,
        interpolation_state: diffusion_states.DiffusionInterpolationState,
        edge_params: geometry.EdgeParams,
        cell_params: geometry.CellParams,
    ):
        """
        Initialize Diffusion granule with configuration.

        calculates all local fields that are used in diffusion within the time loop.

        Args:
            grid:
            config:
            params:
            vertical_grid:
            metric_state:
            interpolation_state:
            edge_params:
            cell_params:
        """
        self.config: DiffusionConfig = config
        self.params: DiffusionParams = params
        self.grid = grid
        self.vertical_grid = vertical_grid
        self.metric_state: diffusion_states.DiffusionMetricState = metric_state
        self.interpolation_state: diffusion_states.DiffusionInterpolationState = interpolation_state
        self.edge_params = edge_params
        self.cell_params = cell_params

        self._allocate_temporary_fields()

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
            self.vertical_grid.interface_physical_height,
            self.diff_multfac_vn,
            self.smag_limit,
            self.enh_smag_fac,
            offset_provider={"Koff": dims.KDim},
        )

        diffusion_utils._init_nabla2_factor_in_upper_damping_zone(
            physical_heights=self.vertical_grid.interface_physical_height,
            k_field=self.vertical_index,
            nrdmax=self.vertical_grid.end_index_of_damping_layer,
            nshift=0,
            heights_nrd_shift=self.vertical_grid.interface_physical_height.ndarray[
                self.vertical_grid.end_index_of_damping_layer + 1
            ].item(),
            heights_1=self.vertical_grid.interface_physical_height.ndarray[1].item(),
            domain={dims.KDim: (1, self.vertical_grid.end_index_of_damping_layer + 1)},
            out=self.diff_multfac_n2w,
            offset_provider={},
        )

        self._determine_horizontal_domains()

        self._initialized = True

    @property
    def initialized(self):
        return self._initialized

    def _allocate_temporary_fields(self):
        self.diff_multfac_vn = field_alloc.allocate_zero_field(dims.KDim, grid=self.grid)
        self.diff_multfac_n2w = field_alloc.allocate_zero_field(dims.KDim, grid=self.grid)
        self.smag_limit = field_alloc.allocate_zero_field(dims.KDim, grid=self.grid)
        self.enh_smag_fac = field_alloc.allocate_zero_field(dims.KDim, grid=self.grid)
        self.u_vert = field_alloc.allocate_zero_field(dims.VertexDim, dims.KDim, grid=self.grid)
        self.v_vert = field_alloc.allocate_zero_field(dims.VertexDim, dims.KDim, grid=self.grid)
        self.kh_smag_e = field_alloc.allocate_zero_field(dims.EdgeDim, dims.KDim, grid=self.grid)
        self.kh_smag_ec = field_alloc.allocate_zero_field(dims.EdgeDim, dims.KDim, grid=self.grid)
        self.z_nabla2_e = field_alloc.allocate_zero_field(dims.EdgeDim, dims.KDim, grid=self.grid)
        self.z_temp = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=self.grid)
        self.diff_multfac_smag = field_alloc.allocate_zero_field(dims.KDim, grid=self.grid)
        # TODO(Magdalena): this is KHalfDim
        self.vertical_index = field_alloc.allocate_indices(
            dims.KDim, grid=self.grid, is_halfdim=True
        )
        self.horizontal_cell_index = field_alloc.allocate_indices(dims.CellDim, grid=self.grid)
        self.horizontal_edge_index = field_alloc.allocate_indices(dims.EdgeDim, grid=self.grid)
        self.w_tmp = gtx.as_field(
            (dims.CellDim, dims.KDim),
            xp.zeros((self.grid.num_cells, self.grid.num_levels + 1), dtype=float),
        )

    def _determine_horizontal_domains(self):
        cell_domain = h_grid.domain(dims.CellDim)
        edge_domain = h_grid.domain(dims.EdgeDim)
        vertex_domain = h_grid.domain(dims.VertexDim)

        def _get_start_index_for_w_diffusion() -> gtx.int32:
            return (
                self.grid.start_index(cell_domain(h_grid.Zone.NUDGING))
                if self.grid.limited_area
                else self.grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
            )

        self._cell_start_interior = self.grid.start_index(cell_domain(h_grid.Zone.INTERIOR))
        self._cell_start_nudging = self.grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._cell_end_local = self.grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._cell_end_halo = self.grid.end_index(cell_domain(h_grid.Zone.HALO))

        self._edge_start_lateral_boundary_level_5 = self.grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._edge_start_nudging = self.grid.start_index(edge_domain(h_grid.Zone.NUDGING))
        self._edge_start_nudging_level_2 = self.grid.start_index(
            edge_domain(h_grid.Zone.NUDGING_LEVEL_2)
        )
        self._edge_end_local = self.grid.end_index(edge_domain(h_grid.Zone.LOCAL))
        self._edge_end_halo = self.grid.end_index(edge_domain(h_grid.Zone.HALO))
        self._edge_end_halo_level_2 = self.grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))

        self._vertex_start_lateral_boundary_level_2 = self.grid.start_index(
            vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._vertex_end_local = self.grid.end_index(vertex_domain(h_grid.Zone.LOCAL))

        self._horizontal_start_index_w_diffusion = _get_start_index_for_w_diffusion()

    def initial_run(
        self,
        diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        prognostic_state: prognostics.PrognosticState,
        dtime: float,
    ):
        """
        Calculate initial diffusion step.

        In ICON at the start of the simulation diffusion is run with a parameter linit = True:

        'For real-data runs, perform an extra diffusion call before the first time
        step because no other filtering of the interpolated velocity field is done'

        This run uses special values for diff_multfac_vn, smag_limit and smag_offset

        """
        diff_multfac_vn = field_alloc.allocate_zero_field(dims.KDim, grid=self.grid)
        smag_limit = field_alloc.allocate_zero_field(dims.KDim, grid=self.grid)

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
        diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        prognostic_state: prognostics.PrognosticState,
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
            dims.CellDim,
            prognostic_state.w,
            prognostic_state.theta_v,
            prognostic_state.exner,
        )
        log.debug("communication of prognostic cell fields: theta, w, exner - done")

    def _do_diffusion_step(
        self,
        diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        prognostic_state: prognostics.PrognosticState,
        dtime: float,
        diff_multfac_vn: fa.KField[float],
        smag_limit: fa.KField[float],
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
        num_levels = self.grid.num_levels
        # dtime dependent: enh_smag_factor,
        scale_k(self.enh_smag_fac, dtime, self.diff_multfac_smag, offset_provider={})

        log.debug("rbf interpolation 1: start")
        mo_intp_rbf_rbf_vec_interpol_vertex(
            p_e_in=prognostic_state.vn,
            ptr_coeff_1=self.interpolation_state.rbf_coeff_1,
            ptr_coeff_2=self.interpolation_state.rbf_coeff_2,
            p_u_out=self.u_vert,
            p_v_out=self.v_vert,
            horizontal_start=self._vertex_start_lateral_boundary_level_2,
            horizontal_end=self._vertex_end_local,
            vertical_start=0,
            vertical_end=num_levels,
            offset_provider=self.grid.offset_providers,
        )
        log.debug("rbf interpolation 1: end")

        # 2.  HALO EXCHANGE -- CALL sync_patch_array_mult u_vert and v_vert
        log.debug("communication rbf extrapolation of vn - start")
        self._exchange.exchange_and_wait(dims.VertexDim, self.u_vert, self.v_vert)
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
            horizontal_start=self._edge_start_lateral_boundary_level_5,
            horizontal_end=self._edge_end_halo_level_2,
            vertical_start=0,
            vertical_end=num_levels,
            offset_provider=self.grid.offset_providers,
        )
        log.debug("running stencil 01 (calculate_nabla2_and_smag_coefficients_for_vn): end")
        if self.config.shear_type != TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND:
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
                horizontal_start=self._cell_start_nudging,
                horizontal_end=self._cell_end_local,
                vertical_start=1,
                vertical_end=num_levels,
                offset_provider=self.grid.offset_providers,
            )
            log.debug(
                "running stencils 02 03 (calculate_diagnostic_quantities_for_turbulence): end"
            )

        # HALO EXCHANGE  IF (discr_vn > 1) THEN CALL sync_patch_array
        # TODO (magdalena) move this up and do asynchronous exchange
        if self.config.type_vn_diffu > 1:
            log.debug("communication rbf extrapolation of z_nable2_e - start")
            self._exchange.exchange_and_wait(dims.EdgeDim, self.z_nabla2_e)
            log.debug("communication rbf extrapolation of z_nable2_e - end")

        log.debug("2nd rbf interpolation: start")
        mo_intp_rbf_rbf_vec_interpol_vertex(
            p_e_in=self.z_nabla2_e,
            ptr_coeff_1=self.interpolation_state.rbf_coeff_1,
            ptr_coeff_2=self.interpolation_state.rbf_coeff_2,
            p_u_out=self.u_vert,
            p_v_out=self.v_vert,
            horizontal_start=self._vertex_start_lateral_boundary_level_2,
            horizontal_end=self._vertex_end_local,
            vertical_start=0,
            vertical_end=num_levels,
            offset_provider=self.grid.offset_providers,
        )
        log.debug("2nd rbf interpolation: end")

        # 6.  HALO EXCHANGE -- CALL sync_patch_array_mult (Vertex Fields)
        log.debug("communication rbf extrapolation of z_nable2_e - start")
        self._exchange.exchange_and_wait(dims.VertexDim, self.u_vert, self.v_vert)
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
            start_2nd_nudge_line_idx_e=self._edge_start_nudging_level_2,
            limited_area=self.grid.limited_area,
            horizontal_start=self._edge_start_lateral_boundary_level_5,
            horizontal_end=self._edge_end_local,
            vertical_start=0,
            vertical_end=num_levels,
            offset_provider=self.grid.offset_providers,
        )
        log.debug("running stencils 04 05 06 (apply_diffusion_to_vn): end")
        log.debug("communication of prognistic.vn : start")
        handle_edge_comm = self._exchange.exchange(dims.EdgeDim, prognostic_state.vn)

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
            type_shear=gtx.int32(self.config.shear_type.value),
            dwdx=diagnostic_state.dwdx,
            dwdy=diagnostic_state.dwdy,
            diff_multfac_w=self.diff_multfac_w,
            diff_multfac_n2w=self.diff_multfac_n2w,
            k=self.vertical_index,
            cell=self.horizontal_cell_index,
            nrdmax=gtx.int32(
                self.vertical_grid.end_index_of_damping_layer + 1
            ),  # +1 since Fortran includes boundaries
            interior_idx=self._cell_start_interior,
            halo_idx=self._cell_end_local,
            horizontal_start=self._horizontal_start_index_w_diffusion,
            horizontal_end=self._cell_end_halo,
            vertical_start=0,
            vertical_end=num_levels,
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
            smallest_vpfloat=constants.DBL_EPS,
            kh_smag_e=self.kh_smag_e,
            horizontal_start=self._edge_start_nudging,
            horizontal_end=self._edge_end_halo,
            vertical_start=(num_levels - 2),
            vertical_end=num_levels,
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
            horizontal_start=self._cell_start_nudging,
            horizontal_end=self._cell_end_local,
            vertical_start=0,
            vertical_end=num_levels,
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
                horizontal_start=self._cell_start_nudging,
                horizontal_end=self._cell_end_local,
                vertical_start=0,
                vertical_end=num_levels,
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
            horizontal_start=self._cell_start_nudging,
            horizontal_end=self._cell_end_local,
            vertical_start=0,
            vertical_end=num_levels,
            offset_provider={},
        )
        log.debug("running stencil 16 (update_theta_and_exner): end")
        handle_edge_comm.wait()  # need to do this here, since we currently only use 1 communication object.
        log.debug("communication of prognogistic.vn - end")
