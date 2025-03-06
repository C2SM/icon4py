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
import icon4py.model.common.grid.states as grid_states
from gt4py.next import int32

import icon4py.model.common.states.prognostic_state as prognostics
from gt4py.next import backend as gtx_backend

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
)
from icon4py.model.common.interpolation.stencils.mo_intp_rbf_rbf_vec_interpol_vertex import (
    mo_intp_rbf_rbf_vec_interpol_vertex,
)

from icon4py.model.common.utils import data_allocation as data_alloc

from icon4py.model.common.orchestration import decorator as dace_orchestration


"""
Diffusion module ported from ICON mo_nh_diffusion.f90.

Supports only diffusion_type (=hdiff_order) 5 from the diffusion namelist.
"""

# flake8: noqa
log = logging.getLogger(__name__)


class DiffusionType(int, enum.Enum):
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


class TurbulenceShearForcingType(int, enum.Enum):
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
        ltkeshs: bool = True,
    ):
        """Set the diffusion configuration parameters with the ICON default values."""
        # parameters from namelist diffusion_nml

        self.diffusion_type: int = diffusion_type

        #: If True, apply diffusion on the vertical wind field
        #: Called 'lhdiff_w' in mo_diffusion_nml.f90
        self.apply_to_vertical_wind: bool = hdiff_w

        #: True apply diffusion on the horizontal wind field, is ONLY used in mo_nh_stepping.f90
        #: Called 'lhdiff_vn' in mo_diffusion_nml.f90
        self.apply_to_horizontal_wind = hdiff_vn

        #:  If True, apply horizontal diffusion to temperature field
        #: Called 'lhdiff_temp' in mo_diffusion_nml.f90
        self.apply_to_temperature: bool = hdiff_temp

        #: If True, compute 3D Smagorinsky diffusion coefficient
        #: Called 'lsmag_3d' in mo_diffusion_nml.f90
        self.compute_3d_smag_coeff: bool = smag_3d

        #: Options for discretizing the Smagorinsky momentum diffusion
        #: Called 'itype_vn_diffu' in mo_diffusion_nml.f90
        self.type_vn_diffu: int = type_vn_diffu

        #: Options for discretizing the Smagorinsky temperature diffusion
        #: Called 'itype_t_diffu' in mo_diffusion_nml.f90
        self.type_t_diffu: int = type_t_diffu

        #: Ratio of e-folding time to (2*)time step
        #: Called 'hdiff_efdt_ratio' in mo_diffusion_nml.f90
        self.hdiff_efdt_ratio: float = hdiff_efdt_ratio

        #: Ratio of e-folding time to time step for w diffusion (NH only)
        #: Called 'hdiff_w_efdt_ratio' in mo_diffusion_nml.f90.
        self.hdiff_w_efdt_ratio: float = hdiff_w_efdt_ratio

        #: Scaling factor for Smagorinsky diffusion at height hdiff_smag_z and below
        #: Called 'hdiff_smag_fac' in mo_diffusion_nml.f90
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
        #: Called 'nudge_max_coeff' in mo_interpol_nml.f90
        self.nudge_max_coeff: float = max_nudging_coeff

        #: Exponential decay rate (in units of cell rows) of the lateral boundary nudging coefficients
        #: Called 'nudge_efold_width' in mo_interpol_nml.f90
        self.nudge_efold_width: float = nudging_decay_rate

        #: Type of shear forcing used in turbulence
        #: Called 'itype_shear' in mo_turbdiff_nml.f90
        self.shear_type = shear_type

        #: Consider separate horizontal shear production in TKE-equation.
        #: Called 'ltkeshs' in mo_turbdiff_nml.f90
        self.ltkeshs = ltkeshs

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
            config.nudge_max_coeff * constants.DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO,
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

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        config: DiffusionConfig,
        params: DiffusionParams,
        vertical_grid: v_grid.VerticalGrid,
        metric_state: diffusion_states.DiffusionMetricState,
        interpolation_state: diffusion_states.DiffusionInterpolationState,
        edge_params: grid_states.EdgeParams,
        cell_params: grid_states.CellParams,
        backend: Optional[gtx_backend.Backend],
        orchestration: bool = False,
        exchange: decomposition.ExchangeRuntime = decomposition.SingleNodeExchange(),
    ):
        self._backend = backend
        self._orchestration = orchestration
        self._exchange = exchange
        self.config = config
        self._params = params
        self._grid = grid
        self._vertical_grid = vertical_grid
        self._metric_state = metric_state
        self._interpolation_state = interpolation_state
        self._edge_params = edge_params
        self._cell_params = cell_params

        self.halo_exchange_wait = decomposition.create_halo_exchange_wait(
            self._exchange
        )  # wait on a communication handle
        self.rd_o_cvd: float = constants.GAS_CONSTANT_DRY_AIR / (
            constants.CPD - constants.GAS_CONSTANT_DRY_AIR
        )
        #: threshold temperature deviation from neighboring grid points hat activates extra diffusion against runaway cooling
        self.thresh_tdiff: float = -5.0
        self._horizontal_start_index_w_diffusion: gtx.int32 = gtx.int32(0)

        self.mo_intp_rbf_rbf_vec_interpol_vertex = mo_intp_rbf_rbf_vec_interpol_vertex.with_backend(
            self._backend
        )
        self.calculate_nabla2_and_smag_coefficients_for_vn = (
            calculate_nabla2_and_smag_coefficients_for_vn.with_backend(self._backend)
        )
        self.calculate_diagnostic_quantities_for_turbulence = (
            calculate_diagnostic_quantities_for_turbulence.with_backend(self._backend)
        )
        self.apply_diffusion_to_vn = apply_diffusion_to_vn.with_backend(self._backend)
        self.apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence = (
            apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence.with_backend(
                self._backend
            )
        )
        self.calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools = (
            calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools.with_backend(
                self._backend
            )
        )
        self.calculate_nabla2_for_theta = calculate_nabla2_for_theta.with_backend(self._backend)
        self.truly_horizontal_diffusion_nabla_of_theta_over_steep_points = (
            truly_horizontal_diffusion_nabla_of_theta_over_steep_points.with_backend(self._backend)
        )
        self.update_theta_and_exner = update_theta_and_exner.with_backend(self._backend)
        self.copy_field = copy_field.with_backend(self._backend)
        self.scale_k = scale_k.with_backend(self._backend)
        self.setup_fields_for_initial_step = setup_fields_for_initial_step.with_backend(
            self._backend
        )

        self.init_diffusion_local_fields_for_regular_timestep = (
            init_diffusion_local_fields_for_regular_timestep.with_backend(self._backend)
        )

        self._allocate_temporary_fields()

        self.nudgezone_diff: float = 0.04 / (params.scaled_nudge_max_coeff + sys.float_info.epsilon)
        self.bdy_diff: float = 0.015 / (params.scaled_nudge_max_coeff + sys.float_info.epsilon)
        self.fac_bdydiff_v: float = (
            math.sqrt(config.substep_as_float) / config.velocity_boundary_diffusion_denominator
        )

        self.smag_offset: float = 0.25 * params.K4 * config.substep_as_float
        self.diff_multfac_w: float = min(1.0 / 48.0, params.K4W * config.substep_as_float)

        self.init_diffusion_local_fields_for_regular_timestep(
            params.K4,
            config.substep_as_float,
            *params.smagorinski_factor,
            *params.smagorinski_height,
            self._vertical_grid.interface_physical_height,
            self.diff_multfac_vn,
            self.smag_limit,
            self.enh_smag_fac,
            offset_provider={"Koff": dims.KDim},
        )

        diffusion_utils._init_nabla2_factor_in_upper_damping_zone(
            physical_heights=self._vertical_grid.interface_physical_height,
            k_field=self.vertical_index,
            nrdmax=self._vertical_grid.end_index_of_damping_layer,
            nshift=0,
            heights_nrd_shift=self._vertical_grid.interface_physical_height.ndarray[
                self._vertical_grid.end_index_of_damping_layer + 1
            ].item(),
            heights_1=self._vertical_grid.interface_physical_height.ndarray[1].item(),
            domain={dims.KDim: (1, self._vertical_grid.end_index_of_damping_layer + 1)},
            out=self.diff_multfac_n2w,
            offset_provider={},
        )

        self._determine_horizontal_domains()

        self.compile_time_connectivities = dace_orchestration.build_compile_time_connectivities(
            self._grid.offset_providers
        )

    def _allocate_temporary_fields(self):
        self.diff_multfac_vn = data_alloc.zero_field(self._grid, dims.KDim, backend=self._backend)
        self.diff_multfac_n2w = data_alloc.zero_field(self._grid, dims.KDim, backend=self._backend)
        self.smag_limit = data_alloc.zero_field(self._grid, dims.KDim, backend=self._backend)
        self.enh_smag_fac = data_alloc.zero_field(self._grid, dims.KDim, backend=self._backend)
        self.u_vert = data_alloc.zero_field(
            self._grid, dims.VertexDim, dims.KDim, backend=self._backend
        )
        self.v_vert = data_alloc.zero_field(
            self._grid, dims.VertexDim, dims.KDim, backend=self._backend
        )
        self.kh_smag_e = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, backend=self._backend
        )
        self.kh_smag_ec = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, backend=self._backend
        )
        self.z_nabla2_e = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, backend=self._backend
        )
        self.z_temp = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, backend=self._backend
        )
        self.diff_multfac_smag = data_alloc.zero_field(self._grid, dims.KDim, backend=self._backend)
        # TODO(Magdalena): this is KHalfDim
        self.vertical_index = data_alloc.index_field(
            self._grid, dims.KDim, extend={dims.KDim: 1}, backend=self._backend
        )
        self.horizontal_cell_index = data_alloc.index_field(
            self._grid, dims.CellDim, backend=self._backend
        )
        self.horizontal_edge_index = data_alloc.index_field(
            self._grid, dims.EdgeDim, backend=self._backend
        )
        self.w_tmp = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, backend=self._backend
        )

    def _determine_horizontal_domains(self):
        cell_domain = h_grid.domain(dims.CellDim)
        edge_domain = h_grid.domain(dims.EdgeDim)
        vertex_domain = h_grid.domain(dims.VertexDim)

        def _get_start_index_for_w_diffusion() -> gtx.int32:
            return (
                self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
                if self._grid.limited_area
                else self._grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
            )

        self._cell_start_interior = self._grid.start_index(cell_domain(h_grid.Zone.INTERIOR))
        self._cell_start_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._cell_end_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._cell_end_halo = self._grid.end_index(cell_domain(h_grid.Zone.HALO))

        self._edge_start_lateral_boundary_level_5 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._edge_start_nudging = self._grid.start_index(edge_domain(h_grid.Zone.NUDGING))
        self._edge_start_nudging_level_2 = self._grid.start_index(
            edge_domain(h_grid.Zone.NUDGING_LEVEL_2)
        )
        self._edge_end_local = self._grid.end_index(edge_domain(h_grid.Zone.LOCAL))
        self._edge_end_halo = self._grid.end_index(edge_domain(h_grid.Zone.HALO))
        self._edge_end_halo_level_2 = self._grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))

        self._vertex_start_lateral_boundary_level_2 = self._grid.start_index(
            vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._vertex_end_local = self._grid.end_index(vertex_domain(h_grid.Zone.LOCAL))

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
        diff_multfac_vn = data_alloc.zero_field(self._grid, dims.KDim, backend=self._backend)
        smag_limit = data_alloc.zero_field(self._grid, dims.KDim, backend=self._backend)

        self.setup_fields_for_initial_step(
            self._params.K4,
            self.config.hdiff_efdt_ratio,
            diff_multfac_vn,
            smag_limit,
            offset_provider={},
        )
        self._do_diffusion_step(
            diagnostic_state, prognostic_state, dtime, diff_multfac_vn, smag_limit, 0.0
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

    @dace_orchestration.orchestrate
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
        # dtime dependent: enh_smag_factor,
        self.scale_k.with_connectivities(self.compile_time_connectivities)(
            self.enh_smag_fac, dtime, self.diff_multfac_smag, offset_provider={}
        )

        log.debug("rbf interpolation 1: start")
        self.mo_intp_rbf_rbf_vec_interpol_vertex.with_connectivities(
            self.compile_time_connectivities
        )(
            p_e_in=prognostic_state.vn,
            ptr_coeff_1=self._interpolation_state.rbf_coeff_1,
            ptr_coeff_2=self._interpolation_state.rbf_coeff_2,
            p_u_out=self.u_vert,
            p_v_out=self.v_vert,
            horizontal_start=self._vertex_start_lateral_boundary_level_2,
            horizontal_end=self._vertex_end_local,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("rbf interpolation 1: end")

        # 2.  HALO EXCHANGE -- CALL sync_patch_array_mult u_vert and v_vert
        log.debug("communication rbf extrapolation of vn - start")
        self._exchange(
            self.u_vert,
            self.v_vert,
            dim=dims.VertexDim,
            wait=True,
        )
        log.debug("communication rbf extrapolation of vn - end")

        log.debug("running stencil 01(calculate_nabla2_and_smag_coefficients_for_vn): start")
        self.calculate_nabla2_and_smag_coefficients_for_vn.with_connectivities(
            self.compile_time_connectivities
        )(
            diff_multfac_smag=self.diff_multfac_smag,
            tangent_orientation=self._edge_params.tangent_orientation,
            inv_primal_edge_length=self._edge_params.inverse_primal_edge_lengths,
            inv_vert_vert_length=self._edge_params.inverse_vertex_vertex_lengths,
            u_vert=self.u_vert,
            v_vert=self.v_vert,
            primal_normal_vert_x=self._edge_params.primal_normal_vert[0],
            primal_normal_vert_y=self._edge_params.primal_normal_vert[1],
            dual_normal_vert_x=self._edge_params.dual_normal_vert[0],
            dual_normal_vert_y=self._edge_params.dual_normal_vert[1],
            vn=prognostic_state.vn,
            smag_limit=smag_limit,
            kh_smag_e=self.kh_smag_e,
            kh_smag_ec=self.kh_smag_ec,
            z_nabla2_e=self.z_nabla2_e,
            smag_offset=smag_offset,
            horizontal_start=self._edge_start_lateral_boundary_level_5,
            horizontal_end=self._edge_end_halo_level_2,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencil 01 (calculate_nabla2_and_smag_coefficients_for_vn): end")
        if (
            self.config.shear_type
            >= TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND
            or self.config.ltkeshs
        ):
            log.debug(
                "running stencils 02 03 (calculate_diagnostic_quantities_for_turbulence): start"
            )
            self.calculate_diagnostic_quantities_for_turbulence.with_connectivities(
                self.compile_time_connectivities
            )(
                kh_smag_ec=self.kh_smag_ec,
                vn=prognostic_state.vn,
                e_bln_c_s=self._interpolation_state.e_bln_c_s,
                geofac_div=self._interpolation_state.geofac_div,
                diff_multfac_smag=self.diff_multfac_smag,
                wgtfac_c=self._metric_state.wgtfac_c,
                div_ic=diagnostic_state.div_ic,
                hdef_ic=diagnostic_state.hdef_ic,
                horizontal_start=self._cell_start_nudging,
                horizontal_end=self._cell_end_local,
                vertical_start=1,
                vertical_end=self._grid.num_levels,
                offset_provider=self._grid.offset_providers,
            )
            log.debug(
                "running stencils 02 03 (calculate_diagnostic_quantities_for_turbulence): end"
            )

        # HALO EXCHANGE  IF (discr_vn > 1) THEN CALL sync_patch_array
        # TODO (magdalena) move this up and do asynchronous exchange
        if self.config.type_vn_diffu > 1:
            log.debug("communication rbf extrapolation of z_nable2_e - start")
            self._exchange(self.z_nabla2_e, dim=dims.EdgeDim, wait=True)
            log.debug("communication rbf extrapolation of z_nable2_e - end")

        log.debug("2nd rbf interpolation: start")
        self.mo_intp_rbf_rbf_vec_interpol_vertex.with_connectivities(
            self.compile_time_connectivities
        )(
            p_e_in=self.z_nabla2_e,
            ptr_coeff_1=self._interpolation_state.rbf_coeff_1,
            ptr_coeff_2=self._interpolation_state.rbf_coeff_2,
            p_u_out=self.u_vert,
            p_v_out=self.v_vert,
            horizontal_start=self._vertex_start_lateral_boundary_level_2,
            horizontal_end=self._vertex_end_local,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("2nd rbf interpolation: end")

        # 6.  HALO EXCHANGE -- CALL sync_patch_array_mult (Vertex Fields)
        log.debug("communication rbf extrapolation of z_nable2_e - start")
        self._exchange(
            self.u_vert,
            self.v_vert,
            dim=dims.VertexDim,
            wait=True,
        )
        log.debug("communication rbf extrapolation of z_nable2_e - end")

        log.debug("running stencils 04 05 06 (apply_diffusion_to_vn): start")
        self.apply_diffusion_to_vn.with_connectivities(self.compile_time_connectivities)(
            u_vert=self.u_vert,
            v_vert=self.v_vert,
            primal_normal_vert_v1=self._edge_params.primal_normal_vert[0],
            primal_normal_vert_v2=self._edge_params.primal_normal_vert[1],
            z_nabla2_e=self.z_nabla2_e,
            inv_vert_vert_length=self._edge_params.inverse_vertex_vertex_lengths,
            inv_primal_edge_length=self._edge_params.inverse_primal_edge_lengths,
            area_edge=self._edge_params.edge_areas,
            kh_smag_e=self.kh_smag_e,
            diff_multfac_vn=diff_multfac_vn,
            nudgecoeff_e=self._interpolation_state.nudgecoeff_e,
            vn=prognostic_state.vn,
            edge=self.horizontal_edge_index,
            nudgezone_diff=self.nudgezone_diff,
            fac_bdydiff_v=self.fac_bdydiff_v,
            start_2nd_nudge_line_idx_e=self._edge_start_nudging_level_2,
            limited_area=self._grid.limited_area,
            horizontal_start=self._edge_start_lateral_boundary_level_5,
            horizontal_end=self._edge_end_local,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug("running stencils 04 05 06 (apply_diffusion_to_vn): end")

        log.debug("communication of prognistic.vn : start")
        handle_edge_comm = self._exchange(prognostic_state.vn, dim=dims.EdgeDim, wait=False)

        log.debug(
            "running stencils 07 08 09 10 (apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence): start"
        )
        # TODO (magdalena) get rid of this copying. So far passing an empty buffer instead did not verify?
        self.copy_field.with_connectivities(self.compile_time_connectivities)(
            prognostic_state.w, self.w_tmp, offset_provider={}
        )

        self.apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence.with_connectivities(
            self.compile_time_connectivities
        )(
            area=self._cell_params.area,
            geofac_n2s=self._interpolation_state.geofac_n2s,
            geofac_grg_x=self._interpolation_state.geofac_grg_x,
            geofac_grg_y=self._interpolation_state.geofac_grg_y,
            w_old=self.w_tmp,
            w=prognostic_state.w,
            type_shear=int32(
                self.config.shear_type.value
            ),  # DaCe parser peculiarity (does not work as gtx.int32)
            dwdx=diagnostic_state.dwdx,
            dwdy=diagnostic_state.dwdy,
            diff_multfac_w=self.diff_multfac_w,
            diff_multfac_n2w=self.diff_multfac_n2w,
            k=self.vertical_index,
            cell=self.horizontal_cell_index,
            nrdmax=int32(  # DaCe parser peculiarity (does not work as gtx.int32)
                self._vertical_grid.end_index_of_damping_layer + 1
            ),  # +1 since Fortran includes boundaries
            interior_idx=self._cell_start_interior,
            halo_idx=self._cell_end_local,
            horizontal_start=self._horizontal_start_index_w_diffusion,
            horizontal_end=self._cell_end_halo,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug(
            "running stencils 07 08 09 10 (apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence): end"
        )

        if self.config.apply_to_temperature:
            log.debug(
                "running fused stencils 11 12 (calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools): start"
            )

            self.calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools.with_connectivities(
                self.compile_time_connectivities
            )(
                theta_v=prognostic_state.theta_v,
                theta_ref_mc=self._metric_state.theta_ref_mc,
                thresh_tdiff=self.thresh_tdiff,
                smallest_vpfloat=constants.DBL_EPS,
                kh_smag_e=self.kh_smag_e,
                horizontal_start=self._edge_start_nudging,
                horizontal_end=self._edge_end_halo,
                vertical_start=(self._grid.num_levels - 2),
                vertical_end=self._grid.num_levels,
                offset_provider=self._grid.offset_providers,
            )
            log.debug(
                "running stencils 11 12 (calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools): end"
            )

            log.debug("running stencils 13 14 (calculate_nabla2_for_theta): start")
            self.calculate_nabla2_for_theta.with_connectivities(self.compile_time_connectivities)(
                kh_smag_e=self.kh_smag_e,
                inv_dual_edge_length=self._edge_params.inverse_dual_edge_lengths,
                theta_v=prognostic_state.theta_v,
                geofac_div=self._interpolation_state.geofac_div,
                z_temp=self.z_temp,
                horizontal_start=self._cell_start_nudging,
                horizontal_end=self._cell_end_local,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
                offset_provider=self._grid.offset_providers,
            )
            log.debug("running stencils 13_14 (calculate_nabla2_for_theta): end")
            log.debug(
                "running stencil 15 (truly_horizontal_diffusion_nabla_of_theta_over_steep_points): start"
            )
            if self.config.apply_zdiffusion_t:
                self.truly_horizontal_diffusion_nabla_of_theta_over_steep_points.with_connectivities(
                    self.compile_time_connectivities
                )(
                    mask=self._metric_state.mask_hdiff,
                    zd_vertoffset=self._metric_state.zd_vertoffset,
                    zd_diffcoef=self._metric_state.zd_diffcoef,
                    geofac_n2s_c=self._interpolation_state.geofac_n2s_c,
                    geofac_n2s_nbh=self._interpolation_state.geofac_n2s_nbh,
                    vcoef=self._metric_state.zd_intcoef,
                    theta_v=prognostic_state.theta_v,
                    z_temp=self.z_temp,
                    horizontal_start=self._cell_start_nudging,
                    horizontal_end=self._cell_end_local,
                    vertical_start=0,
                    vertical_end=self._grid.num_levels,
                    offset_provider=self._grid.offset_providers,
                )

                log.debug(
                    "running fused stencil 15 (truly_horizontal_diffusion_nabla_of_theta_over_steep_points): end"
                )
            log.debug("running stencil 16 (update_theta_and_exner): start")
            self.update_theta_and_exner.with_connectivities(self.compile_time_connectivities)(
                z_temp=self.z_temp,
                area=self._cell_params.area,
                theta_v=prognostic_state.theta_v,
                exner=prognostic_state.exner,
                rd_o_cvd=self.rd_o_cvd,
                horizontal_start=self._cell_start_nudging,
                horizontal_end=self._cell_end_local,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
                offset_provider={},
            )
            log.debug("running stencil 16 (update_theta_and_exner): end")

        self.halo_exchange_wait(
            handle_edge_comm
        )  # need to do this here, since we currently only use 1 communication object.
        log.debug("communication of prognogistic.vn - end")

    # TODO (kotsaloscv): It is unsafe to set it as cached property -demands more testing-
    def orchestration_uid(self) -> str:
        """Unique id based on the runtime state of the Diffusion object. It is used for caching in DaCe Orchestration."""
        members_to_disregard = [
            "_backend",
            "_exchange",
            "_grid",
            *[
                name
                for name in self.__dict__.keys()
                if isinstance(self.__dict__[name], gtx.ffront.decorator.Program)
            ],
        ]
        return dace_orchestration.generate_orchestration_uid(
            self, members_to_disregard=members_to_disregard
        )
