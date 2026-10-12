# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import enum
import logging
import math
import typing
from typing import Final

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

import icon4py.model.common.grid.states as grid_states
import icon4py.model.common.states.prognostic_state as prognostics
from icon4py.model.atmosphere.diffusion import diffusion_states, diffusion_utils
from icon4py.model.atmosphere.diffusion.diffusion_utils import (
    init_diffusion_local_fields_for_regular_timestep,
    scale_k,
    setup_fields_for_initial_step,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_and_smag_coefficients_for_vn import (
    calculate_nabla2_and_smag_coefficients_for_vn,
)
from icon4py.model.atmosphere.diffusion.stencils.diffusion_for_prognostic_fields import (
    diffusion_for_prognostic_fields,
)
from icon4py.model.common import constants, dimension as dims, model_backends, type_alias as ta
from icon4py.model.common.config import config_io, options as common_conf_opt
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, vertical as v_grid
from icon4py.model.common.interpolation.stencils.mo_intp_rbf_rbf_vec_interpol_vertex import (
    mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.model.common.math.stencils import generic_math_operations
from icon4py.model.common.model_options import setup_program
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc


"""
Diffusion module ported from ICON mo_nh_diffusion.f90.

Supports only diffusion_type (=hdiff_order) 5 from the diffusion namelist.
"""

log = logging.getLogger(__name__)


@config_io.register_enum
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


@config_io.register_enum
class SmagorinskyStencilType(int, enum.Enum):
    """
    Type of the reconstruction stencil for the Smagorinsky diffusion of normal wind (vn).

    Note: Called `itype_vn_diffu` in `mo_diffusion_nml.f90`.
    Note: We currently only support type 1 in combination with lsmag_3d=False.
    """

    DIAMOND_VERTICES = (
        1  #: Smagorinsky diffusion of vn with diamond stencil on vertices (only for vn)
    )
    CELLS_AND_VERTICES = 2  #: Smagorinsky diffusion of vn with stencil on neighboring vertices (E2V) and cell centers (E2C)


@config_io.register_enum
class TemperatureDiscretizationType(int, enum.Enum):
    """
    Type of the discretization of the Smagorinsky diffusion of temperature.

    Note: Called `itype_t_diffu` in `mo_diffusion_nml.f90`.
    Note: We currently only support type 2.
    """

    HOMOGENEOUS = 1  #: K Lap(T)
    HETEROGENEOUS = 2  #: Div (K Grad(T))


@config_io.register_enum
class TurbulenceShearForcingType(int, enum.Enum):
    """
    Type of shear forcing used in turbulence.

    Note: called `itype_sher` in `mo_turbdiff_nml.f90`
    """

    VERTICAL_OF_HORIZONTAL_WIND = 0  #: only vertical shear of horizontal wind
    VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND = (
        1  #: as `VERTICAL_ONLY` plus horizontal shear correction
    )
    VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND = (
        2  #: as `VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND` plus shear form vertical velocity
    )
    VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND_LTHESH = 3  #: same as `VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND` but scaling of coarse-grid horizontal shear production term with 1/sqrt(Ri) (if LTKESH = TRUE)


@config_io.register_enum
class ForcingType(int, enum.Enum):
    """
    Type of physics forcing applied to the model.

    Note: called `iforcing` in `mo_run_nml.f90`
    """

    NO_FORCING = 0  #: no physics forcing (diagnostic / idealized runs)
    AES = 2  #: Atmospheric Earth System / ECHAM forcing (iaes)
    NWP = 3  #: Numerical Weather Prediction forcing (inwp)


@dataclasses.dataclass(frozen=True, kw_only=True)
class DiffusionConfig:
    """
    Contains necessary parameter to configure a diffusion run.

    Encapsulates namelist parameters and derived parameters.
    Values should be read from configuration.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    """

    # TODO(halungge): to be read from config
    # TODO(halungge):  handle dependencies on other namelists (see below...)

    diffusion_type: typing.Annotated[
        DiffusionType,
        common_conf_opt.ConfigOption(
            description="Order of Nabla operator for diffusion.",
        ),
    ] = DiffusionType.SMAGORINSKY_4TH_ORDER

    apply_to_vertical_wind: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, apply diffusion to the vertical wind field.",
        ),
    ] = True

    apply_to_horizontal_wind: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If true, apply diffusion on the horizontal wind field.",
        ),
    ] = True

    apply_to_temperature: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, apply horizontal diffusion to temperature field.",
        ),
    ] = True

    apply_smag_diff_to_vertical_wind: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, apply Smagorinsky diffusion to vertical wind field.",
        ),
    ] = False

    compute_3d_smag_coeff: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, compute 3D Smagorinsky diffusion coefficient.",
        ),
    ] = False

    type_vn_diffu: typing.Annotated[
        SmagorinskyStencilType,
        common_conf_opt.ConfigOption(
            description="Reconstruction method used for Smagorinsky diffusion.",
        ),
    ] = SmagorinskyStencilType.DIAMOND_VERTICES

    type_t_diffu: typing.Annotated[
        TemperatureDiscretizationType,
        common_conf_opt.ConfigOption(
            description="Options for discretizing the Smagorinsky temperature diffusion.",
        ),
    ] = TemperatureDiscretizationType.HETEROGENEOUS

    hdiff_efdt_ratio: typing.Annotated[
        ta.wpfloat,
        common_conf_opt.ConfigOption(
            description="Ratio of e-folding time to (2*)time step.",
        ),
    ] = 36.0

    hdiff_w_efdt_ratio: typing.Annotated[
        ta.wpfloat,
        common_conf_opt.ConfigOption(
            description="Ratio of e-folding time to time step for w diffusion (NH only).",
        ),
    ] = 15.0

    # TODO(muellch): The four smagorinsky factors and heights should be in one or two dataclasses.
    smagorinski_scaling_factor: typing.Annotated[
        ta.wpfloat,
        common_conf_opt.ConfigOption(
            description="Smagorinsky factor for z <= smagorinski_scaling_height (constant base value).",
        ),
    ] = 0.015

    smagorinski_scaling_factor2: typing.Annotated[
        ta.wpfloat,
        common_conf_opt.ConfigOption(
            description=(
                "Smagorinsky factor at z = smagorinski_scaling_height2: end of the linear segment and"
                "start of the quadratic segment. The linear slope is (factor2-factor1)/(height2-height1)."
            ),
        ),
    ] = 2e-6 * (1600.0 + 25000.0 + math.sqrt(1600.0 * (1600 + 50000.0)))

    smagorinski_scaling_factor3: typing.Annotated[
        ta.wpfloat,
        common_conf_opt.ConfigOption(
            description=(
                "Smagorinsky factor at z = smagorinski_scaling_height3: interior control point of the"
                "quadratic segment (height2 <= height3 <= height4), used to fit the quadratic coefficients."
            ),
        ),
    ] = 0.0

    smagorinski_scaling_factor4: typing.Annotated[
        ta.wpfloat,
        common_conf_opt.ConfigOption(
            description=(
                "Smagorinsky factor for z >= smagorinski_scaling_height4 (constant asymptotic value)."
                "Also the third control point that defines the quadratic segment together with factor2 and factor3."
            ),
        ),
    ] = 1.0

    smagorinski_scaling_height: typing.Annotated[
        ta.wpfloat,
        common_conf_opt.ConfigOption(
            description=(
                "Lower boundary of the linear segment: factor is constant at smagorinski_scaling_factor "
                "below this height."
            ),
        ),
    ] = 32500.0

    smagorinski_scaling_height2: typing.Annotated[
        ta.wpfloat,
        common_conf_opt.ConfigOption(
            description="Transition height between linear and quadratic segments.",
        ),
    ] = 1600.0 + 50000.0 + math.sqrt(1600.0 * (1600 + 50000.0))

    smagorinski_scaling_height3: typing.Annotated[
        ta.wpfloat,
        common_conf_opt.ConfigOption(
            description="Interior control point height within the quadratic segment (height2 <= height3 <= height4).",
        ),
    ] = 50000.0

    smagorinski_scaling_height4: typing.Annotated[
        ta.wpfloat,
        common_conf_opt.ConfigOption(
            description=(
                "Upper boundary of the quadratic segment: factor is constant at "
                "smagorinski_scaling_factor4 above this height."
            ),
        ),
    ] = 90000.0

    apply_zdiffusion_t: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, apply truly horizontal temperature diffusion over steep slopes.",
        ),
    ] = True

    temperature_boundary_diffusion_denominator: typing.Annotated[
        ta.wpfloat,
        common_conf_opt.ConfigOption(
            description="Denominator for temperature boundary diffusion.",
        ),
    ] = 135.0

    velocity_boundary_diffusion_denominator: typing.Annotated[
        ta.wpfloat,
        common_conf_opt.ConfigOption(
            description="Denominator for velocity boundary diffusion.",
        ),
    ] = 200.0

    shear_type: typing.Annotated[
        TurbulenceShearForcingType,
        common_conf_opt.ConfigOption(
            description="Type of shear forcing used in turbulence.",
        ),
    ] = TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND

    iforcing: typing.Annotated[
        ForcingType,
        common_conf_opt.ConfigOption(
            description="Type of physics forcing.",
        ),
    ] = ForcingType.NO_FORCING

    a_hshr: typing.Annotated[
        ta.wpfloat,
        common_conf_opt.ConfigOption(
            description="Scaling factor for horizontal shear production term.",
        ),
    ] = 1.0

    loutshs: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description=(
                "Output flag for horizontal shear."
                "Called 'loutshs' in mo_turbdiff_nml.f90, "
                "not a namelist parameter: its default is FALSE and "
                "only set to true in fortran `IF (.NOT. ldynamics)`."
            )
        ),
    ] = False

    def __post_init__(self) -> None:
        ta.dataclass_float_to_wp(self)

        self._validate()

    def _validate(self) -> None:
        """Apply consistency checks and validation on configuration parameters."""
        if self.diffusion_type != DiffusionType.SMAGORINSKY_4TH_ORDER:
            raise NotImplementedError(
                "Only diffusion type 5 = `Smagorinsky diffusion with fourth-order background "
                "diffusion` is implemented"
            )

        if self.type_vn_diffu != SmagorinskyStencilType.DIAMOND_VERTICES:
            raise NotImplementedError(
                "Only type_vn_diffu 1 = `Smagorinsky diffusion with diamond stencil on vertices` is implemented"
            )

        if self.type_t_diffu != TemperatureDiscretizationType.HETEROGENEOUS:
            raise NotImplementedError(
                "Only type_t_diffu 2 = `Smagorinsky diffusion with heterogeneous discretization` is implemented"
            )

        if self.apply_smag_diff_to_vertical_wind:
            raise NotImplementedError("Smagorinsky diffusion for vertical wind is not implemented")

        if self.compute_3d_smag_coeff:
            raise NotImplementedError("3D Smagorinsky diffusion computation is not implemented")

        if self.shear_type not in (
            TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND,
            TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND,
            TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND,
        ):
            raise NotImplementedError(
                f"Turbulence Shear only {TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND} "
                f"and {TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND} "
                f"and {TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND} "
                f"implemented"
            )


@dataclasses.dataclass(frozen=True)
class DiffusionParams:
    """Calculates derived quantities depending on the diffusion config."""

    config: dataclasses.InitVar[DiffusionConfig]
    K2: Final[wpfloat] = dataclasses.field(init=False)
    K4: Final[wpfloat] = dataclasses.field(init=False)
    K6: Final[wpfloat] = dataclasses.field(init=False)
    K4W: Final[wpfloat] = dataclasses.field(init=False)
    smagorinski_factor: Final[tuple[wpfloat, wpfloat, wpfloat, wpfloat]] = dataclasses.field(
        init=False
    )
    smagorinski_height: Final[tuple[wpfloat, wpfloat, wpfloat, wpfloat]] = dataclasses.field(
        init=False
    )

    def __post_init__(self, config: DiffusionConfig) -> None:
        object.__setattr__(
            self,
            "K2",
            (
                wpfloat(1.0) / (config.hdiff_efdt_ratio * wpfloat(8.0))
                if config.hdiff_efdt_ratio > wpfloat(0.0)
                else wpfloat(0.0)
            ),
        )
        object.__setattr__(self, "K4", self.K2 / wpfloat(8.0))
        object.__setattr__(self, "K6", self.K2 / wpfloat(64.0))
        object.__setattr__(
            self,
            "K4W",
            (
                wpfloat(1.0) / (config.hdiff_w_efdt_ratio * wpfloat(36.0))
                if config.hdiff_w_efdt_ratio > wpfloat(0.0)
                else wpfloat(0.0)
            ),
        )

        object.__setattr__(
            self,
            "smagorinski_factor",
            (
                config.smagorinski_scaling_factor,
                config.smagorinski_scaling_factor2,
                config.smagorinski_scaling_factor3,
                config.smagorinski_scaling_factor4,
            ),
        )
        object.__setattr__(
            self,
            "smagorinski_height",
            (
                config.smagorinski_scaling_height,
                config.smagorinski_scaling_height2,
                config.smagorinski_scaling_height3,
                config.smagorinski_scaling_height4,
            ),
        )


class Diffusion:
    """Class that configures diffusion and does one diffusion step."""

    def __init__(
        self,
        *,
        grid: icon_grid.IconGrid,
        config: DiffusionConfig,
        params: DiffusionParams,
        vertical_grid: v_grid.VerticalGrid,
        metric_state: diffusion_states.DiffusionMetricState,
        interpolation_state: diffusion_states.DiffusionInterpolationState,
        edge_params: grid_states.EdgeParams,
        cell_params: grid_states.CellParams,
        backend: gtx_typing.Backend
        | model_backends.DeviceType
        | model_backends.BackendDescriptor
        | None,
        exchange: decomposition.ExchangeRuntime,
        ndyn_substeps: int,
        max_nudging_coefficient: state_utils.FloatType,
    ) -> None:
        self._allocator = model_backends.get_allocator(backend)
        self._exchange = exchange
        self.config = config
        self._params = params
        self._grid = grid
        self._vertical_grid = vertical_grid
        self._metric_state = metric_state
        self._interpolation_state = interpolation_state
        self._edge_params = edge_params
        self._cell_params = cell_params
        ndyn_substeps_as_float = wpfloat(ndyn_substeps)
        max_nudging_coefficient = wpfloat(max_nudging_coefficient)

        assert self._cell_params.area is not None

        self.rd_o_cvd: vpfloat = gtx.astype(
            constants.GAS_CONSTANT_DRY_AIR / (constants.CPD - constants.GAS_CONSTANT_DRY_AIR),
            vpfloat,
        )
        #: threshold temperature deviation from neighboring grid points that activates extra diffusion against runaway cooling
        self.thresh_tdiff: wpfloat = wpfloat(-5.0)
        self._horizontal_start_index_w_diffusion: gtx.int32 = gtx.int32(0)

        self.nudgezone_diff: vpfloat = gtx.astype(
            wpfloat(0.04) / (max_nudging_coefficient + constants.WP_EPS), vpfloat
        )
        self.bdy_diff: wpfloat = wpfloat(0.015) / (max_nudging_coefficient + constants.WP_EPS)
        self.fac_bdydiff_v: wpfloat = wpfloat(
            math.sqrt(ndyn_substeps_as_float) / config.velocity_boundary_diffusion_denominator
        )

        self.smag_offset: vpfloat = gtx.astype(
            wpfloat(0.25) * params.K4 * ndyn_substeps_as_float, vpfloat
        )
        self.diff_multfac_w: wpfloat = gtx.astype(
            min(1.0 / 48.0, params.K4W * ndyn_substeps_as_float), wpfloat
        )
        self._determine_horizontal_domains()

        self.mo_intp_rbf_rbf_vec_interpol_vertex = setup_program(
            backend=backend,
            program=mo_intp_rbf_rbf_vec_interpol_vertex,
            constant_args={
                "ptr_coeff_1": self._interpolation_state.rbf_coeff_1,
                "ptr_coeff_2": self._interpolation_state.rbf_coeff_2,
            },
            horizontal_sizes={
                "horizontal_start": self._vertex_start_lateral_boundary_level_2,
                "horizontal_end": self._vertex_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )

        self.calculate_nabla2_and_smag_coefficients_for_vn = setup_program(
            backend=backend,
            program=calculate_nabla2_and_smag_coefficients_for_vn,
            constant_args={
                "tangent_orientation": self._edge_params.tangent_orientation,
                "inv_primal_edge_length": self._edge_params.inverse_primal_edge_lengths,
                "inv_vert_vert_length": self._edge_params.inverse_vertex_vertex_lengths,
                "primal_normal_vert_x": self._edge_params.primal_normal_vert[0],
                "primal_normal_vert_y": self._edge_params.primal_normal_vert[1],
                "dual_normal_vert_x": self._edge_params.dual_normal_vert[0],
                "dual_normal_vert_y": self._edge_params.dual_normal_vert[1],
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_lateral_boundary_level_5,
                "horizontal_end": self._edge_end_halo_level_2,
            },
            vertical_sizes={"vertical_start": 0, "vertical_end": self._grid.num_levels},
            offset_provider=self._grid.connectivities,
        )

        self._compute_diagnostic_quantities = (
            self.config.shear_type
            >= TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND
            or self.config.loutshs
            or self.config.a_hshr > 0.0
        )
        self.diffusion_for_prognostic_fields = setup_program(
            backend=backend,
            program=diffusion_for_prognostic_fields,
            constant_args={
                "e_bln_c_s": self._interpolation_state.e_bln_c_s,
                "geofac_div": self._interpolation_state.geofac_div,
                "wgtfac_c": self._metric_state.wgtfac_c,
                "primal_normal_vert_v1": self._edge_params.primal_normal_vert[0],
                "primal_normal_vert_v2": self._edge_params.primal_normal_vert[1],
                "inv_vert_vert_length": self._edge_params.inverse_vertex_vertex_lengths,
                "inv_primal_edge_length": self._edge_params.inverse_primal_edge_lengths,
                "area_edge": self._edge_params.edge_areas,
                "nudgecoeff_e": self._interpolation_state.nudgecoeff_e,
                "nudgezone_diff": self.nudgezone_diff,
                "fac_bdydiff_v": self.fac_bdydiff_v,
                "limited_area": self._grid.limited_area,
                "geofac_n2s": self._interpolation_state.geofac_n2s,
                "geofac_grg_x": self._interpolation_state.geofac_grg_x,
                "geofac_grg_y": self._interpolation_state.geofac_grg_y,
                "area": self._cell_params.area,
                "diff_multfac_w": self.diff_multfac_w,
                "type_shear": self.config.shear_type,
                "theta_ref_mc": self._metric_state.theta_ref_mc,
                "thresh_tdiff": self.thresh_tdiff,
                "smallest_vpfloat": constants.VP_NEG_HUGE,
                "zd_vertoffset": self._metric_state.zd_vertoffset,
                "zd_diffcoef": self._metric_state.zd_diffcoef,
                "vcoef": self._metric_state.zd_intcoef,
                "geofac_n2s_c": self._interpolation_state.geofac_n2s_c,
                "geofac_n2s_nbh": self._interpolation_state.geofac_n2s_nbh,
                "inv_dual_edge_length": self._edge_params.inverse_dual_edge_lengths,
                "apply_zdiffusion_t": self.config.apply_zdiffusion_t,
                "rd_o_cvd": self.rd_o_cvd,
            },
            variants={
                "compute_diagnostic_quantities": [self._compute_diagnostic_quantities],
                "apply_to_temperature": [self.config.apply_to_temperature],
            },
            horizontal_sizes={
                "edge_horizontal_start": self._edge_start_lateral_boundary_level_5,
                "edge_horizontal_end": self._edge_end_local,
                "start_2nd_nudge_line_idx_e": self._edge_start_nudging_level_2,
                "w_horizontal_start": self._horizontal_start_index_w_diffusion,
                "w_horizontal_end": self._cell_end_halo,
                "cell_horizontal_start": self._cell_start_nudging,
                "cell_horizontal_end": self._cell_end_local,
                "interior_idx": self._cell_start_interior,
                "halo_idx": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
                "nrdmax": gtx.int32(
                    self._vertical_grid.end_index_of_damping_layer + 1
                ),  # +1 since Fortran includes boundaries
                "nlev": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        self.copy_field_on_cell_k = setup_program(
            backend=backend,
            program=generic_math_operations.copy_field_on_cell_k,
            horizontal_sizes={"horizontal_start": 0, "horizontal_end": self._grid.num_cells},
            vertical_sizes={"vertical_start": 0, "vertical_end": self._grid.num_levels},
        )
        self.copy_field_on_cell_khalf = setup_program(
            backend=backend,
            program=generic_math_operations.copy_field_on_cell_khalf,
            horizontal_sizes={"horizontal_start": 0, "horizontal_end": self._grid.num_cells},
            vertical_sizes={"vertical_start": 0, "vertical_end": self._grid.num_levels + 1},
        )
        self.copy_field_on_edge_k = setup_program(
            backend=backend,
            program=generic_math_operations.copy_field_on_edge_k,
            horizontal_sizes={"horizontal_start": 0, "horizontal_end": self._grid.num_edges},
            vertical_sizes={"vertical_start": 0, "vertical_end": self._grid.num_levels},
        )
        self.scale_k = setup_program(backend=backend, program=scale_k)
        self.setup_fields_for_initial_step = setup_program(
            backend=backend, program=setup_fields_for_initial_step
        )

        self.init_diffusion_local_fields_for_regular_timestep = setup_program(
            backend=backend,
            program=init_diffusion_local_fields_for_regular_timestep,
        )

        self._allocate_local_fields(model_backends.get_allocator(backend))

        self.init_diffusion_local_fields_for_regular_timestep(
            params.K4,
            ndyn_substeps_as_float,
            *params.smagorinski_factor,
            *params.smagorinski_height,
            self._vertical_grid.interface_physical_height,
            self.diff_multfac_vn,
            self.smag_limit,
            self.enh_smag_fac,
        )

        setup_program(
            backend=backend,
            program=diffusion_utils.init_nabla2_factor_in_upper_damping_zone,
            constant_args={
                "physical_heights": self._vertical_grid.interface_physical_height,
                "nshift": 0,
                "heights_1": self._vertical_grid.interface_physical_height[1].as_scalar(),
                "heights_nrd_shift": self._vertical_grid.interface_physical_height[
                    self._vertical_grid.end_index_of_damping_layer + 1
                ].as_scalar(),
            },
            vertical_sizes={
                "vertical_start": 1,
                "vertical_end": gtx.int32(self._vertical_grid.end_index_of_damping_layer + 1),
                "end_index_of_damping_layer": self._vertical_grid.end_index_of_damping_layer,
            },
        )(diff_multfac_n2w=self.diff_multfac_n2w)

    def _allocate_local_fields(self, allocator: gtx_typing.Allocator | None) -> None:
        self.diff_multfac_vn = data_alloc.zero_field(
            self._grid, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        )
        self.diff_multfac_n2w = data_alloc.zero_field(
            self._grid, dims.KHalfDim, dtype=ta.wpfloat, allocator=allocator
        )
        # TODO(pstark): smag_limit, enh_smag_fac and diff_multfac_smag are consumed as vpfloat by the
        #  stencils but produced as wpfloat by diffusion_utils; switch to vpfloat once the producers do.
        self.smag_limit = data_alloc.zero_field(
            self._grid, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        )
        self.enh_smag_fac = data_alloc.zero_field(
            self._grid, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        )
        # TODO(pstark): u_vert and v_vert are consumed as vpfloat by the stencils but produced as
        #  wpfloat by mo_intp_rbf_rbf_vec_interpol_vertex.
        self.u_vert = data_alloc.zero_field(
            self._grid, dims.VertexDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        )
        self.v_vert = data_alloc.zero_field(
            self._grid, dims.VertexDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        )
        self.kh_smag_e = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat, allocator=allocator
        )
        self.z_nabla2_e = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        )
        self.diff_multfac_smag = data_alloc.zero_field(
            self._grid, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        )
        self.vertical_index = data_alloc.index_field(self._grid, dims.KHalfDim, allocator=allocator)
        self.horizontal_cell_index = data_alloc.index_field(
            self._grid, dims.CellDim, allocator=allocator
        )
        self.horizontal_edge_index = data_alloc.index_field(
            self._grid, dims.EdgeDim, allocator=allocator
        )
        self.w_tmp = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KHalfDim, dtype=ta.wpfloat, allocator=allocator
        )
        self.theta_v_tmp = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        )
        self.vn_tmp = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        )

    def _determine_horizontal_domains(self) -> None:
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

    def _smag_values_for_run(
        self, initial_run: bool
    ) -> tuple[
        gtx.Field[gtx.Dims[dims.KDim], wpfloat],
        gtx.Field[gtx.Dims[dims.KDim], wpfloat],
        vpfloat,
    ]:
        """Return the (diff_multfac_vn, smag_limit, smag_offset) values for a run.

        The initial run uses special values, see `run`.
        """
        if initial_run:
            diff_multfac_vn = data_alloc.zero_field(
                self._grid, dims.KDim, dtype=ta.wpfloat, allocator=self._allocator
            )
            smag_limit = data_alloc.zero_field(
                self._grid, dims.KDim, dtype=ta.wpfloat, allocator=self._allocator
            )
            self.setup_fields_for_initial_step(
                self._params.K4,
                self.config.hdiff_efdt_ratio,
                diff_multfac_vn,
                smag_limit,
            )
            smag_offset = vpfloat(0.0)
        else:
            diff_multfac_vn = self.diff_multfac_vn
            smag_limit = self.smag_limit
            smag_offset = self.smag_offset
        return diff_multfac_vn, smag_limit, smag_offset

    def run(
        self,
        diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        prognostic_state: prognostics.PrognosticState,
        dtime: wpfloat,
        initial_run: bool = False,
    ) -> None:
        """
        Do one diffusion step.

        In ICON at the start of the simulation diffusion is run with linit=True:

        'For real-data runs, perform an extra diffusion call before the first time
        step because no other filtering of the interpolated velocity field is done'

        The initial run uses special values for diff_multfac_vn, smag_limit and smag_offset.
        """
        diff_multfac_vn, smag_limit, smag_offset = self._smag_values_for_run(initial_run)

        self.scale_k(self.enh_smag_fac, dtime, self.diff_multfac_smag)

        log.debug("rbf interpolation 1: start")
        self.mo_intp_rbf_rbf_vec_interpol_vertex(
            p_e_in=prognostic_state.vn,
            p_u_out=self.u_vert,
            p_v_out=self.v_vert,
        )
        log.debug("rbf interpolation 1: end")

        # 2.  HALO EXCHANGE -- CALL sync_patch_array_mult u_vert and v_vert
        # TODO(phimuell, muellch): Is asynchronous mode okay here.
        log.debug("communication rbf extrapolation of vn - start")
        self._exchange(
            self.u_vert,
            self.v_vert,
            dim=dims.VertexDim,
            full_exchange=True,
            stream=decomposition.DEFAULT_STREAM,
        )
        log.debug("communication rbf extrapolation of vn - end")

        log.debug("running stencil 01(calculate_nabla2_and_smag_coefficients_for_vn): start")
        self.calculate_nabla2_and_smag_coefficients_for_vn(
            diff_multfac_smag=self.diff_multfac_smag,
            u_vert=self.u_vert,
            v_vert=self.v_vert,
            vn=prognostic_state.vn,
            kh_smag_e=self.kh_smag_e,
            z_nabla2_e=self.z_nabla2_e,
        )
        log.debug("running stencil 01 (calculate_nabla2_and_smag_coefficients_for_vn): end")

        # 5.  HALO EXCHANGE -- CALL sync_patch_array(SYNC_E, z_nabla2_e)
        # ICON: mo_nh_diffusion.f90:853. Fill halo edges before second RBF.
        log.debug("communication of z_nabla2_e - start")
        self._exchange.exchange(
            dims.EdgeDim,
            self.z_nabla2_e,
            stream=decomposition.DEFAULT_STREAM,
        )
        log.debug("communication of z_nabla2_e - end")

        log.debug("2nd rbf interpolation: start")
        self.mo_intp_rbf_rbf_vec_interpol_vertex(
            p_e_in=self.z_nabla2_e, p_u_out=self.u_vert, p_v_out=self.v_vert
        )
        log.debug("2nd rbf interpolation: end")

        # 6.  HALO EXCHANGE -- CALL sync_patch_array_mult (Vertex Fields)
        # TODO(phimuell, muellch): Is asynchronous mode okay here.
        log.debug("communication rbf extrapolation of z_nable2_e - start")
        self._exchange(
            self.u_vert,
            self.v_vert,
            dim=dims.VertexDim,
            full_exchange=True,
            stream=decomposition.DEFAULT_STREAM,
        )
        log.debug("communication rbf extrapolation of z_nable2_e - end")

        # The fused program reads neighbour values of the very prognostic fields it
        # updates (through C2E, E2C, C2E2C and C2E2CO gathers), so the pre-diffusion
        # state is frozen into temporary copies first.
        log.debug("copying prognostic fields vn, w, theta_v: start")
        self.copy_field_on_edge_k(field=prognostic_state.vn, output_field=self.vn_tmp)
        self.copy_field_on_cell_khalf(field=prognostic_state.w, output_field=self.w_tmp)
        self.copy_field_on_cell_k(field=prognostic_state.theta_v, output_field=self.theta_v_tmp)
        log.debug("copying prognostic fields vn, w, theta_v: end")

        log.debug("running fused diffusion program (diffusion_for_prognostic_fields): start")
        self.diffusion_for_prognostic_fields(
            kh_smag=self.kh_smag_e,
            smag_limit=smag_limit,
            smag_offset=smag_offset,
            vn_in=self.vn_tmp,
            diff_multfac_smag=self.diff_multfac_smag,
            u_vert=self.u_vert,
            v_vert=self.v_vert,
            z_nabla2_e=self.z_nabla2_e,
            diff_multfac_vn=diff_multfac_vn,
            w_old=self.w_tmp,
            diff_multfac_n2w=self.diff_multfac_n2w,
            theta_v_in=self.theta_v_tmp,
            vn=prognostic_state.vn,
            w=prognostic_state.w,
            dwdx=diagnostic_state.dwdx,
            dwdy=diagnostic_state.dwdy,
            theta_v=prognostic_state.theta_v,
            exner=prognostic_state.exner,
            div_ic=diagnostic_state.div_ic,
            hdef_ic=diagnostic_state.hdef_ic,
            compute_diagnostic_quantities=self._compute_diagnostic_quantities,
            apply_to_temperature=self.config.apply_to_temperature,
        )
        log.debug("running fused diffusion program (diffusion_for_prognostic_fields): end")

        log.debug("communication of prognostic.vn : start")
        self._exchange.exchange(
            dims.EdgeDim,
            prognostic_state.vn,
            stream=decomposition.DEFAULT_STREAM,
        )
        log.debug("communication of prognostic.vn - end")

        # The halo exchange can be skipped in the case of NWP or AES physics because the column-wise physics
        # computations, which happen right after diffusion, do not require the halo lines to be correct and there
        # is another halo exchange after the physics are applied.
        if initial_run or self.config.iforcing not in (ForcingType.NWP, ForcingType.AES):
            if self.config.apply_to_temperature:
                log.debug("communication of prognostic cell fields: theta and exner - start")
                self._exchange.exchange(
                    dims.CellDim,
                    prognostic_state.theta_v,
                    prognostic_state.exner,
                    stream=decomposition.DEFAULT_STREAM,
                )
                log.debug("communication of prognostic cell fields: theta and exner - done")

            log.debug("communication of prognostic cell field: w - start")
            self._exchange.exchange(
                dims.CellDim,
                prognostic_state.w,
                stream=decomposition.DEFAULT_STREAM,
            )
            log.debug("communication of prognostic cell field: w - done")
