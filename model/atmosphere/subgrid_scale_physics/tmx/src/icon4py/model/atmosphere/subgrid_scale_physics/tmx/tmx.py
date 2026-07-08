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
import typing
from typing import Any, Final

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.apply_explicit_vertical_diffusion_cells import (
    apply_explicit_vertical_diffusion_cells,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.apply_horizontal_diffusion_and_update_scalar import (
    apply_horizontal_diffusion_and_update_scalar,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.apply_w_horizontal_diffusion_and_update import (
    apply_w_horizontal_diffusion_and_update,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.assign_constant_viscosity import (
    assign_constant_viscosity,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_brunt_vaisala_frequency import (
    compute_brunt_vaisala_frequency,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_energy_from_temperature import (
    compute_energy_from_temperature,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_scalar_nabla2_flux import (
    compute_scalar_nabla2_flux,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_shear_and_div_of_stress import (
    compute_shear_and_div_of_stress,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_smagorinsky_viscosity import (
    compute_smagorinsky_viscosity,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_static_energy import (
    compute_static_energy,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_surface_energy_flux import (
    compute_surface_energy_flux,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_surface_flux_rhs import (
    compute_surface_flux_rhs,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_temperature_from_energy_and_tendency import (
    compute_temperature_from_energy_and_tendency,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_vertical_integral_diagnostics import (
    compute_vertical_integral_diagnostics,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_virtual_potential_temperature import (
    compute_virtual_potential_temperature,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_vn_from_uv import (
    compute_vn_from_uv,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_vn_horizontal_stress_tendency import (
    compute_vn_horizontal_stress_tendency,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_vn_vertical_diffusion_rhs import (
    compute_vn_vertical_diffusion_rhs,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_w_horizontal_stress_tendency import (
    compute_w_horizontal_stress_tendency,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_w_vertical_diffusion_rhs import (
    compute_w_vertical_diffusion_rhs,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.init_louis_scaling_factor import (
    init_louis_scaling_factor,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.init_smagorinsky_mixing_length import (
    init_smagorinsky_mixing_length,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.interpolate_cell_to_half_levels import (
    interpolate_cell_to_half_levels,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.interpolate_inverse_density_to_edges import (
    interpolate_inverse_density_to_edges,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.interpolate_km_to_edges import (
    interpolate_km_to_edges,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.interpolate_km_to_full_level_cells import (
    interpolate_km_to_full_level_cells,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.interpolate_km_to_vertices import (
    interpolate_km_to_vertices,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.interpolate_shear_to_half_level_cells import (
    interpolate_shear_to_half_level_cells,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.interpolate_vn_to_half_levels_with_boundary import (
    interpolate_vn_to_half_levels_with_boundary,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.modify_w_diffusion_matrix_boundary import (
    modify_w_diffusion_matrix_boundary,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.prepare_tridiagonal_matrix_cells import (
    prepare_tridiagonal_matrix_cells,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.prepare_tridiagonal_matrix_cells_half import (
    prepare_tridiagonal_matrix_cells_half,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.prepare_tridiagonal_matrix_edges import (
    prepare_tridiagonal_matrix_edges,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.solve_vertical_diffusion_cells import (
    solve_vertical_diffusion_cells,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.solve_vertical_diffusion_edges import (
    solve_vertical_diffusion_edges,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.update_exchange_coefficient_diagnostics import (
    update_exchange_coefficient_diagnostics,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.update_horizontal_wind import (
    update_horizontal_wind,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.update_temperature_with_dissipation_heating import (
    update_temperature_with_dissipation_heating,
)
from icon4py.model.common import constants, dimension as dims, model_backends
from icon4py.model.common.config import options as common_conf_opt
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import base as base_grid, horizontal as h_grid
from icon4py.model.common.interpolation.stencils.cell_2_edge_interpolation import (
    cell_2_edge_interpolation,
)
from icon4py.model.common.interpolation.stencils.compute_cell_2_vertex_interpolation import (
    compute_cell_2_vertex_interpolation,
)
from icon4py.model.common.interpolation.stencils.compute_tangential_wind import (
    compute_tangential_wind_wp,
)
from icon4py.model.common.interpolation.stencils.edge_2_cell_vector_rbf_interpolation import (
    edge_2_cell_vector_rbf_interpolation,
)
from icon4py.model.common.interpolation.stencils.interpolate_to_cell_center import (
    interpolate_to_cell_center,
)
from icon4py.model.common.interpolation.stencils.mo_intp_rbf_rbf_vec_interpol_vertex import (
    mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.model.common.math.stencils.generic_math_operations import (
    compute_reciprocal_on_cell_k,
    subtract_cell_field_on_cell_k,
)
from icon4py.model.common.math.stencils.init_cell_kdim_field_with_zero_wp import (
    init_cell_kdim_field_with_zero_wp,
)
from icon4py.model.common.model_options import setup_program
from icon4py.model.common.utils import data_allocation as data_alloc


if typing.TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    import icon4py.model.common.grid.states as grid_states
    from icon4py.model.common import field_type_aliases as fa, type_alias as ta
    from icon4py.model.common.grid import vertical as v_grid


"""
TMX (AES turbulent mixing) scheme, ported from ICON's ``src/atm_phy_aes/tmx``.

This module provides the configuration (:class:`TmxConfig`), derived parameters
(:class:`TmxParams`) and the granule class (:class:`Tmx`). Config default
values are taken from ``vdiff_config_init`` in ``mo_turb_vdiff_config.f90``.
"""


log = logging.getLogger(__name__)


class TurbulenceSolverType(int, enum.Enum):
    """
    Type of the vertical diffusion solver.

    Note: Called ``solver_type`` in ``mo_turb_vdiff_config.f90``.
    """

    EXPLICIT = 1  # explicit time stepping
    IMPLICIT = 2  # implicit time stepping


class EnergyType(int, enum.Enum):
    """
    Type of energy diffused by the temperature (heat) diffusion.

    Note: Called ``energy_type`` in ``mo_turb_vdiff_config.f90``.
    """

    DRY_STATIC = 1  # dry static energy cp*T + g*z
    INTERNAL = 2  # internal energy cv*T


# number of members of the Fortran t_vdiff_config derived type
# (mo_turb_vdiff_config.f90); the echoed aes_vdf_nml namelist holds this many
# values per domain, in declaration order. Must be kept in sync with the
# 'unnamed_index' positions of the TmxConfig options below.
_T_VDIFF_CONFIG_NUM_MEMBERS: Final = 42

# position of 'use_tmx' in t_vdiff_config, used as an order canary
_T_VDIFF_CONFIG_USE_TMX_INDEX: Final = 22


@dataclasses.dataclass(kw_only=True)
class TmxConfig:
    """
    Contains the necessary parameters to configure a tmx run.

    Encapsulates namelist parameters and derived parameters.
    Values should be read from configuration.
    Default values are taken from ``vdiff_config_init`` in the corresponding ICON
    Fortran module ``mo_turb_vdiff_config.f90`` (namelist ``aes_vdf_nml``).
    """

    solver_type: typing.Annotated[
        TurbulenceSolverType,
        common_conf_opt.ConfigOption(
            description="Type of the vertical diffusion solver (explicit or implicit).",
            icon_equivalent=common_conf_opt.IconOption(
                "solver_type", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=23
            ),
        ),
    ] = TurbulenceSolverType.IMPLICIT

    energy_type: typing.Annotated[
        EnergyType,
        common_conf_opt.ConfigOption(
            description="Type of energy diffused by the heat diffusion (dry static or internal).",
            icon_equivalent=common_conf_opt.IconOption(
                "energy_type", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=24
            ),
        ),
    ] = EnergyType.INTERNAL

    dissipation_factor: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Scaling factor for the kinetic energy dissipation heating.",
            icon_equivalent=common_conf_opt.IconOption(
                "dissipation_factor", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=25
            ),
        ),
    ] = 1.0

    use_louis: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, use the Louis (1979) stability correction function "
            "instead of the classic (Lilly 1962) one.",
            icon_equivalent=common_conf_opt.IconOption(
                "use_louis", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=26
            ),
        ),
    ] = True

    use_louis_land: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If False, exclude cells with more than 50% land fraction "
            "from the Louis stability correction.",
            icon_equivalent=common_conf_opt.IconOption(
                "use_louis_land", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=27
            ),
        ),
    ] = True

    use_louis_ice: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If False, exclude cells with more than 50% sea-ice fraction "
            "from the Louis stability correction.",
            icon_equivalent=common_conf_opt.IconOption(
                "use_louis_ice", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=28
            ),
        ),
    ] = True

    louis_constant_b: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Louis constant b of the Louis stability correction function.",
            icon_equivalent=common_conf_opt.IconOption(
                "louis_constant_b", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=29
            ),
        ),
    ] = 4.2

    use_km_const: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, use a constant exchange coefficient instead of the "
            "Smagorinsky model.",
            icon_equivalent=common_conf_opt.IconOption(
                "use_km_const", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=30
            ),
        ),
    ] = False

    km_const: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Constant exchange coefficient used if 'use_km_const' is True [m^2/s].",
            icon_equivalent=common_conf_opt.IconOption(
                "km_const", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=31
            ),
        ),
    ] = 1.0

    use_scale_turb_energy_flux: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, scale the turbulent energy flux by 'scale_turb_energy_flux'.",
            icon_equivalent=common_conf_opt.IconOption(
                "use_scale_turb_energy_flux", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=32
            ),
        ),
    ] = False

    scale_turb_energy_flux: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Scaling factor for the turbulent energy flux used if "
            "'use_scale_turb_energy_flux' is True.",
            icon_equivalent=common_conf_opt.IconOption(
                "scale_turb_energy_flux", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=33
            ),
        ),
    ] = 1.0

    smag_constant: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Smagorinsky constant Cs of the Smagorinsky-Lilly eddy viscosity model.",
            icon_equivalent=common_conf_opt.IconOption(
                "smag_constant", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=34
            ),
        ),
    ] = 0.23

    turb_prandtl: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Turbulent Prandtl number.",
            icon_equivalent=common_conf_opt.IconOption(
                "turb_prandtl", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=35
            ),
        ),
    ] = 0.33333333333  # exact literal from mo_turb_vdiff_config.f90 (not 1/3)

    km_min: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Minimum mass-weighted turbulent viscosity [kg/(m s)].",
            icon_equivalent=common_conf_opt.IconOption(
                "km_min", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=37
            ),
        ),
    ] = 0.001

    max_turb_scale: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Maximum turbulence length scale [m].",
            icon_equivalent=common_conf_opt.IconOption(
                "max_turb_scale", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=38
            ),
        ),
    ] = 300.0

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def from_fortran_dict(cls, atmo_dict: dict[str, Any], **overrides: Any) -> TmxConfig:
        """
        Construct the configuration from the echoed ICON namelists.

        ``aes_vdf_nml`` is a derived-type namelist (``t_vdiff_config``), which
        ICON echoes as an anonymous positional array of the member values in
        declaration order, so the options are located by ``unnamed_index``
        (pinned to mo_turb_vdiff_config.f90) instead of by name. Only the
        first domain is read. The guards below make a change of the Fortran
        type fail loudly instead of silently mis-assigning values.
        """
        flat = atmo_dict["aes_vdf_nml"]["aes_vdf_config"]
        if len(flat) % _T_VDIFF_CONFIG_NUM_MEMBERS != 0:
            raise ValueError(
                f"'aes_vdf_config' has {len(flat)} values, not a multiple of the "
                f"{_T_VDIFF_CONFIG_NUM_MEMBERS} members of t_vdiff_config: the Fortran "
                "type changed and the pinned 'unnamed_index' positions must be revised."
            )
        use_tmx = flat[_T_VDIFF_CONFIG_USE_TMX_INDEX]
        if use_tmx is not True:
            raise ValueError(
                f"expected 'use_tmx' (True) at position {_T_VDIFF_CONFIG_USE_TMX_INDEX} of "
                f"'aes_vdf_config', found {use_tmx!r}: either the run does not use tmx or "
                "the t_vdiff_config member order changed."
            )
        return common_conf_opt.construct_config_from_icon(cls, atmo_dict, **overrides)

    def _validate(self) -> None:
        """Apply consistency checks and validation on configuration parameters."""
        self.solver_type = TurbulenceSolverType(self.solver_type)
        self.energy_type = EnergyType(self.energy_type)

        if self.turb_prandtl <= 0.0:
            raise ValueError(
                f"Invalid argument 'turb_prandtl': should be positive, got {self.turb_prandtl}."
            )
        if self.km_min < 0.0:
            raise ValueError(
                f"Invalid argument 'km_min': should be non-negative, got {self.km_min}."
            )


@dataclasses.dataclass(frozen=True)
class TmxParams:
    """Calculates derived quantities depending on the tmx config."""

    config: dataclasses.InitVar[TmxConfig]
    rturb_prandtl: Final[float] = dataclasses.field(init=False)
    """Reciprocal turbulent Prandtl number (``rturb_prandtl`` in mo_turb_vdiff_config.f90)."""
    von_karman: Final[float] = 0.4
    """Von Karman constant (``ckap`` in mo_turb_vdiff_params.f90 and the local ``kappa`` in
    ``compute_mixing_length`` of mo_tmx_smagorinsky.f90)."""
    mean_cell_area_r2b8: Final[float] = 97294071.23714285
    """Global mean cell area of the R2B8 grid [m^2] (``mean_area_R2B8`` in
    ``compute_scaling_factor_louis`` of mo_tmx_smagorinsky.f90)."""

    def __post_init__(self, config: TmxConfig) -> None:
        object.__setattr__(self, "rturb_prandtl", 1.0 / config.turb_prandtl)


class Tmx:
    """
    TMX (AES turbulent mixing) granule.

    Port of the ``t_vdf`` / ``t_vdf_atmo`` classes driven by ``mo_vdf.f90``.
    Implements the initialization (``Smagorinsky_init`` in
    mo_tmx_smagorinsky.f90 plus the time-independent height above ground) and
    the full atmospheric ``Compute`` sequence of mo_vdf.f90 (:meth:`run`):
    Stage A, the Smagorinsky diagnostics (``Compute_diagnostics`` in
    mo_vdf_atmo.f90), the scalar diffusion stages B
    (``Compute_diffusion_hydrometeors``) and C
    (``Compute_diffusion_temperature``), the momentum diffusion stages D
    (``Compute_diffusion_hor_wind``) and E (``Compute_diffusion_vert_wind``),
    the dissipation-heating energy update F (``Update_energy_tendencies``)
    and the end-of-step diagnostics G (``Update_diagnostics``). The surface
    scheme (``this%sfc%Compute`` in the Fortran) is out of scope; the
    grid-mean surface fluxes are prescribed inputs
    (:class:`tmx_states.TmxSurfaceFluxState`).

    Persistent derived fields (computed once at construction, read every step):
    - ``mix_len_sq``: squared Smagorinsky mixing length (half-level cells),
    - ``louis_factor``: cell-area scaling factor of the Louis constant b
      (zero if ``config.use_louis`` is False, matching the Fortran init),
    - ``ghf``: geometric height of the full levels above the surface
      (recomputed every step in the Fortran code, but time-independent).

    Note: the land and sea-ice fractions (``fract_land`` / ``fract_ice``) are
    not part of :class:`tmx_states.TmxInputState` yet; the granule allocates
    them as zero fields (aqua-planet setup). They are only read if the Louis
    stability correction is switched off over land or ice
    (``use_louis_land`` / ``use_louis_ice`` = False).
    """

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        config: TmxConfig,
        params: TmxParams,
        vertical_grid: v_grid.VerticalGrid | None,
        metric_state: tmx_states.TmxMetricState,
        interpolation_state: tmx_states.TmxInterpolationState,
        edge_params: grid_states.EdgeParams,
        cell_params: grid_states.CellParams,
        backend: gtx_typing.Backend
        | model_backends.DeviceType
        | model_backends.BackendDescriptor
        | None,
        exchange: decomposition.ExchangeRuntime = decomposition.single_node_exchange,
    ) -> None:
        self._allocator = model_backends.get_allocator(backend)
        self._exchange = exchange
        self.config = config
        self._params = params
        self._grid = grid
        # not used by the Smagorinsky diagnostics (Stage A); kept for parity with
        # the other granules and for the later tmx stages.
        self._vertical_grid = vertical_grid
        self._metric_state = metric_state
        self._interpolation_state = interpolation_state
        self._edge_params = edge_params
        self._cell_params = cell_params

        assert self._cell_params.area is not None

        # scaling factor of the turbulent energy flux (``zfactor`` in
        # 'Compute_diffusion_temperature', mo_vdf.f90)
        self._zfactor = (
            self.config.scale_turb_energy_flux if self.config.use_scale_turb_energy_flux else 1.0
        )
        # compile-time variant selector of the energy conversion stencils
        use_internal_energy = self.config.energy_type == EnergyType.INTERNAL

        self.halo_exchange_wait = decomposition.create_halo_exchange_wait(self._exchange)

        num_levels = self._grid.num_levels
        self._determine_horizontal_domains()
        self._allocate_local_fields()

        # 2D views on the quadratic extrapolation coefficients (rows 0..2 in
        # Fortran coefficient order, see the TmxMetricState docstrings); the
        # stencils take them as three separate 2D fields.
        wgtfacq1_c = self._coefficient_fields(self._metric_state.wgtfacq1_c, dims.CellDim)
        wgtfacq_c = self._coefficient_fields(self._metric_state.wgtfacq_c, dims.CellDim)
        wgtfacq1_e = self._coefficient_fields(self._metric_state.wgtfacq1_e, dims.EdgeDim)
        wgtfacq_e = self._coefficient_fields(self._metric_state.wgtfacq_e, dims.EdgeDim)
        # geometric height of the surface (bottom half level), 2D slice of z_ifc
        z_ifc_sfc = gtx.as_field(
            (dims.CellDim,),
            self._metric_state.z_ifc.ndarray[:, num_levels],
            allocator=self._allocator,
        )

        # ---------------------------------------------------------------------
        # Init programs (run once, at the end of __init__)
        # ---------------------------------------------------------------------
        # compute_mixing_length (mo_tmx_smagorinsky.f90): cells rl 3..min_rlcell_int,
        # all half levels
        self.init_smagorinsky_mixing_length = setup_program(
            backend=backend,
            program=init_smagorinsky_mixing_length,
            constant_args={
                "dz_ic": self._metric_state.ddqz_z_half,
                "geopot_agl_ic": self._metric_state.geopot_agl_ifc,
                "cell_area": self._cell_params.area,
                "smag_constant": self.config.smag_constant,
                "max_turb_scale": self.config.max_turb_scale,
                "grav": constants.GRAV,
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_lateral_boundary_level_3,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels + 1),
            },
            offset_provider={},
        )
        # compute_scaling_factor_louis (mo_tmx_smagorinsky.f90): cells rl
        # 3..min_rlcell_int
        self.init_louis_scaling_factor = setup_program(
            backend=backend,
            program=init_louis_scaling_factor,
            constant_args={"cell_area": self._cell_params.area},
            horizontal_sizes={
                "horizontal_start": self._cell_start_lateral_boundary_level_3,
                "horizontal_end": self._cell_end_local,
            },
            offset_provider={},
        )
        # compute_geopotential_height_above_ground (mo_vdf_atmo.f90): tmx t_domain
        # cells (grf_bdywidth_c + 1 .. min_rlcell_int), all full levels.
        # ghf = z_mc - z_ifc_sfc; despite the Fortran name it is the geometric
        # height in meters (gravity is only applied later, e.g. in
        # compute_static_energy). z_ifc_sfc is the surface slice z_ifc[:, nlev],
        # passed as a 2D field because GT4Py offsets are relative and cannot
        # address a fixed absolute K row.
        self.init_height_above_ground = setup_program(
            backend=backend,
            program=subtract_cell_field_on_cell_k,
            constant_args={"minuend": self._metric_state.z_mc, "subtrahend_cell": z_ifc_sfc},
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )

        # ---------------------------------------------------------------------
        # Stage A step programs, in the Fortran call order of Compute_diagnostics
        # (mo_vdf_atmo.f90 l. 343-482)
        # ---------------------------------------------------------------------
        # compute_static_energy: tmx t_domain cells, all full levels
        self.compute_static_energy = setup_program(
            backend=backend,
            program=compute_static_energy,
            constant_args={
                "height_above_ground": self.ghf,
                "spec_heat": constants.CPD,
                "grav": constants.GRAV,
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        # get_virtual_potential_temperature: cells rl 3..min_rlcell_int, all full levels
        self.compute_virtual_potential_temperature = setup_program(
            backend=backend,
            program=compute_virtual_potential_temperature,
            horizontal_sizes={
                "horizontal_start": self._cell_start_lateral_boundary_level_3,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        # vert_intp_full2half_cell_3d (rho -> rho_ic): cells rl 2..min_rlcell_int-2,
        # all half levels (top and bottom rows are extrapolated)
        self.interpolate_cell_to_half_levels = setup_program(
            backend=backend,
            program=interpolate_cell_to_half_levels,
            constant_args={
                "wgtfac_c": self._metric_state.wgtfac_c,
                "wgtfacq1_c_1": wgtfacq1_c[0],
                "wgtfacq1_c_2": wgtfacq1_c[1],
                "wgtfacq1_c_3": wgtfacq1_c[2],
                "wgtfacq_c_1": wgtfacq_c[0],
                "wgtfacq_c_2": wgtfacq_c[1],
                "wgtfacq_c_3": wgtfacq_c[2],
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_lateral_boundary_level_2,
                "horizontal_end": self._cell_end_halo_level_2,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels + 1),
                "nlev": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        # brunt_vaisala_freq: cells rl 3..min_rlcell_int, half levels 1..nlev-1
        # (top and bottom rows are not computed)
        self.compute_brunt_vaisala_frequency = setup_program(
            backend=backend,
            program=compute_brunt_vaisala_frequency,
            constant_args={
                "wgtfac_c": self._metric_state.wgtfac_c,
                "inv_ddqz_z_half": self._metric_state.inv_ddqz_z_half,
                "grav": constants.GRAV,
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_lateral_boundary_level_3,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(1),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        # compute_normal_velocity_edge: edges rl grf_bdywidth_e+1..min_rledge_int,
        # all full levels
        self.compute_vn_from_uv = setup_program(
            backend=backend,
            program=compute_vn_from_uv,
            constant_args={
                "primal_normal_cell_x": self._edge_params.primal_normal_cell[0],
                "primal_normal_cell_y": self._edge_params.primal_normal_cell[1],
                "c_lin_e": self._interpolation_state.c_lin_e,
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_nudging_level_2,
                "horizontal_end": self._edge_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        # cells2verts_scalar (w -> w_vert): vertices rl 2..min_rlvert_int,
        # all half levels
        self.compute_cell_2_vertex_interpolation = setup_program(
            backend=backend,
            program=compute_cell_2_vertex_interpolation,
            constant_args={"c_int": self._interpolation_state.cells_aw_verts},
            horizontal_sizes={
                "horizontal_start": self._vertex_start_lateral_boundary_level_2,
                "horizontal_end": self._vertex_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels + 1),
            },
            offset_provider=self._grid.connectivities,
        )
        # cells2edges_scalar (w -> w_ie): edges rl 2..min_rledge_int-2, all half levels
        self.cell_2_edge_interpolation = setup_program(
            backend=backend,
            program=cell_2_edge_interpolation,
            constant_args={"coeff": self._interpolation_state.c_lin_e},
            horizontal_sizes={
                "horizontal_start": self._edge_start_lateral_boundary_level_2,
                "horizontal_end": self._edge_end_halo_level_2,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels + 1),
            },
            offset_provider=self._grid.connectivities,
        )
        # rbf_vec_interpol_vertex (vn -> u_vert, v_vert): vertices rl
        # 2..min_rlvert_int, all full levels
        self.mo_intp_rbf_rbf_vec_interpol_vertex = setup_program(
            backend=backend,
            program=mo_intp_rbf_rbf_vec_interpol_vertex,
            constant_args={
                "ptr_coeff_1": self._interpolation_state.rbf_coeff_v1,
                "ptr_coeff_2": self._interpolation_state.rbf_coeff_v2,
            },
            horizontal_sizes={
                "horizontal_start": self._vertex_start_lateral_boundary_level_2,
                "horizontal_end": self._vertex_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        # interpolate_normal_velocity_edge_interface (vn -> vn_ie): edges rl
        # 2..min_rledge_int-3, all half levels. There is no icon4py zone for
        # min_rledge_int-3 (third halo line); h_grid.Zone.END is the closest
        # more-inclusive bound (identical on a single node, where there are no
        # halo lines; the extra halo rows are unused boundary values anyway).
        self.interpolate_vn_to_half_levels_with_boundary = setup_program(
            backend=backend,
            program=interpolate_vn_to_half_levels_with_boundary,
            constant_args={
                "wgtfac_e": self._metric_state.wgtfac_e,
                "wgtfacq1_e_1": wgtfacq1_e[0],
                "wgtfacq1_e_2": wgtfacq1_e[1],
                "wgtfacq1_e_3": wgtfacq1_e[2],
                "wgtfacq_e_1": wgtfacq_e[0],
                "wgtfacq_e_2": wgtfacq_e[1],
                "wgtfacq_e_3": wgtfacq_e[2],
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_lateral_boundary_level_2,
                "horizontal_end": self._edge_end_end,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels + 1),
                "nlev": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        # rbf_vec_interpol_edge (vn_ie -> vt_ie): edges rl 3..min_rledge_int-2,
        # all half levels
        self.compute_tangential_wind_wp = setup_program(
            backend=backend,
            program=compute_tangential_wind_wp,
            constant_args={"rbf_vec_coeff_e": self._interpolation_state.rbf_coeff_e},
            horizontal_sizes={
                "horizontal_start": self._edge_start_lateral_boundary_level_3,
                "horizontal_end": self._edge_end_halo_level_2,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels + 1),
            },
            offset_provider=self._grid.connectivities,
        )
        # compute_velocity_gradient_tensor + compute_shear: edges rl
        # 4..min_rledge_int-2, all full levels
        self.compute_shear_and_div_of_stress = setup_program(
            backend=backend,
            program=compute_shear_and_div_of_stress,
            constant_args={
                "primal_normal_vert_x": self._edge_params.primal_normal_vert[0],
                "primal_normal_vert_y": self._edge_params.primal_normal_vert[1],
                "dual_normal_vert_x": self._edge_params.dual_normal_vert[0],
                "dual_normal_vert_y": self._edge_params.dual_normal_vert[1],
                "tangent_orientation": self._edge_params.tangent_orientation,
                "inv_primal_edge_length": self._edge_params.inverse_primal_edge_lengths,
                "inv_vert_vert_length": self._edge_params.inverse_vertex_vertex_lengths,
                "inv_dual_edge_length": self._edge_params.inverse_dual_edge_lengths,
                "inv_ddqz_z_full_e": self._metric_state.inv_ddqz_z_full_e,
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_lateral_boundary_level_4,
                "horizontal_end": self._edge_end_halo_level_2,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        # get_horizontal_divergence_strain_rate_cell (div_of_stress -> div_c):
        # cells rl grf_bdywidth_c+1..min_rlcell_int-1, all full levels.
        # Note: the common stencil is vp-typed; tmx fields are wpfloat, so this
        # only works while vpfloat == wpfloat (i.e. without mixed precision).
        self.interpolate_to_cell_center = setup_program(
            backend=backend,
            program=interpolate_to_cell_center,
            constant_args={"e_bln_c_s": self._interpolation_state.e_bln_c_s},
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_halo,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        # interpolate_rate_of_strain_full2half_edge2cell (shear -> mech_prod):
        # cells rl 3..min_rlcell_int-1, half levels 1..nlev-1 (top and bottom
        # rows are not computed)
        self.interpolate_shear_to_half_level_cells = setup_program(
            backend=backend,
            program=interpolate_shear_to_half_level_cells,
            constant_args={
                "e_bln_c_s": self._interpolation_state.e_bln_c_s,
                "wgtfac_c": self._metric_state.wgtfac_c,
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_lateral_boundary_level_3,
                "horizontal_end": self._cell_end_halo,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(1),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        # Smagorinsky_model / Assign_constant_eddy_viscosity (-> km_ic, kh_ic):
        # cells rl 3..min_rlcell_int, all half levels (rows 0 and nlev are copies
        # of the adjacent interior rows, fused into the stencils)
        if self.config.use_km_const:
            self.compute_viscosity = setup_program(
                backend=backend,
                program=assign_constant_viscosity,
                constant_args={
                    "km_const": self.config.km_const,
                    "rturb_prandtl": self._params.rturb_prandtl,
                },
                horizontal_sizes={
                    "horizontal_start": self._cell_start_lateral_boundary_level_3,
                    "horizontal_end": self._cell_end_local,
                },
                vertical_sizes={
                    "vertical_start": gtx.int32(0),
                    "vertical_end": gtx.int32(num_levels + 1),
                    "nlev": gtx.int32(num_levels),
                },
                offset_provider={},
            )
        else:
            self.compute_viscosity = setup_program(
                backend=backend,
                program=compute_smagorinsky_viscosity,
                constant_args={
                    "mixing_length_sq": self.mix_len_sq,
                    "scaling_factor_louis": self.louis_factor,
                    "fract_land": self.fract_land,
                    "fract_ice": self.fract_ice,
                    "rturb_prandtl": self._params.rturb_prandtl,
                    "louis_constant_b": self.config.louis_constant_b,
                    "use_louis": self.config.use_louis,
                    "use_louis_land": self.config.use_louis_land,
                    "use_louis_ice": self.config.use_louis_ice,
                },
                horizontal_sizes={
                    "horizontal_start": self._cell_start_lateral_boundary_level_3,
                    "horizontal_end": self._cell_end_local,
                },
                vertical_sizes={
                    "vertical_start": gtx.int32(0),
                    "vertical_end": gtx.int32(num_levels + 1),
                    "nlev": gtx.int32(num_levels),
                },
                offset_provider={},
            )
        # interpolate_eddy_viscosity2cell (km_ic -> km_c): cells rl
        # grf_bdywidth_c..min_rlcell_int-1, all full levels (halo cells computed
        # on purpose, km_c is used in the diffusion later)
        self.interpolate_km_to_full_level_cells = setup_program(
            backend=backend,
            program=interpolate_km_to_full_level_cells,
            constant_args={"km_min": self.config.km_min},
            horizontal_sizes={
                "horizontal_start": self._cell_start_lateral_boundary_level_4,
                "horizontal_end": self._cell_end_halo,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        # interpolate_eddy_viscosity2half_vertex (km_ic -> km_iv): vertices rl
        # 5 (= max_rlvert)..min_rlvert_int-1, all half levels
        self.interpolate_km_to_vertices = setup_program(
            backend=backend,
            program=interpolate_km_to_vertices,
            constant_args={
                "cells_aw_verts": self._interpolation_state.cells_aw_verts,
                "km_min": self.config.km_min,
            },
            horizontal_sizes={
                "horizontal_start": self._vertex_start_nudging,
                "horizontal_end": self._vertex_end_halo,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels + 1),
            },
            offset_provider=self._grid.connectivities,
        )
        # interpolate_eddy_viscosity2half_edge (km_ic -> km_ie): edges rl
        # grf_bdywidth_e..min_rledge_int-1, all half levels
        self.interpolate_km_to_edges = setup_program(
            backend=backend,
            program=interpolate_km_to_edges,
            constant_args={
                "c_lin_e": self._interpolation_state.c_lin_e,
                "km_min": self.config.km_min,
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_nudging,
                "horizontal_end": self._edge_end_halo,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels + 1),
            },
            offset_provider=self._grid.connectivities,
        )

        self._setup_scalar_diffusion_programs(backend, num_levels, use_internal_energy)
        self._setup_momentum_diffusion_programs(backend, num_levels)
        self._setup_energy_and_diagnostics_programs(backend, num_levels)

        # ---------------------------------------------------------------------
        # Run the init programs (Smagorinsky_init in mo_tmx_smagorinsky.f90 and
        # compute_geopotential_height_above_ground in mo_vdf_atmo.f90)
        # ---------------------------------------------------------------------
        self.init_smagorinsky_mixing_length(mixing_length_sq=self.mix_len_sq)
        if self.config.use_louis:
            # the Fortran init only computes the Louis scaling factor if the
            # Louis stability correction is enabled; the field stays zero otherwise
            self.init_louis_scaling_factor(scaling_factor_louis=self.louis_factor)
        self.init_height_above_ground(difference=self.ghf)

    def _setup_scalar_diffusion_programs(
        self,
        backend: gtx_typing.Backend
        | model_backends.DeviceType
        | model_backends.BackendDescriptor
        | None,
        num_levels: int,
        use_internal_energy: bool,
    ) -> None:
        """
        Bind the Stage B + C step programs (scalar diffusion:
        Compute_diffusion_hydrometeors l. 585 and Compute_diffusion_temperature
        l. 912 in mo_vdf.f90). All cell loops run on the tmx t_domain cell
        range (grf_bdywidth_c + 1 .. min_rlcell_int), all full levels, unless
        noted otherwise.
        """
        # CALL init(...) zero fills (whole array, tendencies before the solves)
        self.init_cell_kdim_field_with_zero = setup_program(
            backend=backend,
            program=init_cell_kdim_field_with_zero_wp,
            horizontal_sizes={
                "horizontal_start": gtx.int32(0),
                "horizontal_end": self._cell_end_end,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        # inverse air mass (the ``inv_mair`` loops of mo_vdf.f90): inv_mair
        # scales the rows of the vertical diffusion matrix
        # ('prepare_diffusion_matrix') and the surface flux right-hand side
        self.compute_inverse_air_mass = setup_program(
            backend=backend,
            program=compute_reciprocal_on_cell_k,
            constant_args={"output_field": self._inv_air_mass},
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        # prepare_diffusion_matrix (zk = kh_ic): two bindings because zprefac is
        # inlined at compile time (hydrometeors: 1, energy: zfactor)
        prepare_tridiagonal_matrix_constant_args = {
            "inv_mair": self._inv_air_mass,
            "inv_dz": self._metric_state.inv_ddqz_z_half,
        }
        prepare_tridiagonal_matrix_sizes = dict(
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        self.prepare_tridiagonal_matrix_hydrometeors = setup_program(
            backend=backend,
            program=prepare_tridiagonal_matrix_cells,
            constant_args={**prepare_tridiagonal_matrix_constant_args, "zprefac": 1.0},
            **prepare_tridiagonal_matrix_sizes,
        )
        self.prepare_tridiagonal_matrix_energy = setup_program(
            backend=backend,
            program=prepare_tridiagonal_matrix_cells,
            constant_args={**prepare_tridiagonal_matrix_constant_args, "zprefac": self._zfactor},
            **prepare_tridiagonal_matrix_sizes,
        )
        # rhs(nlev) = -sfc_flx * prefac * inv_mair(nlev): single bottom K row;
        # the other rows of self._rhs are zero-allocated and never written
        # (Fortran: rhs(1) = +top_flx * inv_mair(1) with top_flx == 0)
        self.compute_surface_flux_rhs = setup_program(
            backend=backend,
            program=compute_surface_flux_rhs,
            constant_args={"inv_air_mass": self._inv_air_mass},
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(num_levels - 1),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        # vertical solve: 'diffuse_vertical_implicit' or 'diffuse_vertical_explicit'
        # in mo_tmx_numerics.f90, selected by the configured solver type
        solve_vertical_diffusion_constant_args = {
            "a": self._matrix_a,
            "b": self._matrix_b,
            "c": self._matrix_c,
            "rhs": self._rhs,
        }
        self.solve_vertical_diffusion = setup_program(
            backend=backend,
            program=solve_vertical_diffusion_cells
            if self.config.solver_type == TurbulenceSolverType.IMPLICIT
            else apply_explicit_vertical_diffusion_cells,
            constant_args=solve_vertical_diffusion_constant_args,
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        # nabla2_e = kh_ie * grad_horiz(scalar): edges rl grf_bdywidth_e..
        # min_rledge_int-1 (halo edges computed on purpose, the divergence is
        # taken on halo-adjacent cells afterwards)
        self.compute_scalar_nabla2_flux = setup_program(
            backend=backend,
            program=compute_scalar_nabla2_flux,
            constant_args={
                "inv_dual_edge_length": self._edge_params.inverse_dual_edge_lengths,
                "rturb_prandtl": self._params.rturb_prandtl,
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_nudging,
                "horizontal_end": self._edge_end_halo,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        # flux divergence (geofac_div), tendency and state update
        self.apply_horizontal_diffusion_and_update_scalar = setup_program(
            backend=backend,
            program=apply_horizontal_diffusion_and_update_scalar,
            constant_args={
                "nabla2_flux": self._nabla2_flux_e,
                "geofac_div": self._interpolation_state.geofac_div,
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        # temp_to_energy / energy_to_temp + final tendency (mo_vdf_atmo.f90 l.
        # 634/694) and compute_flux_x (l. 753); the energy-type variant is
        # inlined at compile time
        self.compute_energy_from_temperature = setup_program(
            backend=backend,
            program=compute_energy_from_temperature,
            constant_args={
                "height_above_ground": self.ghf,
                "grav": constants.GRAV,
                "use_internal_energy": use_internal_energy,
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        self.compute_surface_energy_flux = setup_program(
            backend=backend,
            program=compute_surface_energy_flux,
            constant_args={"use_internal_energy": use_internal_energy},
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            offset_provider={},
        )
        self.compute_temperature_from_energy_and_tendency = setup_program(
            backend=backend,
            program=compute_temperature_from_energy_and_tendency,
            constant_args={
                "height_above_ground": self.ghf,
                "grav": constants.GRAV,
                "use_internal_energy": use_internal_energy,
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )

    def _setup_momentum_diffusion_programs(
        self,
        backend: gtx_typing.Backend
        | model_backends.DeviceType
        | model_backends.BackendDescriptor
        | None,
        num_levels: int,
    ) -> None:
        """
        Bind the Stage D + E step programs (momentum diffusion:
        Compute_diffusion_hor_wind l. 1207 and Compute_diffusion_vert_wind
        l. 1601 in mo_vdf.f90). The Stage D edge loops run on
        rl grf_bdywidth_e + 1 .. min_rledge_int, all full levels; the Stage E
        cell loops run on the tmx t_domain cell range (grf_bdywidth_c + 1 ..
        min_rlcell_int), half-level rows 2..nlev (1-based), unless noted
        otherwise.
        """
        # CALL init(...) zero fill of the half-level (nlev + 1) cell fields
        # (tend_wa and wa_new before the w solve); same stencil as the
        # full-level variant, one more K row
        self.init_cell_kdim_half_field_with_zero = setup_program(
            backend=backend,
            program=init_cell_kdim_field_with_zero_wp,
            horizontal_sizes={
                "horizontal_start": gtx.int32(0),
                "horizontal_end": self._cell_end_end,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels + 1),
            },
            offset_provider={},
        )

        # ------------------------------------------------------------------
        # Stage D: horizontal wind (vn) diffusion
        # ------------------------------------------------------------------
        # cells2edges_scalar(rho) + reciprocal (-> inv_rhoe)
        self.interpolate_inverse_density_to_edges = setup_program(
            backend=backend,
            program=interpolate_inverse_density_to_edges,
            constant_args={
                "c_lin_e": self._interpolation_state.c_lin_e,
                "inv_rhoe": self._inv_rhoe,
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_nudging_level_2,
                "horizontal_end": self._edge_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        # '1) First get the horizontal tendencies' (-> tot_tend)
        self.compute_vn_horizontal_stress_tendency = setup_program(
            backend=backend,
            program=compute_vn_horizontal_stress_tendency,
            constant_args={
                "inv_rhoe": self._inv_rhoe,
                "primal_normal_vert_x": self._edge_params.primal_normal_vert[0],
                "primal_normal_vert_y": self._edge_params.primal_normal_vert[1],
                "dual_normal_vert_x": self._edge_params.dual_normal_vert[0],
                "dual_normal_vert_y": self._edge_params.dual_normal_vert[1],
                "tangent_orientation": self._edge_params.tangent_orientation,
                "inv_primal_edge_length": self._edge_params.inverse_primal_edge_lengths,
                "inv_vert_vert_length": self._edge_params.inverse_vertex_vertex_lengths,
                "inv_dual_edge_length": self._edge_params.inverse_dual_edge_lengths,
                "tot_tend": self.tot_tend,
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_nudging_level_2,
                "horizontal_end": self._edge_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        # '2) Vertical tendency' rhs loops (interior, top and surface-stress
        # bottom row fused; also fills inv_maire)
        self.compute_vn_vertical_diffusion_rhs = setup_program(
            backend=backend,
            program=compute_vn_vertical_diffusion_rhs,
            constant_args={
                "inv_rhoe": self._inv_rhoe,
                "inv_ddqz_z_full_e": self._metric_state.inv_ddqz_z_full_e,
                "primal_normal_cell_x": self._edge_params.primal_normal_cell[0],
                "primal_normal_cell_y": self._edge_params.primal_normal_cell[1],
                "c_lin_e": self._interpolation_state.c_lin_e,
                "inv_dual_edge_length": self._edge_params.inverse_dual_edge_lengths,
                "rhs": self._edge_rhs,
                "inv_maire": self._inv_maire,
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_nudging_level_2,
                "horizontal_end": self._edge_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
                "nlev": gtx.int32(num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        # prepare_diffusion_matrix on edges (zk = km_ie, lhalflvl=.FALSE.,
        # zprefac absent -> 1)
        self.prepare_tridiagonal_matrix_vn = setup_program(
            backend=backend,
            program=prepare_tridiagonal_matrix_edges,
            constant_args={
                "inv_mair": self._inv_maire,
                "inv_dz": self._metric_state.inv_ddqz_z_half_e,
                "a": self._edge_matrix_a,
                "b": self._edge_matrix_b,
                "c": self._edge_matrix_c,
                "zprefac": 1.0,
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_nudging_level_2,
                "horizontal_end": self._edge_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        # 'diffuse_vertical_implicit' on edges (accumulates onto tot_tend);
        # the tridiagonal solution only enters through the tendency
        self.solve_vn_vertical_diffusion = setup_program(
            backend=backend,
            program=solve_vertical_diffusion_edges,
            constant_args={
                "a": self._edge_matrix_a,
                "b": self._edge_matrix_b,
                "c": self._edge_matrix_c,
                "rhs": self._edge_rhs,
                "new_var": self._diffused_vn,
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_nudging_level_2,
                "horizontal_end": self._edge_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        # rbf_vec_interpol_cell (tot_tend -> tend_u, tend_v): cells rl
        # 2..min_rlcell_int (Fortran default opt_rlstart = 2), all full levels
        self.edge_2_cell_vector_rbf_interpolation = setup_program(
            backend=backend,
            program=edge_2_cell_vector_rbf_interpolation,
            constant_args={
                "ptr_coeff_1": self._interpolation_state.rbf_coeff_c1,
                "ptr_coeff_2": self._interpolation_state.rbf_coeff_c2,
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_lateral_boundary_level_2,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        # final update loop: tmx t_domain cells, all full levels
        self.update_horizontal_wind = setup_program(
            backend=backend,
            program=update_horizontal_wind,
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )

        # ------------------------------------------------------------------
        # Stage E: vertical wind (w) diffusion
        # ------------------------------------------------------------------
        # rbf_vec_interpol_edge on full levels (vn -> vt_e): edges rl
        # 2 (Fortran default opt_rlstart)..min_rledge_int-1
        self.compute_tangential_wind_full_levels = setup_program(
            backend=backend,
            program=compute_tangential_wind_wp,
            constant_args={
                "rbf_vec_coeff_e": self._interpolation_state.rbf_coeff_e,
                "vt": self._vt_e,
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_lateral_boundary_level_2,
                "horizontal_end": self._edge_end_halo,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        # rhs of the w solve (also fills inv_rho_ic and inv_mair_ic)
        self.compute_w_vertical_diffusion_rhs = setup_program(
            backend=backend,
            program=compute_w_vertical_diffusion_rhs,
            constant_args={
                "inv_ddqz_z_half": self._metric_state.inv_ddqz_z_half,
                "rhs": self._w_rhs,
                "inv_rho_ic": self._inv_rho_ic,
                "inv_mair_ic": self._inv_mair_ic,
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(1),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        # prepare_diffusion_matrix on half-level cells (zk = km_c,
        # lhalflvl=.TRUE., minlvl=2, zprefac=2)
        self.prepare_tridiagonal_matrix_w = setup_program(
            backend=backend,
            program=prepare_tridiagonal_matrix_cells_half,
            constant_args={
                "inv_mair": self._inv_mair_ic,
                "inv_dz": self._metric_state.inv_ddqz_z_full,
                "a": self._matrix_a,
                "b": self._matrix_b,
                "c": self._matrix_c,
                "zprefac": 2.0,
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(1),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        # w = 0 top/bottom boundary-condition terms on the main diagonal
        self.modify_w_diffusion_matrix_boundary = setup_program(
            backend=backend,
            program=modify_w_diffusion_matrix_boundary,
            constant_args={
                "b": self._matrix_b,
                "inv_dz": self._metric_state.inv_ddqz_z_full,
                "inv_mair_ic": self._inv_mair_ic,
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(1),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        # 'diffuse_vertical_implicit' on half-level cells (minlvl=2, i.e.
        # vertical_start=1: the scan init is applied at the domain start, row 0
        # stays untouched); the Fortran w solve is implicit regardless of the
        # configured solver type
        self.solve_w_vertical_diffusion = setup_program(
            backend=backend,
            program=solve_vertical_diffusion_cells,
            constant_args={
                "a": self._matrix_a,
                "b": self._matrix_b,
                "c": self._matrix_c,
                "rhs": self._w_rhs,
                "new_var": self._diffused_w,
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(1),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        # '1) Get horizontal tendencies at half level edges' (D31/D32 stress):
        # edges rl grf_bdywidth_e..min_rledge_int-1 (one halo line computed on
        # purpose, the C2E gather of the update runs on halo-adjacent cells)
        self.compute_w_horizontal_stress_tendency = setup_program(
            backend=backend,
            program=compute_w_horizontal_stress_tendency,
            constant_args={
                "inv_ddqz_z_half": self._metric_state.inv_ddqz_z_half,
                "inv_ddqz_z_half_v": self._metric_state.inv_ddqz_z_half_v,
                "vt_e": self._vt_e,
                "primal_normal_cell_x": self._edge_params.primal_normal_cell[0],
                "primal_normal_cell_y": self._edge_params.primal_normal_cell[1],
                "dual_normal_vert_x": self._edge_params.dual_normal_vert[0],
                "dual_normal_vert_y": self._edge_params.dual_normal_vert[1],
                "edge_cell_length": self._metric_state.edge_cell_length,
                "tangent_orientation": self._edge_params.tangent_orientation,
                "inv_primal_edge_length": self._edge_params.inverse_primal_edge_lengths,
                "inv_vert_vert_length": self._edge_params.inverse_vertex_vertex_lengths,
                "inv_dual_edge_length": self._edge_params.inverse_dual_edge_lengths,
                "hori_tend_e": self._hori_tend_e,
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_nudging,
                "horizontal_end": self._edge_end_halo,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(1),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        # e_bln_c_s gather of hori_tend_e, tendency accumulation and w update
        self.apply_w_horizontal_diffusion_and_update = setup_program(
            backend=backend,
            program=apply_w_horizontal_diffusion_and_update,
            constant_args={
                "hori_tend_e": self._hori_tend_e,
                "e_bln_c_s": self._interpolation_state.e_bln_c_s,
                "inv_rho_ic": self._inv_rho_ic,
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(1),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider=self._grid.connectivities,
        )

    def _setup_energy_and_diagnostics_programs(
        self,
        backend: gtx_typing.Backend
        | model_backends.DeviceType
        | model_backends.BackendDescriptor
        | None,
        num_levels: int,
    ) -> None:
        """
        Bind the Stage F + G step programs (``Update_energy_tendencies``
        l. 1938 in mo_vdf.f90 and the ``Update_diagnostics`` of
        mo_vdf_atmo.f90 l. 487 / mo_vdf.f90 l. 354). All loops run on the tmx
        t_domain cell range (grf_bdywidth_c + 1 .. min_rlcell_int), all full
        levels.
        """
        # Stage F: kinetic-energy dissipation heating and final temperature
        # tendency / update
        self.update_temperature_with_dissipation_heating = setup_program(
            backend=backend,
            program=update_temperature_with_dissipation_heating,
            constant_args={"dissipation_factor": self.config.dissipation_factor},
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
                "nlev": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        # Stage G: vertical-integral diagnostics ('compute_internal_energy_vi'
        # and the accumulation loop of Update_diagnostics, mo_vdf_atmo.f90);
        # the running integrals go to granule scratch fields, the bottom rows
        # (the column integrals) are copied to the 2D diagnostics afterwards
        self.compute_vertical_integral_diagnostics = setup_program(
            backend=backend,
            program=compute_vertical_integral_diagnostics,
            constant_args={
                "dz": self._metric_state.ddqz_z_full,
                "cptgz_vi": self._cptgz_vi_run,
                "dissip_ke_vi": self._dissip_ke_vi_run,
                "int_energy_vi": self._int_energy_vi_run,
                "int_energy_vi_tend": self._int_energy_vi_tend_run,
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        # Stage G: full-level km/kh diagnostic assembly (the km/kh loop of
        # Update_diagnostics, mo_vdf.f90; output-only diagnostics)
        self.update_exchange_coefficient_diagnostics = setup_program(
            backend=backend,
            program=update_exchange_coefficient_diagnostics,
            constant_args={
                "km_const": self.config.km_const,
                "rturb_prandtl": self._params.rturb_prandtl,
                "use_km_const": self.config.use_km_const,
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
                "nlev": gtx.int32(num_levels),
            },
            offset_provider={},
        )

    def _allocate_local_fields(self) -> None:
        # squared Smagorinsky mixing length at half-level cell centers [m^2]
        self.mix_len_sq: fa.CellKField[ta.wpfloat] = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=self._allocator
        )
        # cell-area scaling factor of the Louis constant b
        self.louis_factor: fa.CellField[ta.wpfloat] = data_alloc.zero_field(
            self._grid, dims.CellDim, allocator=self._allocator
        )
        # geometric height of the full levels above the surface [m]
        self.ghf: fa.CellKField[ta.wpfloat] = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=self._allocator
        )
        # land / sea-ice fractions; zero (aqua planet) until they are wired
        # through the input state
        self.fract_land: fa.CellField[ta.wpfloat] = data_alloc.zero_field(
            self._grid, dims.CellDim, allocator=self._allocator
        )
        self.fract_ice: fa.CellField[ta.wpfloat] = data_alloc.zero_field(
            self._grid, dims.CellDim, allocator=self._allocator
        )

        # scalar diffusion (Stages B and C) temporaries, matching the local
        # arrays of the scalar diffusion subroutines in mo_vdf.f90
        def _cell_k_field() -> fa.CellKField[ta.wpfloat]:
            return data_alloc.zero_field(
                self._grid, dims.CellDim, dims.KDim, allocator=self._allocator
            )

        # inverse air mass per unit area [m^2/kg] (``inv_mair``)
        self._inv_air_mass: fa.CellKField[ta.wpfloat] = _cell_k_field()
        # rows of the tridiagonal vertical diffusion matrix (``a``, ``b``, ``c``)
        self._matrix_a: fa.CellKField[ta.wpfloat] = _cell_k_field()
        self._matrix_b: fa.CellKField[ta.wpfloat] = _cell_k_field()
        self._matrix_c: fa.CellKField[ta.wpfloat] = _cell_k_field()
        # right-hand side of the vertical diffusion solve (``rhs``); only the
        # bottom K row is ever written, all other rows must stay zero
        self._rhs: fa.CellKField[ta.wpfloat] = _cell_k_field()
        # scratch for the tridiagonal solution of the implicit solver, whose
        # effect only enters through the tendency (the Fortran discards it too:
        # the new state is computed as state + tend * dtime after the
        # horizontal diffusion)
        self._diffused_scalar: fa.CellKField[ta.wpfloat] = _cell_k_field()
        # horizontal turbulent diffusion flux at full-level edges (``nabla2_e``)
        self._nabla2_flux_e: fa.EdgeKField[ta.wpfloat] = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, allocator=self._allocator
        )
        # energy diffused by the heat diffusion, computed from the input
        # temperature and the old moisture state (``energy``)
        self.energy: fa.CellKField[ta.wpfloat] = _cell_k_field()
        # total (vertical + horizontal) diffusion tendency of the energy
        # (``tend_energy``)
        self.tend_energy: fa.CellKField[ta.wpfloat] = _cell_k_field()
        # energy after the diffusion update (``new_energy``)
        self._new_energy: fa.CellKField[ta.wpfloat] = _cell_k_field()
        # grid-mean surface energy flux (``flux_x`` of 'compute_flux_x')
        self._flux_x: fa.CellField[ta.wpfloat] = data_alloc.zero_field(
            self._grid, dims.CellDim, allocator=self._allocator
        )
        # zero surface flux of the tracers without surface exchange (qc, qi)
        self._zero_surface_flux: fa.CellField[ta.wpfloat] = data_alloc.zero_field(
            self._grid, dims.CellDim, allocator=self._allocator
        )

        # Stage G scratch: running (top-down) vertical integrals of the
        # ``*_vi`` diagnostics; the value at the last full level is the column
        # integral that is copied to the 2D fields of the diagnostic state
        # (rows of cells outside the computed domain stay zero, matching the
        # Fortran 'CALL init(...)' of the 2D fields)
        self._cptgz_vi_run: fa.CellKField[ta.wpfloat] = _cell_k_field()
        self._dissip_ke_vi_run: fa.CellKField[ta.wpfloat] = _cell_k_field()
        self._int_energy_vi_run: fa.CellKField[ta.wpfloat] = _cell_k_field()
        self._int_energy_vi_tend_run: fa.CellKField[ta.wpfloat] = _cell_k_field()

        # momentum diffusion (Stages D and E) temporaries, matching the local
        # arrays of Compute_diffusion_hor_wind / Compute_diffusion_vert_wind
        # in mo_vdf.f90
        def _edge_k_field(extend: int = 0) -> fa.EdgeKField[ta.wpfloat]:
            return data_alloc.zero_field(
                self._grid,
                dims.EdgeDim,
                dims.KDim,
                extend={dims.KDim: extend},
                allocator=self._allocator,
            )

        def _cell_half_field() -> fa.CellKField[ta.wpfloat]:
            return data_alloc.zero_field(
                self._grid,
                dims.CellDim,
                dims.KDim,
                extend={dims.KDim: 1},
                allocator=self._allocator,
            )

        # inverse air density at edge midpoints (``inv_rhoe``)
        self._inv_rhoe: fa.EdgeKField[ta.wpfloat] = _edge_k_field()
        # inverse air mass per unit area of the edge layers (``inv_maire``)
        self._inv_maire: fa.EdgeKField[ta.wpfloat] = _edge_k_field()
        # total (horizontal + vertical) vn diffusion tendency (``tot_tend``),
        # kept on the granule for testing. The Fortran zero fill at entry is
        # not replicated: the rows inside the Stage D edge domain are
        # (over)written before they are read, halo rows are synced (S9), and
        # all other rows keep their allocation-time zeros because no stencil
        # ever writes them (they are read by the C2E2C2E gather of the RBF
        # interpolation, as zeros, exactly as in the Fortran).
        self.tot_tend: fa.EdgeKField[ta.wpfloat] = _edge_k_field()
        # rows of the tridiagonal vn diffusion matrix (``za``, ``zb``, ``zc``)
        self._edge_matrix_a: fa.EdgeKField[ta.wpfloat] = _edge_k_field()
        self._edge_matrix_b: fa.EdgeKField[ta.wpfloat] = _edge_k_field()
        self._edge_matrix_c: fa.EdgeKField[ta.wpfloat] = _edge_k_field()
        # right-hand side of the vn diffusion solve (``zrhs``); every row is
        # written by 'compute_vn_vertical_diffusion_rhs' before the solve
        self._edge_rhs: fa.EdgeKField[ta.wpfloat] = _edge_k_field()
        # scratch for the tridiagonal solution of the vn solve (discarded,
        # only the tendency accumulated onto ``tot_tend`` is used)
        self._diffused_vn: fa.EdgeKField[ta.wpfloat] = _edge_k_field()
        # tangential wind at edge midpoints on full levels (``vt_e``)
        self._vt_e: fa.EdgeKField[ta.wpfloat] = _edge_k_field()
        # horizontal D31/D32 stress tendency of w at half-level edges
        # (``hori_tend_e``); rows outside the computed domain (edge rows
        # outside grf_bdywidth_e..min_rledge_int-1 and the top/bottom half
        # levels) keep their allocation-time zeros and are never read
        self._hori_tend_e: fa.EdgeKField[ta.wpfloat] = _edge_k_field(extend=1)
        # right-hand side of the w diffusion solve (``rhs``); only the
        # half-level rows 1..nlev-1 are written and read
        self._w_rhs: fa.CellKField[ta.wpfloat] = _cell_half_field()
        # inverse air density at half-level cell centers (``inv_rho_ic``)
        self._inv_rho_ic: fa.CellKField[ta.wpfloat] = _cell_half_field()
        # inverse air mass per unit area of the half-level layers
        # (``inv_mair_ic``)
        self._inv_mair_ic: fa.CellKField[ta.wpfloat] = _cell_half_field()
        # scratch for the tridiagonal solution of the w solve (discarded,
        # only the tendency is used)
        self._diffused_w: fa.CellKField[ta.wpfloat] = _cell_half_field()

    def _coefficient_fields(
        self, field: gtx.Field, horizontal_dim: gtx.Dimension
    ) -> tuple[gtx.Field, gtx.Field, gtx.Field]:
        """Extract the three 2D extrapolation coefficient fields from rows 0..2."""
        return tuple(  # type: ignore[return-value]
            gtx.as_field((horizontal_dim,), field.ndarray[:, k], allocator=self._allocator)
            for k in range(3)
        )

    def _determine_horizontal_domains(self) -> None:
        cell_domain = h_grid.domain(dims.CellDim)
        edge_domain = h_grid.domain(dims.EdgeDim)
        vertex_domain = h_grid.domain(dims.VertexDim)

        self._cell_start_lateral_boundary_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._cell_start_lateral_boundary_level_3 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        self._cell_start_lateral_boundary_level_4 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
        )
        self._cell_start_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._cell_end_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._cell_end_halo = self._grid.end_index(cell_domain(h_grid.Zone.HALO))
        self._cell_end_halo_level_2 = self._grid.end_index(cell_domain(h_grid.Zone.HALO_LEVEL_2))
        self._cell_end_end = self._grid.end_index(cell_domain(h_grid.Zone.END))

        self._edge_start_lateral_boundary_level_2 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._edge_start_lateral_boundary_level_3 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        self._edge_start_lateral_boundary_level_4 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
        )
        self._edge_start_nudging = self._grid.start_index(edge_domain(h_grid.Zone.NUDGING))
        self._edge_start_nudging_level_2 = self._grid.start_index(
            edge_domain(h_grid.Zone.NUDGING_LEVEL_2)
        )
        self._edge_end_local = self._grid.end_index(edge_domain(h_grid.Zone.LOCAL))
        self._edge_end_halo = self._grid.end_index(edge_domain(h_grid.Zone.HALO))
        self._edge_end_halo_level_2 = self._grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))
        self._edge_end_end = self._grid.end_index(edge_domain(h_grid.Zone.END))

        self._vertex_start_lateral_boundary_level_2 = self._grid.start_index(
            vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._vertex_start_nudging = self._grid.start_index(vertex_domain(h_grid.Zone.NUDGING))
        self._vertex_end_local = self._grid.end_index(vertex_domain(h_grid.Zone.LOCAL))
        self._vertex_end_halo = self._grid.end_index(vertex_domain(h_grid.Zone.HALO))

    def run_diagnostics(
        self,
        input_state: tmx_states.TmxInputState,
        diagnostic_state: tmx_states.TmxDiagnosticState,
    ) -> None:
        """
        Compute the Smagorinsky diagnostics (Stage A).

        Port of ``Compute_diagnostics`` in mo_vdf_atmo.f90 (l. 343-482), executed
        in the Fortran order with halo exchanges at the Fortran sync points.
        ``ghf``, ``mix_len_sq`` and ``louis_factor`` are granule-owned fields
        computed at construction; the corresponding fields of
        ``diagnostic_state`` are not written here.

        Note: the Fortran zero-initializes u_vert/v_vert/w_vert and
        km_iv/km_c/km_ie/kh_ic/km_ic before (re)computing them on possibly
        smaller domains. This is not replicated here: entries outside the
        computed domains keep their allocation-time zeros (or whatever a
        previous call left there); they must not be relied upon.
        """
        log.debug("tmx Stage A (Compute_diagnostics): start")

        self.compute_static_energy(
            temperature=input_state.temperature,
            static_energy=diagnostic_state.cptgz,
        )
        self.compute_virtual_potential_temperature(
            virtual_temperature=input_state.virtual_temperature,
            pressure=input_state.pressure,
            theta_v=diagnostic_state.theta_v,
        )
        self.interpolate_cell_to_half_levels(
            interpolant=input_state.rho,
            interpolation=diagnostic_state.rho_ic,
        )
        self.compute_brunt_vaisala_frequency(
            theta_v=diagnostic_state.theta_v,
            bruvais=diagnostic_state.bruvais,
        )

        # S1: CALL sync_patch_array(SYNC_C, patch, pum1/pvm1) in mo_vdf_atmo.f90
        log.debug("communication of input u, v (cells): start")
        self._exchange.exchange(dims.CellDim, input_state.u, input_state.v)
        log.debug("communication of input u, v (cells): end")

        self.compute_vn_from_uv(
            u=input_state.u,
            v=input_state.v,
            vn=diagnostic_state.vn,
        )

        # S2: CALL sync_patch_array(SYNC_E, patch, vn) in mo_vdf_atmo.f90
        log.debug("communication of vn (edges): start")
        self._exchange.exchange(dims.EdgeDim, diagnostic_state.vn)
        log.debug("communication of vn (edges): end")

        self.compute_cell_2_vertex_interpolation(
            cell_in=input_state.w,
            vert_out=diagnostic_state.w_vert,
        )
        self.cell_2_edge_interpolation(
            in_field=input_state.w,
            out_field=diagnostic_state.w_ie,
        )
        self.mo_intp_rbf_rbf_vec_interpol_vertex(
            p_e_in=diagnostic_state.vn,
            p_u_out=diagnostic_state.u_vert,
            p_v_out=diagnostic_state.v_vert,
        )

        # S3: CALL sync_patch_array_mult(SYNC_V, patch, 3, w_vert, u_vert, v_vert)
        log.debug("communication of w_vert, u_vert, v_vert (vertices): start")
        self._exchange.exchange(
            dims.VertexDim,
            diagnostic_state.w_vert,
            diagnostic_state.u_vert,
            diagnostic_state.v_vert,
        )
        log.debug("communication of w_vert, u_vert, v_vert (vertices): end")

        self.interpolate_vn_to_half_levels_with_boundary(
            vn=diagnostic_state.vn,
            vn_ie=diagnostic_state.vn_ie,
        )
        self.compute_tangential_wind_wp(
            vn=diagnostic_state.vn_ie,
            vt=diagnostic_state.vt_ie,
        )
        self.compute_shear_and_div_of_stress(
            u_vert=diagnostic_state.u_vert,
            v_vert=diagnostic_state.v_vert,
            w_vert=diagnostic_state.w_vert,
            w=input_state.w,
            vn_ie=diagnostic_state.vn_ie,
            vt_ie=diagnostic_state.vt_ie,
            w_ie=diagnostic_state.w_ie,
            shear=diagnostic_state.shear,
            div_stress=diagnostic_state.div_of_stress,
        )
        self.interpolate_to_cell_center(
            interpolant=diagnostic_state.div_of_stress,
            interpolation=diagnostic_state.div_c,
        )
        self.interpolate_shear_to_half_level_cells(
            shear=diagnostic_state.shear,
            mech_prod=diagnostic_state.mech_prod,
        )

        if self.config.use_km_const:
            self.compute_viscosity(
                rho_ic=diagnostic_state.rho_ic,
                km_ic=diagnostic_state.km_ic,
                kh_ic=diagnostic_state.kh_ic,
            )
        else:
            self.compute_viscosity(
                mech_prod=diagnostic_state.mech_prod,
                bruvais=diagnostic_state.bruvais,
                rho_ic=diagnostic_state.rho_ic,
                km_ic=diagnostic_state.km_ic,
                kh_ic=diagnostic_state.kh_ic,
            )

        # S4: CALL sync_patch_array(SYNC_C, patch, kh_ic/km_ic) in
        # mo_tmx_smagorinsky.f90 (l. 299-300); the constant-viscosity branch does
        # not sync in the Fortran code, but the exchange is a no-op there anyway
        # (the full halo is computed).
        log.debug("communication of kh_ic, km_ic (cells): start")
        self._exchange.exchange(dims.CellDim, diagnostic_state.kh_ic, diagnostic_state.km_ic)
        log.debug("communication of kh_ic, km_ic (cells): end")

        self.interpolate_km_to_full_level_cells(
            km_ic=diagnostic_state.km_ic,
            km_c=diagnostic_state.km_c,
        )
        self.interpolate_km_to_vertices(
            km_ic=diagnostic_state.km_ic,
            km_iv=diagnostic_state.km_iv,
        )
        self.interpolate_km_to_edges(
            km_ic=diagnostic_state.km_ic,
            km_ie=diagnostic_state.km_ie,
        )

        log.debug("tmx Stage A (Compute_diagnostics): end")

    def _solve_scalar_vertical_diffusion(
        self,
        var: fa.CellKField[ta.wpfloat],
        tend: fa.CellKField[ta.wpfloat],
        dtime: float,
    ) -> None:
        """
        Vertical diffusion solve of a cell scalar, accumulating onto ``tend``.

        Dispatches on the configured solver type ('diffuse_vertical_implicit' /
        'diffuse_vertical_explicit' in mo_tmx_numerics.f90). The matrix rows
        (``self._matrix_a/b/c``) and the right-hand side (``self._rhs``) are
        bound at construction and must be up to date. The tridiagonal solution
        of the implicit solver goes to the ``self._diffused_scalar`` scratch:
        as in the Fortran, it only enters through the tendency.
        """
        if self.config.solver_type == TurbulenceSolverType.IMPLICIT:
            self.solve_vertical_diffusion(
                var=var,
                new_var=self._diffused_scalar,
                tend=tend,
                dtime=dtime,
            )
        else:
            self.solve_vertical_diffusion(
                var=var,
                tend=tend,
            )

    def run_hydrometeor_diffusion(
        self,
        *,
        input_state: tmx_states.TmxInputState,
        surface_flux_state: tmx_states.TmxSurfaceFluxState,
        diagnostic_state: tmx_states.TmxDiagnosticState,
        tendency_state: tmx_states.TmxTendencyState,
        new_state: tmx_states.TmxNewState,
        dtime: float,
    ) -> None:
        """
        Compute the hydrometeor diffusion (Stage B).

        Port of ``Compute_diffusion_hydrometeors`` in mo_vdf.f90 (l. 585),
        without the optional CO2 tracer (``l_co2``, out of scope). For each of
        qv, qc and qi: implicit (or explicit) vertical diffusion with the
        surface flux entering through the bottom-row right-hand side
        (qv: evapotranspiration, qc/qi: zero), followed by conservative
        horizontal nabla2 diffusion and the state update. The tendencies
        (``ddt_qv/qc/qi``) are zeroed at entry and hold the total (vertical +
        horizontal) diffusion tendency on exit.

        Requires the Stage A diagnostics (``kh_ic``, ``km_ie``) of
        ``diagnostic_state`` to be up to date (``run_diagnostics``).
        """
        log.debug("tmx Stage B (Compute_diffusion_hydrometeors): start")

        self.compute_inverse_air_mass(input_field=input_state.air_mass)
        self.prepare_tridiagonal_matrix_hydrometeors(
            zk=diagnostic_state.kh_ic,
            a=self._matrix_a,
            b=self._matrix_b,
            c=self._matrix_c,
        )

        tracers = (
            (
                "qv",
                input_state.qv,
                tendency_state.ddt_qv,
                new_state.qv,
                surface_flux_state.evapotranspiration,
            ),
            ("qc", input_state.qc, tendency_state.ddt_qc, new_state.qc, self._zero_surface_flux),
            ("qi", input_state.qi, tendency_state.ddt_qi, new_state.qi, self._zero_surface_flux),
        )
        for name, state, tend, new, sfc_flx in tracers:
            self.init_cell_kdim_field_with_zero(field_with_zero_wp=tend)
            self.compute_surface_flux_rhs(sfc_flx=sfc_flx, rhs=self._rhs, prefac=1.0)
            self._solve_scalar_vertical_diffusion(var=state, tend=tend, dtime=dtime)

            # S5: CALL sync_patch_array(SYNC_C, patch, state) in mo_vdf.f90
            # ("include halo points and boundary points because these values
            # will be used in next loop")
            log.debug(f"communication of {name} (cells): start")
            self._exchange.exchange(dims.CellDim, state)
            log.debug(f"communication of {name} (cells): end")

            self.compute_scalar_nabla2_flux(
                scalar=state,
                km_ie=diagnostic_state.km_ie,
                nabla2_flux=self._nabla2_flux_e,
                prefac=1.0,
            )
            self.apply_horizontal_diffusion_and_update_scalar(
                scalar=state,
                rho=input_state.rho,
                new_scalar=new,
                tend=tend,
                dtime=dtime,
            )

        log.debug("tmx Stage B (Compute_diffusion_hydrometeors): end")

    def run_temperature_diffusion(
        self,
        *,
        input_state: tmx_states.TmxInputState,
        surface_flux_state: tmx_states.TmxSurfaceFluxState,
        diagnostic_state: tmx_states.TmxDiagnosticState,
        tendency_state: tmx_states.TmxTendencyState,
        new_state: tmx_states.TmxNewState,
        dtime: float,
    ) -> None:
        """
        Compute the temperature (heat) diffusion (Stage C).

        Port of ``Compute_diffusion_temperature`` in mo_vdf.f90 (l. 912):
        the temperature is converted to the configured energy (dry static or
        internal, using the *old* moisture state), the energy is diffused
        vertically (surface energy flux ``flux_x`` in the bottom-row right-hand
        side) and horizontally like the hydrometeors, and the new temperature
        is recovered from the new energy using the *new* moisture state of
        ``new_state`` -> ``new_state.temperature``,
        ``tendency_state.ddt_temperature = (new_ta - ta) / dtime``.

        The intermediate energy fields are kept on the granule for testing:
        ``self.energy`` (from the old temperature) and ``self.tend_energy``
        (total energy diffusion tendency).

        Requires the Stage A diagnostics (``kh_ic``, ``km_ie``) of
        ``diagnostic_state`` and the hydrometeor diffusion results
        (``new_state.qv/qc/qi``, Stage B) to be up to date.
        """
        log.debug("tmx Stage C (Compute_diffusion_temperature): start")

        self.init_cell_kdim_field_with_zero(field_with_zero_wp=self.tend_energy)

        self.compute_energy_from_temperature(
            temperature=input_state.temperature,
            qv=input_state.qv,
            qc=input_state.qc,
            qi=input_state.qi,
            qr=input_state.qr,
            qs=input_state.qs,
            qg=input_state.qg,
            energy=self.energy,
        )

        # air temperature at the lowest full level, as passed to
        # 'compute_energy_fluxes' by the surface Compute (mo_vdf_sfc.f90; the
        # surface scheme input ``ta`` is bound to the bottom row of the tmx
        # temperature state in mo_interface_aes_tmx.f90)
        temperature_sfc = gtx.as_field(
            (dims.CellDim,),
            input_state.temperature.ndarray[:, self._grid.num_levels - 1],
            allocator=self._allocator,
        )
        self.compute_surface_energy_flux(
            sensible_heat_flux=surface_flux_state.sensible_heat_flux,
            evapotranspiration=surface_flux_state.evapotranspiration,
            temperature_sfc=temperature_sfc,
            flux_x=self._flux_x,
        )

        self.compute_inverse_air_mass(input_field=input_state.air_mass)
        self.prepare_tridiagonal_matrix_energy(
            zk=diagnostic_state.kh_ic,
            a=self._matrix_a,
            b=self._matrix_b,
            c=self._matrix_c,
        )
        self.compute_surface_flux_rhs(sfc_flx=self._flux_x, rhs=self._rhs, prefac=self._zfactor)
        self._solve_scalar_vertical_diffusion(var=self.energy, tend=self.tend_energy, dtime=dtime)

        # S6: CALL sync_patch_array(SYNC_C, patch, energy) in mo_vdf.f90
        log.debug("communication of energy (cells): start")
        self._exchange.exchange(dims.CellDim, self.energy)
        log.debug("communication of energy (cells): end")

        self.compute_scalar_nabla2_flux(
            scalar=self.energy,
            km_ie=diagnostic_state.km_ie,
            nabla2_flux=self._nabla2_flux_e,
            prefac=self._zfactor,
        )
        self.apply_horizontal_diffusion_and_update_scalar(
            scalar=self.energy,
            rho=input_state.rho,
            new_scalar=self._new_energy,
            tend=self.tend_energy,
            dtime=dtime,
        )

        self.compute_temperature_from_energy_and_tendency(
            energy=self._new_energy,
            temperature=input_state.temperature,
            qv=new_state.qv,
            qc=new_state.qc,
            qi=new_state.qi,
            qr=input_state.qr,
            qs=input_state.qs,
            qg=input_state.qg,
            new_temperature=new_state.temperature,
            tend_temperature=tendency_state.ddt_temperature,
            dtime=dtime,
        )

        log.debug("tmx Stage C (Compute_diffusion_temperature): end")

    def run_horizontal_wind_diffusion(
        self,
        *,
        input_state: tmx_states.TmxInputState,
        surface_flux_state: tmx_states.TmxSurfaceFluxState,
        diagnostic_state: tmx_states.TmxDiagnosticState,
        tendency_state: tmx_states.TmxTendencyState,
        new_state: tmx_states.TmxNewState,
        dtime: float,
    ) -> None:
        """
        Compute the horizontal wind diffusion (Stage D).

        Port of ``Compute_diffusion_hor_wind`` in mo_vdf.f90 (l. 1207): the
        horizontal divergence of the 3D stress tensor acting on vn and the
        implicit vertical vn diffusion (surface momentum stress entering
        through the bottom-row right-hand side) accumulate the total edge
        tendency ``self.tot_tend``, which is RBF-interpolated to the cell
        tendencies ``tendency_state.ddt_u/ddt_v``;
        ``new_state.u/v = u/v + ddt_u/v * dtime``.

        Only the implicit vertical solver is wired (the Fortran explicit
        branch of the hor-wind solve has no edge-based icon4py port yet).

        Requires the Stage A diagnostics (``vn``, ``u_vert``, ``v_vert``,
        ``km_c``, ``div_c``, ``km_iv``, ``km_ie``) of ``diagnostic_state`` to
        be up to date (``run_diagnostics``).
        """
        if self.config.solver_type != TurbulenceSolverType.IMPLICIT:
            raise NotImplementedError(
                "tmx Stage D (Compute_diffusion_hor_wind) only implements the "
                "implicit vertical diffusion solver ('diffuse_vertical_explicit' "
                "on edges is not ported)."
            )

        log.debug("tmx Stage D (Compute_diffusion_hor_wind): start")

        # CALL init(tend_u/tend_v): the RBF interpolation only writes cells
        # 2..min_rlcell_int, everything outside must be zero
        self.init_cell_kdim_field_with_zero(field_with_zero_wp=tendency_state.ddt_u)
        self.init_cell_kdim_field_with_zero(field_with_zero_wp=tendency_state.ddt_v)

        # S7: CALL sync_patch_array(SYNC_C, patch, rho) in mo_vdf.f90
        log.debug("communication of rho (cells): start")
        self._exchange.exchange(dims.CellDim, input_state.rho)
        log.debug("communication of rho (cells): end")

        self.interpolate_inverse_density_to_edges(rho=input_state.rho)
        self.compute_vn_horizontal_stress_tendency(
            u_vert=diagnostic_state.u_vert,
            v_vert=diagnostic_state.v_vert,
            vn=diagnostic_state.vn,
            km_c=diagnostic_state.km_c,
            div_c=diagnostic_state.div_c,
            km_iv=diagnostic_state.km_iv,
        )

        # S8: CALL sync_uvml_s of mflux_u and mflux_v in mo_vdf.f90; the
        # Fortran comment notes the sync of the momentum fluxes is needed for
        # the MPI test to pass
        log.debug("communication of u_stress, v_stress (cells, 2D): start")
        self._exchange.exchange(
            dims.CellDim, surface_flux_state.u_stress, surface_flux_state.v_stress
        )
        log.debug("communication of u_stress, v_stress (cells, 2D): end")

        self.compute_vn_vertical_diffusion_rhs(
            w=input_state.w,
            km_ie=diagnostic_state.km_ie,
            u_stress=surface_flux_state.u_stress,
            v_stress=surface_flux_state.v_stress,
        )
        self.prepare_tridiagonal_matrix_vn(zk=diagnostic_state.km_ie)
        self.solve_vn_vertical_diffusion(
            var=diagnostic_state.vn,
            tend=self.tot_tend,
            dtime=dtime,
        )

        # S9: CALL sync_patch_array(SYNC_E, patch, tot_tend) in mo_vdf.f90
        log.debug("communication of tot_tend (edges): start")
        self._exchange.exchange(dims.EdgeDim, self.tot_tend)
        log.debug("communication of tot_tend (edges): end")

        self.edge_2_cell_vector_rbf_interpolation(
            p_e_in=self.tot_tend,
            p_u_out=tendency_state.ddt_u,
            p_v_out=tendency_state.ddt_v,
        )
        self.update_horizontal_wind(
            u=input_state.u,
            v=input_state.v,
            tend_u=tendency_state.ddt_u,
            tend_v=tendency_state.ddt_v,
            new_u=new_state.u,
            new_v=new_state.v,
            dtime=dtime,
        )

        # S10/S11: CALL sync_patch_array_mult(SYNC_C, patch, 2, tend_u, tend_v)
        # and (..., new_state_u, new_state_v) in mo_vdf.f90 (both marked
        # "TODO: Are these necessary?" there; ported as-is)
        log.debug("communication of ddt_u, ddt_v (cells): start")
        self._exchange.exchange(dims.CellDim, tendency_state.ddt_u, tendency_state.ddt_v)
        log.debug("communication of ddt_u, ddt_v (cells): end")
        log.debug("communication of new u, v (cells): start")
        self._exchange.exchange(dims.CellDim, new_state.u, new_state.v)
        log.debug("communication of new u, v (cells): end")

        log.debug("tmx Stage D (Compute_diffusion_hor_wind): end")

    def run_vertical_wind_diffusion(
        self,
        *,
        input_state: tmx_states.TmxInputState,
        diagnostic_state: tmx_states.TmxDiagnosticState,
        tendency_state: tmx_states.TmxTendencyState,
        new_state: tmx_states.TmxNewState,
        dtime: float,
    ) -> None:
        """
        Compute the vertical wind diffusion (Stage E).

        Port of ``Compute_diffusion_vert_wind`` in mo_vdf.f90 (l. 1601): the
        implicit vertical w diffusion on half levels (minlvl = 2, with the
        w = 0 top/bottom boundary conditions folded into the matrix diagonal)
        accumulates onto ``tendency_state.ddt_w``, then the horizontal D31/D32
        stress tendency at half-level edges is interpolated back to cells and
        added; ``new_state.w = w + ddt_w * dtime`` on the interior half levels
        (rows 0 and nlev stay zero, the w = 0 boundary rows). The Fortran w
        solve is implicit regardless of the configured solver type.

        Requires the Stage A diagnostics (``vn``, ``rho_ic``, ``km_c``,
        ``km_ic``, ``km_iv``, ``div_c``, ``u_vert``, ``v_vert``, ``w_vert``,
        ``w_ie``) of ``diagnostic_state`` to be up to date
        (``run_diagnostics``).
        """
        log.debug("tmx Stage E (Compute_diffusion_vert_wind): start")

        # CALL init(tend) / init(new_state): ddt_w is accumulated and new_w is
        # only written on the interior half levels
        self.init_cell_kdim_half_field_with_zero(field_with_zero_wp=tendency_state.ddt_w)
        self.init_cell_kdim_half_field_with_zero(field_with_zero_wp=new_state.w)

        self.compute_tangential_wind_full_levels(vn=diagnostic_state.vn)
        self.compute_w_vertical_diffusion_rhs(
            rho_ic=diagnostic_state.rho_ic,
            km_c=diagnostic_state.km_c,
            div_c=diagnostic_state.div_c,
        )
        self.prepare_tridiagonal_matrix_w(zk=diagnostic_state.km_c)
        self.modify_w_diffusion_matrix_boundary(km_c=diagnostic_state.km_c)
        self.solve_w_vertical_diffusion(
            var=input_state.w,
            tend=tendency_state.ddt_w,
            dtime=dtime,
        )

        self.compute_w_horizontal_stress_tendency(
            u=input_state.u,
            v=input_state.v,
            km_ic=diagnostic_state.km_ic,
            u_vert=diagnostic_state.u_vert,
            v_vert=diagnostic_state.v_vert,
            w_vert=diagnostic_state.w_vert,
            km_iv=diagnostic_state.km_iv,
            w_ie=diagnostic_state.w_ie,
        )
        self.apply_w_horizontal_diffusion_and_update(
            w=input_state.w,
            new_w=new_state.w,
            tend=tendency_state.ddt_w,
            dtime=dtime,
        )

        # S12: CALL sync_patch_array(SYNC_C, patch, new_state) in mo_vdf.f90
        log.debug("communication of new w (cells): start")
        self._exchange.exchange(dims.CellDim, new_state.w)
        log.debug("communication of new w (cells): end")

        log.debug("tmx Stage E (Compute_diffusion_vert_wind): end")

    def run_energy_update(
        self,
        *,
        input_state: tmx_states.TmxInputState,
        surface_flux_state: tmx_states.TmxSurfaceFluxState,
        diagnostic_state: tmx_states.TmxDiagnosticState,
        tendency_state: tmx_states.TmxTendencyState,
        new_state: tmx_states.TmxNewState,
        dtime: float,
    ) -> None:
        """
        Update the temperature tendency with the dissipation heating (Stage F).

        Port of ``Update_energy_tendencies`` in mo_vdf.f90 (l. 1938): the
        kinetic energy dissipated by the horizontal wind diffusion
        (``dissip_ke``, from the old and the Stage D updated winds) plus the
        snow-on-canopy melt cooling at the lowest level (``-q_snocpymlt``,
        non-zero only over land) give the turbulent heating rate
        (``diagnostic_state.heating``), whose temperature tendency is added to
        the heat-diffusion tendency of Stage C:
        ``tendency_state.ddt_temperature += heating / cv_air`` and
        ``new_state.temperature = temperature + ddt_temperature * dtime``.

        Requires the temperature diffusion tendency (Stage C,
        ``tendency_state.ddt_temperature``) and the updated winds (Stage D,
        ``new_state.u/v``) to be up to date.
        """
        log.debug("tmx Stage F (Update_energy_tendencies): start")

        # CALL init(heating): zero fill of the whole array; the update stencil
        # only writes the domain cells
        self.init_cell_kdim_field_with_zero(field_with_zero_wp=diagnostic_state.heating)
        self.update_temperature_with_dissipation_heating(
            u=input_state.u,
            v=input_state.v,
            new_u=new_state.u,
            new_v=new_state.v,
            air_mass=input_state.air_mass,
            cv_air=input_state.cv_air,
            temperature=input_state.temperature,
            tend_temperature=tendency_state.ddt_temperature,
            q_snocpymlt=surface_flux_state.q_snocpymlt,
            dissip_ke=diagnostic_state.dissip_ke,
            heating=diagnostic_state.heating,
            new_temperature=new_state.temperature,
            dtime=dtime,
        )

        # S13: CALL sync_patch_array_mult(SYNC_C, patch, 2, new_state_ta,
        # tend_ta) in mo_vdf.f90 (marked "TODO: Are these necessary?" there;
        # ported as-is)
        log.debug("communication of new temperature, ddt_temperature (cells): start")
        self._exchange.exchange(dims.CellDim, new_state.temperature, tendency_state.ddt_temperature)
        log.debug("communication of new temperature, ddt_temperature (cells): end")

        log.debug("tmx Stage F (Update_energy_tendencies): end")

    def run_update_diagnostics(
        self,
        *,
        input_state: tmx_states.TmxInputState,
        diagnostic_state: tmx_states.TmxDiagnosticState,
        new_state: tmx_states.TmxNewState,
        dtime: float,
    ) -> None:
        """
        Update the end-of-step diagnostics (Stage G).

        Port of the atmospheric part of ``Update_diagnostics`` in
        mo_vdf_atmo.f90 (l. 487) and mo_vdf.f90 (l. 354):

        - ``diagnostic_state.cptgz`` is recomputed from the updated
          temperature,
        - the vertically integrated diagnostics ``cptgz_vi``,
          ``dissip_ke_vi``, ``int_energy_vi`` and ``int_energy_vi_tend``
          (from the old- and new-state internal energies),
        - the full-level exchange coefficient diagnostics ``km`` / ``kh``
          (bottom row: ``km_const`` if ``use_km_const``, else zero — the
          surface exchange coefficients are out of scope).

        The 2m/10m diagnostics and the tile aggregation of the Fortran
        ``Update_diagnostics`` belong to the surface scheme and are out of
        scope.

        Requires all diffusion stages and the energy update (Stage F, for
        ``dissip_ke`` and the final ``new_state.temperature``) to be up to
        date.
        """
        log.debug("tmx Stage G (Update_diagnostics): start")

        # cptgz from the updated temperature (same program binding as Stage A)
        self.compute_static_energy(
            temperature=new_state.temperature,
            static_energy=diagnostic_state.cptgz,
        )
        self.compute_vertical_integral_diagnostics(
            static_energy=diagnostic_state.cptgz,
            dissip_ke=diagnostic_state.dissip_ke,
            rho=input_state.rho,
            temperature=input_state.temperature,
            qv=input_state.qv,
            qc=input_state.qc,
            qi=input_state.qi,
            new_temperature=new_state.temperature,
            new_qv=new_state.qv,
            new_qc=new_state.qc,
            new_qi=new_state.qi,
            qr=input_state.qr,
            qs=input_state.qs,
            qg=input_state.qg,
            dtime=dtime,
        )
        # extract the column integrals (the last full-level row of the running
        # sums) into the 2D diagnostics; a device-side row copy, no compute
        bottom = self._grid.num_levels - 1
        for running_integral, target in (
            (self._cptgz_vi_run, diagnostic_state.cptgz_vi),
            (self._dissip_ke_vi_run, diagnostic_state.dissip_ke_vi),
            (self._int_energy_vi_run, diagnostic_state.int_energy_vi),
            (self._int_energy_vi_tend_run, diagnostic_state.int_energy_vi_tend),
        ):
            target.ndarray[...] = running_integral.ndarray[:, bottom]

        self.update_exchange_coefficient_diagnostics(
            km_ic=diagnostic_state.km_ic,
            kh_ic=diagnostic_state.kh_ic,
            km=diagnostic_state.km,
            kh=diagnostic_state.kh,
        )

        log.debug("tmx Stage G (Update_diagnostics): end")

    def run(
        self,
        *,
        input_state: tmx_states.TmxInputState,
        surface_flux_state: tmx_states.TmxSurfaceFluxState,
        diagnostic_state: tmx_states.TmxDiagnosticState,
        tendency_state: tmx_states.TmxTendencyState,
        new_state: tmx_states.TmxNewState,
        dtime: float,
    ) -> None:
        """
        Run one tmx time step (Stages A to G).

        Port of ``Compute`` in mo_vdf.f90, in the Fortran stage order:
        Smagorinsky diagnostics (A), hydrometeor diffusion (B), temperature
        diffusion (C, using the new moisture state of B), horizontal wind
        diffusion (D), vertical wind diffusion (E), dissipation heating (F,
        using the new winds of D) and the end-of-step diagnostics (G). The
        surface scheme called between A and B in the Fortran
        (``this%sfc%Compute``) is out of scope: the grid-mean surface fluxes
        it would produce are prescribed inputs (``surface_flux_state``).

        On exit, ``tendency_state`` holds the total tmx tendencies of
        temperature, qv/qc/qi, u/v and w, ``new_state`` the corresponding
        updated fields (``new = state + tend * dtime``) and
        ``diagnostic_state`` the Stage A and Stage F/G diagnostics.
        """
        log.debug("tmx run (Compute): start")

        self.run_diagnostics(input_state, diagnostic_state)
        stage_kwargs = dict(
            input_state=input_state,
            surface_flux_state=surface_flux_state,
            diagnostic_state=diagnostic_state,
            tendency_state=tendency_state,
            new_state=new_state,
            dtime=dtime,
        )
        self.run_hydrometeor_diffusion(**stage_kwargs)
        self.run_temperature_diffusion(**stage_kwargs)
        self.run_horizontal_wind_diffusion(**stage_kwargs)
        self.run_vertical_wind_diffusion(
            input_state=input_state,
            diagnostic_state=diagnostic_state,
            tendency_state=tendency_state,
            new_state=new_state,
            dtime=dtime,
        )
        self.run_energy_update(**stage_kwargs)
        self.run_update_diagnostics(
            input_state=input_state,
            diagnostic_state=diagnostic_state,
            new_state=new_state,
            dtime=dtime,
        )

        log.debug("tmx run (Compute): end")
