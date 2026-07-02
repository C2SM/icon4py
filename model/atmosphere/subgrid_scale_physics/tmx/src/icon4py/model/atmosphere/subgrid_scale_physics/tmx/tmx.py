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
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.assign_constant_viscosity import (
    assign_constant_viscosity,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_brunt_vaisala_frequency import (
    compute_brunt_vaisala_frequency,
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
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_tangential_wind_wp import (
    compute_tangential_wind_wp,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_virtual_potential_temperature import (
    compute_virtual_potential_temperature,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_vn_from_uv import (
    compute_vn_from_uv,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.init_height_above_ground import (
    init_height_above_ground,
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
from icon4py.model.common.interpolation.stencils.interpolate_to_cell_center import (
    interpolate_to_cell_center,
)
from icon4py.model.common.interpolation.stencils.mo_intp_rbf_rbf_vec_interpol_vertex import (
    mo_intp_rbf_rbf_vec_interpol_vertex,
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

    EXPLICIT = 1  #: explicit time stepping
    IMPLICIT = 2  #: implicit time stepping


class EnergyType(int, enum.Enum):
    """
    Type of energy diffused by the temperature (heat) diffusion.

    Note: Called ``energy_type`` in ``mo_turb_vdiff_config.f90``.
    """

    DRY_STATIC = 1  #: dry static energy cp*T + g*z
    INTERNAL = 2  #: internal energy cv*T


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
            icon_equivalent=common_conf_opt.IconOption("solver_type", ("aes_vdf_nml",)),
        ),
    ] = TurbulenceSolverType.IMPLICIT

    energy_type: typing.Annotated[
        EnergyType,
        common_conf_opt.ConfigOption(
            description="Type of energy diffused by the heat diffusion (dry static or internal).",
            icon_equivalent=common_conf_opt.IconOption("energy_type", ("aes_vdf_nml",)),
        ),
    ] = EnergyType.INTERNAL

    dissipation_factor: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Scaling factor for the kinetic energy dissipation heating.",
            icon_equivalent=common_conf_opt.IconOption("dissipation_factor", ("aes_vdf_nml",)),
        ),
    ] = 1.0

    use_louis: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, use the Louis (1979) stability correction function "
            "instead of the classic (Lilly 1962) one.",
            icon_equivalent=common_conf_opt.IconOption("use_louis", ("aes_vdf_nml",)),
        ),
    ] = True

    use_louis_land: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If False, exclude cells with more than 50% land fraction "
            "from the Louis stability correction.",
            icon_equivalent=common_conf_opt.IconOption("use_louis_land", ("aes_vdf_nml",)),
        ),
    ] = True

    use_louis_ice: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If False, exclude cells with more than 50% sea-ice fraction "
            "from the Louis stability correction.",
            icon_equivalent=common_conf_opt.IconOption("use_louis_ice", ("aes_vdf_nml",)),
        ),
    ] = True

    louis_constant_b: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Louis constant b of the Louis stability correction function.",
            icon_equivalent=common_conf_opt.IconOption("louis_constant_b", ("aes_vdf_nml",)),
        ),
    ] = 4.2

    use_km_const: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, use a constant exchange coefficient instead of the "
            "Smagorinsky model.",
            icon_equivalent=common_conf_opt.IconOption("use_km_const", ("aes_vdf_nml",)),
        ),
    ] = False

    km_const: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Constant exchange coefficient used if 'use_km_const' is True [m^2/s].",
            icon_equivalent=common_conf_opt.IconOption("km_const", ("aes_vdf_nml",)),
        ),
    ] = 1.0

    use_scale_turb_energy_flux: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, scale the turbulent energy flux by 'scale_turb_energy_flux'.",
            icon_equivalent=common_conf_opt.IconOption(
                "use_scale_turb_energy_flux", ("aes_vdf_nml",)
            ),
        ),
    ] = False

    scale_turb_energy_flux: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Scaling factor for the turbulent energy flux used if "
            "'use_scale_turb_energy_flux' is True.",
            icon_equivalent=common_conf_opt.IconOption("scale_turb_energy_flux", ("aes_vdf_nml",)),
        ),
    ] = 1.0

    smag_constant: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Smagorinsky constant Cs of the Smagorinsky-Lilly eddy viscosity model.",
            icon_equivalent=common_conf_opt.IconOption("smag_constant", ("aes_vdf_nml",)),
        ),
    ] = 0.23

    turb_prandtl: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Turbulent Prandtl number.",
            icon_equivalent=common_conf_opt.IconOption("turb_prandtl", ("aes_vdf_nml",)),
        ),
    ] = 0.33333333333  #: exact literal from mo_turb_vdiff_config.f90 (not 1/3)

    km_min: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Minimum mass-weighted turbulent viscosity [kg/(m s)].",
            icon_equivalent=common_conf_opt.IconOption("km_min", ("aes_vdf_nml",)),
        ),
    ] = 0.001

    max_turb_scale: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Maximum turbulence length scale [m].",
            icon_equivalent=common_conf_opt.IconOption("max_turb_scale", ("aes_vdf_nml",)),
        ),
    ] = 300.0

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def from_fortran_dict(cls, atmo_dict: dict[str, Any], **overrides: Any) -> TmxConfig:
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
    Currently implements the initialization (``Smagorinsky_init`` in
    mo_tmx_smagorinsky.f90 plus the time-independent height above ground) and
    Stage A, the Smagorinsky diagnostics (``Compute_diagnostics`` in
    mo_vdf_atmo.f90).

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
        #: not used by the Smagorinsky diagnostics (Stage A); kept for parity with
        #: the other granules and for the later tmx stages.
        self._vertical_grid = vertical_grid
        self._metric_state = metric_state
        self._interpolation_state = interpolation_state
        self._edge_params = edge_params
        self._cell_params = cell_params

        assert self._cell_params.area is not None

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
        #: geometric height of the surface (bottom half level), 2D slice of z_ifc
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
        # cells (grf_bdywidth_c + 1 .. min_rlcell_int), all full levels
        self.init_height_above_ground = setup_program(
            backend=backend,
            program=init_height_above_ground,
            constant_args={"z_mc": self._metric_state.z_mc, "z_ifc_sfc": z_ifc_sfc},
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

        # ---------------------------------------------------------------------
        # Run the init programs (Smagorinsky_init in mo_tmx_smagorinsky.f90 and
        # compute_geopotential_height_above_ground in mo_vdf_atmo.f90)
        # ---------------------------------------------------------------------
        self.init_smagorinsky_mixing_length(mixing_length_sq=self.mix_len_sq)
        if self.config.use_louis:
            # the Fortran init only computes the Louis scaling factor if the
            # Louis stability correction is enabled; the field stays zero otherwise
            self.init_louis_scaling_factor(scaling_factor_louis=self.louis_factor)
        self.init_height_above_ground(height_above_ground=self.ghf)

    def _allocate_local_fields(self) -> None:
        #: squared Smagorinsky mixing length at half-level cell centers [m^2]
        self.mix_len_sq: fa.CellKField[ta.wpfloat] = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=self._allocator
        )
        #: cell-area scaling factor of the Louis constant b
        self.louis_factor: fa.CellField[ta.wpfloat] = data_alloc.zero_field(
            self._grid, dims.CellDim, allocator=self._allocator
        )
        #: geometric height of the full levels above the surface [m]
        self.ghf: fa.CellKField[ta.wpfloat] = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=self._allocator
        )
        #: land / sea-ice fractions; zero (aqua planet) until they are wired
        #: through the input state
        self.fract_land: fa.CellField[ta.wpfloat] = data_alloc.zero_field(
            self._grid, dims.CellDim, allocator=self._allocator
        )
        self.fract_ice: fa.CellField[ta.wpfloat] = data_alloc.zero_field(
            self._grid, dims.CellDim, allocator=self._allocator
        )

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
