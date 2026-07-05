# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
import logging
import math
import typing
from typing import Any, Final

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.aggregate_surface_tiles import (
    aggregate_surface_tiles,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.compute_potential_temperatures import (
    compute_potential_temperatures,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.compute_surface_density import (
    compute_surface_density,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.compute_surface_exchange_coefficients import (
    compute_surface_exchange_first_guess,
    obukhov_businger_step,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.compute_surface_fluxes_ice import (
    compute_surface_fluxes_ice,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.compute_surface_fluxes_ocean import (
    compute_surface_fluxes_ocean,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.compute_surface_roughness_ocean import (
    compute_surface_roughness_ocean,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.compute_surface_saturation_humidity import (
    compute_surface_saturation_humidity,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.compute_surface_stress_land import (
    compute_surface_stress_land,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.compute_surface_wind_speed import (
    compute_surface_wind_speed,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.sea_ice import (
    compute_ice_nonsolar_forcing,
    set_ice_albedo,
    set_ice_temp_zerolayer,
)
from icon4py.model.common import dimension as dims, model_backends, type_alias as ta
from icon4py.model.common.config import options as common_conf_opt
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.model_options import setup_program
from icon4py.model.common.utils import data_allocation as data_alloc


if typing.TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states
    from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface import surface_states
    from icon4py.model.common.grid import base as base_grid


"""
TMX surface scheme, ported from ICON's ``src/atm_phy_aes/tmx``.

Provides the configuration (:class:`TmxSurfaceConfig`), derived parameters
(:class:`TmxSurfaceParams`) and the granule (:class:`TmxSurface`). The surface
scheme runs before the atmospheric diffusion (``mo_vdf.f90``: surface stage
then atmosphere stage) and produces a ``TmxSurfaceFluxState``. Config defaults
are taken from ``vdiff_config_init`` in ``mo_turb_vdiff_config.f90`` and
``mo_sea_ice_nml.f90``.
"""


log = logging.getLogger(__name__)


# Member count of the Fortran t_vdiff_config derived type and the position of
# its 'use_tmx' canary (mo_turb_vdiff_config.f90). The surface config reads the
# same echoed aes_vdf_nml namelist as TmxConfig; keep these in sync with tmx.py.
_T_VDIFF_CONFIG_NUM_MEMBERS: Final = 42
_T_VDIFF_CONFIG_USE_TMX_INDEX: Final = 22

# ln 2, precomputed to keep the function call out of the dataclass default (ruff RUF009).
_LN2: Final[float] = math.log(2.0)


@dataclasses.dataclass(kw_only=True)
class TmxSurfaceConfig:
    """
    Configuration of the tmx surface scheme.

    Turbulence/roughness options are read from ``aes_vdf_nml`` (``t_vdiff_config``,
    mo_turb_vdiff_config.f90) by positional member index; sea-ice options come
    from ``sea_ice_nml`` (mo_sea_ice_nml.f90) and are wired to the namelist in
    a later milestone (``icon_equivalent`` still ``None``).
    """

    fsl: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Fraction of the first-level height at which surface fluxes are evaluated.",
            icon_equivalent=common_conf_opt.IconOption(
                "fsl", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=15
            ),
        ),
    ] = 0.4

    z0m_min: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Minimum roughness length for momentum [m].",
            icon_equivalent=common_conf_opt.IconOption(
                "z0m_min", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=18
            ),
        ),
    ] = 1.5e-5

    z0m_ice: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Roughness length for momentum over ice [m].",
            icon_equivalent=common_conf_opt.IconOption(
                "z0m_ice", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=19
            ),
        ),
    ] = 1.0e-3

    z0m_oce: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="First-guess roughness length for momentum over ocean [m].",
            icon_equivalent=common_conf_opt.IconOption(
                "z0m_oce", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=20
            ),
        ),
    ] = 1.0e-3

    min_sfc_wind: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Minimum surface wind speed in the free-convection limit [m/s].",
            icon_equivalent=common_conf_opt.IconOption(
                "min_sfc_wind", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=39
            ),
        ),
    ] = 0.3

    wind_g: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Wind-gust parameter for the ocean scalar fluxes [m/s].",
            icon_equivalent=common_conf_opt.IconOption(
                "wind_g", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=40
            ),
        ),
    ] = 3.0

    # Sea-ice options (sea_ice_nml); namelist wiring deferred to milestone S3.
    ice_thermodynamics_type: typing.Annotated[
        int,
        common_conf_opt.ConfigOption(
            description="Sea-ice thermodynamics model; only the Semtner zero-layer model (1) "
            "is supported.",
        ),
    ] = 1

    ice_layer_heat_capacity_thickness: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Stabilising slab thickness of the ice heat-capacity term (``hci_layer``) [m].",
        ),
    ] = 0.10

    ocean_freezing_temperature: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Sea-water freezing temperature (``Tf``) [degC].",
        ),
    ] = -1.80

    use_no_flux_gradients: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, drop the nonsolar-flux temperature derivative in the ice "
            "surface energy balance.",
        ),
    ] = True

    ice_albedo_scheme: typing.Annotated[
        int,
        common_conf_opt.ConfigOption(
            description="Sea-ice albedo scheme; only the temperature-weighted scheme (1) is supported.",
        ),
    ] = 1

    prescribed_flux_mode: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, bypass the surface physics and use the caller-provided fluxes "
            "(ICON isrfc_type==1 / atmosphere-only setup).",
        ),
    ] = False

    land_qsat_from_star: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, use the prescribed land surface saturation humidity (JSBACH "
            "'seb_qsat_star') in the land exchange solver (Fortran-faithful); if False, "
            "approximate it as saturation over water at the land skin temperature.",
        ),
    ] = True

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def from_fortran_dict(cls, atmo_dict: dict[str, Any], **overrides: Any) -> TmxSurfaceConfig:
        """
        Construct the configuration from the echoed ICON namelists.

        Reads the turbulence/roughness members of the ``aes_vdf_nml``
        (``t_vdiff_config``) positional array, located by ``unnamed_index``
        pinned to mo_turb_vdiff_config.f90. Sea-ice options keep their defaults
        (or come from ``overrides``). The guards fail loudly if the Fortran
        type changed.
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
        """Apply consistency checks on configuration parameters."""
        if self.ice_thermodynamics_type != 1:
            raise ValueError(
                "Invalid argument 'ice_thermodynamics_type': only the Semtner zero-layer model "
                f"(1) is supported, got {self.ice_thermodynamics_type}."
            )
        if self.ice_albedo_scheme != 1:
            raise ValueError(
                "Invalid argument 'ice_albedo_scheme': only the temperature-weighted scheme (1) "
                f"is supported, got {self.ice_albedo_scheme}."
            )
        if self.min_sfc_wind <= 0.0:
            raise ValueError(
                f"Invalid argument 'min_sfc_wind': should be positive, got {self.min_sfc_wind}."
            )
        if self.z0m_min <= 0.0:
            raise ValueError(f"Invalid argument 'z0m_min': should be positive, got {self.z0m_min}.")


@dataclasses.dataclass(frozen=True)
class TmxSurfaceParams:
    """Compile-time surface parameters (mo_turb_vdiff_params.f90, mo_vdf_diag_smag.f90)."""

    von_karman: Final[float] = 0.4
    """Von Karman constant (``ckap`` in mo_turb_vdiff_params.f90)."""
    charnock: Final[float] = 0.018
    """Charnock constant (``cchar`` in mo_turb_vdiff_params.f90)."""
    viscous_coeff: Final[float] = 0.11
    """Viscous coefficient of the ocean roughness (``viscous_coeff`` in mo_turb_vdiff_params.f90)."""
    bsm: Final[float] = 5.0
    """Businger stable-momentum constant (``bsm`` in mo_vdf_diag_smag.f90)."""
    bsh: Final[float] = 5.0
    """Businger stable-heat constant (``bsh`` in mo_vdf_diag_smag.f90)."""
    bum: Final[float] = 16.0
    """Businger unstable-momentum constant (``bum`` in mo_vdf_diag_smag.f90)."""
    buh: Final[float] = 16.0
    """Businger unstable-heat constant (``buh`` in mo_vdf_diag_smag.f90)."""
    air_kinematic_viscosity: Final[float] = 1.5e-5
    """Kinematic viscosity of air (``nu`` in the Charnock roughness) [m^2/s]; verify in S1."""
    half_pi: Final[float] = math.pi / 2.0
    """pi/2 (``pi_2`` in the Businger unstable momentum profile)."""
    ln2: Final[float] = _LN2
    """ln 2 (``ln2`` in the Businger similarity profiles)."""


class TmxSurface:
    """
    TMX surface-flux granule.

    Port of ``t_vdf_sfc`` (mo_vdf_sfc.f90 ``Compute_diagnostics`` + ``Compute``,
    mo_tmx_surface.f90). Runs before the atmospheric diffusion and fills a
    ``TmxSurfaceFluxState`` from prescribed SST (ocean), a prognostic sea-ice
    surface temperature (ice_fast) and prescribed land fluxes.

    Skeleton milestone (S0): only the prescribed-flux bypass is wired; the bulk
    physics is added from S1 onwards.
    """

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        config: TmxSurfaceConfig,
        params: TmxSurfaceParams,
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
        # 'use_no_flux_gradients' (default) drops the dnonsolardt term in the ice
        # surface energy balance -> flag 0; otherwise 1 (mo_ice_zerolayer.f90).
        self._nfg_flag = 0.0 if config.use_no_flux_gradients else 1.0
        self._determine_horizontal_domains()
        self._allocate_local_fields()
        self._setup_programs(backend)

    def _determine_horizontal_domains(self) -> None:
        cell_domain = h_grid.domain(dims.CellDim)
        self._cell_start = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._cell_end = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))

    def _allocate_local_fields(self) -> None:
        """Allocate the per-step scratch fields (2D cell fields, zero-initialized)."""

        def scratch() -> object:
            return data_alloc.zero_field(
                self._grid, dims.CellDim, dtype=ta.wpfloat, allocator=self._allocator
            )

        # shared per-tile diagnostics (recomputed for each tile)
        self._wind_rel = scratch()
        self._rho_sfc = scratch()
        self._theta_atm = scratch()
        self._theta_sfc = scratch()
        self._qsat_ocean = scratch()
        self._qsat_land = scratch()
        self._rough_m = scratch()
        self._rough_h = scratch()
        # Obukhov / Businger ping-pong buffers and per-tile transfer coefficients
        self._km_a = scratch()
        self._kh_a = scratch()
        self._km_b = scratch()
        self._kh_b = scratch()
        self._kh_ocean = scratch()
        self._km_ice = scratch()
        self._kh_ice = scratch()
        self._km_land = scratch()
        self._kh_land = scratch()
        # zero placeholder (inactive tile contributions in the aggregation)
        self._zero_tile = scratch()
        # sea-ice forcing, energy-balance and albedo scratch
        self._ice_nonsolar = scratch()
        self._ice_dnonsolardt = scratch()
        self._ice_qtop = scratch()
        self._ice_qbot = scratch()
        self._ice_albvisdir = scratch()
        self._ice_albvisdif = scratch()
        self._ice_albnirdir = scratch()
        self._ice_albnirdif = scratch()
        self._qsat_ice = scratch()
        # fixed sea-ice momentum roughness (z0m_ice); filled once
        self._rough_ice = scratch()
        self._rough_ice.ndarray[...] = self.config.z0m_ice
        # per-tile flux scratch (aggregated into the grid mean)
        self._evap_ocean = scratch()
        self._latent_ocean = scratch()
        self._sensible_ocean = scratch()
        self._ustress_ocean = scratch()
        self._vstress_ocean = scratch()
        self._evap_ice = scratch()
        self._ustress_ice = scratch()
        self._vstress_ice = scratch()
        self._ustress_land = scratch()
        self._vstress_land = scratch()

    def _setup_programs(self, backend: object) -> None:
        horizontal_sizes = {"horizontal_start": self._cell_start, "horizontal_end": self._cell_end}
        self.compute_surface_wind_speed = setup_program(
            program=compute_surface_wind_speed,
            backend=backend,
            constant_args={"min_sfc_wind": self.config.min_sfc_wind},
            horizontal_sizes=horizontal_sizes,
            offset_provider={},
        )
        self.compute_surface_saturation_humidity = setup_program(
            program=compute_surface_saturation_humidity,
            backend=backend,
            constant_args={"over_ice": False},
            horizontal_sizes=horizontal_sizes,
            offset_provider={},
        )
        self.compute_surface_density = setup_program(
            program=compute_surface_density,
            backend=backend,
            horizontal_sizes=horizontal_sizes,
            offset_provider={},
        )
        self.compute_potential_temperatures = setup_program(
            program=compute_potential_temperatures,
            backend=backend,
            horizontal_sizes=horizontal_sizes,
            offset_provider={},
        )
        self.compute_surface_roughness_ocean = setup_program(
            program=compute_surface_roughness_ocean,
            backend=backend,
            constant_args={
                "charnock": self._params.charnock,
                "viscous_coeff": self._params.viscous_coeff,
                "kinematic_viscosity": self._params.air_kinematic_viscosity,
                "z0m_min": self.config.z0m_min,
            },
            horizontal_sizes=horizontal_sizes,
            offset_provider={},
        )
        self.compute_surface_exchange_first_guess = setup_program(
            program=compute_surface_exchange_first_guess,
            backend=backend,
            horizontal_sizes=horizontal_sizes,
            offset_provider={},
        )
        self.obukhov_businger_step = setup_program(
            program=obukhov_businger_step,
            backend=backend,
            horizontal_sizes=horizontal_sizes,
            offset_provider={},
        )
        self.compute_surface_fluxes_ocean = setup_program(
            program=compute_surface_fluxes_ocean,
            backend=backend,
            constant_args={"wind_gustiness": self.config.wind_g},
            horizontal_sizes=horizontal_sizes,
            offset_provider={},
        )
        # sea-ice tile: saturation over ice, non-solar forcing, zero-layer energy
        # balance, albedo and the (gustiness-free, sublimation) bulk fluxes
        self.compute_surface_saturation_humidity_ice = setup_program(
            program=compute_surface_saturation_humidity,
            backend=backend,
            constant_args={"over_ice": True},
            horizontal_sizes=horizontal_sizes,
            offset_provider={},
        )
        self.compute_ice_nonsolar_forcing = setup_program(
            program=compute_ice_nonsolar_forcing,
            backend=backend,
            horizontal_sizes=horizontal_sizes,
            offset_provider={},
        )
        self.set_ice_temp_zerolayer = setup_program(
            program=set_ice_temp_zerolayer,
            backend=backend,
            constant_args={
                "freezing_temperature": self.config.ocean_freezing_temperature,
                "heat_capacity_thickness": self.config.ice_layer_heat_capacity_thickness,
                "nonsolar_gradient_flag": self._nfg_flag,
            },
            horizontal_sizes=horizontal_sizes,
            offset_provider={},
        )
        self.set_ice_albedo = setup_program(
            program=set_ice_albedo,
            backend=backend,
            horizontal_sizes=horizontal_sizes,
            offset_provider={},
        )
        self.compute_surface_fluxes_ice = setup_program(
            program=compute_surface_fluxes_ice,
            backend=backend,
            horizontal_sizes=horizontal_sizes,
            offset_provider={},
        )
        self.compute_surface_stress_land = setup_program(
            program=compute_surface_stress_land,
            backend=backend,
            horizontal_sizes=horizontal_sizes,
            offset_provider={},
        )
        self.aggregate_surface_tiles = setup_program(
            program=aggregate_surface_tiles,
            backend=backend,
            horizontal_sizes=horizontal_sizes,
            offset_provider={},
        )

    def run(
        self,
        *,
        input_state: surface_states.SurfaceInputState,
        surface_state: surface_states.SurfaceState,
        flux_state: tmx_states.TmxSurfaceFluxState,
        dtime: float,
    ) -> None:
        """Compute one surface step, filling ``flux_state`` in place."""
        if self.config.prescribed_flux_mode:
            # Bypass: the fluxes are provided by the caller in 'flux_state'
            # (ICON isrfc_type==1 / the atmosphere-only tmx setup).
            return
        self._run_ocean(input_state, surface_state)
        self._run_ice(input_state, surface_state, dtime)
        self._run_land(input_state)
        self._aggregate(input_state, surface_state, flux_state)
        self._exchange.exchange(
            dims.CellDim,
            flux_state.evapotranspiration,
            flux_state.sensible_heat_flux,
            flux_state.u_stress,
            flux_state.v_stress,
        )

    def _tile_exchange(  # noqa: PLR0917 [too-many-positional-arguments] internal per-tile helper
        self,
        input_state: surface_states.SurfaceInputState,
        temperature_sfc: object,
        qsat_sfc: object,
        rough_m: object,
        km_out: object,
        kh_out: object,
    ) -> None:
        """Density, potential temperatures and the 5-iteration exchange solver for one tile.

        ``self._wind_rel`` must have been computed for the tile beforehand. The
        final transfer coefficients are written to ``km_out`` / ``kh_out``; the
        first four iterations ping-pong between the a/b buffers, the fifth writes
        the result out directly.
        """
        inp = input_state
        self.compute_surface_density(
            surface_pressure=inp.psfc,
            temperature_sfc=temperature_sfc,
            qsat_sfc=qsat_sfc,
            rho_sfc=self._rho_sfc,
        )
        self.compute_potential_temperatures(
            temperature_atm=inp.ta,
            pressure_atm=inp.pa,
            temperature_sfc=temperature_sfc,
            surface_pressure=inp.psfc,
            theta_atm=self._theta_atm,
            theta_sfc=self._theta_sfc,
        )
        self.compute_surface_exchange_first_guess(
            theta_atm=self._theta_atm,
            theta_sfc=self._theta_sfc,
            wind_rel=self._wind_rel,
            rough_m=rough_m,
            dz=inp.dz,
            km=self._km_a,
            kh=self._kh_a,
        )
        src_km, src_kh, dst_km, dst_kh = self._km_a, self._kh_a, self._km_b, self._kh_b
        for _ in range(4):
            self.obukhov_businger_step(
                km_in=src_km,
                kh_in=src_kh,
                theta_atm=self._theta_atm,
                theta_sfc=self._theta_sfc,
                qsat_sfc=qsat_sfc,
                qa=inp.qa,
                wind_rel=self._wind_rel,
                rough_m=rough_m,
                dz=inp.dz,
                km_out=dst_km,
                kh_out=dst_kh,
            )
            src_km, dst_km = dst_km, src_km
            src_kh, dst_kh = dst_kh, src_kh
        self.obukhov_businger_step(
            km_in=src_km,
            kh_in=src_kh,
            theta_atm=self._theta_atm,
            theta_sfc=self._theta_sfc,
            qsat_sfc=qsat_sfc,
            qa=inp.qa,
            wind_rel=self._wind_rel,
            rough_m=rough_m,
            dz=inp.dz,
            km_out=km_out,
            kh_out=kh_out,
        )

    def _run_ocean(
        self,
        input_state: surface_states.SurfaceInputState,
        surface_state: surface_states.SurfaceState,
    ) -> None:
        """Ocean-tile fluxes from prescribed SST (Charnock roughness, water qsat)."""
        inp = input_state
        self.compute_surface_wind_speed(
            ua=inp.ua,
            va=inp.va,
            reference_u=inp.ocean_u,
            reference_v=inp.ocean_v,
            wind_rel=self._wind_rel,
        )
        self.compute_surface_saturation_humidity(
            temperature_sfc=inp.sst, surface_pressure=inp.psfc, qsat_sfc=self._qsat_ocean
        )
        # Charnock roughness uses the momentum transfer coefficient of the previous step
        self.compute_surface_roughness_ocean(
            wind_rel=self._wind_rel,
            km=surface_state.ocean_km,
            rough_m=self._rough_m,
            rough_h=self._rough_h,
        )
        # the exchange solver writes the new km into surface_state.ocean_km (Charnock lag)
        self._tile_exchange(
            inp, inp.sst, self._qsat_ocean, self._rough_m, surface_state.ocean_km, self._kh_ocean
        )
        self.compute_surface_fluxes_ocean(
            rho_sfc=self._rho_sfc,
            kh=self._kh_ocean,
            km=surface_state.ocean_km,
            wind_rel=self._wind_rel,
            qa=inp.qa,
            qsat_sfc=self._qsat_ocean,
            ta=inp.ta,
            temperature_sfc=inp.sst,
            ua=inp.ua,
            va=inp.va,
            ocean_u=inp.ocean_u,
            ocean_v=inp.ocean_v,
            evapotranspiration=self._evap_ocean,
            latent_hflx=self._latent_ocean,
            sensible_hflx=self._sensible_ocean,
            u_stress=self._ustress_ocean,
            v_stress=self._vstress_ocean,
        )

    def _run_ice(
        self,
        input_state: surface_states.SurfaceInputState,
        surface_state: surface_states.SurfaceState,
        dtime: float,
    ) -> None:
        """Sea-ice tile: prognostic skin temperature (ice_fast) then bulk fluxes.

        Port of the ice branch of 'Compute' (mo_vdf_sfc.f90:413-452,
        mo_tmx_surface.f90:266-410). The zero-layer energy balance is forced by
        the ice-tile heat fluxes of the *previous* atmospheric step (lagged in
        ``surface_state``); the freshly computed fluxes are written straight back
        into the lagged state to force the next step. The driver swaps
        ``tsurf_ice_old <- tsurf_ice_new`` after the step.
        """
        inp = input_state
        # non-solar forcing on the old skin temperature and the lagged tile fluxes
        self.compute_ice_nonsolar_forcing(
            lwflx_net=inp.lwflx_net,
            lhflx=surface_state.lagged_ice_lhflx,
            shflx=surface_state.lagged_ice_shflx,
            tsurf_old=surface_state.tsurf_ice_old,
            emissivity=inp.emissivity,
            nonsolar=self._ice_nonsolar,
            dnonsolardt=self._ice_dnonsolardt,
        )
        # Semtner zero-layer update: old -> new skin temperature
        self.set_ice_temp_zerolayer(
            tsurf_old=surface_state.tsurf_ice_old,
            hi=inp.ice_thickness,
            hs=surface_state.snow_thickness,
            swnet=inp.swflx_net,
            nonsolar=self._ice_nonsolar,
            dnonsolardt=self._ice_dnonsolardt,
            tsurf_new=surface_state.tsurf_ice_new,
            qtop=self._ice_qtop,
            qbot=self._ice_qbot,
            dtime=dtime,
        )
        # albedo diagnostic on the new skin temperature (closes the SW feedback)
        self.set_ice_albedo(
            tsurf_new=surface_state.tsurf_ice_new,
            hi=inp.ice_thickness,
            hs=surface_state.snow_thickness,
            albvisdir=self._ice_albvisdir,
            albvisdif=self._ice_albvisdif,
            albnirdir=self._ice_albnirdir,
            albnirdif=self._ice_albnirdif,
        )
        # exchange coefficients and fluxes on the new skin temperature; wind is
        # relative to the ice drift, roughness is fixed (z0m_ice), qsat over ice
        self.compute_surface_wind_speed(
            ua=inp.ua,
            va=inp.va,
            reference_u=inp.ice_u,
            reference_v=inp.ice_v,
            wind_rel=self._wind_rel,
        )
        self.compute_surface_saturation_humidity_ice(
            temperature_sfc=surface_state.tsurf_ice_new,
            surface_pressure=inp.psfc,
            qsat_sfc=self._qsat_ice,
        )
        self._tile_exchange(
            inp,
            surface_state.tsurf_ice_new,
            self._qsat_ice,
            self._rough_ice,
            self._km_ice,
            self._kh_ice,
        )
        # store the latent/sensible fluxes into the lagged state: they feed both
        # the aggregation this step and the ice forcing of the next step
        self.compute_surface_fluxes_ice(
            rho_sfc=self._rho_sfc,
            kh=self._kh_ice,
            km=self._km_ice,
            wind_rel=self._wind_rel,
            qa=inp.qa,
            qsat_sfc=self._qsat_ice,
            ta=inp.ta,
            temperature_sfc=surface_state.tsurf_ice_new,
            ua=inp.ua,
            va=inp.va,
            ice_u=inp.ice_u,
            ice_v=inp.ice_v,
            evapotranspiration=self._evap_ice,
            latent_hflx=surface_state.lagged_ice_lhflx,
            sensible_hflx=surface_state.lagged_ice_shflx,
            u_stress=self._ustress_ice,
            v_stress=self._vstress_ice,
        )

    def _run_land(self, input_state: surface_states.SurfaceInputState) -> None:
        """Land-tile: prescribed LH/SH/evap (JSBACH cut line) and a bulk momentum stress.

        The heat/moisture fluxes are prescribed inputs; only the momentum stress
        is bulk-computed from an exchange coefficient over the prescribed land
        roughness. The land surface saturation humidity used by the exchange
        solver is the prescribed JSBACH ``seb_qsat_star`` or a
        saturation-over-water approximation (``config.land_qsat_from_star``).
        """
        inp = input_state
        # resting surface: reference velocity is zero
        self.compute_surface_wind_speed(
            ua=inp.ua,
            va=inp.va,
            reference_u=self._zero_tile,
            reference_v=self._zero_tile,
            wind_rel=self._wind_rel,
        )
        if self.config.land_qsat_from_star:
            land_qsat = inp.land_qsat_star
        else:
            self.compute_surface_saturation_humidity(
                temperature_sfc=inp.land_tskin, surface_pressure=inp.psfc, qsat_sfc=self._qsat_land
            )
            land_qsat = self._qsat_land
        self._tile_exchange(
            inp, inp.land_tskin, land_qsat, inp.land_rough_m, self._km_land, self._kh_land
        )
        self.compute_surface_stress_land(
            rho_sfc=self._rho_sfc,
            km=self._km_land,
            wind_rel=self._wind_rel,
            ua=inp.ua,
            va=inp.va,
            u_stress=self._ustress_land,
            v_stress=self._vstress_land,
        )

    def _aggregate(
        self,
        input_state: surface_states.SurfaceInputState,
        surface_state: surface_states.SurfaceState,
        flux_state: tmx_states.TmxSurfaceFluxState,
    ) -> None:
        """Fraction-weighted aggregation of the per-tile fluxes to the grid mean.

        The canopy snow-melt heating (``q_snocpymlt``) is a land-only diagnostic;
        the ocean and ice tiles contribute zero.
        """
        inp = input_state

        def aggregate(
            field_ocean: object, field_ice: object, field_land: object, grid_mean: object
        ) -> None:
            self.aggregate_surface_tiles(
                field_ocean=field_ocean,
                field_ice=field_ice,
                field_land=field_land,
                fraction_ocean=inp.frac_oce,
                fraction_ice=inp.frac_ice,
                fraction_land=inp.frac_lnd,
                grid_mean=grid_mean,
            )

        # the ice sensible flux is the lagged-state field just written by _run_ice
        aggregate(
            self._evap_ocean, self._evap_ice, inp.land_evapotrans, flux_state.evapotranspiration
        )
        aggregate(
            self._sensible_ocean,
            surface_state.lagged_ice_shflx,
            inp.land_sensible_hflx,
            flux_state.sensible_heat_flux,
        )
        aggregate(self._ustress_ocean, self._ustress_ice, self._ustress_land, flux_state.u_stress)
        aggregate(self._vstress_ocean, self._vstress_ice, self._vstress_land, flux_state.v_stress)
        aggregate(self._zero_tile, self._zero_tile, inp.land_q_snocpymlt, flux_state.q_snocpymlt)
