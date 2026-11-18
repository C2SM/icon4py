# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The surface scheme of ICON's AES physics: the surface fluxes that tmx mixes into the atmosphere."""

from __future__ import annotations

import logging
import typing

from icon4py.model.atmosphere.subgrid_scale_physics.surface import (
    config as surface_config,
    surface_states,
)
from icon4py.model.atmosphere.subgrid_scale_physics.surface.stencils import prescribed_flux_stencils
from icon4py.model.common import dimension as dims, model_backends, type_alias as ta
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.model_options import setup_program
from icon4py.model.common.utils import data_allocation as data_alloc


if typing.TYPE_CHECKING:
    from icon4py.model.common import field_type_aliases as fa
    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.common.states.surface_flux_state import SurfaceFluxState


log = logging.getLogger(__name__)


class Surface:
    """
    One step of the surface scheme (``Compute`` of ``t_vdf_sfc`` in mo_vdf_sfc.f90), restricted
    to the prescribed fluxes of ``isrfc_type = 1`` over a sea surface of fixed temperature
    (``ape_sst_case = sst_const``), the only scheme with reference data.

    The surface temperature is a constant field built from the configuration, as ICON's ocean
    tile holds it without a mixed-layer ocean (mo_vdf_sfc.f90 zeroes its tendency). The scheme
    runs on the prognostic cells (``grf_bdywidth_c + 1`` to ``min_rlcell_int`` in
    mo_tmx_field_class.f90), the range tmx consumes the fluxes on.

    ICON's interactive surface (``isrfc_type = 0``) is tiled: each of the ocean, sea-ice and
    land tiles has its own surface temperature and saturation humidity and computes its own
    fluxes, which are aggregated with the tile fractions into the fields tmx receives. Porting
    it means a loop over the tiles around the flux computation and the aggregation after it,
    with a surface state holding the per-tile temperatures; the prescribed scheme is the
    one-tile case with fraction one.
    """

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        config: surface_config.SurfaceConfig,
        backend: model_backends.BackendLike,
    ) -> None:
        self.surface_temperature: fa.CellField[ta.wpfloat] = data_alloc.constant_field(
            grid,
            config.sea_surface_temperature,
            dims.CellDim,
            dtype=ta.wpfloat,
            allocator=model_backends.get_allocator(backend),
        )
        """Sea surface temperature (``ts``) [K], constant."""

        cell_domain = h_grid.domain(dims.CellDim)
        self._compute_prescribed_surface_fluxes = setup_program(
            backend=backend,
            program=prescribed_flux_stencils.compute_prescribed_surface_fluxes,
            constant_args={
                "surface_temperature": self.surface_temperature,
                "shflx": ta.wpfloat(config.kinematic_sensible_heat_flux),
                "lhflx": ta.wpfloat(config.kinematic_latent_heat_flux),
            },
            horizontal_sizes={
                "horizontal_start": grid.start_index(cell_domain(h_grid.Zone.NUDGING)),
                "horizontal_end": grid.end_index(cell_domain(h_grid.Zone.LOCAL)),
            },
            offset_provider={},
        )

    def run(self, *, input_state: surface_states.SurfaceInputState, out: SurfaceFluxState) -> None:
        """
        Compute the surface fluxes of the step into `out`.

        Writes the heat and moisture fluxes and the energy fluxes they carry; the wind stress
        and `q_snocpymlt` of `out` are left as they are (zero from `allocate`), as ICON's
        prescribed-flux branch leaves them.
        """
        log.debug("surface fluxes (prescribed): start")
        self._compute_prescribed_surface_fluxes(
            surface_pressure=input_state.surface_pressure,
            temperature=input_state.temperature,
            evapotranspiration=out.evapotranspiration,
            sensible_heat_flux=out.sensible_heat_flux,
            latent_heat_flux=out.latent_heat_flux,
            thermal_energy_flux=out.thermal_energy_flux,
            vapor_energy_flux=out.vapor_energy_flux,
        )
        log.debug("surface fluxes (prescribed): end")
