# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test of the tmx surface exchange-coefficient stage.

Reproduces ``Compute_diagnostics`` (mo_vdf_sfc.f90) for the ocean tile of the
aqua-planet archive: relative wind, saturation humidity, surface density,
potential temperatures, Charnock roughness and the 5-iteration
Monin-Obukhov / Businger exchange solver, verified against ``tmx-surface-exchange``.

This stage is computed regardless of ``isrfc_type`` (that switch only bypasses
the surface *fluxes*), so it is validatable on the ``isrfc_type==1`` archive.
The Charnock roughness at step n uses the momentum coefficient of step n-1, so
``ocean_km`` is seeded from the previous step's savepoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface import surface, surface_states
from icon4py.model.common import model_backends
from icon4py.model.common.utils import fortran_config
from icon4py.model.testing import definitions

from ..fixtures import *  # noqa: F403
from .surface_utils import construct_surface_input_state
from .utils import TMX_DATES, assert_scaled_allclose


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.decomposition import definitions as decomposition
    from icon4py.model.common.grid import icon as icon_grid_
    from icon4py.model.testing import serialbox as sb


# The surface scheme also runs at the initialization step (00:00:00), so the
# previous-step momentum coefficient of the first verified step is available.
_INIT_DATE = "2008-09-01T00:00:00.000"
_PREV_DATE = {TMX_DATES[0]: _INIT_DATE, TMX_DATES[1]: TMX_DATES[0]}


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, date",
    [(definitions.Experiments.EXCLAIM_APE_AES, date) for date in TMX_DATES],
)
def test_surface_exchange_coefficients_single_step(
    *,
    data_provider: sb.IconSerialDataProvider,
    icon_grid: icon_grid_.IconGrid,
    backend: gtx_typing.Backend | None,
    date: str,
    experiment_description: definitions.ExperimentDescription,
    process_props: decomposition.ProcessProperties,
) -> None:
    allocator = model_backends.get_allocator(backend)
    atmo_dict = load_fortran_dict(  # noqa: F405 (from ..fixtures import *)
        experiment_description=experiment_description,
        process_props=process_props,
        fname=fortran_config.ATM_DICT_FNAME,
    )
    config = surface.TmxSurfaceConfig.from_fortran_dict(atmo_dict)
    params = surface.TmxSurfaceParams()
    granule = surface.TmxSurface(grid=icon_grid, config=config, params=params, backend=backend)

    entry_savepoint = data_provider.from_savepoint_tmx_surface_entry(date=date)
    exchange_savepoint = data_provider.from_savepoint_tmx_surface_exchange(date=date)
    previous_exchange_savepoint = data_provider.from_savepoint_tmx_surface_exchange(
        date=_PREV_DATE[date]
    )

    input_state = construct_surface_input_state(entry_savepoint, icon_grid, allocator)
    surface_state = surface_states.SurfaceState.allocate(icon_grid, allocator=allocator)
    # Charnock roughness lag: seed with the previous step's momentum coefficient
    surface_state.ocean_km.ndarray[...] = previous_exchange_savepoint.km_oce().ndarray

    granule._run_ocean(input_state, surface_state)

    # (produced field, reference savepoint accessor, name). thetav_atm/thetav_oce
    # and moist_rich_oce are Fortran diagnostics not needed by the flux state, so
    # they are not computed by the granule and not verified here.
    fields = (
        (surface_state.ocean_km, exchange_savepoint.km_oce(), "km_oce"),
        (granule._kh_ocean, exchange_savepoint.kh_oce(), "kh_oce"),
        (granule._rough_m, exchange_savepoint.rough_m_oce(), "rough_m_oce"),
        (granule._rough_h, exchange_savepoint.rough_h_oce(), "rough_h_oce"),
        (granule._qsat_ocean, exchange_savepoint.qsat_oce(), "qsat_oce"),
        (granule._rho_sfc, exchange_savepoint.rho_oce(), "rho_oce"),
        (granule._theta_atm, exchange_savepoint.theta_atm(), "theta_atm"),
        (granule._theta_sfc, exchange_savepoint.theta_oce(), "theta_oce"),
        (granule._wind_rel, exchange_savepoint.wind_rel_oce(), "wind_rel_oce"),
    )
    for actual, desired, name in fields:
        assert_scaled_allclose(actual.asnumpy(), desired.asnumpy(), rtol=1e-9, err_msg=name)
