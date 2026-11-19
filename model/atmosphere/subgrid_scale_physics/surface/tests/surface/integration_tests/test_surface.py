# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.surface import (
    config as surface_config,
    surface,
    surface_states,
)
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.states.surface_flux_state import SurfaceFluxState
from icon4py.model.testing import definitions, test_utils

from ..fixtures import *  # noqa: F403


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid_
    from icon4py.model.testing import serialbox as sb


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, date",
    [
        (definitions.Experiments.EXCLAIM_APE_AES, date)
        for date in definitions.Experiments.EXCLAIM_APE_AES.dates
    ],
)
def test_surface_run_single_step(
    *,
    data_provider: sb.IconSerialDataProvider,
    icon_grid: icon_grid_.IconGrid,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
    date: str,
) -> None:
    """Against the `tmx-surface-fluxes` savepoint: the fluxes ICON's surface scheme hands to tmx."""
    config = experiment.config.surface
    assert config is not None
    assert config.surface_flux_type is surface_config.SurfaceFluxType.FIXED_HEAT_FLUXES
    component = surface.Surface(grid=icon_grid, config=config, backend=backend)

    entry_savepoint = data_provider.from_savepoint_tmx_entry(date=date)
    input_state = surface_states.SurfaceInputState(
        # views of the surface half level and the lowest full level
        surface_pressure=entry_savepoint.pres_ifc()[dims.KHalfDim(icon_grid.num_levels)],
        temperature=entry_savepoint.ta()[dims.KDim(icon_grid.num_levels - 1)],
    )
    out = SurfaceFluxState.allocate(icon_grid, allocator=model_backends.get_allocator(backend))
    component.run(input_state=input_state, out=out)

    reference = data_provider.from_savepoint_tmx_surface_fluxes(date=date)
    for name, computed, desired in (
        ("hfss", out.sensible_heat_flux, reference.hfss()),
        ("evspsbl", out.evapotranspiration, reference.evspsbl()),
        ("tauu", out.u_stress, reference.tauu()),
        ("tauv", out.v_stress, reference.tauv()),
        ("q_snocpymlt", out.q_snocpymlt, reference.q_snocpymlt()),
    ):
        test_utils.assert_dallclose(
            computed.asnumpy(), desired.asnumpy(), rtol=1.0e-14, err_msg=name
        )
    # the one flux that is not zero in this configuration
    assert np.all(np.abs(out.sensible_heat_flux.asnumpy()) > 0.0)
    # the energy fluxes are not serialized: ufts = hfss, and lhflx = 0 zeroes ufvs
    test_utils.assert_dallclose(
        out.thermal_energy_flux.asnumpy(), reference.hfss().asnumpy(), rtol=1.0e-14, err_msg="ufts"
    )
    assert np.all(out.vapor_energy_flux.asnumpy() == 0.0)
