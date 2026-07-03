# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests of the Tmx granule scalar diffusion stages (M4).

Constructs the granule from the serialized ICON state (exp.exclaim_ape_aesPhys)
and verifies one call of ``run_hydrometeor_diffusion`` (Stage B) against the
tmx-hydro-exit savepoint and one call of ``run_temperature_diffusion``
(Stage C) against the tmx-temperature-exit savepoint. Both stages are seeded
from the tmx-diagnostics-exit savepoint (``kh_ic``, ``km_ie``) and the
temperature stage additionally from the tmx-hydro-exit savepoint (the new
moisture state), so that failures do not cascade between the stages.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx, tmx_states
from icon4py.model.common import model_backends
from icon4py.model.testing import definitions, test_utils

from ..fixtures import *  # noqa: F403
from .utils import (
    TMX_DATES,
    assert_scaled_allclose,
    construct_input_state,
    construct_interpolation_state,
    construct_metric_state,
    construct_surface_flux_state,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid_
    from icon4py.model.testing import serialbox as sb


@dataclasses.dataclass
class _GranuleSetup:
    granule: tmx.Tmx
    input_state: tmx_states.TmxInputState
    surface_flux_state: tmx_states.TmxSurfaceFluxState
    diagnostic_state: tmx_states.TmxDiagnosticState
    tendency_state: tmx_states.TmxTendencyState
    new_state: tmx_states.TmxNewState
    dtime: float


def _setup_granule(  # noqa: PLR0917 [too-many-positional-arguments]
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    backend: gtx_typing.Backend | None,
    date: str,
    config: tmx.TmxConfig,
    dtime: float,
) -> _GranuleSetup:
    """Construct the granule and its states, seeded from the savepoints."""
    allocator = model_backends.get_allocator(backend)
    init_savepoint = data_provider.from_savepoint_tmx_init()
    entry_savepoint = data_provider.from_savepoint_tmx_entry(date=date)
    surface_fluxes_savepoint = data_provider.from_savepoint_tmx_surface_fluxes(date=date)
    diagnostics_exit_savepoint = data_provider.from_savepoint_tmx_diagnostics_exit(date=date)

    granule = tmx.Tmx(
        grid=icon_grid,
        config=config,
        params=tmx.TmxParams(config),
        vertical_grid=None,
        metric_state=construct_metric_state(
            metrics_savepoint, init_savepoint, grid_savepoint, allocator
        ),
        interpolation_state=construct_interpolation_state(interpolation_savepoint),
        edge_params=grid_savepoint.construct_edge_geometry(),
        cell_params=grid_savepoint.construct_cell_geometry(),
        backend=backend,
    )

    # seed the Stage A diagnostics consumed by the scalar diffusion from the
    # diagnostics-exit savepoint instead of running Stage A (failures there
    # must not cascade into these tests)
    diagnostic_state = dataclasses.replace(
        tmx_states.TmxDiagnosticState.allocate(icon_grid, allocator=allocator),
        kh_ic=diagnostics_exit_savepoint.kh_ic(),
        km_ie=diagnostics_exit_savepoint.km_ie(),
    )

    return _GranuleSetup(
        granule=granule,
        input_state=construct_input_state(entry_savepoint),
        surface_flux_state=construct_surface_flux_state(surface_fluxes_savepoint),
        diagnostic_state=diagnostic_state,
        tendency_state=tmx_states.TmxTendencyState.allocate(icon_grid, allocator=allocator),
        new_state=tmx_states.TmxNewState.allocate(icon_grid, allocator=allocator),
        dtime=dtime,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, date",
    [(definitions.Experiments.EXCLAIM_APE_AES, date) for date in TMX_DATES],
)
def test_tmx_run_hydrometeor_diffusion_single_step(  # noqa: PLR0917 [too-many-positional-arguments]
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    backend: gtx_typing.Backend | None,
    date: str,
    tmx_config: tmx.TmxConfig,
    tmx_dtime: float,
) -> None:
    setup = _setup_granule(
        data_provider,
        grid_savepoint,
        metrics_savepoint,
        interpolation_savepoint,
        icon_grid,
        backend,
        date,
        tmx_config,
        tmx_dtime,
    )
    exit_savepoint = data_provider.from_savepoint_tmx_hydro_exit(date=date)

    setup.granule.run_hydrometeor_diffusion(
        setup.input_state,
        setup.surface_flux_state,
        setup.diagnostic_state,
        setup.tendency_state,
        setup.new_state,
        setup.dtime,
    )

    fields = (
        (setup.tendency_state.ddt_qv, exit_savepoint.tend_qv(), "tend_qv"),
        (setup.tendency_state.ddt_qc, exit_savepoint.tend_qc(), "tend_qc"),
        (setup.tendency_state.ddt_qi, exit_savepoint.tend_qi(), "tend_qi"),
        (setup.new_state.qv, exit_savepoint.qv_new(), "qv_new"),
        (setup.new_state.qc, exit_savepoint.qc_new(), "qc_new"),
        (setup.new_state.qi, exit_savepoint.qi_new(), "qi_new"),
    )
    for actual, desired, name in fields:
        assert_scaled_allclose(actual.asnumpy(), desired.asnumpy(), err_msg=name)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, date",
    [(definitions.Experiments.EXCLAIM_APE_AES, date) for date in TMX_DATES],
)
def test_tmx_run_temperature_diffusion_single_step(  # noqa: PLR0917 [too-many-positional-arguments]
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    backend: gtx_typing.Backend | None,
    date: str,
    tmx_config: tmx.TmxConfig,
    tmx_dtime: float,
) -> None:
    setup = _setup_granule(
        data_provider,
        grid_savepoint,
        metrics_savepoint,
        interpolation_savepoint,
        icon_grid,
        backend,
        date,
        tmx_config,
        tmx_dtime,
    )
    hydro_exit_savepoint = data_provider.from_savepoint_tmx_hydro_exit(date=date)
    exit_savepoint = data_provider.from_savepoint_tmx_temperature_exit(date=date)

    # seed the new moisture state (energy_type = 2 recovers the temperature
    # with the *new* tracers) from the hydro-exit savepoint instead of running
    # Stage B
    new_state = dataclasses.replace(
        setup.new_state,
        qv=hydro_exit_savepoint.qv_new(),
        qc=hydro_exit_savepoint.qc_new(),
        qi=hydro_exit_savepoint.qi_new(),
    )

    setup.granule.run_temperature_diffusion(
        setup.input_state,
        setup.surface_flux_state,
        setup.diagnostic_state,
        setup.tendency_state,
        new_state,
        setup.dtime,
    )

    # the energy computed from the (old) input state is a direct chain
    # (no solve); the serialized field includes the halo values updated by the
    # sync before the horizontal diffusion, which is a no-op on a single node
    test_utils.assert_dallclose(
        setup.granule.energy.asnumpy(), exit_savepoint.energy().asnumpy(), err_msg="energy"
    )
    fields = (
        (setup.granule.tend_energy, exit_savepoint.tend_energy(), "tend_energy"),
        (new_state.temperature, exit_savepoint.ta_new(), "ta_new"),
        (setup.tendency_state.ddt_temperature, exit_savepoint.tend_ta(), "tend_ta"),
    )
    for actual, desired, name in fields:
        assert_scaled_allclose(actual.asnumpy(), desired.asnumpy(), err_msg=name)
