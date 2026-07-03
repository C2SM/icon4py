# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests of the Tmx granule momentum diffusion stages (M5).

Constructs the granule from the serialized ICON state (exp.exclaim_ape_aesPhys)
and verifies one call of ``run_horizontal_wind_diffusion`` (Stage D) against
the tmx-hor-wind-exit savepoint and one call of ``run_vertical_wind_diffusion``
(Stage E) against the tmx-vert-wind-exit savepoint. Both stages are seeded
from the tmx-diagnostics-exit savepoint (the Stage A diagnostics they consume)
instead of running Stage A, so that failures do not cascade between the
stages. Stage E only reads the input state and the Stage A diagnostics, so no
Stage D outputs need to be seeded.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx, tmx_states
from icon4py.model.common import model_backends
from icon4py.model.testing import definitions

from ..fixtures import *  # noqa: F403
from .utils import (
    TMX_DATES,
    assert_scaled_allclose,
    construct_config,
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
) -> _GranuleSetup:
    """Construct the granule and its states, seeded from the savepoints."""
    allocator = model_backends.get_allocator(backend)
    init_savepoint = data_provider.from_savepoint_tmx_init()
    entry_savepoint = data_provider.from_savepoint_tmx_entry(date=date)
    surface_fluxes_savepoint = data_provider.from_savepoint_tmx_surface_fluxes(date=date)
    diagnostics_exit_savepoint = data_provider.from_savepoint_tmx_diagnostics_exit(date=date)

    config = construct_config(init_savepoint)
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

    # seed the Stage A diagnostics consumed by the momentum diffusion from the
    # diagnostics-exit savepoint instead of running Stage A (failures there
    # must not cascade into these tests)
    diagnostic_state = dataclasses.replace(
        tmx_states.TmxDiagnosticState.allocate(icon_grid, allocator=allocator),
        vn=diagnostics_exit_savepoint.vn(),
        u_vert=diagnostics_exit_savepoint.u_vert(),
        v_vert=diagnostics_exit_savepoint.v_vert(),
        w_vert=diagnostics_exit_savepoint.w_vert(),
        w_ie=diagnostics_exit_savepoint.w_ie(),
        rho_ic=diagnostics_exit_savepoint.rho_ic(),
        div_c=diagnostics_exit_savepoint.div_c(),
        km_c=diagnostics_exit_savepoint.km_c(),
        km_ic=diagnostics_exit_savepoint.km_ic(),
        km_iv=diagnostics_exit_savepoint.km_iv(),
        km_ie=diagnostics_exit_savepoint.km_ie(),
    )

    return _GranuleSetup(
        granule=granule,
        input_state=construct_input_state(entry_savepoint),
        surface_flux_state=construct_surface_flux_state(surface_fluxes_savepoint),
        diagnostic_state=diagnostic_state,
        tendency_state=tmx_states.TmxTendencyState.allocate(icon_grid, allocator=allocator),
        new_state=tmx_states.TmxNewState.allocate(icon_grid, allocator=allocator),
        dtime=init_savepoint.dtime(),
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, date",
    [(definitions.Experiments.APE_AES, date) for date in TMX_DATES],
)
def test_tmx_run_horizontal_wind_diffusion_single_step(  # noqa: PLR0917 [too-many-positional-arguments]
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    backend: gtx_typing.Backend | None,
    date: str,
) -> None:
    setup = _setup_granule(
        data_provider,
        grid_savepoint,
        metrics_savepoint,
        interpolation_savepoint,
        icon_grid,
        backend,
        date,
    )
    exit_savepoint = data_provider.from_savepoint_tmx_hor_wind_exit(date=date)

    setup.granule.run_horizontal_wind_diffusion(
        setup.input_state,
        setup.surface_flux_state,
        setup.diagnostic_state,
        setup.tendency_state,
        setup.new_state,
        setup.dtime,
    )

    fields = (
        (setup.granule.tot_tend, exit_savepoint.tot_tend(), "tot_tend"),
        (setup.tendency_state.ddt_u, exit_savepoint.tend_ua(), "tend_ua"),
        (setup.tendency_state.ddt_v, exit_savepoint.tend_va(), "tend_va"),
        (setup.new_state.u, exit_savepoint.ua_new(), "ua_new"),
        (setup.new_state.v, exit_savepoint.va_new(), "va_new"),
    )
    for actual, desired, name in fields:
        assert_scaled_allclose(actual.asnumpy(), desired.asnumpy(), err_msg=name)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, date",
    [(definitions.Experiments.APE_AES, date) for date in TMX_DATES],
)
def test_tmx_run_vertical_wind_diffusion_single_step(  # noqa: PLR0917 [too-many-positional-arguments]
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    backend: gtx_typing.Backend | None,
    date: str,
) -> None:
    setup = _setup_granule(
        data_provider,
        grid_savepoint,
        metrics_savepoint,
        interpolation_savepoint,
        icon_grid,
        backend,
        date,
    )
    exit_savepoint = data_provider.from_savepoint_tmx_vert_wind_exit(date=date)
    # tend_wa is compared against the tmx-exit savepoint: on GPU runs the
    # tmx-vert-wind-exit serialization races with the still-running ASYNC(1)
    # horizontal-stress kernel, so its tend_wa copy predates the horizontal
    # contribution (wa_new, updated to host a moment later, is complete, and
    # nothing modifies tend_wa between the two savepoints). Verified on the
    # v06 archive: wa_new == wa + tend_wa(tmx-exit) * dtime to 3e-19, while
    # tend_wa(tmx-vert-wind-exit) is off by exactly the horizontal term.
    final_savepoint = data_provider.from_savepoint_tmx_exit(date=date)

    setup.granule.run_vertical_wind_diffusion(
        setup.input_state,
        setup.diagnostic_state,
        setup.tendency_state,
        setup.new_state,
        setup.dtime,
    )

    fields = (
        (setup.tendency_state.ddt_w, final_savepoint.tend_wa(), "tend_wa"),
        (setup.new_state.w, exit_savepoint.wa_new(), "wa_new"),
    )
    for actual, desired, name in fields:
        assert_scaled_allclose(actual.asnumpy(), desired.asnumpy(), err_msg=name)
