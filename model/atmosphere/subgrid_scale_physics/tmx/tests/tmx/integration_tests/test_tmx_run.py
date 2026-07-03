# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end integration test of the Tmx granule (M6).

Constructs the granule from the serialized ICON state (exp.exclaim_ape_aesPhys)
and verifies one full ``run`` (Stages A to G) from the tmx-entry /
tmx-surface-fluxes savepoints against the tmx-exit savepoint (final
tendencies, dissipation heating and vertically integrated diagnostics).
Unlike the per-stage tests, nothing is seeded from intermediate savepoints:
this exercises the complete Fortran ``Compute`` sequence of mo_vdf.f90.
"""

from __future__ import annotations

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


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, date",
    [(definitions.Experiments.APE_AES, date) for date in TMX_DATES],
)
def test_tmx_full_run_single_step(  # noqa: PLR0917 [too-many-positional-arguments]
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    backend: gtx_typing.Backend | None,
    date: str,
) -> None:
    allocator = model_backends.get_allocator(backend)
    init_savepoint = data_provider.from_savepoint_tmx_init()
    entry_savepoint = data_provider.from_savepoint_tmx_entry(date=date)
    surface_fluxes_savepoint = data_provider.from_savepoint_tmx_surface_fluxes(date=date)
    exit_savepoint = data_provider.from_savepoint_tmx_exit(date=date)

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

    diagnostic_state = tmx_states.TmxDiagnosticState.allocate(icon_grid, allocator=allocator)
    tendency_state = tmx_states.TmxTendencyState.allocate(icon_grid, allocator=allocator)
    new_state = tmx_states.TmxNewState.allocate(icon_grid, allocator=allocator)

    granule.run(
        construct_input_state(entry_savepoint),
        construct_surface_flux_state(surface_fluxes_savepoint),
        diagnostic_state,
        tendency_state,
        new_state,
        init_savepoint.dtime(),
    )

    # final tendencies and Stage F diagnostics
    fields = (
        (tendency_state.ddt_temperature, exit_savepoint.tend_ta(), "tend_ta"),
        (tendency_state.ddt_qv, exit_savepoint.tend_qv(), "tend_qv"),
        (tendency_state.ddt_qc, exit_savepoint.tend_qc(), "tend_qc"),
        (tendency_state.ddt_qi, exit_savepoint.tend_qi(), "tend_qi"),
        (tendency_state.ddt_u, exit_savepoint.tend_ua(), "tend_ua"),
        (tendency_state.ddt_v, exit_savepoint.tend_va(), "tend_va"),
        (tendency_state.ddt_w, exit_savepoint.tend_wa(), "tend_wa"),
        (diagnostic_state.heating, exit_savepoint.heating(), "heating"),
        (diagnostic_state.dissip_ke, exit_savepoint.dissip_ke(), "dissip_ke"),
    )
    for actual, desired, name in fields:
        assert_scaled_allclose(actual.asnumpy(), desired.asnumpy(), err_msg=name)

    # Stage G vertically integrated diagnostics (2D)
    integrals = (
        (diagnostic_state.cptgz_vi, exit_savepoint.cptgzvi(), "cptgzvi"),
        (diagnostic_state.dissip_ke_vi, exit_savepoint.dissip_ke_vi(), "dissip_ke_vi"),
        (diagnostic_state.int_energy_vi, exit_savepoint.int_energy_vi(), "int_energy_vi"),
        (
            diagnostic_state.int_energy_vi_tend,
            exit_savepoint.tend_int_energy_vi(),
            "tend_int_energy_vi",
        ),
    )
    for actual, desired, name in integrals:
        assert_scaled_allclose(actual.asnumpy(), desired.asnumpy(), err_msg=name)

    # Stage G km/kh diagnostics: the bottom (nlev) row is excluded, it holds
    # the tile-aggregated surface exchange coefficients in the Fortran
    # (km_sfc/kh_sfc from mo_vdf_diag_smag.f90, out of scope of the
    # atmosphere-only port; the granule writes zero there)
    nlev = icon_grid.num_levels
    for actual, desired, name in (
        (diagnostic_state.km, exit_savepoint.km(), "km"),
        (diagnostic_state.kh, exit_savepoint.kh(), "kh"),
    ):
        assert_scaled_allclose(
            actual.asnumpy()[:, : nlev - 1],
            desired.asnumpy()[:, : nlev - 1],
            err_msg=name,
        )
