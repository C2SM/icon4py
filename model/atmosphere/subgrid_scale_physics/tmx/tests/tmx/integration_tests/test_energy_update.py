# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test of the tmx energy update component.

Constructs the component from the serialized ICON state (exp.exclaim_ape_aesPhys), with the
diagnostics taken from the tmx-diagnostics-exit savepoint and the diffused state from the
tmx-hydro-exit, tmx-temperature-exit and tmx-hor-wind-exit savepoints, and verifies one call
of `run` against the tmx-exit savepoint.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import energy_update, tmx_states
from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.testing import definitions, test_utils

from ..fixtures import *  # noqa: F403
from .utils import (
    RTOL,
    TMX_DATES,
    construct_input_state,
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
    [(definitions.Experiments.EXCLAIM_APE_AES, date) for date in TMX_DATES],
)
def test_tmx_run_energy_update_single_step(
    *,
    data_provider: sb.IconSerialDataProvider,
    metrics_savepoint: sb.MetricSavepoint,
    icon_grid: icon_grid_.IconGrid,
    backend: gtx_typing.Backend | None,
    date: str,
    experiment: definitions.Experiment,
) -> None:
    tmx_config = experiment.config.tmx
    assert tmx_config is not None
    allocator = model_backends.get_allocator(backend)
    diagnostics_savepoint = data_provider.from_savepoint_tmx_diagnostics_exit(date=date)
    hydro_savepoint = data_provider.from_savepoint_tmx_hydro_exit(date=date)
    temperature_savepoint = data_provider.from_savepoint_tmx_temperature_exit(date=date)
    hor_wind_savepoint = data_provider.from_savepoint_tmx_hor_wind_exit(date=date)
    exit_savepoint = data_provider.from_savepoint_tmx_exit(date=date)

    component = energy_update.EnergyUpdate(
        grid=icon_grid,
        metric_state=construct_metric_state(
            metrics_savepoint=metrics_savepoint,
            init_savepoint=data_provider.from_savepoint_tmx_init(),
            allocator=allocator,
        ),
        backend=backend,
        exchange=decomposition.SingleNodeExchange(),
        dissipation_factor=tmx_config.dissipation_factor,
        turb_prandtl=tmx_config.turb_prandtl,
        use_km_const=tmx_config.use_km_const,
        km_const=tmx_config.km_const,
    )
    diagnostic_state = dataclasses.replace(
        tmx_states.TmxDiagnosticState.allocate(icon_grid, allocator=allocator),
        km_ic=diagnostics_savepoint.km_ic(),
        kh_ic=diagnostics_savepoint.kh_ic(),
    )
    tendency_state = dataclasses.replace(
        tmx_states.TmxTendencyState.allocate(icon_grid, allocator=allocator),
        tend_temperature=temperature_savepoint.tend_ta(),
    )
    new_state = dataclasses.replace(
        tmx_states.TmxNewState.allocate(icon_grid, allocator=allocator),
        qv=hydro_savepoint.qv_new(),
        qc=hydro_savepoint.qc_new(),
        qi=hydro_savepoint.qi_new(),
        u=hor_wind_savepoint.ua_new(),
        v=hor_wind_savepoint.va_new(),
    )

    component.run(
        input_state=construct_input_state(data_provider.from_savepoint_tmx_entry(date=date)),
        surface_flux_state=construct_surface_flux_state(
            data_provider.from_savepoint_tmx_surface_fluxes(date=date)
        ),
        diagnostic_state=diagnostic_state,
        tendency_state=tendency_state,
        new_state=new_state,
        # ICON runs tmx with the model time step (`init_tmx`), `dt_vdf` only sets how
        # often tmx is called
        dtime=experiment.config.driver.dtime.total_seconds(),
    )

    # the surface level of km and kh is the surface exchange coefficient, written only with
    # `use_km_const`
    exchange_coefficient_levels = slice(
        None, None if tmx_config.use_km_const else icon_grid.num_levels - 1
    )
    # (computed, reference, absolute tolerance)
    fields = {
        "tend_ta": (tendency_state.tend_temperature, exit_savepoint.tend_ta(), 1.0e-18),
        "heating": (diagnostic_state.heating, exit_savepoint.heating(), 3.0e-13),
        "dissip_ke": (diagnostic_state.dissip_ke, exit_savepoint.dissip_ke(), 3.0e-13),
        "cptgzvi": (diagnostic_state.cptgz_vi, exit_savepoint.cptgzvi(), 3.0e-6),
        "dissip_ke_vi": (diagnostic_state.dissip_ke_vi, exit_savepoint.dissip_ke_vi(), 2.0e-12),
        "int_energy_vi": (diagnostic_state.int_energy_vi, exit_savepoint.int_energy_vi(), 3.0e-6),
        "tend_int_energy_vi": (
            diagnostic_state.int_energy_vi_tend,
            exit_savepoint.tend_int_energy_vi(),
            7.0e-9,
        ),
    }
    for name, (computed, reference, atol) in fields.items():
        test_utils.assert_dallclose(
            computed.asnumpy(), reference.asnumpy(), rtol=RTOL, atol=atol, err_msg=name
        )
    for name, computed, reference, atol in (
        ("km", diagnostic_state.km, exit_savepoint.km(), 0.0),
        ("kh", diagnostic_state.kh, exit_savepoint.kh(), 0.0),
    ):
        test_utils.assert_dallclose(
            computed.asnumpy()[:, exchange_coefficient_levels],
            reference.asnumpy()[:, exchange_coefficient_levels],
            rtol=RTOL,
            atol=atol,
            err_msg=name,
        )
