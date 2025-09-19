# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from devtools import Timer
import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    microphysics_options as mphys_options,
    single_moment_six_class_gscp_graupel as graupel,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
    tracer_state as tracers,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions, test_utils

from ..fixtures import *  # noqa: F403


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid
    from icon4py.model.testing import serialbox as sb


@pytest.mark.embedded_static_args
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, model_top_height",
    [
        (definitions.Experiments.WEISMAN_KLEMP_TORUS, 30000.0),
    ],
)
@pytest.mark.parametrize(
    "date", ["2008-09-01T02:00:00.000"]
)
def test_graupel(
    experiment: definitions.Experiments,
    model_top_height: ta.wpfloat,
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    icon_grid: icon_grid.IconGrid,
    lowest_layer_thickness: ta.wpfloat,
    backend: gtx_typing.Backend,
):
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
    )
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
    )

    metric_state = graupel.MetricStateIconGraupel(
        ddqz_z_full=metrics_savepoint.ddqz_z_full(),
    )

    entry_savepoint = data_provider.from_savepoint_weisman_klemp_graupel_entry(date=date)
    exit_savepoint = data_provider.from_savepoint_weisman_klemp_graupel_exit(date=date)

    print()
    solid = entry_savepoint.qc().ndarray + entry_savepoint.qr().ndarray + entry_savepoint.qi().ndarray + entry_savepoint.qs().ndarray + entry_savepoint.qg().ndarray
    xp = data_alloc.import_array_ns(solid)
    solid_count = xp.where(solid > 0.0, True, False)
    print("DEBUG: ", icon_grid.num_cells, xp.count_nonzero(solid_count))

    dtime = entry_savepoint.dtime()

    tracer_state = tracers.TracerState(
        qv=entry_savepoint.qv(),
        qc=entry_savepoint.qc(),
        qr=entry_savepoint.qr(),
        qi=entry_savepoint.qi(),
        qs=entry_savepoint.qs(),
        qg=entry_savepoint.qg(),
    )
    prognostic_state = prognostics.PrognosticState(
        rho=entry_savepoint.rho(),
        vn=None,
        w=None,
        exner=None,
        theta_v=None,
    )
    diagnostic_state = diagnostics.DiagnosticState(
        temperature=entry_savepoint.temperature(),
        virtual_temperature=None,
        pressure=entry_savepoint.pressure(),
        pressure_ifc=None,
        u=None,
        v=None,
    )

    graupel_config = graupel.SingleMomentSixClassIconGraupelConfig(
        liquid_autoconversion_option=mphys_options.LiquidAutoConversionType.SEIFERT_BEHENG,
        ice_stickeff_min=0.075,
        power_law_coeff_for_ice_mean_fall_speed=1.25,
        exponent_for_density_factor_in_ice_sedimentation=0.33,
        power_law_coeff_for_snow_fall_speed=20.0,
        rain_mu=0.0,
        rain_n0=1.0,
        snow2graupel_riming_coeff=0.5,
    )

    graupel_microphysics = graupel.SingleMomentSixClassIconGraupel(
        graupel_config=graupel_config,
        grid=icon_grid,
        metric_state=metric_state,
        vertical_params=vertical_params,
        backend=backend,
    )

    qnc = entry_savepoint.qnc()

    timer_first_timestep = Timer("TimeLoop: first time step", dp=6)
    timer_after_first_timestep = Timer("TimeLoop: after first time step", dp=6)
    for time_step in range(900):
        timer = timer_first_timestep if time_step == 0 else timer_after_first_timestep
        temperature_tendency = data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=backend
        )
        qv_tendency = data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=backend
        )
        qc_tendency = data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=backend
        )
        qr_tendency = data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=backend
        )
        qi_tendency = data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=backend
        )
        qs_tendency = data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=backend
        )
        qg_tendency = data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=backend
        )

        timer.start()
        graupel_microphysics.run(
            dtime,
            prognostic_state.rho,
            diagnostic_state.temperature,
            diagnostic_state.pressure,
            tracer_state.qv,
            tracer_state.qc,
            tracer_state.qr,
            tracer_state.qi,
            tracer_state.qs,
            tracer_state.qg,
            qnc,
            temperature_tendency,
            qv_tendency,
            qc_tendency,
            qr_tendency,
            qi_tendency,
            qs_tendency,
            qg_tendency,
        )
        timer.capture()

        new_qv = entry_savepoint.qv().ndarray + qv_tendency.ndarray * dtime
        new_qc = entry_savepoint.qc().ndarray + qc_tendency.ndarray * dtime
        new_qr = entry_savepoint.qr().ndarray + qr_tendency.ndarray * dtime
        new_qi = entry_savepoint.qi().ndarray + qi_tendency.ndarray * dtime
        new_qs = entry_savepoint.qs().ndarray + qs_tendency.ndarray * dtime
        new_qg = entry_savepoint.qg().ndarray + qg_tendency.ndarray * dtime

        new_temperature = (
            entry_savepoint.temperature().ndarray + temperature_tendency.ndarray * dtime
        )
        tracer_state = tracers.TracerState(
            qv=gtx.as_field((dims.CellDim, dims.KDim), data=new_qv, allocator=backend),
            qc=gtx.as_field((dims.CellDim, dims.KDim), data=new_qc, allocator=backend),
            qr=gtx.as_field((dims.CellDim, dims.KDim), data=new_qr, allocator=backend),
            qi=gtx.as_field((dims.CellDim, dims.KDim), data=new_qi, allocator=backend),
            qs=gtx.as_field((dims.CellDim, dims.KDim), data=new_qs, allocator=backend),
            qg=gtx.as_field((dims.CellDim, dims.KDim), data=new_qg, allocator=backend),
        )
        diagnostic_state = diagnostics.DiagnosticState(
            temperature=gtx.as_field((dims.CellDim, dims.KDim), data=new_temperature, allocator=backend),
            virtual_temperature=None,
            pressure=entry_savepoint.pressure(),
            pressure_ifc=None,
            u=None,
            v=None,
        )


    timer_first_timestep.summary(True)
    if time_step > 1:  # in case only one time step was run
        timer_after_first_timestep.summary(True)

    
    
    assert test_utils.dallclose(
        new_temperature,
        exit_savepoint.temperature().asnumpy(),
    )
    assert test_utils.dallclose(
        new_qv,
        exit_savepoint.qv().asnumpy(),
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qc,
        exit_savepoint.qc().asnumpy(),
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qr,
        exit_savepoint.qr().asnumpy(),
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qi,
        exit_savepoint.qi().asnumpy(),
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qs,
        exit_savepoint.qs().asnumpy(),
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qg,
        exit_savepoint.qg().asnumpy(),
        atol=1.0e-12,
    )

    assert test_utils.dallclose(
        graupel_microphysics.rain_precipitation_flux.asnumpy()[:, -1],
        exit_savepoint.rain_flux().asnumpy()[:],
        atol=9.0e-11,
    )
    assert test_utils.dallclose(
        graupel_microphysics.snow_precipitation_flux.asnumpy()[:, -1],
        exit_savepoint.snow_flux().asnumpy()[:],
        atol=9.0e-11,
    )
    assert test_utils.dallclose(
        graupel_microphysics.graupel_precipitation_flux.asnumpy()[:, -1],
        exit_savepoint.graupel_flux().asnumpy()[:],
        atol=9.0e-11,
    )
    assert test_utils.dallclose(
        graupel_microphysics.ice_precipitation_flux.asnumpy()[:, -1],
        exit_savepoint.ice_flux().asnumpy()[:],
        atol=9.0e-11,
    )
