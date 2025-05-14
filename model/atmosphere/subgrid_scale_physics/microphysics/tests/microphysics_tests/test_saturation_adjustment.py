# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    saturation_adjustment as satad,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
    tracer_state as tracers,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils, helpers


@pytest.mark.parametrize(
    "experiment, model_top_height, damping_height, stretch_factor",
    [
        (dt_utils.WEISMAN_KLEMP_EXPERIMENT, 30000.0, 8000.0, 0.85),
    ],
)
@pytest.mark.parametrize(
    "date", ["2008-09-01T01:59:48.000", "2008-09-01T01:59:52.000", "2008-09-01T01:59:56.000"]
)
@pytest.mark.parametrize(
    "location, diagnose_values", [("nwp-gscp-interface", False), ("interface-nwp", True)]
)
def test_saturation_adjustement(
    experiment,
    location,
    diagnose_values,
    model_top_height,
    damping_height,
    stretch_factor,
    date,
    data_provider,
    grid_savepoint,
    metrics_savepoint,
    icon_grid,
    lowest_layer_thickness,
    backend,
):
    satad_init = data_provider.from_savepoint_satad_init(location=location, date=date)
    satad_exit = data_provider.from_savepoint_satad_exit(location=location, date=date)

    config = satad.SaturationAdjustmentConfig(
        tolerance=1e-3,
        max_iter=10,
        diagnose_variables_from_new_temperature=diagnose_values,
    )
    dtime = 2.0

    vertical_config = v_grid.VerticalGridConfig(icon_grid.num_levels)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )

    metric_state = satad.MetricStateSaturationAdjustment(
        ddqz_z_full=metrics_savepoint.ddqz_z_full()
    )

    saturation_adjustment = satad.SaturationAdjustment(
        config=config,
        grid=icon_grid,
        metric_state=metric_state,
        vertical_params=vertical_params,
        backend=backend,
    )
    tracer_state = tracers.TracerState(
        qv=satad_init.qv(),
        qc=satad_init.qc(),
        qr=satad_init.qr(),
        qi=satad_init.qi(),
        qs=satad_init.qs(),
        qg=satad_init.qg(),
    )
    prognostic_state = prognostics.PrognosticState(
        rho=satad_init.rho(),
        vn=data_alloc.zero_field(icon_grid, dims.EdgeDim, dims.KDim, dtype=float),
        w=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        exner=satad_init.exner(),
        theta_v=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
    )
    diagnostic_state = diagnostics.DiagnosticState(
        temperature=satad_init.temperature(),
        virtual_temperature=satad_init.virtual_temperature(),
        pressure=satad_init.pressure(),
        pressure_ifc=satad_init.pressure_ifc(),
        u=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        v=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
    )

    # WHEN
    # run saturation adjustment
    saturation_adjustment.run(
        dtime=dtime,
        prognostic_state=prognostic_state,
        diagnostic_state=diagnostic_state,
        tracer_state=tracer_state,
    )

    # apply tendencies

    updated_qv = tracer_state.qv.asnumpy() + saturation_adjustment.qv_tendency.asnumpy() * dtime
    updated_qc = tracer_state.qc.asnumpy() + saturation_adjustment.qc_tendency.asnumpy() * dtime
    updated_temperature = (
        diagnostic_state.temperature.asnumpy()
        + saturation_adjustment.temperature_tendency.asnumpy() * dtime
    )
    updated_exner = (
        prognostic_state.exner.asnumpy() + saturation_adjustment.exner_tendency.asnumpy() * dtime
    )
    if diagnose_values:
        pressure = (
            diagnostic_state.pressure.asnumpy()
            + saturation_adjustment.pressure_tendency.asnumpy() * dtime
        )
        pressure_ifc = (
            diagnostic_state.pressure_ifc.asnumpy()
            + saturation_adjustment.pressure_ifc_tendency.asnumpy() * dtime
        )
        virtual_temperature = (
            diagnostic_state.virtual_temperature.asnumpy()
            + saturation_adjustment.virtual_temperature_tendency.asnumpy() * dtime
        )
    else:
        pressure = diagnostic_state.pressure.asnumpy()
        pressure_ifc = diagnostic_state.pressure_ifc.asnumpy()
        virtual_temperature = diagnostic_state.virtual_temperature.asnumpy()

    assert helpers.dallclose(
        updated_qv,
        satad_exit.qv().asnumpy(),
        atol=1.0e-13,
    )
    assert helpers.dallclose(
        updated_qc,
        satad_exit.qc().asnumpy(),
        atol=1.0e-13,
    )
    assert helpers.dallclose(
        updated_temperature,
        satad_exit.temperature().asnumpy(),
        atol=1.0e-13,
    )
    assert helpers.dallclose(
        virtual_temperature,
        satad_exit.virtual_temperature().asnumpy(),
        atol=1.0e-13,
    )
    assert helpers.dallclose(
        updated_exner,
        satad_exit.exner().asnumpy(),
        atol=1.0e-13,
    )
    assert helpers.dallclose(
        pressure,
        satad_exit.pressure().asnumpy(),
        atol=1.0e-13,
    )
    assert helpers.dallclose(
        pressure_ifc,
        satad_exit.pressure_ifc().asnumpy(),
        atol=1.0e-13,
    )
