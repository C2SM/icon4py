# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    saturation_adjustment as satad,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.diagnostic_calculations.stencils import (
    calculate_tendency,
    diagnose_pressure,
    diagnose_surface_pressure,
)
from icon4py.model.common.grid import horizontal as h_grid, vertical as v_grid
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
    backend,
):
    satad_init = data_provider.from_savepoint_satad_init(location=location, date=date)
    satad_exit = data_provider.from_savepoint_satad_exit(location=location, date=date)

    config = satad.SaturationAdjustmentConfig(
        tolerance=1e-3,
        max_iter=10,
    )
    dtime = 2.0

    vertical_config = v_grid.VerticalGridConfig(icon_grid.num_levels)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )
    temperature_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, backend=backend
    )
    qv_tendency = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, backend=backend)
    qc_tendency = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, backend=backend)
    virtual_temperature_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, backend=backend
    )
    exner_tendency = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, backend=backend)

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

    # run saturation adjustment
    saturation_adjustment.run(
        dtime=dtime,
        rho=prognostic_state.rho,
        temperature=diagnostic_state.temperature,
        qv=tracer_state.qv,
        qc=tracer_state.qc,
        temperature_tendency=temperature_tendency,
        qv_tendency=qv_tendency,
        qc_tendency=qc_tendency,
    )

    updated_qv = tracer_state.qv.asnumpy() + qv_tendency.asnumpy() * dtime
    updated_qc = tracer_state.qc.asnumpy() + qc_tendency.asnumpy() * dtime
    updated_temperature = (
        diagnostic_state.temperature.asnumpy() + temperature_tendency.asnumpy() * dtime
    )

    if diagnose_values:
        cell_domain = h_grid.domain(dims.CellDim)
        start_cell_nudging = icon_grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        end_cell_local = icon_grid.start_index(cell_domain(h_grid.Zone.END))
        calculate_tendency.calculate_virtual_temperature_tendency.with_backend(backend)(
            dtime=dtime,
            qv=gtx.as_field((dims.CellDim, dims.KDim), updated_qv, allocator=backend),
            qc=gtx.as_field((dims.CellDim, dims.KDim), updated_qc, allocator=backend),
            qi=tracer_state.qi,
            qr=tracer_state.qr,
            qs=tracer_state.qs,
            qg=tracer_state.qg,
            temperature=gtx.as_field(
                (dims.CellDim, dims.KDim), updated_temperature, allocator=backend
            ),
            virtual_temperature=diagnostic_state.virtual_temperature,
            virtual_temperature_tendency=virtual_temperature_tendency,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=vertical_params.kstart_moist,
            vertical_end=icon_grid.num_levels,
            offset_provider={},
        )

        updated_virtual_temperature = (
            diagnostic_state.virtual_temperature.asnumpy()
            + virtual_temperature_tendency.asnumpy() * dtime
        )

        calculate_tendency.calculate_exner_tendency.with_backend(backend)(
            dtime=dtime,
            virtual_temperature=diagnostic_state.virtual_temperature,
            virtual_temperature_tendency=virtual_temperature_tendency,
            exner=prognostic_state.exner,
            exner_tendency=exner_tendency,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=vertical_params.kstart_moist,
            vertical_end=icon_grid.num_levels,
            offset_provider={},
        )

        updated_exner = prognostic_state.exner.asnumpy() + exner_tendency.asnumpy() * dtime

        diagnose_surface_pressure.diagnose_surface_pressure.with_backend(backend)(
            gtx.as_field((dims.CellDim, dims.KDim), updated_exner, allocator=backend),
            gtx.as_field((dims.CellDim, dims.KDim), updated_virtual_temperature, allocator=backend),
            metric_state.ddqz_z_full,
            diagnostic_state.pressure_ifc,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=icon_grid.num_levels,
            vertical_end=gtx.int32(icon_grid.num_levels + 1),
            offset_provider={"Koff": dims.KDim},
        )

        diagnose_pressure.diagnose_pressure.with_backend(backend)(
            metric_state.ddqz_z_full,
            gtx.as_field((dims.CellDim, dims.KDim), updated_virtual_temperature, allocator=backend),
            diagnostic_state.surface_pressure,
            diagnostic_state.pressure,
            diagnostic_state.pressure_ifc,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=gtx.int32(0),
            vertical_end=icon_grid.num_levels,
            offset_provider={},
        )
    else:
        updated_exner = prognostic_state.exner.asnumpy()
        updated_virtual_temperature = diagnostic_state.virtual_temperature.asnumpy()

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
        updated_virtual_temperature,
        satad_exit.virtual_temperature().asnumpy(),
        atol=1.0e-13,
    )
    assert helpers.dallclose(
        updated_exner,
        satad_exit.exner().asnumpy(),
        atol=1.0e-13,
    )
    assert helpers.dallclose(
        diagnostic_state.pressure.asnumpy(),
        satad_exit.pressure().asnumpy(),
        atol=1.0e-13,
    )
    assert helpers.dallclose(
        diagnostic_state.pressure_ifc.asnumpy(),
        satad_exit.pressure_ifc().asnumpy(),
        atol=1.0e-13,
    )
