# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.grid.horizontal as h_grid
from icon4py.model.common import dimension as dims
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.grid import simple, vertical as v_grid
from icon4py.model.common.interpolation.stencils import edge_2_cell_vector_rbf_interpolation as rbf
from icon4py.model.common.physics.thermodynamics import (
    compute_pressure,
    compute_temperature,
    compute_tendencies,
)
from icon4py.model.common.states import diagnostic_state as diagnostics, tracer_states as tracers
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions as test_defs, test_utils
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    process_props,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [test_defs.Experiments.JW])
def test_compute_virtual_temperature_and_temperature(
    data_provider: sb.IconSerialDataProvider, icon_grid: base_grid.Grid, backend: gtx_typing.Backend
) -> None:
    diagnostic_reference_savepoint = data_provider.from_savepoint_diagnostics_initial()
    temperature_ref = diagnostic_reference_savepoint.temperature().asnumpy()
    virtual_temperature_ref = diagnostic_reference_savepoint.virtual_temperature().asnumpy()
    initial_prognostic_savepoint = data_provider.from_savepoint_prognostics_initial()
    exner = initial_prognostic_savepoint.exner_now()
    theta_v = initial_prognostic_savepoint.theta_v_now()

    temperature = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend
    )
    virtual_temperature = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend
    )

    qv = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend)
    qc = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend)
    qr = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend)
    qi = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend)
    qs = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend)
    qg = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend)

    compute_temperature.compute_virtual_temperature_and_temperature.with_backend(backend)(
        qv=qv,
        qc=qc,
        qr=qr,
        qi=qi,
        qs=qs,
        qg=qg,
        theta_v=theta_v,
        exner=exner,
        virtual_temperature=virtual_temperature,
        temperature=temperature,
        horizontal_start=0,
        horizontal_end=icon_grid.end_index(h_grid.domain(dims.CellDim)(h_grid.Zone.END)),
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={},
    )

    # only temperature is tested because there is no moisture in the JW test. i.e. temperature = virtual_temperature
    assert test_utils.dallclose(
        temperature.asnumpy(),
        temperature_ref,
    )
    assert test_utils.dallclose(
        virtual_temperature.asnumpy(),
        virtual_temperature_ref,
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [test_defs.Experiments.JW])
def test_compute_surface_pressure(
    data_provider: sb.IconSerialDataProvider,
    icon_grid: base_grid.Grid,
    backend: gtx_typing.Backend,
    metrics_savepoint: sb.MetricSavepoint,
) -> None:
    initial_diagnostic_savepoint = data_provider.from_savepoint_diagnostics_initial()
    surface_pressure_ref = initial_diagnostic_savepoint.pressure_sfc().asnumpy()
    initial_prognostic_savepoint = data_provider.from_savepoint_prognostics_initial()
    exner = initial_prognostic_savepoint.exner_now()
    virtual_temperature = initial_diagnostic_savepoint.virtual_temperature()
    ddqz_z_full = metrics_savepoint.ddqz_z_full()

    surface_pressure = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KHalfDim, dtype=float, allocator=backend
    )

    cell_domain = h_grid.domain(dims.CellDim)

    compute_pressure.compute_surface_pressure.with_backend(backend)(
        exner=exner,
        virtual_temperature=virtual_temperature,
        ddqz_z_full=ddqz_z_full,
        surface_pressure=surface_pressure,
        horizontal_start=0,
        horizontal_end=icon_grid.end_index(cell_domain(h_grid.Zone.END)),
        vertical_start=icon_grid.num_levels,
        vertical_end=icon_grid.num_levels + 1,
        offset_provider={},
    )

    assert test_utils.dallclose(
        surface_pressure.asnumpy()[:, icon_grid.num_levels],
        surface_pressure_ref,
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [test_defs.Experiments.JW])
def test_compute_hydrostatic_pressure(
    data_provider: sb.IconSerialDataProvider,
    icon_grid: base_grid.Grid,
    backend: gtx_typing.Backend,
    metrics_savepoint: sb.MetricSavepoint,
) -> None:
    ddqz_z_full = metrics_savepoint.ddqz_z_full()

    diagnostics_reference_savepoint = data_provider.from_savepoint_diagnostics_initial()
    virtual_temperature = diagnostics_reference_savepoint.temperature()
    surface_pressure = diagnostics_reference_savepoint.pressure_sfc()

    pressure_ifc_ref = diagnostics_reference_savepoint.pressure_ifc().asnumpy()
    pressure_ref = diagnostics_reference_savepoint.pressure().asnumpy()

    pressure = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend
    )
    cell_domain = h_grid.domain(dims.CellDim)

    pressure_ifc = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KHalfDim, dtype=float, allocator=backend
    )

    pressure_ifc.ndarray[:, -1] = surface_pressure.ndarray

    compute_pressure.compute_hydrostatic_pressure.with_backend(backend)(
        ddqz_z_full,
        virtual_temperature,
        surface_pressure,
        pressure,
        data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend),
        pressure_ifc,
        horizontal_start=0,
        horizontal_end=icon_grid.end_index(cell_domain(h_grid.Zone.END)),
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={},
    )

    assert test_utils.dallclose(pressure_ifc_ref, pressure_ifc.asnumpy())

    assert test_utils.dallclose(
        pressure_ref,
        pressure.asnumpy(),
    )


@pytest.mark.parametrize(
    "experiment_description",
    [test_defs.Experiments.WEISMAN_KLEMP_TORUS],
)
@pytest.mark.parametrize(
    "date", ["2008-09-01T01:59:48.000", "2008-09-01T01:59:52.000", "2008-09-01T01:59:56.000"]
)
@pytest.mark.parametrize("location", ["interface-nwp"])
@pytest.mark.datatest
def test_diagnostic_update_after_saturation_adjustement(  # noqa: PLR0917 [too-many-positional-arguments]
    location: str,
    date: str,
    experiment: test_defs.Experiment,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    icon_grid: base_grid.Grid,
    backend: gtx_typing.Backend,
) -> None:
    satad_init = data_provider.from_savepoint_satad_init(location=location, date=date)
    satad_exit = data_provider.from_savepoint_satad_exit(location=location, date=date)

    dtime = 2.0

    vertical_config = v_grid.VerticalGridConfig(icon_grid.num_levels)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
    )
    virtual_temperature_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, allocator=backend
    )
    exner_tendency = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)

    tracer_state = tracers.TracerState(
        qv=satad_exit.qv(),
        qc=satad_exit.qc(),
        qr=satad_init.qr(),
        qi=satad_init.qi(),
        qs=satad_init.qs(),
        qg=satad_init.qg(),
    )
    exner = satad_init.exner()

    diagnostic_state = diagnostics.DiagnosticState(
        temperature=satad_exit.temperature(),
        virtual_temperature=satad_init.virtual_temperature(),
        pressure=satad_init.pressure(),
        pressure_ifc=satad_init.pressure_ifc(),
        u=None,
        v=None,
    )

    cell_domain = h_grid.domain(dims.CellDim)
    start_cell_nudging = icon_grid.start_index(cell_domain(h_grid.Zone.NUDGING))
    end_cell_local = icon_grid.start_index(cell_domain(h_grid.Zone.END))
    compute_tendencies.compute_virtual_temperature_tendency.with_backend(backend)(
        dtime=dtime,
        qv=tracer_state.qv,
        qc=tracer_state.qc,
        qi=tracer_state.qi,
        qr=tracer_state.qr,
        qs=tracer_state.qs,
        qg=tracer_state.qg,
        temperature=diagnostic_state.temperature,
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

    compute_tendencies.compute_exner_tendency.with_backend(backend)(
        dtime=dtime,
        virtual_temperature=diagnostic_state.virtual_temperature,
        virtual_temperature_tendency=virtual_temperature_tendency,
        exner=exner,
        out=exner_tendency,
        domain={
            dims.CellDim: (start_cell_nudging, end_cell_local),
            dims.KDim: (vertical_params.kstart_moist, icon_grid.num_levels),
        },
        offset_provider={},
    )

    updated_exner = exner.asnumpy() + exner_tendency.asnumpy() * dtime

    compute_pressure.compute_surface_pressure.with_backend(backend)(
        gtx.as_field((dims.CellDim, dims.KDim), updated_exner, allocator=backend),
        gtx.as_field((dims.CellDim, dims.KDim), updated_virtual_temperature, allocator=backend),
        metrics_savepoint.ddqz_z_full(),
        diagnostic_state.pressure_ifc,
        horizontal_start=start_cell_nudging,
        horizontal_end=end_cell_local,
        vertical_start=icon_grid.num_levels,
        vertical_end=gtx.int32(icon_grid.num_levels + 1),
        offset_provider={},
    )

    compute_pressure.compute_hydrostatic_pressure.with_backend(backend)(
        metrics_savepoint.ddqz_z_full(),
        gtx.as_field((dims.CellDim, dims.KDim), updated_virtual_temperature, allocator=backend),
        diagnostic_state.surface_pressure,
        diagnostic_state.pressure,
        data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend),
        diagnostic_state.pressure_ifc,
        horizontal_start=start_cell_nudging,
        horizontal_end=end_cell_local,
        vertical_start=gtx.int32(0),
        vertical_end=icon_grid.num_levels,
        offset_provider={},
    )

    assert test_utils.dallclose(
        updated_virtual_temperature,
        satad_exit.virtual_temperature().asnumpy(),
        atol=1.0e-13,
    )
    assert test_utils.dallclose(
        updated_exner,
        satad_exit.exner().asnumpy(),
        atol=1.0e-13,
    )
    assert test_utils.dallclose(
        diagnostic_state.pressure.asnumpy(),
        satad_exit.pressure().asnumpy(),
        atol=1.0e-13,
    )
    assert test_utils.dallclose(
        diagnostic_state.pressure_ifc.asnumpy(),
        satad_exit.pressure_ifc().asnumpy(),
        atol=1.0e-13,
    )
