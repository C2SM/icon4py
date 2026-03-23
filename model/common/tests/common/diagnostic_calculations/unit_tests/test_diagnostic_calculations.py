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
import pytest

import icon4py.model.common.grid.horizontal as h_grid
from icon4py.model.common import dimension as dims
from icon4py.model.common.diagnostic_calculations import stencils as diagnostic_stencils
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.interpolation.stencils import compute_edge_2_cell_vector_interpolation
from icon4py.model.common.states import diagnostic_state as diagnostics, tracer_state
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions, test_utils
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    processor_props,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [definitions.Experiments.JW])
def test_diagnose_temperature(
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
    tracers = tracer_state.TracerState(
        qv=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend
        ),
        qc=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend
        ),
        qr=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend
        ),
        qi=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend
        ),
        qs=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend
        ),
        qg=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend
        ),
    )

    diagnostic_stencils.diagnose_virtual_temperature_and_temperature_from_exner.with_backend(
        backend
    )(
        virtual_temperature=virtual_temperature,
        temperature=temperature,
        tracers=tracers,
        theta_v=theta_v,
        exner=exner,
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
@pytest.mark.parametrize("experiment", [definitions.Experiments.JW])
def test_diagnose_meridional_and_zonal_winds(
    data_provider: sb.IconSerialDataProvider,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: base_grid.Grid,
    backend: gtx_typing.Backend,
) -> None:
    prognostics_init_savepoint = data_provider.from_savepoint_prognostics_initial()
    vn = prognostics_init_savepoint.vn_now()
    rbv_vec_coeff_c1 = interpolation_savepoint.rbf_vec_coeff_c1()
    rbv_vec_coeff_c2 = interpolation_savepoint.rbf_vec_coeff_c2()

    diagnostics_reference_savepoint = data_provider.from_savepoint_diagnostics_initial()
    u_ref = diagnostics_reference_savepoint.zonal_wind().asnumpy()
    v_ref = diagnostics_reference_savepoint.meridional_wind().asnumpy()

    u = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend)
    v = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend)

    cell_domain = h_grid.domain(dims.CellDim)
    cell_end_lateral_boundary_level_2 = icon_grid.end_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    end_cell_end = icon_grid.end_index(cell_domain(h_grid.Zone.END))

    compute_edge_2_cell_vector_interpolation.compute_edge_2_cell_vector_interpolation.with_backend(
        backend
    )(
        p_e_in=vn,
        ptr_coeff_1=rbv_vec_coeff_c1,
        ptr_coeff_2=rbv_vec_coeff_c2,
        p_u_out=u,
        p_v_out=v,
        horizontal_start=cell_end_lateral_boundary_level_2,
        horizontal_end=end_cell_end,
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={
            "C2E2C2E": icon_grid.get_connectivity("C2E2C2E"),
        },
    )

    assert test_utils.dallclose(
        u.asnumpy(),
        u_ref,
    )

    assert test_utils.dallclose(
        v.asnumpy(),
        v_ref,
        atol=1.0e-13,
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [definitions.Experiments.JW])
def test_diagnose_surface_pressure(
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
        icon_grid, dims.CellDim, dims.KDim, dtype=float, extend={dims.KDim: 1}, allocator=backend
    )

    cell_domain = h_grid.domain(dims.CellDim)

    diagnostic_stencils.diagnose_surface_pressure.with_backend(backend)(
        surface_pressure=surface_pressure,
        exner=exner,
        virtual_temperature=virtual_temperature,
        ddqz_z_full=ddqz_z_full,
        horizontal_start=0,
        horizontal_end=icon_grid.end_index(cell_domain(h_grid.Zone.END)),
        vertical_start=icon_grid.num_levels,
        vertical_end=icon_grid.num_levels + 1,
        offset_provider={"Koff": dims.KDim},
    )

    assert test_utils.dallclose(
        surface_pressure.asnumpy()[:, icon_grid.num_levels],
        surface_pressure_ref,
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [definitions.Experiments.JW])
def test_diagnose_pressure(
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

    pressure_at_half_levels = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=float, extend={dims.KDim: 1}, allocator=backend
    )

    pressure_at_half_levels.ndarray[:, -1] = surface_pressure.ndarray

    diagnostic_stencils.diagnose_pressure.with_backend(backend)(
        pressure=pressure,
        pressure_at_half_levels=pressure_at_half_levels,
        surface_pressure=surface_pressure,
        virtual_temperature=virtual_temperature,
        ddqz_z_full=ddqz_z_full,
        horizontal_start=0,
        horizontal_end=icon_grid.end_index(cell_domain(h_grid.Zone.END)),
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={},
    )

    assert test_utils.dallclose(pressure_ifc_ref, pressure_at_half_levels.asnumpy())

    assert test_utils.dallclose(
        pressure_ref,
        pressure.asnumpy(),
    )


@pytest.mark.parametrize(
    "experiment, model_top_height, damping_height, stretch_factor",
    [(definitions.Experiments.WEISMAN_KLEMP_TORUS, 30000.0, 8000.0, 0.85)],
)
@pytest.mark.parametrize(
    "date", ["2008-09-01T01:59:48.000", "2008-09-01T01:59:52.000", "2008-09-01T01:59:56.000"]
)
@pytest.mark.parametrize("location", [("interface-nwp")])
@pytest.mark.datatest
def test_diagnostic_update_after_saturation_adjustement(
    location: str,
    date: str,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    icon_grid: base_grid.Grid,
    backend: gtx_typing.Backend,
) -> None:
    satad_init = data_provider.from_savepoint_satad_init(location=location, date=date)
    satad_exit = data_provider.from_savepoint_satad_exit(location=location, date=date)

    vertical_config = v_grid.VerticalGridConfig(icon_grid.num_levels)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
    )

    tracers = tracer_state.TracerState(
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
        pressure_at_half_levels=satad_init.pressure_ifc(),
        u=None,
        v=None,
    )

    cell_domain = h_grid.domain(dims.CellDim)
    start_cell_nudging = icon_grid.start_index(cell_domain(h_grid.Zone.NUDGING))
    end_cell_local = icon_grid.start_index(cell_domain(h_grid.Zone.END))
    diagnostic_stencils.diagnose_virtual_temperature_and_exner.with_backend(backend)(
        virtual_temperature=diagnostic_state.virtual_temperature,
        exner=exner,
        tracers=tracers,
        temperature=diagnostic_state.temperature,
        horizontal_start=start_cell_nudging,
        horizontal_end=end_cell_local,
        vertical_start=vertical_params.kstart_moist,
        vertical_end=icon_grid.num_levels,
        offset_provider={},
    )

    diagnostic_stencils.diagnose_surface_pressure.with_backend(backend)(
        surface_pressure=diagnostic_state.pressure_at_half_levels,
        exner=exner,
        virtual_temperature=diagnostic_state.virtual_temperature,
        ddqz_z_full=metrics_savepoint.ddqz_z_full(),
        horizontal_start=start_cell_nudging,
        horizontal_end=end_cell_local,
        vertical_start=icon_grid.num_levels,
        vertical_end=gtx.int32(icon_grid.num_levels + 1),
        offset_provider={"Koff": dims.KDim},
    )

    diagnostic_stencils.diagnose_pressure.with_backend(backend)(
        pressure=diagnostic_state.pressure,
        pressure_at_half_levels=diagnostic_state.pressure_at_half_levels,
        surface_pressure=diagnostic_state.surface_pressure,
        virtual_temperature=diagnostic_state.virtual_temperature,
        ddqz_z_full=metrics_savepoint.ddqz_z_full(),
        horizontal_start=start_cell_nudging,
        horizontal_end=end_cell_local,
        vertical_start=gtx.int32(0),
        vertical_end=icon_grid.num_levels,
        offset_provider={},
    )

    assert test_utils.dallclose(
        diagnostic_state.virtual_temperature.asnumpy(),
        satad_exit.virtual_temperature().asnumpy(),
        atol=1.0e-13,
    )
    assert test_utils.dallclose(
        exner.asnumpy(),
        satad_exit.exner().asnumpy(),
        atol=1.0e-13,
    )
    assert test_utils.dallclose(
        diagnostic_state.pressure.asnumpy(),
        satad_exit.pressure().asnumpy(),
        atol=1.0e-13,
    )
    assert test_utils.dallclose(
        diagnostic_state.pressure_at_half_levels.asnumpy(),
        satad_exit.pressure_ifc().asnumpy(),
        atol=1.0e-13,
    )
