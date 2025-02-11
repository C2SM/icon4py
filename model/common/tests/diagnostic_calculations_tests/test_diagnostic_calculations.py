# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import icon4py.model.common.grid.horizontal as h_grid
from icon4py.model.common import constants as phy_const, dimension as dims
from icon4py.model.common.diagnostic_calculations.stencils import (
    diagnose_pressure as pressure,
    diagnose_surface_pressure as surface_pressure,
    diagnose_temperature as temperature,
)
from icon4py.model.common.interpolation.stencils import (
    edge_2_cell_vector_rbf_interpolation as rbf,
)
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state,
    tracer_state as tracers,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils, helpers


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment",
    [
        dt_utils.JABW_EXPERIMENT,
    ],
)
def test_diagnose_temperature(
    experiment,
    data_provider,
    icon_grid,
    backend,
):
    sp = data_provider.from_savepoint_jabw_final()
    icon_diagnostics_output_sp = data_provider.from_savepoint_jabw_diagnostic()
    prognostic_state_now = prognostic_state.PrognosticState(
        rho=sp.rho(),
        w=None,
        vn=sp.vn(),
        exner=sp.exner(),
        theta_v=sp.theta_v(),
    )
    diagnostic_state = diagnostics.DiagnosticState(
        temperature=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend
        ),
        virtual_temperature=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend
        ),
        pressure=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend
        ),
        pressure_ifc=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, extend={dims.KDim: 1}, backend=backend
        ),
        u=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend),
        v=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend),
    )
    tracer_state = tracers.TracerState(
        qv=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend),
        qc=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend),
        qr=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend),
        qi=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend),
        qs=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend),
        qg=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend),
    )

    temperature.diagnose_virtual_temperature_and_temperature.with_backend(backend)(
        tracer_state.qv,
        tracer_state.qc,
        tracer_state.qr,
        tracer_state.qi,
        tracer_state.qs,
        tracer_state.qg,
        prognostic_state_now.theta_v,
        prognostic_state_now.exner,
        diagnostic_state.virtual_temperature,
        diagnostic_state.temperature,
        phy_const.RV_O_RD_MINUS_1,
        horizontal_start=0,
        horizontal_end=icon_grid.end_index(h_grid.domain(dims.CellDim)(h_grid.Zone.END)),
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={},
    )

    # only temperature is tested because there is no moisture in the JW test. i.e. temperature = virtual_temperature
    assert helpers.dallclose(
        diagnostic_state.temperature.asnumpy(),
        icon_diagnostics_output_sp.temperature().asnumpy(),
    )
    assert helpers.dallclose(
        diagnostic_state.virtual_temperature.asnumpy(),
        icon_diagnostics_output_sp.temperature().asnumpy(),
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment",
    [
        dt_utils.JABW_EXPERIMENT,
    ],
)
def test_diagnose_meridional_and_zonal_winds(
    experiment,
    data_provider,
    icon_grid,
    backend,
):
    sp = data_provider.from_savepoint_jabw_final()
    icon_diagnostics_output_sp = data_provider.from_savepoint_jabw_diagnostic()
    prognostic_state_now = prognostic_state.PrognosticState(
        rho=sp.rho(),
        w=None,
        vn=sp.vn(),
        exner=sp.exner(),
        theta_v=sp.theta_v(),
    )
    diagnostic_state = diagnostics.DiagnosticState(
        temperature=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend
        ),
        pressure=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend
        ),
        pressure_ifc=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, extend={dims.KDim: 1}, backend=backend
        ),
        u=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend),
        v=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend),
        virtual_temperature=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend
        ),
    )
    cell_domain = h_grid.domain(dims.CellDim)
    rbv_vec_coeff_c1 = data_provider.from_interpolation_savepoint().rbf_vec_coeff_c1()
    rbv_vec_coeff_c2 = data_provider.from_interpolation_savepoint().rbf_vec_coeff_c2()
    cell_end_lateral_boundary_level_2 = icon_grid.end_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    end_cell_end = icon_grid.end_index(cell_domain(h_grid.Zone.END))
    rbf.edge_2_cell_vector_rbf_interpolation.with_backend(backend)(
        prognostic_state_now.vn,
        rbv_vec_coeff_c1,
        rbv_vec_coeff_c2,
        diagnostic_state.u,
        diagnostic_state.v,
        cell_end_lateral_boundary_level_2,
        end_cell_end,
        0,
        icon_grid.num_levels,
        offset_provider={
            "C2E2C2E": icon_grid.get_offset_provider("C2E2C2E"),
        },
    )

    assert helpers.dallclose(
        diagnostic_state.u.asnumpy(),
        icon_diagnostics_output_sp.zonal_wind().asnumpy(),
    )

    assert helpers.dallclose(
        diagnostic_state.v.asnumpy(),
        icon_diagnostics_output_sp.meridional_wind().asnumpy(),
        atol=1.0e-13,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment",
    [
        dt_utils.JABW_EXPERIMENT,
    ],
)
def test_diagnose_pressure(
    experiment,
    data_provider,
    icon_grid,
    backend,
):
    sp = data_provider.from_savepoint_jabw_final()
    icon_diagnostics_output_sp = data_provider.from_savepoint_jabw_diagnostic()
    prognostic_state_now = prognostic_state.PrognosticState(
        rho=sp.rho(),
        w=None,
        vn=sp.vn(),
        exner=sp.exner(),
        theta_v=sp.theta_v(),
    )
    diagnostic_state = diagnostics.DiagnosticState(
        temperature=sp.temperature(),
        virtual_temperature=sp.temperature(),
        pressure=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend
        ),
        pressure_ifc=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, extend={dims.KDim: 1}, backend=backend
        ),
        u=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend),
        v=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend),
    )

    cell_domain = h_grid.domain(dims.CellDim)
    surface_pressure.diagnose_surface_pressure.with_backend(backend)(
        prognostic_state_now.exner,
        diagnostic_state.virtual_temperature,
        data_provider.from_metrics_savepoint().ddqz_z_full(),
        diagnostic_state.pressure_ifc,
        phy_const.CPD_O_RD,
        phy_const.P0REF,
        phy_const.GRAV_O_RD,
        horizontal_start=0,
        horizontal_end=icon_grid.end_index(cell_domain(h_grid.Zone.END)),
        vertical_start=icon_grid.num_levels,
        vertical_end=icon_grid.num_levels + 1,
        offset_provider={"Koff": dims.KDim},
    )

    pressure.diagnose_pressure.with_backend(backend)(
        data_provider.from_metrics_savepoint().ddqz_z_full(),
        diagnostic_state.virtual_temperature,
        diagnostic_state.surface_pressure,
        diagnostic_state.pressure,
        diagnostic_state.pressure_ifc,
        phy_const.GRAV_O_RD,
        horizontal_start=0,
        horizontal_end=icon_grid.end_index(cell_domain(h_grid.Zone.END)),
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={},
    )
    assert helpers.dallclose(
        diagnostic_state.surface_pressure.asnumpy(),
        icon_diagnostics_output_sp.pressure_sfc().asnumpy(),
    )

    assert helpers.dallclose(
        diagnostic_state.pressure_ifc.asnumpy(),
        icon_diagnostics_output_sp.pressure_ifc().asnumpy(),
    )

    assert helpers.dallclose(
        diagnostic_state.pressure.asnumpy(),
        icon_diagnostics_output_sp.pressure().asnumpy(),
    )
