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
from icon4py.model.common.test_utils import datatest_utils as dt_utils, helpers as helpers


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
        temperature=helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        virtual_temperature=helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        pressure=helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        pressure_ifc=helpers.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, extend={dims.KDim: 1}
        ),
        u=helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        v=helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
    )
    tracer_state = tracers.TracerState(
        qv=helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        qc=helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        qr=helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        qi=helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        qs=helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        qg=helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
    )

    temperature.diagnose_virtual_temperature_and_temperature(
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
        horizontal_end=icon_grid.get_end_index(
            dims.CellDim, h_grid.HorizontalMarkerIndex.end(dims.CellDim)
        ),
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
        temperature=helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        pressure=helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        pressure_ifc=helpers.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, extend={dims.KDim: 1}
        ),
        u=helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        v=helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        virtual_temperature=helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
    )

    rbv_vec_coeff_c1 = data_provider.from_interpolation_savepoint().rbf_vec_coeff_c1()
    rbv_vec_coeff_c2 = data_provider.from_interpolation_savepoint().rbf_vec_coeff_c2()
    grid_idx_cell_start_plus1 = icon_grid.get_end_index(
        dims.CellDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.CellDim) + 1
    )
    grid_idx_cell_end = icon_grid.get_end_index(
        dims.CellDim, h_grid.HorizontalMarkerIndex.end(dims.CellDim)
    )
    rbf.edge_2_cell_vector_rbf_interpolation(
        prognostic_state_now.vn,
        rbv_vec_coeff_c1,
        rbv_vec_coeff_c2,
        diagnostic_state.u,
        diagnostic_state.v,
        grid_idx_cell_start_plus1,
        grid_idx_cell_end,
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
        pressure=helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        pressure_ifc=helpers.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, extend={dims.KDim: 1}
        ),
        u=helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        v=helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        virtual_temperature=sp.temperature(),
    )

    surface_pressure.diagnose_surface_pressure(
        prognostic_state_now.exner,
        diagnostic_state.virtual_temperature,
        data_provider.from_metrics_savepoint().ddqz_z_full(),
        diagnostic_state.pressure_ifc,
        phy_const.CPD_O_RD,
        phy_const.P0REF,
        phy_const.GRAV_O_RD,
        horizontal_start=0,
        horizontal_end=icon_grid.get_end_index(
            dims.CellDim, h_grid.HorizontalMarkerIndex.end(dims.CellDim)
        ),
        vertical_start=icon_grid.num_levels,
        vertical_end=icon_grid.num_levels + 1,
        offset_provider={"Koff": dims.KDim},
    )

    pressure.diagnose_pressure(
        data_provider.from_metrics_savepoint().ddqz_z_full(),
        diagnostic_state.virtual_temperature,
        diagnostic_state.pressure_sfc,
        diagnostic_state.pressure,
        diagnostic_state.pressure_ifc,
        phy_const.GRAV_O_RD,
        horizontal_start=0,
        horizontal_end=icon_grid.get_end_index(
            dims.CellDim, h_grid.HorizontalMarkerIndex.end(dims.CellDim)
        ),
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={},
    )

    assert helpers.dallclose(
        diagnostic_state.pressure_sfc.asnumpy(),
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
