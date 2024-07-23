# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later


import pytest

from icon4py.model.common.constants import CPD_O_RD, GRAV_O_RD, P0REF
from icon4py.model.common.diagnostic_calculations.stencils.diagnose_pressure import (
    diagnose_pressure,
)
from icon4py.model.common.diagnostic_calculations.stencils.diagnose_surface_pressure import (
    diagnose_surface_pressure,
)
from icon4py.model.common.diagnostic_calculations.stencils.diagnose_temperature import (
    diagnose_temperature,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex
from icon4py.model.common.interpolation.stencils.edge_2_cell_vector_rbf_interpolation import (
    edge_2_cell_vector_rbf_interpolation,
)
from icon4py.model.common.states.diagnostic_state import DiagnosticState
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils.datatest_utils import JABW_EXPERIMENT
from icon4py.model.common.test_utils.helpers import dallclose, zero_field


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment",
    [
        JABW_EXPERIMENT,
    ],
)
def test_diagnose_temperature(
    experiment,
    data_provider,
    icon_grid,
):
    sp = data_provider.from_savepoint_jabw_final()
    icon_diagnostics_output_sp = data_provider.from_savepoint_jabw_diagnostic()
    prognostic_state_now = PrognosticState(
        rho=sp.rho(),
        w=None,
        vn=sp.vn(),
        exner=sp.exner(),
        theta_v=sp.theta_v(),
    )
    diagnostic_state = DiagnosticState(
        temperature=zero_field(icon_grid, CellDim, KDim, dtype=float),
        pressure=zero_field(icon_grid, CellDim, KDim, dtype=float),
        pressure_ifc=zero_field(icon_grid, CellDim, KDim, dtype=float, extend={KDim: 1}),
        u=zero_field(icon_grid, CellDim, KDim, dtype=float),
        v=zero_field(icon_grid, CellDim, KDim, dtype=float),
    )

    diagnose_temperature(
        prognostic_state_now.theta_v,
        prognostic_state_now.exner,
        diagnostic_state.temperature,
        icon_grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
        icon_grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
        0,
        icon_grid.num_levels,
        offset_provider={},
    )

    assert dallclose(
        diagnostic_state.temperature.asnumpy(),
        icon_diagnostics_output_sp.temperature().asnumpy(),
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment",
    [
        JABW_EXPERIMENT,
    ],
)
def test_diagnose_meridional_and_zonal_winds(
    experiment,
    data_provider,
    icon_grid,
):
    sp = data_provider.from_savepoint_jabw_final()
    icon_diagnostics_output_sp = data_provider.from_savepoint_jabw_diagnostic()
    prognostic_state_now = PrognosticState(
        rho=sp.rho(),
        w=None,
        vn=sp.vn(),
        exner=sp.exner(),
        theta_v=sp.theta_v(),
    )
    diagnostic_state = DiagnosticState(
        temperature=zero_field(icon_grid, CellDim, KDim, dtype=float),
        pressure=zero_field(icon_grid, CellDim, KDim, dtype=float),
        pressure_ifc=zero_field(icon_grid, CellDim, KDim, dtype=float, extend={KDim: 1}),
        u=zero_field(icon_grid, CellDim, KDim, dtype=float),
        v=zero_field(icon_grid, CellDim, KDim, dtype=float),
    )

    rbv_vec_coeff_c1 = data_provider.from_interpolation_savepoint().rbf_vec_coeff_c1()
    rbv_vec_coeff_c2 = data_provider.from_interpolation_savepoint().rbf_vec_coeff_c2()
    grid_idx_cell_start_plus1 = icon_grid.get_end_index(
        CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 1
    )
    grid_idx_cell_end = icon_grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim))
    edge_2_cell_vector_rbf_interpolation(
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

    assert dallclose(
        diagnostic_state.u.asnumpy(),
        icon_diagnostics_output_sp.zonal_wind().asnumpy(),
    )

    assert dallclose(
        diagnostic_state.v.asnumpy(),
        icon_diagnostics_output_sp.meridional_wind().asnumpy(),
        atol=1.0e-13,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment",
    [
        JABW_EXPERIMENT,
    ],
)
def test_diagnose_pressure(
    experiment,
    data_provider,
    icon_grid,
):
    sp = data_provider.from_savepoint_jabw_final()
    icon_diagnostics_output_sp = data_provider.from_savepoint_jabw_diagnostic()
    prognostic_state_now = PrognosticState(
        rho=sp.rho(),
        w=None,
        vn=sp.vn(),
        exner=sp.exner(),
        theta_v=sp.theta_v(),
    )
    diagnostic_state = DiagnosticState(
        temperature=sp.temperature(),
        pressure=zero_field(icon_grid, CellDim, KDim, dtype=float),
        pressure_ifc=zero_field(icon_grid, CellDim, KDim, dtype=float, extend={KDim: 1}),
        u=zero_field(icon_grid, CellDim, KDim, dtype=float),
        v=zero_field(icon_grid, CellDim, KDim, dtype=float),
    )

    diagnose_surface_pressure(
        prognostic_state_now.exner,
        diagnostic_state.temperature,
        data_provider.from_metrics_savepoint().ddqz_z_full(),
        diagnostic_state.pressure_ifc,
        CPD_O_RD,
        P0REF,
        GRAV_O_RD,
        horizontal_start=icon_grid.get_start_index(
            CellDim, HorizontalMarkerIndex.interior(CellDim)
        ),
        horizontal_end=icon_grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
        vertical_start=icon_grid.num_levels,
        vertical_end=icon_grid.num_levels + 1,
        offset_provider={"Koff": KDim},
    )

    diagnose_pressure(
        data_provider.from_metrics_savepoint().ddqz_z_full(),
        diagnostic_state.temperature,
        diagnostic_state.pressure_sfc,
        diagnostic_state.pressure,
        diagnostic_state.pressure_ifc,
        GRAV_O_RD,
        icon_grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
        icon_grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
        0,
        icon_grid.num_levels,
        offset_provider={},
    )

    assert dallclose(
        diagnostic_state.pressure_sfc.asnumpy(),
        icon_diagnostics_output_sp.pressure_sfc().asnumpy(),
    )

    assert dallclose(
        diagnostic_state.pressure_ifc.asnumpy(),
        icon_diagnostics_output_sp.pressure_ifc().asnumpy(),
    )

    assert dallclose(
        diagnostic_state.pressure.asnumpy(),
        icon_diagnostics_output_sp.pressure().asnumpy(),
    )
