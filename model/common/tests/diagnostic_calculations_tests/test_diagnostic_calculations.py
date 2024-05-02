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

import os

import numpy as np
import pytest

from icon4py.model.common.constants import CPD_O_RD, GRAV_O_RD, P0REF
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
    "experiment, rank, debug",
    [
        (JABW_EXPERIMENT, 0, False),
    ],
)
def test_diagnostic_calculations_in_jabw(
    experiment,
    ranked_data_path,
    rank,
    data_provider,
    icon_grid,
    debug,
):
    sp = data_provider.from_savepoint_jabw_final()
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

    # TODO (Chia Rui): remember to uncomment this computation when the bug in gt4py is removed
    """
    diagnose_pressure(
        data_provider.from_metrics_savepoint().ddqz_z_full(),
        diagnostic_state.temperature,
        diagnostic_state.pressure_sfc,
        diagnostic_state.pressure,
        diagnostic_state.pressure_ifc,
        icon_grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
        icon_grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
        0,
        icon_grid.num_levels,
        offset_provider={"Koff": KDim},
    )
    """

    icon_diagnostics_output_sp = data_provider.from_savepoint_jabw_diagnostic()

    if debug:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = script_dir + "/"

        def printing(ref, predict, title: str):
            with open(base_dir + "analysis_" + title + ".dat", "w") as f:
                if len(ref.shape) == 1:
                    cell_size = ref.shape[0]
                    k_size = 0
                elif len(ref.shape) == 2:
                    cell_size = ref.shape[0]
                    k_size = ref.shape[1]
                else:
                    print("Dimension is not 1 or 2, do nothing in printing for ", title)
                    return
                print(title, cell_size, k_size)
                difference = np.abs(ref - predict)
                print(np.max(difference), np.min(difference))
                if k_size > 0:
                    for i in range(cell_size):
                        for k in range(k_size):
                            f.write("{0:7d} {1:7d}".format(i, k))
                            f.write(
                                " {0:.20e} {1:.20e} {2:.20e} ".format(
                                    difference[i, k], ref[i, k], predict[i, k]
                                )
                            )
                            f.write("\n")
                else:
                    for i in range(cell_size):
                        f.write("{0:7d}".format(i))
                        f.write(
                            " {0:.20e} {1:.20e} {2:.20e} ".format(difference[i], ref[i], predict[i])
                        )
                        f.write("\n")

        printing(
            data_provider.from_savepoint_jabw_final().rho().asnumpy(),
            prognostic_state_now.rho.asnumpy(),
            "rho",
        )

        printing(
            data_provider.from_savepoint_jabw_final().exner().asnumpy(),
            prognostic_state_now.exner.asnumpy(),
            "exner",
        )

        printing(
            data_provider.from_savepoint_jabw_final().theta_v().asnumpy(),
            prognostic_state_now.theta_v.asnumpy(),
            "theta_v",
        )

        printing(
            data_provider.from_savepoint_jabw_final().vn().asnumpy(),
            prognostic_state_now.vn.asnumpy(),
            "vn",
        )

        printing(
            data_provider.from_savepoint_jabw_final().w().asnumpy(),
            prognostic_state_now.w.asnumpy(),
            "w",
        )

        printing(
            data_provider.from_savepoint_jabw_final().pressure().asnumpy(),
            diagnostic_state.pressure.asnumpy(),
            "pressure",
        )

        printing(
            data_provider.from_savepoint_jabw_final().temperature().asnumpy(),
            diagnostic_state.temperature.asnumpy(),
            "temperature",
        )

        printing(
            data_provider.from_savepoint_jabw_init().pressure_sfc().asnumpy(),
            diagnostic_state.pressure_sfc.asnumpy(),
            "pressure_sfc",
        )

        printing(
            icon_diagnostics_output_sp.u().asnumpy(),
            diagnostic_state.u.asnumpy(),
            "u",
        )

        printing(
            icon_diagnostics_output_sp.v().asnumpy(),
            diagnostic_state.v.asnumpy(),
            "v",
        )

        printing(
            icon_diagnostics_output_sp.temperature().asnumpy(),
            diagnostic_state.temperature.asnumpy(),
            "temperature_diag",
        )

        printing(
            icon_diagnostics_output_sp.pressure_sfc().asnumpy(),
            diagnostic_state.pressure_sfc.asnumpy(),
            "pressure_sfc_diag",
        )

    assert dallclose(
        diagnostic_state.u.asnumpy(),
        icon_diagnostics_output_sp.zonal_Wind().asnumpy(),
    )

    assert dallclose(
        diagnostic_state.v.asnumpy(),
        icon_diagnostics_output_sp.meridional_Wind().asnumpy(),
        atol=1.0e-13,
    )

    assert dallclose(
        diagnostic_state.temperature.asnumpy(),
        icon_diagnostics_output_sp.temperature().asnumpy(),
    )

    assert dallclose(
        diagnostic_state.pressure_ifc.asnumpy()[:, -1],
        icon_diagnostics_output_sp.pressure_sfc().asnumpy(),
    )

    # TODO (Chia Rui): to compare pressure
