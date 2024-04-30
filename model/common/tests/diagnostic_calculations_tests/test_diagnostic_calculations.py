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

from icon4py.model.atmosphere.dycore.state_utils.utils import _allocate
from icon4py.model.common.constants import CPD_O_RD, GRAV_O_RD, P0REF
from icon4py.model.common.diagnostic_calculations.stencils.diagnose_temperature import diagnose_temperature
from icon4py.model.common.diagnostic_calculations.stencils.diagnose_pressure import diagnose_pressure
from icon4py.model.common.diagnostic_calculations.stencils.diagnose_surface_pressure import diagnose_surface_pressure
from icon4py.model.common.diagnostic_calculations.stencils.init_exner_pr import init_exner_pr
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex
from icon4py.model.common.interpolation.stencils.edge_2_cell_vector_rbf_interpolation import (
    edge_2_cell_vector_rbf_interpolation,
)
from icon4py.model.common.test_utils.datatest_utils import JABW_EXPERIMENT
from icon4py.model.common.test_utils.helpers import dallclose
from icon4py.model.driver.initialization_utils import model_initialization_jabw


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, experiment_name, fname_prefix, rank, debug",
    [
        (JABW_EXPERIMENT, "jabw", "icon_pydycore", 0, False),
    ],
)
def test_diagnostic_calculations_in_jabw(
    experiment,
    datapath,
    experiment_name,
    fname_prefix,
    rank,
    data_provider,
    grid_savepoint,
    icon_grid,
    debug,
):
    edge_geometry = grid_savepoint.construct_edge_geometry()
    cell_geometry = grid_savepoint.construct_cell_geometry()

    (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        divdamp_fac_o2,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    ) = model_initialization_jabw(
        icon_grid, cell_geometry, edge_geometry, datapath, fname_prefix, rank
    )

    init_exner_pr(
        prognostic_state_now.exner,
        data_provider.from_metrics_savepoint().exner_ref_mc(),
        solve_nonhydro_diagnostic_state.exner_pr,
        icon_grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
        icon_grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
        0,
        icon_grid.num_levels,
        offset_provider={},
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
    ref_u = _allocate(CellDim, KDim, grid=icon_grid)
    ref_v = _allocate(CellDim, KDim, grid=icon_grid)
    edge_2_cell_vector_rbf_interpolation(
        prognostic_state_now.vn,
        rbv_vec_coeff_c1,
        rbv_vec_coeff_c2,
        ref_u,
        ref_v,
        grid_idx_cell_start_plus1,
        grid_idx_cell_end,
        0,
        icon_grid.num_levels,
        offset_provider={
            "C2E2C2E": icon_grid.get_offset_provider("C2E2C2E"),
        },
    )

    exner_nlev_minus2 = prognostic_state_now.exner[:, icon_grid.num_levels - 3]
    temperature_nlev = diagnostic_state.temperature[:, icon_grid.num_levels - 1]
    temperature_nlev_minus1 = diagnostic_state.temperature[:, icon_grid.num_levels - 2]
    temperature_nlev_minus2 = diagnostic_state.temperature[:, icon_grid.num_levels - 3]
    ddqz_z_full_nlev = data_provider.from_metrics_savepoint().ddqz_z_full()[
        :, icon_grid.num_levels - 1
    ]
    ddqz_z_full_nlev_minus1 = data_provider.from_metrics_savepoint().ddqz_z_full()[
        :, icon_grid.num_levels - 2
    ]
    ddqz_z_full_nlev_minus2 = data_provider.from_metrics_savepoint().ddqz_z_full()[
        :, icon_grid.num_levels - 3
    ]
    diagnose_surface_pressure(
        exner_nlev_minus2,
        temperature_nlev,
        temperature_nlev_minus1,
        temperature_nlev_minus2,
        ddqz_z_full_nlev,
        ddqz_z_full_nlev_minus1,
        ddqz_z_full_nlev_minus2,
        diagnostic_state.pressure_sfc,
        CPD_O_RD,
        P0REF,
        GRAV_O_RD,
        icon_grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
        icon_grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
        offset_provider={},
    )

    # TODO (Chia Rui): remember to uncomment this computation when the bug in gt4py is removed
    '''
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
        offset_provider={}
    )
    '''

    icon_diagnostics_output_sp = data_provider.from_savepoint_jabw_first_output()

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

        printing(
            icon_diagnostics_output_sp.exner_pr().asnumpy(),
            solve_nonhydro_diagnostic_state.exner_pr.asnumpy(),
            "exner_pr",
        )

    assert dallclose(
        diagnostic_state.u.asnumpy(),
        icon_diagnostics_output_sp.u().asnumpy(),
    )

    assert dallclose(
        diagnostic_state.v.asnumpy(),
        icon_diagnostics_output_sp.v().asnumpy(),
        atol=1.0e-13,
    )

    assert dallclose(
        diagnostic_state.temperature.asnumpy(),
        icon_diagnostics_output_sp.temperature().asnumpy(),
    )

    assert dallclose(
        diagnostic_state.pressure_sfc.asnumpy(),
        icon_diagnostics_output_sp.pressure_sfc().asnumpy(),
    )

    assert dallclose(
        ref_u.asnumpy(),
        icon_diagnostics_output_sp.u().asnumpy(),
    )

    assert dallclose(
        ref_v.asnumpy(),
        icon_diagnostics_output_sp.v().asnumpy(),
        atol=1.0e-13,
    )

    assert dallclose(
        solve_nonhydro_diagnostic_state.exner_pr.asnumpy(),
        icon_diagnostics_output_sp.exner_pr().asnumpy(),
        atol=3.0e-15,
    )

    # TODO (Chia Rui): to compare pressure

