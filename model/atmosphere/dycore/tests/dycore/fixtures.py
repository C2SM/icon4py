# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from icon4py.model.testing.fixtures.datatest import (
    backend,
    damping_height,
    data_provider,
    decomposition_info,
    download_ser_data,
    experiment,
    flat_height,
    grid_savepoint,
    htop_moist_proc,
    icon_grid,
    interpolation_savepoint,
    istep_exit,
    istep_init,
    linit,
    lowest_layer_thickness,
    maximal_layer_thickness,
    metrics_savepoint,
    model_top_height,
    ndyn_substeps,
    processor_props,
    ranked_data_path,
    savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_exit,
    savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_init,
    savepoint_nonhydro_step_final,
    savepoint_dycore_30_to_38_init,
    savepoint_dycore_30_to_38_exit,
    savepoint_velocity_exit,
    savepoint_velocity_init,
    savepoint_vertically_implicit_dycore_solver_init,
    step_date_exit,
    step_date_init,
    stretch_factor,
    substep_exit,
    substep_init,
    top_height_limit_for_maximal_layer_thickness,
)
