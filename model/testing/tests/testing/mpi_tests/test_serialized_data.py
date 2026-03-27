# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs, mpi_decomposition
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    parallel_helpers,
    serialbox,
)
from icon4py.model.testing.fixtures import (
    backend,
    backend_like,
    data_provider,
    decomposition_info,
    download_ser_data,
    experiment,
    istep_exit,
    linit,
    processor_props,
    savepoint_diffusion_exit,
    savepoint_nonhydro_exit,
    step_date_exit,
    substep_exit,
)
from icon4py.model.testing.fixtures.datatest import _download_ser_data


if mpi_decomposition.mpi4py is None:
    pytest.skip("Skipping parallel tests on single node installation", allow_module_level=True)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "experiment, istep_exit, substep_exit, step_date_exit, linit",
    [
        (
            test_defs.Experiments.JW,
            2,
            5,
            "2008-09-01T00:05:00.000",
            False,
        ),
    ],
)
def test_single_vs_multirank_serialized_data(
    experiment: test_defs.Experiment,
    istep_exit: int,
    substep_exit: int,
    step_date_exit: str,
    linit: bool,
    processor_props: decomp_defs.ProcessProperties,
    decomposition_info: decomp_defs.DecompositionInfo,
    savepoint_diffusion_exit: serialbox.IconDiffusionExitSavepoint,
    savepoint_nonhydro_exit: serialbox.IconNonHydroExitSavepoint,
    backend_like: model_backends.BackendLike,
) -> None:
    parallel_helpers.check_comm_size(processor_props)

    # All ranks independently load the single-rank reference data (comm_size=1, rank=0).
    # File locking in download_test_data makes concurrent downloads safe.
    single_rank_processor_props = decomp_defs.get_processor_properties(
        decomp_defs.get_runtype(with_mpi=False)
    )
    _download_ser_data(experiment, single_rank_processor_props)
    single_rank_datapath = dt_utils.get_datapath_for_experiment(
        experiment, single_rank_processor_props
    )
    single_rank_data_provider = dt_utils.create_icon_serial_data_provider(
        single_rank_datapath, rank=0, backend=None
    )

    sp_diffusion_exit_single = single_rank_data_provider.from_savepoint_diffusion_exit(
        linit=linit, date=step_date_exit
    )
    sp_nonhydro_exit_single = single_rank_data_provider.from_savepoint_nonhydro_exit(
        istep=istep_exit, date=step_date_exit, substep=substep_exit
    )

    for dim, global_field, local_field in [
        (dims.EdgeDim, sp_diffusion_exit_single.vn(), savepoint_diffusion_exit.vn()),
        (dims.CellDim, sp_diffusion_exit_single.w(), savepoint_diffusion_exit.w()),
        (dims.CellDim, sp_diffusion_exit_single.exner(), savepoint_diffusion_exit.exner()),
        (dims.CellDim, sp_diffusion_exit_single.theta_v(), savepoint_diffusion_exit.theta_v()),
        (dims.CellDim, sp_nonhydro_exit_single.rho_new(), savepoint_nonhydro_exit.rho_new()),
    ]:
        parallel_helpers.check_local_global_field(
            decomposition_info=decomposition_info,
            processor_props=processor_props,
            dim=dim,
            global_reference_field=global_field.asnumpy(),
            local_field=local_field.asnumpy(),
            check_halos=True,
            atol=0.0,
            non_blocking = True,
        )
