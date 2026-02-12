# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging

import gt4py.next as gtx
import pytest

from icon4py.model.common.decomposition import definitions as decomposition, mpi_decomposition
from icon4py.model.common.grid import grid_refinement, horizontal as h_grid
from icon4py.model.testing import definitions, serialbox
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    grid_savepoint,
    processor_props,
)

from .. import utils


if mpi_decomposition.mpi4py is None:
    pytest.skip("Skipping parallel tests on single node installation", allow_module_level=True)

_log = logging.getLogger(__name__)


@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_compute_domain_bounds(
    dim: gtx.Dimension,
    experiment: definitions.Experiment,
    grid_savepoint: serialbox.IconGridSavepoint,
    processor_props: decomposition.ProcessProperties,
) -> None:
    if processor_props.is_single_rank() and experiment == definitions.Experiments.EXCLAIM_APE:
        pytest.xfail(
            "end index data for single node APE are all 0 - re- serialization should fix that (patch%cells%end_index vs patch%cells%end_idx)"
        )

    ref_grid = grid_savepoint.construct_icon_grid(backend=None, keep_skip_values=True)
    decomposition_info = grid_savepoint.construct_decomposition_info()
    refin_ctrl = {dim: grid_savepoint.refin_ctrl(dim) for dim in utils.main_horizontal_dims()}
    start_indices, end_indices = grid_refinement.compute_domain_bounds(
        dim, refin_ctrl, decomposition_info
    )
    for domain in h_grid.get_domains_for_dim(dim):
        ref_start_index = ref_grid.start_index(domain)
        ref_end_index = ref_grid.end_index(domain)
        computed_start = start_indices[domain]
        computed_end = end_indices[domain]
        _log.info(
            f"rank = {processor_props.rank}/{processor_props.comm_size}: domain={domain} : start = {computed_start} end = {computed_end} "
        )
        assert (
            computed_start == ref_start_index
        ), f"rank={processor_props.rank}/{processor_props.comm_size} - experiment = {experiment.name}: start_index for {domain} does not match: is {computed_start}, expected {ref_start_index}"
        assert (
            computed_end == ref_end_index
        ), f"rank={processor_props.rank}/{processor_props.comm_size} - experiment = {experiment.name}: end_index for {domain} does not match: is {computed_end}, expected {ref_end_index}"
