# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from collections.abc import Iterable

import pytest

from icon4py.model.common.decomposition import definitions
from icon4py.model.common.decomposition.mpi_decomposition import get_multinode_properties


log = logging.getLogger(__file__)


def check_comm_size(
    props: definitions.ProcessProperties, sizes: tuple[int, ...] = (1, 2, 4)
) -> None:
    if props.comm_size not in sizes:
        pytest.xfail(f"wrong comm size: {props.comm_size}: test only works for comm-sizes: {sizes}")


@pytest.fixture(scope="session")
def processor_props(request: pytest.FixtureRequest) -> Iterable[definitions.ProcessProperties]:
    runtype = definitions.get_runtype(with_mpi=True)
    yield get_multinode_properties(runtype)


def log_process_properties(
    props: definitions.ProcessProperties, level: int = logging.DEBUG
) -> None:
    log.info(f"rank={props.rank}/{props.comm_size}")


def log_local_field_size(
    decomposition_info: definitions.DecompositionInfo, level: int = logging.DEBUG
) -> None:
    log.info(
        f"local grid size: cells={decomposition_info.num_cells}, edges={decomposition_info.num_edges}, vertices={decomposition_info.num_vertices}"
    )
