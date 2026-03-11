# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import logging

import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions


log = logging.getLogger(__file__)


def check_comm_size(
    props: definitions.ProcessProperties, sizes: tuple[int, ...] = (1, 2, 4)
) -> None:
    if props.comm_size not in sizes:
        pytest.xfail(f"wrong comm size: {props.comm_size}: test only works for comm-sizes: {sizes}")


def log_process_properties(props: definitions.ProcessProperties) -> None:
    log.info(f"rank={props.rank}/{props.comm_size}")


def log_local_field_size(decomposition_info: definitions.DecompositionInfo) -> None:
    log.info(
        f"local grid size: cells={decomposition_info.global_index(dims.CellDim).size}, "
        f"edges={decomposition_info.global_index(dims.EdgeDim).size}, "
        f"vertices={decomposition_info.global_index(dims.VertexDim).size}"
    )
