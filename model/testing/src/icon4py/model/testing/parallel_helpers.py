# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import logging
import operator

import numpy as np
import pytest
from gt4py import next as gtx

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions, definitions as decomp_defs


log = logging.getLogger(__file__)
_log = log


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


def gather_field(field: np.ndarray, props: decomp_defs.ProcessProperties) -> tuple:
    constant_dims = tuple(field.shape[1:])
    _log.info(f"gather_field on rank={props.rank} - gathering field of local shape {field.shape}")
    constant_length = functools.reduce(operator.mul, constant_dims, 1)
    local_sizes = np.array(props.comm.gather(field.size, root=0))
    if props.rank == 0:
        recv_buffer = np.empty(np.sum(local_sizes), dtype=field.dtype)
        _log.info(
            f"gather_field on rank = {props.rank} - setup receive buffer with size {sum(local_sizes)} on rank 0"
        )
    else:
        recv_buffer = None

    props.comm.Gatherv(sendbuf=field, recvbuf=(recv_buffer, local_sizes), root=0)
    if props.rank == 0:
        local_first_dim = tuple(sz // constant_length for sz in local_sizes)
        _log.info(
            f" gather_field on rank = 0: computed local dims {local_first_dim} - constant dims {constant_dims}"
        )
        gathered_field = recv_buffer.reshape((-1, *constant_dims))  # type: ignore [union-attr]
    else:
        gathered_field = None
        local_first_dim = field.shape
    return local_first_dim, gathered_field


def check_local_global_field(
    decomposition_info: decomp_defs.DecompositionInfo,
    processor_props: decomp_defs.ProcessProperties,  # F811 # fixture
    dim: gtx.Dimension,
    global_reference_field: np.ndarray,
    local_field: np.ndarray,
    check_halos: bool,
    atol: float,
) -> None:
    if dim == dims.KDim:
        np.testing.assert_allclose(global_reference_field, local_field)
        return

    _log.info(
        f" rank= {processor_props.rank}/{processor_props.comm_size}----exchanging field of main dim {dim}"
    )
    assert (
        local_field.shape[0]
        == decomposition_info.global_index(dim, decomp_defs.DecompositionInfo.EntryType.ALL).shape[
            0
        ]
    )

    # Compare halo against global reference field
    if check_halos:
        print("checking halos")
        np.testing.assert_allclose(
            global_reference_field[
                decomposition_info.global_index(dim, decomp_defs.DecompositionInfo.EntryType.HALO)
            ],
            local_field[
                decomposition_info.local_index(dim, decomp_defs.DecompositionInfo.EntryType.HALO)
            ],
            atol=atol,
            verbose=True,
        )

    # Compare owned local field, excluding halos, against global reference
    # field, by gathering owned entries to the first rank. This ensures that in
    # total we have the full global field distributed on all ranks.
    owned_entries = local_field[
        decomposition_info.local_index(dim, decomp_defs.DecompositionInfo.EntryType.OWNED)
    ]
    gathered_sizes, gathered_field = gather_field(owned_entries, processor_props)

    global_index_sizes, gathered_global_indices = gather_field(
        decomposition_info.global_index(dim, decomp_defs.DecompositionInfo.EntryType.OWNED),
        processor_props,
    )

    if processor_props.rank == 0:
        _log.info(f"rank = {processor_props.rank}: asserting gathered fields: ")

        assert np.all(
            gathered_sizes == global_index_sizes
        ), f"gathered field sizes do not match:  {dim} {gathered_sizes} - {global_index_sizes}"
        _log.info(
            f"rank = {processor_props.rank}: Checking field size on dim ={dim}: --- gathered sizes {gathered_sizes} = {sum(gathered_sizes)}"
        )
        _log.info(
            f"rank = {processor_props.rank}:                      --- gathered field has size {gathered_sizes}"
        )
        sorted_ = np.zeros(global_reference_field.shape, dtype=gtx.float64)  # type: ignore [attr-defined]
        sorted_[gathered_global_indices] = gathered_field
        _log.info(
            f" rank = {processor_props.rank}: SHAPES: global reference field {global_reference_field.shape}, gathered = {gathered_field.shape}"
        )

        print("checking interior")
        np.testing.assert_allclose(sorted_, global_reference_field, atol=1e-9, verbose=True)
