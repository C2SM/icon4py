# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import serialbox  # type: ignore[import-untyped]

from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import grid_manager as gm


log = logging.getLogger(__name__)


@dataclasses.dataclass
class FromFileConfig:
    """Parameters for the file-based topography."""

    #: Path to the serialised data directory (typically ``<experiment>/ser_data``).
    data_path: pathlib.Path


def read_from_file(
    *,
    config: FromFileConfig,
    grid_manager: gm.GridManager,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
) -> data_alloc.NDArray:
    """Read the topography field from a serialised ICON snapshot.

    Always reads from the single-rank (global) serialbox archive
    ``icon_pydycore_rank0`` and selects local rows using the rank's global
    cell indices.  The caller is responsible for pointing ``config.data_path``
    at the single-rank ser_data directory (``mpitask1_*``), which contains
    the full global topography field.

    ``local_indices`` is converted to numpy before indexing because on GPU
    backends ``decomposition_info.global_index()`` returns a CuPy array,
    which cannot index a numpy serialbox buffer.
    """
    allocator = model_backends.get_allocator(backend)
    array_ns = data_alloc.import_array_ns(allocator)

    data_path = config.data_path

    ser = serialbox.Serializer(serialbox.OpenModeKind.Read, str(data_path), "icon_pydycore_rank0")
    sp = ser.savepoint["smooth-topo-savepoint"].as_savepoint()
    log.debug("Reading topography from %s / icon_pydycore_rank0", data_path)

    # global_index() may return CuPy on GPU backends; np.asarray converts safely.
    local_indices = data_alloc.as_numpy(
        grid_manager.decomposition_info.global_index(
            dims.CellDim, decomposition_defs.DecompositionInfo.EntryType.ALL
        )
    )
    global_topo = np.squeeze(ser.read("topography", sp).astype(float))
    return array_ns.asarray(global_topo[local_indices])
