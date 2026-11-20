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
    backend: model_backends.BackendLike,
    exchange: decomposition_defs.ExchangeRuntime,
) -> data_alloc.NDArray:
    """Read the topography field from a serialised ICON snapshot.

    Always reads from the single-rank (global) serialbox archive
    ``icon_pydycore_rank0`` and selects local rows using the rank's global
    cell indices. The caller is responsible for pointing ``config.data_path``
    at the single-rank ser_data directory (``mpitask1_*``), which contains
    the full global topography field.

    ``ser.read`` always returns a numpy array; it is moved to the device via
    ``array_ns.asarray`` before indexing so that the selection stays on the
    device on GPU backends.
    """
    allocator = model_backends.get_allocator(backend)
    array_ns = data_alloc.import_array_ns(allocator)

    data_path = config.data_path

    # TODO(icon4py): the archive name is hardcoded to the single-rank (global) archive
    # because the per-rank archives follow ICON's domain decomposition, not icon4py's,
    # so each rank selects its rows by global index instead. Make this explicit
    # (e.g. in the config) rather than hardcoding 'rank0'.
    ser = serialbox.Serializer(serialbox.OpenModeKind.Read, str(data_path), "icon_pydycore_rank0")
    sp = ser.savepoint["smooth-topo-savepoint"].as_savepoint()
    log.debug("Reading topography from %s / icon_pydycore_rank0", data_path)

    local_indices = grid_manager.decomposition_info.global_index(
        dims.CellDim, decomposition_defs.DecompositionInfo.EntryType.ALL
    )
    global_topo = array_ns.asarray(np.squeeze(ser.read("topography", sp).astype(float)))
    return global_topo[local_indices]
