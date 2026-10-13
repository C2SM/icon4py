# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from icon4py.model.common.config import config_io
from icon4py.model.common.topography import from_file as from_file_topo
from icon4py.model.common.topography.analytical import (
    flat_topography as flat_topo,
    gaussian_hill as gausshill_topo,
    jablonowski_williamson as jw_topo,
)
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.decomposition import definitions as decomposition_defs
    from icon4py.model.common.grid import grid_manager as gm

log = logging.getLogger(__name__)


type TOPO_CONFIG = (
    flat_topo.FlatTopographyConfig
    | jw_topo.JablonowskiWilliamsonConfig
    | gausshill_topo.GaussianHillConfig
    | from_file_topo.FromFileConfig
)

config_io.register_config_union(
    TOPO_CONFIG.__value__,
    {
        "flat": flat_topo.FlatTopographyConfig,
        "jablonowski_williamson": jw_topo.JablonowskiWilliamsonConfig,
        "gaussian_hill": gausshill_topo.GaussianHillConfig,
        "from_file": from_file_topo.FromFileConfig,
    },
)


def create(
    *,
    config: TOPO_CONFIG,
    grid_manager: gm.GridManager,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
) -> data_alloc.NDArray:
    """Create topography array by dispatching on the type of ``config.config``."""
    match config:
        case flat_topo.FlatTopographyConfig():
            return flat_topo.flat_topography(grid_manager=grid_manager)
        case jw_topo.JablonowskiWilliamsonConfig():
            return jw_topo.jablonowski_williamson(config=config, grid_manager=grid_manager)
        case gausshill_topo.GaussianHillConfig():
            return gausshill_topo.gaussian_hill(config=config, grid_manager=grid_manager)
        case from_file_topo.FromFileConfig():
            return from_file_topo.read_from_file(
                config=config,
                grid_manager=grid_manager,
                backend=backend,
                exchange=exchange,
            )
        case _:
            raise TypeError(f"Unknown topography config type: {type(config.config)!r}")
