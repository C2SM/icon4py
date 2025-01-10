# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from icon4py.model.testing.datatest_fixtures import (
    damping_height,
    data_provider,
    download_ser_data,
    experiment,
    flat_height,
    grid_savepoint,
    htop_moist_proc,
    icon_grid,
    interpolation_savepoint,
    lowest_layer_thickness,
    maximal_layer_thickness,
    metrics_savepoint,
    model_top_height,
    ndyn_substeps,
    processor_props,
    ranked_data_path,
    stretch_factor,
    top_height_limit_for_maximal_layer_thickness,
)
from icon4py.model.testing.helpers import backend, grid


# Make sure custom icon4py pytest hooks are loaded
try:
    import sys

    _ = sys.modules["icon4py.model.testing.pytest_config"]
except KeyError:
    from icon4py.model.testing.pytest_config import *  # noqa: F403 [undefined-local-with-import-star]


__all__ = [
    # imported fixtures:
    "lowest_layer_thickness",
    "maximal_layer_thickness",
    "model_top_height",
    "stretch_factor",
    "top_height_limit_for_maximal_layer_thickness",
    "flat_height",
    "htop_moist_proc",
    "damping_height",
    "data_provider",
    "download_ser_data",
    "experiment",
    "grid_savepoint",
    "icon_grid",
    "interpolation_savepoint",
    "metrics_savepoint",
    "ndyn_substeps",
    "processor_props",
    "ranked_data_path",
    "backend",
    "grid",
]
