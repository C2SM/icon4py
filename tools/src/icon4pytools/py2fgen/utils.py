# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import importlib
import os
from pathlib import Path

from gt4py.next.common import Dimension
from gt4py.next.type_system.type_specifications import FieldType, ScalarKind, ScalarType, TypeSpec


def parse_type_spec(type_spec: TypeSpec) -> tuple[list[Dimension], ScalarKind]:
    if isinstance(type_spec, ScalarType):
        return [], type_spec.kind
    elif isinstance(type_spec, FieldType):
        return type_spec.dims, type_spec.dtype.kind
    else:
        raise ValueError(f"Unsupported type specification: {type_spec}")


def flatten_and_get_unique_elts(list_of_lists: list[list[str]]) -> list[str]:
    return sorted(set(item for sublist in list_of_lists for item in sublist))


def get_local_test_grid(grid_folder: str):
    test_folder = "testdata"
    module_spec = importlib.util.find_spec("icon4pytools")

    if module_spec and module_spec.origin:
        # following namespace package conventions the root is three levels down
        repo_root = Path(module_spec.origin).parents[3]
        return os.path.join(repo_root, test_folder, "grids", grid_folder)
    else:
        raise FileNotFoundError(
            "The `icon4pytools` package could not be found. Ensure the package is installed "
            "and accessible. Alternatively, set the 'ICON_GRID_LOC' environment variable "
            "explicitly to specify the location."
        )


def get_icon_grid_loc(grid_folder: str):
    env_path = os.environ.get("ICON_GRID_LOC")
    if env_path is not None:
        return os.path.join(env_path, grid_folder)
    else:
        return get_local_test_grid(grid_folder)


def get_grid_filename():
    env_path = os.environ.get("ICON_GRID_NAME")
    if env_path is not None:
        return env_path
    return "grid.nc"
