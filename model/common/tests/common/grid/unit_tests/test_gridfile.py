# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import gridfile
from icon4py.model.testing import definitions, grid_utils as gridtest_utils
from icon4py.model.testing.fixtures import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    grid_savepoint,
    processor_props,
)


if TYPE_CHECKING:
    from icon4py.model.testing import serialbox


@pytest.mark.with_netcdf
def test_grid_file_dimension() -> None:
    grid_descriptor = definitions.Grids.R02B04_GLOBAL
    global_grid_file = str(gridtest_utils.resolve_full_grid_file_name(grid_descriptor))
    parser = gridfile.GridFile(global_grid_file, transformation=gridfile.NoTransformation())
    try:
        parser.open()
        assert (
            parser.dimension(gridfile.DynamicDimension.CELL_NAME) == grid_descriptor.sizes["cell"]
        )
        assert (
            parser.dimension(gridfile.DynamicDimension.VERTEX_NAME)
            == grid_descriptor.sizes["vertex"]
        )
        assert (
            parser.dimension(gridfile.DynamicDimension.EDGE_NAME) == grid_descriptor.sizes["edge"]
        )
    except Exception:
        pytest.fail()
    finally:
        parser.close()


@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_file_vertex_cell_edge_dimensions(
    experiment: definitions.Experiment, grid_savepoint: serialbox.IconGridSavepoint
) -> None:
    file = gridtest_utils.resolve_full_grid_file_name(experiment.grid)
    parser = gridfile.GridFile(str(file), gridfile.ToZeroBasedIndexTransformation())
    try:
        parser.open()
        assert parser.dimension(gridfile.DynamicDimension.CELL_NAME) == grid_savepoint.num(
            dims.CellDim
        )
        assert parser.dimension(gridfile.DynamicDimension.VERTEX_NAME) == grid_savepoint.num(
            dims.VertexDim
        )
        assert parser.dimension(gridfile.DynamicDimension.EDGE_NAME) == grid_savepoint.num(
            dims.EdgeDim
        )
    except Exception as error:
        pytest.fail(f"reading of dimension from netcdf failed: {error}")
    finally:
        parser.close()


@pytest.mark.parametrize("apply_transformation", (True, False))
def test_int_variable(experiment: definitions.Experiment, apply_transformation: bool) -> None:
    file = gridtest_utils.resolve_full_grid_file_name(experiment.grid)
    with gridfile.GridFile(str(file), gridfile.ToZeroBasedIndexTransformation()) as parser:
        edge_dim = parser.dimension(gridfile.DynamicDimension.EDGE_NAME)
        # use a test field that does not contain Pentagons
        test_field = parser.int_variable(
            gridfile.ConnectivityName.C2E, apply_transformation=apply_transformation
        )
        min_value = 0 if apply_transformation else 1
        max_value = edge_dim - 1 if apply_transformation else edge_dim
        assert min_value == np.min(test_field)
        assert max_value == np.max(test_field)


_index_selection: Iterable[list[int]] = [
    [0, 1, 2, 3, 4, 5],
    [],
    [0, 2, 4, 6, 7, 8, 24, 57],
    [1, 2, 12, 13, 23, 24, 513],
]


@pytest.mark.parametrize(
    "selection",
    _index_selection,
)
def test_index_read_for_1d_fields(experiment: definitions.Experiment, selection: list[int]) -> None:
    file = gridtest_utils.resolve_full_grid_file_name(experiment.grid)
    with gridfile.GridFile(str(file), gridfile.ToZeroBasedIndexTransformation()) as parser:
        indices_to_read = np.asarray(selection) if len(selection) > 0 else None
        full_field = parser.variable(gridfile.CoordinateName.CELL_LATITUDE)
        selective_field = parser.variable(
            gridfile.CoordinateName.CELL_LATITUDE, indices=indices_to_read
        )
        assert np.allclose(full_field[indices_to_read], selective_field)


@pytest.mark.parametrize(
    "selection",
    _index_selection,
)
@pytest.mark.parametrize(
    "field",
    (
        gridfile.ConnectivityName.V2E,
        gridfile.ConnectivityName.V2C,
        gridfile.ConnectivityName.E2V,
    ),
)
@pytest.mark.parametrize("apply_offset", (True, False))
def test_index_read_for_2d_connectivity(
    experiment: definitions.Experiment,
    selection: list[int],
    field: gridfile.FieldName,
    apply_offset: bool,
) -> None:
    file = gridtest_utils.resolve_full_grid_file_name(experiment.grid)
    with gridfile.GridFile(str(file), gridfile.ToZeroBasedIndexTransformation()) as parser:
        indices_to_read = np.asarray(selection) if len(selection) > 0 else None
        full_field = parser.int_variable(field, transpose=True, apply_transformation=apply_offset)
        selective_field = parser.int_variable(
            field,
            indices=indices_to_read,
            transpose=True,
            apply_transformation=apply_offset,
        )
        assert np.allclose(full_field[indices_to_read], selective_field)
