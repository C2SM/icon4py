# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition, mpi_decomposition
from icon4py.model.common.grid import horizontal as h_grid, vertical as v_grid
from icon4py.model.common.math import helpers as math_helpers
from icon4py.model.common.states import factory
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import parallel_helpers

from ..fixtures import (
    backend,
    data_provider,
    decomposition_info,
    download_ser_data,
    grid_savepoint,
    processor_props,
)
from ..unit_tests.test_factory import SimpleFieldSource


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.testing import serialbox as sb


if mpi_decomposition.mpi4py is None:
    pytest.skip("Skipping parallel tests on single node installation", allow_module_level=True)


cell_domain = h_grid.domain(dims.CellDim)
k_domain = v_grid.domain(dims.KDim)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_program_provider_exchange(
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    grid_savepoint: sb.IconGridSavepoint,
    backend: gtx_typing.Backend | None,
) -> None:
    parallel_helpers.check_comm_size(processor_props, sizes=(2, 4))
    exchange = decomposition.create_exchange(processor_props, decomposition_info)
    grid = grid_savepoint.construct_icon_grid(backend=backend)

    number = processor_props.rank + 10
    input_field = data_alloc.constant_field(
        grid, number, dims.CellDim, dims.KDim, allocator=backend
    )
    source = SimpleFieldSource(
        data_={"in": (input_field, {"standard_name": "in", "units": ""})},
        backend=backend,
        grid=grid,
    )
    source._exchange = exchange
    provider = factory.ProgramFieldProvider(
        func=math_helpers.average_two_vertical_levels_downwards_on_cells,
        domain={
            # Use full local horizontal range so the field is computed before implicit exchange.
            dims.CellDim: (cell_domain(h_grid.Zone.LOCAL), cell_domain(h_grid.Zone.END)),
            dims.KDim: (k_domain(v_grid.Zone.TOP), k_domain(v_grid.Zone.BOTTOM)),
        },
        fields={"average": "out"},
        deps={"input_field": "in"},
        do_exchange=True,
    )
    source.register_provider(provider)
    field = source.get("out")

    halo_points = data_alloc.as_numpy(
        decomposition_info.local_index(dims.CellDim, decomposition.DecompositionInfo.EntryType.HALO)
    )
    owned_points = data_alloc.as_numpy(
        decomposition_info.local_index(
            dims.CellDim, decomposition.DecompositionInfo.EntryType.OWNED
        )
    )
    assert (field.ndarray[owned_points, :] == number).all()
    assert not (field.ndarray[halo_points, :] == number).all()


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_numpy_provider_exchange(
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    grid_savepoint: sb.IconGridSavepoint,
    backend: gtx_typing.Backend | None,
) -> None:
    parallel_helpers.check_comm_size(processor_props, sizes=(2, 4))
    exchange = decomposition.create_exchange(processor_props, decomposition_info)
    grid = grid_savepoint.construct_icon_grid(backend=backend)

    number = processor_props.rank + 10
    input_field = data_alloc.constant_field(
        grid, number, dims.CellDim, dims.KDim, allocator=backend
    )
    source = SimpleFieldSource(
        data_={"in": (input_field, {"standard_name": "in", "units": ""})},
        backend=backend,
        grid=grid,
    )
    source._exchange = exchange

    def identity(ar: data_alloc.NDArray) -> data_alloc.NDArray:
        return ar

    provider = factory.NumpyDataProvider(
        func=identity,
        domain=(dims.CellDim, dims.KDim),
        fields=("out",),
        deps={"ar": "in"},
        do_exchange=True,
    )
    source.register_provider(provider)
    field = source.get("out")

    halo_points = data_alloc.as_numpy(
        decomposition_info.local_index(dims.CellDim, decomposition.DecompositionInfo.EntryType.HALO)
    )
    owned_points = data_alloc.as_numpy(
        decomposition_info.local_index(
            dims.CellDim, decomposition.DecompositionInfo.EntryType.OWNED
        )
    )
    assert (field.ndarray[owned_points, :] == number).all()
    assert not (field.ndarray[halo_points, :] == number).all()
