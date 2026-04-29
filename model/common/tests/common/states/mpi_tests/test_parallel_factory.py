# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from gt4py import next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.decomposition import definitions as decomposition, mpi_decomposition
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.states import factory
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import parallel_helpers

from ..fixtures import (
    backend,
    data_provider,
    decomposition_info,
    download_ser_data,
    experiment,
    grid_savepoint,
    process_props,
)
from ..unit_tests.test_factory import SimpleFieldSource


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.testing import serialbox as sb


if mpi_decomposition.mpi4py is None:
    pytest.skip("Skipping parallel tests on single node installation", allow_module_level=True)


@gtx.field_operator
def _fill_op(field: fa.EdgeField[gtx.int32], value: gtx.int32) -> fa.EdgeField[gtx.int32]:
    return field + value


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def _fill_edges(
    field: fa.EdgeField[gtx.int32],
    value: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _fill_op(field, value, out=field, domain={dims.EdgeDim: (horizontal_start, horizontal_end)})


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize("do_exchange", [True, False])
def test_program_provider_exchange(
    do_exchange: bool,
    process_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    grid_savepoint: sb.IconGridSavepoint,
    backend: gtx_typing.Backend | None,
) -> None:
    parallel_helpers.check_comm_size(process_props, sizes=(2, 4))
    exchange = decomposition.create_exchange(process_props, decomposition_info)
    grid = grid_savepoint.construct_icon_grid(backend=backend)

    number = process_props.rank + 10
    source = SimpleFieldSource(
        data_={},
        backend=backend,
        grid=grid,
    ).with_metadata({"out": {"dtype": np.int32, "standard_name": "out", "units": ""}})
    source._exchange = exchange
    edge_domain = h_grid.domain(dims.EdgeDim)
    provider = factory.ProgramFieldProvider(
        func=_fill_edges,
        domain={
            dims.EdgeDim: (edge_domain(h_grid.Zone.LOCAL), edge_domain(h_grid.Zone.END)),
        },
        fields={"field": "out"},
        deps={},
        params={"value": number},
        do_exchange=do_exchange,
    )
    source.register_provider(provider)
    field = source.get("out")

    halo_points = decomposition_info.local_index(
        dims.EdgeDim, decomposition.DecompositionInfo.EntryType.HALO
    )
    owned_points = decomposition_info.local_index(
        dims.EdgeDim, decomposition.DecompositionInfo.EntryType.OWNED
    )
    valid_values = {r + 10 for r in range(process_props.comm_size)}
    arr = field.ndarray

    assert (arr[owned_points] == number).all()  # type: ignore[attr-defined]
    if do_exchange:
        assert (arr[halo_points] != number).all()  # type: ignore[attr-defined]
        assert set(arr[halo_points].flatten()).issubset(valid_values)  # type: ignore[attr-defined]
    else:
        assert (arr[halo_points] == number).all()  # type: ignore[attr-defined]


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize("do_exchange", [True, False])
def test_numpy_provider_exchange(
    do_exchange: bool,
    process_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    grid_savepoint: sb.IconGridSavepoint,
    backend: gtx_typing.Backend | None,
) -> None:
    parallel_helpers.check_comm_size(process_props, sizes=(2, 4))
    exchange = decomposition.create_exchange(process_props, decomposition_info)
    grid = grid_savepoint.construct_icon_grid(backend=backend)

    number = process_props.rank + 10
    source = SimpleFieldSource(
        data_={},
        backend=backend,
        grid=grid,
    )
    source._exchange = exchange

    xp = data_alloc.import_array_ns(backend)

    def make_constant(size: int, value: int) -> data_alloc.NDArray:
        return xp.full(size, value)

    provider = factory.NumpyDataProvider(
        func=make_constant,
        domain=(dims.EdgeDim,),
        fields=("out",),
        deps={},
        params={"size": grid.size[dims.EdgeDim], "value": number},
        do_exchange=do_exchange,
    )
    source.register_provider(provider)
    field = source.get("out")

    halo_points = decomposition_info.local_index(
        dims.EdgeDim, decomposition.DecompositionInfo.EntryType.HALO
    )
    owned_points = decomposition_info.local_index(
        dims.EdgeDim, decomposition.DecompositionInfo.EntryType.OWNED
    )
    valid_values = {r + 10 for r in range(process_props.comm_size)}
    arr = field.ndarray

    assert (arr[owned_points] == number).all()  # type: ignore[attr-defined]
    if do_exchange:
        assert (arr[halo_points] != number).all()  # type: ignore[attr-defined]
        assert set(arr[halo_points].flatten()).issubset(valid_values)  # type: ignore[attr-defined]
    else:
        assert (arr[halo_points] == number).all()  # type: ignore[attr-defined]
