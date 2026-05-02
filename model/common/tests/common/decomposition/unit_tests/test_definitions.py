# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import sys
import warnings

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next import common as gtx_common

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import simple
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions as test_defs
from icon4py.model.testing.fixtures.datatest import (  # import fixtures form test_utils
    backend,
    data_provider,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    process_props,
)

from ...grid import utils as grid_utils


@pytest.mark.parametrize("process_props", [False], indirect=True)
def test_create_single_node_runtime_without_mpi(process_props):  # fixture
    decomposition_info = definitions.DecompositionInfo()
    exchange = definitions.create_exchange(process_props, decomposition_info)

    assert isinstance(exchange, definitions.SingleNodeExchange)


def get_neighbor_tables_for_simple_grid() -> dict[str, data_alloc.NDArray]:
    grid = simple.simple_grid()
    neighbor_tables = {
        k: v.ndarray
        for k, v in grid.connectivities.items()
        if gtx_common.is_neighbor_connectivity(v)
    }
    return neighbor_tables


offsets = [dims.E2C, dims.E2V, dims.C2E, dims.C2E2C, dims.V2C, dims.V2E, dims.C2V, dims.E2C2V]


@pytest.mark.parametrize("dim", grid_utils.main_horizontal_dims())
def test_decomposition_info_single_node_empty_halo(dim: gtx.Dimension) -> None:
    manager = grid_utils.run_grid_manager(
        test_defs.Grids.MCH_CH_R04B09_DSL, keep_skip_values=True, backend=None
    )

    decomposition_info = manager.decomposition_info
    for level in (
        definitions.DecompositionFlag.FIRST_HALO_LEVEL,
        definitions.DecompositionFlag.SECOND_HALO_LEVEL,
        definitions.DecompositionFlag.THIRD_HALO_LEVEL,
    ):
        assert decomposition_info.get_halo_size(dim, level) == 0
        assert np.count_nonzero(decomposition_info.halo_level_mask(dim, level)) == 0
    assert (
        decomposition_info.get_halo_size(dim, definitions.DecompositionFlag.OWNED)
        == manager.grid.size[dim]
    )


@pytest.mark.parametrize(
    "flag, expected",
    [
        (definitions.DecompositionFlag.OWNED, False),
        (definitions.DecompositionFlag.SECOND_HALO_LEVEL, True),
        (definitions.DecompositionFlag.THIRD_HALO_LEVEL, True),
        (definitions.DecompositionFlag.FIRST_HALO_LEVEL, True),
    ],
)
def test_decomposition_info_is_distributed(flag, expected) -> None:
    mesh = simple.simple_grid(allocator=None, num_levels=10)
    decomp = definitions.DecompositionInfo()
    decomp.set_dimension(
        dims.CellDim,
        np.arange(mesh.num_cells),
        np.arange(mesh.num_cells),
        np.full((mesh.num_cells,), flag),
    )
    assert decomp.is_distributed() == expected


def test_single_node_exchange_warns_on_first_use(monkeypatch):
    monkeypatch.setattr(definitions.SingleNodeExchange, "_warning_emitted", False)

    exchange = definitions.SingleNodeExchange()

    with pytest.warns(RuntimeWarning, match="SingleNodeExchange"):
        exchange.start(dims.CellDim)


def test_single_node_exchange_does_not_warn_on_construction_or_repeat_use(monkeypatch):
    monkeypatch.setattr(definitions.SingleNodeExchange, "_warning_emitted", False)

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        exchange = definitions.SingleNodeExchange()

    assert len(recorded_warnings) == 0

    with pytest.warns(RuntimeWarning, match="SingleNodeExchange"):
        exchange.exchange(dims.CellDim)

    with warnings.catch_warnings(record=True) as repeated_warnings:
        warnings.simplefilter("always")
        exchange.start(dims.CellDim)

    assert len(repeated_warnings) == 0


def _assert_warning_points_to_call_site(monkeypatch, func, expected_line):
    monkeypatch.setattr(SingleNodeExchange, "_warning_emitted", False)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        func()

    assert len(caught_warnings) == 1
    assert caught_warnings[0].filename == __file__
    assert caught_warnings[0].lineno == expected_line


def test_single_node_exchange_warning_points_to_call_site(monkeypatch):
    exchange = SingleNodeExchange()

    exchange_line = sys._getframe().f_lineno + 1
    _assert_warning_points_to_call_site(
        monkeypatch, lambda: exchange.start(dims.CellDim), exchange_line
    )

    wait_line = sys._getframe().f_lineno + 1
    _assert_warning_points_to_call_site(
        monkeypatch, lambda: exchange.exchange(dims.CellDim), wait_line
    )
