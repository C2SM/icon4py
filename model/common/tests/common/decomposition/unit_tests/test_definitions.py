# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import math

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
        k: v.ndarray for k, v in grid.connectivities.items() if gtx_common.is_neighbor_table(v)
    }
    return neighbor_tables


offsets = [dims.E2C, dims.E2V, dims.C2E, dims.C2E2C, dims.V2C, dims.V2E, dims.C2V, dims.E2C2V]


@pytest.mark.parametrize("dim", dims.horizontal_dims())
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


def test_decomposition_info_as_host_returns_self_for_host_buffers() -> None:
    decomp = definitions.DecompositionInfo()
    decomp.set_dimension(
        dims.CellDim,
        np.asarray([2, 0, 1], dtype=np.int64),
        np.asarray([True, True, False]),
        np.where(
            [True, True, False],
            definitions.DecompositionFlag.OWNED.value,
            definitions.DecompositionFlag.FIRST_HALO_LEVEL.value,
        ),
    )
    assert decomp.as_host() is decomp


def test_decomposition_info_as_host_copies_device_buffers() -> None:
    cp = pytest.importorskip("cupy")
    decomp = definitions.DecompositionInfo()
    decomp.set_dimension(
        dims.CellDim,
        cp.asarray([2, 0, 1]),
        cp.asarray([True, True, False]),
        cp.full((3,), definitions.DecompositionFlag.OWNED.value),
    )
    host = decomp.as_host()
    assert host is not decomp
    for array in (
        host.global_index(dims.CellDim),
        host.owner_mask(dims.CellDim),
        host.halo_levels(dims.CellDim),
    ):
        assert isinstance(array, np.ndarray)


def _sum_on_emulated_ranks(values: np.ndarray, num_ranks: int) -> float:
    """Sum ``values`` as ``GlobalReductions`` does: integer partial sums per rank, combined exactly."""
    chunks = np.array_split(values, num_ranks)
    abs_max = float(np.max(np.abs(values)))
    shift = definitions.fixed_point_shift(abs_max)
    other_partial_sums = [
        definitions.fixed_point_partial_sums(chunk, shift) for chunk in chunks[1:]
    ]

    def combine(integer_sum: int, fraction_sum: int) -> tuple[int, int]:
        return (
            integer_sum + sum(partial[0] for partial in other_partial_sums),
            fraction_sum + sum(partial[1] for partial in other_partial_sums),
        )

    return definitions.order_insensitive_sum(chunks[0], abs_max, combine_partial_sums=combine)


@pytest.mark.parametrize("num_ranks", [1, 2, 3, 7])
def test_order_insensitive_sum_does_not_depend_on_the_decomposition(num_ranks: int) -> None:
    rng = np.random.default_rng(42)
    size = 100_003
    # magnitudes spread over many binades, both signs
    values = rng.lognormal(mean=0.0, sigma=4.0, size=size) * rng.choice([-1.0, 1.0], size=size)

    single_rank = definitions.SingleNodeReductions().sum(values)

    assert _sum_on_emulated_ranks(values, num_ranks) == single_rank
    # the only error is the truncation of every contribution below abs_max * 2**-60
    abs_max = float(np.max(np.abs(values)))
    assert abs(single_rank - math.fsum(values)) <= size * abs_max * 2.0**-59


def test_single_node_sum_and_mean() -> None:
    reductions = definitions.SingleNodeReductions()
    rng = np.random.default_rng(7)
    values = rng.random((37, 5)) - 0.25

    assert reductions.sum(values) == pytest.approx(math.fsum(values.ravel()), rel=1e-15)
    assert reductions.mean(values) == reductions.sum(values) / values.size
    assert reductions.sum(np.array([-2.5, 1.25, 4.0])) == 2.75
    assert reductions.sum(np.zeros(5)) == 0.0
    assert reductions.sum(np.array([1, 2, 3])) == 6
    assert math.isnan(reductions.sum(np.array([1.0, np.nan])))
    assert reductions.sum(np.array([1.0, np.inf])) == np.inf


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_sum_of_finite_values_overflows_to_infinity_like_a_plain_sum(sign: float) -> None:
    values = sign * np.array([1e308, 1e308])
    with np.errstate(over="ignore"):
        assert definitions.SingleNodeReductions().sum(values) == np.sum(values) == sign * np.inf
