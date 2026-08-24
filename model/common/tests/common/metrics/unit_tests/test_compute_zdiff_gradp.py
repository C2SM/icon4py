# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.grid.horizontal as h_grid
from icon4py.model.common import dimension as dims
from icon4py.model.common.metrics.compute_zdiff_gradp import (
    compute_zdiff_gradp,
    compute_zdiff_gradp_v2,
)
from icon4py.model.common.metrics.metric_fields import compute_flat_max_idx
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import test_utils
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    process_props,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


def _main_reference(  # noqa: PLR0912
    *,
    e2c: np.ndarray,
    z_me: np.ndarray,
    z_mc: np.ndarray,
    z_ifc: np.ndarray,
    flat_idx: np.ndarray,
    topography: np.ndarray,
    nlev: int,
    horizontal_start: int,
    horizontal_start_1: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Element-by-element reference matching the main-branch loop semantics."""
    nedges = e2c.shape[0]
    z_aux1 = np.maximum(topography[e2c[:, 0]], topography[e2c[:, 1]])
    z_aux2 = z_aux1 - 5.0

    zdiff_gradp = np.zeros_like(z_mc[e2c])
    jk_field = np.arange(nlev, dtype=np.int32)
    zdiff_gradp[horizontal_start:, :, :] = (
        np.expand_dims(z_me, axis=1)[horizontal_start:, :, :] - z_mc[e2c][horizontal_start:, :, :]
    )
    vertidx_gradp = np.broadcast_to(jk_field[None, None, :], (nedges, 2, nlev)).copy()
    vertoffset_gradp = np.broadcast_to(jk_field[None, None, :], (nedges, 2, nlev)).copy()

    for je in range(horizontal_start, nedges):
        for jk in range(int(flat_idx[je]) + 1, nlev):
            param = np.zeros((nlev,), dtype=bool)
            for jk1 in range(int(flat_idx[je]), nlev):
                if jk1 == nlev - 1 or (
                    z_me[je, jk] <= z_ifc[e2c[je, 0], jk1]
                    and z_me[je, jk] >= z_ifc[e2c[je, 0], jk1 + 1]
                ):
                    param[jk1] = True
            idx = int(np.where(param)[0][0])
            vertidx_gradp[je, 0, jk] = idx
            zdiff_gradp[je, 0, jk] = z_me[je, jk] - z_mc[e2c[je, 0], idx]

        jk_start = int(flat_idx[je])
        for jk in range(int(flat_idx[je]) + 1, nlev):
            for jk1 in range(jk_start, nlev):
                if jk1 == nlev - 1 or (
                    z_me[je, jk] <= z_ifc[e2c[je, 1], jk1]
                    and z_me[je, jk] >= z_ifc[e2c[je, 1], jk1 + 1]
                ):
                    vertidx_gradp[je, 1, jk] = jk1
                    zdiff_gradp[je, 1, jk] = z_me[je, jk] - z_mc[e2c[je, 1], jk1]
                    jk_start = jk1
                    break

    for je in range(horizontal_start_1, nedges):
        jk_start = int(flat_idx[je])
        for jk in range(int(flat_idx[je]) + 1, nlev):
            if z_me[je, jk] < z_aux2[je]:
                for jk1 in range(jk_start, nlev):
                    if jk1 == nlev - 1 or (
                        z_aux2[je] <= z_ifc[e2c[je, 0], jk1]
                        and z_aux2[je] >= z_ifc[e2c[je, 0], jk1 + 1]
                    ):
                        vertidx_gradp[je, 0, jk] = jk1
                        zdiff_gradp[je, 0, jk] = z_aux2[je] - z_mc[e2c[je, 0], jk1]
                        jk_start = jk1
                        break

        jk_start = int(flat_idx[je])
        for jk in range(int(flat_idx[je]) + 1, nlev):
            if z_me[je, jk] < z_aux2[je]:
                for jk1 in range(jk_start, nlev):
                    if jk1 == nlev - 1 or (
                        z_aux2[je] <= z_ifc[e2c[je, 1], jk1]
                        and z_aux2[je] >= z_ifc[e2c[je, 1], jk1 + 1]
                    ):
                        vertidx_gradp[je, 1, jk] = jk1
                        zdiff_gradp[je, 1, jk] = z_aux2[je] - z_mc[e2c[je, 1], jk1]
                        jk_start = jk1
                        break

    vertoffset_gradp = vertidx_gradp - vertoffset_gradp
    return zdiff_gradp, vertoffset_gradp


@pytest.mark.level("unit")
@pytest.mark.datatest
@pytest.mark.parametrize("compute_fn", [compute_zdiff_gradp, compute_zdiff_gradp_v2])
def test_compute_zdiff_gradp(  # noqa: PLR0917
    icon_grid: base_grid.Grid,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    backend: gtx_typing.Backend,
    compute_fn: Callable[..., tuple[data_alloc.NDArray, data_alloc.NDArray]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ICON4PY_VALIDATE_ZDIFF_GRADP", "1")

    zdiff_gradp_ref = metrics_savepoint.zdiff_gradp()
    vertoffset_gradp_ref = metrics_savepoint.vertoffset_gradp()

    e2c = icon_grid.get_connectivity("E2C").ndarray
    z_ifc = metrics_savepoint.z_ifc()
    z_ifc_ground_level = z_ifc.ndarray[:, icon_grid.num_levels]
    z_mc = metrics_savepoint.z_mc()
    c_lin_e = interpolation_savepoint.c_lin_e().ndarray
    k_lev = data_alloc.index_field(
        icon_grid, dims.KDim, extend={dims.KDim: 1}, dtype=gtx.int32, allocator=backend
    )
    edge_domain = h_grid.domain(dims.EdgeDim)
    horizontal_start_edge = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    start_nudging = icon_grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))

    z_me = np.sum(z_mc.ndarray[e2c] * np.expand_dims(c_lin_e, axis=-1), axis=1)

    flat_idx_np = compute_flat_max_idx(
        e2c=e2c,
        z_me=z_me,
        z_ifc=z_ifc.ndarray,
        k_lev=k_lev.ndarray,
    )

    zdiff_gradp_full_field, vertoffset_gradp_full_field = compute_fn(
        e2c=e2c,
        z_me=z_me,
        z_mc=z_mc.ndarray,
        z_ifc=metrics_savepoint.z_ifc().ndarray,
        flat_idx=flat_idx_np,
        topography=z_ifc_ground_level,
        nlev=icon_grid.num_levels,
        horizontal_start=horizontal_start_edge,
        horizontal_start_1=start_nudging,
    )

    assert test_utils.dallclose(
        data_alloc.as_numpy(zdiff_gradp_full_field),
        zdiff_gradp_ref.asnumpy(),
        atol=1e-10,
        rtol=1.0e-9,
    )

    assert test_utils.dallclose(
        data_alloc.as_numpy(vertoffset_gradp_full_field),
        vertoffset_gradp_ref.asnumpy(),
        atol=1e-10,
        rtol=1.0e-9,
    )


@pytest.mark.level("unit")
@pytest.mark.parametrize("validation_enabled", ["1", "0"])
def test_compute_zdiff_gradp_endpoint_forcing(
    monkeypatch: pytest.MonkeyPatch, validation_enabled: str
) -> None:
    monkeypatch.setenv("ICON4PY_VALIDATE_ZDIFF_GRADP", validation_enabled)

    nedges = 4
    ncells = 4
    nlev = 8
    hs = 0
    hs1 = 2
    e0 = 2
    cell0 = 0
    cell1 = 1

    topography = np.array([1000.0, 1000.0, 2000.0, 1500.0], dtype=np.float64)
    # Force the phase-2 endpoint on e0 per the amended Benchmark contract.
    topography[cell1] = 30000.0 - (nlev - 2) * (30000.0 - topography[cell0]) / nlev + 6.0

    z_ifc = np.empty((ncells, nlev + 1), dtype=np.float64)
    for c in range(ncells):
        top = topography[c]
        for k in range(nlev + 1):
            z_ifc[c, k] = 30000.0 - k * (30000.0 - top) / nlev

    z_mc = 0.5 * (z_ifc[:, :-1] + z_ifc[:, 1:])
    c_lin_e = np.full((nedges, 2), 0.5, dtype=np.float64)
    e2c = np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=np.int64)
    z_me = np.sum(z_mc[e2c] * c_lin_e[:, :, None], axis=1)

    flat_idx = np.zeros((nedges,), dtype=np.int32)
    flat_idx[e0] = nlev - 2
    z_me[e0, nlev - 1] = z_ifc[cell0, nlev - 1]

    z_aux1 = np.maximum(topography[e2c[:, 0]], topography[e2c[:, 1]])
    z_aux2 = z_aux1 - 5.0

    # Precondition asserts from the amended Benchmark contract.
    assert flat_idx[e0] < nlev - 1
    assert z_aux2[e0] > z_ifc[cell0, flat_idx[e0]]
    for c in (cell0, cell1):
        for jk in range(flat_idx[e0] + 1, nlev):
            assert z_me[e0, jk] <= z_ifc[e2c[e0, c], flat_idx[e0]]
    assert e0 >= hs1
    assert z_me[e0, nlev - 1] < z_aux2[e0]

    golden_zdiff, golden_vert = _main_reference(
        e2c=e2c,
        z_me=z_me,
        z_mc=z_mc,
        z_ifc=z_ifc,
        flat_idx=flat_idx,
        topography=topography,
        nlev=nlev,
        horizontal_start=hs,
        horizontal_start_1=hs1,
    )

    candidate_zdiff, candidate_vert = compute_zdiff_gradp_v2(
        e2c=e2c,
        z_me=z_me,
        z_mc=z_mc,
        z_ifc=z_ifc,
        flat_idx=flat_idx,
        topography=topography,
        nlev=nlev,
        horizontal_start=gtx.int32(hs),
        horizontal_start_1=gtx.int32(hs1),
    )

    baseline_zdiff, baseline_vert = compute_zdiff_gradp(
        e2c=e2c,
        z_me=z_me,
        z_mc=z_mc,
        z_ifc=z_ifc,
        flat_idx=flat_idx,
        topography=topography,
        nlev=nlev,
        horizontal_start=gtx.int32(hs),
        horizontal_start_1=gtx.int32(hs1),
    )

    assert np.allclose(candidate_zdiff, golden_zdiff)
    assert np.array_equal(candidate_vert, golden_vert)

    assert not np.allclose(baseline_zdiff, golden_zdiff) or not np.array_equal(
        baseline_vert, golden_vert
    )
