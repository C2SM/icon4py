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
import icon4py.model.common.metrics.compute_zdiff_gradp as _zdiff_mod
from icon4py.model.common import dimension as dims
from icon4py.model.common.metrics.compute_zdiff_gradp import (
    _exact_phase1_cell0,
    _exact_query_succ,
    _first_match_scan_reference,
    compute_zdiff_gradp,
    compute_zdiff_gradp_dispatch,
    compute_zdiff_gradp_exact,
    compute_zdiff_gradp_exact_v2,
    compute_zdiff_gradp_exact_v3,
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
@pytest.mark.parametrize(
    "compute_fn",
    [
        compute_zdiff_gradp,
        compute_zdiff_gradp_v2,
        compute_zdiff_gradp_exact,
        compute_zdiff_gradp_exact_v2,
        compute_zdiff_gradp_exact_v3,
        compute_zdiff_gradp_dispatch,
    ],
)
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
@pytest.mark.parametrize("validation_enabled", [True, False])
def test_compute_zdiff_gradp_endpoint_forcing(
    monkeypatch: pytest.MonkeyPatch, validation_enabled: bool
) -> None:
    if validation_enabled:
        monkeypatch.setenv("ICON4PY_VALIDATE_ZDIFF_GRADP", "1")
    else:
        monkeypatch.setenv("ICON4PY_VALIDATE_ZDIFF_GRADP", "0")

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

    exact_zdiff, exact_vert = compute_zdiff_gradp_exact(
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

    assert np.allclose(exact_zdiff, golden_zdiff)
    assert np.array_equal(exact_vert, golden_vert)

    exact3_zdiff, exact3_vert = compute_zdiff_gradp_exact_v3(
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
    assert np.allclose(exact3_zdiff, golden_zdiff)
    assert np.array_equal(exact3_vert, golden_vert)
    assert _zdiff_mod._LAST_EXACT_V3_PATH == "fast"

    assert not np.allclose(baseline_zdiff, golden_zdiff) or not np.array_equal(
        baseline_vert, golden_vert
    )


@pytest.mark.level("unit")
@pytest.mark.parametrize(
    "compute_fn",
    [
        compute_zdiff_gradp_v2,
        compute_zdiff_gradp_exact,
        compute_zdiff_gradp_exact_v2,
        compute_zdiff_gradp_exact_v3,
        compute_zdiff_gradp_dispatch,
    ],
)
def test_compute_zdiff_gradp_nan_validation(
    monkeypatch: pytest.MonkeyPatch, compute_fn: Callable[..., tuple[np.ndarray, np.ndarray]]
) -> None:
    nedges = 4
    ncells = 4
    nlev = 8
    hs = 0
    hs1 = 2

    topography = np.array([1000.0, 1000.0, 2000.0, 1500.0], dtype=np.float64)
    z_ifc = np.empty((ncells, nlev + 1), dtype=np.float64)
    for c in range(ncells):
        top = topography[c]
        for k in range(nlev + 1):
            z_ifc[c, k] = 30000.0 - k * (30000.0 - top) / nlev

    z_mc = 0.5 * (z_ifc[:, :-1] + z_ifc[:, 1:])
    c_lin_e = np.full((nedges, 2), 0.5, dtype=np.float64)
    e2c = np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=np.int64)
    z_me = np.sum(z_mc[e2c] * c_lin_e[:, :, None], axis=1)
    z_me[0, 2] = np.nan

    # Place NaN at jk > flat_idx so the kernels actually see it.
    flat_idx = np.full((nedges,), 1, dtype=np.int32)

    # Validation ON (default): ValueError before compute.
    monkeypatch.setenv("ICON4PY_VALIDATE_ZDIFF_GRADP", "1")
    with pytest.raises(ValueError):
        compute_fn(
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

    # Validation OFF: defined nlev-1 fallback, no crash for all variants.
    monkeypatch.setenv("ICON4PY_VALIDATE_ZDIFF_GRADP", "0")
    zdiff_gradp, vertoffset_gradp = compute_fn(
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
    assert zdiff_gradp.shape == (nedges, 2, nlev)
    assert vertoffset_gradp.shape == (nedges, 2, nlev)
    assert np.all(np.isfinite(vertoffset_gradp))


def _build_p3_e3_violation_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
    int,
    int,
]:
    nedges = 4
    ncells = 4
    nlev = 8
    hs = 0
    hs1 = 2
    e0 = 2
    cell0 = 0
    fi = 2

    topography = np.array([1000.0, 1000.0, 2000.0, 1500.0], dtype=np.float64)
    z_ifc = np.empty((ncells, nlev + 1), dtype=np.float64)
    for c in range(ncells):
        top = topography[c]
        for k in range(nlev + 1):
            z_ifc[c, k] = 30000.0 - k * (30000.0 - top) / nlev

    z_mc = 0.5 * (z_ifc[:, :-1] + z_ifc[:, 1:])
    c_lin_e = np.full((nedges, 2), 0.5, dtype=np.float64)
    e2c = np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=np.int64)
    z_me = np.sum(z_mc[e2c] * c_lin_e[:, :, None], axis=1)

    flat_idx = np.full((nedges,), fi, dtype=np.int32)

    # Force strictly increasing z_me over the active segment, crossing bracket
    # boundaries upward so the carry semantics differ from a fresh per-jk scan.
    z_me[e0, fi + 1] = 0.5 * (z_ifc[cell0, fi + 1] + z_ifc[cell0, fi + 2])
    z_me[e0, fi + 2] = 0.5 * (z_ifc[cell0, fi] + z_ifc[cell0, fi + 1])
    z_me[e0, fi + 3] = 0.5 * (z_ifc[cell0, fi - 1] + z_ifc[cell0, fi])
    z_me[e0, fi + 4] = 0.5 * (z_ifc[cell0, fi - 2] + z_ifc[cell0, fi - 1])
    z_me[e0, nlev - 1] = z_ifc[cell0, 0] - 100.0
    return e2c, z_me, z_mc, z_ifc, flat_idx, topography, nlev, hs, hs1, e0, 0


@pytest.mark.level("unit")
@pytest.mark.parametrize("validation_enabled", [True, False])
def test_compute_zdiff_gradp_e3_violation(
    monkeypatch: pytest.MonkeyPatch, validation_enabled: bool
) -> None:
    if validation_enabled:
        monkeypatch.setenv("ICON4PY_VALIDATE_ZDIFF_GRADP", "1")
    else:
        monkeypatch.setenv("ICON4PY_VALIDATE_ZDIFF_GRADP", "0")

    e2c, z_me, z_mc, z_ifc, flat_idx, topography, nlev, hs, hs1, e0, _cell0 = (
        _build_p3_e3_violation_inputs()
    )

    # Sanity-check that E3 is violated on the forced edge.
    fi = int(flat_idx[e0])
    assert any(z_me[e0, k] < z_me[e0, k + 1] for k in range(fi + 1, nlev - 1))

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

    exact2_zdiff, exact2_vert = compute_zdiff_gradp_exact_v2(
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
    exact3_zdiff, exact3_vert = compute_zdiff_gradp_exact_v3(
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

    if validation_enabled:
        assert np.allclose(exact2_zdiff, golden_zdiff)
        assert np.array_equal(exact2_vert, golden_vert)
        assert _zdiff_mod._LAST_EXACT_V2_PATH == "carry"
        assert np.allclose(exact3_zdiff, golden_zdiff)
        assert np.array_equal(exact3_vert, golden_vert)
        assert _zdiff_mod._LAST_EXACT_V3_PATH == "carry"
    else:
        # Validation OFF: exact_v2/exact_v3 take the fast path without E3 check.
        assert _zdiff_mod._LAST_EXACT_V2_PATH == "fast"
        assert _zdiff_mod._LAST_EXACT_V3_PATH == "fast"
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
    # The baseline vectorized path does not replicate main's carry for E3 violations.
    baseline_matches = np.allclose(baseline_zdiff, golden_zdiff) and np.array_equal(
        baseline_vert, golden_vert
    )

    if validation_enabled:
        with pytest.raises(ValueError):
            compute_zdiff_gradp_v2(
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

        dispatch_zdiff, dispatch_vert = compute_zdiff_gradp_dispatch(
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
        assert np.allclose(dispatch_zdiff, golden_zdiff)
        assert np.array_equal(dispatch_vert, golden_vert)
        assert _zdiff_mod._LAST_DISPATCH_PATH == "exact"
        assert not baseline_matches
    else:
        v2_zdiff, v2_vert = compute_zdiff_gradp_v2(
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
        v2_matches = np.allclose(v2_zdiff, golden_zdiff) and np.array_equal(v2_vert, golden_vert)
        assert not v2_matches

        dispatch_zdiff, dispatch_vert = compute_zdiff_gradp_dispatch(
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
        dispatch_matches = np.allclose(dispatch_zdiff, golden_zdiff) and np.array_equal(
            dispatch_vert, golden_vert
        )
        assert not dispatch_matches
        assert _zdiff_mod._LAST_DISPATCH_PATH == "fast"


@pytest.mark.level("unit")
def test_compute_zdiff_gradp_dispatch_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ICON4PY_VALIDATE_ZDIFF_GRADP", "1")

    def _check_case(
        builder: Callable[
            [],
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                int,
                int,
                int,
                int,
                int,
            ],
        ],
        expected_dispatch_path: str,
        expect_golden: bool,
    ) -> None:
        e2c, z_me, z_mc, z_ifc, flat_idx, topography, nlev, hs, hs1, _e0, _cell0 = builder()
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
        dispatch_zdiff, dispatch_vert = compute_zdiff_gradp_dispatch(
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
        assert expected_dispatch_path == _zdiff_mod._LAST_DISPATCH_PATH
        if expect_golden:
            assert np.allclose(dispatch_zdiff, golden_zdiff)
            assert np.array_equal(dispatch_vert, golden_vert)

    # Endpoint forcing: E1/E3 valid -> dispatch routes to v2 (fast) and matches golden.
    _check_case(_build_endpoint_forcing_inputs, "fast", True)
    # P1 interior tie: E1/E3 valid but E2 tie detected -> dispatch routes to exact fallback.
    _check_case(_build_p1_interior_tie_inputs, "exact", True)
    _check_case(_build_p2_zero_thickness_inputs, "exact", True)
    # P3 E3 violation: E3 invalid -> dispatch routes to exact_v2 fallback.
    _check_case(_build_p3_e3_violation_inputs, "exact", True)


@pytest.mark.level("unit")
@pytest.mark.parametrize("validation_enabled", [True, False])
def test_compute_zdiff_gradp_interior_tie(
    monkeypatch: pytest.MonkeyPatch, validation_enabled: bool
) -> None:
    """P1: an exact z_me == z_ifc interior interface is caught and routed to exact."""
    if validation_enabled:
        monkeypatch.setenv("ICON4PY_VALIDATE_ZDIFF_GRADP", "1")
    else:
        monkeypatch.setenv("ICON4PY_VALIDATE_ZDIFF_GRADP", "0")

    e2c, z_me, z_mc, z_ifc, flat_idx, topography, nlev, hs, hs1, _e0, _cell0 = (
        _build_p1_interior_tie_inputs()
    )

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

    if validation_enabled:
        with pytest.raises(ValueError, match="exact interior tie"):
            compute_zdiff_gradp_v2(
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

        dispatch_zdiff, dispatch_vert = compute_zdiff_gradp_dispatch(
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
        assert np.allclose(dispatch_zdiff, golden_zdiff)
        assert np.array_equal(dispatch_vert, golden_vert)
        assert _zdiff_mod._LAST_DISPATCH_PATH == "exact"
    else:
        v2_zdiff, v2_vert = compute_zdiff_gradp_v2(
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
        v2_matches = np.allclose(v2_zdiff, golden_zdiff) and np.array_equal(v2_vert, golden_vert)
        assert not v2_matches


def _build_endpoint_forcing_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
    int,
    int,
]:
    nedges = 4
    ncells = 4
    nlev = 8
    hs = 0
    hs1 = 2
    e0 = 2
    cell0 = 0
    cell1 = 1

    topography = np.array([1000.0, 1000.0, 2000.0, 1500.0], dtype=np.float64)
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

    return e2c, z_me, z_mc, z_ifc, flat_idx, topography, nlev, hs, hs1, e0, cell0


def _build_p1_interior_tie_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
    int,
    int,
]:
    nedges = 4
    ncells = 4
    nlev = 8
    hs = 0
    hs1 = 2
    e0 = 2
    cell0 = 0
    fi = 2
    tie_level = fi + 1

    topography = np.array([1000.0, 1000.0, 2000.0, 1500.0], dtype=np.float64)
    z_ifc = np.empty((ncells, nlev + 1), dtype=np.float64)
    for c in range(ncells):
        top = topography[c]
        for k in range(nlev + 1):
            z_ifc[c, k] = 30000.0 - k * (30000.0 - top) / nlev

    z_mc = 0.5 * (z_ifc[:, :-1] + z_ifc[:, 1:])
    c_lin_e = np.full((nedges, 2), 0.5, dtype=np.float64)
    e2c = np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=np.int64)
    z_me = np.sum(z_mc[e2c] * c_lin_e[:, :, None], axis=1)
    z_me[e0, fi + 1] = z_ifc[cell0, tie_level]

    flat_idx = np.full((nedges,), fi, dtype=np.int32)

    return e2c, z_me, z_mc, z_ifc, flat_idx, topography, nlev, hs, hs1, e0, cell0


def _build_p2_zero_thickness_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
    int,
    int,
]:
    nedges = 4
    ncells = 4
    nlev = 8
    hs = 0
    hs1 = 2
    e0 = 2
    cell0 = 0
    fi = 1
    thin_level = 4

    topography = np.array([1000.0, 1000.0, 2000.0, 1500.0], dtype=np.float64)
    z_ifc = np.empty((ncells, nlev + 1), dtype=np.float64)
    for c in range(ncells):
        top = topography[c]
        for k in range(nlev + 1):
            z_ifc[c, k] = 30000.0 - k * (30000.0 - top) / nlev

    z_ifc[cell0, thin_level + 1] = z_ifc[cell0, thin_level]

    z_mc = 0.5 * (z_ifc[:, :-1] + z_ifc[:, 1:])
    c_lin_e = np.full((nedges, 2), 0.5, dtype=np.float64)
    e2c = np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=np.int64)
    z_me = np.sum(z_mc[e2c] * c_lin_e[:, :, None], axis=1)
    z_me[e0, fi + 1] = 0.5 * (z_ifc[cell0, thin_level - 1] + z_ifc[cell0, thin_level + 2])

    flat_idx = np.full((nedges,), fi, dtype=np.int32)

    return e2c, z_me, z_mc, z_ifc, flat_idx, topography, nlev, hs, hs1, e0, cell0


@pytest.mark.level("unit")
def test_compute_zdiff_gradp_dispatch_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    nedges = 4
    ncells = 4
    nlev = 8
    hs = 0
    hs1 = 2

    topography = np.array([1000.0, 1000.0, 2000.0, 1500.0], dtype=np.float64)
    z_ifc = np.empty((ncells, nlev + 1), dtype=np.float64)
    for c in range(ncells):
        top = topography[c]
        for k in range(nlev + 1):
            z_ifc[c, k] = 30000.0 - k * (30000.0 - top) / nlev

    z_mc = 0.5 * (z_ifc[:, :-1] + z_ifc[:, 1:])
    c_lin_e = np.full((nedges, 2), 0.5, dtype=np.float64)
    e2c = np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=np.int64)
    z_me = np.sum(z_mc[e2c] * c_lin_e[:, :, None], axis=1)
    z_me[0, 2] = np.nan

    flat_idx = np.full((nedges,), 1, dtype=np.int32)

    monkeypatch.setenv("ICON4PY_VALIDATE_ZDIFF_GRADP", "1")
    with pytest.raises(ValueError):
        compute_zdiff_gradp_dispatch(
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

    monkeypatch.setenv("ICON4PY_VALIDATE_ZDIFF_GRADP", "0")
    zdiff_gradp, vertoffset_gradp = compute_zdiff_gradp_dispatch(
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
    assert zdiff_gradp.shape == (nedges, 2, nlev)
    assert vertoffset_gradp.shape == (nedges, 2, nlev)
    assert np.all(np.isfinite(vertoffset_gradp))


def _build_chunking_test_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
]:
    nedges = 20
    ncells = 16
    nlev = 8
    hs = 4
    hs1 = 10

    topography = np.linspace(0.0, 3000.0, ncells).astype(np.float64)
    z_ifc = np.empty((ncells, nlev + 1), dtype=np.float64)
    for c in range(ncells):
        top = topography[c]
        for k in range(nlev + 1):
            z_ifc[c, k] = 30000.0 - k * (30000.0 - top) / nlev

    z_mc = 0.5 * (z_ifc[:, :-1] + z_ifc[:, 1:])
    c_lin_e = np.full((nedges, 2), 0.5, dtype=np.float64)
    c0 = np.arange(nedges) % ncells
    c1 = (c0 + 1) % ncells
    e2c = np.stack([c0, c1], axis=1)
    z_me = np.sum(z_mc[e2c] * c_lin_e[:, :, None], axis=1)
    flat_idx = np.full((nedges,), 2, dtype=np.int32)
    return e2c, z_me, z_mc, z_ifc, flat_idx, topography, nlev, hs, hs1


def _build_chunking_carry_test_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
    int,
    int,
]:
    nedges = 20
    ncells = 16
    nlev = 8
    hs = 4
    hs1 = 10
    e0 = 10
    cell0 = 0
    fi = 2

    topography = np.linspace(0.0, 3000.0, ncells).astype(np.float64)
    z_ifc = np.empty((ncells, nlev + 1), dtype=np.float64)
    for c in range(ncells):
        top = topography[c]
        for k in range(nlev + 1):
            z_ifc[c, k] = 30000.0 - k * (30000.0 - top) / nlev

    z_mc = 0.5 * (z_ifc[:, :-1] + z_ifc[:, 1:])
    c_lin_e = np.full((nedges, 2), 0.5, dtype=np.float64)
    c0 = np.arange(nedges) % ncells
    c1 = (c0 + 1) % ncells
    e2c = np.stack([c0, c1], axis=1)
    z_me = np.sum(z_mc[e2c] * c_lin_e[:, :, None], axis=1)
    flat_idx = np.full((nedges,), fi, dtype=np.int32)

    # Force strictly increasing z_me over the active segment on e0.
    z_me[e0, fi + 1] = 0.5 * (z_ifc[cell0, fi + 1] + z_ifc[cell0, fi + 2])
    z_me[e0, fi + 2] = 0.5 * (z_ifc[cell0, fi] + z_ifc[cell0, fi + 1])
    z_me[e0, fi + 3] = 0.5 * (z_ifc[cell0, fi - 1] + z_ifc[cell0, fi])
    z_me[e0, fi + 4] = 0.5 * (z_ifc[cell0, fi - 2] + z_ifc[cell0, fi - 1])
    z_me[e0, nlev - 1] = z_ifc[cell0, 0] - 100.0

    return e2c, z_me, z_mc, z_ifc, flat_idx, topography, nlev, hs, hs1, e0, cell0


@pytest.mark.level("unit")
def test_compute_zdiff_gradp_exact_v2_chunking_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ICON4PY_VALIDATE_ZDIFF_GRADP", "1")
    e2c, z_me, z_mc, z_ifc, flat_idx, topography, nlev, hs, hs1 = _build_chunking_test_inputs()

    uncapped_zdiff, uncapped_vert = compute_zdiff_gradp_exact_v2(
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
    assert _zdiff_mod._LAST_EXACT_V2_PATH == "fast"

    monkeypatch.setattr(_zdiff_mod, "_EXACT_V2_MAX_TABLE_BYTES", 1000)
    capped_zdiff, capped_vert = compute_zdiff_gradp_exact_v2(
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
    assert _zdiff_mod._LAST_EXACT_V2_PATH == "fast"

    assert np.allclose(capped_zdiff, uncapped_zdiff)
    assert np.array_equal(capped_vert, uncapped_vert)


@pytest.mark.level("unit")
def test_compute_zdiff_gradp_exact_v2_chunking_carry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ICON4PY_VALIDATE_ZDIFF_GRADP", "1")
    e2c, z_me, z_mc, z_ifc, flat_idx, topography, nlev, hs, hs1, _e0, _cell0 = (
        _build_chunking_carry_test_inputs()
    )

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

    uncapped_zdiff, uncapped_vert = compute_zdiff_gradp_exact_v2(
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
    assert _zdiff_mod._LAST_EXACT_V2_PATH == "carry"
    assert np.allclose(uncapped_zdiff, golden_zdiff)
    assert np.array_equal(uncapped_vert, golden_vert)

    monkeypatch.setattr(_zdiff_mod, "_EXACT_V2_MAX_TABLE_BYTES", 1000)
    capped_zdiff, capped_vert = compute_zdiff_gradp_exact_v2(
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
    assert _zdiff_mod._LAST_EXACT_V2_PATH == "carry"
    assert np.allclose(capped_zdiff, golden_zdiff)
    assert np.array_equal(capped_vert, uncapped_vert)


@pytest.mark.level("unit")
def test_first_match_scan_reference_boundary_cases() -> None:
    """Reference scan matches exact_v2 fresh-gather semantics on edge cases."""
    nedges = 8
    ncells = 8
    nlev = 16

    # Hand-built decreasing columns; topography is irrelevant for the scan.
    topography = np.linspace(0.0, 3000.0, ncells).astype(np.float64)
    z_ifc = np.empty((ncells, nlev + 1), dtype=np.float64)
    for c in range(ncells):
        top = topography[c]
        for k in range(nlev + 1):
            z_ifc[c, k] = 30000.0 - k * (30000.0 - top) / nlev

    # Perturb one column to create a zero-thickness level (E1 violation) and
    # interior ties; the reference scan must still match the fresh gather.
    z_ifc[0, nlev // 2 + 1] = z_ifc[0, nlev // 2]

    c_lin_e = np.full((nedges, 2), 0.5, dtype=np.float64)
    c0 = np.arange(nedges) % ncells
    c1 = (c0 + 1) % ncells
    e2c = np.stack([c0, c1], axis=1)
    z_mc = 0.5 * (z_ifc[:, :-1] + z_ifc[:, 1:])
    z_me = np.sum(z_mc[e2c] * c_lin_e[:, :, None], axis=1)

    # Boundary-case flat_idx values: 0, near nlev-2, and interior values.
    flat_idx = np.array(
        [0, 0, nlev - 2, nlev - 2, nlev // 2, nlev // 2, 1, nlev - 3],
        dtype=np.int32,
    )

    # Make one query fall below every real bracket so the result is nlev-1.
    z_me[0, 0] = z_ifc[e2c[0, 0], -1] - 100.0
    # Make one query equal an interior interface (tie case).
    z_me[2, 0] = z_ifc[e2c[2, 0], nlev // 2]

    fi = flat_idx.astype(np.int64)
    z_ifc_e0 = z_ifc[e2c[:, 0], :]
    z_ifc_e1 = z_ifc[e2c[:, 1], :]

    # Phase-1 style query (nedges, nlev).
    succ0 = _exact_query_succ(z_ifc_e0, z_me, np)
    jk1_fresh = _exact_phase1_cell0(succ0, fi, np)
    jk1_ref = _first_match_scan_reference(z_ifc_e0, z_me, fi)
    np.testing.assert_array_equal(jk1_ref, jk1_fresh)

    # Phase-2 style query (nedges, 1).
    z_aux2 = np.linspace(100.0, 500.0, nedges).astype(np.float64)
    z_aux2_v = z_aux2[:, None]
    succ2 = _exact_query_succ(z_ifc_e1, z_aux2_v, np)
    jk1_aux_fresh = _exact_phase1_cell0(succ2, fi, np)
    jk1_aux_ref = _first_match_scan_reference(z_ifc_e1, z_aux2_v, fi)
    np.testing.assert_array_equal(jk1_aux_ref, jk1_aux_fresh)

    # The reference scan also produces a value for inactive jk<=fi rows;
    # those rows are discarded by valid_jk in assembly, but they must still
    # match the fresh-gather value.
    for e in range(nedges):
        for q in range(nlev):
            if q <= flat_idx[e]:
                assert jk1_ref[e, q] == jk1_fresh[e, q]
