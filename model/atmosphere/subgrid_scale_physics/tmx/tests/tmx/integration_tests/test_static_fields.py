# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatest: verify that factory-built TmxMetricState and TmxInterpolationState
match the savepoint-built reference states field-by-field.

Parametrized over the EXCLAIM_APE_AES experiment (v08 archive, auto-downloaded).
Run with the compiled CPU backend — the embedded backend currently fails on
``concat_where`` domain inference in ``compute_ddqz_z_half``::

    uv run --group test --frozen pytest \
        model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/integration_tests/test_static_fields.py \
        --datatest-only --backend=gtfn_cpu -v
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import static_fields
from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import (
    geometry as grid_geometry,
    geometry_attributes,
    geometry_config as geometry_configuration,
    gridfile,
    vertical as v_grid,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.testing import definitions
from icon4py.model.testing.fixtures.datatest import topography_savepoint

from ..fixtures import *  # noqa: F403  (re-exports experiment, decomposition_info, etc.)
from .utils import assert_scaled_allclose, construct_interpolation_state, construct_metric_state


@pytest.fixture  # type: ignore[no-redef]  # deliberately shadows the fixtures.py import
def icon_grid(grid_savepoint, backend):
    """Grid with production skip-value semantics (invalid neighbors are negative).

    The vertex-RBF coefficient solve requires pentagon vertices' missing V2E
    neighbors to be negative (its documented contract); the shared fixture keeps
    ICON's repeated-index padding, which makes the twelve pentagon interpolation
    matrices singular. The driver's grid-file path (keep_skip_values=True,
    0-padding -> -1) meets the contract — mirror it here.
    """
    return grid_savepoint.construct_icon_grid(
        backend=backend, keep_skip_values=True, with_repeated_index=False
    )


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid_
    from icon4py.model.testing import serialbox as sb


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description",
    [definitions.Experiments.EXCLAIM_APE_AES],
)
def test_factory_static_states_match_savepoints(
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    backend: gtx_typing.Backend | None,
    experiment: definitions.Experiment,
    process_props: decomposition_defs.ProcessProperties,
    decomposition_info: decomposition_defs.DecompositionInfo,
    topography_savepoint: sb.TopographySavepoint,
) -> None:
    """Compare every field of the factory-built states against the savepoint reference.

    The factory-built states are constructed in the same way as
    ``driver_utils.create_static_field_factories`` would for a real run:
    a ``GridGeometry`` is built from the savepoint geometry fields,
    ``InterpolationFieldsFactory`` and ``MetricsFieldsFactory`` are
    constructed on top.
    """
    allocator = model_backends.get_allocator(backend)

    # ------------------------------------------------------------------
    # Reference states from the serialized ICON savepoints
    # ------------------------------------------------------------------
    init_savepoint = data_provider.from_savepoint_tmx_init()

    metric_ref = construct_metric_state(
        metrics_savepoint=metrics_savepoint,
        init_savepoint=init_savepoint,
        grid_savepoint=grid_savepoint,
        allocator=allocator,
    )
    interp_ref = construct_interpolation_state(interpolation_savepoint)

    # ------------------------------------------------------------------
    # Build the three factory sources from savepoint grid geometry data
    # ------------------------------------------------------------------
    exchange = decomposition_defs.create_exchange(process_props, decomposition_info)
    global_reductions = decomposition_defs.create_reduction(process_props, decomposition_info)

    extra_fields = {
        gridfile.GeometryName.CELL_AREA: grid_savepoint.cell_areas(),
        gridfile.GeometryName.EDGE_LENGTH: grid_savepoint.primal_edge_length(),
        gridfile.GeometryName.DUAL_EDGE_LENGTH: grid_savepoint.dual_edge_length(),
        gridfile.GeometryName.EDGE_CELL_DISTANCE: grid_savepoint.edge_cell_length(),
        gridfile.GeometryName.EDGE_VERTEX_DISTANCE: grid_savepoint.edge_vert_length(),
        gridfile.GeometryName.DUAL_AREA: grid_savepoint.vertex_dual_area(),
        gridfile.GeometryName.TANGENT_ORIENTATION: grid_savepoint.tangent_orientation(),
        gridfile.GeometryName.CELL_NORMAL_ORIENTATION: grid_savepoint.edge_orientation(),
        gridfile.GeometryName.EDGE_ORIENTATION_ON_VERTEX: grid_savepoint.vertex_edge_orientation(),
    }

    geometry_source = grid_geometry.GridGeometry(
        grid=icon_grid,
        decomposition_info=decomposition_info,
        backend=backend,
        coordinates=grid_savepoint.coordinates(),
        extra_fields=extra_fields,
        metadata=geometry_attributes.attrs,
        config=geometry_configuration.GeometryConfig(),
        process_props=process_props,
        exchange=exchange,
        global_reductions=global_reductions,
    )

    interpolation_source = interpolation_factory.InterpolationFieldsFactory(
        config=experiment.config.interpolation,
        grid=icon_grid,
        decomposition_info=decomposition_info,
        geometry_source=geometry_source,
        backend=backend,
        metadata=interpolation_attributes.attrs,
        exchange=exchange,
    )

    vertical_grid = v_grid.VerticalGrid(
        experiment.config.vertical_grid,
        grid_savepoint.vct_a(),
        grid_savepoint.vct_b(),
    )

    metrics_source = metrics_factory.MetricsFieldsFactory(
        grid=icon_grid,
        vertical_grid=vertical_grid,
        decomposition_info=decomposition_info,
        geometry_source=geometry_source,
        topography=topography_savepoint.topo_c(),
        interpolation_source=interpolation_source,
        backend=backend,
        metadata=metrics_attributes.attrs,
        config=experiment.config.metrics,
        exchange=exchange,
        global_reductions=global_reductions,
    )

    # ------------------------------------------------------------------
    # Build the states under test
    # ------------------------------------------------------------------
    metric_actual, interp_actual = static_fields.build_tmx_static_states(
        grid=icon_grid,
        geometry_source=geometry_source,
        interpolation_source=interpolation_source,
        metrics_source=metrics_source,
        backend=backend,
    )

    # ------------------------------------------------------------------
    # Field-by-field comparison
    # ------------------------------------------------------------------
    # Note: inv_ddqz_z_half_e and inv_ddqz_z_half_v are derived via numpy
    # weighted-averaging without an MPI halo exchange; comparison on halo
    # cells may fail in multi-rank runs.  In the single-rank (EXCLAIM_APE_AES)
    # test context this is not an issue.
    for f in dataclasses.fields(metric_ref):
        actual = getattr(metric_actual, f.name)
        ref = getattr(metric_ref, f.name)
        assert_scaled_allclose(
            actual.asnumpy(),
            ref.asnumpy(),
            err_msg=f.name,
        )

    # Solver-recomputed fields: icon4py's batched-LU RBF solve differs from ICON's
    # Cholesky at round-off (sanctioned in common test_rbf_interpolation.py's
    # RBF_TOLERANCES: cell atol 3.1e-9 on the aquaplanet grid; v08 measures
    # 3.25e-9 abs). Those fields get a wider absolute floor; every pass-through
    # or identically-derived field keeps the tight default.
    atol_scale_overrides = {
        "rbf_coeff_c1": 5.0e-9,
        "rbf_coeff_c2": 5.0e-9,
        "rbf_coeff_v1": 2.0e-9,
        "rbf_coeff_v2": 2.0e-9,
    }
    for f in dataclasses.fields(interp_ref):
        actual = getattr(interp_actual, f.name)
        ref = getattr(interp_ref, f.name)
        assert_scaled_allclose(
            actual.asnumpy(),
            ref.asnumpy(),
            atol_scale=atol_scale_overrides.get(f.name, 1.0e-9),
            err_msg=f.name,
        )
