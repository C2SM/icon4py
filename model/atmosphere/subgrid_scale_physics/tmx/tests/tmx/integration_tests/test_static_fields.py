# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatest: verify that factory-built TmxMetricState and TmxInterpolationState
match the savepoint-built reference states field-by-field.

This test is parametrized over the EXCLAIM_APE_AES experiment and requires the
v06 serialized data archive to be downloaded.  Because the archive server
currently returns HTTP 403, this test *cannot be executed*; it is written and
will at least COLLECT (import errors are fixed, data-download failure at
fixture time is expected and acceptable).

Data-test can be unblocked once the v06 archive is accessible by running::

    uv run pytest model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/integration_tests/test_static_fields.py -v -m datatest
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

    for f in dataclasses.fields(interp_ref):
        actual = getattr(interp_actual, f.name)
        ref = getattr(interp_ref, f.name)
        assert_scaled_allclose(
            actual.asnumpy(),
            ref.asnumpy(),
            err_msg=f.name,
        )
