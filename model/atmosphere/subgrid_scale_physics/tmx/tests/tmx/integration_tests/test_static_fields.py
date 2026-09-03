# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Verify that the factory-built tmx static states match the serialized ICON ones.

The metric and interpolation fields tmx needs are registered in the common
field factories, so the states can be built with
``TmxMetricState.from_sources`` / ``TmxInterpolationState.from_sources``
instead of being read from the tmx-init savepoint. This test pins the two
paths against each other.

Needs a compiled backend: the embedded backend fails on the ``concat_where``
domain inference in ``compute_ddqz_z_half``::

    uv run --group test --frozen pytest --datatest-only --backend=gtfn_cpu \
        model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/integration_tests/test_static_fields.py
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states
from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import (
    geometry as grid_geometry,
    geometry_attributes,
    geometry_config,
    gridfile,
    vertical as v_grid,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.testing import definitions, test_utils
from icon4py.model.testing.fixtures.datatest import topography_savepoint

from ..fixtures import *  # noqa: F403
from .utils import construct_interpolation_state, construct_metric_state


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid_
    from icon4py.model.testing import serialbox as sb


# The vertex-RBF solve requires the missing V2E neighbours of pentagon vertices to
# be negative; the shared fixture keeps ICON's repeated-index padding, which makes
# those twelve interpolation matrices singular. Mirror the driver's grid-file path.
@pytest.fixture  # type: ignore[no-redef]
def icon_grid(grid_savepoint, backend):
    return grid_savepoint.construct_icon_grid(
        backend=backend, keep_skip_values=True, with_repeated_index=False
    )


# Largest deviation measured against the v08 archive, rounded up to one
# significant digit. Fields not listed here are reproduced bit for bit.
# The rbf_coeff_* floors are the widest: icon4py's batched-LU solve differs from
# ICON's Cholesky at round-off (same effect as the sanctioned tolerances in
# common's test_rbf_interpolation.py).
_ATOL = {
    "inv_ddqz_z_full_e": 5.0e-18,
    "inv_ddqz_z_half_e": 5.0e-18,
    "inv_ddqz_z_half_v": 1.0e-17,
    "wgtfac_e": 2.0e-16,
    "wgtfacq_e": 4.0e-16,
    "wgtfacq1_e": 4.0e-16,
    "geopot_agl_ifc": 5.0e-11,
    "c_lin_e": 2.0e-16,
    "e_bln_c_s": 6.0e-14,
    "cells_aw_verts": 2.0e-16,
    "rbf_coeff_v1": 3.0e-10,
    "rbf_coeff_v2": 3.0e-10,
    "rbf_coeff_e": 5.0e-13,
    "rbf_coeff_c1": 5.0e-9,
    "rbf_coeff_c2": 5.0e-9,
}


@pytest.mark.datatest
# through 'compute_ddqz_z_half', as in common's 'test_factory_ddqz_z_half'
@pytest.mark.uses_concat_where
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.EXCLAIM_APE_AES])
def test_factory_static_states_match_savepoints(
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    topography_savepoint: sb.TopographySavepoint,
    icon_grid: icon_grid_.IconGrid,
    backend: gtx_typing.Backend | None,
    experiment: definitions.Experiment,
    process_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
) -> None:
    allocator = model_backends.get_allocator(backend)

    init_savepoint = data_provider.from_savepoint_tmx_init()
    metric_ref = construct_metric_state(
        metrics_savepoint=metrics_savepoint,
        init_savepoint=init_savepoint,
        grid_savepoint=grid_savepoint,
        allocator=allocator,
    )
    interp_ref = construct_interpolation_state(interpolation_savepoint)

    geometry_source = grid_geometry.GridGeometry(
        grid=icon_grid,
        decomposition_info=decomposition_info,
        backend=backend,
        coordinates=grid_savepoint.coordinates(),
        extra_fields={
            gridfile.GeometryName.CELL_AREA: grid_savepoint.cell_areas(),
            gridfile.GeometryName.EDGE_LENGTH: grid_savepoint.primal_edge_length(),
            gridfile.GeometryName.DUAL_EDGE_LENGTH: grid_savepoint.dual_edge_length(),
            gridfile.GeometryName.EDGE_CELL_DISTANCE: grid_savepoint.edge_cell_length(),
            gridfile.GeometryName.EDGE_VERTEX_DISTANCE: grid_savepoint.edge_vert_length(),
            gridfile.GeometryName.DUAL_AREA: grid_savepoint.vertex_dual_area(),
            gridfile.GeometryName.TANGENT_ORIENTATION: grid_savepoint.tangent_orientation(),
            gridfile.GeometryName.CELL_NORMAL_ORIENTATION: grid_savepoint.edge_orientation(),
            gridfile.GeometryName.EDGE_ORIENTATION_ON_VERTEX: grid_savepoint.vertex_edge_orientation(),
        },
        metadata=geometry_attributes.attrs,
        config=geometry_config.GeometryConfig(),
        process_props=process_props,
    )

    interpolation_source = interpolation_factory.InterpolationFieldsFactory(
        config=experiment.config.interpolation,
        grid=icon_grid,
        decomposition_info=decomposition_info,
        geometry_source=geometry_source,
        backend=backend,
        metadata=interpolation_attributes.attrs,
        process_props=process_props,
    )

    metrics_source = metrics_factory.MetricsFieldsFactory(
        grid=icon_grid,
        vertical_grid=v_grid.VerticalGrid(
            experiment.config.vertical_grid, grid_savepoint.vct_a(), grid_savepoint.vct_b()
        ),
        decomposition_info=decomposition_info,
        geometry_source=geometry_source,
        topography=topography_savepoint.topo_c(),
        interpolation_source=interpolation_source,
        backend=backend,
        metadata=metrics_attributes.attrs,
        config=experiment.config.metrics,
        process_props=process_props,
    )

    metric_actual = tmx_states.TmxMetricState.from_sources(
        metrics=metrics_source, geometry=geometry_source
    )
    interp_actual = tmx_states.TmxInterpolationState.from_sources(
        interpolation=interpolation_source
    )

    for state_actual, state_ref in ((metric_actual, metric_ref), (interp_actual, interp_ref)):
        for field in dataclasses.fields(state_ref):
            test_utils.assert_dallclose(
                getattr(state_actual, field.name).asnumpy(),
                getattr(state_ref, field.name).asnumpy(),
                rtol=1.0e-12,
                atol=_ATOL.get(field.name, 0.0),
                err_msg=field.name,
            )
