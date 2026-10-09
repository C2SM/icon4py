# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The static states of the tmx component built from the field factories match the savepoints."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import static_fields
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.testing import definitions, grid_utils, test_utils

from ..fixtures import *  # noqa: F403
from .utils import construct_interpolation_state, construct_metric_state


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.testing import serialbox as sb


# The interpolation factory solves the RBF coefficients by a batched LU, ICON by Cholesky;
# the differences measured in common/tests/common/interpolation (RBF_TOLERANCES, the R02B04
# grid of the aquaplanet experiments).
_RBF_ATOL = {
    "rbf_coeff_c1": 5e-9,  # 3.25e-9 measured on gtfn_cpu
    "rbf_coeff_c2": 3.1e-9,
    "rbf_coeff_e": 8.0e-14,
    "rbf_coeff_v1": 3.0e-10,
    "rbf_coeff_v2": 3.0e-10,
}
# measured in common/tests/common/metrics/test_metrics_factory.py
_METRIC_ATOL = {"wgtfacq1_e": 4.0e-16, "geopot_agl_ifc": 5.0e-11}


def _construct_field_sources(
    *,
    backend: gtx_typing.Backend | None,
    experiment: definitions.Experiment,
    grid_savepoint: sb.IconGridSavepoint,
    topography_savepoint: sb.TopographySavepoint,
    process_props: decomposition.ProcessProperties,
) -> tuple[interpolation_factory.InterpolationFieldsFactory, metrics_factory.MetricsFieldsFactory]:
    """The field factories as the driver builds them, from the experiment's grid file."""
    geometry = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    interpolation_source = interpolation_factory.InterpolationFieldsFactory(
        grid=geometry.grid,
        decomposition_info=geometry._decomposition_info,
        geometry_source=geometry,
        backend=backend,
        config=experiment.config.interpolation,
        metadata=interpolation_attributes.attrs,
        process_props=process_props,
    )
    metrics_source = metrics_factory.MetricsFieldsFactory(
        grid=geometry.grid,
        vertical_grid=v_grid.VerticalGrid(
            experiment.config.vertical_grid, grid_savepoint.vct_a(), grid_savepoint.vct_b()
        ),
        decomposition_info=geometry._decomposition_info,
        geometry_source=geometry,
        topography=topography_savepoint.topo_c(),
        interpolation_source=interpolation_source,
        config=experiment.config.metrics,
        backend=backend,
        metadata=metrics_attributes.attrs,
        process_props=process_props,
    )
    return interpolation_source, metrics_source


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.EXCLAIM_APE_AES])
def test_static_states_from_field_sources_match_savepoints(
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    topography_savepoint: sb.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
) -> None:
    interpolation_source, metrics_source = _construct_field_sources(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
        process_props=decomposition.SingleNodeProcessProperties(),
    )
    metric_state = static_fields.build_metric_state(metrics_source)
    interpolation_state = static_fields.build_interpolation_state(interpolation_source)

    metric_state_ref = construct_metric_state(
        metrics_savepoint=metrics_savepoint,
        init_savepoint=data_provider.from_savepoint_tmx_init(),
        allocator=model_backends.get_allocator(backend),
    )
    interpolation_state_ref = construct_interpolation_state(interpolation_savepoint)

    for name, reference in _leaves(metric_state_ref):
        field = getattr(metric_state, name)
        assert field.domain == reference.domain, name
        test_utils.assert_dallclose(
            field.asnumpy(), reference.asnumpy(), atol=_METRIC_ATOL.get(name, 0.0), err_msg=name
        )
    for name, reference in _leaves(interpolation_state_ref):
        field = getattr(interpolation_state, name)
        assert field.domain.dims == reference.domain.dims, name
        test_utils.assert_dallclose(
            field.asnumpy(), reference.asnumpy(), atol=_RBF_ATOL.get(name, 0.0), err_msg=name
        )


def _leaves(state: object) -> list[tuple[str, object]]:
    return [(f.name, getattr(state, f.name)) for f in dataclasses.fields(state)]
