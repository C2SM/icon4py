# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Verify the v2 diffusion bindings against serialized ICON reference data.

Drives the real v2 path: build the static-field factories from the raw grid geometry
(`gm.coordinates` / `gm.geometry_fields` -- what ICON would pass), inject the serialized
RBF coefficients and mean_cell_area, assemble the diffusion granule states, and compare
them to the serialized ICON interpolation/metrics savepoints.
"""

from __future__ import annotations

import pytest

from icon4py.bindings.v2 import diffusion_setup, factory_setup
from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import geometry_attributes, vertical as v_grid
from icon4py.model.common.interpolation import interpolation_attributes
from icon4py.model.common.states import factory as states_factory
from icon4py.model.testing import grid_utils as gridtest_utils, test_utils as test_helpers
from icon4py.model.testing.fixtures.datatest import (  # noqa: F401 [needed for fixture resolution]
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
    topography_savepoint,
)


_sources_cache: dict[str, factory_setup.StaticFieldSources] = {}


def _get_sources(
    backend, experiment, grid_savepoint, interpolation_savepoint, topography_savepoint
) -> factory_setup.StaticFieldSources:
    key = experiment.name + "_" + (backend.name if backend is not None else "embedded")
    sources = _sources_cache.get(key)
    if sources is None:
        gm = gridtest_utils.get_grid_manager_from_experiment(
            experiment,
            keep_skip_values=True,
            allocator=model_backends.get_allocator(backend),
        )
        vertical_grid = v_grid.VerticalGrid(
            experiment.config.vertical_grid, grid_savepoint.vct_a(), grid_savepoint.vct_b()
        )
        sources = factory_setup.build_static_field_sources(
            grid=gm.grid,
            decomposition_info=gm.decomposition_info,
            coordinates=gm.coordinates,
            extra_fields=gm.geometry_fields,
            vertical_grid=vertical_grid,
            topography=topography_savepoint.topo_c(),
            interpolation_config=experiment.config.interpolation,
            metrics_config=experiment.config.metrics,
            rbf_vec_coeff_v1=interpolation_savepoint.rbf_vec_coeff_v1(),
            rbf_vec_coeff_v2=interpolation_savepoint.rbf_vec_coeff_v2(),
            mean_cell_area=float(grid_savepoint.mean_cell_area()),
            backend=backend,
            exchange=decomposition.single_node_exchange,
            reductions=decomposition.single_node_reductions,
        )
        _sources_cache[key] = sources
    return sources


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_injected_rbf_and_mean_cell_area(
    backend, experiment, grid_savepoint, interpolation_savepoint, topography_savepoint
) -> None:
    """The Fortran-passed RBF coeffs and mean_cell_area must be used verbatim (atol=0)."""
    sources = _get_sources(
        backend, experiment, grid_savepoint, interpolation_savepoint, topography_savepoint
    )
    rbf_v1_ref = interpolation_savepoint.rbf_vec_coeff_v1().asnumpy()
    rbf_v2_ref = interpolation_savepoint.rbf_vec_coeff_v2().asnumpy()
    assert test_helpers.dallclose(
        sources.interpolation.get(interpolation_attributes.RBF_VEC_COEFF_V1).asnumpy(),
        rbf_v1_ref,
        rtol=0.0,
        atol=0.0,
    )
    assert test_helpers.dallclose(
        sources.interpolation.get(interpolation_attributes.RBF_VEC_COEFF_V2).asnumpy(),
        rbf_v2_ref,
        rtol=0.0,
        atol=0.0,
    )
    injected = float(grid_savepoint.mean_cell_area())
    used = sources.geometry.get(
        geometry_attributes.MEAN_CELL_AREA, states_factory.RetrievalType.SCALAR
    )
    assert float(used) == injected


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_diffusion_metric_state_matches_icon(
    backend, experiment, grid_savepoint, interpolation_savepoint, metrics_savepoint, topography_savepoint
) -> None:
    sources = _get_sources(
        backend, experiment, grid_savepoint, interpolation_savepoint, topography_savepoint
    )
    state = diffusion_setup.assemble_diffusion_metric_state(sources.metrics)

    assert test_helpers.dallclose(
        state.theta_ref_mc.asnumpy(), metrics_savepoint.theta_ref_mc().asnumpy(), atol=1e-9
    )
    assert test_helpers.dallclose(
        state.wgtfac_c.asnumpy(), metrics_savepoint.wgtfac_c().asnumpy(), rtol=1e-9
    )
    assert test_helpers.dallclose(
        state.zd_diffcoef.asnumpy(), metrics_savepoint.zd_diffcoef().asnumpy(), atol=1e-10
    )
    assert test_helpers.dallclose(
        state.zd_intcoef.asnumpy(), metrics_savepoint.zd_intcoef().asnumpy(), atol=1e-8
    )
    assert test_helpers.dallclose(
        state.zd_vertoffset.asnumpy(), metrics_savepoint.zd_vertoffset().asnumpy()
    )


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_diffusion_interpolation_state_matches_icon(
    backend, experiment, grid_savepoint, interpolation_savepoint, topography_savepoint
) -> None:
    sources = _get_sources(
        backend, experiment, grid_savepoint, interpolation_savepoint, topography_savepoint
    )
    state = diffusion_setup.assemble_diffusion_interpolation_state(sources.interpolation)
    geofac_grg_x_ref, geofac_grg_y_ref = interpolation_savepoint.geofac_grg()

    assert test_helpers.dallclose(
        state.e_bln_c_s.asnumpy(), interpolation_savepoint.e_bln_c_s().asnumpy(), rtol=1e-10
    )
    assert test_helpers.dallclose(
        state.geofac_div.asnumpy(), interpolation_savepoint.geofac_div().asnumpy()
    )
    assert test_helpers.dallclose(
        state.geofac_n2s.asnumpy(), interpolation_savepoint.geofac_n2s().asnumpy()
    )
    assert test_helpers.dallclose(
        state.geofac_grg_x.asnumpy(), geofac_grg_x_ref.asnumpy(), rtol=1e-11, atol=1.1e-16
    )
    assert test_helpers.dallclose(
        state.geofac_grg_y.asnumpy(), geofac_grg_y_ref.asnumpy(), rtol=1e-11, atol=1.1e-16
    )
    assert test_helpers.dallclose(
        state.nudgecoeff_e.asnumpy(), interpolation_savepoint.nudgecoeff_e().asnumpy()
    )
    # injected from Fortran -> must be bit-identical
    assert test_helpers.dallclose(
        state.rbf_coeff_1.asnumpy(),
        interpolation_savepoint.rbf_vec_coeff_v1().asnumpy(),
        rtol=0.0,
        atol=0.0,
    )
    assert test_helpers.dallclose(
        state.rbf_coeff_2.asnumpy(),
        interpolation_savepoint.rbf_vec_coeff_v2().asnumpy(),
        rtol=0.0,
        atol=0.0,
    )
