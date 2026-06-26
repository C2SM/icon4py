# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Verify the v2 solve-nonhydro (dycore) bindings against serialized ICON reference data.

Drives the real v2 path: build the static-field factories from the raw grid geometry
(`gm.coordinates` / `gm.geometry_fields` -- what ICON would pass), inject the serialized
RBF coefficients (v1, v2, e) and mean_cell_area, assemble the dycore granule states, and
compare them to the serialized ICON interpolation/metrics savepoints.
"""

from __future__ import annotations

import numpy as np
import pytest

from icon4py.bindings.v2 import dycore_setup, factory_setup
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import geometry_attributes, horizontal as h_grid, vertical as v_grid
from icon4py.model.common.states import factory as states_factory
from icon4py.model.testing import grid_utils as gridtest_utils, test_utils as test_helpers
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
            rbf_vec_coeff_e=interpolation_savepoint.rbf_vec_coeff_e(),
            mean_cell_area=float(grid_savepoint.mean_cell_area()),
            backend=backend,
            exchange=decomposition.single_node_exchange,
            reductions=decomposition.single_node_reductions,
        )
        _sources_cache[key] = sources
    return sources


def _np(x):
    return x.asnumpy() if hasattr(x, "asnumpy") else np.asarray(x)


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_dycore_interpolation_state_matches_icon(
    backend, experiment, grid_savepoint, interpolation_savepoint, topography_savepoint
) -> None:
    sources = _get_sources(
        backend, experiment, grid_savepoint, interpolation_savepoint, topography_savepoint
    )
    state = dycore_setup.assemble_dycore_interpolation_state(sources.interpolation)
    sp = interpolation_savepoint

    # factory-derived fields: tolerances mirror the validated interpolation-factory tests.
    factory_checks = [
        ("c_lin_e", state.c_lin_e, sp.c_lin_e(), dict()),
        ("c_intp", state.c_intp, sp.c_intp(), dict()),
        ("e_flx_avg", state.e_flx_avg, sp.e_flx_avg(), dict(atol=1e-12)),
        ("geofac_grdiv", state.geofac_grdiv, sp.geofac_grdiv(), dict()),
        (
            "pos_on_tplane_e_1",
            state.pos_on_tplane_e_1,
            sp.pos_on_tplane_e_x(),
            dict(atol=1e-8, rtol=1e-9),
        ),
        (
            "pos_on_tplane_e_2",
            state.pos_on_tplane_e_2,
            sp.pos_on_tplane_e_y(),
            dict(atol=1e-8, rtol=1e-9),
        ),
        ("e_bln_c_s", state.e_bln_c_s, sp.e_bln_c_s(), dict(rtol=1e-10)),
        ("geofac_div", state.geofac_div, sp.geofac_div(), dict()),
        ("geofac_n2s", state.geofac_n2s, sp.geofac_n2s(), dict()),
        ("nudgecoeff_e", state.nudgecoeff_e, sp.nudgecoeff_e(), dict()),
    ]
    for name, got, ref, tol in factory_checks:
        assert test_helpers.dallclose(_np(got), _np(ref), **tol), f"interpolation field {name}"

    # geofac_rot is only initialized from the second lateral-boundary level inward in ICON;
    # the factory computes it everywhere, so compare only the interior (cf. test_get_geofac_rot).
    grid = sources.interpolation.grid
    vertex_start = grid.start_index(
        h_grid.domain(dims.VertexDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    assert test_helpers.dallclose(
        _np(state.geofac_rot)[vertex_start:, :], _np(sp.geofac_rot())[vertex_start:, :]
    )

    geofac_grg_x_ref, geofac_grg_y_ref = sp.geofac_grg()
    assert test_helpers.dallclose(
        _np(state.geofac_grg_x), _np(geofac_grg_x_ref), rtol=1e-11, atol=1.1e-16
    )
    assert test_helpers.dallclose(
        _np(state.geofac_grg_y), _np(geofac_grg_y_ref), rtol=1e-11, atol=1.1e-16
    )

    # injected from Fortran -> must be bit-identical
    assert test_helpers.dallclose(
        _np(state.rbf_coeff_1), _np(sp.rbf_vec_coeff_v1()), rtol=0.0, atol=0.0
    )
    assert test_helpers.dallclose(
        _np(state.rbf_coeff_2), _np(sp.rbf_vec_coeff_v2()), rtol=0.0, atol=0.0
    )
    assert test_helpers.dallclose(
        _np(state.rbf_vec_coeff_e), _np(sp.rbf_vec_coeff_e()), rtol=0.0, atol=0.0
    )


@pytest.mark.level("integration")
@pytest.mark.uses_concat_where  # exner_exfac et al. use concat_where (xfails on embedded backend)
@pytest.mark.datatest
def test_dycore_metric_state_matches_icon(  # noqa: PLR0917 [too-many-positional-arguments]
    backend,
    experiment,
    grid_savepoint,
    interpolation_savepoint,
    metrics_savepoint,
    topography_savepoint,
) -> None:
    sources = _get_sources(
        backend, experiment, grid_savepoint, interpolation_savepoint, topography_savepoint
    )
    state = dycore_setup.assemble_dycore_metric_state(sources.metrics)
    ms = metrics_savepoint

    # (assembled field, ICON reference, tolerances). Defaults catch wiring mistakes
    # (a misrouted field is off by orders of magnitude); the few loose entries match the
    # corresponding test_metrics_factory.py tolerances.
    checks = [
        (state.rayleigh_w, ms.rayleigh_w(), dict(rtol=1e-8)),
        (state.time_extrapolation_parameter_for_exner, ms.exner_exfac(), dict(atol=1e-8)),
        (state.reference_exner_at_cells_on_model_levels, ms.exner_ref_mc(), dict(rtol=1e-8)),
        (state.wgtfac_c, ms.wgtfac_c(), dict(rtol=1e-9)),
        (state.wgtfacq_c, ms.wgtfacq_c(), dict(rtol=1e-8)),
        (state.inv_ddqz_z_full, ms.inv_ddqz_z_full(), dict(rtol=1e-8)),
        (state.reference_rho_at_cells_on_model_levels, ms.rho_ref_mc(), dict(rtol=1e-8)),
        (state.reference_theta_at_cells_on_model_levels, ms.theta_ref_mc(), dict(atol=1e-9)),
        (state.exner_w_explicit_weight_parameter, ms.vwind_expl_wgt(), dict(rtol=1e-9)),
        (
            state.ddz_of_reference_exner_at_cells_on_half_levels,
            ms.d_exner_dz_ref_ic(),
            dict(atol=1e-9),
        ),
        (state.ddqz_z_half, ms.ddqz_z_half(), dict(rtol=1e-8)),
        (state.reference_theta_at_cells_on_half_levels, ms.theta_ref_ic(), dict(atol=1e-9)),
        (state.d2dexdz2_fac1_mc, ms.d2dexdz2_fac1_mc(), dict(atol=1e-11)),
        (state.d2dexdz2_fac2_mc, ms.d2dexdz2_fac2_mc(), dict(atol=1e-11)),
        (state.reference_rho_at_edges_on_model_levels, ms.rho_ref_me(), dict(rtol=1e-8)),
        (state.reference_theta_at_edges_on_model_levels, ms.theta_ref_me(), dict(rtol=1e-8)),
        (state.ddxn_z_full, ms.ddxn_z_full(), dict(atol=1e-8)),
        (state.ddxt_z_full, ms.ddxt_z_full(), dict(rtol=1.0e-5, atol=1.0e-8)),
        (state.pg_exdist, ms.pg_exdist_dsl(), dict(atol=1.0e-5)),
        (state.ddqz_z_full_e, ms.ddqz_z_full_e(), dict(rtol=1e-8)),
        (state.wgtfac_e, ms.wgtfac_e(), dict(rtol=1e-9)),
        (state.wgtfacq_e, ms.wgtfacq_e(), dict(rtol=1e-8)),
        (state.exner_w_implicit_weight_parameter, ms.vwind_impl_wgt(), dict(rtol=1e-9)),
        (state.horizontal_mask_for_3d_divdamp, ms.hmask_dd3d(), dict(rtol=1e-9)),
        (state.scaling_factor_for_3d_divdamp, ms.scalfac_dd3d(), dict(rtol=1e-9)),
        (state.coeff1_dwdz, ms.coeff1_dwdz(), dict(rtol=1e-8)),
        (state.coeff2_dwdz, ms.coeff2_dwdz(), dict(rtol=1e-8)),
        (state.coeff_gradekin, ms.coeff_gradekin(), dict(rtol=1e-8)),
    ]
    for got, ref, tol in checks:
        assert test_helpers.dallclose(_np(got), _np(ref), **tol)

    # zdiff_gradp / vertoffset_gradp: ICON leaves them uninitialized below the second lateral
    # boundary level, so compare only the interior (cf. test_factory_zdiff_gradp).
    grid = sources.metrics._grid
    edge_start = grid.start_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    assert test_helpers.dallclose(
        _np(state.zdiff_gradp)[edge_start:],
        _np(ms.zdiff_gradp())[edge_start:],
        atol=1e-10,
        rtol=1e-9,
    )
    assert np.array_equal(
        _np(state.vertoffset_gradp)[edge_start:], _np(ms.vertoffset_gradp())[edge_start:]
    )

    # bool mask and integer scalar: exact
    assert np.array_equal(_np(state.mask_prog_halo_c), _np(ms.mask_prog_halo_c()))
    assert int(state.nflat_gradp) == int(grid_savepoint.nflat_gradp())


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_injected_mean_cell_area_used(
    backend, experiment, grid_savepoint, interpolation_savepoint, topography_savepoint
) -> None:
    sources = _get_sources(
        backend, experiment, grid_savepoint, interpolation_savepoint, topography_savepoint
    )
    injected = float(grid_savepoint.mean_cell_area())
    used = sources.geometry.get(
        geometry_attributes.MEAN_CELL_AREA, states_factory.RetrievalType.SCALAR
    )
    assert float(used) == injected
