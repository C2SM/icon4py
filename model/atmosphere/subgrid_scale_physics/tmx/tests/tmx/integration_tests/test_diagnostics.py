# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test of the tmx diagnostics component (Smagorinsky).

Constructs the component from the serialized ICON state (exp.exclaim_ape_aesPhys),
verifies the init fields against the tmx-init savepoint and one call of
``run`` against the tmx-diagnostics-exit savepoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import diagnostics, tmx_states
from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.testing import definitions, test_utils

from ..fixtures import *  # noqa: F403
from .utils import construct_input_state, construct_interpolation_state, construct_metric_state


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid_
    from icon4py.model.testing import serialbox as sb


@pytest.mark.datatest
@pytest.mark.single_precision_ready
@pytest.mark.parametrize(
    "experiment_description, date",
    [
        (definitions.Experiments.EXCLAIM_APE_AES, date)
        for date in definitions.Experiments.EXCLAIM_APE_AES.dates[1:]
    ],
)
def test_tmx_init_and_run_diagnostics_single_step(
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    backend: gtx_typing.Backend | None,
    date: str,
    experiment: definitions.Experiment,
) -> None:
    tmx_config = experiment.config.tmx
    assert tmx_config is not None
    allocator = model_backends.get_allocator(backend)
    init_savepoint = data_provider.from_savepoint_tmx_init()
    entry_savepoint = data_provider.from_savepoint_tmx_entry(date=date)
    exit_savepoint = data_provider.from_savepoint_tmx_diagnostics_exit(date=date)

    metric_state = construct_metric_state(
        metrics_savepoint=metrics_savepoint,
        init_savepoint=init_savepoint,
        allocator=allocator,
    )
    component = diagnostics.Diagnostics(
        grid=icon_grid,
        metric_state=metric_state,
        interpolation_state=construct_interpolation_state(interpolation_savepoint),
        edge_params=grid_savepoint.construct_edge_geometry(),
        cell_params=grid_savepoint.construct_cell_geometry(),
        backend=backend,
        exchange=decomposition.SingleNodeExchange(),
        turb_prandtl=tmx_config.turb_prandtl,
        smag_constant=tmx_config.smag_constant,
        max_turb_scale=tmx_config.max_turb_scale,
        km_min=tmx_config.km_min,
        km_const=tmx_config.km_const,
        use_km_const=tmx_config.use_km_const,
        louis_constant_b=tmx_config.louis_constant_b,
        use_louis=tmx_config.use_louis,
        use_louis_land=tmx_config.use_louis_land,
        use_louis_ice=tmx_config.use_louis_ice,
    )

    # Smagorinsky_init runs in the constructor; 'ghf' is only serialized at diagnostics exit.
    # All three match exactly on every backend in double precision.
    for name, computed, reference in (
        ("mixing_length_sq", component.mixing_length_sq, init_savepoint.mix_len_sq()),
        (
            "scaling_factor_louis",
            component.scaling_factor_louis,
            init_savepoint.scaling_factor_louis(),
        ),
        ("height_above_ground", metric_state.height_above_ground, exit_savepoint.ghf()),
    ):
        if test_utils.wp_is_dp:
            np.testing.assert_array_equal(computed.asnumpy(), reference.asnumpy(), err_msg=name)
        else:
            test_utils.assert_dallclose(computed.asnumpy(), reference.asnumpy(), err_msg=name)

    diagnostic_state = tmx_states.TmxDiagnosticState.allocate(icon_grid, allocator=allocator)
    component.run(
        input_state=construct_input_state(entry_savepoint), diagnostic_state=diagnostic_state
    )

    nlev = icon_grid.num_levels
    # K rows are excluded only where the Fortran leaves them dead:
    # - bruvais: brunt_vaisala_freq (mo_nh_vert_interp_les.f90) computes
    #   jk = 2..nlev (1-based), i.e. rows 1..nlev-1; rows 0 and nlev are never
    #   written.
    # - mech_prod: interpolate_rate_of_strain_full2half_edge2cell
    #   (mo_vdf_atmo.f90) computes jk = 2..nlev (1-based), i.e. rows
    #   1..nlev-1; rows 0 and nlev are never written.
    # - rho_ic: NOT excluded; vert_intp_full2half_cell_3d
    #   (mo_nh_vert_interp_les.f90) writes all rows, including row 0
    #   (wgtfacq1_c extrapolation) and row nlev (wgtfacq_c extrapolation).
    interior = slice(1, nlev)
    everything = slice(None)
    # (attribute and savepoint accessor, K slice compared, atol, rtol), chosen as described in
    # `assert_tmx_exit_fields`
    fields = (
        # measured atol=5.8e-11, rtol=2.2e-16
        ("cptgz", everything, 0.0, 3.0e-16),
        # measured atol=1.1e-13, rtol=4.1e-16
        ("theta_v", everything, 0.0, 5.0e-16),
        # measured atol=4.4e-16, rtol=4.0e-16
        ("rho_ic", everything, 0.0, 5.0e-16),
        # measured atol=3.6e-17, rtol=8.5e-10
        ("bruvais", interior, 0.0, 1.0e-9),
        # measured atol=1.4e-14, rtol=1.1e-13
        ("vn", everything, 0.0, 2.0e-13),
        # measured atol=3.3e-16, rtol=6.7e-13
        ("w_vert", everything, 0.0, 8.0e-13),
        # measured atol=1.1e-16, rtol=9.0e-13
        ("w_ie", everything, 0.0, 1.0e-12),
        # measured atol=2.1e-14, rtol=8.3e-7
        ("u_vert", everything, 3.0e-14, 0.0),
        # measured atol=4.0e-15, rtol=2.0e-5
        ("v_vert", everything, 5.0e-15, 0.0),
        # measured atol=2.1e-14, rtol=3.9e-14
        ("vn_ie", everything, 0.0, 5.0e-14),
        # measured atol=1.4e-14, rtol=9.9e-11
        ("vt_ie", everything, 0.0, 2.0e-10),
        # measured atol=4.3e-18, rtol=4.3e-12
        ("shear", everything, 0.0, 5.0e-12),
        # measured atol=2.6e-19, rtol=2.3e-9
        ("div_of_stress", everything, 3.0e-19, 0.0),
        # measured atol=2.0e-19, rtol=7.9e-10
        ("div_c", everything, 0.0, 9.0e-10),
        # measured atol=1.7e-18, rtol=3.1e-13
        ("mech_prod", interior, 0.0, 4.0e-13),
        # measured atol=7.7e-11, rtol=3.9e-11
        ("km_ic", everything, 0.0, 5.0e-11),
        # measured atol=2.3e-10, rtol=3.9e-11
        ("kh_ic", everything, 0.0, 5.0e-11),
        # measured atol=3.8e-11, rtol=3.5e-11
        ("km_c", everything, 0.0, 4.0e-11),
        # measured atol=1.4e-11, rtol=5.1e-12
        ("km_iv", everything, 0.0, 6.0e-12),
        # measured atol=3.8e-11, rtol=1.9e-11
        ("km_ie", everything, 0.0, 3.0e-11),
    )
    # single precision (#970): one atol per field and the scaled shared rtol
    single_precision_atol = {
        "cptgz": 7.0e-11,
        "theta_v": 2.0e-13,
        "rho_ic": 6.0e-16,
        "bruvais": 5.0e-8,
        "vn": 1.5e-5,
        "w_vert": 2.0e-7,
        "w_ie": 2.0e-7,
        "u_vert": 1.5e-5,
        "v_vert": 4.0e-6,
        "vn_ie": 2.0e-5,
        "vt_ie": 1.5e-5,
        "shear": 3.0e-9,
        "div_of_stress": 2.0e-10,
        "div_c": 2.0e-10,
        "mech_prod": 2.0e-9,
        "km_ic": 0.1,
        "kh_ic": 0.3,
        "km_c": 0.05,
        "km_iv": 0.03,
        "km_ie": 0.06,
    }
    single_precision_rtol = test_utils.scale_tol(3.0e-12)
    for name, k_slice, atol, rtol in fields:
        test_utils.assert_dallclose(
            getattr(diagnostic_state, name).asnumpy()[:, k_slice],
            getattr(exit_savepoint, name)().asnumpy()[:, k_slice],
            atol=atol if test_utils.wp_is_dp else single_precision_atol[name],
            rtol=rtol if test_utils.wp_is_dp else single_precision_rtol,
            err_msg=name,
        )
