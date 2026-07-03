# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test of the Tmx granule Stage A (Smagorinsky diagnostics).

Constructs the granule from the serialized ICON state (exp.exclaim_ape_aesPhys),
verifies the init fields against the tmx-init savepoint and one call of
``run_diagnostics`` against the tmx-diagnostics-exit savepoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx, tmx_states
from icon4py.model.common import model_backends
from icon4py.model.testing import definitions, test_utils

from ..fixtures import *  # noqa: F403
from .utils import (
    TMX_DATES,
    assert_scaled_allclose,
    construct_input_state,
    construct_interpolation_state,
    construct_metric_state,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid_
    from icon4py.model.testing import serialbox as sb


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, date",
    [(definitions.Experiments.EXCLAIM_APE_AES, date) for date in TMX_DATES],
)
def test_tmx_init_and_run_diagnostics_single_step(  # noqa: PLR0917 [too-many-positional-arguments]
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    backend: gtx_typing.Backend | None,
    date: str,
    tmx_config: tmx.TmxConfig,
) -> None:
    allocator = model_backends.get_allocator(backend)
    init_savepoint = data_provider.from_savepoint_tmx_init()
    entry_savepoint = data_provider.from_savepoint_tmx_entry(date=date)
    exit_savepoint = data_provider.from_savepoint_tmx_diagnostics_exit(date=date)

    params = tmx.TmxParams(tmx_config)

    granule = tmx.Tmx(
        grid=icon_grid,
        config=tmx_config,
        params=params,
        # the vertical grid is not used by the Smagorinsky diagnostics (Stage A)
        vertical_grid=None,
        metric_state=construct_metric_state(
            metrics_savepoint, init_savepoint, grid_savepoint, allocator
        ),
        interpolation_state=construct_interpolation_state(interpolation_savepoint),
        edge_params=grid_savepoint.construct_edge_geometry(),
        cell_params=grid_savepoint.construct_cell_geometry(),
        backend=backend,
    )

    # init fields, computed in the granule constructor (Smagorinsky_init in
    # mo_tmx_smagorinsky.f90; ghf is only serialized at diagnostics-exit)
    test_utils.assert_dallclose(
        granule.mix_len_sq.asnumpy(), init_savepoint.mix_len_sq().asnumpy(), err_msg="mix_len_sq"
    )
    test_utils.assert_dallclose(
        granule.louis_factor.asnumpy(),
        init_savepoint.scaling_factor_louis().asnumpy(),
        err_msg="louis_factor",
    )
    test_utils.assert_dallclose(
        granule.ghf.asnumpy(), exit_savepoint.ghf().asnumpy(), err_msg="ghf"
    )

    diagnostic_state = tmx_states.TmxDiagnosticState.allocate(icon_grid, allocator=allocator)
    granule.run_diagnostics(construct_input_state(entry_savepoint), diagnostic_state)

    nlev = icon_grid.num_levels
    #: (diagnostic state attribute, exit savepoint accessor, K slice compared)
    #: K rows are excluded only where the Fortran leaves them dead:
    #: - bruvais: brunt_vaisala_freq (mo_nh_vert_interp_les.f90) computes
    #:   jk = 2..nlev (1-based), i.e. rows 1..nlev-1; rows 0 and nlev are never
    #:   written.
    #: - mech_prod: interpolate_rate_of_strain_full2half_edge2cell
    #:   (mo_vdf_atmo.f90) computes jk = 2..nlev (1-based), i.e. rows
    #:   1..nlev-1; rows 0 and nlev are never written.
    #: - rho_ic: NOT excluded; vert_intp_full2half_cell_3d
    #:   (mo_nh_vert_interp_les.f90) writes all rows, including row 0
    #:   (wgtfacq1_c extrapolation) and row nlev (wgtfacq_c extrapolation).
    interior = slice(1, nlev)
    everything = slice(None)
    fields = (
        ("cptgz", "cptgz", everything),
        ("theta_v", "theta_v", everything),
        ("rho_ic", "rho_ic", everything),
        ("bruvais", "bruvais", interior),
        ("vn", "vn", everything),
        ("w_vert", "w_vert", everything),
        ("w_ie", "w_ie", everything),
        ("u_vert", "u_vert", everything),
        ("v_vert", "v_vert", everything),
        ("vn_ie", "vn_ie", everything),
        ("vt_ie", "vt_ie", everything),
        ("shear", "shear", everything),
        ("div_of_stress", "div_of_stress", everything),
        ("div_c", "div_c", everything),
        ("mech_prod", "mech_prod", interior),
        ("km_ic", "km_ic", everything),
        ("kh_ic", "kh_ic", everything),
        ("km_c", "km_c", everything),
        ("km_iv", "km_iv", everything),
        ("km_ie", "km_ie", everything),
    )
    for attr_name, accessor_name, k_slice in fields:
        actual = getattr(diagnostic_state, attr_name).asnumpy()[:, k_slice]
        desired = getattr(exit_savepoint, accessor_name)().asnumpy()[:, k_slice]
        assert_scaled_allclose(actual, desired, err_msg=attr_name)
