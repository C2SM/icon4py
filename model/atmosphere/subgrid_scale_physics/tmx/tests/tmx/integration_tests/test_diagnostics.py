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

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import diagnostics, tmx_states
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.config import TmxConfig
from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.testing import definitions, test_utils

from ..fixtures import *  # noqa: F403
from .utils import (
    RTOL,
    TMX_DATES,
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
def test_tmx_init_and_run_diagnostics_single_step(
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    backend: gtx_typing.Backend | None,
    date: str,
    tmx_config: TmxConfig,
) -> None:
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

    # Smagorinsky_init runs in the constructor; 'ghf' is only serialized at diagnostics exit
    test_utils.assert_dallclose(
        component.mixing_length_sq.asnumpy(),
        init_savepoint.mix_len_sq().asnumpy(),
        err_msg="mixing_length_sq",
    )
    test_utils.assert_dallclose(
        component.scaling_factor_louis.asnumpy(),
        init_savepoint.scaling_factor_louis().asnumpy(),
        err_msg="scaling_factor_louis",
    )
    test_utils.assert_dallclose(
        metric_state.height_above_ground.asnumpy(),
        exit_savepoint.ghf().asnumpy(),
        err_msg="height_above_ground",
    )

    diagnostic_state = tmx_states.TmxDiagnosticState.allocate(icon_grid, allocator=allocator)
    component.run(construct_input_state(entry_savepoint), diagnostic_state)

    nlev = icon_grid.num_levels
    # (diagnostic state attribute, exit savepoint accessor, K slice compared, absolute tolerance)
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
    fields = (
        ("cptgz", "cptgz", everything, 7.0e-11),
        ("theta_v", "theta_v", everything, 2.0e-13),
        ("rho_ic", "rho_ic", everything, 6.0e-16),
        ("bruvais", "bruvais", interior, 5.0e-17),
        ("vn", "vn", everything, 2.0e-14),
        ("w_vert", "w_vert", everything, 4.0e-16),
        ("w_ie", "w_ie", everything, 2.0e-16),
        ("u_vert", "u_vert", everything, 3.0e-14),
        ("v_vert", "v_vert", everything, 5.0e-15),
        ("vn_ie", "vn_ie", everything, 3.0e-14),
        ("vt_ie", "vt_ie", everything, 2.0e-14),
        ("shear", "shear", everything, 6.0e-18),
        ("div_of_stress", "div_of_stress", everything, 4.0e-19),
        ("div_c", "div_c", everything, 3.0e-19),
        ("mech_prod", "mech_prod", interior, 3.0e-18),
        ("km_ic", "km_ic", everything, 1.0e-10),
        ("kh_ic", "kh_ic", everything, 3.0e-10),
        ("km_c", "km_c", everything, 5.0e-11),
        ("km_iv", "km_iv", everything, 2.0e-11),
        ("km_ie", "km_ie", everything, 5.0e-11),
    )
    for attr_name, accessor_name, k_slice, atol in fields:
        test_utils.assert_dallclose(
            getattr(diagnostic_state, attr_name).asnumpy()[:, k_slice],
            getattr(exit_savepoint, accessor_name)().asnumpy()[:, k_slice],
            rtol=RTOL,
            atol=atol,
            err_msg=attr_name,
        )
