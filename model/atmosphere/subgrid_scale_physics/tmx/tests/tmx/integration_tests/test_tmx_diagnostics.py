# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test of the Tmx granule Stage A (Smagorinsky diagnostics).

Constructs the granule from the serialized ICON state (exp.exclaim_ape_aesPhys_sb),
verifies the init fields against the tmx-init savepoint and one call of
``run_diagnostics`` against the tmx-diagnostics-exit savepoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx, tmx_states
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.testing import definitions, test_utils

from ..fixtures import *  # noqa: F403


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing
    import numpy as np

    from icon4py.model.common.grid import icon as icon_grid_
    from icon4py.model.testing import serialbox as sb


# TODO(port_turbulence): verify against the actual timestamps once the
# exclaim_ape_aesPhys_sb archive is generated (run start 2008-09-01T00:00:00Z,
# dtime = 300 s). Keep in sync with integration_tests/test_savepoints.py.
TMX_DATE = "2008-09-01T00:05:00.000"


def _flip_back(field: gtx.Field) -> np.ndarray:
    """
    Reverse the K rows of a 3-row extrapolation coefficient field.

    The metrics savepoint accessors (`wgtfacq_c`/`wgtfacq_e`) flip the Fortran
    coefficient order (row k multiplies the k+1-th full level counted from the
    relevant boundary); the tmx metric state stores the coefficients in Fortran
    order, see the TmxMetricState docstrings.
    """
    array = field.asnumpy()
    assert array.shape[1] == 3
    return array[:, ::-1]


def _construct_config(init_savepoint: sb.TmxInitSavepoint) -> tmx.TmxConfig:
    """Construct a TmxConfig matching the tmx-init savepoint scalars."""
    return tmx.TmxConfig(
        solver_type=tmx.TurbulenceSolverType(int(init_savepoint.solver_type())),
        energy_type=tmx.EnergyType(int(init_savepoint.energy_type())),
        dissipation_factor=init_savepoint.dissipation_factor(),
        use_louis=init_savepoint.use_louis(),
        louis_constant_b=init_savepoint.louis_constant_b(),
        use_km_const=init_savepoint.use_km_const(),
        km_const=init_savepoint.km_const(),
        smag_constant=init_savepoint.smag_constant(),
        turb_prandtl=init_savepoint.turb_prandtl(),
        km_min=init_savepoint.km_min(),
        max_turb_scale=init_savepoint.max_turb_scale(),
    )


def _construct_metric_state(
    metrics_savepoint: sb.MetricSavepoint,
    init_savepoint: sb.TmxInitSavepoint,
    allocator: gtx_typing.Allocator | None,
) -> tmx_states.TmxMetricState:
    inv_ddqz_z_full = metrics_savepoint.inv_ddqz_z_full()
    ddqz_z_full = metrics_savepoint.ddqz_z_full()
    if ddqz_z_full is None:  # optionally registered in the savepoint
        ddqz_z_full = gtx.as_field(
            (dims.CellDim, dims.KDim), 1.0 / inv_ddqz_z_full.asnumpy(), allocator=allocator
        )
    return tmx_states.TmxMetricState(
        ddqz_z_full=ddqz_z_full,
        inv_ddqz_z_full=inv_ddqz_z_full,
        ddqz_z_half=metrics_savepoint.ddqz_z_half(),
        inv_ddqz_z_half=init_savepoint.inv_ddqz_z_half(),
        inv_ddqz_z_full_e=init_savepoint.inv_ddqz_z_full_e(),
        inv_ddqz_z_half_e=init_savepoint.inv_ddqz_z_half_e(),
        inv_ddqz_z_half_v=init_savepoint.inv_ddqz_z_half_v(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        wgtfac_e=metrics_savepoint.wgtfac_e(),
        wgtfacq_c=gtx.as_field(
            (dims.CellDim, dims.KDim),
            _flip_back(metrics_savepoint.wgtfacq_c()),
            allocator=allocator,
        ),
        wgtfacq1_c=init_savepoint.wgtfacq1_c(),
        wgtfacq_e=gtx.as_field(
            (dims.EdgeDim, dims.KDim),
            _flip_back(metrics_savepoint.wgtfacq_e()),
            allocator=allocator,
        ),
        wgtfacq1_e=init_savepoint.wgtfacq1_e(),
        geopot_agl_ifc=init_savepoint.geopot_agl_ifc(),
        z_mc=metrics_savepoint.z_mc(),
        z_ifc=metrics_savepoint.z_ifc(),
    )


def _construct_interpolation_state(
    interpolation_savepoint: sb.InterpolationSavepoint,
) -> tmx_states.TmxInterpolationState:
    return tmx_states.TmxInterpolationState(
        c_lin_e=interpolation_savepoint.c_lin_e(),
        e_bln_c_s=interpolation_savepoint.e_bln_c_s(),
        geofac_div=interpolation_savepoint.geofac_div(),
        # `c_intp` is `p_int_state%cells_aw_verts` in the serialization
        cells_aw_verts=interpolation_savepoint.c_intp(),
        rbf_coeff_v1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_v2=interpolation_savepoint.rbf_vec_coeff_v2(),
        rbf_coeff_e=interpolation_savepoint.rbf_vec_coeff_e(),
        rbf_coeff_c1=interpolation_savepoint.rbf_vec_coeff_c1(),
        rbf_coeff_c2=interpolation_savepoint.rbf_vec_coeff_c2(),
    )


def _construct_input_state(entry_savepoint: sb.TmxEntrySavepoint) -> tmx_states.TmxInputState:
    return tmx_states.TmxInputState(
        temperature=entry_savepoint.ta(),
        virtual_temperature=entry_savepoint.tempv(),
        pressure=entry_savepoint.pres(),
        pressure_ifc=entry_savepoint.pres_ifc(),
        u=entry_savepoint.ua(),
        v=entry_savepoint.va(),
        w=entry_savepoint.wa(),
        qv=entry_savepoint.qv(),
        qc=entry_savepoint.qc(),
        qi=entry_savepoint.qi(),
        qr=entry_savepoint.qr(),
        qs=entry_savepoint.qs(),
        qg=entry_savepoint.qg(),
        rho=entry_savepoint.rho(),
        air_mass=entry_savepoint.mair(),
        cv_air=entry_savepoint.cvair(),
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.APE_AES])
def test_tmx_init_and_run_diagnostics_single_step(  # noqa: PLR0917 [too-many-positional-arguments]
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    backend: gtx_typing.Backend | None,
) -> None:
    allocator = model_backends.get_allocator(backend)
    init_savepoint = data_provider.from_savepoint_tmx_init()
    entry_savepoint = data_provider.from_savepoint_tmx_entry(date=TMX_DATE)
    exit_savepoint = data_provider.from_savepoint_tmx_diagnostics_exit(date=TMX_DATE)

    config = _construct_config(init_savepoint)
    params = tmx.TmxParams(config)
    assert params.rturb_prandtl == pytest.approx(init_savepoint.rturb_prandtl(), rel=1e-14)

    granule = tmx.Tmx(
        grid=icon_grid,
        config=config,
        params=params,
        # the vertical grid is not used by the Smagorinsky diagnostics (Stage A)
        vertical_grid=None,
        metric_state=_construct_metric_state(metrics_savepoint, init_savepoint, allocator),
        interpolation_state=_construct_interpolation_state(interpolation_savepoint),
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
    granule.run_diagnostics(_construct_input_state(entry_savepoint), diagnostic_state)

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
        test_utils.assert_dallclose(actual, desired, err_msg=attr_name)
