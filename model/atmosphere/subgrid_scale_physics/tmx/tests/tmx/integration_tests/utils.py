# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers of the tmx integration datatests: state constructors from
the serialized ICON data (exp.exclaim_ape_aesPhys savepoints)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states
from icon4py.model.common import dimension as dims
from icon4py.model.testing import test_utils


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.testing import serialbox as sb


#: Serialized timesteps of the exclaim_ape_aesPhys archive (run start
#: 2008-09-01T00:00:00Z, dtime = 300 s). The archive also holds the
#: 00:00:00 step, but that is the call made during model initialization,
#: so the verification tests parametrize over the subsequent steps only.
TMX_DATES: tuple[str, ...] = ("2008-09-01T00:05:00.000", "2008-09-01T00:10:00.000")


def assert_scaled_allclose(
    actual: np.ndarray,
    desired: np.ndarray,
    *,
    rtol: float = 1.0e-11,
    atol_scale: float = 1.0e-9,
    err_msg: str = "",
) -> None:
    """
    'assert_dallclose' with an absolute tolerance scaled to the reference field.

    The tmx fields verified against the Fortran reference agree to a few ulp of
    the field magnitude, but a plain relative tolerance blows up on near-zero
    entries (e.g. tendencies crossing zero, v-wind on a zonally symmetric
    aquaplanet). ``atol = atol_scale * max|desired|`` gives every field an
    absolute floor tied to its own scale; the measured normalized deviations
    (max abs diff / max|desired|) on the v06 archive are all below 6e-11, and
    the largest relative deviations away from zero are below 3e-12.
    """
    atol = atol_scale * float(np.max(np.abs(desired)))
    test_utils.assert_dallclose(actual, desired, rtol=rtol, atol=atol, err_msg=err_msg)


def verify_full_run_fields(
    diagnostic_state: tmx_states.TmxDiagnosticState,
    tendency_state: tmx_states.TmxTendencyState,
    exit_savepoint: sb.TmxExitSavepoint,
    num_levels: int,
) -> None:
    """Verify the outputs of a full ``Tmx.run`` against the tmx-exit savepoint."""
    # final tendencies and Stage F diagnostics
    fields = (
        (tendency_state.ddt_temperature, exit_savepoint.tend_ta(), "tend_ta"),
        (tendency_state.ddt_qv, exit_savepoint.tend_qv(), "tend_qv"),
        (tendency_state.ddt_qc, exit_savepoint.tend_qc(), "tend_qc"),
        (tendency_state.ddt_qi, exit_savepoint.tend_qi(), "tend_qi"),
        (tendency_state.ddt_u, exit_savepoint.tend_ua(), "tend_ua"),
        (tendency_state.ddt_v, exit_savepoint.tend_va(), "tend_va"),
        (tendency_state.ddt_w, exit_savepoint.tend_wa(), "tend_wa"),
        (diagnostic_state.heating, exit_savepoint.heating(), "heating"),
        (diagnostic_state.dissip_ke, exit_savepoint.dissip_ke(), "dissip_ke"),
    )
    for actual, desired, name in fields:
        assert_scaled_allclose(actual.asnumpy(), desired.asnumpy(), err_msg=name)

    # Stage G vertically integrated diagnostics (2D)
    integrals = (
        (diagnostic_state.cptgz_vi, exit_savepoint.cptgzvi(), "cptgzvi"),
        (diagnostic_state.dissip_ke_vi, exit_savepoint.dissip_ke_vi(), "dissip_ke_vi"),
        (diagnostic_state.int_energy_vi, exit_savepoint.int_energy_vi(), "int_energy_vi"),
        (
            diagnostic_state.int_energy_vi_tend,
            exit_savepoint.tend_int_energy_vi(),
            "tend_int_energy_vi",
        ),
    )
    for actual, desired, name in integrals:
        assert_scaled_allclose(actual.asnumpy(), desired.asnumpy(), err_msg=name)

    # Stage G km/kh diagnostics: the bottom (nlev) row is excluded, it holds
    # the tile-aggregated surface exchange coefficients in the Fortran
    # (km_sfc/kh_sfc from mo_vdf_diag_smag.f90, out of scope of the
    # atmosphere-only port; the granule writes zero there)
    for actual, desired, name in (
        (diagnostic_state.km, exit_savepoint.km(), "km"),
        (diagnostic_state.kh, exit_savepoint.kh(), "kh"),
    ):
        assert_scaled_allclose(
            actual.asnumpy()[:, : num_levels - 1],
            desired.asnumpy()[:, : num_levels - 1],
            err_msg=name,
        )


def flip_back(field: gtx.Field) -> np.ndarray:
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


def construct_metric_state(
    metrics_savepoint: sb.MetricSavepoint,
    init_savepoint: sb.TmxInitSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
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
            flip_back(metrics_savepoint.wgtfacq_c()),
            allocator=allocator,
        ),
        wgtfacq1_c=init_savepoint.wgtfacq1_c(),
        wgtfacq_e=gtx.as_field(
            (dims.EdgeDim, dims.KDim),
            flip_back(metrics_savepoint.wgtfacq_e()),
            allocator=allocator,
        ),
        wgtfacq1_e=init_savepoint.wgtfacq1_e(),
        geopot_agl_ifc=init_savepoint.geopot_agl_ifc(),
        z_mc=metrics_savepoint.z_mc(),
        z_ifc=metrics_savepoint.z_ifc(),
        # a grid-geometry field, not part of the common EdgeParams (see the
        # TmxMetricState docstring)
        edge_cell_length=grid_savepoint.edge_cell_length(),
    )


def construct_interpolation_state(
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


def construct_input_state(entry_savepoint: sb.TmxEntrySavepoint) -> tmx_states.TmxInputState:
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


def construct_surface_flux_state(
    surface_fluxes_savepoint: sb.TmxSurfaceFluxesSavepoint,
) -> tmx_states.TmxSurfaceFluxState:
    return tmx_states.TmxSurfaceFluxState(
        evapotranspiration=surface_fluxes_savepoint.evspsbl(),
        sensible_heat_flux=surface_fluxes_savepoint.hfss(),
        u_stress=surface_fluxes_savepoint.tauu(),
        v_stress=surface_fluxes_savepoint.tauv(),
        q_snocpymlt=surface_fluxes_savepoint.q_snocpymlt(),
    )
