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

import f90nml
import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states
from icon4py.model.common import dimension as dims
from icon4py.model.common.metrics import metric_fields
from icon4py.model.testing import datatest_utils as dt_utils, definitions, test_utils


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing
    import numpy as np

    from icon4py.model.common.decomposition import definitions as decomposition
    from icon4py.model.testing import serialbox as sb


# Serialized timesteps of the exclaim_ape_aesPhys archive. The first one is the
# call made during model initialization, so the verification tests parametrize
# over the subsequent steps only.
TMX_DATES: tuple[str, ...] = definitions.Experiments.EXCLAIM_APE_AES.dates[1:]

# Relative tolerance of all tmx integration datatests.
RTOL: float = test_utils.scale_tol(3.0e-12)

# Tolerances of the tmx datatests, per field: the worst deviation from the serialized ICON
# fields measured on the five backends (CSCS, v13 archive), times 1.1 and rounded up to one
# digit. Only the bound that fits the field is enforced: rtol where the relative deviation
# is at roundoff level (at most 1e-9), atol otherwise; the other bound is 0.0, and the
# comment above the field gives the value it would have.


def construct_metric_state(
    *,
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
    z_mc = metrics_savepoint.z_mc()
    z_ifc = metrics_savepoint.z_ifc()
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
        wgtfacq_c=metrics_savepoint.wgtfacq_c(),
        wgtfacq1_c=init_savepoint.wgtfacq1_c(),
        wgtfacq_e=metrics_savepoint.wgtfacq_e(),
        wgtfacq1_e=init_savepoint.wgtfacq1_e(),
        geopot_agl_ifc=init_savepoint.geopot_agl_ifc(),
        height_above_ground=gtx.as_field(
            (dims.CellDim, dims.KDim),
            metric_fields.compute_height_above_surface(z=z_mc.asnumpy(), z_ifc=z_ifc.asnumpy()),
            allocator=allocator,
        ),
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
        u=entry_savepoint.ua(),
        v=entry_savepoint.va(),
        w=entry_savepoint.wa(),
        qv=entry_savepoint.qv(),
        qc=entry_savepoint.qc(),
        qi=entry_savepoint.qi(),
        qr=entry_savepoint.qr(),
        qs=entry_savepoint.qs(),
        qg=entry_savepoint.qg(),
        air_mass=entry_savepoint.mair(),
        cv_air=entry_savepoint.cvair(),
        rho=entry_savepoint.rho(),
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


def assert_tmx_exit_fields(
    *,
    tendency_state: tmx_states.TmxTendencyState,
    diagnostic_state: tmx_states.TmxDiagnosticState,
    exit_savepoint: sb.TmxExitSavepoint,
    use_km_const: bool,
    owner_mask: np.ndarray,
) -> None:
    """
    Assert that the outputs of a tmx step match the tmx-exit savepoint.

    The tendencies ICON exchanges (temperature, u, v) are compared on all cells, including
    the halo; every other output only on the owned cells, because ICON leaves its halo
    unsynced.
    """
    num_levels = diagnostic_state.km.ndarray.shape[1]
    # the surface level of km and kh is the surface exchange coefficient, written only with
    # `use_km_const`
    # TODO(jcanton): drop this slicing once the tmx surface scheme, which computes km_sfc and
    # kh_sfc, is ported.
    exchange_coefficient_levels = slice(None, None if use_km_const else num_levels - 1)
    # (computed, reference, atol, rtol), chosen as described at the top of this module
    synced_fields = {"tend_ta", "tend_ua", "tend_va"}
    fields = {
        # rtol 9.0
        "tend_ta": (tendency_state.tend_temperature, exit_savepoint.tend_ta(), 2.0e-15, 0.0),
        # rtol inf
        "tend_qv": (tendency_state.tend_qv, exit_savepoint.tend_qv(), 3.0e-18, 0.0),
        # rtol 4.0e-5
        "tend_qc": (tendency_state.tend_qc, exit_savepoint.tend_qc(), 6.0e-19, 0.0),
        # rtol 2.0e-7
        "tend_qi": (tendency_state.tend_qi, exit_savepoint.tend_qi(), 7.0e-22, 0.0),
        # rtol 6.0e-4
        "tend_ua": (tendency_state.tend_u, exit_savepoint.tend_ua(), 2.0e-16, 0.0),
        # rtol 2.0
        "tend_va": (tendency_state.tend_v, exit_savepoint.tend_va(), 4.0e-17, 0.0),
        # rtol 5.0e-5
        "tend_wa": (tendency_state.tend_w, exit_savepoint.tend_wa(), 2.0e-17, 0.0),
        # rtol 2.0e-3
        "heating": (diagnostic_state.heating, exit_savepoint.heating(), 8.0e-13, 0.0),
        # rtol 2.0e-3
        "dissip_ke": (diagnostic_state.dissip_ke, exit_savepoint.dissip_ke(), 8.0e-13, 0.0),
        # atol 3.0e-6
        "cptgzvi": (diagnostic_state.cptgz_vi, exit_savepoint.cptgzvi(), 0.0, 8.0e-16),
        # rtol 2.0e-8
        "dissip_ke_vi": (
            diagnostic_state.dissip_ke_vi,
            exit_savepoint.dissip_ke_vi(),
            3.0e-12,
            0.0,
        ),
        # atol 2.0e-6
        "int_energy_vi": (
            diagnostic_state.int_energy_vi,
            exit_savepoint.int_energy_vi(),
            0.0,
            8.0e-16,
        ),
        # atol 6.0e-9
        "tend_int_energy_vi": (
            diagnostic_state.tend_int_energy_vi,
            exit_savepoint.tend_int_energy_vi(),
            0.0,
            7.0e-11,
        ),
    }
    for name, (computed, reference, atol, rtol) in fields.items():
        compared = slice(None) if name in synced_fields else owner_mask
        test_utils.assert_dallclose(
            computed.asnumpy()[compared],
            reference.asnumpy()[compared],
            atol=atol,
            rtol=rtol,
            err_msg=name,
        )
    for name, computed, reference, atol, rtol in (
        # atol 9.0e-11
        ("km", diagnostic_state.km, exit_savepoint.km(), 0.0, 5.0e-11),
        # atol 3.0e-10
        ("kh", diagnostic_state.kh, exit_savepoint.kh(), 0.0, 5.0e-11),
    ):
        test_utils.assert_dallclose(
            computed.asnumpy()[owner_mask, exchange_coefficient_levels],
            reference.asnumpy()[owner_mask, exchange_coefficient_levels],
            atol=atol,
            rtol=rtol,
            err_msg=name,
        )


# the echoed namelist; every other `NAMELIST_*` file of an archive is the input namelist
_NAMELIST_ATM_FNAME = "NAMELIST_ICON_output_atm"


def read_input_namelist(
    experiment_description: definitions.ExperimentDescription,
    process_props: decomposition.ProcessProperties,
) -> dict:
    """Read the experiment-specific (input) namelist shipped with the archive."""
    experiment_path = dt_utils.get_path_for_experiment(experiment_description, process_props)
    candidates = [c for c in experiment_path.glob("NAMELIST_*") if c.name != _NAMELIST_ATM_FNAME]
    assert len(candidates) == 1, (
        f"expected one input namelist in {experiment_path}, got {candidates}"
    )
    return f90nml.read(candidates[0]).todict()
