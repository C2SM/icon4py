# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import icon4py.model.common.settings as settings
import icon4py.model.common.test_utils.helpers as helpers
from icon4py.model.common.io import cf_utils
from icon4py.model.common.metrics import metrics_factory as mf

# TODO: mf is metrics_fields in metrics_factory.py. We should change `mf` either here or there
from icon4py.model.common.metrics.metrics_factory import interpolation_savepoint
from icon4py.model.common.states import factory as states_factory


def test_factory(icon_grid, metrics_savepoint):
    factory = mf.fields_factory
    factory.with_grid(icon_grid).with_allocator(settings.backend)
    factory.get("height_on_interface_levels", states_factory.RetrievalType.FIELD)
    factory.get(cf_utils.INTERFACE_LEVEL_STANDARD_NAME, states_factory.RetrievalType.FIELD)

    factory.get("height", states_factory.RetrievalType.FIELD)

    inv_ddqz_full_ref = metrics_savepoint.inv_ddqz_z_full()
    inv_ddqz_z_full = factory.get("inv_ddqz_z_full", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(inv_ddqz_z_full.asnumpy(), inv_ddqz_full_ref.asnumpy())

    ddq_z_half_ref = metrics_savepoint.ddqz_z_half()
    ddqz_z_half_full = factory.get("ddqz_z_half", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(ddqz_z_half_full.asnumpy(), ddq_z_half_ref.asnumpy())

    scalfac_dd3d_ref = metrics_savepoint.scalfac_dd3d()
    scalfac_dd3d_full = factory.get("scalfac_dd3d", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(scalfac_dd3d_full.asnumpy(), scalfac_dd3d_ref.asnumpy())

    rayleigh_w_ref = metrics_savepoint.rayleigh_w()
    rayleigh_w_full = factory.get("rayleigh_w", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(rayleigh_w_full.asnumpy(), rayleigh_w_ref.asnumpy())

    coeff1_dwdz_full_ref = metrics_savepoint.coeff1_dwdz_full()
    coeff2_dwdz_full_ref = metrics_savepoint.coeff2_dwdz_full()
    coeff1_dwdz_full = factory.get("coeff1_dwdz_full", states_factory.RetrievalType.FIELD)
    coeff2_dwdz_full = factory.get("coeff2_dwdz_full", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(coeff1_dwdz_full.asnumpy(), coeff1_dwdz_full_ref.asnumpy())
    assert helpers.dallclose(coeff2_dwdz_full.asnumpy(), coeff2_dwdz_full_ref.asnumpy())

    d2dexdz2_fac1_mc_ref = metrics_savepoint.d2dexdz2_fac1_mc()
    d2dexdz2_fac2_mc_ref = metrics_savepoint.d2dexdz2_fac2_mc()
    d2dexdz2_fac1_mc_full = factory.get("d2dexdz2_fac1_mc", states_factory.RetrievalType.FIELD)
    d2dexdz2_fac2_mc_full = factory.get("d2dexdz2_fac2_mc", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(d2dexdz2_fac1_mc_full.asnumpy(), d2dexdz2_fac1_mc_ref.asnumpy())
    assert helpers.dallclose(d2dexdz2_fac2_mc_full.asnumpy(), d2dexdz2_fac2_mc_ref.asnumpy())

    ddxt_z_half_e_ref = metrics_savepoint.ddxt_z_half_e()
    ddxt_z_half_e_full = factory.get("ddxt_z_half_e", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(ddxt_z_half_e_full.asnumpy(), ddxt_z_half_e_ref.asnumpy())

    ddxn_z_full_ref = metrics_savepoint.ddxn_z_full()
    ddxn_z_full = factory.get("ddxn_z_full", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(ddxn_z_full.asnumpy(), ddxn_z_full_ref.asnumpy())

    vwind_impl_wgt_ref = metrics_savepoint.vwind_impl_wgt()
    vwind_impl_wgt_full = factory.get("vwind_impl_wgt", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(vwind_impl_wgt_full.asnumpy(), vwind_impl_wgt_ref.asnumpy())

    vwind_expl_wgt_ref = metrics_savepoint.vwind_expl_wgt()
    vwind_expl_wgt_full = factory.get("vwind_expl_wgt", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(vwind_expl_wgt_full.asnumpy(), vwind_expl_wgt_ref.asnumpy())

    exner_exfac_ref = metrics_savepoint.exner_exfac()
    exner_exfac_full = factory.get("exner_exfac", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(exner_exfac_full.asnumpy(), exner_exfac_ref.asnumpy())

    pg_edgeidx_dsl_ref = metrics_savepoint.pg_edgeidx_dsl()
    pg_edgeidx_dsl_full = factory.get("pg_edgeidx_dsl", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(pg_edgeidx_dsl_full.asnumpy(), pg_edgeidx_dsl_ref.asnumpy())

    pg_exdist_dsl_ref = metrics_savepoint.pg_exdist_dsl()
    pg_exdist_dsl_full = factory.get("pg_exdist_dsl", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(pg_exdist_dsl_full.asnumpy(), pg_exdist_dsl_ref.asnumpy())

    mask_prog_halo_c_ref = metrics_savepoint.mask_prog_halo_c()
    mask_prog_halo_c_full = factory.get("mask_prog_halo_c", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(mask_prog_halo_c_full.asnumpy(), mask_prog_halo_c_ref.asnumpy())

    bdy_halo_c_ref = metrics_savepoint.bdy_halo_c()
    bdy_halo_c_full = factory.get("bdy_halo_c", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(bdy_halo_c_full.asnumpy(), bdy_halo_c_ref.asnumpy())

    hmask_dd3d_ref = metrics_savepoint.hmask_dd3d()
    hmask_dd3d_full = factory.get("hmask_dd3d", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(hmask_dd3d_full.asnumpy(), hmask_dd3d_ref.asnumpy())

    zdiff_gradp_full_field = factory.get("zdiff_gradp", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(
        zdiff_gradp_full_field, metrics_savepoint.zdiff_gradp().asnumpy(), rtol=1.0e-5
    )

    nudgecoeffs_e_full = factory.get("nudgecoeffs_e", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(nudgecoeffs_e_full, interpolation_savepoint.nudgecoeff_e())

    coeff_gradekin_full = factory.get(
        "coeff_gradekin", states_factory.RetrievalType.FIELD
    )  # TODO: FIELD or DATARRAY?
    assert helpers.dallclose(coeff_gradekin_full, metrics_savepoint.coeff_gradekin().asnumpy())

    wgtfacq_e = factory.get(
        "weighting_factor_for_quadratic_interpolation_to_edge_center",
        states_factory.RetrievalType.FIELD,
    )  # TODO: FIELD or DATARRAY?
    assert helpers.dallclose(
        wgtfacq_e.asnumpy(), metrics_savepoint.wgtfacq_e_dsl(icon_grid.num_levels + 1).asnumpy()
    )

    mask_hdiff = factory.get(
        "mask_hdiff", states_factory.RetrievalType.FIELD
    )  # TODO: FIELD or DATARRAY?
    zd_diffcoef_dsl = factory.get(
        "zd_diffcoef_dsl", states_factory.RetrievalType.FIELD
    )  # TODO: FIELD or DATARRAY?
    zd_vertoffset_dsl = factory.get(
        "zd_vertoffset_dsl", states_factory.RetrievalType.FIELD
    )  # TODO: FIELD or DATARRAY?
    zd_intcoef_dsl = factory.get(
        "zd_intcoef_dsl", states_factory.RetrievalType.FIELD
    )  # TODO: FIELD or DATARRAY?
    assert helpers.dallclose(mask_hdiff, metrics_savepoint.mask_hdiff().asnumpy())
    assert helpers.dallclose(
        zd_diffcoef_dsl, metrics_savepoint.zd_diffcoef().asnumpy(), rtol=1.0e-11
    )
    assert helpers.dallclose(zd_vertoffset_dsl, metrics_savepoint.zd_vertoffset().asnumpy())
    assert helpers.dallclose(zd_intcoef_dsl, metrics_savepoint.zd_intcoef().asnumpy())
