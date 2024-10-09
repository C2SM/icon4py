# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import icon4py.model.common.test_utils.helpers as helpers
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.metrics import metrics_factory as mf

# TODO: mf is metrics_fields in metrics_factory.py. We should change `mf` either here or there
from icon4py.model.common.states import factory as states_factory
from icon4py.model.common.states.metadata import INTERFACE_LEVEL_STANDARD_NAME


def test_factory_inv_ddqz_z(
    grid_savepoint, icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    factory = mf.fields_factory
    num_levels = grid_savepoint.num(dims.KDim)
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels), vct_a, vct_b)
    factory.with_grid(icon_grid, vertical_grid).with_backend(backend)

    factory.get("height_on_interface_levels", states_factory.RetrievalType.FIELD)
    factory.get(INTERFACE_LEVEL_STANDARD_NAME, states_factory.RetrievalType.FIELD)

    inv_ddqz_full_ref = metrics_savepoint.inv_ddqz_z_full()
    inv_ddqz_z_full = factory.get("inv_ddqz_z_full", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(inv_ddqz_z_full.asnumpy(), inv_ddqz_full_ref.asnumpy())


def test_factory_ddq_z_half(
    grid_savepoint, icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    factory = mf.fields_factory
    num_levels = grid_savepoint.num(dims.KDim)
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels), vct_a, vct_b)
    factory.with_grid(icon_grid, vertical_grid).with_backend(backend)

    factory.get("height_on_interface_levels", states_factory.RetrievalType.FIELD)
    factory.get("height", states_factory.RetrievalType.FIELD)
    factory.get(INTERFACE_LEVEL_STANDARD_NAME, states_factory.RetrievalType.FIELD)

    ddq_z_half_ref = metrics_savepoint.ddqz_z_half()
    # check TODOs in stencil
    ddqz_z_half_full = factory.get(
        "functional_determinant_of_metrics_on_interface_levels", states_factory.RetrievalType.FIELD
    )
    assert helpers.dallclose(ddqz_z_half_full.asnumpy(), ddq_z_half_ref.asnumpy())


def test_factory_scalfac_dd3d(
    grid_savepoint, icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    factory = mf.fields_factory
    num_levels = grid_savepoint.num(dims.KDim)
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels), vct_a, vct_b)
    factory.with_grid(icon_grid, vertical_grid).with_backend(backend)

    scalfac_dd3d_ref = metrics_savepoint.scalfac_dd3d()
    scalfac_dd3d_full = factory.get("scalfac_dd3d", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(scalfac_dd3d_full.asnumpy(), scalfac_dd3d_ref.asnumpy())


def test_factory_rayleigh_w(
    grid_savepoint, icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    factory = mf.fields_factory
    num_levels = grid_savepoint.num(dims.KDim)
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_grid = v_grid.VerticalGrid(
        v_grid.VerticalGridConfig(num_levels, rayleigh_damping_height=12500.0), vct_a, vct_b
    )
    factory.with_grid(icon_grid, vertical_grid).with_backend(backend)

    rayleigh_w_ref = metrics_savepoint.rayleigh_w()
    rayleigh_w_full = factory.get("rayleigh_w", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(rayleigh_w_full.asnumpy(), rayleigh_w_ref.asnumpy())


def test_factory_coeffs_dwdz(
    grid_savepoint, icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    factory = mf.fields_factory
    num_levels = grid_savepoint.num(dims.KDim)
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels), vct_a, vct_b)
    factory.with_grid(icon_grid, vertical_grid).with_backend(backend)
    factory.get("height_on_interface_levels", states_factory.RetrievalType.FIELD)
    factory.get(
        "functional_determinant_of_metrics_on_interface_levels", states_factory.RetrievalType.FIELD
    )

    coeff1_dwdz_full_ref = metrics_savepoint.coeff1_dwdz()
    coeff2_dwdz_full_ref = metrics_savepoint.coeff2_dwdz()
    coeff1_dwdz_full = factory.get("coeff1_dwdz", states_factory.RetrievalType.FIELD)
    coeff2_dwdz_full = factory.get("coeff2_dwdz", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(coeff1_dwdz_full.asnumpy(), coeff1_dwdz_full_ref.asnumpy())
    assert helpers.dallclose(coeff2_dwdz_full.asnumpy(), coeff2_dwdz_full_ref.asnumpy())


def test_factory_ref_mc(
    grid_savepoint, icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    factory = mf.fields_factory
    num_levels = grid_savepoint.num(dims.KDim)
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels), vct_a, vct_b)
    factory.with_grid(icon_grid, vertical_grid).with_backend(backend)
    factory.get("height", states_factory.RetrievalType.FIELD)

    theta_ref_mc_ref = metrics_savepoint.theta_ref_mc()
    exner_ref_mc_ref = metrics_savepoint.exner_ref_mc()
    theta_ref_mc_full = factory.get("theta_ref_mc", states_factory.RetrievalType.FIELD)
    exner_ref_mc_full = factory.get("exner_ref_mc", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(exner_ref_mc_ref.asnumpy(), exner_ref_mc_full.asnumpy())
    assert helpers.dallclose(theta_ref_mc_ref.asnumpy(), theta_ref_mc_full.asnumpy())


def test_factory_facs_mc(
    grid_savepoint, icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    factory = mf.fields_factory
    num_levels = grid_savepoint.num(dims.KDim)
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels), vct_a, vct_b)
    factory.with_grid(icon_grid, vertical_grid).with_backend(backend)

    factory.get("height", states_factory.RetrievalType.FIELD)
    factory.get("inv_ddqz_z_full", states_factory.RetrievalType.FIELD)
    factory.get("theta_ref_mc", states_factory.RetrievalType.FIELD)
    factory.get("exner_ref_mc", states_factory.RetrievalType.FIELD)

    d2dexdz2_fac1_mc_ref = metrics_savepoint.d2dexdz2_fac1_mc()
    d2dexdz2_fac2_mc_ref = metrics_savepoint.d2dexdz2_fac2_mc()
    d2dexdz2_fac1_mc_full = factory.get("d2dexdz2_fac1_mc", states_factory.RetrievalType.FIELD)
    d2dexdz2_fac2_mc_full = factory.get("d2dexdz2_fac2_mc", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(d2dexdz2_fac1_mc_full.asnumpy(), d2dexdz2_fac1_mc_ref.asnumpy())
    assert helpers.dallclose(d2dexdz2_fac2_mc_full.asnumpy(), d2dexdz2_fac2_mc_ref.asnumpy())


def test_factory_ddxn_z_full(
    grid_savepoint, icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    factory = mf.fields_factory
    num_levels = grid_savepoint.num(dims.KDim)
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels), vct_a, vct_b)
    factory.with_grid(icon_grid, vertical_grid).with_backend(backend)
    factory.get("ddxn_z_half_e", states_factory.RetrievalType.FIELD)

    ddxn_z_full_ref = metrics_savepoint.ddxn_z_full()
    ddxn_z_full = factory.get("ddxn_z_full", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(ddxn_z_full.asnumpy(), ddxn_z_full_ref.asnumpy())


def test_factory_vwind_impl_wgt(
    grid_savepoint, icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    factory = mf.fields_factory
    num_levels = grid_savepoint.num(dims.KDim)
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels), vct_a, vct_b)
    factory.with_grid(icon_grid, vertical_grid).with_backend(backend)

    factory.get("ddxn_z_half_e", states_factory.RetrievalType.FIELD)
    factory.get("ddxt_z_half_e", states_factory.RetrievalType.FIELD)
    factory.get("height_on_interface_levels", states_factory.RetrievalType.FIELD)
    factory.get("dual_edge_length", states_factory.RetrievalType.FIELD)

    vwind_impl_wgt_ref = metrics_savepoint.vwind_impl_wgt()
    vwind_impl_wgt_full = factory.get("vwind_impl_wgt", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(vwind_impl_wgt_full.asnumpy(), vwind_impl_wgt_ref.asnumpy())


def test_factory_vwind_expl_wgt(
    grid_savepoint, icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    factory = mf.fields_factory
    num_levels = grid_savepoint.num(dims.KDim)
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels), vct_a, vct_b)
    factory.with_grid(icon_grid, vertical_grid).with_backend(backend)
    factory.get("vwind_impl_wgt", states_factory.RetrievalType.FIELD)

    vwind_expl_wgt_ref = metrics_savepoint.vwind_expl_wgt()
    vwind_expl_wgt_full = factory.get("vwind_expl_wgt", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(vwind_expl_wgt_full.asnumpy(), vwind_expl_wgt_ref.asnumpy())


def test_factory_exner_exfac(
    grid_savepoint, icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    factory = mf.fields_factory
    num_levels = grid_savepoint.num(dims.KDim)
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels), vct_a, vct_b)
    factory.with_grid(icon_grid, vertical_grid).with_backend(backend)

    factory.get("ddxn_z_full", states_factory.RetrievalType.FIELD)
    factory.get("dual_edge_length", states_factory.RetrievalType.FIELD)

    exner_exfac_ref = metrics_savepoint.exner_exfac()
    exner_exfac_full = factory.get("exner_exfac", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(exner_exfac_full.asnumpy(), exner_exfac_ref.asnumpy(), rtol=1.0e-10)


def test_factory_pg_edgeidx_dsl(
    grid_savepoint, icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    factory = mf.fields_factory
    num_levels = grid_savepoint.num(dims.KDim)
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels), vct_a, vct_b)
    factory.with_grid(icon_grid, vertical_grid).with_backend(backend)

    factory.get("pg_edgeidx", states_factory.RetrievalType.FIELD)
    factory.get("pg_vertidx", states_factory.RetrievalType.FIELD)

    pg_edgeidx_dsl_ref = metrics_savepoint.pg_edgeidx_dsl()
    pg_edgeidx_dsl_full = factory.get("pg_edgeidx_dsl", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(pg_edgeidx_dsl_full.asnumpy(), pg_edgeidx_dsl_ref.asnumpy())


def test_factory_pg_exdist_dsl(
    grid_savepoint, icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    factory = mf.fields_factory
    num_levels = grid_savepoint.num(dims.KDim)
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels), vct_a, vct_b)
    factory.with_grid(icon_grid, vertical_grid).with_backend(backend)

    factory.get("z_ifc_sliced", states_factory.RetrievalType.FIELD)
    factory.get("height", states_factory.RetrievalType.FIELD)
    factory.get("cell_to_edge_interpolation_coefficient", states_factory.RetrievalType.FIELD)
    factory.get("e_owner_mask", states_factory.RetrievalType.FIELD)
    factory.get("flat_idx_max", states_factory.RetrievalType.FIELD)
    factory.get(INTERFACE_LEVEL_STANDARD_NAME, states_factory.RetrievalType.FIELD)
    factory.get("e_lev", states_factory.RetrievalType.FIELD)

    pg_exdist_dsl_ref = metrics_savepoint.pg_exdist()
    pg_exdist_dsl_full = factory.get("pg_exdist_dsl", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(pg_exdist_dsl_full.asnumpy(), pg_exdist_dsl_ref.asnumpy(), rtol=1.0e-9)


def test_factory_mask_bdy_prog_halo_c(
    grid_savepoint, icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    factory = mf.fields_factory
    num_levels = grid_savepoint.num(dims.KDim)
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels), vct_a, vct_b)
    factory.with_grid(icon_grid, vertical_grid)

    factory.get("c_refin_ctrl", states_factory.RetrievalType.FIELD)

    mask_prog_halo_c_ref = metrics_savepoint.mask_prog_halo_c()
    mask_prog_halo_c_full = factory.get("mask_prog_halo_c", states_factory.RetrievalType.FIELD)
    bdy_halo_c_ref = metrics_savepoint.bdy_halo_c()
    bdy_halo_c_full = factory.get("bdy_halo_c", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(mask_prog_halo_c_full.asnumpy(), mask_prog_halo_c_ref.asnumpy())
    assert helpers.dallclose(bdy_halo_c_full.asnumpy(), bdy_halo_c_ref.asnumpy())


def test_factory_hmask_dd3d(
    grid_savepoint, icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    factory = mf.fields_factory
    num_levels = grid_savepoint.num(dims.KDim)
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels), vct_a, vct_b)
    factory.with_grid(icon_grid, vertical_grid).with_backend(backend)

    factory.get("e_refin_ctrl", states_factory.RetrievalType.FIELD)

    hmask_dd3d_ref = metrics_savepoint.hmask_dd3d()
    hmask_dd3d_full = factory.get("hmask_dd3d", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(hmask_dd3d_full.asnumpy(), hmask_dd3d_ref.asnumpy())


def test_factory_zdiff_gradp(
    grid_savepoint, icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    factory = mf.fields_factory
    num_levels = grid_savepoint.num(dims.KDim)
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels), vct_a, vct_b)
    factory.with_grid(icon_grid, vertical_grid).with_backend(backend)

    factory.get("z_ifc_sliced", states_factory.RetrievalType.FIELD)
    factory.get("cell_to_edge_interpolation_coefficient", states_factory.RetrievalType.FIELD)
    factory.get("height", states_factory.RetrievalType.FIELD)
    factory.get("height_on_interface_levels", states_factory.RetrievalType.FIELD)
    factory.get("flat_idx_max", states_factory.RetrievalType.FIELD)

    zdiff_gradp_ref = metrics_savepoint.zdiff_gradp().asnumpy()
    zdiff_gradp_full_field = factory.get("zdiff_gradp", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(zdiff_gradp_full_field.asnumpy(), zdiff_gradp_ref, rtol=1.0e-5)


def test_factory_coeff_gradekin(
    grid_savepoint, icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    factory = mf.fields_factory
    num_levels = grid_savepoint.num(dims.KDim)
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels), vct_a, vct_b)
    factory.with_grid(icon_grid, vertical_grid).with_backend(backend)

    factory.get("edge_cell_length", states_factory.RetrievalType.FIELD)
    factory.get("inv_dual_edge_length", states_factory.RetrievalType.FIELD)

    coeff_gradekin_ref = metrics_savepoint.coeff_gradekin()
    coeff_gradekin_full = factory.get("coeff_gradekin", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(coeff_gradekin_full.asnumpy(), coeff_gradekin_ref.asnumpy())


def test_factory_wgtfacq_e(
    grid_savepoint, icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    factory = mf.fields_factory
    num_levels = grid_savepoint.num(dims.KDim)
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels), vct_a, vct_b)
    factory.with_grid(icon_grid, vertical_grid).with_backend(backend)

    factory.get("height_on_interface_levels", states_factory.RetrievalType.FIELD)

    wgtfacq_e = factory.get(
        "weighting_factor_for_quadratic_interpolation_to_edge_center",
        states_factory.RetrievalType.FIELD,
    )
    wgtfacq_e_ref = metrics_savepoint.wgtfacq_e_dsl(wgtfacq_e.shape[1])
    assert helpers.dallclose(wgtfacq_e.asnumpy(), wgtfacq_e_ref.asnumpy())


def test_factory_diffusion(
    grid_savepoint, icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    factory = mf.fields_factory
    num_levels = grid_savepoint.num(dims.KDim)
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels), vct_a, vct_b)
    factory.with_grid(icon_grid, vertical_grid).with_backend(backend)

    factory.get("height", states_factory.RetrievalType.FIELD)
    factory.get("max_nbhgt", states_factory.RetrievalType.FIELD)
    factory.get("c_owner_mask", states_factory.RetrievalType.FIELD)
    factory.get("maxslp_avg", states_factory.RetrievalType.FIELD)
    factory.get("maxhgtd_avg", states_factory.RetrievalType.FIELD)

    mask_hdiff = factory.get("mask_hdiff", states_factory.RetrievalType.FIELD)
    zd_diffcoef_dsl = factory.get("zd_diffcoef_dsl", states_factory.RetrievalType.FIELD)
    zd_vertoffset_dsl = factory.get("zd_vertoffset_dsl", states_factory.RetrievalType.FIELD)
    zd_intcoef_dsl = factory.get("zd_intcoef_dsl", states_factory.RetrievalType.FIELD)
    assert helpers.dallclose(mask_hdiff.asnumpy(), metrics_savepoint.mask_hdiff().asnumpy())
    assert helpers.dallclose(
        zd_diffcoef_dsl.asnumpy(), metrics_savepoint.zd_diffcoef().asnumpy(), rtol=1.0e-11
    )
    assert helpers.dallclose(
        zd_vertoffset_dsl.asnumpy(), metrics_savepoint.zd_vertoffset().asnumpy()
    )
    assert helpers.dallclose(zd_intcoef_dsl.asnumpy(), metrics_savepoint.zd_intcoef().asnumpy())
