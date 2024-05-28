# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import math

import numpy as np
import pytest
from gt4py.next import as_field
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common import constants
from icon4py.model.common.dimension import (
    C2E2CDim,
    CellDim,
    E2CDim,
    ECDim,
    EdgeDim,
    KDim,
    V2CDim,
    VertexDim,
)
from icon4py.model.common.grid.horizontal import (
    HorizontalMarkerIndex,
    _compute_cells2verts,
)
from icon4py.model.common.interpolation.stencils.cell_2_edge_interpolation import (
    _cell_2_edge_interpolation,
    cell_2_edge_interpolation,
)
from icon4py.model.common.math.helpers import average_cell_kdim_level_up
from icon4py.model.common.metrics.metric_fields import (
    _compute_flat_idx,
    _compute_max_nbhgt,
    _compute_maxslp_maxhgtd,
    _compute_pg_edgeidx_vertidx,
    _compute_z_aux2,
    compute_bdy_halo_c,
    compute_coeff_dwdz,
    compute_d2dexdz2_fac_mc,
    compute_ddqz_z_full_and_inverse,
    compute_ddqz_z_half,
    compute_ddxn_z_full,
    compute_ddxn_z_half_e,
    compute_ddxt_z_half_e,
    compute_exner_exfac,
    compute_hmask_dd3d,
    compute_mask_prog_halo_c,
    compute_pg_edgeidx_dsl,
    compute_pg_exdist_dsl,
    compute_rayleigh_w,
    compute_scalfac_dd3d,
    compute_vwind_expl_wgt,
    compute_vwind_impl_wgt,
    compute_wgtfac_e,
    compute_z_maxslp_avg_z_maxhgtd_avg,
    compute_z_mc,
)
from icon4py.model.common.metrics.stencils.compute_diffusion_metrics import (
    _compute_i_params,
    _compute_k_start_end,
    _compute_mask_hdiff,
    _compute_zd_diffcoef_dsl,
    _compute_zd_vertoffset_dsl,
)
from icon4py.model.common.metrics.stencils.compute_zdiff_gradp_dsl import compute_zdiff_gradp_dsl
from icon4py.model.common.test_utils.datatest_utils import (
    GLOBAL_EXPERIMENT,
    REGIONAL_EXPERIMENT,
)
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    constant_field,
    dallclose,
    flatten_first_two_dims,
    is_python,
    is_roundtrip,
    random_field,
    zero_field,
)


class TestComputeZMc(StencilTest):
    PROGRAM = compute_z_mc
    OUTPUTS = ("z_mc",)

    @staticmethod
    def reference(
        grid,
        z_ifc: np.array,
        **kwargs,
    ) -> dict:
        shp = z_ifc.shape
        z_mc = 0.5 * (z_ifc + np.roll(z_ifc, shift=-1, axis=1))[:, : shp[1] - 1]
        return dict(z_mc=z_mc)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        z_mc = zero_field(grid, CellDim, KDim)
        z_if = random_field(grid, CellDim, KDim, extend={KDim: 1})
        horizontal_start = int32(0)
        horizontal_end = grid.num_cells
        vertical_start = int32(0)
        vertical_end = grid.num_levels

        return dict(
            z_mc=z_mc,
            z_ifc=z_if,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )


@pytest.mark.datatest
def test_compute_ddq_z_half(icon_grid, metrics_savepoint, backend):
    if is_python(backend):
        pytest.skip("skipping: unsupported backend")
    ddq_z_half_ref = metrics_savepoint.ddqz_z_half()
    z_ifc = metrics_savepoint.z_ifc()
    z_mc = zero_field(icon_grid, CellDim, KDim, extend={KDim: 1})
    nlevp1 = icon_grid.num_levels + 1
    k_index = as_field((KDim,), np.arange(nlevp1, dtype=int32))
    compute_z_mc.with_backend(backend)(
        z_ifc,
        z_mc,
        horizontal_start=int32(0),
        horizontal_end=icon_grid.num_cells,
        vertical_start=int32(0),
        vertical_end=int32(icon_grid.num_levels),
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )
    ddqz_z_half = zero_field(icon_grid, CellDim, KDim, extend={KDim: 1})

    compute_ddqz_z_half.with_backend(backend=backend)(
        z_ifc=z_ifc,
        z_mc=z_mc,
        k=k_index,
        nlev=icon_grid.num_levels,
        ddqz_z_half=ddqz_z_half,
        horizontal_start=0,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=nlevp1,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    assert dallclose(ddqz_z_half.asnumpy(), ddq_z_half_ref.asnumpy())


@pytest.mark.datatest
def test_compute_ddqz_z_full_and_inverse(icon_grid, metrics_savepoint, backend):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    z_ifc = metrics_savepoint.z_ifc()
    inv_ddqz_full_ref = metrics_savepoint.inv_ddqz_z_full()
    ddqz_z_full = zero_field(icon_grid, CellDim, KDim)
    inv_ddqz_z_full = zero_field(icon_grid, CellDim, KDim)

    compute_ddqz_z_full_and_inverse.with_backend(backend)(
        z_ifc=z_ifc,
        ddqz_z_full=ddqz_z_full,
        inv_ddqz_z_full=inv_ddqz_z_full,
        horizontal_start=int32(0),
        horizontal_end=icon_grid.num_cells,
        vertical_start=int32(0),
        vertical_end=icon_grid.num_levels,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    assert dallclose(inv_ddqz_z_full.asnumpy(), inv_ddqz_full_ref.asnumpy())


# TODO: convert this to a stenciltest once it is possible to have only KDim in domain
@pytest.mark.datatest
def test_compute_scalfac_dd3d(icon_grid, metrics_savepoint, grid_savepoint, backend):
    scalfac_dd3d_ref = metrics_savepoint.scalfac_dd3d()
    scalfac_dd3d_full = zero_field(icon_grid, KDim)
    divdamp_trans_start = 12500.0
    divdamp_trans_end = 17500.0
    divdamp_type = 3

    compute_scalfac_dd3d.with_backend(backend=backend)(
        vct_a=grid_savepoint.vct_a(),
        scalfac_dd3d=scalfac_dd3d_full,
        divdamp_trans_start=divdamp_trans_start,
        divdamp_trans_end=divdamp_trans_end,
        divdamp_type=divdamp_type,
        vertical_start=int32(0),
        vertical_end=icon_grid.num_levels,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    assert dallclose(scalfac_dd3d_ref.asnumpy(), scalfac_dd3d_full.asnumpy())


# TODO: convert this to a stenciltest once it is possible to have only KDim in domain
@pytest.mark.datatest
def test_compute_rayleigh_w(icon_grid, metrics_savepoint, grid_savepoint, backend):
    rayleigh_w_ref = metrics_savepoint.rayleigh_w()
    vct_a_1 = grid_savepoint.vct_a().asnumpy()[0]
    rayleigh_w_full = zero_field(icon_grid, KDim, extend={KDim: 1})
    rayleigh_type = 2
    rayleigh_coeff = 5.0
    damping_height = 12500.0
    compute_rayleigh_w.with_backend(backend=backend)(
        rayleigh_w=rayleigh_w_full,
        vct_a=grid_savepoint.vct_a(),
        damping_height=damping_height,
        rayleigh_type=rayleigh_type,
        rayleigh_classic=constants.RayleighType.RAYLEIGH_CLASSIC,
        rayleigh_klemp=constants.RayleighType.RAYLEIGH_KLEMP,
        rayleigh_coeff=rayleigh_coeff,
        vct_a_1=vct_a_1,
        pi_const=math.pi,
        vertical_start=int32(0),
        vertical_end=grid_savepoint.nrdmax().item() + 1,
        offset_provider={},
    )

    assert dallclose(rayleigh_w_full.asnumpy(), rayleigh_w_ref.asnumpy())


@pytest.mark.datatest
def test_compute_coeff_dwdz(icon_grid, metrics_savepoint, grid_savepoint, backend):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    coeff1_dwdz_ref = metrics_savepoint.coeff1_dwdz()
    coeff2_dwdz_ref = metrics_savepoint.coeff2_dwdz()

    coeff1_dwdz_full = zero_field(icon_grid, CellDim, KDim)
    coeff2_dwdz_full = zero_field(icon_grid, CellDim, KDim)
    ddqz_z_full = as_field((CellDim, KDim), 1 / metrics_savepoint.inv_ddqz_z_full().asnumpy())

    compute_coeff_dwdz.with_backend(backend=backend)(
        ddqz_z_full=ddqz_z_full,
        z_ifc=metrics_savepoint.z_ifc(),
        coeff1_dwdz=coeff1_dwdz_full,
        coeff2_dwdz=coeff2_dwdz_full,
        horizontal_start=int32(0),
        horizontal_end=icon_grid.num_cells,
        vertical_start=int32(1),
        vertical_end=int32(icon_grid.num_levels),
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    assert dallclose(coeff1_dwdz_full.asnumpy(), coeff1_dwdz_ref.asnumpy())
    assert dallclose(coeff2_dwdz_full.asnumpy(), coeff2_dwdz_ref.asnumpy())


@pytest.mark.datatest
def test_compute_d2dexdz2_fac_mc(icon_grid, metrics_savepoint, grid_savepoint, backend):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    z_ifc = metrics_savepoint.z_ifc()
    z_mc = zero_field(icon_grid, CellDim, KDim)
    compute_z_mc.with_backend(backend)(
        z_ifc=z_ifc,
        z_mc=z_mc,
        horizontal_start=int32(0),
        horizontal_end=icon_grid.num_cells,
        vertical_start=int32(0),
        vertical_end=int32(icon_grid.num_levels),
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    d2dexdz2_fac1_mc_ref = metrics_savepoint.d2dexdz2_fac1_mc()
    d2dexdz2_fac2_mc_ref = metrics_savepoint.d2dexdz2_fac2_mc()

    d2dexdz2_fac1_mc_full = zero_field(icon_grid, CellDim, KDim)
    d2dexdz2_fac2_mc_full = zero_field(icon_grid, CellDim, KDim)
    cpd = constants.CPD
    grav = constants.GRAV
    del_t_bg = constants.DEL_T_BG
    h_scal_bg = constants._H_SCAL_BG

    compute_d2dexdz2_fac_mc.with_backend(backend=backend)(
        theta_ref_mc=metrics_savepoint.theta_ref_mc(),
        inv_ddqz_z_full=metrics_savepoint.inv_ddqz_z_full(),
        exner_ref_mc=metrics_savepoint.exner_ref_mc(),
        z_mc=z_mc,
        d2dexdz2_fac1_mc=d2dexdz2_fac1_mc_full,
        d2dexdz2_fac2_mc=d2dexdz2_fac2_mc_full,
        cpd=cpd,
        grav=grav,
        del_t_bg=del_t_bg,
        h_scal_bg=h_scal_bg,
        igradp_method=int32(3),
        horizontal_start=int32(0),
        horizontal_end=icon_grid.num_cells,
        vertical_start=int32(0),
        vertical_end=int32(icon_grid.num_levels),
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    assert dallclose(d2dexdz2_fac1_mc_full.asnumpy(), d2dexdz2_fac1_mc_ref.asnumpy())
    assert dallclose(d2dexdz2_fac2_mc_full.asnumpy(), d2dexdz2_fac2_mc_ref.asnumpy())


@pytest.mark.datatest
def test_compute_ddxt_z_full_e(
    grid_savepoint, interpolation_savepoint, icon_grid, metrics_savepoint
):
    z_ifc = metrics_savepoint.z_ifc()
    tangent_orientation = grid_savepoint.tangent_orientation()
    inv_primal_edge_length = grid_savepoint.inverse_primal_edge_lengths()
    ddxt_z_full_ref = metrics_savepoint.ddxt_z_full().asnumpy()
    horizontal_start_vertex = icon_grid.get_start_index(
        VertexDim,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1,
    )
    horizontal_end_vertex = icon_grid.get_end_index(
        VertexDim,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) - 1,
    )
    horizontal_start_edge = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 2,
    )
    horizontal_end_edge = icon_grid.get_end_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) - 1,
    )
    vertical_start = 0
    vertical_end = icon_grid.num_levels + 1
    cells_aw_verts = interpolation_savepoint.c_intp().asnumpy()
    z_ifv = zero_field(icon_grid, VertexDim, KDim, extend={KDim: 1})
    _compute_cells2verts(
        z_ifc,
        as_field((VertexDim, V2CDim), cells_aw_verts),
        out=z_ifv,
        offset_provider={"V2C": icon_grid.get_offset_provider("V2C")},
        domain={
            VertexDim: (horizontal_start_vertex, horizontal_end_vertex),
            KDim: (vertical_start, vertical_end),
        },
    )
    ddxt_z_half_e = zero_field(icon_grid, EdgeDim, KDim, extend={KDim: 1})
    compute_ddxt_z_half_e(
        z_ifv=z_ifv,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
        ddxt_z_half_e=ddxt_z_half_e,
        horizontal_start=horizontal_start_edge,
        horizontal_end=horizontal_end_edge,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
        offset_provider={"E2V": icon_grid.get_offset_provider("E2V")},
    )
    ddxt_z_full = zero_field(icon_grid, EdgeDim, KDim)
    compute_ddxn_z_full(
        z_ddxnt_z_half_e=ddxt_z_half_e,
        ddxn_z_full=ddxt_z_full,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    assert np.allclose(ddxt_z_full.asnumpy(), ddxt_z_full_ref)


@pytest.mark.datatest
def test_compute_vwind_expl_wgt(icon_grid, metrics_savepoint, backend):
    vwind_expl_wgt_full = zero_field(icon_grid, CellDim)
    vwind_expl_wgt_ref = metrics_savepoint.vwind_expl_wgt()
    vwind_impl_wgt = metrics_savepoint.vwind_impl_wgt()

    compute_vwind_expl_wgt.with_backend(backend)(
        vwind_impl_wgt=vwind_impl_wgt,
        vwind_expl_wgt=vwind_expl_wgt_full,
        horizontal_start=int32(0),
        horizontal_end=icon_grid.num_cells,
        offset_provider={"C2E": icon_grid.get_offset_provider("C2E")},
    )

    assert dallclose(vwind_expl_wgt_full.asnumpy(), vwind_expl_wgt_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", (REGIONAL_EXPERIMENT, GLOBAL_EXPERIMENT))
def test_compute_ddqz_z_full_e(
    grid_savepoint, interpolation_savepoint, icon_grid, metrics_savepoint, backend
):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    ddqz_z_full = as_field((CellDim, KDim), 1.0 / metrics_savepoint.inv_ddqz_z_full().asnumpy())
    c_lin_e = interpolation_savepoint.c_lin_e()
    ddqz_z_full_e_ref = metrics_savepoint.ddqz_z_full_e().asnumpy()
    vertical_start = 0
    vertical_end = icon_grid.num_levels
    ddqz_z_full_e = zero_field(icon_grid, EdgeDim, KDim)
    # TODO: perhaps write a program with ddqz_z_full_e name and call fieldop _cells2edges... from there
    # TODO: This way it's clear where this field is computed and we cna more easily avoid duplicates
    cell_2_edge_interpolation.with_backend(backend)(
        in_field=ddqz_z_full,
        coeff=c_lin_e,
        out_field=ddqz_z_full_e,
        horizontal_start=0,
        horizontal_end=ddqz_z_full_e.shape[0],
        vertical_start=vertical_start,
        vertical_end=vertical_end,
        offset_provider={"E2C": icon_grid.get_offset_provider("E2C")},
    )
    assert np.allclose(ddqz_z_full_e.asnumpy(), ddqz_z_full_e_ref)


@pytest.mark.datatest
def test_compute_ddxn_z_full(
    grid_savepoint, interpolation_savepoint, icon_grid, metrics_savepoint, backend
):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    z_ifc = metrics_savepoint.z_ifc()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    ddxn_z_full_ref = metrics_savepoint.ddxn_z_full().asnumpy()
    horizontal_start = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
    )
    horizontal_end = icon_grid.get_end_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) - 1,
    )
    vertical_start = 0
    vertical_end = icon_grid.num_levels + 1
    ddxn_z_half_e = zero_field(icon_grid, EdgeDim, KDim, extend={KDim: 1})
    compute_ddxn_z_half_e.with_backend(backend)(
        z_ifc=z_ifc,
        inv_dual_edge_length=inv_dual_edge_length,
        ddxn_z_half_e=ddxn_z_half_e,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
        offset_provider={"E2C": icon_grid.get_offset_provider("E2C")},
    )
    ddxn_z_full = zero_field(icon_grid, EdgeDim, KDim)
    compute_ddxn_z_full.with_backend(backend)(
        z_ddxnt_z_half_e=ddxn_z_half_e,
        ddxn_z_full=ddxn_z_full,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    assert np.allclose(ddxn_z_full.asnumpy(), ddxn_z_full_ref)


@pytest.mark.datatest
def test_compute_ddxt_z_full(
    grid_savepoint, interpolation_savepoint, icon_grid, metrics_savepoint, backend
):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    z_ifc = metrics_savepoint.z_ifc()
    tangent_orientation = grid_savepoint.tangent_orientation()
    inv_primal_edge_length = grid_savepoint.inverse_primal_edge_lengths()
    ddxt_z_full_ref = metrics_savepoint.ddxt_z_full().asnumpy()
    horizontal_start_vertex = icon_grid.get_start_index(
        VertexDim,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1,
    )
    horizontal_end_vertex = icon_grid.get_end_index(
        VertexDim,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) - 1,
    )
    horizontal_start_edge = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 2,
    )
    horizontal_end_edge = icon_grid.get_end_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) - 1,
    )
    vertical_start = 0
    vertical_end = icon_grid.num_levels + 1
    cells_aw_verts = interpolation_savepoint.c_intp().asnumpy()
    z_ifv = zero_field(icon_grid, VertexDim, KDim, extend={KDim: 1})
    _compute_cells2verts(
        z_ifc,
        as_field((VertexDim, V2CDim), cells_aw_verts),
        out=z_ifv,
        offset_provider={"V2C": icon_grid.get_offset_provider("V2C")},
        domain={
            VertexDim: (horizontal_start_vertex, horizontal_end_vertex),
            KDim: (vertical_start, vertical_end),
        },
    )
    ddxt_z_half_e = zero_field(icon_grid, EdgeDim, KDim, extend={KDim: 1})
    compute_ddxt_z_half_e.with_backend(backend)(
        z_ifv=z_ifv,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
        ddxt_z_half_e=ddxt_z_half_e,
        horizontal_start=horizontal_start_edge,
        horizontal_end=horizontal_end_edge,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
        offset_provider={"E2V": icon_grid.get_offset_provider("E2V")},
    )
    ddxt_z_full = zero_field(icon_grid, EdgeDim, KDim)
    compute_ddxn_z_full.with_backend(backend)(
        z_ddxnt_z_half_e=ddxt_z_half_e,
        ddxn_z_full=ddxt_z_full,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    assert np.allclose(ddxt_z_full.asnumpy(), ddxt_z_full_ref)


@pytest.mark.datatest
def test_compute_exner_exfac(
    grid_savepoint, interpolation_savepoint, icon_grid, metrics_savepoint, backend
):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    horizontal_start = icon_grid.get_start_index(
        CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 1
    )
    exner_exfac = constant_field(icon_grid, constants.exner_expol, CellDim, KDim)
    exner_exfac_ref = metrics_savepoint.exner_exfac()
    compute_exner_exfac.with_backend(backend)(
        ddxn_z_full=metrics_savepoint.ddxn_z_full(),
        dual_edge_length=grid_savepoint.dual_edge_length(),
        exner_exfac=exner_exfac,
        exner_expol=constants.exner_expol,
        horizontal_start=horizontal_start,
        horizontal_end=icon_grid.num_cells,
        vertical_start=int32(0),
        vertical_end=icon_grid.num_levels,
        offset_provider={"C2E": icon_grid.get_offset_provider("C2E")},
    )

    assert dallclose(exner_exfac.asnumpy(), exner_exfac_ref.asnumpy(), rtol=1.0e-10)


# TODO
@pytest.mark.datatest
def test_compute_zdiff_gradp_dsl(icon_grid, metrics_savepoint, interpolation_savepoint, backend):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    zdiff_gradp_ref = metrics_savepoint.zdiff_gradp()
    z_mc = zero_field(icon_grid, CellDim, KDim)
    z_ifc = metrics_savepoint.z_ifc()
    k_lev = as_field((KDim,), np.arange(icon_grid.num_levels, dtype=int))
    z_me = zero_field(icon_grid, EdgeDim, KDim)
    horizontal_start_edge = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
    )
    end_edge_nudging = icon_grid.get_end_index(EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim))
    compute_z_mc.with_backend(backend)(
        z_ifc,
        z_mc,
        horizontal_start=int32(0),
        horizontal_end=icon_grid.num_cells,
        vertical_start=int32(0),
        vertical_end=int32(icon_grid.num_levels),
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )
    _cell_2_edge_interpolation(
        in_field=z_mc,
        coeff=interpolation_savepoint.c_lin_e(),
        out=z_me,
        offset_provider={"E2C": icon_grid.get_offset_provider("E2C")},
    )
    flat_idx = zero_field(icon_grid, EdgeDim, KDim)
    _compute_flat_idx(
        z_me=z_me,
        z_ifc=z_ifc,
        k_lev=k_lev,
        out=flat_idx,
        domain={
            EdgeDim: (horizontal_start_edge, icon_grid.num_edges),
            KDim: (int32(0), icon_grid.num_levels),
        },
        offset_provider={
            "E2C": icon_grid.get_offset_provider("E2C"),
            "Koff": icon_grid.get_offset_provider("Koff"),
        },
    )
    flat_idx_np = np.amax(flat_idx.asnumpy(), axis=1)
    z_ifc_sliced = as_field((CellDim,), z_ifc.asnumpy()[:, icon_grid.num_levels])
    z_aux2 = zero_field(icon_grid, EdgeDim)
    _compute_z_aux2(
        z_ifc=z_ifc_sliced,
        out=z_aux2,
        domain={EdgeDim: (end_edge_nudging, icon_grid.num_edges)},
        offset_provider={"E2C": icon_grid.get_offset_provider("E2C")},
    )

    zdiff_gradp_full_np = compute_zdiff_gradp_dsl(
        e2c=icon_grid.connectivities[E2CDim],
        z_me=z_me.asnumpy(),
        z_mc=z_mc.asnumpy(),
        z_ifc=metrics_savepoint.z_ifc().asnumpy(),
        flat_idx=flat_idx_np,
        z_aux2=z_aux2.asnumpy(),
        nlev=icon_grid.num_levels,
        horizontal_start=horizontal_start_edge,
        horizontal_start_1=end_edge_nudging,
        nedges=icon_grid.num_edges,
    )
    zdiff_gradp_full_field = flatten_first_two_dims(
        ECDim, KDim, field=as_field((EdgeDim, E2CDim, KDim), zdiff_gradp_full_np)
    )
    assert dallclose(zdiff_gradp_full_field.asnumpy(), zdiff_gradp_ref.asnumpy(), rtol=1.0e-5)


@pytest.mark.datatest
def test_compute_vwind_impl_wgt(
    icon_grid, grid_savepoint, metrics_savepoint, interpolation_savepoint, backend
):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    z_ifc = metrics_savepoint.z_ifc()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    z_ddxn_z_half_e = zero_field(icon_grid, EdgeDim, KDim, extend={KDim: 1})
    tangent_orientation = grid_savepoint.tangent_orientation()
    inv_primal_edge_length = grid_savepoint.inverse_primal_edge_lengths()
    z_ddxt_z_half_e = zero_field(icon_grid, EdgeDim, KDim, extend={KDim: 1})
    z_ifv = zero_field(icon_grid, VertexDim, KDim, extend={KDim: 1})
    horizontal_start = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
    )
    horizontal_end = icon_grid.get_end_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) - 1,
    )
    vertical_start = 0
    vertical_end = icon_grid.num_levels + 1

    compute_ddxn_z_half_e(
        z_ifc=z_ifc,
        inv_dual_edge_length=inv_dual_edge_length,
        ddxn_z_half_e=z_ddxn_z_half_e,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
        offset_provider={"E2C": icon_grid.get_offset_provider("E2C")},
    )

    horizontal_start_edge = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 2,
    )
    horizontal_end_edge = icon_grid.get_end_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) - 1,
    )

    horizontal_start_vertex = icon_grid.get_start_index(
        VertexDim,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1,
    )
    horizontal_end_vertex = icon_grid.get_end_index(
        VertexDim,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) - 1,
    )
    _compute_cells2verts(
        z_ifc,
        interpolation_savepoint.c_intp(),
        out=z_ifv,
        offset_provider={"V2C": icon_grid.get_offset_provider("V2C")},
        domain={
            VertexDim: (horizontal_start_vertex, horizontal_end_vertex),
            KDim: (vertical_start, vertical_end),
        },
    )

    compute_ddxt_z_half_e(
        z_ifv=z_ifv,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
        ddxt_z_half_e=z_ddxt_z_half_e,
        horizontal_start=horizontal_start_edge,
        horizontal_end=horizontal_end_edge,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
        offset_provider={"E2V": icon_grid.get_offset_provider("E2V")},
    )

    horizontal_start_cell = icon_grid.get_start_index(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) + 1,
    )
    vwind_impl_wgt_ref = metrics_savepoint.vwind_impl_wgt()
    dual_edge_length = grid_savepoint.dual_edge_length()
    vwind_offctr = 0.2
    vwind_impl_wgt_full = constant_field(icon_grid, 0.5 + vwind_offctr, CellDim)
    vwind_impl_wgt_k = constant_field(icon_grid, 0.7, CellDim, KDim)

    compute_vwind_impl_wgt.with_backend(backend)(
        z_ddxn_z_half_e=as_field((EdgeDim,), z_ddxn_z_half_e.asnumpy()[:, icon_grid.num_levels]),
        z_ddxt_z_half_e=as_field((EdgeDim,), z_ddxt_z_half_e.asnumpy()[:, icon_grid.num_levels]),
        dual_edge_length=dual_edge_length,
        vct_a=grid_savepoint.vct_a(),
        z_ifc=metrics_savepoint.z_ifc(),
        vwind_impl_wgt=vwind_impl_wgt_full,
        vwind_impl_wgt_k=vwind_impl_wgt_k,
        vwind_offctr=vwind_offctr,
        horizontal_start=horizontal_start_cell,
        horizontal_end=icon_grid.num_cells,
        vertical_start=max(10, icon_grid.num_levels - 8),
        vertical_end=icon_grid.num_levels,
        offset_provider={
            "C2E": icon_grid.get_offset_provider("C2E"),
            "Koff": icon_grid.get_offset_provider("Koff"),
        },
    )

    vwind_impl_wgt = np.amax(vwind_impl_wgt_k.asnumpy(), axis=1)
    assert dallclose(vwind_impl_wgt_ref.asnumpy(), vwind_impl_wgt)


@pytest.mark.datatest
def test_compute_wgtfac_e(metrics_savepoint, interpolation_savepoint, icon_grid, backend):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    wgtfac_e = zero_field(icon_grid, EdgeDim, KDim, extend={KDim: 1})
    wgtfac_e_ref = metrics_savepoint.wgtfac_e()
    compute_wgtfac_e.with_backend(backend)(
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        c_lin_e=interpolation_savepoint.c_lin_e(),
        wgtfac_e=wgtfac_e,
        horizontal_start=0,
        horizontal_end=wgtfac_e.shape[0],
        vertical_start=0,
        vertical_end=icon_grid.num_levels + 1,
        offset_provider={"E2C": icon_grid.get_offset_provider("E2C")},
    )
    assert dallclose(wgtfac_e.asnumpy(), wgtfac_e_ref.asnumpy())


@pytest.mark.datatest
def test_compute_pg_exdist_dsl(
    metrics_savepoint, interpolation_savepoint, icon_grid, grid_savepoint, backend
):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    pg_exdist_ref = metrics_savepoint.pg_exdist()
    nlev = icon_grid.num_levels
    k_lev = as_field((KDim,), np.arange(nlev, dtype=int))
    pg_edgeidx = zero_field(icon_grid, EdgeDim, KDim, dtype=int)
    pg_vertidx = zero_field(icon_grid, EdgeDim, KDim, dtype=int)
    pg_exdist_dsl = zero_field(icon_grid, EdgeDim, KDim)
    z_me = zero_field(icon_grid, EdgeDim, KDim)
    z_aux2 = zero_field(icon_grid, EdgeDim)
    z_mc = zero_field(icon_grid, CellDim, KDim)
    flat_idx = zero_field(icon_grid, EdgeDim, KDim)
    z_ifc = metrics_savepoint.z_ifc()
    z_ifc_sliced = as_field((CellDim,), z_ifc.asnumpy()[:, nlev])
    start_edge_nudging = icon_grid.get_end_index(EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim))
    horizontal_start_edge = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 2,
    )
    e_lev = as_field((EdgeDim,), np.arange(icon_grid.num_edges, dtype=int32))

    average_cell_kdim_level_up(
        z_ifc, out=z_mc, offset_provider={"Koff": icon_grid.get_offset_provider("Koff")}
    )
    _cell_2_edge_interpolation(
        in_field=z_mc,
        coeff=interpolation_savepoint.c_lin_e(),
        out=z_me,
        offset_provider={"E2C": icon_grid.get_offset_provider("E2C")},
    )
    _compute_z_aux2(
        z_ifc=z_ifc_sliced,
        out=z_aux2,
        domain={EdgeDim: (start_edge_nudging, icon_grid.num_edges)},
        offset_provider={"E2C": icon_grid.get_offset_provider("E2C")},
    )

    _compute_flat_idx(
        z_me=z_me,
        z_ifc=z_ifc,
        k_lev=k_lev,
        out=flat_idx,
        domain={
            EdgeDim: (horizontal_start_edge, icon_grid.num_edges),
            KDim: (int32(0), icon_grid.num_levels),
        },
        offset_provider={
            "E2C": icon_grid.get_offset_provider("E2C"),
            "Koff": icon_grid.get_offset_provider("Koff"),
        },
    )
    flat_idx_np = np.amax(flat_idx.asnumpy(), axis=1)

    compute_pg_exdist_dsl.with_backend(backend)(
        z_aux2=z_aux2,
        z_me=z_me,
        e_owner_mask=grid_savepoint.e_owner_mask(),
        flat_idx_max=as_field((EdgeDim,), flat_idx_np, dtype=int),
        k_lev=k_lev,
        pg_exdist_dsl=pg_exdist_dsl,
        horizontal_start=start_edge_nudging,
        horizontal_end=icon_grid.num_edges,
        vertical_start=int(0),
        vertical_end=nlev,
        offset_provider={},
    )

    _compute_pg_edgeidx_vertidx(
        c_lin_e=interpolation_savepoint.c_lin_e(),
        z_ifc=z_ifc,
        z_aux2=z_aux2,
        e_owner_mask=grid_savepoint.e_owner_mask(),
        flat_idx_max=as_field((EdgeDim,), flat_idx_np, dtype=int),
        e_lev=e_lev,
        k_lev=k_lev,
        pg_edgeidx=pg_edgeidx,
        pg_vertidx=pg_vertidx,
        out=(pg_edgeidx, pg_vertidx),
        domain={EdgeDim: (start_edge_nudging, icon_grid.num_edges), KDim: (int(0), nlev)},
        offset_provider={
            "E2C": icon_grid.get_offset_provider("E2C"),
            "Koff": icon_grid.get_offset_provider("Koff"),
        },
    )

    pg_edgeidx_dsl = zero_field(icon_grid, EdgeDim, KDim, dtype=bool)
    pg_edgeidx_dsl_ref = metrics_savepoint.pg_edgeidx_dsl()

    compute_pg_edgeidx_dsl.with_backend(backend)(
        pg_edgeidx=pg_edgeidx,
        pg_vertidx=pg_vertidx,
        pg_edgeidx_dsl=pg_edgeidx_dsl,
        horizontal_start=int(0),
        horizontal_end=icon_grid.num_edges,
        vertical_start=int(0),
        vertical_end=icon_grid.num_levels,
        offset_provider={},
    )

    assert dallclose(pg_exdist_ref.asnumpy(), pg_exdist_dsl.asnumpy(), rtol=1.0e-9)
    assert dallclose(pg_edgeidx_dsl_ref.asnumpy(), pg_edgeidx_dsl.asnumpy())


@pytest.mark.datatest
def test_compute_mask_prog_halo_c(metrics_savepoint, icon_grid, grid_savepoint, backend):
    mask_prog_halo_c_full = zero_field(icon_grid, CellDim, dtype=bool)
    c_refin_ctrl = grid_savepoint.refin_ctrl(CellDim)
    mask_prog_halo_c_ref = metrics_savepoint.mask_prog_halo_c()
    horizontal_start = icon_grid.get_start_index(CellDim, HorizontalMarkerIndex.local(CellDim) - 1)
    horizontal_end = icon_grid.get_end_index(CellDim, HorizontalMarkerIndex.local(CellDim))
    compute_mask_prog_halo_c.with_backend(backend)(
        c_refin_ctrl=c_refin_ctrl,
        mask_prog_halo_c=mask_prog_halo_c_full,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        offset_provider={},
    )
    assert dallclose(mask_prog_halo_c_full.asnumpy(), mask_prog_halo_c_ref.asnumpy())


@pytest.mark.datatest
def test_compute_bdy_halo_c(metrics_savepoint, icon_grid, grid_savepoint, backend):
    bdy_halo_c_full = zero_field(icon_grid, CellDim, dtype=bool)
    c_refin_ctrl = grid_savepoint.refin_ctrl(CellDim)
    bdy_halo_c_ref = metrics_savepoint.bdy_halo_c()
    horizontal_start = icon_grid.get_start_index(CellDim, HorizontalMarkerIndex.local(CellDim) - 1)
    horizontal_end = icon_grid.get_end_index(CellDim, HorizontalMarkerIndex.local(CellDim))
    compute_bdy_halo_c(
        c_refin_ctrl=c_refin_ctrl,
        bdy_halo_c=bdy_halo_c_full,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        offset_provider={},
    )

    assert dallclose(bdy_halo_c_full.asnumpy(), bdy_halo_c_ref.asnumpy())


@pytest.mark.datatest
def test_compute_hmask_dd3d(metrics_savepoint, icon_grid, grid_savepoint, backend):
    hmask_dd3d_full = zero_field(icon_grid, EdgeDim)
    e_refin_ctrl = grid_savepoint.refin_ctrl(EdgeDim)
    horizontal_start = icon_grid.get_start_index(
        CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 1
    )
    hmask_dd3d_ref = metrics_savepoint.hmask_dd3d()
    compute_hmask_dd3d(
        e_refin_ctrl=e_refin_ctrl,
        hmask_dd3d=hmask_dd3d_full,
        grf_nudge_start_e=int32(constants.grf_nudge_start_e),
        grf_nudgezone_width=int32(constants.grf_nudgezone_width),
        horizontal_start=horizontal_start,
        horizontal_end=icon_grid.num_edges,
        offset_provider={},
    )

    dallclose(hmask_dd3d_full.asnumpy(), hmask_dd3d_ref.asnumpy())


@pytest.mark.datatest
def test_compute_diffusion_metrics(
    metrics_savepoint, interpolation_savepoint, icon_grid, grid_savepoint, backend
):
    backend = None
    mask_hdiff = zero_field(icon_grid, CellDim, KDim, dtype=bool).asnumpy()
    zd_vertoffset_dsl = zero_field(icon_grid, CellDim, C2E2CDim, KDim).asnumpy()
    nlev = icon_grid.num_levels
    mask_hdiff_ref = metrics_savepoint.mask_hdiff()
    zd_vertoffset_ref = metrics_savepoint.zd_vertoffset()
    # TODO: inquire to Magda as to why 3316 works instead of cell_nudging
    # cell_nudging = icon_grid.get_end_index(CellDim, HorizontalMarkerIndex.nudging(CellDim))
    cell_nudging = 3316
    cell_lateral = icon_grid.get_start_index(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) + 1,
    )
    z_maxslp_avg = zero_field(icon_grid, CellDim, KDim)
    z_maxhgtd_avg = zero_field(icon_grid, CellDim, KDim)
    zd_diffcoef_dsl = zero_field(icon_grid, CellDim, KDim).asnumpy()
    c_bln_avg = interpolation_savepoint.c_bln_avg()
    thslp_zdiffu = (
        0.02  # TODO: import from Fortran, note: same variable in diffusion has different value
    )
    thhgtd_zdiffu = (
        125  # TODO: import from Fortran, note: same variable in diffusion has different value
    )
    c_owner_mask = grid_savepoint.c_owner_mask().asnumpy()
    maxslp = zero_field(icon_grid, CellDim, KDim)
    maxhgtd = zero_field(icon_grid, CellDim, KDim)
    max_nbhgt = zero_field(icon_grid, CellDim)
    c2e2c = icon_grid.connectivities[C2E2CDim]
    nbidx = constant_field(icon_grid, 1, CellDim, C2E2CDim, KDim, dtype=int).asnumpy()

    _compute_maxslp_maxhgtd(
        ddxn_z_full=metrics_savepoint.ddxn_z_full(),
        dual_edge_length=grid_savepoint.dual_edge_length(),
        out=(maxslp, maxhgtd),
        domain={CellDim: (cell_lateral, icon_grid.num_cells), KDim: (int32(0), nlev)},
        offset_provider={"C2E": icon_grid.get_offset_provider("C2E")},
    )

    z_ifc = metrics_savepoint.z_ifc()
    z_mc = zero_field(icon_grid, CellDim, KDim, extend={KDim: 1})
    compute_z_mc.with_backend(backend)(
        z_ifc,
        z_mc,
        horizontal_start=int32(0),
        horizontal_end=icon_grid.num_cells,
        vertical_start=int32(0),
        vertical_end=nlev,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )
    z_mc_off = z_mc.asnumpy()[c2e2c]

    compute_z_maxslp_avg_z_maxhgtd_avg.with_backend(backend)(
        maxslp=maxslp,
        maxhgtd=maxhgtd,
        c_bln_avg_0=as_field((CellDim,), c_bln_avg.asnumpy()[:, 0]),
        c_bln_avg_1=as_field((CellDim,), c_bln_avg.asnumpy()[:, 1]),
        c_bln_avg_2=as_field((CellDim,), c_bln_avg.asnumpy()[:, 2]),
        c_bln_avg_3=as_field((CellDim,), c_bln_avg.asnumpy()[:, 3]),
        z_maxslp_avg=z_maxslp_avg,
        z_maxhgtd_avg=z_maxhgtd_avg,
        horizontal_start=cell_lateral,
        horizontal_end=int32(icon_grid.num_cells),
        vertical_start=0,
        vertical_end=nlev,
        offset_provider={"C2E2C": icon_grid.get_offset_provider("C2E2C")},
    )

    _compute_max_nbhgt(
        z_mc_nlev=as_field((CellDim,), z_mc.asnumpy()[:, nlev - 1]),
        out=max_nbhgt,
        domain={CellDim: (int32(1), int32(icon_grid.num_cells))},
        offset_provider={"C2E2C": icon_grid.get_offset_provider("C2E2C")},
    )

    k_start, k_end = _compute_k_start_end(
        z_mc.asnumpy(),
        max_nbhgt.asnumpy(),
        z_maxslp_avg.asnumpy(),
        z_maxhgtd_avg.asnumpy(),
        c_owner_mask,
        thslp_zdiffu,
        thhgtd_zdiffu,
        cell_nudging,
        icon_grid.num_cells,
        nlev,
    )

    i_indlist, i_listreduce, ji = _compute_i_params(
        k_start,
        k_end,
        z_maxslp_avg.asnumpy(),
        z_maxhgtd_avg.asnumpy(),
        c_owner_mask,
        thslp_zdiffu,
        thhgtd_zdiffu,
        cell_nudging,
        icon_grid.num_cells,
        nlev,
    )

    i_listdim = ji - i_listreduce
    mask_hdiff = _compute_mask_hdiff(mask_hdiff, k_start, k_end, i_indlist, i_listdim)
    zd_diffcoef_dsl = _compute_zd_diffcoef_dsl(
        z_maxslp_avg.asnumpy(),
        z_maxhgtd_avg.asnumpy(),
        k_start,
        k_end,
        i_indlist,
        i_listdim,
        zd_diffcoef_dsl,
        thslp_zdiffu,
        thhgtd_zdiffu,
    )
    zd_vertoffset_dsl = _compute_zd_vertoffset_dsl(
        k_start,
        k_end,
        z_mc.asnumpy(),
        z_mc_off,
        nbidx,
        i_indlist,
        i_listdim,
        zd_vertoffset_dsl,
        nlev,
    )

    assert dallclose(mask_hdiff, mask_hdiff_ref.asnumpy())
    assert dallclose(zd_diffcoef_dsl, metrics_savepoint.zd_diffcoef().asnumpy(), rtol=1.0e-11)
    assert dallclose(zd_vertoffset_dsl, zd_vertoffset_ref.asnumpy())
