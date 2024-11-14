# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import math

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import horizontal
from icon4py.model.common.interpolation.stencils.cell_2_edge_interpolation import (
    cell_2_edge_interpolation,
)
from icon4py.model.common.interpolation.stencils.compute_cell_2_vertex_interpolation import (
    compute_cell_2_vertex_interpolation,
)
from icon4py.model.common.math.helpers import average_cell_kdim_level_up
from icon4py.model.common.metrics.compute_vwind_impl_wgt import (
    compute_vwind_impl_wgt,
)
from icon4py.model.common.metrics.metric_fields import (
    MetricsConfig,
    _compute_flat_idx,
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
    compute_wgtfac_e,
    compute_z_mc,
)
from icon4py.model.common.test_utils import datatest_utils as dt_utils
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    constant_field,
    dallclose,
    is_python,
    is_roundtrip,
    random_field,
    zero_field,
)


cell_domain = horizontal.domain(dims.CellDim)
edge_domain = horizontal.domain(dims.EdgeDim)
vertex_domain = horizontal.domain(dims.VertexDim)


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
        z_mc = zero_field(grid, dims.CellDim, dims.KDim)
        z_if = random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        horizontal_start = 0
        horizontal_end = grid.num_cells
        vertical_start = 0
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
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_ddq_z_half(icon_grid, metrics_savepoint, backend):
    if is_python(backend):
        pytest.skip("skipping: unsupported backend")
    ddq_z_half_ref = metrics_savepoint.ddqz_z_half()
    z_ifc = metrics_savepoint.z_ifc()
    z_mc = zero_field(icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
    nlevp1 = icon_grid.num_levels + 1
    k_index = gtx.as_field((dims.KDim,), np.arange(nlevp1, dtype=gtx.int32))
    compute_z_mc.with_backend(backend)(
        z_ifc,
        z_mc,
        horizontal_start=0,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=gtx.int32(icon_grid.num_levels),
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )
    ddqz_z_half = zero_field(icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})

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
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_ddqz_z_full_and_inverse(icon_grid, metrics_savepoint, backend):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    z_ifc = metrics_savepoint.z_ifc()
    inv_ddqz_full_ref = metrics_savepoint.inv_ddqz_z_full()
    ddqz_z_full = zero_field(icon_grid, dims.CellDim, dims.KDim)
    inv_ddqz_z_full = zero_field(icon_grid, dims.CellDim, dims.KDim)

    compute_ddqz_z_full_and_inverse.with_backend(backend)(
        z_ifc=z_ifc,
        ddqz_z_full=ddqz_z_full,
        inv_ddqz_z_full=inv_ddqz_z_full,
        horizontal_start=0,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    assert dallclose(inv_ddqz_z_full.asnumpy(), inv_ddqz_full_ref.asnumpy())


# TODO: convert this to a stenciltest once it is possible to have only dims.KDim in domain
@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_scalfac_dd3d(icon_grid, metrics_savepoint, grid_savepoint, backend):
    scalfac_dd3d_ref = metrics_savepoint.scalfac_dd3d()
    scalfac_dd3d_full = zero_field(icon_grid, dims.KDim)
    divdamp_trans_start = 12500.0
    divdamp_trans_end = 17500.0
    divdamp_type = 3

    compute_scalfac_dd3d.with_backend(backend=backend)(
        vct_a=grid_savepoint.vct_a(),
        scalfac_dd3d=scalfac_dd3d_full,
        divdamp_trans_start=divdamp_trans_start,
        divdamp_trans_end=divdamp_trans_end,
        divdamp_type=divdamp_type,
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    assert dallclose(scalfac_dd3d_ref.asnumpy(), scalfac_dd3d_full.asnumpy())


# TODO: convert this to a stenciltest once it is possible to have only dims.KDim in domain
@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT])
def test_compute_rayleigh_w(icon_grid, experiment, metrics_savepoint, grid_savepoint, backend):
    rayleigh_w_ref = metrics_savepoint.rayleigh_w()
    vct_a_1 = grid_savepoint.vct_a().asnumpy()[0]
    rayleigh_w_full = zero_field(icon_grid, dims.KDim, extend={dims.KDim: 1})
    rayleigh_type = 2
    rayleigh_coeff = 0.1 if experiment == dt_utils.GLOBAL_EXPERIMENT else 5.0
    damping_height = 50000.0 if experiment == dt_utils.GLOBAL_EXPERIMENT else 12500.0
    compute_rayleigh_w.with_backend(backend=backend)(
        rayleigh_w=rayleigh_w_full,
        vct_a=grid_savepoint.vct_a(),
        damping_height=damping_height,
        rayleigh_type=rayleigh_type,
        rayleigh_classic=constants.RayleighType.CLASSIC,
        rayleigh_klemp=constants.RayleighType.KLEMP,
        rayleigh_coeff=rayleigh_coeff,
        vct_a_1=vct_a_1,
        pi_const=math.pi,
        vertical_start=0,
        vertical_end=grid_savepoint.nrdmax().item() + 1,
        offset_provider={},
    )

    assert dallclose(rayleigh_w_full.asnumpy(), rayleigh_w_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_coeff_dwdz(icon_grid, metrics_savepoint, grid_savepoint, backend):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    coeff1_dwdz_ref = metrics_savepoint.coeff1_dwdz()
    coeff2_dwdz_ref = metrics_savepoint.coeff2_dwdz()

    coeff1_dwdz_full = zero_field(icon_grid, dims.CellDim, dims.KDim)
    coeff2_dwdz_full = zero_field(icon_grid, dims.CellDim, dims.KDim)
    ddqz_z_full = gtx.as_field(
        (dims.CellDim, dims.KDim), 1 / metrics_savepoint.inv_ddqz_z_full().asnumpy()
    )

    compute_coeff_dwdz.with_backend(backend=backend)(
        ddqz_z_full=ddqz_z_full,
        z_ifc=metrics_savepoint.z_ifc(),
        coeff1_dwdz=coeff1_dwdz_full,
        coeff2_dwdz=coeff2_dwdz_full,
        horizontal_start=0,
        horizontal_end=icon_grid.num_cells,
        vertical_start=1,
        vertical_end=gtx.int32(icon_grid.num_levels),
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    assert dallclose(coeff1_dwdz_full.asnumpy(), coeff1_dwdz_ref.asnumpy())
    assert dallclose(coeff2_dwdz_full.asnumpy(), coeff2_dwdz_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_d2dexdz2_fac_mc(icon_grid, metrics_savepoint, grid_savepoint, backend):
    backend = None
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    z_ifc = metrics_savepoint.z_ifc()
    z_mc = zero_field(icon_grid, dims.CellDim, dims.KDim)
    compute_z_mc.with_backend(backend)(
        z_ifc=z_ifc,
        z_mc=z_mc,
        horizontal_start=0,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=gtx.int32(icon_grid.num_levels),
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    d2dexdz2_fac1_mc_ref = metrics_savepoint.d2dexdz2_fac1_mc()
    d2dexdz2_fac2_mc_ref = metrics_savepoint.d2dexdz2_fac2_mc()

    d2dexdz2_fac1_mc_full = zero_field(icon_grid, dims.CellDim, dims.KDim)
    d2dexdz2_fac2_mc_full = zero_field(icon_grid, dims.CellDim, dims.KDim)
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
        igradp_method=3,
        igradp_constant=3,
        horizontal_start=0,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=gtx.int32(icon_grid.num_levels),
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    assert dallclose(d2dexdz2_fac1_mc_full.asnumpy(), d2dexdz2_fac1_mc_ref.asnumpy())
    assert dallclose(d2dexdz2_fac2_mc_full.asnumpy(), d2dexdz2_fac2_mc_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_ddxt_z_full_e(
    grid_savepoint, interpolation_savepoint, icon_grid, metrics_savepoint, backend
):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    z_ifc = metrics_savepoint.z_ifc()
    tangent_orientation = grid_savepoint.tangent_orientation()
    inv_primal_edge_length = grid_savepoint.inverse_primal_edge_lengths()
    ddxt_z_full_ref = metrics_savepoint.ddxt_z_full().asnumpy()
    horizontal_start_vertex = icon_grid.start_index(
        vertex_domain(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    horizontal_end_vertex = icon_grid.end_index(vertex_domain(horizontal.Zone.INTERIOR))
    horizontal_start_edge = icon_grid.start_index(
        edge_domain(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_3)
    )

    horizontal_end_edge = icon_grid.end_index(edge_domain(horizontal.Zone.INTERIOR))
    vertical_start = 0
    vertical_end = icon_grid.num_levels + 1
    cells_aw_verts = interpolation_savepoint.c_intp().asnumpy()
    z_ifv = zero_field(icon_grid, dims.VertexDim, dims.KDim, extend={dims.KDim: 1})
    compute_cell_2_vertex_interpolation.with_backend(backend)(
        z_ifc,
        gtx.as_field((dims.VertexDim, dims.V2CDim), cells_aw_verts),
        z_ifv,
        horizontal_start=horizontal_start_vertex,
        horizontal_end=horizontal_end_vertex,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
        offset_provider={"V2C": icon_grid.get_offset_provider("V2C")},
    )
    ddxt_z_half_e = zero_field(icon_grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1})
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
    ddxt_z_full = zero_field(icon_grid, dims.EdgeDim, dims.KDim)
    compute_ddxn_z_full.with_backend(backend)(
        z_ddxnt_z_half_e=ddxt_z_half_e,
        ddxn_z_full=ddxt_z_full,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    assert np.allclose(ddxt_z_full.asnumpy(), ddxt_z_full_ref)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_vwind_expl_wgt(icon_grid, metrics_savepoint, backend):
    vwind_expl_wgt_full = zero_field(icon_grid, dims.CellDim)
    vwind_expl_wgt_ref = metrics_savepoint.vwind_expl_wgt()
    vwind_impl_wgt = metrics_savepoint.vwind_impl_wgt()

    compute_vwind_expl_wgt.with_backend(backend)(
        vwind_impl_wgt=vwind_impl_wgt,
        vwind_expl_wgt=vwind_expl_wgt_full,
        horizontal_start=0,
        horizontal_end=icon_grid.num_cells,
        offset_provider={"C2E": icon_grid.get_offset_provider("C2E")},
    )

    assert dallclose(vwind_expl_wgt_full.asnumpy(), vwind_expl_wgt_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_ddqz_z_full_e(
    grid_savepoint, interpolation_savepoint, icon_grid, metrics_savepoint, backend
):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    ddqz_z_full = gtx.as_field(
        (dims.CellDim, dims.KDim), 1.0 / metrics_savepoint.inv_ddqz_z_full().asnumpy()
    )
    c_lin_e = interpolation_savepoint.c_lin_e()
    ddqz_z_full_e_ref = metrics_savepoint.ddqz_z_full_e().asnumpy()
    vertical_start = 0
    vertical_end = icon_grid.num_levels
    ddqz_z_full_e = zero_field(icon_grid, dims.EdgeDim, dims.KDim)
    cell_2_edge_interpolation.with_backend(backend)(
        in_field=ddqz_z_full,
        coeff=c_lin_e,
        out_field=ddqz_z_full_e,
        horizontal_start=0,
        horizontal_end=icon_grid.num_edges,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
        offset_provider={"E2C": icon_grid.get_offset_provider("E2C")},
    )
    assert np.allclose(ddqz_z_full_e.asnumpy(), ddqz_z_full_e_ref)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_ddxn_z_full(
    grid_savepoint, interpolation_savepoint, icon_grid, metrics_savepoint, backend
):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    z_ifc = metrics_savepoint.z_ifc()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    ddxn_z_full_ref = metrics_savepoint.ddxn_z_full().asnumpy()
    horizontal_start = icon_grid.start_index(edge_domain(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_2))
    horizontal_end = icon_grid.end_index(edge_domain(horizontal.Zone.INTERIOR))
    vertical_start = 0
    vertical_end = icon_grid.num_levels + 1
    ddxn_z_half_e = zero_field(icon_grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1})
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
    ddxn_z_full = zero_field(icon_grid, dims.EdgeDim, dims.KDim)
    compute_ddxn_z_full.with_backend(backend)(
        z_ddxnt_z_half_e=ddxn_z_half_e,
        ddxn_z_full=ddxn_z_full,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    assert np.allclose(ddxn_z_full.asnumpy(), ddxn_z_full_ref)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_ddxt_z_full(
    grid_savepoint, interpolation_savepoint, icon_grid, metrics_savepoint, backend
):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    z_ifc = metrics_savepoint.z_ifc()
    tangent_orientation = grid_savepoint.tangent_orientation()
    inv_primal_edge_length = grid_savepoint.inverse_primal_edge_lengths()
    ddxt_z_full_ref = metrics_savepoint.ddxt_z_full().asnumpy()
    horizontal_start_vertex = icon_grid.start_index(
        vertex_domain(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    horizontal_end_vertex = icon_grid.end_index(vertex_domain(horizontal.Zone.INTERIOR))
    horizontal_start_edge = icon_grid.start_index(
        edge_domain(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_3)
    )
    horizontal_end_edge = icon_grid.end_index(edge_domain(horizontal.Zone.INTERIOR))
    vertical_start = 0
    vertical_end = icon_grid.num_levels + 1
    cells_aw_verts = interpolation_savepoint.c_intp().asnumpy()
    z_ifv = zero_field(icon_grid, dims.VertexDim, dims.KDim, extend={dims.KDim: 1})
    compute_cell_2_vertex_interpolation(
        z_ifc,
        gtx.as_field((dims.VertexDim, dims.V2CDim), cells_aw_verts),
        z_ifv,
        offset_provider={"V2C": icon_grid.get_offset_provider("V2C")},
        horizontal_start=horizontal_start_vertex,
        horizontal_end=horizontal_end_vertex,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
    )
    ddxt_z_half_e = zero_field(icon_grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1})
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
    ddxt_z_full = zero_field(icon_grid, dims.EdgeDim, dims.KDim)
    compute_ddxn_z_full.with_backend(backend)(
        z_ddxnt_z_half_e=ddxt_z_half_e,
        ddxn_z_full=ddxt_z_full,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    assert np.allclose(ddxt_z_full.asnumpy(), ddxt_z_full_ref)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_exner_exfac(
    grid_savepoint, experiment, interpolation_savepoint, icon_grid, metrics_savepoint, backend
):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")

    horizontal_start = icon_grid.start_index(cell_domain(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_2))
    config = (
        MetricsConfig(exner_expol=0.333)
        if experiment == dt_utils.REGIONAL_EXPERIMENT
        else MetricsConfig()
    )

    exner_exfac = constant_field(icon_grid, config.exner_expol, dims.CellDim, dims.KDim)
    exner_exfac_ref = metrics_savepoint.exner_exfac()
    compute_exner_exfac.with_backend(backend)(
        ddxn_z_full=metrics_savepoint.ddxn_z_full(),
        dual_edge_length=grid_savepoint.dual_edge_length(),
        exner_exfac=exner_exfac,
        exner_expol=config.exner_expol,
        horizontal_start=horizontal_start,
        horizontal_end=icon_grid.num_cells,
        vertical_start=gtx.int32(0),
        vertical_end=icon_grid.num_levels,
        offset_provider={"C2E": icon_grid.get_offset_provider("C2E")},
    )

    assert dallclose(exner_exfac.asnumpy(), exner_exfac_ref.asnumpy(), rtol=1.0e-10)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_vwind_impl_wgt(
    icon_grid, experiment, grid_savepoint, metrics_savepoint, interpolation_savepoint, backend
):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    z_ifc = metrics_savepoint.z_ifc()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    z_ddxn_z_half_e = zero_field(icon_grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1})
    tangent_orientation = grid_savepoint.tangent_orientation()
    inv_primal_edge_length = grid_savepoint.inverse_primal_edge_lengths()
    z_ddxt_z_half_e = zero_field(icon_grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1})
    z_ifv = zero_field(icon_grid, dims.VertexDim, dims.KDim, extend={dims.KDim: 1})
    horizontal_start = icon_grid.start_index(edge_domain(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_2))

    horizontal_end = icon_grid.end_index(edge_domain(horizontal.Zone.INTERIOR))

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

    horizontal_start_edge = icon_grid.start_index(
        edge_domain(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_3)
    )
    horizontal_end_edge = icon_grid.end_index(edge_domain(horizontal.Zone.INTERIOR))

    horizontal_start_vertex = icon_grid.start_index(
        vertex_domain(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    horizontal_end_vertex = icon_grid.end_index(vertex_domain(horizontal.Zone.INTERIOR))
    compute_cell_2_vertex_interpolation(
        z_ifc,
        interpolation_savepoint.c_intp(),
        z_ifv,
        offset_provider={"V2C": icon_grid.get_offset_provider("V2C")},
        horizontal_start=horizontal_start_vertex,
        horizontal_end=horizontal_end_vertex,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
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

    horizontal_start_cell = icon_grid.start_index(
        cell_domain(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    vwind_impl_wgt_ref = metrics_savepoint.vwind_impl_wgt()
    dual_edge_length = grid_savepoint.dual_edge_length()
    vwind_offctr = 0.2
    vwind_impl_wgt_full = constant_field(icon_grid, 0.5 + vwind_offctr, dims.CellDim)
    init_val = 0.65 if experiment == dt_utils.GLOBAL_EXPERIMENT else 0.7
    vwind_impl_wgt_k = constant_field(icon_grid, init_val, dims.CellDim, dims.KDim)

    vwind_impl_wgt = compute_vwind_impl_wgt(
        backend=backend,
        icon_grid=icon_grid,
        vct_a=grid_savepoint.vct_a(),
        z_ifc=metrics_savepoint.z_ifc(),
        z_ddxn_z_half_e=gtx.as_field(
            (dims.EdgeDim,), z_ddxn_z_half_e.asnumpy()[:, icon_grid.num_levels]
        ),
        z_ddxt_z_half_e=gtx.as_field(
            (dims.EdgeDim,), z_ddxt_z_half_e.asnumpy()[:, icon_grid.num_levels]
        ),
        dual_edge_length=dual_edge_length,
        vwind_impl_wgt_full=vwind_impl_wgt_full,
        vwind_impl_wgt_k=vwind_impl_wgt_k,
        global_exp=dt_utils.GLOBAL_EXPERIMENT,
        experiment=experiment,
        vwind_offctr=vwind_offctr,
        horizontal_start_cell=horizontal_start_cell,
    )
    assert dallclose(vwind_impl_wgt_ref.asnumpy(), vwind_impl_wgt)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_wgtfac_e(metrics_savepoint, interpolation_savepoint, icon_grid, backend):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    wgtfac_e = zero_field(icon_grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1})
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
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_pg_exdist_dsl(
    metrics_savepoint, interpolation_savepoint, icon_grid, grid_savepoint, backend
):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    pg_exdist_ref = metrics_savepoint.pg_exdist()
    nlev = icon_grid.num_levels
    k_lev = gtx.as_field((dims.KDim,), np.arange(nlev, dtype=gtx.int32))
    pg_edgeidx = zero_field(icon_grid, dims.EdgeDim, dims.KDim, dtype=gtx.int32)
    pg_vertidx = zero_field(icon_grid, dims.EdgeDim, dims.KDim, dtype=gtx.int32)
    pg_exdist_dsl = zero_field(icon_grid, dims.EdgeDim, dims.KDim)
    z_me = zero_field(icon_grid, dims.EdgeDim, dims.KDim)
    z_aux2 = zero_field(icon_grid, dims.EdgeDim)
    z_mc = zero_field(icon_grid, dims.CellDim, dims.KDim)
    flat_idx = zero_field(icon_grid, dims.EdgeDim, dims.KDim)
    z_ifc = metrics_savepoint.z_ifc()
    z_ifc_sliced = gtx.as_field((dims.CellDim,), z_ifc.asnumpy()[:, nlev])
    start_edge_nudging = icon_grid.end_index(edge_domain(horizontal.Zone.NUDGING))
    horizontal_start_edge = icon_grid.start_index(
        edge_domain(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_3)
    )

    e_lev = gtx.as_field((dims.EdgeDim,), np.arange(icon_grid.num_edges, dtype=gtx.int32))

    average_cell_kdim_level_up.with_backend(backend)(
        z_ifc, out=z_mc, offset_provider={"Koff": icon_grid.get_offset_provider("Koff")}
    )
    cell_2_edge_interpolation.with_backend(backend)(
        in_field=z_mc,
        coeff=interpolation_savepoint.c_lin_e(),
        out_field=z_me,
        horizontal_start=0,
        horizontal_end=icon_grid.num_edges,
        vertical_start=0,
        vertical_end=nlev,
        offset_provider={"E2C": icon_grid.get_offset_provider("E2C")},
    )
    _compute_z_aux2(
        z_ifc=z_ifc_sliced,
        out=z_aux2,
        domain={dims.EdgeDim: (start_edge_nudging, icon_grid.num_edges)},
        offset_provider={"E2C": icon_grid.get_offset_provider("E2C")},
    )

    _compute_flat_idx(
        z_me=z_me,
        z_ifc=z_ifc,
        k_lev=k_lev,
        out=flat_idx,
        domain={
            dims.EdgeDim: (horizontal_start_edge, icon_grid.num_edges),
            dims.KDim: (gtx.int32(0), icon_grid.num_levels),
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
        flat_idx_max=gtx.as_field((dims.EdgeDim,), flat_idx_np, dtype=gtx.int32),
        k_lev=k_lev,
        pg_exdist_dsl=pg_exdist_dsl,
        horizontal_start=start_edge_nudging,
        horizontal_end=icon_grid.num_edges,
        vertical_start=0,
        vertical_end=nlev,
        offset_provider={},
    )

    _compute_pg_edgeidx_vertidx(
        c_lin_e=interpolation_savepoint.c_lin_e(),
        z_ifc=z_ifc,
        z_aux2=z_aux2,
        e_owner_mask=grid_savepoint.e_owner_mask(),
        flat_idx_max=gtx.as_field((dims.EdgeDim,), flat_idx_np, dtype=gtx.int32),
        e_lev=e_lev,
        k_lev=k_lev,
        pg_edgeidx=pg_edgeidx,
        pg_vertidx=pg_vertidx,
        out=(pg_edgeidx, pg_vertidx),
        domain={dims.EdgeDim: (start_edge_nudging, icon_grid.num_edges), dims.KDim: (0, nlev)},
        offset_provider={
            "E2C": icon_grid.get_offset_provider("E2C"),
            "Koff": icon_grid.get_offset_provider("Koff"),
        },
    )

    pg_edgeidx_dsl = zero_field(icon_grid, dims.EdgeDim, dims.KDim, dtype=bool)
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
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_mask_prog_halo_c(metrics_savepoint, icon_grid, grid_savepoint, backend):
    mask_prog_halo_c_full = zero_field(icon_grid, dims.CellDim, dtype=bool)
    c_refin_ctrl = grid_savepoint.refin_ctrl(dims.CellDim)
    mask_prog_halo_c_ref = metrics_savepoint.mask_prog_halo_c()
    horizontal_start = icon_grid.start_index(cell_domain(horizontal.Zone.HALO))
    horizontal_end = icon_grid.end_index(cell_domain(horizontal.Zone.LOCAL))
    compute_mask_prog_halo_c.with_backend(backend)(
        c_refin_ctrl=c_refin_ctrl,
        mask_prog_halo_c=mask_prog_halo_c_full,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        offset_provider={},
    )
    assert dallclose(mask_prog_halo_c_full.asnumpy(), mask_prog_halo_c_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_bdy_halo_c(metrics_savepoint, icon_grid, grid_savepoint, backend):
    bdy_halo_c_full = zero_field(icon_grid, dims.CellDim, dtype=bool)
    c_refin_ctrl = grid_savepoint.refin_ctrl(dims.CellDim)
    bdy_halo_c_ref = metrics_savepoint.bdy_halo_c()
    horizontal_start = icon_grid.start_index(cell_domain(horizontal.Zone.HALO))
    horizontal_end = icon_grid.end_index(cell_domain(horizontal.Zone.LOCAL))

    compute_bdy_halo_c(
        c_refin_ctrl=c_refin_ctrl,
        bdy_halo_c=bdy_halo_c_full,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        offset_provider={},
    )

    assert dallclose(bdy_halo_c_full.asnumpy(), bdy_halo_c_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_hmask_dd3d(metrics_savepoint, icon_grid, grid_savepoint, backend):
    hmask_dd3d_full = zero_field(icon_grid, dims.EdgeDim)
    e_refin_ctrl = grid_savepoint.refin_ctrl(dims.EdgeDim)
    horizontal_start = icon_grid.start_index(cell_domain(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_2))
    hmask_dd3d_ref = metrics_savepoint.hmask_dd3d()
    compute_hmask_dd3d(
        e_refin_ctrl=e_refin_ctrl,
        hmask_dd3d=hmask_dd3d_full,
        grf_nudge_start_e=gtx.int32(horizontal._GRF_NUDGEZONE_START_EDGES),
        grf_nudgezone_width=gtx.int32(horizontal._GRF_NUDGEZONE_WIDTH),
        horizontal_start=horizontal_start,
        horizontal_end=icon_grid.num_edges,
        offset_provider={},
    )

    dallclose(hmask_dd3d_full.asnumpy(), hmask_dd3d_ref.asnumpy())
