# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import math
import pathlib

import gt4py.next as gtx
import numpy as np
from gt4py.next import as_field

import icon4py.model.common.states.factory as factory
from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro import (
    HorizontalPressureDiscretizationType,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid, vertical as v_grid
from icon4py.model.common.interpolation.stencils import cell_2_edge_interpolation
from icon4py.model.common.io import cf_utils
from icon4py.model.common.metrics import (
    compute_coeff_gradekin,
    compute_diffusion_metrics,
    compute_nudgecoeffs,
    compute_vwind_impl_wgt,
    compute_wgtfac_c,
    compute_wgtfacq,
    compute_zdiff_gradp_dsl,
    metric_fields as mf,
)
from icon4py.model.common.settings import xp
from icon4py.model.common.test_utils import datatest_utils as dt_utils, serialbox_utils as sb

# we need to register a couple of fields from the serializer. Those should get replaced one by one.
from icon4py.model.common.test_utils.helpers import constant_field


dt_utils.TEST_DATA_ROOT = pathlib.Path(__file__).parent / "testdata"
properties = decomposition.get_processor_properties(decomposition.get_runtype(with_mpi=False))
path = dt_utils.get_datapath_for_experiment(
    dt_utils.get_ranked_data_path(dt_utils.SERIALIZED_DATA_PATH, properties)
)

data_provider = sb.IconSerialDataProvider(
    "icon_pydycore", str(path.absolute()), False, mpi_rank=properties.rank
)

# z_ifc (computable from vertical grid for model without topography)
metrics_savepoint = data_provider.from_metrics_savepoint()

# interpolation fields also for now passing as precomputed fields
interpolation_savepoint = data_provider.from_interpolation_savepoint()
# can get geometry fields as pre computed fields from the grid_savepoint
root, level = dt_utils.get_global_grid_params(dt_utils.REGIONAL_EXPERIMENT)
grid_id = dt_utils.get_grid_id_for_experiment(dt_utils.REGIONAL_EXPERIMENT)
grid_savepoint = data_provider.from_savepoint_grid(grid_id, root, level)
nlev = grid_savepoint.num(dims.KDim)
cell_domain = h_grid.domain(dims.CellDim)
edge_domain = h_grid.domain(dims.EdgeDim)
vertex_domain = h_grid.domain(dims.VertexDim)
#######

# start build up factory:

# used for vertical domain below: should go away once vertical grid provids start_index and end_index like interface
grid = grid_savepoint.global_grid_params

# TODO: this will go in a future ConfigurationProvider
experiment = dt_utils.GLOBAL_EXPERIMENT
init_val = 0.65 if experiment == dt_utils.GLOBAL_EXPERIMENT else 0.7
vwind_offctr = 0.2
divdamp_trans_start = 12500.0
divdamp_trans_end = 17500.0
divdamp_type = 3
damping_height = 50000.0 if dt_utils.GLOBAL_EXPERIMENT else 12500.0
rayleigh_coeff = 0.1 if dt_utils.GLOBAL_EXPERIMENT else 5.0
vct_a_1 = grid_savepoint.vct_a().asnumpy()[0]
nudge_max_coeff = 0.375
nudge_efold_width = 2.0
nudge_zone_width = 10
thslp_zdiffu = 0.02
thhgtd_zdiffu = 125

interface_model_height = metrics_savepoint.z_ifc()
c_lin_e = interpolation_savepoint.c_lin_e()
c_bln_avg = interpolation_savepoint.c_bln_avg()
k_index = gtx.as_field((dims.KDim,), xp.arange(nlev + 1, dtype=gtx.int32))
vct_a = grid_savepoint.vct_a()
theta_ref_mc = metrics_savepoint.theta_ref_mc()
exner_ref_mc = metrics_savepoint.exner_ref_mc()
c_refin_ctrl = grid_savepoint.refin_ctrl(dims.CellDim)
e_refin_ctrl = grid_savepoint.refin_ctrl(dims.EdgeDim)
dual_edge_length = grid_savepoint.dual_edge_length()
tangent_orientation = grid_savepoint.tangent_orientation()
inv_primal_edge_length = grid_savepoint.inverse_primal_edge_lengths()
inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
cells_aw_verts = interpolation_savepoint.c_intp().asnumpy()
cells_aw_verts_field = gtx.as_field((dims.VertexDim, dims.V2CDim), cells_aw_verts)
icon_grid = grid_savepoint.construct_icon_grid(on_gpu=False)
vwind_impl_wgt_full = constant_field(icon_grid, 0.5 + vwind_offctr, dims.CellDim)
vwind_impl_wgt_k = constant_field(icon_grid, init_val, dims.CellDim, dims.KDim)
k_lev = gtx.as_field((dims.KDim,), np.arange(nlev, dtype=gtx.int32))
e_lev = gtx.as_field((dims.EdgeDim,), np.arange(icon_grid.num_edges, dtype=gtx.int32))
e_owner_mask = grid_savepoint.e_owner_mask()
c_owner_mask = grid_savepoint.c_owner_mask()
edge_cell_length = grid_savepoint.edge_cell_length()


fields_factory = factory.FieldsFactory()

fields_factory.register_provider(
    factory.PrecomputedFieldsProvider(
        {
            "height_on_interface_levels": interface_model_height,
            "cell_to_edge_interpolation_coefficient": c_lin_e,
            "c_bln_avg": c_bln_avg,
            cf_utils.INTERFACE_LEVEL_STANDARD_NAME: k_index,
            "vct_a": vct_a,
            "theta_ref_mc": theta_ref_mc,
            "exner_ref_mc": exner_ref_mc,
            "c_refin_ctrl": c_refin_ctrl,
            "e_refin_ctrl": e_refin_ctrl,
            "dual_edge_length": dual_edge_length,
            "tangent_orientation": tangent_orientation,
            "inv_primal_edge_length": inv_primal_edge_length,
            "inv_dual_edge_length": inv_dual_edge_length,
            "cells_aw_verts_field": cells_aw_verts_field,
            "vwind_impl_wgt_full": vwind_impl_wgt_full,
            "vwind_impl_wgt_k": vwind_impl_wgt_k,
            "k_lev": k_lev,
            "e_lev": e_lev,
            "e_owner_mask": e_owner_mask,
            "c_owner_mask": c_owner_mask,
            "edge_cell_length": edge_cell_length,
        }
    )
)


height_provider = factory.ProgramFieldProvider(
    func=mf.compute_z_mc,
    domain={
        dims.CellDim: (cell_domain(h_grid.Zone.LOCAL), cell_domain(h_grid.Zone.END)),
        dims.KDim: (
            v_grid.domain(dims.KDim)(v_grid.Zone.TOP),
            v_grid.domain(dims.KDim)(v_grid.Zone.BOTTOM),
        ),
    },
    fields={"z_mc": "height"},
    deps={"z_ifc": "height_on_interface_levels"},
)
fields_factory.register_provider(height_provider)

compute_ddqz_z_half_provider = factory.ProgramFieldProvider(
    func=mf.compute_ddqz_z_half,
    deps={
        "z_ifc": "height_on_interface_levels",
        "z_mc": "height",
        "k_index": cf_utils.INTERFACE_LEVEL_STANDARD_NAME,
    },
    domain={
        dims.CellDim: (
            cell_domain(h_grid.Zone.LOCAL),
            cell_domain(h_grid.Zone.LOCAL),
        ),
        dims.KDim: (
            v_grid.domain(dims.KDim)(v_grid.Zone.TOP),
            v_grid.domain(dims.KDim)(v_grid.Zone.BOTTOM),  # TODO: edit dimension - KHalfDim
        ),
    },
    fields={"ddqz_z_half": "ddqz_z_half"},
    params={"nlev": nlev},
)
fields_factory.register_provider(compute_ddqz_z_half_provider)

ddqz_z_full_and_inverse_provider = factory.ProgramFieldProvider(
    func=mf.compute_ddqz_z_full_and_inverse,
    deps={
        "z_ifc": "height_on_interface_levels",
    },
    domain={
        dims.CellDim: (
            cell_domain(h_grid.Zone.LOCAL),
            cell_domain(h_grid.Zone.LOCAL),
        ),
        dims.KDim: (
            v_grid.domain(dims.KDim)(v_grid.Zone.TOP),
            v_grid.domain(dims.KDim)(v_grid.Zone.BOTTOM),
        ),
    },
    fields={"ddqz_z_full": "ddqz_z_full", "inv_ddqz_z_full": "inv_ddqz_z_full"},
)
fields_factory.register_provider(ddqz_z_full_and_inverse_provider)


compute_scalfac_dd3d_provider = factory.ProgramFieldProvider(
    func=mf.compute_scalfac_dd3d,
    deps={
        "vct_a": "vct_a",
    },
    domain={
        dims.KDim: (
            v_grid.domain(dims.KDim)(v_grid.Zone.TOP),
            v_grid.domain(dims.KDim)(v_grid.Zone.BOTTOM),
        )
    },
    fields={"scalfac_dd3d": "scalfac_dd3d"},
    params={
        "divdamp_trans_start": divdamp_trans_start,
        "divdamp_trans_end": divdamp_trans_end,
        "divdamp_type": divdamp_type,
    },
)
fields_factory.register_provider(compute_scalfac_dd3d_provider)


compute_rayleigh_w_provider = factory.ProgramFieldProvider(
    func=mf.compute_rayleigh_w,
    deps={
        "vct_a": "vct_a",
    },
    domain={
        dims.KDim: (0, grid_savepoint.nrdmax().item() + 1),
    },
    fields={"rayleigh_w": "rayleigh_w"},
    params={
        "damping_height": damping_height,
        "rayleigh_type": 2,
        "rayleigh_classic": constants.RayleighType.CLASSIC,
        "rayleigh_klemp": constants.RayleighType.KLEMP,
        "rayleigh_coeff": rayleigh_coeff,
        "vct_a_1": vct_a_1,
        "pi_const": math.pi,
    },
)
fields_factory.register_provider(compute_rayleigh_w_provider)

compute_coeff_dwdz_provider = factory.ProgramFieldProvider(
    func=mf.compute_coeff_dwdz,
    deps={
        "ddqz_z_full": "ddqz_z_full",
        "z_ifc": "height_on_interface_levels",
    },
    domain={
        dims.CellDim: (
            cell_domain(h_grid.Zone.LOCAL),
            cell_domain(h_grid.Zone.LOCAL),
        ),
        dims.KDim: (
            v_grid.domain(dims.KDim)(v_grid.Zone.TOP),  # TODO: edit bounds - actual start at 1
            v_grid.domain(dims.KDim)(v_grid.Zone.BOTTOM),
        ),
    },
    fields={"coeff1_dwdz_full": "coeff1_dwdz_full", "coeff2_dwdz_full": "coeff2_dwdz_full"},
)
fields_factory.register_provider(compute_coeff_dwdz_provider)

compute_d2dexdz2_fac_mc_provider = factory.ProgramFieldProvider(
    func=mf.compute_d2dexdz2_fac_mc,
    deps={
        "theta_ref_mc": "theta_ref_mc",
        "inv_ddqz_z_full": "inv_ddqz_z_full",
        "exner_ref_mc": "exner_ref_mc",
        "z_mc": "height",
    },
    domain={
        dims.CellDim: (
            cell_domain(h_grid.Zone.LOCAL),
            cell_domain(h_grid.Zone.LOCAL),
        ),
        dims.KDim: (
            v_grid.domain(dims.KDim)(v_grid.Zone.TOP),
            v_grid.domain(dims.KDim)(v_grid.Zone.BOTTOM),
        ),
    },
    fields={"d2dexdz2_fac1_mc": "d2dexdz2_fac1_mc", "d2dexdz2_fac2_mc": "d2dexdz2_fac2_mc"},
    params={
        "cpd": constants.CPD,
        "grav": constants.GRAV,
        "del_t_bg": constants.DEL_T_BG,
        "h_scal_bg": constants._H_SCAL_BG,
        "igradp_method": 3,
        "igradp_constant": HorizontalPressureDiscretizationType.TAYLOR_HYDRO,
    },
)
fields_factory.register_provider(compute_d2dexdz2_fac_mc_provider)

compute_cell_2_vertex_interpolation_provider = factory.ProgramFieldProvider(
    func=mf.compute_cell_2_vertex_interpolation,
    deps={
        "cell_in": "height_on_interface_levels",
        "c_int": "cells_aw_verts_field",
    },
    domain={
        dims.CellDim: (
            vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
            vertex_domain(h_grid.Zone.INTERIOR),
        ),
        dims.KDim: (
            v_grid.domain(dims.KDim)(v_grid.Zone.TOP),
            v_grid.domain(dims.KDim)(v_grid.Zone.BOTTOM),  # TODO: edit dimension - KHalfDim
        ),
    },
    fields={"z_ifv": "z_ifv"},
)
fields_factory.register_provider(compute_cell_2_vertex_interpolation_provider)

compute_ddxt_z_half_e_provider = factory.ProgramFieldProvider(
    func=mf.compute_ddxt_z_half_e,
    deps={
        "z_ifv": "z_ifv",
        "inv_primal_edge_length": "inv_primal_edge_length",
        "tangent_orientation": "inv_primal_edge_length",
    },
    domain={
        dims.CellDim: (
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3),
            edge_domain(h_grid.Zone.INTERIOR),
        ),
        dims.KDim: (nlev, nlev + 1),  # TODO: edit dimension - KHalfDim
    },
    fields={"ddxt_z_half_e": "ddxt_z_half_e"},
)
fields_factory.register_provider(compute_ddxt_z_half_e_provider)


compute_ddxn_z_full_provider = factory.ProgramFieldProvider(
    func=mf.compute_ddxn_z_full,
    deps={
        "ddxt_z_half_e": "ddxt_z_half_e",
    },
    domain={},
    fields={"ddxn_z_full": "ddxn_z_full"},
)
fields_factory.register_provider(compute_ddxn_z_full_provider)


compute_ddxn_z_half_e_provider = factory.ProgramFieldProvider(
    func=mf.compute_ddxn_z_half_e,
    deps={
        "z_ifc": "height_on_interface_levels",
        "inv_dual_edge_length": "inv_dual_edge_length",
    },
    domain={
        dims.EdgeDim: (
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
            edge_domain(h_grid.Zone.INTERIOR),
        ),
        dims.KDim: (nlev, nlev + 1),  # TODO: edit dimension - KHalfDim
    },
    fields={"ddxn_z_half_e": "ddxn_z_half_e"},
)
fields_factory.register_provider(compute_ddxn_z_half_e_provider)


compute_vwind_impl_wgt_provider = factory.NumpyFieldsProvider(
    func=compute_vwind_impl_wgt.compute_vwind_impl_wgt,
    domain={
        dims.CellDim: (
            edge_domain(h_grid.Zone.LOCAL),
            edge_domain(h_grid.Zone.LOCAL),
        ),
        dims.KDim: (
            v_grid.domain(dims.KDim)(v_grid.Zone.TOP),
            v_grid.domain(dims.KDim)(v_grid.Zone.BOTTOM),
        ),
    },
    fields=["vwind_impl_wgt"],
    deps={
        "vct_a": "vct_a",
        "z_ifc": "height_on_interface_levels",
        "z_ddxn_z_half_e": "z_ddxn_z_half_e",
        "z_ddxt_z_half_e": "z_ddxt_z_half_e",
        "dual_edge_length": "dual_edge_length",
        "vwind_impl_wgt_full": "vwind_impl_wgt_full",
        "vwind_impl_wgt_k": "vwind_impl_wgt_k",
    },
    params={
        "backend": "backend",
        "icon_grid": "icon_grid",
        "global_exp": "global_exp",
        "experiment": "experiment",
        "vwind_offctr": "vwind_offctr",
        "horizontal_start_cell": "horizontal_start_cell",
    },
)
fields_factory.register_provider(compute_vwind_impl_wgt_provider)

compute_vwind_expl_wgt_provider = factory.ProgramFieldProvider(
    func=mf.compute_vwind_expl_wgt,
    deps={
        "vwind_impl_wgt": "vwind_impl_wgt",
    },
    domain={
        dims.CellDim: (
            cell_domain(h_grid.Zone.LOCAL),
            cell_domain(h_grid.Zone.LOCAL),
        ),
    },
    fields={"vwind_expl_wgt": "vwind_expl_wgt"},
)

compute_exner_exfac_provider = factory.ProgramFieldProvider(
    func=mf.compute_exner_exfac,
    deps={
        "ddxn_z_full": "ddxn_z_full",
        "dual_edge_length": "dual_edge_length",
    },
    domain={
        dims.CellDim: (
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
            cell_domain(h_grid.Zone.LOCAL),
        ),
        dims.KDim: (
            v_grid.domain(dims.KDim)(v_grid.Zone.TOP),
            v_grid.domain(dims.KDim)(v_grid.Zone.BOTTOM),
        ),
    },
    fields={"exner_exfac": "exner_exfac"},
    params={"exner_expol": "exner_expol"},
)
fields_factory.register_provider(compute_exner_exfac_provider)

compute_wgtfac_c_provider = factory.ProgramFieldProvider(
    func=compute_wgtfac_c.compute_wgtfac_c,
    deps={
        "z_ifc": "z_ifc",
        "k": "k_index",
    },
    domain={dims.CellDim: (0, icon_grid.num_cells), dims.KDim: (0, icon_grid.num_levels)},
    fields={"wgtfac_c": "wgtfac_c"},
    params={"nlev": icon_grid.num_levels},
)
fields_factory.register_provider(compute_wgtfac_c_provider)

compute_wgtfac_e_provider = factory.ProgramFieldProvider(
    func=mf.compute_wgtfac_e,
    deps={
        "wgtfac_c": "wgtfac_c",
        "c_lin_e": "cell_to_edge_interpolation_coefficient",
    },
    domain={
        dims.CellDim: (
            edge_domain(h_grid.Zone.LOCAL),
            edge_domain(h_grid.Zone.LOCAL),
        ),
        dims.KDim: (
            v_grid.domain(dims.KDim)(v_grid.Zone.TOP),
            v_grid.domain(dims.KDim)(v_grid.Zone.BOTTOM),  # TODO: edit dimension - KHalfDim
        ),
    },
    fields={"wgtfac_e": "wgtfac_e"},
)
fields_factory.register_provider(compute_wgtfac_e_provider)

compute_compute_z_aux2 = factory.ProgramFieldProvider(
    func=mf.compute_z_aux2,
    deps={"z_ifc_sliced": "z_ifc_sliced"},
    domain={
        dims.EdgeDim: (
            edge_domain(h_grid.Zone.NUDGING),  # TODO: check if this is really end (also in mf)
            edge_domain(h_grid.Zone.LOCAL),
        )
    },
    fields={"z_aux2": "z_aux2"},
)

cell_2_edge_interpolation_provider = factory.ProgramFieldProvider(
    func=cell_2_edge_interpolation.cell_2_edge_interpolation,
    deps={"in_field": "height", "coeff": "c_lin_e"},
    domain={
        dims.EdgeDim: (
            edge_domain(h_grid.Zone.LOCAL),
            edge_domain(h_grid.Zone.LOCAL),
        ),
        dims.KDim: (
            v_grid.domain(dims.KDim)(v_grid.Zone.TOP),
            v_grid.domain(dims.KDim)(v_grid.Zone.BOTTOM),
        ),
    },
    fields={"z_me": "z_me"},
)

compute_flat_idx_provider = factory.ProgramFieldProvider(
    func=mf.compute_flat_idx,
    deps={
        "z_me": "z_me",
        "z_ifc": "height_on_interface_levels",
        "k_lev": "k_lev",
    },
    domain={
        dims.EdgeDim: (
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3),
            edge_domain(h_grid.Zone.LOCAL),
        ),
        dims.KDim: (
            v_grid.domain(dims.KDim)(v_grid.Zone.TOP),
            v_grid.domain(dims.KDim)(v_grid.Zone.BOTTOM),
        ),
    },
    fields={"flat_idx": "flat_idx"},
)
fields_factory.register_provider(compute_flat_idx_provider)


compute_pg_edgeidx_vertidx_provider = factory.ProgramFieldProvider(
    func=mf.compute_pg_edgeidx_vertidx,
    deps={
        "c_lin_e": "c_lin_e",
        "z_ifc": "height_on_interface_levels",
        "z_aux2": "z_aux2",
        "e_owner_mask": "e_owner_mask",
        "flat_idx_max": gtx.as_field(
            (dims.EdgeDim,),
            np.amax(compute_flat_idx_provider.fields()["flat_idx"].asnumpy(), axis=1),
            dtype=gtx.int32,
        ),
        "e_lev": "e_lev",
        "k_lev": "k_lev",
    },
    domain={
        dims.EdgeDim: (
            edge_domain(h_grid.Zone.NUDGING),
            edge_domain(h_grid.Zone.LOCAL),
        ),
        dims.KDim: (
            v_grid.domain(dims.KDim)(v_grid.Zone.TOP),
            v_grid.domain(dims.KDim)(v_grid.Zone.BOTTOM),
        ),
    },
    fields={"pg_edgeidx": "pg_edgeidx", "pg_vertidx": "pg_vertidx"},
)

compute_pg_edgeidx_dsl_provider = factory.ProgramFieldProvider(
    func=mf.compute_pg_edgeidx_dsl,
    deps={"pg_edgeidx": "pg_edgeidx", "pg_vertidx": "pg_vertidx"},
    domain={
        dims.EdgeDim: (
            edge_domain(h_grid.Zone.LOCAL),
            edge_domain(h_grid.Zone.LOCAL),
        ),
        dims.KDim: (
            v_grid.domain(dims.KDim)(v_grid.Zone.TOP),
            v_grid.domain(dims.KDim)(v_grid.Zone.BOTTOM),
        ),
    },
    fields={"pg_edgeidx_dsl": "pg_edgeidx_dsl"},
)
fields_factory.register_provider(compute_pg_edgeidx_dsl_provider)


compute_pg_exdist_dsl_provider = factory.ProgramFieldProvider(
    func=mf.compute_pg_exdist_dsl,
    deps={
        "z_aux2": "z_aux2",
        "z_me": "z_me",
        "e_owner_mask": "e_owner_mask",
        "flat_idx_max": gtx.as_field(
            (dims.EdgeDim,),
            np.amax(compute_flat_idx_provider.fields()["flat_idx"].asnumpy(), axis=1),
            dtype=gtx.int32,
        ),
        "k_lev": "k_lev",
    },
    domain={
        dims.CellDim: (
            edge_domain(h_grid.Zone.NUDGING),
            edge_domain(h_grid.Zone.LOCAL),
        ),
        dims.KDim: (
            v_grid.domain(dims.KDim)(v_grid.Zone.TOP),
            v_grid.domain(dims.KDim)(v_grid.Zone.BOTTOM),
        ),
    },
    fields={"pg_exdist_dsl": "pg_exdist_dsl"},
)
fields_factory.register_provider(compute_pg_exdist_dsl_provider)


compute_mask_prog_halo_c_provider = factory.ProgramFieldProvider(
    func=mf.compute_mask_prog_halo_c,
    deps={
        "c_refin_ctrl": "c_refin_ctrl",
    },
    domain={
        dims.CellDim: (
            cell_domain(h_grid.Zone.HALO),
            cell_domain(h_grid.Zone.LOCAL),
        ),
    },
    fields={"mask_prog_halo_c": "mask_prog_halo_c"},
)
fields_factory.register_provider(compute_mask_prog_halo_c_provider)


compute_bdy_halo_c_provider = factory.ProgramFieldProvider(
    func=mf.compute_bdy_halo_c,
    deps={
        "c_refin_ctrl": "c_refin_ctrl",
    },
    domain={
        dims.CellDim: (
            cell_domain(h_grid.Zone.HALO),
            cell_domain(h_grid.Zone.LOCAL),
        ),
    },
    fields={"bdy_halo_c": "bdy_halo_c"},
)
fields_factory.register_provider(compute_bdy_halo_c_provider)


compute_hmask_dd3d_provider = factory.ProgramFieldProvider(
    func=mf.compute_hmask_dd3d,
    deps={
        "e_refin_ctrl": "e_refin_ctrl",
    },
    domain={
        dims.CellDim: (
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
            cell_domain(h_grid.Zone.LOCAL),
        )
    },
    fields={"hmask_dd3d": "hmask_dd3d"},
    params={
        "grf_nudge_start_e": gtx.int32(h_grid._GRF_NUDGEZONE_START_EDGES),
        "grf_nudgezone_width": gtx.int32(h_grid._GRF_NUDGEZONE_WIDTH),
    },
)
fields_factory.register_provider(compute_hmask_dd3d_provider)


compute_zdiff_gradp_dsl_provider = factory.NumpyFieldsProvider(
    func=compute_zdiff_gradp_dsl.compute_zdiff_gradp_dsl,
    domain={},
    fields=["zdiff_gradp"],
    deps={
        "z_me": "z_me",
        "z_mc": "height",
        "z_ifc": "height_on_interface_levels",
        "flat_idx_max": gtx.as_field(
            (dims.EdgeDim,),
            np.amax(compute_flat_idx_provider.fields()["flat_idx"].asnumpy(), axis=1),
            dtype=gtx.int32,
        ),
        "z_aux2": "z_aux2",
    },
    offsets={"e2c": dims.E2CDim},
    params={
        "nlev": icon_grid.num_levels,
        "horizontal_start": icon_grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        ),
        "horizontal_start_1": icon_grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2)),
        "nedges": icon_grid.num_edges,
    },
)
fields_factory.register_provider(compute_zdiff_gradp_dsl_provider)

compute_nudgecoeffs_provider = factory.ProgramFieldProvider(
    func=compute_nudgecoeffs.compute_nudgecoeffs,
    deps={
        "refin_ctrl": "e_refin_ctrl",
    },
    domain={
        dims.EdgeDim: (
            edge_domain(h_grid.Zone.NUDGING_LEVEL_2),
            edge_domain(h_grid.Zone.LOCAL),
        )
    },
    fields={"nudgecoeffs_e": "nudgecoeffs_e"},
    params={
        "grf_nudge_start_e": gtx.int32(h_grid._GRF_NUDGEZONE_START_EDGES),
        "nudge_max_coeffs": "nudge_max_coeffs",
        "nudge_efold_width": "nudge_efold_width",
        "nudge_zone_width": "nudge_zone_width",
    },
)
fields_factory.register_provider(compute_nudgecoeffs_provider)


compute_coeff_gradekin_provider = factory.NumpyFieldsProvider(
    func=compute_coeff_gradekin.compute_coeff_gradekin,
    domain={dims.EdgeDim: (0, icon_grid.num_edges)},
    fields=["coeff_gradekin"],
    deps={
        "edge_cell_length": "edge_cell_length",
        "inv_dual_edge_length": icon_grid.num_levels,
    },
    params={
        "horizontal_start": edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
        "horizontal_end": icon_grid.num_edges,
    },
)
fields_factory.register_provider(compute_coeff_gradekin_provider)


compute_wgtfacq_c_provider = factory.NumpyFieldsProvider(
    func=compute_wgtfacq.compute_wgtfacq_c_dsl,
    domain={dims.CellDim: (0, icon_grid.num_cells), dims.KDim: (0, icon_grid.num_levels)},
    fields=["weighting_factor_for_quadratic_interpolation_to_cell_surface"],
    deps={"z_ifc": "height_on_interface_levels"},
    params={"nlev": icon_grid.num_levels},
)

fields_factory.register_provider(compute_wgtfacq_c_provider)


compute_wgtfacq_e_provider = factory.NumpyFieldsProvider(
    func=compute_wgtfacq.compute_wgtfacq_e_dsl,
    deps={
        "z_ifc": "height_on_interface_levels",
        "c_lin_e": "cell_to_edge_interpolation_coefficient",
        "wgtfacq_c_dsl": "weighting_factor_for_quadratic_interpolation_to_cell_surface",
    },
    offsets={"e2c": dims.E2CDim},
    domain={dims.EdgeDim: (0, icon_grid.num_edges), dims.KDim: (0, icon_grid.num_levels)},
    fields=["weighting_factor_for_quadratic_interpolation_to_edge_center"],
    params={"n_edges": icon_grid.num_edges, "nlev": icon_grid.num_levels},
)

fields_factory.register_provider(compute_wgtfacq_e_provider)

compute_max_nbhgt_provider = factory.ProgramFieldProvider(
    func=mf.compute_max_nbhgt,
    deps={
        "z_mc_nlev": as_field(
            (dims.CellDim,), height_provider.fields()["z_mc"].asnumpy()[:, nlev - 1]
        ),
    },
    domain={
        dims.CellDim: (
            cell_domain(h_grid.Zone.NUDGING),
            icon_grid.num_cells,
        )
    },
    fields={"max_nbhgt": "max_nbhgt"},
)
fields_factory.register_provider(compute_max_nbhgt_provider)

compute_maxslp_maxhgtd_provider = factory.ProgramFieldProvider(
    func=mf.compute_maxslp_maxhgtd,
    deps={
        "ddxn_z_full": "ddxn_z_full",
        "dual_edge_length": "dual_edge_length",
    },
    domain={
        dims.CellDim: (
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
            icon_grid.num_cells,
        ),
        dims.KDim: (
            v_grid.domain(dims.KDim)(v_grid.Zone.TOP),
            v_grid.domain(dims.KDim)(v_grid.Zone.BOTTOM),
        ),
    },
    fields={"maxslp": "maxslp", "maxhgtd": "maxhgtd"},
)
fields_factory.register_provider(compute_maxslp_maxhgtd_provider)

compute_weighted_cell_neighbor_sum_provider = factory.ProgramFieldProvider(
    func=mf.compute_weighted_cell_neighbor_sum,
    deps={
        "maxslp": "maxslp",
        "maxhgtd": "maxhgtd",
        "c_bln_avg": "c_bln_avg",
    },
    domain={
        dims.CellDim: (
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
            icon_grid.num_cells,
        ),
        dims.KDim: (
            v_grid.domain(dims.KDim)(v_grid.Zone.TOP),
            v_grid.domain(dims.KDim)(v_grid.Zone.BOTTOM),
        ),
    },
    fields={"z_maxslp_avg": "z_maxslp_avg", "z_maxhgtd_avg": "z_maxhgtd_avg"},
)
fields_factory.register_provider(compute_weighted_cell_neighbor_sum_provider)

compute_diffusion_metrics_provider = factory.NumpyFieldsProvider(
    func=compute_diffusion_metrics.compute_diffusion_metrics,
    deps={
        "z_mc": "height",
        "max_nbhgt": "max_nbhgt",
        "c_owner_mask": "c_owner_mask",
        "z_maxslp_avg": "z_maxslp_avg",
        "z_maxhgtd_avg": "z_maxhgtd_avg",
    },
    offsets={"c2e2c": dims.C2E2CDim},
    domain={dims.CellDim: (0, icon_grid.num_cells), dims.KDim: (0, icon_grid.num_levels)},
    fields=["mask_hdiff", "zd_diffcoef_dsl", "zd_intcoef_dsl", "zd_vertoffset_dsl"],
    params={
        "thslp_zdiffu": thslp_zdiffu,
        "thhgtd_zdiffu": thhgtd_zdiffu,
        "n_c2e2c": icon_grid.connectivities[dims.C2E2CDim].shape[1],
        "cell_nudging": cell_domain(h_grid.Zone.NUDGING),
        "n_cells": icon_grid.num_cells,
        "nlev": icon_grid.num_levels,
    },
)

fields_factory.register_provider(compute_diffusion_metrics_provider)
