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

import icon4py.model.common.states.factory as factory
from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro import (
    HorizontalPressureDiscretizationType,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal
from icon4py.model.common.io import cf_utils
from icon4py.model.common.metrics import metric_fields as mf
from icon4py.model.common.settings import xp
from icon4py.model.common.test_utils import datatest_utils as dt_utils, serialbox_utils as sb


# we need to register a couple of fields from the serializer. Those should get replaced one by one.

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
#######

# start build up factory:

# used for vertical domain below: should go away once vertical grid provids start_index and end_index like interface
grid = grid_savepoint.global_grid_params

interface_model_height = metrics_savepoint.z_ifc()
c_lin_e = interpolation_savepoint.c_lin_e()
k_index = gtx.as_field((dims.KDim,), xp.arange(nlev + 1, dtype=gtx.int32))
vct_a = grid_savepoint.vct_a()
theta_ref_mc = metrics_savepoint.theta_ref_mc()
exner_ref_mc = metrics_savepoint.exner_ref_mc()
wgtfac_c = metrics_savepoint.wgtfac_c()
c_refin_ctrl = grid_savepoint.refin_ctrl(dims.CellDim)
e_refin_ctrl = grid_savepoint.refin_ctrl(dims.EdgeDim)
dual_edge_length = grid_savepoint.dual_edge_length()
tangent_orientation = grid_savepoint.tangent_orientation()
inv_primal_edge_length = grid_savepoint.inverse_primal_edge_lengths()
cells_aw_verts = interpolation_savepoint.c_intp().asnumpy()
cells_aw_verts_field = gtx.as_field((dims.VertexDim, dims.V2CDim), cells_aw_verts)

fields_factory = factory.FieldsFactory()

fields_factory.register_provider(
    factory.PrecomputedFieldsProvider(
        {
            "height_on_interface_levels": interface_model_height,
            "cell_to_edge_interpolation_coefficient": c_lin_e,
            cf_utils.INTERFACE_LEVEL_STANDARD_NAME: k_index,
            "vct_a": vct_a,
            "theta_ref_mc": theta_ref_mc,
            "exner_ref_mc": exner_ref_mc,
            "wgtfac_c": wgtfac_c,
            "c_refin_ctrl": c_refin_ctrl,
            "e_refin_ctrl": e_refin_ctrl,
            "dual_edge_length": dual_edge_length,
            "tangent_orientation": tangent_orientation,
            "inv_primal_edge_length": inv_primal_edge_length,
            "cells_aw_verts_field": cells_aw_verts_field,
        }
    )
)


height_provider = factory.ProgramFieldProvider(
    func=mf.compute_z_mc,
    domain={
        dims.CellDim: (
            horizontal.HorizontalMarkerIndex.local(dims.CellDim),
            horizontal.HorizontalMarkerIndex.end(dims.CellDim),
        ),
        dims.KDim: (0, nlev),
    },
    fields={"z_mc": "height"},
    deps={"z_ifc": "height_on_interface_levels"},
)
fields_factory.register_provider(height_provider)

ddqz_z_full_and_inverse_provider = factory.ProgramFieldProvider(
    func=mf.compute_ddqz_z_full_and_inverse,
    deps={
        "z_ifc": "height_on_interface_levels",
    },
    domain={
        dims.CellDim: (
            horizontal.HorizontalMarkerIndex.local(dims.CellDim),
            horizontal.HorizontalMarkerIndex.end(dims.CellDim),
        ),
        dims.KDim: (0, nlev),
    },
    fields={"ddqz_z_full": "ddqz_z_full", "inv_ddqz_z_full": "inv_ddqz_z_full"},
)
fields_factory.register_provider(ddqz_z_full_and_inverse_provider)

compute_ddqz_z_half_provider = factory.ProgramFieldProvider(
    func=mf.compute_ddqz_z_half,
    deps={
        "z_ifc": "height_on_interface_levels",
        "z_mc": "height",
        "k_index": cf_utils.INTERFACE_LEVEL_STANDARD_NAME,
    },
    domain={
        dims.CellDim: (
            horizontal.HorizontalMarkerIndex.local(dims.CellDim),
            horizontal.HorizontalMarkerIndex.end(dims.CellDim),
        ),
        dims.KDim: (0, nlev + 1),
    },
    fields={"ddqz_z_half": "ddqz_z_half"},
    params={"nlev": nlev},
)
fields_factory.register_provider(compute_ddqz_z_half_provider)

# TODO: this should include experiment param as in test_metric_fields
damping_height = 50000.0 if dt_utils.GLOBAL_EXPERIMENT else 12500.0
rayleigh_coeff = 0.1 if dt_utils.GLOBAL_EXPERIMENT else 5.0
vct_a_1 = grid_savepoint.vct_a().asnumpy()[0]

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
            horizontal.HorizontalMarkerIndex.local(dims.CellDim),
            horizontal.HorizontalMarkerIndex.end(dims.CellDim),
        ),
        dims.KDim: (1, nlev),
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
            horizontal.HorizontalMarkerIndex.local(dims.CellDim),
            horizontal.HorizontalMarkerIndex.end(dims.CellDim),
        ),
        dims.KDim: (0, nlev),
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
            horizontal.HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 1,
            horizontal.HorizontalMarkerIndex.end(dims.VertexDim)
            - 1,  # TODO: upper bound is lateral boundary as well
        ),
        dims.KDim: (0, nlev + 1),
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
            horizontal.HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 2,
            horizontal.HorizontalMarkerIndex.end(dims.EdgeDim)
            - 1,  # TODO: upper bound is lateral boundary as well
        ),
        dims.KDim: (0, nlev + 1),
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

compute_exner_exfac_provider = factory.ProgramFieldProvider(
    func=mf.compute_exner_exfac,
    deps={
        "ddxn_z_full": "ddxn_z_full",
        "dual_edge_length": "dual_edge_length",
    },
    domain={
        dims.CellDim: (
            horizontal.HorizontalMarkerIndex.lateral_boundary(dims.CellDim) + 1,
            horizontal.HorizontalMarkerIndex.end(dims.CellDim),
        ),
        dims.KDim: (0, nlev),
    },
    fields={"exner_exfac": "exner_exfac"},
    params={"exner_expol": "exner_expol"},
)
fields_factory.register_provider(compute_exner_exfac_provider)

# # TODO: need to do compute_vwind_impl_wgt first
# compute_vwind_expl_wgt_provider = factory.ProgramFieldProvider(
#     func=mf.compute_vwind_expl_wgt,
#     deps={
#         "vwind_impl_wgt": "vwind_impl_wgt",
#     },
#     domain={
#         dims.CellDim: (
#             horizontal.HorizontalMarkerIndex.local(dims.CellDim),
#             horizontal.HorizontalMarkerIndex.end(dims.CellDim),
#         ),
#     },
#     fields={"vwind_expl_wgt": "vwind_expl_wgt"},
# )


compute_wgtfac_e_provider = factory.ProgramFieldProvider(
    func=mf.compute_wgtfac_e,
    deps={
        "wgtfac_c": "wgtfac_c",
        "c_lin_e": "cell_to_edge_interpolation_coefficient",
    },
    domain={
        dims.CellDim: (
            horizontal.HorizontalMarkerIndex.local(dims.EdgeDim),
            horizontal.HorizontalMarkerIndex.end(dims.EdgeDim),  # TODO: check this bound
        ),
        dims.KDim: (0, nlev + 1),
    },
    fields={"wgtfac_e": "wgtfac_e"},
)
fields_factory.register_provider(compute_wgtfac_e_provider)


# TODO: lots of dependencies
# compute_pg_edgeidx_dsl_provider = factory.ProgramFieldProvider(
#     func=mf.compute_hmask_dd3d,
#     deps={
#         "c_refin_ctrl": "c_refin_ctrl",
#     },
#     domain={
#         dims.CellDim: (
#             horizontal.HorizontalMarkerIndex.local(dims.EdgeDim),
#             horizontal.HorizontalMarkerIndex.end(dims.EdgeDim),
#         ),
#         dims.KDim: (0, nlev),
#     },
#     fields={"pg_edgeidx_dsl": "pg_edgeidx_dsl"},
# )


compute_mask_prog_halo_c_provider = factory.ProgramFieldProvider(
    func=mf.compute_mask_prog_halo_c,
    deps={
        "c_refin_ctrl": "c_refin_ctrl",
    },
    domain={
        dims.CellDim: (
            horizontal.HorizontalMarkerIndex.local(dims.CellDim) - 1,
            horizontal.HorizontalMarkerIndex.end(dims.CellDim),
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
            horizontal.HorizontalMarkerIndex.local(dims.CellDim) - 1,
            horizontal.HorizontalMarkerIndex.end(dims.CellDim),
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
            horizontal.HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim),
            horizontal.HorizontalMarkerIndex.end(dims.EdgeDim),  # TODO: check this bound
        ),
    },
    fields={"hmask_dd3d": "hmask_dd3d"},
    params={
        "grf_nudge_start_e": gtx.int32(horizontal._GRF_NUDGEZONE_START_EDGES),
        "grf_nudgezone_width": gtx.int32(horizontal._GRF_NUDGEZONE_WIDTH),
    },
)
fields_factory.register_provider(compute_hmask_dd3d_provider)
