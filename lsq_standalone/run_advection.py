# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy
import pytest
import functools
from collections.abc import Callable, Mapping, Sequence
from gt4py import next as gtx
from gt4py.next import allocators as gtx_allocators, backend as gtx_backend
from mesh_generator import mesh_generator

from icon4py.model.atmosphere.advection import advection, advection_states
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.fixtures.datatest import (
#    backend,
    data_provider,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    processor_props,
    ranked_data_path
)

from model.atmosphere.advection.tests.advection.fixtures import advection_exit_savepoint, advection_init_savepoint
from model.atmosphere.advection.tests.advection.utils import (
    construct_config,
    construct_diagnostic_exit_state,
    construct_diagnostic_init_state,
    construct_interpolation_state,
    construct_least_squares_state,
    construct_metric_state,
    construct_prep_adv,
    log_serialized,
    verify_advection_fields
)
from icon4py.model.common.grid.base import HorizontalGridSize, GridConfig
from icon4py.model.common.grid import (
    geometry,
    geometry_attributes as geometry_attrs,
    horizontal as h_grid,
    icon,
    refinement,
    base
)
from icon4py.model.common.interpolation import (
    interpolation_attributes as attrs,
    interpolation_fields,
    rbf_interpolation as rbf,
    interpolation_attributes as interpolation_attrs
)
from icon4py.model.common.states import factory
import icon4py.model.common.grid.states as grid_states

# ntracer legend for the serialization data used here in test_advection:
# ------------------------------------
# ntracer          |  0, 1, 2, 3, 4 |
# ------------------------------------
# ivadv_tracer     |  3, 0, 0, 2, 3 |
# itype_hlimit     |  3, 4, 3, 0, 0 |
# itype_vlimit     |  1, 0, 0, 2, 1 |
# ihadv_tracer     | 52, 2, 2, 0, 0 |
# ------------------------------------

config = construct_config(
    horizontal_advection_type=advection.HorizontalAdvectionType.LINEAR_2ND_ORDER,
    horizontal_advection_limiter=advection.HorizontalAdvectionLimiter.NO_LIMITER,
    vertical_advection_type=advection.VerticalAdvectionType.NO_ADVECTION,
    vertical_advection_limiter=advection.VerticalAdvectionLimiter.NO_LIMITER
)
(nodes,
     c2v_table,
     e2v_table,
     v2c_table,
     v2e_table,
     e2c_table,
     e2c2v_table,
     c2e_table,
     e2c2e0_table,
     e2c2e_table,
     c2e2cO_table,
     c2e2c_table,
     c2e2c2e_table,
     c2e2c2e2c_table,
     cartesian_vertex_coordinates,
     cartesian_cell_centers,
     cartesian_edge_centers,
     primal_edge_length,
     edge_orientation,
     area
) = mesh_generator()
neighbor_tables = {
  dims.C2E2C: c2e2c_table,
  dims.C2E: c2e_table,
  dims.E2C: e2c_table,
  dims.V2E: v2e_table,
  dims.E2V: e2v_table,
  dims.V2C: v2c_table,
  dims.C2V: c2v_table,
  dims.E2C2E: e2c2e_table,
#  v2e2v_table
  dims.C2E2C2E: c2e2c2e_table
}
gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], float]
#h_grid_domain = h_grid.Domain(dims.VertexDim, dims.V2CDim)
h_grid_domain_start = h_grid.Domain(dims.VertexDim, h_grid.Zone.INTERIOR)
h_grid_domain_end = h_grid.Domain(dims.VertexDim, h_grid.Zone.END)
grid_config = base.GridConfig(
            horizontal_config=base.HorizontalGridSize(
                num_cells=2048,
                num_vertices=1024,
                num_edges=3072
            ),
            vertical_size=65,
            limited_area=False
        )
start_indices= Mapping[h_grid_domain_start, gtx.int32()]
end_indices= Mapping[h_grid_domain_end, gtx.int32()]
global_properties = icon.GlobalGridParams()
mesh = icon.icon_grid(
    id_ = 'void',
#    allocator= gtx_allocators.FieldBufferAllocationUtil,
    allocator= None,
    config= grid_config,
    neighbor_tables= neighbor_tables,
    start_indices= start_indices,
    end_indices= end_indices,
    global_properties= global_properties,
    refinement_control= None
)
geofac_div = data_alloc.zero_field(mesh, dims.CellDim, dims.C2EDim)
primal_edge_length = gtx.as_field((dims.EdgeDim,), primal_edge_length)
edge_orientation = gtx.as_field((dims.CellDim, dims.C2EDim), edge_orientation)
area = gtx.as_field((dims.CellDim,), area)
geofac_div = interpolation_fields.compute_geofac_div(primal_edge_length, edge_orientation, area, out= geofac_div, offset_provider={"C2E": mesh.get_connectivity("C2E")})
backend = gtx_backend
backend = None
backend = model_backends.BACKENDS["gtfn_cpu"]
_xp = data_alloc.import_array_ns(backend)
characteristic_length = 11.1
_config = {
            "divavg_cntrwgt": 0.5,
            "weighting_factor": 0.0,
            "max_nudging_coefficient": 0.375,
            "nudge_efold_width": 2.0,
            "nudge_zone_width": 10,
            "rbf_kernel_cell": rbf.DEFAULT_RBF_KERNEL[rbf.RBFDimension.CELL],
            "rbf_kernel_edge": rbf.DEFAULT_RBF_KERNEL[rbf.RBFDimension.EDGE],
            "rbf_kernel_vertex": rbf.DEFAULT_RBF_KERNEL[rbf.RBFDimension.VERTEX],
            "rbf_scale_cell": rbf.compute_default_rbf_scale(
                characteristic_length, rbf.RBFDimension.CELL
            ),
            "rbf_scale_edge": rbf.compute_default_rbf_scale(
                characteristic_length, rbf.RBFDimension.EDGE
            ),
            "rbf_scale_vertex": rbf.compute_default_rbf_scale(
                characteristic_length, rbf.RBFDimension.VERTEX
            )
}
#rbf_vec_coeff_e = factory.NumpyFieldsProvider(
#            func=functools.partial(rbf.compute_rbf_interpolation_coeffs_edge, array_ns=_xp),
#            fields=(attrs.RBF_VEC_COEFF_E),
#            domain=(dims.CellDim, dims.E2C2EDim),
#            deps={
#                "edge_lat": geometry_attrs.EDGE_LAT,
#                "edge_lon": geometry_attrs.EDGE_LON,
#                "edge_center_x": geometry_attrs.EDGE_CENTER_X,
#                "edge_center_y": geometry_attrs.EDGE_CENTER_Y,
#                "edge_center_z": geometry_attrs.EDGE_CENTER_Z,
#                "edge_normal_x": geometry_attrs.EDGE_NORMAL_X,
#                "edge_normal_y": geometry_attrs.EDGE_NORMAL_Y,
#                "edge_normal_z": geometry_attrs.EDGE_NORMAL_Z,
#                "edge_dual_normal_u": geometry_attrs.EDGE_DUAL_U,
#                "edge_dual_normal_v": geometry_attrs.EDGE_DUAL_V
#            },
#            connectivities={"rbf_offset": dims.E2C2EDim},
#            params={
#                "rbf_kernel": _config["rbf_kernel_edge"].value,
#                "scale_factor": _config["rbf_scale_edge"],
#                "horizontal_start": mesh.start_index(
#                    domain=h_grid.Domain(dims.EdgeDim, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
#                )
##                    domain=(dims.CellDim, dims.E2C2EDim))
#            }
#)
rbf_vec_coeff_e = rbf.compute_rbf_interpolation_coeffs_edge(
            edge_lat = geometry_attrs.EDGE_LAT,
            edge_lon = geometry_attrs.EDGE_LON,
            edge_center_x = geometry_attrs.EDGE_CENTER_X,
            edge_center_y = geometry_attrs.EDGE_CENTER_Y,
            edge_center_z = geometry_attrs.EDGE_CENTER_Z,
            edge_normal_x = geometry_attrs.EDGE_NORMAL_X,
            edge_normal_y = geometry_attrs.EDGE_NORMAL_Y,
            edge_normal_z = geometry_attrs.EDGE_NORMAL_Z,
            edge_dual_normal_u = geometry_attrs.EDGE_DUAL_U,
            edge_dual_normal_v = geometry_attrs.EDGE_DUAL_V,
#            rbf_offset = rbf.construct_rbf_matrix_offsets_tables_for_edges(mesh),
            rbf_offset = dims.E2C2EDim,
            rbf_kernel = _config["rbf_kernel_edge"].value,
            scale_factor = _config["rbf_scale_edge"],
            horizontal_start = mesh.start_index(
                    domain=h_grid.Domain(dims.EdgeDim, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
            )
)
#rbf_vec_coeff_e = data_alloc.random_field(mesh, dims.EdgeDim, dims.E2C2EDim)
pos_on_tplane_e_x = cartesian_edge_centers[:, 0]
pos_on_tplane_e_y = cartesian_edge_centers[:, 1]
pos_on_tplane_e_1 = cartesian_vertex_coordinates[e2v_table[:, 0], :]
pos_on_tplane_e_2 = cartesian_vertex_coordinates[e2v_table[:, 1], :]
interpolation_state = advection_states.AdvectionInterpolationState(
    geofac_div=geofac_div,
    rbf_vec_coeff_e=rbf_vec_coeff_e,
    pos_on_tplane_e_1=pos_on_tplane_e_1,
    pos_on_tplane_e_2=pos_on_tplane_e_2
)
least_squares_state = advection_states.AdvectionLeastSquaresState(
        lsq_pseudoinv_1=1,
        lsq_pseudoinv_2=2
)
constant_f = data_alloc.constant_field(mesh, 1.0, dims.KDim)
#ddqz_z_full_np = numpy.reciprocal(savepoint.inv_ddqz_z_full().asnumpy())
ddqz_z_full_np = f"inverse_of_{interpolation_attrs.C_LIN_E}"
ddqz_z_full_np = 0
metric_state = advection_states.AdvectionMetricState(
        deepatmo_divh=constant_f,
        deepatmo_divzl=constant_f,
        deepatmo_divzu=constant_f,
#        ddqz_z_full=gtx.as_field((dims.CellDim, dims.KDim), ddqz_z_full_np, allocator=backend),
        ddqz_z_full=None
)
tangent_orientation = None
tangent_orientation = edge_orientation
inverse_primal_edge_lengths = f"inverse_of_{geometry_attrs.EDGE_LENGTH}"
inverse_dual_edge_lengths = f"inverse_of_{geometry_attrs.DUAL_EDGE_LENGTH}"
inverse_vertex_vertex_lengths = geometry_attrs.INVERSE_VERTEX_VERTEX_LENGTH
primal_normal_vert_x = geometry_attrs.VERTEX_X
primal_normal_vert_y = geometry_attrs.VERTEX_Y
dual_normal_vert_x = geometry_attrs.EDGE_TANGENT_VERTEX_U
dual_normal_vert_y = geometry_attrs.EDGE_TANGENT_VERTEX_V
primal_normal_cell_x = geometry_attrs.CELL_CENTER_X
primal_normal_cell_y = geometry_attrs.CELL_CENTER_Y
dual_normal_cell_x = geometry_attrs.EDGE_DUAL_U #?
dual_normal_cell_y = geometry_attrs.EDGE_DUAL_V #?
edge_areas = geometry_attrs.EDGE_AREA
coriolis_frequency = None
edge_center_lat = None
edge_center_lon = None
primal_normal_x = None
primal_normal_y = None
#coriolis_frequency = 0e0
#edge_center_lat = 0e0
#edge_center_lon = 0e0
#primal_normal_x = 0e0
#primal_normal_y = 0e0
edge_params = grid_states.EdgeParams (
            tangent_orientation=tangent_orientation,
            inverse_primal_edge_lengths=inverse_primal_edge_lengths,
            inverse_dual_edge_lengths=inverse_dual_edge_lengths,
            inverse_vertex_vertex_lengths=inverse_vertex_vertex_lengths,
            primal_normal_vert_x=primal_normal_vert_x,
            primal_normal_vert_y=primal_normal_vert_y,
            dual_normal_vert_x=dual_normal_vert_x,
            dual_normal_vert_y=dual_normal_vert_y,
            primal_normal_cell_x=primal_normal_cell_x,
            dual_normal_cell_x=dual_normal_cell_x,
            primal_normal_cell_y=primal_normal_cell_y,
            dual_normal_cell_y=dual_normal_cell_y,
            edge_areas=edge_areas,
            coriolis_frequency=coriolis_frequency,
            edge_center_lat=edge_center_lat,
            edge_center_lon=edge_center_lon,
            primal_normal_x=primal_normal_x,
            primal_normal_y=primal_normal_y
)
cell_center_lat = geometry_attrs.CELL_LAT
cell_center_lon = geometry_attrs.CELL_LON
cell_params = grid_states.CellParams (
            cell_center_lat=cell_center_lat,
            cell_center_lon=cell_center_lon,
            area=area
)

advection_granule = advection.convert_config_to_advection(
    config=config,
    grid=mesh,
    interpolation_state=interpolation_state,
    least_squares_state=least_squares_state,
    metric_state=metric_state,
    edge_params=edge_params,
    cell_params=cell_params,
    backend=backend
)

diagnostic_state = advection_states.AdvectionDiagnosticState (
        airmass_now= data_alloc.constant_field(mesh, 1e0, dims.CellDim, dims.KDim, backend=backend),
        airmass_new= data_alloc.constant_field(mesh, 1e0, dims.CellDim, dims.KDim, backend=backend),
        grf_tend_tracer=None,
        hfl_tracer=data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, backend=backend),
        vfl_tracer=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, backend=backend
        )
)
prep_adv = advection_states.AdvectionPrepAdvState(
        vn_traj=data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, backend=backend),
        mass_flx_me=data_alloc.zero_field(mesh, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, backend=backend),
        mass_flx_ic=data_alloc.zero_field(mesh, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, backend=backend)
)
p_tracer_now = data_alloc.zero_field(mesh, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, backend=backend)
p_tracer_new = data_alloc.zero_field(mesh, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, backend=backend)
dtime = 0e0

advection_granule.run(
    diagnostic_state=diagnostic_state,
    prep_adv=prep_adv,
    p_tracer_now=p_tracer_now,
    p_tracer_new=p_tracer_new,
    dtime=dtime
)
