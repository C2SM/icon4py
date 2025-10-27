# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import functools
import logging
import pathlib

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import netCDF4 as nc4

from icon4py.model.atmosphere.diffusion import diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.common import dimension as dims, field_type_aliases as fa, utils as common_utils
from icon4py.model.common.decomposition import (
    definitions as decomposition,
    mpi_decomposition as mpi_decomp,
)
from icon4py.model.common.grid import (
    base,
    icon as icon_grid,
    states as grid_states,
    vertical as v_grid,
    grid_manager as gm,
    geometry as grid_geometry,
    geometry_attributes as geometry_meta,
)
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver.testcases import gauss3d, jablonowski_williamson # TODO (Yilu) should be get rid of
from icon4py.model.testing import definitions, grid_utils, serialbox as sb

from model.common.src.icon4py.model.common.constants import RayleighType

log = logging.getLogger(__name__)

_LOGGING_LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "critical": logging.CRITICAL,
}


def read_icon_grid(
    path: pathlib.Path,
    grid_file: pathlib.Path,
    backend: gtx_typing.Backend,
    rank: int = 0,
    ser_type: SerializationType = SerializationType.SB,
) -> icon_grid.IconGrid:
    """
    Read icon grid.

    Args:
        path: path where to find the input data
        grid_file: path of the grid
        backend: GT4Py backend
        rank: mpi rank of the current compute node
        ser_type: type of input data. Currently only 'sb (serialbox)' is supported. It reads from ppser serialized test data
    Returns:  IconGrid parsed from a given input type.
    """
    if ser_type == SerializationType.SB:
        return _grid_savepoint(
            backend=backend,
            path=path,
            grid_file=grid_file,
            rank=rank,
        ).construct_icon_grid(backend=backend)
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def read_initial_state(
    grid: icon_grid.IconGrid,
    cell_param: grid_states.CellParams,
    edge_param: grid_states.EdgeParams,
    path: pathlib.Path,
    backend: gtx_typing.Backend,
    rank=0,
) -> tuple[
    diffusion_states.DiffusionDiagnosticState,
    dycore_states.DiagnosticStateNonHydro,
    dycore_states.PrepAdvection,
    float,
    diagnostics.DiagnosticState,
    prognostics.PrognosticState,
    prognostics.PrognosticState,
]:
    """
    Read initial prognostic and diagnostic fields for the jablonowski_williamson test case.

    Args:
        grid: IconGrid
        cell_param: cell properties
        edge_param: edge properties
        path: path to the serialized input data
        backend: GT4Py backend
        rank: mpi rank of the current compute node
        experiment_type: (optional) defaults to ANY=any, type of initial condition to be read

    Returns:  A tuple containing Diagnostic variables for diffusion and solve_nonhydro granules,
        PrepAdvection, second order divdamp factor, diagnostic variables, and two prognostic
        variables (now and next).
    """

    (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        divdamp_fac_o2,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    ) = jablonowski_williamson.model_initialization_jabw(
        grid=grid,
        cell_param=cell_param,
        edge_param=edge_param,
        path=path,
        backend=backend,
        rank=rank,
    )

    return (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        divdamp_fac_o2,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    )

# TODO (Yilu) grid or grid_manager can be fixtures?
def read_geometry_fields(
    grid: icon_grid.IconGrid,
    vertical_grid_config: v_grid.VerticalGridConfig,
    backend: gtx_typing.Backend,
)-> tuple[
    decomposition.DecompositionInfo,
    grid_geometry.GridGeometry,
    grid_states.EdgeParams,
    grid_states.CellParams,
    v_grid.VerticalGrid,
    fa.CellField[bool],
]:
    """
        Read geometry fields containing grid properties, not from serialbox data.

        Args:
            grid: grid file
            vertical_grid_config: Vertical grid configuration
            backend: GT4py backend

        Returns: a tuple containing fields describing edges, cells, vertical properties of the model
            the data is originally obtained from the grid file (horizontal fields) or some special input files.
    """

    grid_manager = grid_utils.get_grid_manager_from_identifier(
        grid=grid, num_levels=80, keep_skip_values=True, backend=backend
    )
    mesh = grid_manager.mesh

    coordinates = mesh.coordinates
    geometry_input_fields = mesh.geometry_fields

    decomposition_info = grid_utils.construct_decomposition_info(mesh, backend)

    geometry_field_source = grid_geometry.GridGeometry(
        grid=mesh,
        decomposition_info=decomposition_info,
        backend=backend,
        coordinates=coordinates,
        extra_fields=geometry_input_fields,
        metadata=geometry_meta.attrs,
    )
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_grid_config, backend)

    vertical_grid = v_grid.VerticalGrid(
        config=vertical_grid_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )

    cell_geometry = grid_states.CellParams(
        cell_center_lat=geometry_field_source.get(geometry_meta.CELL_LAT),
        cell_center_lon=geometry_field_source.get(geometry_meta.CELL_LON),
        area=geometry_field_source.get(geometry_meta.CELL_AREA),
    )
    edge_geometry = grid_states.EdgeParams(
        tangent_orientation=geometry_field_source.get(geometry_meta.TANGENT_ORIENTATION),
        inverse_primal_edge_lengths=geometry_field_source.get(
            f"inverse_of_{geometry_meta.EDGE_LENGTH}"
        ),
        inverse_dual_edge_lengths=geometry_field_source.get(
            f"inverse_of_{geometry_meta.DUAL_EDGE_LENGTH}"
        ),
        inverse_vertex_vertex_lengths=geometry_field_source.get(
            f"inverse_of_{geometry_meta.VERTEX_VERTEX_LENGTH}"
        ),
        primal_normal_vert_x=geometry_field_source.get(geometry_meta.EDGE_NORMAL_VERTEX_U),
        primal_normal_vert_y=geometry_field_source.get(geometry_meta.EDGE_NORMAL_VERTEX_V),
        dual_normal_vert_x=geometry_field_source.get(geometry_meta.EDGE_TANGENT_VERTEX_U),
        dual_normal_vert_y=geometry_field_source.get(geometry_meta.EDGE_NORMAL_VERTEX_V),
        primal_normal_cell_x=geometry_field_source.get(geometry_meta.EDGE_NORMAL_CELL_U),
        dual_normal_cell_x=geometry_field_source.get(geometry_meta.EDGE_TANGENT_CELL_U),
        primal_normal_cell_y=geometry_field_source.get(geometry_meta.EDGE_NORMAL_CELL_V),
        dual_normal_cell_y=geometry_field_source.get(geometry_meta.EDGE_TANGENT_CELL_V),
        edge_areas=geometry_field_source.get(geometry_meta.EDGE_AREA),
        coriolis_frequency=geometry_field_source.get(geometry_meta.CORIOLIS_PARAMETER),
        edge_center_lat=geometry_field_source.get(geometry_meta.EDGE_LAT),
        edge_center_lon=geometry_field_source.get(geometry_meta.EDGE_LON),
        primal_normal_x=geometry_field_source.get(geometry_meta.EDGE_NORMAL_U),
        primal_normal_y=geometry_field_source.get(geometry_meta.EDGE_NORMAL_V),
    )

    c_owner_mask =gtx.as_field(
            (dims.CellDim,),
            decomposition_info.owner_mask(dims.CellDim),
            allocator=backend,  # type: ignore[arg-type]
        ),

    return decomposition_info, geometry_field_source, edge_geometry, cell_geometry, vertical_grid, c_owner_mask


# TODO (Yilu) can be deleted?
# TODO(OngChia): cannot be cached (@functools.cache) after adding backend. TypeError: unhashable type: 'CompiledbFactory'
def _serial_data_provider(
    backend: gtx_typing.Backend,
    path: pathlib.Path,
    rank: int,
) -> sb.IconSerialDataProvider:
    return sb.IconSerialDataProvider(
        backend=backend,
        fname_prefix="icon_pydycore",
        path=str(path.absolute()),
        do_print=False,
        mpi_rank=rank,
    )


# TODO(OngChia): cannot be cached (@functools.cache) after adding backend. TypeError: unhashable type: 'CompiledbFactory'
def _grid_savepoint(
    backend: gtx_typing.Backend,
    path: pathlib.Path,
    grid_file: pathlib.Path,
    rank: int,
) -> sb.IconGridSavepoint:
    global_grid_params, grid_uuid = _create_grid_global_params(grid_file)
    sp = _serial_data_provider(backend, path, rank).from_savepoint_grid(
        grid_uuid,
        global_grid_params.grid_shape,
    )
    return sp

def read_static_fields(
    grid: icon_grid.IconGrid,
    vertical_grid_config: v_grid.VerticalGridConfig,
    backend: gtx_typing.Backend,
) -> tuple[
    diffusion_states.DiffusionMetricState,
    diffusion_states.DiffusionInterpolationState,
    dycore_states.MetricStateNonHydro,
    dycore_states.InterpolationState,
    diagnostics.DiagnosticMetricState,
]:
    """
    Read fields for metric and interpolation state.

     Args:
        path: path to the serialized input data
        grid_file: path of the grid
        backend: GT4Py backend
        rank: mpi rank, defaults to 0 for serial run
        ser_type: (optional) defaults to SB=serialbox, type of input data to be read

    Returns:
        a tuple containing the metric_state and interpolation state,
        the fields are precalculated in the icon setup.

    """
    grid_manager = grid_utils.get_grid_manager_from_identifier(
        grid=grid, num_levels=80, keep_skip_values=True, backend=backend
    )
    mesh = grid_manager.mesh

    (
        decomposition_info,
        geometry_field_source,
        edge_param,
        cell_param,
        vertical_grid,
        c_owner_mask,
    ) = read_geometry_fields(
        grid=grid,
        vertical_grid_config=vertical_grid_config,
        backend=backend)

    interpolation_field_source = interpolation_factory.InterpolationFieldsFactory(
        grid=mesh,
        decomposition_info=decomposition_info,
        geometry_source=geometry_field_source,
        backend=backend,
        metadata=interpolation_attributes.attrs,
    )

    metrics_field_source = metrics_factory.MetricsFieldsFactory(
        grid=mesh,
        vertical_grid=vertical_grid,
        decomposition_info=decomposition_info,
        geometry_source=geometry_field_source,
        topography=gtx.as_field((dims.CellDim,), data=topo_c), # TODO (Yilu) shall I use the old topo_c for jbw?
        interpolation_source=interpolation_field_source,
        backend=backend,
        metadata=metrics_attributes.attrs,
        rayleigh_type=RayleighType.KLEMP,
        rayleigh_coeff=5.0,
        exner_expol=0.333,
        vwind_offctr=0.2,
    )

    diffusion_interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=interpolation_field_source.get(interpolation_attributes.E_BLN_C_S),
        rbf_coeff_1=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_V1),
        rbf_coeff_2=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_V2),
        geofac_div=interpolation_field_source.get(interpolation_attributes.GEOFAC_DIV),
        geofac_n2s=interpolation_field_source.get(interpolation_attributes.GEOFAC_N2S),
        geofac_grg_x=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_X),
        geofac_grg_y=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_Y),
        nudgecoeff_e=interpolation_field_source.get(interpolation_attributes.NUDGECOEFFS_E),
    )
    diffusion_metric_state = diffusion_states.DiffusionMetricState(
        mask_hdiff=metrics_field_source.get(metrics_attributes.MASK_HDIFF),
        theta_ref_mc=metrics_field_source.get(metrics_attributes.THETA_REF_MC),
        wgtfac_c=metrics_field_source.get(metrics_attributes.WGTFAC_C),
        zd_intcoef=metrics_field_source.get(metrics_attributes.ZD_INTCOEF_DSL),
        zd_vertoffset=metrics_field_source.get(metrics_attributes.ZD_VERTOFFSET_DSL),
        zd_diffcoef=metrics_field_source.get(metrics_attributes.ZD_DIFFCOEF_DSL),
    )

    solve_nonhydro_interpolation_state = dycore_states.InterpolationState(
        c_lin_e=interpolation_field_source.get(interpolation_attributes.C_LIN_E),
        c_intp=interpolation_field_source.get(interpolation_attributes.CELL_AW_VERTS),
        e_flx_avg=interpolation_field_source.get(interpolation_attributes.E_FLX_AVG),
        geofac_grdiv=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRDIV),
        geofac_rot=interpolation_field_source.get(interpolation_attributes.GEOFAC_ROT),
        pos_on_tplane_e_1=interpolation_field_source.get(
            interpolation_attributes.POS_ON_TPLANE_E_X
        ),
        pos_on_tplane_e_2=interpolation_field_source.get(
            interpolation_attributes.POS_ON_TPLANE_E_Y
        ),
        rbf_vec_coeff_e=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_E),
        e_bln_c_s=interpolation_field_source.get(interpolation_attributes.E_BLN_C_S),
        rbf_coeff_1=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_V1),
        rbf_coeff_2=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_V2),
        geofac_div=interpolation_field_source.get(interpolation_attributes.GEOFAC_DIV),
        geofac_n2s=interpolation_field_source.get(interpolation_attributes.GEOFAC_N2S),
        geofac_grg_x=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_X),
        geofac_grg_y=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_Y),
        nudgecoeff_e=interpolation_field_source.get(interpolation_attributes.NUDGECOEFFS_E),
    )

    solve_nonhydro_metric_state = dycore_states.MetricStateNonHydro(
        bdy_halo_c=metrics_field_source.get(metrics_attributes.BDY_HALO_C),
        mask_prog_halo_c=metrics_field_source.get(metrics_attributes.MASK_PROG_HALO_C),
        rayleigh_w=metrics_field_source.get(metrics_attributes.RAYLEIGH_W),
        time_extrapolation_parameter_for_exner=metrics_field_source.get(
            metrics_attributes.EXNER_EXFAC
        ),
        reference_exner_at_cells_on_model_levels=metrics_field_source.get(
            metrics_attributes.EXNER_REF_MC
        ),
        wgtfac_c=metrics_field_source.get(metrics_attributes.WGTFAC_C),
        wgtfacq_c=metrics_field_source.get(metrics_attributes.WGTFACQ_C),
        inv_ddqz_z_full=metrics_field_source.get(metrics_attributes.INV_DDQZ_Z_FULL),
        reference_rho_at_cells_on_model_levels=metrics_field_source.get(
            metrics_attributes.RHO_REF_MC
        ),
        reference_theta_at_cells_on_model_levels=metrics_field_source.get(
            metrics_attributes.THETA_REF_MC
        ),
        exner_w_explicit_weight_parameter=metrics_field_source.get(
            metrics_attributes.EXNER_W_EXPLICIT_WEIGHT_PARAMETER
        ),
        ddz_of_reference_exner_at_cells_on_half_levels=metrics_field_source.get(
            metrics_attributes.D_EXNER_DZ_REF_IC
        ),
        ddqz_z_half=metrics_field_source.get(metrics_attributes.DDQZ_Z_HALF),
        reference_theta_at_cells_on_half_levels=metrics_field_source.get(
            metrics_attributes.THETA_REF_IC
        ),
        d2dexdz2_fac1_mc=metrics_field_source.get(metrics_attributes.D2DEXDZ2_FAC1_MC),
        d2dexdz2_fac2_mc=metrics_field_source.get(metrics_attributes.D2DEXDZ2_FAC1_MC),
        reference_rho_at_edges_on_model_levels=metrics_field_source.get(
            metrics_attributes.RHO_REF_ME
        ),
        reference_theta_at_edges_on_model_levels=metrics_field_source.get(
            metrics_attributes.THETA_REF_ME
        ),
        ddxn_z_full=metrics_field_source.get(metrics_attributes.DDXN_Z_FULL),
        zdiff_gradp=metrics_field_source.get(metrics_attributes.ZDIFF_GRADP),
        vertoffset_gradp=metrics_field_source.get(metrics_attributes.VERTOFFSET_GRADP),
        nflat_gradp=metrics_field_source.get(metrics_attributes.NFLAT_GRADP),
        pg_edgeidx_dsl=metrics_field_source.get(metrics_attributes.PG_EDGEIDX_DSL),
        pg_exdist=metrics_field_source.get(metrics_attributes.PG_EDGEDIST_DSL),
        ddqz_z_full_e=metrics_field_source.get(metrics_attributes.DDQZ_Z_FULL_E),
        ddxt_z_full=metrics_field_source.get(metrics_attributes.DDXT_Z_FULL),
        wgtfac_e=metrics_field_source.get(metrics_attributes.WGTFAC_E),
        wgtfacq_e=metrics_field_source.get(metrics_attributes.WGTFACQ_E),
        exner_w_implicit_weight_parameter=metrics_field_source.get(
            metrics_attributes.EXNER_W_IMPLICIT_WEIGHT_PARAMETER
        ),
        horizontal_mask_for_3d_divdamp=metrics_field_source.get(
            metrics_attributes.HORIZONTAL_MASK_FOR_3D_DIVDAMP
        ),
        scaling_factor_for_3d_divdamp=metrics_field_source.get(
            metrics_attributes.SCALING_FACTOR_FOR_3D_DIVDAMP
        ),
        coeff1_dwdz=metrics_field_source.get(metrics_attributes.COEFF1_DWDZ),
        coeff2_dwdz=metrics_field_source.get(metrics_attributes.COEFF2_DWDZ),
        coeff_gradekin=metrics_field_source.get(metrics_attributes.COEFF_GRADEKIN),
    )

    diagnostic_metric_state = diagnostics.DiagnosticMetricState(
        ddqz_z_full=metrics_field_source.get(metrics_attributes.DDQZ_Z_FULL),
        rbf_vec_coeff_c1=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_C1),
        rbf_vec_coeff_c2=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_C1),
    )

    return (
        diffusion_metric_state,
        diffusion_interpolation_state,
        solve_nonhydro_metric_state,
        solve_nonhydro_interpolation_state,
        diagnostic_metric_state,
    )



def configure_logging(
    run_path: pathlib.Path,
    experiment_name: str,
    logging_level: str,
    processor_procs: decomposition.ProcessProperties = None,
) -> None:
    """
    Configure logging.

    Log output is sent to console and to a file.

    Args:
        run_path: path to the output folder where the logfile should be stored
        experiment_name: name of the simulation
        enable_output: enable output logging messages above debug level
        processor_procs: ProcessProperties

    """
    if logging_level not in _LOGGING_LEVELS:
        raise ValueError(
            f"Invalid logging level {logging_level}, please make sure that the logging level matches either {' / '.join([k for k in _LOGGING_LEVELS])}"
        )

    logfile = run_path.joinpath(f"log_{experiment_name}_{datetime.now(datetime.timezone.utc)}")
    logfile.touch(exist_ok=False)

    logging.basicConfig(
        level=_LOGGING_LEVELS[logging_level],
        format="%(asctime)s %(filename)-20s (%(lineno)-4d) : %(funcName)-20s:  %(levelname)-8s %(message)s",
        filemode="w",
        filename=logfile,
    )
    console_handler = logging.StreamHandler()
    # TODO(OngChia): modify here when single_dispatch is ready
    console_handler.addFilter(mpi_decomp.ParallelLogger(processor_procs))

    log_format = "{rank} {asctime} - {filename}: {funcName:<20}: {levelname:<7} {message}"
    formatter = logging.Formatter(fmt=log_format, style="{", defaults={"rank": None})
    console_handler.setFormatter(formatter)
    console_handler.setLevel(_LOGGING_LEVELS[logging_level])
    logging.getLogger("").addHandler(console_handler)


@functools.cache
def _create_grid_global_params(
    grid_file: pathlib.Path,
) -> tuple[icon_grid.GlobalGridParams, str]:
    """
    Create global grid params and its uuid.

    Args:
        grid_file: path of the grid file

    Returns:
        global_grid_params: GlobalGridParams
        grid_uuid: id (uuid) of the horizontal grid
    """
    grid = nc4.Dataset(grid_file, "r", format="NETCDF4")
    grid_root = grid.getncattr("grid_root")
    grid_level = grid.getncattr("grid_level")
    grid_uuid = grid.getncattr("uuidOfHGrid")
    try:
        grid_geometry_type = base.GeometryType(grid.getncattr("grid_geometry"))
    except AttributeError:
        log.warning(
            "Global attribute grid_geometry is not found in the grid. Icosahedral grid is assumed."
        )
        grid_geometry_type = base.GeometryType.ICOSAHEDRON
    grid.close()
    global_grid_params = icon_grid.GlobalGridParams(
        grid_shape=icon_grid.GridShape(
            geometry_type=grid_geometry_type,
            subdivision=icon_grid.GridSubdivision(root=grid_root, level=grid_level),
        ),
    )
    return global_grid_params, grid_uuid
