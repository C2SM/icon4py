# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import logging
import pathlib

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
from model.common.src.icon4py.model.common.constants import RayleighType

from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import (
    definitions as decomposition_defs,
    mpi_decomposition as mpi_decomp,
)
from icon4py.model.common.grid import (
    geometry as grid_geometry,
    geometry_attributes as geometry_meta,
    grid_manager as gm,
    icon as icon_grid,
    states as grid_states,
    vertical as v_grid,
)
from icon4py.model.common.initialization.jablonowski_williamson_topography import (
    jablonowski_williamson_topography,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver.testcases import jablonowski_williamson


log = logging.getLogger(__name__)

_LOGGING_LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "critical": logging.CRITICAL,
}


def create_mesh(
    grid_file_path: pathlib.Path,
    vertical_grid_config: v_grid.VerticalGridConfig,
    backend: gtx_typing.Backend,
) -> tuple[
    decomposition_defs.DecompositionInfo,
    icon_grid.IconGrid,
]:
    grid_manager = gm.GridManager(
        gm.ToZeroBasedIndexTransformation(),
        grid_file_path,
        vertical_grid_config,
    )
    grid_manager(backend=backend, keep_skip_values=True)
    mesh = grid_manager.grid

    xp = data_alloc.import_array_ns(backend)

    def _add_dimension(dim: gtx.Dimension) -> None:
        indices = data_alloc.index_field(mesh, dim, allocator=backend)
        owner_mask = xp.ones((mesh.size[dim],), dtype=bool)
        decomposition_info.with_dimension(dim, indices.ndarray, owner_mask)

    decomposition_info = decomposition_defs.DecompositionInfo(klevels=mesh.num_levels)
    _add_dimension(dims.EdgeDim)
    _add_dimension(dims.VertexDim)
    _add_dimension(dims.CellDim)

    return mesh, decomposition_info


def create_vertical_grid(
    vertical_grid_config: v_grid.VerticalGridConfig,
    backend: gtx_typing.Backend,
) -> v_grid.VerticalGrid:
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_grid_config, backend)

    vertical_grid = v_grid.VerticalGrid(
        config=vertical_grid_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )
    return vertical_grid


def create_geometry_factory(
    mesh: icon_grid.IconGrid,
    decomposition_info: decomposition_defs.DecompositionInfo,
    backend: gtx_typing.Backend,
) -> grid_geometry.GridGeometry:
    geometry_field_source = grid_geometry.GridGeometry(
        grid=mesh,
        decomposition_info=decomposition_info,
        backend=backend,
        coordinates=mesh.coordinates,
        extra_fields=mesh.geometry_fields,
        metadata=geometry_meta.attrs,
    )

    return geometry_field_source


def create_topography(
    geometry_field_source: grid_geometry.GridGeometry,
    backend: gtx_typing.Backend,
) -> data_alloc.NDArray:
    topo_c = jablonowski_williamson_topography(
        cell_lat=geometry_field_source.get(geometry_meta.CELL_LAT).ndarray,
        u0=35.0,
        backend=backend,
    )
    return topo_c


def create_interpolation_metrics_factories(
    mesh: icon_grid.IconGrid,
    decomposition_info: decomposition_defs.DecompositionInfo,
    geometry_field_source: grid_geometry.GridGeometry,
    vertical_grid: v_grid.VerticalGrid,
    topo_c: data_alloc.NDArray,
    backend: gtx_typing.Backend,
) -> tuple[
    interpolation_factory.InterpolationFieldsFactory,
    metrics_factory.MetricsFieldsFactory,
]:
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
        topography=gtx.as_field((dims.CellDim,), data=topo_c),
        interpolation_source=interpolation_field_source,
        backend=backend,
        metadata=metrics_attributes.attrs,
        rayleigh_type=RayleighType.KLEMP,
        rayleigh_coeff=5.0,
        exner_expol=0.333,
        vwind_offctr=0.2,
    )

    return interpolation_field_source, metrics_field_source


def initialize_granule(
    mesh: icon_grid.IconGrid,
    decomposition_info: decomposition_defs.DecompositionInfo,
    vertical_grid: v_grid.VerticalGrid,
    diffusion_config: diffusion.DiffusionConfig,
    solve_nh_config: solve_nh.NonHydrostaticConfig,
    geometry_field_source: grid_geometry.GridGeometry,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    exchange: decomposition_defs.ExchangeRuntime,
    backend: gtx_typing.Backend,
) -> tuple[
    diffusion.Diffusion,
    solve_nh.SolveNonhydro,
]:
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

    diffusion_params = diffusion.DiffusionParams(diffusion_config)

    diffusion_granule = diffusion.Diffusion(
        grid=mesh,
        config=diffusion_config,
        params=diffusion_params,
        vertical_grid=vertical_grid,
        metric_state=diffusion_metric_state,
        interpolation_state=diffusion_interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        backend=backend,
        exchange=exchange,
    )

    nonhydro_params = solve_nh.NonHydrostaticParams(solve_nh_config)

    solve_nonhydro_granule = solve_nh.SolveNonhydro(
        grid=mesh,
        backend=backend,
        config=solve_nh_config,
        params=nonhydro_params,
        metric_state_nonhydro=solve_nonhydro_metric_state,
        interpolation_state=solve_nonhydro_interpolation_state,
        vertical_params=vertical_grid,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=gtx.as_field(
            (dims.CellDim,),
            decomposition_info.owner_mask(dims.CellDim),
            allocator=backend,  # type: ignore[arg-type]
        ),
    )

    return diffusion_granule, solve_nonhydro_granule


def read_initial_state(
    grid: icon_grid.IconGrid,
    geometry_field_source: grid_geometry.GridGeometry,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend: gtx_typing.Backend,
) -> tuple[
    diffusion_states.DiffusionDiagnosticState,
    dycore_states.DiagnosticStateNonHydro,
    dycore_states.PrepAdvection,
    diagnostics.DiagnosticState,
    prognostics.PrognosticState,
    prognostics.PrognosticState,
]:
    """
    Read initial prognostic and diagnostic fields for the jablonowski_williamson test case.

    Args:
        grid: IconGrid
        geometry_field_source: geometry field factory,
        interpolation_field_source: interpolation field factory,
        metrics_field_source: metrics field factory,
        backend: GT4Py backend

    Returns:  A tuple containing Diagnostic variables for diffusion and solve_nonhydro granules,
        PrepAdvection, diagnostic variables, and two prognostic
        variables (now and next).
    """

    (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    ) = jablonowski_williamson.model_initialization_jabw(
        grid=grid,
        geometry_field_source=geometry_field_source,
        interpolation_field_source=interpolation_field_source,
        metrics_field_source=metrics_field_source,
        backend=backend,
    )

    return (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    )


def configure_logging(
    output_path: pathlib.Path,
    experiment_name: str,
    logging_level: str,
    processor_procs: decomposition_defs.ProcessProperties = None,
) -> None:
    """
    Configure logging.

    Log output is sent to console and to a file.

    Args:
        output_path: path to the output folder where the logfile should be stored
        experiment_name: name of the simulation
        enable_output: enable output logging messages above debug level
        processor_procs: ProcessProperties

    """
    if logging_level not in _LOGGING_LEVELS:
        raise ValueError(
            f"Invalid logging level {logging_level}, please make sure that the logging level matches either {' / '.join([k for k in _LOGGING_LEVELS])}"
        )

    logfile = output_path.joinpath(f"log_{experiment_name}_{datetime.now(datetime.timezone.utc)}")
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
