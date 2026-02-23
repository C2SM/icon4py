# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import dataclasses
import logging
import pathlib
import sys
import time
from types import ModuleType
from typing import Any

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    model_options,
)
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
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import config as driver_config, driver_states


log = logging.getLogger(__name__)

_LOGGING_LEVELS: dict[str, int] = {
    "notset": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def create_grid_manager(
    grid_file_path: pathlib.Path,
    vertical_grid_config: v_grid.VerticalGridConfig,
    allocator: gtx_typing.FieldBufferAllocationUtil,
    global_reductions: decomposition_defs.Reductions = decomposition_defs.single_node_reductions,
) -> gm.GridManager:
    grid_manager = gm.GridManager(
        gm.ToZeroBasedIndexTransformation(), grid_file_path, vertical_grid_config, global_reductions
    )
    grid_manager(allocator=allocator, keep_skip_values=True)

    return grid_manager


def create_decomposition_info(
    grid_manager: gm.GridManager,
    allocator: gtx_typing.FieldBufferAllocationUtil,
) -> decomposition_defs.DecompositionInfo:
    decomposition_info = decomposition_defs.DecompositionInfo()
    xp = data_alloc.import_array_ns(allocator)

    def _add_dimension(dim: gtx.Dimension) -> None:
        indices = data_alloc.index_field(grid_manager.grid, dim, allocator=allocator)
        owner_mask = xp.ones((grid_manager.grid.size[dim],), dtype=bool)
        decomposition_info.with_dimension(dim, indices.ndarray, owner_mask)

    _add_dimension(dims.EdgeDim)
    _add_dimension(dims.VertexDim)
    _add_dimension(dims.CellDim)

    return decomposition_info


def create_vertical_grid(
    vertical_grid_config: v_grid.VerticalGridConfig,
    allocator: gtx_typing.FieldBufferAllocationUtil,
) -> v_grid.VerticalGrid:
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(
        vertical_config=vertical_grid_config, allocator=allocator
    )

    vertical_grid = v_grid.VerticalGrid(
        config=vertical_grid_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )
    return vertical_grid


def create_static_field_factories(
    grid_manager: gm.GridManager,
    decomposition_info: decomposition_defs.DecompositionInfo,
    vertical_grid: v_grid.VerticalGrid,
    cell_topography: data_alloc.NDArray,
    backend: model_backends.BackendLike,
) -> driver_states.StaticFieldFactories:
    concrete_backend = model_options.customize_backend(program=None, backend=backend)
    geometry_field_source = grid_geometry.GridGeometry(
        grid=grid_manager.grid,
        decomposition_info=decomposition_info,
        backend=concrete_backend,
        coordinates=grid_manager.coordinates,
        extra_fields=grid_manager.geometry_fields,
        metadata=geometry_meta.attrs,
    )

    interpolation_field_source = interpolation_factory.InterpolationFieldsFactory(
        grid=grid_manager.grid,
        decomposition_info=decomposition_info,
        geometry_source=geometry_field_source,
        backend=concrete_backend,
        metadata=interpolation_attributes.attrs,
    )

    metrics_field_source = metrics_factory.MetricsFieldsFactory(
        grid=grid_manager.grid,
        vertical_grid=vertical_grid,
        decomposition_info=decomposition_info,
        geometry_source=geometry_field_source,
        topography=cell_topography,
        interpolation_source=interpolation_field_source,
        backend=concrete_backend,
        metadata=metrics_attributes.attrs,
        rayleigh_type=constants.RayleighType.KLEMP,
        rayleigh_coeff=0.1,
        exner_expol=0.333,
        vwind_offctr=0.2,
    )

    return driver_states.StaticFieldFactories(
        geometry_field_source, interpolation_field_source, metrics_field_source
    )


def initialize_granules(
    grid: icon_grid.IconGrid,
    vertical_grid: v_grid.VerticalGrid,
    diffusion_config: diffusion.DiffusionConfig,
    solve_nh_config: solve_nh.NonHydrostaticConfig,
    static_field_factories: driver_states.StaticFieldFactories,
    exchange: decomposition_defs.ExchangeRuntime,
    owner_mask: fa.CellField[bool],
    backend: model_backends.BackendLike,
) -> tuple[
    diffusion.Diffusion,
    solve_nh.SolveNonhydro,
]:
    geometry_field_source = static_field_factories.geometry_field_source
    interpolation_field_source = static_field_factories.interpolation_field_source
    metrics_field_source = static_field_factories.metrics_field_source

    log.info("creating cell geometry")
    cell_geometry = grid_states.CellParams(
        cell_center_lat=geometry_field_source.get(geometry_meta.CELL_LAT),
        cell_center_lon=geometry_field_source.get(geometry_meta.CELL_LON),
        area=geometry_field_source.get(geometry_meta.CELL_AREA),
        mean_cell_area=geometry_field_source.get(geometry_meta.MEAN_CELL_AREA),
    )

    log.info("creating edge geometry")
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

    log.info("creating diffusion interpolation state")
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

    log.info("creating diffusion metric state")
    diffusion_metric_state = diffusion_states.DiffusionMetricState(
        mask_hdiff=metrics_field_source.get(metrics_attributes.MASK_HDIFF),
        theta_ref_mc=metrics_field_source.get(metrics_attributes.THETA_REF_MC),
        wgtfac_c=metrics_field_source.get(metrics_attributes.WGTFAC_C),
        zd_intcoef=metrics_field_source.get(metrics_attributes.ZD_INTCOEF_DSL),
        zd_vertoffset=metrics_field_source.get(metrics_attributes.ZD_VERTOFFSET_DSL),
        zd_diffcoef=metrics_field_source.get(metrics_attributes.ZD_DIFFCOEF_DSL),
    )

    log.info("creating solve nonhydro interpolation state")
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

    log.info("creating solve nonhydro metric state")
    solve_nonhydro_metric_state = dycore_states.MetricStateNonHydro(
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
        grid=grid,
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
        grid=grid,
        backend=backend,
        config=solve_nh_config,
        params=nonhydro_params,
        metric_state_nonhydro=solve_nonhydro_metric_state,
        interpolation_state=solve_nonhydro_interpolation_state,
        vertical_params=vertical_grid,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=owner_mask,
    )

    return diffusion_granule, solve_nonhydro_granule


def find_maximum_from_field(
    input_field: gtx.Field, array_ns: ModuleType
) -> tuple[tuple[int, ...], float]:
    max_indices = array_ns.unravel_index(
        array_ns.abs(input_field.ndarray).argmax(),
        input_field.ndarray.shape,
    )
    return max_indices, input_field.ndarray[max_indices]


def display_icon4py_logo_in_log_file() -> None:
    """
    Print out icon4py signature and some important information of the initial setup to the log file.

                                                               ___
          -------                                    //      ||   \
            | |                                     //       ||    |
            | |       __      _ _        _ _       //  ||    ||___/
            | |     //       /   \     |/   \     //_ _||_   ||        \\      //
            | |    ||       |     |    |     |    --------   ||         \\    //
            | |     \\__     \_ _/     |     |         ||    ||          \\  //
          -------                                                           //
                                                                           //
                                                              = = = = = = //
    """
    boundary_line = ["*" * 91]
    icon4py_signature = []
    icon4py_signature += boundary_line
    empty_line = ["*" + 89 * " " + "*"]
    for _ in range(3):
        icon4py_signature += empty_line

    icon4py_signature += [
        "*                                                                ___                      *"
    ]
    icon4py_signature += [
        r"*            -------                                    //      ||   \                    *"
    ]
    icon4py_signature += [
        "*              | |                                     //       ||    |                   *"
    ]
    icon4py_signature += [
        "*              | |       __      _ _        _ _       //  ||    ||___/                    *"
    ]
    icon4py_signature += [
        r"*              | |     //       /   \     |/   \     //_ _||_   ||        \\      //      *"
    ]
    icon4py_signature += [
        r"*              | |    ||       |     |    |     |    --------   ||         \\    //       *"
    ]
    icon4py_signature += [
        r"*              | |     \\__     \_ _/     |     |         ||    ||          \\  //        *"
    ]
    icon4py_signature += [
        "*            -------                                                           //         *"
    ]
    icon4py_signature += [
        "*                                                                             //          *"
    ]
    icon4py_signature += [
        "*                                                                = = = = = = //           *"
    ]

    for _ in range(3):
        icon4py_signature += empty_line
    icon4py_signature += boundary_line
    icon4py_signature = "\n".join(icon4py_signature)
    log.info(f"{icon4py_signature}")


def display_driver_setup_in_log_file(
    n_time_steps: int,
    vertical_params,
    config: driver_config.DriverConfig,
) -> None:
    log.info("===== ICON4Py Driver Configuration =====")
    log.info(f"Experiment name        : {config.experiment_name}")
    log.info(f"Time step (dtime)      : {config.dtime.total_seconds()} s")
    log.info(f"Start date             : {config.start_date}")
    log.info(f"End date               : {config.end_date}")
    log.info(f"Number of timesteps    : {n_time_steps}")
    log.info(f"Initial ndyn_substeps  : {config.ndyn_substeps}")
    log.info(f"Vertical CFL threshold : {config.vertical_cfl_threshold}")
    log.info(f"Second-order divdamp   : {config.apply_extra_second_order_divdamp}")
    log.info(f"Statistics enabled     : {config.enable_statistics_output}")
    log.info("")

    log.info("==== Vertical Grid Parameters ====")
    log.info(vertical_params)
    consts = constants.PhysicsConstants()
    log.info("==== Physical Constants ====")
    for name, value in consts.__class__.__dict__.items():
        if name.startswith("_") or callable(value):
            continue
        log.info(f"{name:30s}: {value}")


@dataclasses.dataclass
class _InfoFormatter(logging.Formatter):
    style: str
    default_fmt: str
    info_fmt: str
    defaults: dict[str, Any] | None

    _info_formatter: logging.Formatter = dataclasses.field(init=False)

    def __post_init__(self):
        super().__init__(fmt=self.default_fmt, style=self.style, defaults=self.defaults)
        self._info_formatter = logging.Formatter(
            fmt=self.info_fmt,
            style=self.style,
        )

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            return self._info_formatter.format(record)
        return super().format(record)


def make_handler(
    logging_level: int | None,
    log_filter: logging.Filter | None,
    formatter: str | logging.Formatter | None,
    file_name: str | None,
) -> logging.Handler:
    handler = (
        logging.StreamHandler(stream=sys.stdout)
        if file_name is None
        else logging.FileHandler(filename=file_name)
    )
    if log_filter is not None:
        handler.addFilter(log_filter)
    if formatter is not None:
        if isinstance(formatter, str):
            formatter = logging.Formatter(fmt=formatter, style="{")
        handler.setFormatter(formatter)
    if logging_level is not None:
        handler.setLevel(logging_level)
    return handler


def configure_logging(
    logging_level: str,
    processor_procs: decomposition_defs.ProcessProperties = None,
) -> None:
    """
    Configure logging.

    Log output with user-defined logging level across the entire icon4py, except
    for the driver whose logging level is fixed at debug, is sent to console
    (stdout) and the error message is sent to stderr.

    Args:
        logging_level: log level
        processor_procs: ProcessProperties

    """
    if logging_level.lower() not in _LOGGING_LEVELS:
        raise ValueError(
            f"Invalid logging level {logging_level}, please make sure that the logging level matches either {' / '.join([*_LOGGING_LEVELS.keys()])}"
        )

    logging.Formatter.converter = time.localtime  # set to local time instead of utc

    # TODO(OngChia): modify here when single_dispatch is ready
    log_filter = mpi_decomp.ParallelLogger(processor_procs)
    formatter = _InfoFormatter(
        style="{",
        default_fmt="{rank} {asctime} - {filename}: {funcName:<20}: {levelname:<7} {message}",
        info_fmt="{message}",
        defaults={"rank": None},
    )
    handler = make_handler(
        logging_level=logging.DEBUG,
        log_filter=log_filter,
        formatter=formatter,
        file_name=None,
    )
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            handler,
        ],
    )
    driver_module_name = __name__[: __name__.rindex(".")]
    logging.getLogger("icon4py.model").setLevel(_LOGGING_LEVELS[logging_level])
    logging.getLogger(driver_module_name).setLevel(logging.DEBUG)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("factory.generate").setLevel(logging.WARNING)
    logging.getLogger("blib2to3").setLevel(logging.WARNING)

    display_icon4py_logo_in_log_file()


def get_backend_from_name(backend_name: str) -> model_backends.BackendLike:
    if backend_name not in model_backends.BACKENDS:
        raise ValueError(
            f"Invalid driver backend: {backend_name}. \n"
            f"Available backends are {', '.join([*model_backends.BACKENDS.keys()])}"
        )
    backend = model_backends.BACKENDS[backend_name]
    log.info(f"Backend name used for the model: {backend_name}")
    log.info(f"BackendLike derived from the backend name: {backend}")
    return backend
