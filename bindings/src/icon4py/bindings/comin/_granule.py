# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
icon4py's grid and diffusion granule, built through icon4py's API.

'Granule' builds the grid ('grid_init'), the diffusion granule ('diffusion_init') and runs it
('diffusion_run') from ICON's data. Each function does what the function of the same name of
py2fgen's bindings does ('icon4py.bindings.grid_wrapper.grid_init',
'icon4py.bindings.diffusion_wrapper.diffusion_init' and 'diffusion_run'): the same functions
of icon4py, in the same order, on inputs of the same names, forms and values. So the granule
computes the same on both routes, and the py2fgen probe ('_probe.py') can compare the inputs
and the objects built from them, one by one. Two differences:
- 'diffusion_init' gets icon4py's 'DiffusionConfig', which 'diffusion_config' builds from
  ICON's namelist output with 'DiffusionConfig.from_fortran_dict', instead of 23 of its 25
  configuration values (the other two, 'ndyn_substeps' and 'nudge_max_coeff', which icon4py's
  'Diffusion' takes besides the configuration, it gets as they are);
- it always waits for GT4Py's compilation: ComIn's Python adapter holds the GIL between
  callbacks, so a compilation in the background would stall.
The state lives on the instance, not in module globals.

PARAMS lists the inputs of each function: arrays ('_arguments.ArrayParam': dtype, rank, Field
dimensions, memory space, presence, padding), scalars (their Python type) and the
configuration object.
"""

import dataclasses
import logging
import typing
from collections.abc import Callable, Mapping
from typing import Any, Final

import gt4py.next as gtx
import numpy as np

import icon4py.model.common.grid.states as grid_states
import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.bindings import common as bindings_common, debug_utils
from icon4py.bindings.comin import _arguments, _config
from icon4py.model.atmosphere.diffusion.diffusion import Diffusion, DiffusionConfig, DiffusionParams
from icon4py.model.atmosphere.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
)
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.config import options
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import icon as icon_grid, vertical
from icon4py.model.common.states.prognostic_state import PrognosticState


log = logging.getLogger(__name__)

F64: Final = np.dtype(np.float64)
I32: Final = np.dtype(np.int32)
BOOL: Final = np.dtype(np.bool_)


def _arrays(*params: _arguments.ArrayParam) -> dict[str, _arguments.Param]:
    return {p.name: p for p in params}


def _field(name: str, *field_dims: gtx.Dimension, **kwargs: Any) -> _arguments.ArrayParam:
    """A Field input (on the device on the GPU), float64 unless 'dtype' is given."""
    kwargs.setdefault("dtype", F64)
    return _arguments.ArrayParam(name, rank=len(field_dims), dims=field_dims, **kwargs)


def _host(name: str, dtype: np.dtype, location: str | None = None) -> _arguments.ArrayParam:
    """A one-dimensional NumPy input, also on the GPU."""
    return _arguments.ArrayParam(name, dtype, rank=1, host=True, location=location)


_C, _E, _V, _K = dims.CellDim, dims.EdgeDim, dims.VertexDim, dims.KDim
_KH = dims.KHalfDim

GRID_INIT: Final[Mapping[str, _arguments.Param]] = {
    **_arrays(
        *(
            _host(f"{loc}_{end}s", I32)
            for loc in ("cell", "vertex", "edge")
            for end in ("start", "end")
        ),
        _field("c2e", _C, dims.C2EDim, dtype=I32, location="cell"),
        _field("e2c", _E, dims.E2CDim, dtype=I32, location="edge"),
        _field("c2e2c", _C, dims.C2E2CDim, dtype=I32, location="cell"),
        _field("e2c2e", _E, dims.E2C2EDim, dtype=I32, location="edge"),
        _field("e2v", _E, dims.E2VDim, dtype=I32, location="edge"),
        _field("v2e", _V, dims.V2EDim, dtype=I32, location="vertex"),
        _field("v2c", _V, dims.V2CDim, dtype=I32, location="vertex"),
        _field("e2c2v", _E, dims.E2C2VDim, dtype=I32, location="edge"),
        _field("c2v", _C, dims.C2VDim, dtype=I32, location="cell"),
        _host("c_owner_mask", BOOL, "cell"),
        _host("e_owner_mask", BOOL, "edge"),
        _host("v_owner_mask", BOOL, "vertex"),
        _host("c_glb_index", I32, "cell"),
        _host("e_glb_index", I32, "edge"),
        _host("v_glb_index", I32, "vertex"),
        *(
            _field(name, _E, location="edge")
            for name in (
                "tangent_orientation",
                "inverse_primal_edge_lengths",
                "inv_dual_edge_length",
                "inv_vert_vert_length",
                "edge_areas",
                "f_e",
            )
        ),
        *(
            _field(name, _C, location="cell")
            for name in ("cell_center_lat", "cell_center_lon", "cell_areas")
        ),
        *(
            _field(f"{normal}_{xy}", _E, neighbours, location="edge")
            for normal, neighbours in (
                ("primal_normal_vert", dims.E2C2VDim),
                ("dual_normal_vert", dims.E2C2VDim),
                ("primal_normal_cell", dims.E2CDim),
                ("dual_normal_cell", dims.E2CDim),
            )
            for xy in ("x", "y")
        ),
        *(
            _field(name, _E, location="edge")
            for name in ("edge_center_lat", "edge_center_lon", "primal_normal_x", "primal_normal_y")
        ),
        _field("vct_a", _KH),
    ),
    "lowest_layer_thickness": float,
    "model_top_height": float,
    "stretch_factor": float,
    "flat_height": float,
    "rayleigh_damping_height": float,
    "mean_cell_area": float,
    "comm_id": int,
    "num_vertices": int,
    "num_cells": int,
    "num_edges": int,
    "vertical_size": int,
    "limited_area": bool,
    "backend": int,
}
"""The inputs of 'grid_init' (those of py2fgen's 'grid_init')."""

DIFFUSION_INIT: Final[Mapping[str, _arguments.Param]] = {
    **_arrays(
        _field("theta_ref_mc", _C, _K, location="cell"),
        _field("wgtfac_c", _C, _KH, location="cell"),
        _field("e_bln_c_s", _C, dims.C2EDim, location="cell"),
        _field("geofac_div", _C, dims.C2EDim, location="cell"),
        _field("geofac_grg_x", _C, dims.C2E2CODim, location="cell"),
        _field("geofac_grg_y", _C, dims.C2E2CODim, location="cell"),
        _field("geofac_n2s", _C, dims.C2E2CODim, location="cell"),
        _field("nudgecoeff_e", _E, location="edge"),
        _arguments.ArrayParam("rbf_vec_coeff_v", F64, rank=3, location="vertex", axis=2),
        _arguments.ArrayParam("zd_cellidx", I32, rank=2, optional=True),
        _arguments.ArrayParam("zd_vertidx", I32, rank=2, optional=True),
        _arguments.ArrayParam("zd_intcoef", F64, rank=2, optional=True),
        _arguments.ArrayParam("zd_diffcoef", F64, rank=1, optional=True),
    ),
    "config": DiffusionConfig,
    "ndyn_substeps": int,
    "nudge_max_coeff": float,
    "backend": int,
}
"""
The inputs of 'diffusion_init': the arrays, 'ndyn_substeps', 'nudge_max_coeff' and 'backend' of
py2fgen's 'diffusion_init', and icon4py's 'DiffusionConfig' in place of its other 23
configuration values ('diffusion_config').
"""

DIFFUSION_RUN: Final[Mapping[str, _arguments.Param]] = {
    **_arrays(
        _field("w", _C, _KH, location="cell"),
        _field("vn", _E, _K, location="edge"),
        _field("exner", _C, _K, location="cell"),
        _field("theta_v", _C, _K, location="cell"),
        _field("rho", _C, _K, location="cell"),
        *(
            _field(name, _C, _KH, optional=True, location="cell")
            for name in ("hdef_ic", "div_ic", "dwdx", "dwdy")
        ),
    ),
    "dtime": float,
    "linit": bool,
}
"""The inputs of 'diffusion_run' (those of py2fgen's 'diffusion_run')."""

PARAMS: Final[Mapping[str, Mapping[str, _arguments.Param]]] = {
    "grid_init": GRID_INIT,
    "diffusion_init": DIFFUSION_INIT,
    "diffusion_run": DIFFUSION_RUN,
}
"""The functions of 'Granule' and their inputs, in call order."""


@dataclasses.dataclass
class GridState:
    """What 'grid_init' builds (as py2fgen's 'grid_wrapper.GridState', plus the decomposition)."""

    grid: icon_grid.IconGrid
    vertical_grid: vertical.VerticalGrid
    edge_geometry: grid_states.EdgeParams
    cell_geometry: grid_states.CellParams
    exchange_runtime: decomposition_defs.ExchangeRuntime
    decomposition_info: decomposition_defs.DecompositionInfo | None


class Granule:
    """icon4py's grid and diffusion granule of one plugin."""

    def __init__(self) -> None:
        self.grid_state: GridState | None = None
        self.diffusion: Diffusion | None = None
        self.dummy_field_factory: Callable[..., gtx.Field] | None = None

    def grid_init(
        self,
        *,
        cell_starts: np.ndarray,
        cell_ends: np.ndarray,
        vertex_starts: np.ndarray,
        vertex_ends: np.ndarray,
        edge_starts: np.ndarray,
        edge_ends: np.ndarray,
        c2e: Any,
        e2c: Any,
        c2e2c: Any,
        e2c2e: Any,
        e2v: Any,
        v2e: Any,
        v2c: Any,
        e2c2v: Any,
        c2v: Any,
        c_owner_mask: np.ndarray,
        e_owner_mask: np.ndarray,
        v_owner_mask: np.ndarray,
        c_glb_index: np.ndarray,
        e_glb_index: np.ndarray,
        v_glb_index: np.ndarray,
        tangent_orientation: gtx.Field,
        inverse_primal_edge_lengths: gtx.Field,
        inv_dual_edge_length: gtx.Field,
        inv_vert_vert_length: gtx.Field,
        edge_areas: gtx.Field,
        f_e: gtx.Field,
        cell_center_lat: gtx.Field,
        cell_center_lon: gtx.Field,
        cell_areas: gtx.Field,
        primal_normal_vert_x: gtx.Field,
        primal_normal_vert_y: gtx.Field,
        dual_normal_vert_x: gtx.Field,
        dual_normal_vert_y: gtx.Field,
        primal_normal_cell_x: gtx.Field,
        primal_normal_cell_y: gtx.Field,
        dual_normal_cell_x: gtx.Field,
        dual_normal_cell_y: gtx.Field,
        edge_center_lat: gtx.Field,
        edge_center_lon: gtx.Field,
        primal_normal_x: gtx.Field,
        primal_normal_y: gtx.Field,
        vct_a: gtx.Field,
        lowest_layer_thickness: float,
        model_top_height: float,
        stretch_factor: float,
        flat_height: float,
        rayleigh_damping_height: float,
        mean_cell_area: float,
        comm_id: int | None,
        num_vertices: int,
        num_cells: int,
        num_edges: int,
        vertical_size: int,
        limited_area: bool,
        backend: int,
    ) -> None:
        """The grid, its decomposition and geometry, and the vertical grid."""
        on_gpu = c2e.array_ns != np
        actual_backend = bindings_common.select_backend(
            bindings_common.BackendIntEnum(backend), on_gpu=on_gpu
        )
        allocator = model_backends.get_allocator(actual_backend)

        decomposition_info = None
        process_props: decomposition_defs.ProcessProperties
        if comm_id is None:
            process_props = decomposition_defs.SingleNodeProcessProperties()
            exchange_runtime: decomposition_defs.ExchangeRuntime = (
                decomposition_defs.SingleNodeExchange()
            )
        else:
            process_props, decomposition_info, exchange_runtime = (
                bindings_common.construct_decomposition(
                    c_glb_index=c_glb_index,
                    e_glb_index=e_glb_index,
                    v_glb_index=v_glb_index,
                    c_owner_mask=c_owner_mask,
                    e_owner_mask=e_owner_mask,
                    v_owner_mask=v_owner_mask,
                    num_cells=num_cells,
                    num_edges=num_edges,
                    num_vertices=num_vertices,
                    comm_id=comm_id,
                )
            )

        grid = bindings_common.construct_icon_grid(
            cell_starts=cell_starts,
            cell_ends=cell_ends,
            vertex_starts=vertex_starts,
            vertex_ends=vertex_ends,
            edge_starts=edge_starts,
            edge_ends=edge_ends,
            c2e=c2e.ndarray,
            e2c=e2c.ndarray,
            c2e2c=c2e2c.ndarray,
            e2c2e=e2c2e.ndarray,
            e2v=e2v.ndarray,
            v2e=v2e.ndarray,
            v2c=v2c.ndarray,
            e2c2v=e2c2v.ndarray,
            c2v=c2v.ndarray,
            grid_id="icon_grid",
            num_vertices=num_vertices,
            num_cells=num_cells,
            num_edges=num_edges,
            vertical_size=vertical_size,
            limited_area=limited_area,
            distributed=not process_props.is_single_rank(),
            allocator=allocator,
        )

        if decomposition_info is not None:
            debug_utils.print_grid_decomp_info(
                icon_grid=grid,
                process_props=process_props,
                decomposition_info=decomposition_info,
                num_cells=num_cells,
                num_edges=num_edges,
                num_verts=num_vertices,
            )

        vertical_config = vertical.VerticalGridConfig(
            num_levels=vertical_size,
            lowest_layer_thickness=lowest_layer_thickness,
            model_top_height=model_top_height,
            stretch_factor=stretch_factor,
            rayleigh_damping_height=rayleigh_damping_height,
            flat_height=flat_height,
        )
        vertical_grid = vertical.VerticalGrid(config=vertical_config, vct_a=vct_a, vct_b=None)

        edge_params = grid_states.EdgeParams(
            tangent_orientation=tangent_orientation,
            inverse_primal_edge_lengths=inverse_primal_edge_lengths,
            inverse_dual_edge_lengths=inv_dual_edge_length,
            inverse_vertex_vertex_lengths=inv_vert_vert_length,
            primal_normal_vert=(primal_normal_vert_x, primal_normal_vert_y),
            dual_normal_vert=(dual_normal_vert_x, dual_normal_vert_y),
            primal_normal_cell=(primal_normal_cell_x, primal_normal_cell_y),
            dual_normal_cell=(dual_normal_cell_x, dual_normal_cell_y),
            edge_areas=edge_areas,
            coriolis_frequency=f_e,
            edge_center=(edge_center_lat, edge_center_lon),
            primal_normal=(primal_normal_x, primal_normal_y),
        )
        cell_params = grid_states.CellParams(
            cell_center_lat=cell_center_lat,
            cell_center_lon=cell_center_lon,
            area=cell_areas,
            mean_cell_area=mean_cell_area,
        )
        self.grid_state = GridState(
            grid=grid,
            vertical_grid=vertical_grid,
            edge_geometry=edge_params,
            cell_geometry=cell_params,
            exchange_runtime=exchange_runtime,
            decomposition_info=decomposition_info,
        )

    def diffusion_init(
        self,
        *,
        theta_ref_mc: Any,
        wgtfac_c: gtx.Field,
        e_bln_c_s: gtx.Field,
        geofac_div: gtx.Field,
        geofac_grg_x: gtx.Field,
        geofac_grg_y: gtx.Field,
        geofac_n2s: gtx.Field,
        nudgecoeff_e: gtx.Field,
        rbf_vec_coeff_v: Any,
        zd_cellidx: Any,
        zd_vertidx: Any,
        zd_intcoef: Any,
        zd_diffcoef: Any,
        config: DiffusionConfig,
        ndyn_substeps: int,
        nudge_max_coeff: float,
        backend: int,
    ) -> None:
        """The diffusion granule on the grid of 'grid_init'."""
        grid_state = self.grid_state
        if grid_state is None:
            raise RuntimeError(
                "icon4py ComIn plugin: 'grid_init' must run before 'diffusion_init'."
            )
        xp = theta_ref_mc.array_ns
        on_gpu = xp != np
        actual_backend = bindings_common.select_backend(
            bindings_common.BackendIntEnum(backend), on_gpu=on_gpu
        )
        log.debug(f"backend {getattr(actual_backend, 'name', actual_backend)}, on_gpu={on_gpu}")
        allocator = model_backends.get_allocator(actual_backend)

        diffusion_params = DiffusionParams(config)

        nlev = wgtfac_c.domain[dims.KHalfDim].unit_range.stop - 1
        cell_k_domain = gtx.domain(
            {dims.CellDim: wgtfac_c.domain[dims.CellDim].unit_range, dims.KDim: nlev}
        )
        c2e2c_size = geofac_grg_x.domain[dims.C2E2CODim].unit_range.stop - 1
        cell_c2e2c_k_domain = gtx.domain(
            {
                dims.CellDim: wgtfac_c.domain[dims.CellDim].unit_range,
                dims.C2E2CDim: c2e2c_size,
                dims.KDim: nlev,
            }
        )

        zd_diffcoef_field: Any
        zd_intcoef_field: Any
        zd_vertoffset_field: Any
        if zd_cellidx is None:
            # no truly horizontal temperature diffusion, or an empty list on this rank
            assert zd_vertidx is None and zd_intcoef is None and zd_diffcoef is None
            zd_diffcoef_field = gtx.zeros(
                cell_k_domain, dtype=theta_ref_mc.dtype, allocator=allocator
            )
            zd_intcoef_field = gtx.zeros(
                cell_c2e2c_k_domain, dtype=wgtfac_c.dtype, allocator=allocator
            )
            zd_vertoffset_field = gtx.zeros(
                cell_c2e2c_k_domain, dtype=xp.int32, allocator=allocator
            )
        else:
            # the lists as fields: the first row of 'zd_cellidx' is the cell, those of
            # 'zd_vertidx' the level of the cell (row 0) and of its three neighbours
            cells = zd_cellidx[0, :]
            vertoffset = zd_vertidx[1:, :] - zd_vertidx[0, :]
            levels = zd_vertidx[0, :]
            zd_diffcoef_field = data_alloc.scattered_field(
                domain=cell_k_domain,
                values=zd_diffcoef,
                indices=(
                    data_alloc.adjust_fortran_indices(cells),
                    data_alloc.adjust_fortran_indices(levels),
                ),
                default_value=gtx.float64(0.0),
                allocator=allocator,
            )
            zd_intcoef_field = data_alloc.scattered_field(
                domain=cell_c2e2c_k_domain,
                values=zd_intcoef.T,
                indices=(
                    data_alloc.adjust_fortran_indices(cells),
                    slice(None),
                    data_alloc.adjust_fortran_indices(levels),
                ),
                default_value=gtx.float64(0.0),
                allocator=allocator,
            )
            zd_vertoffset_field = data_alloc.scattered_field(
                domain=cell_c2e2c_k_domain,
                values=vertoffset.T,
                indices=(
                    data_alloc.adjust_fortran_indices(cells),
                    slice(None),
                    data_alloc.adjust_fortran_indices(levels),
                ),
                default_value=gtx.int32(0),
                allocator=allocator,
            )

        metric_state = DiffusionMetricState(
            theta_ref_mc=theta_ref_mc,
            wgtfac_c=wgtfac_c,
            zd_intcoef=zd_intcoef_field,
            zd_vertoffset=zd_vertoffset_field,
            zd_diffcoef=zd_diffcoef_field,
        )
        # the two components of the vertices' RBF vector coefficients, transposed
        rbf_coeff_1 = gtx.as_field(
            [dims.VertexDim, dims.V2EDim],
            xp.transpose(rbf_vec_coeff_v[:, 0, :]),
            allocator=allocator,
        )
        rbf_coeff_2 = gtx.as_field(
            [dims.VertexDim, dims.V2EDim],
            xp.transpose(rbf_vec_coeff_v[:, 1, :]),
            allocator=allocator,
        )
        interpolation_state = DiffusionInterpolationState(
            e_bln_c_s=e_bln_c_s,
            rbf_coeff_1=rbf_coeff_1,
            rbf_coeff_2=rbf_coeff_2,
            geofac_div=geofac_div,
            geofac_n2s=geofac_n2s,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            nudgecoeff_e=nudgecoeff_e,
        )
        self.diffusion = Diffusion(
            grid=grid_state.grid,
            config=config,
            params=diffusion_params,
            vertical_grid=grid_state.vertical_grid,
            metric_state=metric_state,
            interpolation_state=interpolation_state,
            edge_params=grid_state.edge_geometry,
            cell_params=grid_state.cell_geometry,
            backend=actual_backend,
            exchange=grid_state.exchange_runtime,
            ndyn_substeps=ndyn_substeps,
            max_nudging_coefficient=nudge_max_coeff,
        )
        self.dummy_field_factory = bindings_common.cached_dummy_field_factory(allocator)
        gtx.wait_for_compilation()

    def diffusion_run(
        self,
        *,
        w: gtx.Field,
        vn: gtx.Field,
        exner: gtx.Field,
        theta_v: gtx.Field,
        rho: gtx.Field,
        hdef_ic: gtx.Field | None,
        div_ic: gtx.Field | None,
        dwdx: gtx.Field | None,
        dwdy: gtx.Field | None,
        dtime: float,
        linit: bool,
    ) -> None:
        """One diffusion call on the given fields (written in place)."""
        diffusion, dummy = self.diffusion, self.dummy_field_factory
        if diffusion is None or dummy is None:
            raise RuntimeError(
                "icon4py ComIn plugin: 'diffusion_init' must run before 'diffusion_run'."
            )
        prognostic_state = PrognosticState(w=w, vn=vn, exner=exner, theta_v=theta_v, rho=rho)
        if hdef_ic is None:
            hdef_ic = dummy("hdef_ic", domain=w.domain, dtype=w.dtype)
        if div_ic is None:
            div_ic = dummy("div_ic", domain=w.domain, dtype=w.dtype)
        if dwdx is None:
            dwdx = dummy("dwdx", domain=w.domain, dtype=w.dtype)
        if dwdy is None:
            dwdy = dummy("dwdy", domain=w.domain, dtype=w.dtype)
        diagnostic_state = DiffusionDiagnosticState(
            hdef_ic=hdef_ic, div_ic=div_ic, dwdx=dwdx, dwdy=dwdy
        )
        diffusion.run(
            diagnostic_state=diagnostic_state,
            prognostic_state=prognostic_state,
            dtime=dtime,
            initial_run=linit,
        )

    def release(self) -> None:
        self.grid_state = None
        self.diffusion = None
        self.dummy_field_factory = None


# ---- the configuration -----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Entry:
    """A namelist entry that icon4py's DiffusionConfig reads from ICON ('IconOption')."""

    field: str
    group: str
    name: str
    domain_value: bool
    """A per-domain array: the value of domain 1 ('list_to_value')."""
    kind: type
    """The Python type of the value in the namelist output (bool, int or float)."""


def _kind(annotation: Any) -> type:
    base = (
        typing.get_args(annotation)[0]
        if typing.get_origin(annotation) is typing.Annotated
        else annotation
    )
    for kind in (bool, float, int):  # bool before int; an int enumeration reads an int
        if isinstance(base, type) and issubclass(base, kind):
            return kind
    raise TypeError(f"DiffusionConfig: no namelist type for {annotation!r}.")


def diffusion_entries() -> tuple[Entry, ...]:
    """The namelist entries that 'DiffusionConfig.from_fortran_dict' reads, in field order."""
    hints = typing.get_type_hints(DiffusionConfig, include_extras=True)
    entries = []
    for name, option in options.ConfigOption.iter_from_config_class(DiffusionConfig):
        icon = option.icon_equivalent
        if icon is None or (isinstance(icon, options.IconOption) and not icon.read_from_icon):
            continue
        if not isinstance(icon, options.IconOption) or len(icon.path) != 1:
            raise NotImplementedError(
                f"DiffusionConfig.{name}: {icon!r} is not one namelist entry."
            )
        entries.append(Entry(name, icon.path[0], icon.name, icon.list_to_value, _kind(hints[name])))
    return tuple(entries)


def diffusion_config(namelists: _config.Namelists) -> tuple[DiffusionConfig | None, list[str]]:
    """
    icon4py's DiffusionConfig from ICON's namelist output: 'DiffusionConfig.from_fortran_dict',
    after a check of every entry it reads (present, readable, of its type), with ICON's
    configured value of the field it does not read from ICON ('_config.Loutshs.configured'; the
    plugin replaces it by ICON's value at run time in the secondary constructor). Returns the
    configuration and the problems ('None' and the problems if there are any).
    """
    errors = []
    for entry in diffusion_entries():
        where = f"{entry.group}: {entry.name} (DiffusionConfig.{entry.field})"
        try:
            values = namelists.get(entry.group, entry.name)
        except _config.NamelistError as error:
            errors.append(f"{where}: {error}")
            continue
        if values is None:
            errors.append(f"{_config.NAMELIST_FILE} has no {where}.")
        elif not entry.domain_value and len(values) != 1:
            errors.append(f"{where}: {len(values)} values, expected one.")
        elif type(values[0]) is not entry.kind:
            errors.append(f"{where} = {values[0]!r}, expected a {entry.kind.__name__}.")
    loutshs, problems = _config.loutshs(namelists)
    errors += problems
    if errors or loutshs is None:
        return None, errors
    try:
        overrides = {_config.LOUTSHS_FIELD: loutshs.configured}
        return DiffusionConfig.from_fortran_dict(namelists.fortran_dict(), **overrides), []
    except (NotImplementedError, ValueError, TypeError, KeyError) as error:
        return None, [f"DiffusionConfig.from_fortran_dict: {error!r}"]
