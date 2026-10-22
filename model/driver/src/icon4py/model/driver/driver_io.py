# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Driver-side glue between the model state and the ``icon4py.model.common.io`` module.

The variables a run can write come from four sources:

- the prognostic state (:func:`prognostic_state_to_dataarrays`),
- the active tracers (:func:`tracer_state_to_dataarrays`),
- the standard diagnostics diagnosed from the prognostic state and the tracers
  (:class:`DiagnosticsComputer`, :func:`diagnostic_fields_to_dataarrays`),
- the diagnostic outputs of the physics processes, such as the precipitation fluxes
  (:func:`physics_diagnostic_outputs`, :func:`physics_diagnostics_to_dataarrays`).

:func:`validate_output_variables` checks the field groups of an ``IOConfig`` against
these sources at setup; :class:`OutputStateAssembler` assembles, at a capture step,
exactly the variables the capturing groups request. :func:`default_io_config` builds
the built-in default group (every variable of the prognostic state and the standard
diagnostics), and :func:`create_io_monitor` the ``IOMonitor`` that writes one file
series per field group.
"""

import pathlib
import uuid
from collections.abc import Collection, Mapping
from typing import Any, Final, NamedTuple, Self

import gt4py.next as gtx
import xarray as xr

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
from icon4py.model.common import dimension as dims, exceptions, time, type_alias as ta
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import base as grid_base, horizontal as h_grid, vertical as v_grid
from icon4py.model.common.interpolation import interpolation_attributes as intp_attr
from icon4py.model.common.interpolation.stencils import edge_2_cell_vector_rbf_interpolation as rbf
from icon4py.model.common.io import io as common_io, utils as io_utils
from icon4py.model.common.metrics import metrics_attributes as metrics_attr
from icon4py.model.common.physics.thermodynamics import compute_pressure, compute_temperature
from icon4py.model.common.states import (
    data as state_data,
    factory,
    prognostic_state as prognostics,
    static_fields,
    tracer_states,
)
from icon4py.model.common.utils import data_allocation as data_alloc


#: File-name stub of the built-in default output group (a counter + the backend's suffix,
#: ``.nc`` or ``.zarr``, is appended).
DEFAULT_OUTPUT_BASENAME: Final[str] = "icon4py_output"


# --------------------------------------------------------------------------------------
# Prognostic fields
# --------------------------------------------------------------------------------------


#: Prognostic output variables, selected by CF name from the
#: ``states.data.PROGNOSTIC_CF_ATTRIBUTES`` catalog (which also holds fields the driver does
#: not output, e.g. ``tangential_velocity``). The metadata and the state attribute
#: (``icon_var_name``) come from that catalog; this list only selects which entries to emit.
PROGNOSTIC_VARIABLES: Final[list[str]] = [
    "air_density",
    "exner_function",
    "virtual_potential_temperature",
    "upward_air_velocity",
    "normal_velocity",
]


def prognostic_state_to_dataarrays(
    prognostic_state: prognostics.PrognosticState,
    variables: Collection[str] | None = None,
) -> dict[str, xr.DataArray]:
    """Assemble a CF/UGRID-annotated model-state dict from a ``PrognosticState``."""
    selected = PROGNOSTIC_VARIABLES if variables is None else variables

    state: dict[str, xr.DataArray] = {}
    for name in selected:
        try:
            metadata = state_data.PROGNOSTIC_CF_ATTRIBUTES[name]
        except KeyError as err:
            raise ValueError(
                f"Unknown prognostic output variable '{name}'. "
                f"Known variables are: {PROGNOSTIC_VARIABLES}."
            ) from err
        assert metadata.icon_var_name is not None, (
            f"prognostic output '{name}' must declare icon_var_name to be read from the state"
        )
        field = getattr(prognostic_state, metadata.icon_var_name)
        state[name] = io_utils.to_data_array(
            field,
            metadata,
            to_host=True,
        )
    return state


# --------------------------------------------------------------------------------------
# Tracers
# --------------------------------------------------------------------------------------


def tracer_state_to_dataarrays(
    tracer_state: tracer_states.TracerState, variables: Collection[str]
) -> dict[str, xr.DataArray]:
    """Assemble CF/UGRID-annotated DataArrays of the requested active tracers.

    The variables are the keys of ``states.data.COMMON_TRACER_CF_ATTRIBUTES`` (``qv``,
    ``qc``, ...), which are also the names of the ``TracerState`` fields.
    """
    active: dict[str, gtx.Field] = {
        tracer.name: tracer.field for tracer in tracer_state.active_fields()
    }
    state: dict[str, xr.DataArray] = {}
    for name in variables:
        try:
            metadata = state_data.COMMON_TRACER_CF_ATTRIBUTES[name]
        except KeyError as err:
            raise ValueError(
                f"Unknown tracer output variable '{name}'. Known tracers are: "
                f"{', '.join(state_data.COMMON_TRACER_CF_ATTRIBUTES)}."
            ) from err
        if name not in active:
            raise ValueError(f"Tracer output variable '{name}' is not active in the tracer state.")
        state[name] = io_utils.to_data_array(active[name], metadata, to_host=True)
    return state


# --------------------------------------------------------------------------------------
# Diagnostic fields
# --------------------------------------------------------------------------------------


DIAGNOSTIC_VARIABLES: Final[list[str]] = [
    "eastward_wind",
    "northward_wind",
    "temperature",
    "virtual_temperature",
    "pressure",
    "surface_pressure",
]

#: The output variables of the built-in default group (prognostic + diagnostic).
DEFAULT_OUTPUT_VARIABLES: Final[list[str]] = [*PROGNOSTIC_VARIABLES, *DIAGNOSTIC_VARIABLES]


class _Programs(NamedTuple):
    """Which programs of the ``DiagnosticsComputer`` a set of variables needs."""

    temperature: bool
    winds: bool
    pressure: bool

    @classmethod
    def needed_by(cls, variables: Collection[str]) -> Self:
        requested = frozenset(variables)
        pressure = bool(requested & {"pressure", "surface_pressure"})
        return cls(
            # the pressure program integrates the virtual temperature hydrostatically
            temperature=pressure or bool(requested & {"temperature", "virtual_temperature"}),
            winds=bool(requested & {"eastward_wind", "northward_wind"}),
            pressure=pressure,
        )


class DiagnosticsComputer:
    """Diagnoses the requested standard diagnostic output fields.

    Three programs: temperature and virtual temperature from ``theta_v``, ``exner`` and
    the active tracers (an inactive species counts as zero, so a dry run gives the
    dry-air temperature); the cell winds by RBF interpolation of ``vn``; and the pressure
    on the model levels and at the surface by vertical integration from the diagnosed
    surface pressure, which needs the virtual temperature. :meth:`compute` runs only the
    programs the variables of the call need (``_Programs``), so a group that writes just
    the winds pays for neither temperature nor pressure.

    The scratch/output buffers are allocated **once** and reused on every call (no
    per-step allocation, significant on GPU backends); the static inputs of the
    requested families are read once from the field sources at construction.

    TODO(kotsaloscv): refactor once the driver groups model state -- derive these
    diagnostic buffers from that shared grouping instead of recomputing them here.
    Physics relies heavily on the same diagnostic variables, so they should be shared
    rather than duplicated across IO and physics.
    """

    def __init__(
        self,
        *,
        grid: grid_base.Grid,
        backend: gtx.typing.Backend | None,
        variables: Collection[str],
        metrics: factory.FieldSource,
        interpolation: factory.FieldSource,
    ) -> None:
        requested = frozenset(variables)
        if unknown := requested - set(DIAGNOSTIC_VARIABLES):
            raise ValueError(
                f"Unknown diagnostic output variables: {', '.join(sorted(unknown))}. Known "
                f"variables are: {', '.join(DIAGNOSTIC_VARIABLES)}."
            )
        self._variables = requested
        self._grid = grid
        self._backend = backend
        self._num_levels = grid.num_levels
        cell_domain = h_grid.domain(dims.CellDim)
        self._end_cell_end = grid.end_index(cell_domain(h_grid.Zone.END))
        self._cell_lateral_boundary_level_2 = grid.end_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )

        def _zero_full() -> gtx.Field:
            return data_alloc.zero_field(
                grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
            )

        def _zero_interface() -> gtx.Field:
            return data_alloc.zero_field(
                grid,
                dims.CellDim,
                dims.KHalfDim,
                dtype=ta.wpfloat,
                allocator=backend,
            )

        # stands in for every inactive tracer species (read only, so one buffer serves all)
        self._zero_tracer = _zero_full()
        self._temperature = _zero_full()
        self._virtual_temperature = _zero_full()
        self._u = _zero_full()
        self._v = _zero_full()
        self._pressure = _zero_full()
        self._pressure_ifc_on_model_levels = _zero_full()
        # Typed as Any: gt4py's NDArrayObject protocol does not expose __setitem__, so the
        # in-place copy of the surface row in ``compute`` would not type-check against the
        # precise Field type.
        self._pressure_on_cells_half_levels: Any = _zero_interface()
        self._surface_pressure: Any = data_alloc.zero_field(
            grid, dims.CellDim, dtype=ta.wpfloat, allocator=backend
        )

        # static inputs, fetched only for the programs some variable needs
        programs = _Programs.needed_by(requested)
        self._ddqz_z_full: gtx.Field | None = (
            metrics.get(metrics_attr.DDQZ_Z_FULL) if programs.pressure else None
        )
        self._rbf_vec_coeff_c1: gtx.Field | None = None
        self._rbf_vec_coeff_c2: gtx.Field | None = None
        if programs.winds:
            self._rbf_vec_coeff_c1 = interpolation.get(intp_attr.RBF_VEC_COEFF_C1)
            self._rbf_vec_coeff_c2 = interpolation.get(intp_attr.RBF_VEC_COEFF_C2)

    @property
    def variables(self) -> frozenset[str]:
        """The diagnostic variables this computer can provide."""
        return self._variables

    def _tracer_or_zero(self, field: gtx.Field | None) -> gtx.Field:
        return self._zero_tracer if field is None else field

    def compute(
        self,
        prognostic_state: prognostics.PrognosticState,
        tracer_state: tracer_states.TracerState,
        variables: Collection[str],
    ) -> dict[str, gtx.Field]:
        """Diagnose the requested variables (a subset of those given at construction).

        Buffers are overwritten in place; the returned fields are those buffers.
        """
        requested = frozenset(variables)
        if unknown := requested - self._variables:
            raise ValueError(
                f"Diagnostic output variables not configured for this computer: "
                f"{', '.join(sorted(unknown))}."
            )
        backend = self._backend
        num_levels = self._num_levels
        end_cell_end = self._end_cell_end
        programs = _Programs.needed_by(requested)

        if programs.temperature:
            compute_temperature.compute_virtual_temperature_and_temperature.with_backend(backend)(
                qv=self._tracer_or_zero(tracer_state.qv),
                qc=self._tracer_or_zero(tracer_state.qc),
                qi=self._tracer_or_zero(tracer_state.qi),
                qr=self._tracer_or_zero(tracer_state.qr),
                qs=self._tracer_or_zero(tracer_state.qs),
                qg=self._tracer_or_zero(tracer_state.qg),
                theta_v=prognostic_state.theta_v,
                exner=prognostic_state.exner,
                virtual_temperature=self._virtual_temperature,
                temperature=self._temperature,
                horizontal_start=0,
                horizontal_end=end_cell_end,
                vertical_start=0,
                vertical_end=num_levels,
                offset_provider={},
            )

        if programs.winds:
            assert self._rbf_vec_coeff_c1 is not None and self._rbf_vec_coeff_c2 is not None
            rbf.edge_2_cell_vector_rbf_interpolation.with_backend(backend)(
                p_e_in=prognostic_state.vn,
                ptr_coeff_1=self._rbf_vec_coeff_c1,
                ptr_coeff_2=self._rbf_vec_coeff_c2,
                p_u_out=self._u,
                p_v_out=self._v,
                horizontal_start=self._cell_lateral_boundary_level_2,
                horizontal_end=end_cell_end,
                vertical_start=0,
                vertical_end=num_levels,
                offset_provider={"C2E2C2E": self._grid.get_connectivity("C2E2C2E")},
            )

        if programs.pressure:
            assert self._ddqz_z_full is not None
            compute_pressure.compute_surface_and_hydrostatic_pressure.with_backend(backend)(
                exner=prognostic_state.exner,
                virtual_temperature=self._virtual_temperature,
                ddqz_z_full=self._ddqz_z_full,
                pressure=self._pressure,
                pressure_ifc_on_model_levels=self._pressure_ifc_on_model_levels,
                pressure_ifc=self._pressure_on_cells_half_levels,
                horizontal_start=0,
                horizontal_end=end_cell_end,
                vertical_start=0,
                vertical_end=num_levels,
                offset_provider={},
            )
            # the surface pressure is the bottom row of the half-level pressure
            self._surface_pressure.ndarray[:] = self._pressure_on_cells_half_levels.ndarray[
                :, num_levels
            ]

        buffers = {
            "eastward_wind": self._u,
            "northward_wind": self._v,
            "temperature": self._temperature,
            "virtual_temperature": self._virtual_temperature,
            "pressure": self._pressure,
            "surface_pressure": self._surface_pressure,
        }
        return {name: buffers[name] for name in DIAGNOSTIC_VARIABLES if name in requested}


def diagnostic_fields_to_dataarrays(
    diagnostic_fields: Mapping[str, gtx.Field],
) -> dict[str, xr.DataArray]:
    """Assemble CF/UGRID-annotated DataArrays from computed diagnostic fields."""
    state: dict[str, xr.DataArray] = {}
    for name, field in diagnostic_fields.items():
        metadata = state_data.DIAGNOSTIC_CF_ATTRIBUTES[name]
        state[name] = io_utils.to_data_array(
            field,
            metadata,
            to_host=True,
        )
    return state


# --------------------------------------------------------------------------------------
# Physics diagnostics
# --------------------------------------------------------------------------------------


def physics_diagnostic_outputs(
    physics_diagnostics: physics_state.DiagnosticsStore | None,
) -> dict[str, physics_state.DiagnosticOutput]:
    """The diagnostic outputs of the physics processes offered as output variables.

    Keyed by the CF ``standard_name`` of the output's metadata, e.g. ``precipitation_flux``
    for the ``pflx`` output of muphys; surface (purely horizontal) outputs are offered like
    the others. Without physics there are none. The buffers hold the values of the last
    step the process ran, and zeros before that.
    """
    outputs: dict[str, physics_state.DiagnosticOutput] = {}
    if physics_diagnostics is None:
        return outputs
    for output in physics_diagnostics.outputs():
        name = output.metadata.standard_name
        if name in outputs:
            raise ValueError(
                f"Physics diagnostic output '{name}' is provided by more than one process: "
                f"{outputs[name].process_name}, {output.process_name}."
            )
        outputs[name] = output
    return outputs


def physics_diagnostics_to_dataarrays(
    outputs: Mapping[str, physics_state.DiagnosticOutput], variables: Collection[str]
) -> dict[str, xr.DataArray]:
    """Assemble CF/UGRID-annotated DataArrays of the requested physics diagnostics."""
    return {
        name: io_utils.to_data_array(outputs[name].field, outputs[name].metadata, to_host=True)
        for name in variables
    }


# --------------------------------------------------------------------------------------
# Output configuration
# --------------------------------------------------------------------------------------


def default_io_config(
    output_path: pathlib.Path = common_io.DEFAULT_OUTPUT_PATH,
    *,
    backend: common_io.OutputBackend = common_io.OutputBackend.ZARR,
    mode: common_io.OutputMode = common_io.OutputMode.DISTRIBUTED,
) -> common_io.IOConfig:
    """The built-in default output configuration (``--enable-output`` without an ``io`` section).

    One field group holding all prognostic and diagnostic variables, captured every
    step, written below ``output_path`` in the given format and write strategy.
    """
    return common_io.IOConfig(
        output_path=output_path,
        field_groups=[
            common_io.FieldGroupIOConfig(
                basename=DEFAULT_OUTPUT_BASENAME,
                variables=list(DEFAULT_OUTPUT_VARIABLES),
                backend=backend,
                mode=mode,
                nc_title="ICON4Py output",
                nc_comment="Fields computed by ICON4Py.",
            )
        ],
    )


def requested_output_variables(io_config: common_io.IOConfig) -> set[str]:
    """The union of the variables of all field groups."""
    return {name for group in io_config.field_groups for name in group.variables}


def output_variable_sources(
    *,
    tracer_config: tracer_states.TracerConfig | None,
    physics_outputs: Mapping[str, physics_state.DiagnosticOutput],
) -> dict[str, list[str]]:
    """Every variable the run can write, with the names of the sources providing it.

    A variable is normally provided by exactly one source; a physics output whose
    standard name coincides with another source's variable name is ambiguous and
    rejected by :func:`validate_output_variables` if a field group requests it.
    """
    sources: dict[str, Collection[str]] = {
        "prognostic": PROGNOSTIC_VARIABLES,
        "diagnostic": DIAGNOSTIC_VARIABLES,
        "tracer": tracer_config.active_names if tracer_config is not None else (),
        "physics": physics_outputs,
    }
    providers: dict[str, list[str]] = {}
    for source, names in sources.items():
        for name in names:
            providers.setdefault(name, []).append(source)
    return providers


def validate_output_variables(
    io_config: common_io.IOConfig, sources: Mapping[str, Collection[str]]
) -> None:
    """Fail at setup if a field group requests a variable the run does not provide.

    ``sources`` maps every available variable to the sources providing it (see
    :func:`output_variable_sources`); a requested variable with several sources is
    rejected as ambiguous.
    """
    for group in io_config.field_groups:
        if unknown := sorted(set(group.variables) - set(sources)):
            raise exceptions.InvalidConfigError(
                f"Unknown output variables in field group '{group.basename}': "
                f"{', '.join(unknown)}. Available output variables are: "
                f"{', '.join(sorted(sources))}."
            )
        for name in group.variables:
            if len(sources[name]) > 1:
                raise exceptions.InvalidConfigError(
                    f"Output variable '{name}' of field group '{group.basename}' is "
                    f"ambiguous: provided by the {' and the '.join(sources[name])} fields."
                )


def validate_output_config(
    io_config: common_io.IOConfig,
    *,
    tracer_config: tracer_states.TracerConfig | None,
    physics_diagnostics: physics_state.DiagnosticsStore | None,
) -> None:
    """Check the field groups against the variables the run provides (see above)."""
    validate_output_variables(
        io_config,
        output_variable_sources(
            tracer_config=tracer_config,
            physics_outputs=physics_diagnostic_outputs(physics_diagnostics),
        ),
    )


class OutputStateAssembler:
    """Assembles the output state the capturing field groups request from its sources.

    The available variables are the prognostics, the standard diagnostics, the active
    tracers and the physics diagnostics of the run; the field groups are validated
    against them on construction, so an unknown variable fails before the first time
    step. The diagnostics computer is only built if some group requests a diagnostic.
    """

    def __init__(
        self,
        *,
        io_config: common_io.IOConfig,
        grid: grid_base.Grid,
        backend: gtx.typing.Backend | None,
        static_field_factories: static_fields.StaticFieldFactories,
        tracer_config: tracer_states.TracerConfig | None,
        physics_diagnostics: physics_state.DiagnosticsStore | None,
    ) -> None:
        self._tracer_names: tuple[str, ...] = (
            tracer_config.active_names if tracer_config is not None else ()
        )
        self._physics_outputs = physics_diagnostic_outputs(physics_diagnostics)
        sources = output_variable_sources(
            tracer_config=tracer_config, physics_outputs=self._physics_outputs
        )
        validate_output_variables(io_config, sources)
        self._available = frozenset(sources)

        requested = requested_output_variables(io_config)
        requested_diagnostics = [name for name in DIAGNOSTIC_VARIABLES if name in requested]
        self._diagnostics_computer: DiagnosticsComputer | None = (
            DiagnosticsComputer(
                grid=grid,
                backend=backend,
                variables=requested_diagnostics,
                metrics=static_field_factories.metrics,
                interpolation=static_field_factories.interpolation,
            )
            if requested_diagnostics
            else None
        )

    @property
    def available_variables(self) -> frozenset[str]:
        """The variables this run can write."""
        return self._available

    def assemble(
        self,
        variables: Collection[str],
        prognostic_state: prognostics.PrognosticState,
        tracer_state: tracer_states.TracerState,
    ) -> dict[str, xr.DataArray]:
        """The CF/UGRID-annotated DataArrays of exactly the requested variables.

        The DataArrays reference the live state (see ``io.utils.to_data_array``): they
        must be written before the next step mutates it.
        """
        requested = set(variables)
        state: dict[str, xr.DataArray] = {}
        if prognostic := [name for name in PROGNOSTIC_VARIABLES if name in requested]:
            state.update(prognostic_state_to_dataarrays(prognostic_state, prognostic))
        if tracers := [name for name in self._tracer_names if name in requested]:
            state.update(tracer_state_to_dataarrays(tracer_state, tracers))
        if diagnostics := [name for name in DIAGNOSTIC_VARIABLES if name in requested]:
            assert self._diagnostics_computer is not None
            fields = self._diagnostics_computer.compute(prognostic_state, tracer_state, diagnostics)
            state.update(diagnostic_fields_to_dataarrays(fields))
        if physics := [name for name in self._physics_outputs if name in requested]:
            state.update(physics_diagnostics_to_dataarrays(self._physics_outputs, physics))
        return state


# --------------------------------------------------------------------------------------
# Monitor factory
# --------------------------------------------------------------------------------------


def create_io_monitor(
    *,
    io_config: common_io.IOConfig,
    grid_file_path: pathlib.Path,
    grid: grid_base.Grid,
    vertical_grid: v_grid.VerticalGrid,
    dtime: time.RelativeTime,
    process_props: decomposition_defs.ProcessProperties,
    decomposition_info: decomposition_defs.DecompositionInfo | None,
) -> common_io.IOMonitor:
    """Build the ``IOMonitor`` writing the field groups of ``io_config`` below its
    ``output_path``.

    In a distributed run (multi-rank ``process_props``) ``decomposition_info`` is
    required; each group's ``mode`` selects how the ranks write (see
    ``common_io.OutputMode``).
    """
    return common_io.IOMonitor(
        config=io_config,
        vertical_size=vertical_grid,
        horizontal_size=grid.config.horizontal_config,
        grid_file_name=grid_file_path,
        # Grid.id holds the file's `uuidOfHGrid` as a string; the IO layer wants a UUID.
        grid_id=uuid.UUID(grid.id),
        dtime=dtime,
        process_props=process_props,
        decomposition_info=decomposition_info,
    )
