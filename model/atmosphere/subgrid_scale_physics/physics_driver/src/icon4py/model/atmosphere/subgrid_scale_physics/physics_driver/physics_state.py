# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The two-layer coupling pieces of the PhysicsDriver.

The ``PhysicsState`` layer the processes read, and the ``Tendencies`` that advance it
sequentially, as ICON AES does: each process sees the state the previous ones have already
updated, and exner, theta_v and vn are updated once at the end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import gt4py.next as gtx

from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    model_options,
    type_alias as ta,
)
from icon4py.model.common.grid import geometry_attributes
from icon4py.model.common.interpolation import interpolation_attributes
from icon4py.model.common.interpolation.stencils.compute_vn_from_uv import compute_vn_from_uv
from icon4py.model.common.interpolation.stencils.edge_2_cell_vector_rbf_interpolation import (
    edge_2_cell_vector_rbf_interpolation,
)
from icon4py.model.common.math.stencils import generic_math_operations
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.physics.thermodynamics import (
    compute_pressure,
    compute_temperature,
    compute_tendencies,
)
from icon4py.model.common.states import model
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    from collections.abc import Mapping

    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.common.states import (
        diagnostic_state,
        factory,
        prognostic_state as prognostics,
        tracer_states,
    )


# The six moisture species physics requires from the TracerState.
MOISTURE_SPECIES: Final = ("qv", "qc", "qi", "qr", "qs", "qg")


def tendency_name(name: str) -> str:
    """The output name of the tendency of the state field ``name``."""
    return f"tend_{name}"


class PhysicsState:
    """What the processes see in one step (ICON dyn2phy, ``field%ta`` / ``diag%``), in three parts.

    TODO (Yilu): this state and ``Tendencies`` are the untyped stand-ins for the typed
    states of #1510; restructure them on top of it once it is merged.

    1. Set pointers into the model state (``exner``, ``theta_v``, ``rho``, ``vn``,
       ``w``, ``tracers``, and ``air_mass``, the dycore's air mass per unit area that
       ICON hands to the physics as ``mair``).
    2. ``diagnostics``, the caller's ``DiagnosticState``, linked and diagnosed once at
       ``PhysicsDriver.run`` entry. The virtual temperature and the pressure stay at
       step start; ``Tendencies.advance`` updates ``temperature``, ``u`` and ``v`` in
       place after each process, as it does ``w`` and the moisture tracers.
    3. The diagnostic (non-tendency) output buffers of each
       process. They are allocated once, when the driver is built, and written in
       place by their component. They persist across steps, holding the values of
       the last computed step (zeros before the process has first run).
       ``compute_diagnostics`` never touches them. A process reads another's output
       by name through ``diagnostic_output``, as tmx reads the surface fluxes.

    ``compute_diagnostics`` binds and diagnoses at entry (dyn2phy); ``update_prognostics``
    writes exner, theta_v and vn back at exit (phy2dyn).
    """

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        geometry: factory.FieldSource,
        interpolation: factory.FieldSource,
        metrics: factory.FieldSource,
        backend: gtx_typing.Backend | None = None,
    ) -> None:
        self._grid = grid
        self._backend = backend
        num_cells = grid.num_cells
        num_levels = grid.num_levels
        num_edges = grid.num_edges

        full_horizontal = {
            "horizontal_start": gtx.int32(0),
            "horizontal_end": gtx.int32(num_cells),
        }
        full_vertical = {
            "vertical_start": gtx.int32(0),
            "vertical_end": gtx.int32(num_levels),
        }
        edge_horizontal = {
            "horizontal_start": gtx.int32(0),
            "horizontal_end": gtx.int32(num_edges),
        }

        self._ddqz_z_full = metrics.get(metrics_attributes.DDQZ_Z_FULL)

        self._diagnose_temperature = model_options.setup_program(
            program=compute_temperature.compute_virtual_temperature_and_temperature,
            backend=backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        self._compute_surface_and_hydrostatic_pressure = model_options.setup_program(
            program=compute_pressure.compute_surface_and_hydrostatic_pressure,
            backend=backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        # RBF reconstruction of cell-centre wind from edge-normal vn
        self._rbf_interpolation = model_options.setup_program(
            program=edge_2_cell_vector_rbf_interpolation,
            backend=backend,
            constant_args={
                "ptr_coeff_1": interpolation.get(interpolation_attributes.RBF_VEC_COEFF_C1),
                "ptr_coeff_2": interpolation.get(interpolation_attributes.RBF_VEC_COEFF_C2),
            },
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider=grid.connectivities,
        )

        # Scratch for the pressure scan: a scan's range is deduced from its single
        # output domain, so the half-level result lands on model levels first and
        # compute_surface_and_hydrostatic_pressure copies it up (see that program).
        self._pressure_ifc_on_model_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=backend
        )

        # The exit (phy2dyn) programs of update_prognostics
        self._apply_tendency_vn = model_options.setup_program(
            program=generic_math_operations.compute_field_a_plus_coeff_times_field_b_on_edge_k,
            backend=backend,
            horizontal_sizes=edge_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        self._compute_virtual_temperature_tendency = model_options.setup_program(
            program=compute_tendencies.compute_virtual_temperature_tendency,
            backend=backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        self._update_exner_and_theta_v = model_options.setup_program(
            program=compute_temperature.update_exner_and_theta_v,
            backend=backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        self._update_exner_theta_v_and_perturbed_exner = model_options.setup_program(
            program=compute_temperature.update_exner_theta_v_and_perturbed_exner,
            backend=backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        self._compute_vn_from_uv = model_options.setup_program(
            program=compute_vn_from_uv,
            backend=backend,
            constant_args={
                "primal_normal_cell_x": geometry.get(geometry_attributes.EDGE_NORMAL_CELL_U),
                "primal_normal_cell_y": geometry.get(geometry_attributes.EDGE_NORMAL_CELL_V),
                "c_lin_e": interpolation.get(interpolation_attributes.C_LIN_E),
            },
            horizontal_sizes=edge_horizontal,
            vertical_sizes=full_vertical,
            offset_provider=grid.connectivities,
        )

        # Scratch buffers — allocated once
        self._virtual_temperature_tendency = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=backend
        )
        self._vn_tendency = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=backend)

        # Pointers into the model state — bound by every compute_diagnostics call
        self.exner: gtx.Field | None = None
        self.theta_v: gtx.Field | None = None
        self.rho: gtx.Field | None = None
        self.vn: gtx.Field | None = None
        self.w: gtx.Field | None = None
        self.tracers: tracer_states.TracerState | None = None
        self.air_mass: gtx.Field | None = None
        # The caller's diagnostic state — linked by every compute_diagnostics call
        self._diagnostics: diagnostic_state.DiagnosticState | None = None

        # Diagnostic output buffers, keyed by process name, then by output name
        self._process_diagnostics: dict[str, dict[str, gtx.Field]] = {}

    @property
    def diagnostics(self) -> diagnostic_state.DiagnosticState:
        """The caller's diagnostic state, linked by ``compute_diagnostics``."""
        assert self._diagnostics is not None, "compute_diagnostics must run first"
        return self._diagnostics

    def compute_diagnostics(
        self,
        prognostic: prognostics.PrognosticState,
        tracers: tracer_states.TracerState,
        diagnostics: diagnostic_state.DiagnosticState,
        air_mass: fa.CellKField[ta.wpfloat] | None = None,
    ) -> None:
        """Bind the model-state pointers, link ``diagnostics`` and diagnose into it (dyn2phy).

        `air_mass` is optional because only some processes need it; a process that does
        reports its absence.

        After this call the facade is complete: every model-state field the physics
        may touch is reachable as ``state.<name>`` — the raw PrognosticState, DiagnosticState,
        and TracerState never appear below the PhysicsState layer.
        """
        # Physics needs all six moisture species; TracerState fields are optional (a
        # tracer may be inactive per TracerConfig), so fail loudly once here rather
        # than feed None into the physics.
        if missing := [name for name in MOISTURE_SPECIES if getattr(tracers, name) is None]:
            raise ValueError(
                "Physics requires all moisture species to be active in the 'TracerState', "
                f"missing: {missing}."
            )

        # Pointers into the model state — same memory, physics names (no copies)
        self.exner = prognostic.exner
        self.theta_v = prognostic.theta_v
        self.rho = prognostic.rho
        self.vn = prognostic.vn
        self.w = prognostic.w
        self.tracers = tracers
        self.air_mass = air_mass
        self._diagnostics = diagnostics

        # 1. Virtual temperature and temperature
        self._diagnose_temperature(
            qv=tracers.qv,
            qc=tracers.qc,
            qi=tracers.qi,
            qr=tracers.qr,
            qs=tracers.qs,
            qg=tracers.qg,
            theta_v=prognostic.theta_v,
            exner=prognostic.exner,
            virtual_temperature=self.diagnostics.virtual_temperature,
            temperature=self.diagnostics.temperature,
        )

        # 2. Surface pressure at the bottom interface, then the full pressure column
        self._compute_surface_and_hydrostatic_pressure(
            exner=prognostic.exner,
            virtual_temperature=self.diagnostics.virtual_temperature,
            ddqz_z_full=self._ddqz_z_full,
            pressure=self.diagnostics.pressure,
            pressure_ifc_on_model_levels=self._pressure_ifc_on_model_levels,
            pressure_ifc=self.diagnostics.pressure_ifc,
        )

        # 3. Cell-centre (u, v) from edge-normal vn via RBF
        self._rbf_interpolation(
            p_e_in=prognostic.vn,
            p_u_out=self.diagnostics.u,
            p_v_out=self.diagnostics.v,
        )

    def allocate_process_diagnostics(
        self, process_name: str, outputs_properties: Mapping[str, model.FieldMetaData]
    ) -> dict[str, gtx.Field]:
        """Allocate the process's diagnostic output buffers from their metadata and keep them.

        The buffers are the home of each process's diagnostics because the model state has
        no fields for them: the ``DiagnosticState`` holds only what ``compute_diagnostics``
        fills.

        TODO (Yilu): give them a model-level home the output layer can write, with every
        diagnostic bound to it. #1510 types them as each process's ``Output`` state; build
        on that.

        Complement rule: everything whose metadata ``kind`` is NOT ``TENDENCY``
        is a diagnostic. Tendencies stay component-owned: ``Tendencies`` sums
        them across processes, and recycling needs each process's last tendency.

        A diagnostic name has one provider: a second process declaring it is rejected, so
        that ``diagnostic_output`` can hand it to a reader by name alone.
        """
        if process_name in self._process_diagnostics:
            raise ValueError(f"Process '{process_name}' already has output buffers.")
        buffers: dict[str, gtx.Field] = {}
        for name, props in outputs_properties.items():
            if props.kind == model.FieldKind.TENDENCY:
                continue
            if (provider := self._provider_of(name)) is not None:
                raise ValueError(
                    f"Diagnostic output '{name}' is already provided by process '{provider}'."
                )
            if props.dims is None:
                raise ValueError(
                    f"Diagnostic output '{name}' of process '{process_name}' declares no 'dims', "
                    "which are needed to allocate its buffer."
                )
            buffers[name] = data_alloc.zero_field(self._grid, *props.dims, allocator=self._backend)
        self._process_diagnostics[process_name] = buffers
        return buffers

    def _provider_of(self, name: str) -> str | None:
        return next(
            (process for process, buffers in self._process_diagnostics.items() if name in buffers),
            None,
        )

    def diagnostic_output(self, name: str) -> gtx.Field:
        """The diagnostic output ``name`` of the one process that provides it.

        This is how a process reads another's output, as tmx reads the surface fluxes:
        by name through the state, without knowing which process computes it.
        """
        if (provider := self._provider_of(name)) is None:
            raise KeyError(
                f"No process provides the diagnostic '{name}'; is the process computing it enabled?"
            )
        return self._process_diagnostics[provider][name]

    def update_prognostics(
        self,
        sums: Mapping[str, gtx.Field],
        dt_seconds: float,
        perturbed_exner: fa.CellKField[ta.wpfloat] | None = None,
    ) -> tuple[list[gtx.Field], list[gtx.Field]]:
        """Write exner, theta_v, the perturbed exner and vn back at exit (phy2dyn).

        The tracers, temperature, u, v and w were already advanced in place by
        ``Tendencies.advance``; exner and theta_v follow from the final temperature and
        moisture, and vn from the summed wind tendencies ``sums``.

        Returns the fields the physics wrote this step, as ``(cell_fields, edge_fields)``:
        the advanced tracers and w, exner, theta_v, the perturbed exner when given, and vn
        when a process produced a wind tendency. The driver syncs exactly these at the end
        of the step.

        TODO (Yilu): ICON clips negative tracers at the entry and the exit of the interface
        (``iqneg_d2p`` / ``iqneg_p2d``: set to zero and move the deficit over dt into the
        tendency; default off, the APE experiment sets both to 2). Port it with the two
        config members before the exner update here and in ``compute_diagnostics``; the
        APE archive's config has to be regenerated for the switches.
        """
        tracers = self.tracers
        assert tracers is not None, "compute_diagnostics must run before update_prognostics"
        assert self.w is not None and self.exner is not None
        assert self.theta_v is not None and self.vn is not None

        self._compute_virtual_temperature_tendency(
            dtime=dt_seconds,
            qv=tracers.qv,
            qc=tracers.qc,
            qi=tracers.qi,
            qr=tracers.qr,
            qs=tracers.qs,
            qg=tracers.qg,
            temperature=self.diagnostics.temperature,
            virtual_temperature=self.diagnostics.virtual_temperature,
            virtual_temperature_tendency=self._virtual_temperature_tendency,
        )
        if perturbed_exner is None:
            self._update_exner_and_theta_v(
                rho=self.rho,
                virtual_temperature=self.diagnostics.virtual_temperature,
                virtual_temperature_tendency=self._virtual_temperature_tendency,
                dtime=dt_seconds,
                exner=self.exner,
                theta_v=self.theta_v,
            )
        else:
            self._update_exner_theta_v_and_perturbed_exner(
                rho=self.rho,
                virtual_temperature=self.diagnostics.virtual_temperature,
                virtual_temperature_tendency=self._virtual_temperature_tendency,
                dtime=dt_seconds,
                exner=self.exner,
                theta_v=self.theta_v,
                perturbed_exner=perturbed_exner,
            )

        # one projection of the summed (u, v) tendencies onto vn, when a process produced them
        wind_tendencies = {tendency_name("u"), tendency_name("v")} & sums.keys()
        if len(wind_tendencies) == 1:
            (declared,) = wind_tendencies
            raise ValueError(
                f"the horizontal wind tendencies are applied as a pair; got only {declared}"
            )
        written_edges: list[gtx.Field] = []
        if wind_tendencies:
            self._compute_vn_from_uv(
                u=sums[tendency_name("u")],
                v=sums[tendency_name("v")],
                vn=self._vn_tendency,
            )
            self._apply_tendency_vn(
                field_a=self.vn,
                coeff=dt_seconds,
                field_b=self._vn_tendency,
                output_field=self.vn,
            )
            written_edges.append(self.vn)

        written_cells: list[gtx.Field] = [getattr(tracers, name) for name in MOISTURE_SPECIES]
        written_cells += [self.w, self.exner, self.theta_v]
        if perturbed_exner is not None:
            written_cells.append(perturbed_exner)
        return written_cells, written_edges


class Tendencies:
    """The tendency sums of one timestep, and the in-place advance of the state.

    TODO (Yilu): the sums are the untyped stand-in for a typed state of #1510, one field
    per tendency; restructure them on top of it once it is merged.

    One timestep is ``zero``, then ``accumulate`` and ``advance`` once per process,
    then ``PhysicsState.update_prognostics``.

    Sums are keyed by tendency name (``tend_<field>``) and allocated on the first
    contribution, so which sums exist depends on which processes ran. Only outputs whose
    metadata carries ``kind`` ``TENDENCY`` accumulate, and each must be the tendency of
    temperature, u, v, w or a moisture tracer; the rest are diagnostics, written by the
    components directly into the buffers the ``PhysicsState`` allocates for them.

    The coupling is sequential, as ICON AES (``fc_xxx = 1``): ``advance`` adds each
    process's tendencies to the tracers, temperature and u/v/w in place, so the next
    process reads the advanced state. ``PhysicsState.update_prognostics`` does not add
    them again: it only recomputes exner and theta_v from the final temperature and
    moisture, and updates vn once from the summed (u, v) tendencies, when a process
    produced them.
    """

    def __init__(self, *, grid: base_grid.Grid, backend: gtx_typing.Backend | None = None) -> None:
        num_cells = grid.num_cells
        num_levels = grid.num_levels

        full_horizontal = {
            "horizontal_start": gtx.int32(0),
            "horizontal_end": gtx.int32(num_cells),
        }
        full_vertical = {
            "vertical_start": gtx.int32(0),
            "vertical_end": gtx.int32(num_levels),
        }

        self._apply_tendency = model_options.setup_program(
            program=generic_math_operations.compute_field_a_plus_coeff_times_field_b_on_cell_k,
            backend=backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        self._apply_tendency_on_half_levels = model_options.setup_program(
            program=generic_math_operations.compute_field_a_plus_coeff_times_field_b_on_cell_khalf,
            backend=backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels + 1),
            },
            offset_provider={},
        )
        self._backend = backend
        self._sums: dict[str, gtx.Field] = {}

    @property
    def sums(self) -> Mapping[str, gtx.Field]:
        """The tendency sums over the processes so far this step, keyed by tendency name."""
        return self._sums

    def zero(self) -> None:
        """Reset all sums; called by the driver at the start of every run."""
        for buffer in self._sums.values():
            buffer.ndarray[...] = 0.0  # type: ignore[index] # NDArrayObject Protocol doesn't support this

    def accumulate(
        self, outputs: dict[str, gtx.Field], outputs_properties: dict[str, model.FieldMetaData]
    ) -> None:
        """Add a process's tendency outputs to the per-variable sums.

        Element-wise sum with no neighbor access, so a plain array operation on
        the field buffers rather than a stencil. A process that did not run this
        step has no outputs and contributes nothing.
        """
        if not outputs:
            return
        for name, props in outputs_properties.items():
            if props.kind != model.FieldKind.TENDENCY:
                continue
            field = outputs[name]
            if (buffer := self._sums.get(name)) is None:
                buffer = self._sums[name] = gtx.zeros(
                    field.domain, dtype=field.dtype, allocator=self._backend
                )
            buffer.ndarray[...] += field.ndarray  # type: ignore[index] # NDArrayObject Protocol doesn't support this

    def advance(
        self,
        state: PhysicsState,
        outputs: dict[str, gtx.Field],
        outputs_properties: dict[str, model.FieldMetaData],
        dt_seconds: float,
    ) -> None:
        """Add ``dt_seconds`` times a process's tendencies to the state in place.

        This is the sequential coupling: the next process reads the advanced temperature,
        winds, w and tracers, as ICON's interfaces advance their physics state.
        """
        if not outputs:
            return
        tracers = state.tracers
        assert tracers is not None and state.w is not None, "compute_diagnostics must run first"
        targets = {
            tendency_name("temperature"): state.diagnostics.temperature,
            tendency_name("u"): state.diagnostics.u,
            tendency_name("v"): state.diagnostics.v,
            tendency_name("w"): state.w,
            **{tendency_name(name): getattr(tracers, name) for name in MOISTURE_SPECIES},
        }
        for name, props in outputs_properties.items():
            if props.kind != model.FieldKind.TENDENCY:
                continue
            if (target := targets.get(name)) is None:
                raise ValueError(
                    f"Tendency output '{name}' is not the tendency of an advanced field; "
                    f"expected one of {list(targets)}."
                )
            field = outputs[name]
            apply_tendency = (
                self._apply_tendency_on_half_levels
                if dims.KHalfDim in target.domain.dims
                else self._apply_tendency
            )
            apply_tendency(field_a=target, coeff=dt_seconds, field_b=field, output_field=target)
