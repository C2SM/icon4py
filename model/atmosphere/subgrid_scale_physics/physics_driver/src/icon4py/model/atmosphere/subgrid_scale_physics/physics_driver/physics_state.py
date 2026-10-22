# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The two-layer coupling pieces of the PhysicsDriver.

The ``PhysicsState`` facade the processes read, and the ``Tendencies`` that advance it
sequentially, as ICON AES does: each process sees the state the previous ones have already
updated, and exner, theta_v and vn are updated once at the end.
"""

from __future__ import annotations

import dataclasses
import typing
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


@dataclasses.dataclass(frozen=True)
class AdvancedState:
    """The state fields the processes' tendencies advance in place, as pointers (no copies).

    A process names the tendency of a field ``tendency_name(<field name>)``.
    """

    temperature: fa.CellKField[ta.wpfloat]
    u: fa.CellKField[ta.wpfloat]
    v: fa.CellKField[ta.wpfloat]
    w: fa.CellKHalfField[ta.wpfloat]
    qv: fa.CellKField[ta.wpfloat]
    qc: fa.CellKField[ta.wpfloat]
    qi: fa.CellKField[ta.wpfloat]
    qr: fa.CellKField[ta.wpfloat]
    qs: fa.CellKField[ta.wpfloat]
    qg: fa.CellKField[ta.wpfloat]

    @classmethod
    def field_dims(cls) -> dict[str, tuple[gtx.Dimension, ...]]:
        hints = typing.get_type_hints(cls)
        return {
            field.name: typing.get_args(typing.get_args(hints[field.name])[0])
            for field in dataclasses.fields(cls)
        }


class PhysicsState:
    """What the processes see in one step (ICON dyn2phy, ``field%ta`` / ``diag%``), in four parts.

    TODO (Yilu): The states in this module need to be vigorously improved when state and component is ready.
    They are workaround so that the refactoring for sequential run is minimal and the advanced variables are clear to users/developers.

    1. Set pointers into the model state (``exner``, ``theta_v``, ``rho``, ``vn``,
       ``w``, ``tracers``).
    2. ``diagnostics``, the caller's ``DiagnosticState``, linked and diagnosed once at
       ``PhysicsDriver.run`` entry. The virtual temperature and the pressure stay at
       step start.
    3. ``advanced``, an ``AdvancedState`` of pointers to the fields that
       ``Tendencies.advance`` updates in place after each process: ``temperature``,
       ``u`` and ``v`` of ``diagnostics``, ``w`` and the moisture tracers.
    4. ``process_outputs``, the diagnostic (non-tendency) output buffers of each
       process. They are allocated once, when the driver is built, and written in
       place by their component. They persist across steps, holding the values of
       the last computed step (zeros before the process has first run).
       ``compute_diagnostics`` never touches them.
    """

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        interpolation: factory.FieldSource,
        metrics: factory.FieldSource,
        backend: gtx_typing.Backend | None = None,
    ) -> None:
        self._grid = grid
        self._backend = backend
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

        # Pointers into the model state — bound by every compute_diagnostics call
        self.exner: gtx.Field | None = None
        self.theta_v: gtx.Field | None = None
        self.rho: gtx.Field | None = None
        self.vn: gtx.Field | None = None
        self.w: gtx.Field | None = None
        self.tracers: tracer_states.TracerState | None = None
        self.advanced: AdvancedState | None = None
        # The caller's diagnostic state — linked by every compute_diagnostics call
        self._diagnostics: diagnostic_state.DiagnosticState | None = None

        # Diagnostic output buffers, keyed by process name, then by output name
        self._process_outputs: dict[str, dict[str, gtx.Field]] = {}

    @property
    def diagnostics(self) -> diagnostic_state.DiagnosticState:
        """The caller's diagnostic state, linked by ``compute_diagnostics``."""
        assert self._diagnostics is not None, "compute_diagnostics must run first"
        return self._diagnostics

    @property
    def process_outputs(self) -> Mapping[str, Mapping[str, gtx.Field]]:
        """The diagnostic output buffers of every process, keyed by process name.

        Keyed per process, so two processes may declare the same output name.
        """
        return self._process_outputs

    def allocate_process_outputs(
        self, process_name: str, outputs_properties: Mapping[str, model.FieldMetaData]
    ) -> dict[str, gtx.Field]:
        """Allocate the process's diagnostic output buffers from their metadata and keep them.

        (TODO Yilu): I think these should also be linked back to some global variables.
        And not all outputs should be allocated or linked here.

        Complement rule: everything whose metadata ``kind`` is NOT ``TENDENCY``
        is a diagnostic. Tendencies stay component-owned: ``Tendencies`` sums
        them across processes, and recycling needs each process's last tendency.
        """
        if process_name in self._process_outputs:
            raise ValueError(f"Process '{process_name}' already has output buffers.")
        buffers: dict[str, gtx.Field] = {}
        for name, props in outputs_properties.items():
            if props.kind == model.FieldKind.TENDENCY:
                continue
            if props.dims is None:
                raise ValueError(
                    f"Diagnostic output '{name}' of process '{process_name}' declares no 'dims', "
                    "which are needed to allocate its buffer."
                )
            buffers[name] = data_alloc.zero_field(self._grid, *props.dims, allocator=self._backend)
        self._process_outputs[process_name] = buffers
        return buffers

    def compute_diagnostics(
        self,
        prognostic: prognostics.PrognosticState,
        tracers: tracer_states.TracerState,
        diagnostics: diagnostic_state.DiagnosticState,
    ) -> None:
        """Bind the model-state pointers, link ``diagnostics`` and diagnose into it (dyn2phy).

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
        self._diagnostics = diagnostics
        self.advanced = AdvancedState(
            temperature=self.diagnostics.temperature,
            u=self.diagnostics.u,
            v=self.diagnostics.v,
            w=prognostic.w,
            **{name: getattr(tracers, name) for name in MOISTURE_SPECIES},
        )

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


class Tendencies:
    """The tendency sums of one timestep, and their application to the state.

    One timestep is ``zero``, then ``advance`` once per process,
    then ``update_prognostics``.

    There is one sum per ``AdvancedState`` field, keyed by its tendency name
    (``tend_<field>``) and allocated as zeros at construction. Only outputs whose
    metadata carries ``kind`` ``TENDENCY`` accumulate, and each must be the tendency
    of an ``AdvancedState`` field; the rest are diagnostics, written by the
    components directly into the ``PhysicsState.process_outputs`` buffers.

    The coupling is sequential, as ICON AES (``fc_xxx = 1``): ``advance`` adds each
    process's tendencies to the tracers, temperature and u/v/w in place, so the next
    process reads the advanced state. ``update_prognostics`` does not add them again: it
    only recomputes exner and theta_v from the final temperature and moisture, and updates
    vn once from the summed (u, v) tendencies.
    """

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        geometry: factory.FieldSource,
        interpolation: factory.FieldSource,
        backend: gtx_typing.Backend | None = None,
    ) -> None:
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
        # TODO(Yilu): rename the following variables including acc?
        self._tv_tendency = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self._ddt_vn = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=backend)

        self._acc: dict[str, gtx.Field] = {
            tendency_name(name): data_alloc.zero_field(grid, *field_dims, allocator=backend)
            for name, field_dims in AdvancedState.field_dims().items()
        }

    @property
    def acc(self) -> Mapping[str, gtx.Field]:
        """The tendencies accumulated so far this step, keyed by output name."""
        return self._acc

    def zero(self) -> None:
        """Reset all sums; called by the driver at the start of every run."""
        for buffer in self._acc.values():
            buffer.ndarray[...] = 0.0  # type: ignore[index] # NDArrayObject Protocol doesn't support this

    def advance(
        self,
        state: PhysicsState,
        outputs: dict[str, gtx.Field],
        outputs_properties: dict[str, model.FieldMetaData],
        dt_seconds: float,
    ) -> None:
        """Sum this process's tendencies and add them to the state in place."""
        if not outputs:
            return
        advanced = state.advanced
        assert advanced is not None
        targets = {
            tendency_name(field.name): getattr(advanced, field.name)
            for field in dataclasses.fields(advanced)
        }
        for name, props in outputs_properties.items():
            if props.kind != model.FieldKind.TENDENCY:
                continue
            if (buffer := self._acc.get(name)) is None:
                raise ValueError(
                    f"Tendency output '{name}' is not the tendency of an 'AdvancedState' field; "
                    f"expected one of {list(self._acc)}."
                )
            field = outputs[name]
            target = targets[name]
            apply_tendency = (
                self._apply_tendency_on_half_levels
                if dims.KHalfDim in target.domain.dims
                else self._apply_tendency
            )
            # the sum over the processes, for the final vn and virtual temperature updates
            apply_tendency(
                field_a=buffer, coeff=ta.wpfloat(1.0), field_b=field, output_field=buffer
            )
            # the state the next process reads (ICON's interfaces advance ``field%*`` in place)
            apply_tendency(field_a=target, coeff=dt_seconds, field_b=field, output_field=target)

    def update_prognostics(
        self,
        state: PhysicsState,
        dt_seconds: float,
        perturbed_exner: fa.CellKField[ta.wpfloat] | None = None,
    ) -> None:
        acc = self._acc
        tracers = state.tracers
        assert tracers is not None, "compute_diagnostics must run before update_prognostics"

        # the tracers, temperature, u, v and w were already advanced by 'advance'
        self._compute_virtual_temperature_tendency(
            dtime=dt_seconds,
            qv=tracers.qv,
            qc=tracers.qc,
            qi=tracers.qi,
            qr=tracers.qr,
            qs=tracers.qs,
            qg=tracers.qg,
            temperature=state.diagnostics.temperature,
            virtual_temperature=state.diagnostics.virtual_temperature,
            virtual_temperature_tendency=self._tv_tendency,
        )
        if perturbed_exner is None:
            self._update_exner_and_theta_v(
                rho=state.rho,
                virtual_temperature=state.diagnostics.virtual_temperature,
                virtual_temperature_tendency=self._tv_tendency,
                dtime=dt_seconds,
                exner=state.exner,
                theta_v=state.theta_v,
            )
        else:
            self._update_exner_theta_v_and_perturbed_exner(
                rho=state.rho,
                virtual_temperature=state.diagnostics.virtual_temperature,
                virtual_temperature_tendency=self._tv_tendency,
                dtime=dt_seconds,
                exner=state.exner,
                theta_v=state.theta_v,
                perturbed_exner=perturbed_exner,
            )

        self._compute_vn_from_uv(
            u=acc[tendency_name("u")],
            v=acc[tendency_name("v")],
            vn=self._ddt_vn,
        )
        self._apply_tendency_vn(
            field_a=state.vn,
            coeff=dt_seconds,
            field_b=self._ddt_vn,
            output_field=state.vn,
        )
