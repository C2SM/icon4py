# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The two-layer coupling pieces of the PhysicsDriver.

All physics processes are coupled in PARALLEL: every process computes its
tendencies from the same frozen step-entry state, and the summed tendencies
are applied to the prognostic state exactly once, after all processes.

Deliberately NOT implemented here: ICON AES couples microphyscs and turbulence sequentially
(each process sees the previous one's provisional update). Reintroducing that
would require per-process state advances and working tracer buffers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, model_options
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
from icon4py.model.common.states import diagnostic_state
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.common.states import factory, prognostic_state as prognostics, tracer_states


# The six moisture species physics requires from the TracerState.
MOISTURE_SPECIES: Final = ("qv", "qc", "qi", "qr", "qs", "qg")


class EntryState:
    """The PhysicsState facade: model-state pointers + diagnosed fields (ICON dyn2phy, ``field%ta`` / ``diag%``).

    Diagnosed once at ``PhysicsDriver.run`` entry; NEVER updated between processes —
    all processes are parallel-coupled and see the same step-entry state. The
    diagnostics gathered here used to be duplicated in each process's state adapter.
    Pressure is step-start by construction, as in ICON.
    """

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        interpolation: factory.FieldSource,
        metrics: factory.FieldSource,
        backend: gtx_typing.Backend | None = None,
    ) -> None:
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

        self.diagnostics = diagnostic_state.initialize_diagnostic_state(grid, backend)
        # Scratch for the pressure scan: a scan's range is deduced from its single
        # output domain, so the half-level result lands on model levels first and
        # compute_surface_and_hydrostatic_pressure copies it up (see that program).
        self._pressure_ifc_on_model_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=backend
        )

        # Pointers into the model state — bound by every diagnose_from call
        self.exner: gtx.Field | None = None
        self.theta_v: gtx.Field | None = None
        self.rho: gtx.Field | None = None
        self.vn: gtx.Field | None = None
        self.w: gtx.Field | None = None
        self.tracers: tracer_states.TracerState | None = None

    def diagnose_from(
        self,
        prognostic: prognostics.PrognosticState,
        tracers: tracer_states.TracerState,
    ) -> None:
        """Bind the model-state pointers and diagnose the physics fields (dyn2phy).

        After this call the facade is complete: every model-state field the physics
        may touch is reachable as ``entry_state.<name>`` — the raw PrognosticState
        and TracerState never appear below the PhysicsState layer. Strictly
        read-only for the processes: only the driver's single apply step writes,
        through the pointers bound here.
        """
        # Physics needs all six moisture species; TracerState fields are optional (a
        # tracer may be inactive per TracerConfig), so fail loudly once here rather
        # than feed None into the physics.
        missing = [name for name in MOISTURE_SPECIES if getattr(tracers, name) is None]
        if missing:
            raise ValueError(
                f"physics requires all moisture species active in the TracerState; missing: {missing}"
            )

        # Pointers into the model state — same memory, physics names (no copies)
        self.exner = prognostic.exner
        self.theta_v = prognostic.theta_v
        self.rho = prognostic.rho
        self.vn = prognostic.vn
        self.w = prognostic.w
        self.tracers = tracers

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


class TendencyAccumulators:
    """Per-variable tendency sums over the processes of one timestep (ICON ``tend%*_phy``).

    Buffers are keyed by output name (``tend_*``) and allocated lazily from the
    first contributing field's domain, so cell-(K) and cell-(K+1) shapes work
    alike. Only outputs whose metadata carries ``kind == "tendency"`` accumulate;
    the rest are diagnostics, written by the granules directly into the
    layer-owned buffers of the ``DiagnosticsStore``.
    """

    def __init__(self, *, backend: gtx_typing.Backend | None = None) -> None:
        self._backend = backend
        self.acc: dict[str, gtx.Field] = {}

    def zero(self) -> None:
        """Reset all accumulators; called by the driver at the start of every run."""
        for buffer in self.acc.values():
            buffer.ndarray[...] = 0.0  # type: ignore[index] # NDArrayObject Protocol doesn't support this

    def accumulate(self, outputs: dict, outputs_properties: dict) -> None:
        """Add a process's tendency outputs to the per-variable sums.

        Element-wise sum with no neighbor access — a plain array operation on the
        field buffers, valid for numpy and cupy backing storage alike.
        """
        for name, props in outputs_properties.items():
            if props.get("kind") != "tendency":
                continue
            field = outputs[name]
            buffer = self.acc.get(name)
            if buffer is None:
                buffer = gtx.zeros(field.domain, dtype=field.dtype, allocator=self._backend)
                self.acc[name] = buffer
            buffer.ndarray[...] += field.ndarray  # type: ignore[index] # NDArrayObject Protocol doesn't support this


class ApplyToPrognostic:
    """The single application of the accumulated tendencies to the model state.

    The phy2dyn conversion of ``mo_interface_iconam_aes`` (``:513`` and following),
    executed once per timestep — previously each process's ``scatter_to_prognostic``
    repeated the exner/theta_v EOS update. Order: tracers first (the EOS update uses
    the final moisture), then temperature -> exner/theta_v, then the winds.
    Tendencies absent from the accumulators are skipped (e.g. no ``tend_u/v/w`` in a
    muphys-only configuration).
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
        # w has KDim+1 half-levels — same stencil, but domain extends to nlev+1
        self._apply_tendency_w = model_options.setup_program(
            program=generic_math_operations.compute_field_a_plus_coeff_times_field_b_on_cell_k,
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
        self._new_te = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self._tv_tendency = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self._ddt_vn = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=backend)

    def __call__(
        self,
        entry_state: EntryState,
        accumulators: TendencyAccumulators,
        dt_seconds: float,
    ) -> None:
        """Write the accumulated tendencies to the model state — through the facade's pointers."""
        acc = accumulators.acc
        tracers = entry_state.tracers
        assert tracers is not None, "diagnose_from must run before apply"

        # 1. Tracers: q += dt * sum of tendencies (mo_interface_iconam_aes:513)
        for name in MOISTURE_SPECIES:
            key = f"tend_{name}"
            if key in acc:
                tracer = getattr(tracers, name)
                self._apply_tendency(
                    field_a=tracer,
                    coeff=dt_seconds,
                    field_b=acc[key],
                    output_field=tracer,
                )

        # 2. Temperature -> exner/theta_v: ONE exact-EOS update from the entry T
        #    plus the summed tendency, with the final (post step 1) moisture.
        if "tend_temperature" in acc:
            self._apply_tendency(
                field_a=entry_state.diagnostics.temperature,
                coeff=dt_seconds,
                field_b=acc["tend_temperature"],
                output_field=self._new_te,
            )
            self._compute_virtual_temperature_tendency(
                dtime=dt_seconds,
                qv=tracers.qv,
                qc=tracers.qc,
                qi=tracers.qi,
                qr=tracers.qr,
                qs=tracers.qs,
                qg=tracers.qg,
                temperature=self._new_te,
                virtual_temperature=entry_state.diagnostics.virtual_temperature,
                virtual_temperature_tendency=self._tv_tendency,
            )
            self._update_exner_and_theta_v(
                rho=entry_state.rho,
                virtual_temperature=entry_state.diagnostics.virtual_temperature,
                virtual_temperature_tendency=self._tv_tendency,
                dtime=dt_seconds,
                exner=entry_state.exner,
                theta_v=entry_state.theta_v,
            )

        # 3. Winds: ONE projection of the summed (u, v) tendencies onto edge normals
        if "tend_u" in acc:
            self._compute_vn_from_uv(
                u=acc["tend_u"],
                v=acc["tend_v"],
                vn=self._ddt_vn,
            )
            self._apply_tendency_vn(
                field_a=entry_state.vn,
                coeff=dt_seconds,
                field_b=self._ddt_vn,
                output_field=entry_state.vn,
            )

        # 4. w (KDim+1 half-levels)
        if "tend_w" in acc:
            self._apply_tendency_w(
                field_a=entry_state.w,
                coeff=dt_seconds,
                field_b=acc["tend_w"],
                output_field=entry_state.w,
            )


class DiagnosticsStore:
    """Layer-owned storage of the process output diagnostics (ICON ``field%`` spirit).

    Allocates every non-tendency output from its declared metadata (``dims``,
    optionally ``is_on_half_levels``) at driver construction and hands the
    buffers to the component (direct write): one buffer per field, owned here,
    written by the granule. Keyed per process for order-independence and
    collision-safety. Values are the last computed step's; zeros before a
    process first fires.
    """

    def __init__(self, *, grid: base_grid.Grid, backend: gtx_typing.Backend | None = None) -> None:
        self._grid = grid
        self._backend = backend
        self._store: dict[str, dict[str, gtx.Field]] = {}

    def allocate(self, process_name: str, outputs_properties: dict) -> dict[str, gtx.Field]:
        """Allocate the process's diagnostic buffers from their metadata and keep them.

        Complement rule: everything whose metadata ``kind`` is NOT ``"tendency"``
        is a diagnostic — tendencies stay component-owned (the accumulators sum
        across processes and recycling needs each process's last tendency).
        """
        buffers: dict[str, gtx.Field] = {}
        for name, props in outputs_properties.items():
            if props.get("kind") == "tendency":
                continue
            extend = {dims.KDim: 1} if props.get("is_on_half_levels") else None
            buffers[name] = data_alloc.zero_field(
                self._grid, *props["dims"], extend=extend, allocator=self._backend
            )
        self._store[process_name] = buffers
        return buffers

    def __getitem__(self, process_name: str) -> dict[str, gtx.Field]:
        return self._store[process_name]
