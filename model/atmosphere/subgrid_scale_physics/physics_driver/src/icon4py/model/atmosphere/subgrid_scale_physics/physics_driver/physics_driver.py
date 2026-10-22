# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``PhysicsDriver`` and its processes."""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import (
    physics_state,
    static_fields,
)
from icon4py.model.common import dimension as dims, framework as fw, model_backends, model_options
from icon4py.model.common.framework import quantities as qty, states
from icon4py.model.common.interpolation.stencils.compute_vn_from_uv import compute_vn_from_uv
from icon4py.model.common.interpolation.stencils.edge_2_cell_vector_rbf_interpolation import (
    edge_2_cell_vector_rbf_interpolation,
)
from icon4py.model.common.math.stencils import generic_math_operations
from icon4py.model.common.physics.thermodynamics import (
    compute_pressure,
    compute_temperature,
    compute_tendencies,
)
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    from icon4py.model.common.grid import base as base_grid


__all__ = ["PhysicsDriver"]


def _present[Q: fw.Quantity](field: fw.Field[Q] | None) -> fw.Field[Q]:
    """A tracer leaf the physics requires."""
    if field is None:
        raise ValueError("physics requires all moisture species active in the TracerState")
    return field


def _array(field: fw.Field[Any]) -> Any:
    # NDArrayObject declares no __setitem__
    return field.ndarray


class PhysicsDriver(fw.Component):
    """
    Runs the physics processes under parallel coupling (ICON AES's aes_phy_main).

    In one step the entry state is diagnosed once from the prognostics, every process active on
    the step reads that same entry state, the tendencies of their outputs are summed and the sum
    is applied to the prognostics once, at the end. The processes never write the prognostics.

    Not implemented: ICON AES couples microphysics and turbulence sequentially (each process
    sees the previous one's provisional update), which would need per-process state advances.

    In place like the diffusion: the caller passes the same buffers as input and output; `run`
    copies the input into `outputs` first where they differ. `rho` is read only.
    """

    class Input(fw.State):
        vn: fw.Field[qty.VnOnEdgeK]
        w: fw.Field[qty.WOnCellKHalf]
        exner: fw.Field[qty.ExnerOnCellK]
        theta_v: fw.Field[qty.ThetaVOnCellK]
        rho: fw.Field[qty.RhoOnCellK]
        qv: fw.Field[qty.QvOnCellK] | None = None
        qc: fw.Field[qty.QcOnCellK] | None = None
        qi: fw.Field[qty.QiOnCellK] | None = None
        qr: fw.Field[qty.QrOnCellK] | None = None
        qs: fw.Field[qty.QsOnCellK] | None = None
        qg: fw.Field[qty.QgOnCellK] | None = None
        dtime: datetime.timedelta
        # the end of the step being integrated
        simulation_current_datetime: datetime.datetime

    class Output(fw.State):
        vn: fw.Field[qty.VnOnEdgeK]
        w: fw.Field[qty.WOnCellKHalf]
        exner: fw.Field[qty.ExnerOnCellK]
        theta_v: fw.Field[qty.ThetaVOnCellK]
        qv: fw.Field[qty.QvOnCellK] | None = None
        qc: fw.Field[qty.QcOnCellK] | None = None
        qi: fw.Field[qty.QiOnCellK] | None = None
        qr: fw.Field[qty.QrOnCellK] | None = None
        qs: fw.Field[qty.QsOnCellK] | None = None
        qg: fw.Field[qty.QgOnCellK] | None = None

    def __init__(
        self,
        processes: Sequence[fw.PhysicsProcess],
        *,
        grid: base_grid.Grid,
        static_state: static_fields.PhysicsDriverStaticState,
        backend: model_backends.BackendLike = None,
    ) -> None:
        self.grid = grid
        self.allocator = model_backends.get_allocator(backend)
        self.processes = tuple(processes)
        # the diagnostics the processes read, derived here each step
        self.diagnostics = fw.allocate(states.Diagnostics, grid, self.allocator)
        # one sum per tendency the processes emit, by output name, made when first seen
        self.accumulators: dict[str, fw.Field[Any]] = {}

        cells = {"horizontal_start": gtx.int32(0), "horizontal_end": gtx.int32(grid.num_cells)}
        edges = {"horizontal_start": gtx.int32(0), "horizontal_end": gtx.int32(grid.num_edges)}
        levels = {"vertical_start": gtx.int32(0), "vertical_end": gtx.int32(grid.num_levels)}
        half_levels = {
            "vertical_start": gtx.int32(0),
            "vertical_end": gtx.int32(grid.num_levels + 1),
        }

        self._ddqz_z_full = static_state.ddqz_z_full
        self._diagnose_temperature = model_options.setup_program(
            program=compute_temperature.compute_virtual_temperature_and_temperature,
            backend=backend,
            horizontal_sizes=cells,
            vertical_sizes=levels,
            offset_provider={},
        )
        self._compute_surface_and_hydrostatic_pressure = model_options.setup_program(
            program=compute_pressure.compute_surface_and_hydrostatic_pressure,
            backend=backend,
            horizontal_sizes=cells,
            vertical_sizes=levels,
            offset_provider={},
        )
        # RBF reconstruction of the cell-centre wind from the edge-normal vn
        self._rbf_interpolation = model_options.setup_program(
            program=edge_2_cell_vector_rbf_interpolation,
            backend=backend,
            constant_args={
                "ptr_coeff_1": static_state.rbf_vec_coeff_c1,
                "ptr_coeff_2": static_state.rbf_vec_coeff_c2,
            },
            horizontal_sizes=cells,
            vertical_sizes=levels,
            offset_provider=grid.connectivities,
        )
        self._apply_tendency = model_options.setup_program(
            program=generic_math_operations.compute_field_a_plus_coeff_times_field_b_on_cell_k,
            backend=backend,
            horizontal_sizes=cells,
            vertical_sizes=levels,
            offset_provider={},
        )
        self._apply_tendency_w = model_options.setup_program(
            program=generic_math_operations.compute_field_a_plus_coeff_times_field_b_on_cell_khalf,
            backend=backend,
            horizontal_sizes=cells,
            vertical_sizes=half_levels,
            offset_provider={},
        )
        self._apply_tendency_vn = model_options.setup_program(
            program=generic_math_operations.compute_field_a_plus_coeff_times_field_b_on_edge_k,
            backend=backend,
            horizontal_sizes=edges,
            vertical_sizes=levels,
            offset_provider={},
        )
        self._compute_virtual_temperature_tendency = model_options.setup_program(
            program=compute_tendencies.compute_virtual_temperature_tendency,
            backend=backend,
            horizontal_sizes=cells,
            vertical_sizes=levels,
            offset_provider={},
        )
        self._update_exner_and_theta_v = model_options.setup_program(
            program=compute_temperature.update_exner_and_theta_v,
            backend=backend,
            horizontal_sizes=cells,
            vertical_sizes=levels,
            offset_provider={},
        )
        self._compute_vn_from_uv = model_options.setup_program(
            program=compute_vn_from_uv,
            backend=backend,
            constant_args={
                "primal_normal_cell_x": static_state.primal_normal_cell_x,
                "primal_normal_cell_y": static_state.primal_normal_cell_y,
                "c_lin_e": static_state.c_lin_e,
            },
            horizontal_sizes=edges,
            vertical_sizes=levels,
            offset_provider=grid.connectivities,
        )

        # A scan's range is deduced from its single output domain, so the half-level pressure
        # lands on model levels first and compute_surface_and_hydrostatic_pressure copies it up.
        self._pressure_ifc_on_model_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=self.allocator
        )
        self._new_temperature = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=self.allocator
        )
        self._virtual_temperature_tendency = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=self.allocator
        )
        self._vn_tendency = data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, allocator=self.allocator
        )

    def run(self, inputs: Input, outputs: Output) -> Output:
        entry = self._diagnose(inputs)
        self._zero_accumulators()
        # the time control reads the start of the step being integrated
        step_start = inputs.simulation_current_datetime - inputs.dtime
        for process in self.processes:
            if (output := process.step(entry, step_start, inputs.dtime)) is not None:
                self._accumulate(output)
        self._apply(entry, inputs.dtime.total_seconds(), outputs)
        return outputs

    def _diagnose(self, inputs: Input) -> physics_state.EntryState:
        """The entry state: the inputs as given and the diagnostics derived from them (dyn2phy)."""
        # The tracers are optional leaves (a tracer may be inactive in the TracerConfig): fail
        # here rather than feed None to the physics.
        if missing := [name for name in states.TRACERS if getattr(inputs, name) is None]:
            raise ValueError(
                f"physics requires all moisture species active in the TracerState; missing: {missing}"
            )
        diagnostics = self.diagnostics
        entry = physics_state.EntryState(
            vn=inputs.vn,
            w=inputs.w,
            exner=inputs.exner,
            theta_v=inputs.theta_v,
            rho=inputs.rho,
            qv=_present(inputs.qv),
            qc=_present(inputs.qc),
            qi=_present(inputs.qi),
            qr=_present(inputs.qr),
            qs=_present(inputs.qs),
            qg=_present(inputs.qg),
            temperature=diagnostics.temperature,
            virtual_temperature=diagnostics.virtual_temperature,
            pressure=diagnostics.pressure,
            pressure_ifc=diagnostics.pressure_ifc,
            u=diagnostics.u,
            v=diagnostics.v,
        )
        # 1. virtual temperature and temperature
        self._diagnose_temperature(
            qv=entry.qv,
            qc=entry.qc,
            qi=entry.qi,
            qr=entry.qr,
            qs=entry.qs,
            qg=entry.qg,
            theta_v=entry.theta_v,
            exner=entry.exner,
            virtual_temperature=entry.virtual_temperature,
            temperature=entry.temperature,
        )
        # 2. surface pressure at the bottom interface, then the full pressure column
        self._compute_surface_and_hydrostatic_pressure(
            exner=entry.exner,
            virtual_temperature=entry.virtual_temperature,
            ddqz_z_full=self._ddqz_z_full,
            pressure=entry.pressure,
            pressure_ifc_on_model_levels=self._pressure_ifc_on_model_levels,
            pressure_ifc=entry.pressure_ifc,
        )
        # 3. cell-centre (u, v) from the edge-normal vn
        self._rbf_interpolation(p_e_in=entry.vn, p_u_out=entry.u, p_v_out=entry.v)
        return entry

    def _zero_accumulators(self) -> None:
        for accumulator in self.accumulators.values():
            _array(accumulator)[...] = 0.0

    def _accumulate(self, output: fw.State) -> None:
        """
        Add every `Tendency` leaf of a process output to the sum of its name.

        Element-wise with no neighbour access, so a plain array operation rather than a stencil.
        """
        for name, field in fw.fields(output).items():
            if not issubclass(field.quantity, fw.Tendency):
                continue
            if (accumulator := self.accumulators.get(name)) is None:
                accumulator = self.accumulators[name] = fw.zeros(
                    field.quantity, self.grid, self.allocator
                )
            _array(accumulator)[...] += field.ndarray

    @staticmethod
    def _continue_in_place(entry: physics_state.EntryState, out: Output) -> None:
        """`out` continues the entry's prognostics: they are updated in place."""
        for name, target in fw.fields(out).items():
            source: fw.Field[Any] = getattr(entry, name)
            if target.ndarray is not source.ndarray:
                _array(target)[...] = source.ndarray

    def _apply(self, entry: physics_state.EntryState, dt_seconds: float, out: Output) -> None:
        """
        The summed tendencies into `out`, once (ICON's phy2dyn).

        The order matters: the tracers first, because the exner/theta_v update uses the final
        moisture, then the temperature, then the winds. A tendency no process produced is not
        applied.
        """
        self._continue_in_place(entry, out)
        acc = self.accumulators

        # 1. tracers: q += dt * sum of the tendencies (mo_interface_iconam_aes:513)
        for name in states.TRACERS:
            if (tendency := acc.get(f"tend_{name}")) is not None:
                tracer = _present(getattr(out, name))
                self._apply_tendency(
                    field_a=tracer, coeff=dt_seconds, field_b=tendency, output_field=tracer
                )

        # 2. temperature -> exner/theta_v: one exact-EOS update from the entry temperature plus
        #    the summed tendency, with the final (post step 1) moisture
        if "tend_temperature" in acc:
            self._apply_tendency(
                field_a=entry.temperature,
                coeff=dt_seconds,
                field_b=acc["tend_temperature"],
                output_field=self._new_temperature,
            )
            self._compute_virtual_temperature_tendency(
                dtime=dt_seconds,
                qv=_present(out.qv),
                qc=_present(out.qc),
                qi=_present(out.qi),
                qr=_present(out.qr),
                qs=_present(out.qs),
                qg=_present(out.qg),
                temperature=self._new_temperature,
                virtual_temperature=entry.virtual_temperature,
                virtual_temperature_tendency=self._virtual_temperature_tendency,
            )
            self._update_exner_and_theta_v(
                rho=entry.rho,
                virtual_temperature=entry.virtual_temperature,
                virtual_temperature_tendency=self._virtual_temperature_tendency,
                dtime=dt_seconds,
                exner=out.exner,
                theta_v=out.theta_v,
            )

        # 3. winds: one projection of the summed (u, v) tendencies onto the edge normals
        wind_tendencies = {"tend_u", "tend_v"} & acc.keys()
        if len(wind_tendencies) == 1:
            (declared,) = wind_tendencies
            raise ValueError(
                f"the horizontal wind tendencies are applied as a pair; got only {declared}"
            )
        if wind_tendencies:
            self._compute_vn_from_uv(u=acc["tend_u"], v=acc["tend_v"], vn=self._vn_tendency)
            self._apply_tendency_vn(
                field_a=out.vn,
                coeff=dt_seconds,
                field_b=self._vn_tendency,
                output_field=out.vn,
            )

        # 4. w, on half levels
        if "tend_w" in acc:
            self._apply_tendency_w(
                field_a=out.w,
                coeff=dt_seconds,
                field_b=acc["tend_w"],
                output_field=out.w,
            )
