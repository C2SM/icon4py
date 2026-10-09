# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``PhysicsDriver`` and its processes."""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.process_time_control import (
    ProcessTimeControl,
)
from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    model_options,
    type_alias as ta,
)
from icon4py.model.common.components import framework as fw, quantities as qty, states
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
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.common.states import factory


__all__ = ["PhysicsDriver", "PhysicsProcess", "ProcessTimeControl", "Step", "bind"]

type Step = Callable[[physics_state.PhysicsState], fw.State]


def bind[I: fw.State, O: fw.State](
    run: Callable[[I], O], collect_input: Callable[[physics_state.PhysicsState], I]
) -> Callable[[physics_state.PhysicsState], O]:
    """
    A process step: a component's `run` on what its `collect_input` picks from the physics state.

    Checks once that the two belong together (another component's `collect_input` is a type
    error); the driver then sees the step only as a `Step`.
    """
    return lambda state: run(collect_input(state))


@dataclasses.dataclass(frozen=True)
class PhysicsProcess:
    """One physics process: its step and its time control, which belongs to the composer."""

    name: str
    step: Step
    time_control: ProcessTimeControl


def _present[Q: fw.Quantity](field: fw.Field[Q] | None) -> fw.Field[Q]:
    """A tracer leaf the physics requires."""
    if field is None:
        raise ValueError("physics requires all moisture species active in the TracerState")
    return field


def _array(field: fw.Field[Any]) -> Any:
    # NDArrayObject declares no __setitem__
    return field.data.ndarray


class _Advanced(fw.State):
    """The leaves of the physics state the driver advances, in its own buffers."""

    temperature: fw.Field[qty.TemperatureOnCellK]
    qv: fw.Field[qty.QvOnCellK]
    qc: fw.Field[qty.QcOnCellK]
    qi: fw.Field[qty.QiOnCellK]
    qr: fw.Field[qty.QrOnCellK]
    qs: fw.Field[qty.QsOnCellK]
    qg: fw.Field[qty.QgOnCellK]
    u: fw.Field[qty.UOnCellK]
    v: fw.Field[qty.VOnCellK]
    w: fw.Field[qty.WOnCellKHalf]


class PhysicsDriver(fw.Component):
    """
    Runs the physics processes coupled sequentially, as ICON AES (aes_phy_main, fc = 1).

    In one step the physics state is diagnosed once from the prognostics (dyn2phy) and the
    processes run on it in order. After each process, also when its last output is reused,
    its tendencies are added to their sums and advance the physics state's temperature,
    tracers, u, v and w by dt times them: the next process reads that provisional update. At
    the end the sums are applied once to the entry values, the prognostics and the diagnostics
    derived from them (phy2dyn). The processes never write the prognostics.

    In place like the diffusion: the composer passes the same buffers as input and output, `run`
    copies the input into `out` first where they differ. `rho` is read only.
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
        processes: Sequence[PhysicsProcess],
        *,
        grid: base_grid.Grid,
        ddqz_z_full: fa.CellKField[ta.wpfloat],
        rbf_vec_coeff_c1: gtx.Field[Any, ta.wpfloat],
        rbf_vec_coeff_c2: gtx.Field[Any, ta.wpfloat],
        primal_normal_cell_x: gtx.Field[Any, ta.wpfloat],
        primal_normal_cell_y: gtx.Field[Any, ta.wpfloat],
        c_lin_e: gtx.Field[Any, ta.wpfloat],
        backend: model_backends.BackendLike = None,
    ) -> None:
        super().__init__(grid, model_backends.get_allocator(backend))
        self.processes = tuple(processes)
        # the diagnostics derived here each step, the entry values of the final update
        self.diagnostics = fw.allocate(states.Diagnostics, grid, self.allocator)
        # the leaves of the physics state advanced after each process, reset to the entry
        # values each step
        self._advanced = fw.allocate(_Advanced, grid, self.allocator)
        # each of them by the name of the tendency that advances it
        self._advanced_by_tendency = {
            f"tend_{declaration.name}": field for declaration, field in self._advanced.leaves()
        }
        # one sum per tendency the processes emit, by output name, made when first seen
        self.accumulators: dict[str, fw.Field[Any]] = {}
        # the last output of each process, reused on the steps it is not active
        self.outputs: dict[str, fw.State] = {}

        cells = {"horizontal_start": gtx.int32(0), "horizontal_end": gtx.int32(grid.num_cells)}
        edges = {"horizontal_start": gtx.int32(0), "horizontal_end": gtx.int32(grid.num_edges)}
        levels = {"vertical_start": gtx.int32(0), "vertical_end": gtx.int32(grid.num_levels)}
        half_levels = {
            "vertical_start": gtx.int32(0),
            "vertical_end": gtx.int32(grid.num_levels + 1),
        }

        self._ddqz_z_full = ddqz_z_full
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
            constant_args={"ptr_coeff_1": rbf_vec_coeff_c1, "ptr_coeff_2": rbf_vec_coeff_c2},
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
                "primal_normal_cell_x": primal_normal_cell_x,
                "primal_normal_cell_y": primal_normal_cell_y,
                "c_lin_e": c_lin_e,
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

    @classmethod
    def from_sources(
        cls,
        processes: Sequence[PhysicsProcess],
        *,
        grid: base_grid.Grid,
        geometry: factory.FieldSource,
        interpolation: factory.FieldSource,
        metrics: factory.FieldSource,
        backend: model_backends.BackendLike = None,
    ) -> PhysicsDriver:
        """The driver with its static fields read from the factories."""
        return cls(
            processes,
            grid=grid,
            ddqz_z_full=metrics.get(metrics_attributes.DDQZ_Z_FULL),
            rbf_vec_coeff_c1=interpolation.get(interpolation_attributes.RBF_VEC_COEFF_C1),
            rbf_vec_coeff_c2=interpolation.get(interpolation_attributes.RBF_VEC_COEFF_C2),
            primal_normal_cell_x=geometry.get(geometry_attributes.EDGE_NORMAL_CELL_U),
            primal_normal_cell_y=geometry.get(geometry_attributes.EDGE_NORMAL_CELL_V),
            c_lin_e=interpolation.get(interpolation_attributes.C_LIN_E),
            backend=backend,
        )

    def run(self, inputs: Input, out: Output | None = None) -> Output:
        out = self.buffers(out)
        state = self._diagnose(inputs)
        self._zero_accumulators()
        dt_seconds = inputs.dtime.total_seconds()
        # the time control reads the start of the step being integrated
        step_start = inputs.simulation_current_datetime - inputs.dtime
        for process in self.processes:
            time_control = process.time_control
            time_control.validate_interval(inputs.dtime)
            # outside its window a process contributes nothing
            if not time_control.is_in_window(step_start):
                continue
            # a process computes when it fires, and on its first step in the window when there
            # is no output to reuse yet; otherwise its last output is used again
            if time_control.is_active(step_start) or process.name not in self.outputs:
                self.outputs[process.name] = process.step(state)
            self._accumulate_and_advance(self.outputs[process.name], dt_seconds)
        self._apply(inputs, dt_seconds, out)
        return out

    def _diagnose(self, inputs: Input) -> physics_state.PhysicsState:
        """
        The physics state at the start of the step (dyn2phy): the diagnostics derived from the
        inputs, the advanced leaves reset to the entry values.
        """
        # The tracers are optional leaves (a tracer may be inactive in the TracerConfig): fail
        # here rather than feed None to the physics.
        if missing := [name for name in states.TRACERS if getattr(inputs, name) is None]:
            raise ValueError(
                f"physics requires all moisture species active in the TracerState; missing: {missing}"
            )
        diagnostics = self.diagnostics
        # 1. virtual temperature and temperature
        self._diagnose_temperature(
            qv=_present(inputs.qv).data,
            qc=_present(inputs.qc).data,
            qi=_present(inputs.qi).data,
            qr=_present(inputs.qr).data,
            qs=_present(inputs.qs).data,
            qg=_present(inputs.qg).data,
            theta_v=inputs.theta_v.data,
            exner=inputs.exner.data,
            virtual_temperature=diagnostics.virtual_temperature.data,
            temperature=diagnostics.temperature.data,
        )
        # 2. surface pressure at the bottom interface, then the full pressure column
        self._compute_surface_and_hydrostatic_pressure(
            exner=inputs.exner.data,
            virtual_temperature=diagnostics.virtual_temperature.data,
            ddqz_z_full=self._ddqz_z_full,
            pressure=diagnostics.pressure.data,
            pressure_ifc_on_model_levels=self._pressure_ifc_on_model_levels,
            pressure_ifc=diagnostics.pressure_ifc.data,
        )
        # 3. cell-centre (u, v) from the edge-normal vn
        self._rbf_interpolation(
            p_e_in=inputs.vn.data, p_u_out=diagnostics.u.data, p_v_out=diagnostics.v.data
        )
        # 4. the advanced leaves start from the entry values
        entry: dict[str, fw.Field[Any]] = {
            "temperature": diagnostics.temperature,
            "u": diagnostics.u,
            "v": diagnostics.v,
            "w": inputs.w,
            **{name: _present(getattr(inputs, name)) for name in states.TRACERS},
        }
        advanced = self._advanced
        for declaration, buffer in advanced.leaves():
            _array(buffer)[...] = entry[declaration.name].data.ndarray
        return physics_state.PhysicsState(
            vn=inputs.vn,
            w=advanced.w,
            exner=inputs.exner,
            theta_v=inputs.theta_v,
            rho=inputs.rho,
            qv=advanced.qv,
            qc=advanced.qc,
            qi=advanced.qi,
            qr=advanced.qr,
            qs=advanced.qs,
            qg=advanced.qg,
            temperature=advanced.temperature,
            virtual_temperature=diagnostics.virtual_temperature,
            pressure=diagnostics.pressure,
            pressure_ifc=diagnostics.pressure_ifc,
            u=advanced.u,
            v=advanced.v,
        )

    def _zero_accumulators(self) -> None:
        for accumulator in self.accumulators.values():
            _array(accumulator)[...] = 0.0

    def _accumulate_and_advance(self, output: fw.State, dt_seconds: float) -> None:
        """
        Add every `Tendency` leaf `tend_<name>` of a process output to the sum of its name and
        advance the physics state's `<name>` by dt times it, for the next process.

        Element-wise with no neighbour access, so plain array operations rather than stencils.
        """
        targets = self._advanced_by_tendency
        for declaration, field in output.leaves():
            if not issubclass(declaration.quantity, fw.Tendency):
                continue
            if (target := targets.get(declaration.name)) is None:
                raise ValueError(
                    f"Tendency '{declaration.name}' is not one of those that advance the physics "
                    f"state: {', '.join(targets)}."
                )
            if (accumulator := self.accumulators.get(declaration.name)) is None:
                accumulator = self.accumulators[declaration.name] = fw.zeros(
                    declaration.quantity, self.grid, self.allocator
                )
            tendency = field.data.ndarray
            _array(accumulator)[...] += tendency
            _array(target)[...] += dt_seconds * tendency

    @staticmethod
    def _continue_in_place(inputs: Input, out: Output) -> None:
        """`out` continues the inputs: the prognostics are updated in place."""
        for declaration, target in out.leaves():
            source: fw.Field[Any] = getattr(inputs, declaration.name)
            if target.data is not source.data:
                _array(target)[...] = source.data.ndarray

    def _apply(self, inputs: Input, dt_seconds: float, out: Output) -> None:
        """
        The summed tendencies into `out`, once, from the entry values (ICON's phy2dyn).

        From the inputs and the diagnostics derived from them, not from the advanced physics
        state, which already holds the tendencies. The order matters: the tracers first,
        because the exner/theta_v update uses the final moisture, then the temperature, then
        the winds. A tendency no process produced is not applied.
        """
        self._continue_in_place(inputs, out)
        acc = self.accumulators
        diagnostics = self.diagnostics

        # 1. tracers: q += dt * sum of the tendencies (mo_interface_iconam_aes:513)
        for name in states.TRACERS:
            if (tendency := acc.get(f"tend_{name}")) is not None:
                tracer = _present(getattr(out, name)).data
                self._apply_tendency(
                    field_a=tracer, coeff=dt_seconds, field_b=tendency.data, output_field=tracer
                )

        # 2. temperature -> exner/theta_v: one exact-EOS update from the diagnosed (entry)
        #    temperature plus the summed tendency, with the final (post step 1) moisture
        if "tend_temperature" in acc:
            self._apply_tendency(
                field_a=diagnostics.temperature.data,
                coeff=dt_seconds,
                field_b=acc["tend_temperature"].data,
                output_field=self._new_temperature,
            )
            self._compute_virtual_temperature_tendency(
                dtime=dt_seconds,
                qv=_present(out.qv).data,
                qc=_present(out.qc).data,
                qi=_present(out.qi).data,
                qr=_present(out.qr).data,
                qs=_present(out.qs).data,
                qg=_present(out.qg).data,
                temperature=self._new_temperature,
                virtual_temperature=diagnostics.virtual_temperature.data,
                virtual_temperature_tendency=self._virtual_temperature_tendency,
            )
            self._update_exner_and_theta_v(
                rho=inputs.rho.data,
                virtual_temperature=diagnostics.virtual_temperature.data,
                virtual_temperature_tendency=self._virtual_temperature_tendency,
                dtime=dt_seconds,
                exner=out.exner.data,
                theta_v=out.theta_v.data,
            )

        # 3. winds: one projection of the summed (u, v) tendencies onto the edge normals
        wind_tendencies = {"tend_u", "tend_v"} & acc.keys()
        if len(wind_tendencies) == 1:
            (declared,) = wind_tendencies
            raise ValueError(
                f"the horizontal wind tendencies are applied as a pair; got only {declared}"
            )
        if wind_tendencies:
            self._compute_vn_from_uv(
                u=acc["tend_u"].data, v=acc["tend_v"].data, vn=self._vn_tendency
            )
            self._apply_tendency_vn(
                field_a=out.vn.data,
                coeff=dt_seconds,
                field_b=self._vn_tendency,
                output_field=out.vn.data,
            )

        # 4. w, on half levels
        if "tend_w" in acc:
            self._apply_tendency_w(
                field_a=out.w.data,
                coeff=dt_seconds,
                field_b=acc["tend_w"].data,
                output_field=out.w.data,
            )
