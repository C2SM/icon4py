# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import (
    physics_process,
    physics_state,
    static_fields,
)
from icon4py.model.common import (
    dimension as dims,
    field,
    framework as fw,
    model_backends,
    model_options,
    quantity,
    time,
)
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
    from icon4py.model.common.decomposition import definitions as decomposition
    from icon4py.model.common.grid import base as base_grid


class PhysicsDriver(fw.Component["PhysicsDriver.Input", "PhysicsDriver.Output"]):
    """
    Runs the physics processes under ICON AES's sequential coupling (aes_phy_main).

    In one step the physics state is diagnosed once from the prognostics, then the processes
    run in list order. Each process's tendencies are summed over the processes and added to
    the physics state in place before the next one runs, so each process sees the state the
    previous ones advanced. At the end exner, theta_v, the perturbed exner (when given) and vn
    are updated once from the advanced state and the summed wind tendencies (phy2dyn), and the
    halos of what the physics wrote are synced.
    """

    class Input(fw.State):
        vn: field.Field[qty.VnOnEdgeK]
        w: field.Field[qty.WOnCellKHalf]
        exner: field.Field[qty.ExnerOnCellK]
        theta_v: field.Field[qty.ThetaVOnCellK]
        rho: field.Field[qty.RhoOnCellK]
        qv: field.Field[qty.QvOnCellK] | None = None
        qc: field.Field[qty.QcOnCellK] | None = None
        qi: field.Field[qty.QiOnCellK] | None = None
        qr: field.Field[qty.QrOnCellK] | None = None
        qs: field.Field[qty.QsOnCellK] | None = None
        qg: field.Field[qty.QgOnCellK] | None = None
        air_mass: field.Field[qty.AirMassOnCellK] | None = None
        simulation_current_datetime: time.AbsoluteTime

    class Output(fw.State):
        vn: field.Field[qty.VnOnEdgeK]
        w: field.Field[qty.WOnCellKHalf]
        exner: field.Field[qty.ExnerOnCellK]
        theta_v: field.Field[qty.ThetaVOnCellK]
        qv: field.Field[qty.QvOnCellK] | None = None
        qc: field.Field[qty.QcOnCellK] | None = None
        qi: field.Field[qty.QiOnCellK] | None = None
        qr: field.Field[qty.QrOnCellK] | None = None
        qs: field.Field[qty.QsOnCellK] | None = None
        qg: field.Field[qty.QgOnCellK] | None = None
        temperature: field.Field[qty.TemperatureOnCellK]
        virtual_temperature: field.Field[qty.VirtualTemperatureOnCellK]
        pressure: field.Field[qty.PressureOnCellK]
        pressure_ifc: field.Field[qty.PressureOnCellKHalf]
        u: field.Field[qty.UOnCellK]
        v: field.Field[qty.VOnCellK]
        perturbed_exner: field.Field[qty.PerturbedExnerOnCellK] | None = None

    def __init__(
        self,
        processes: Sequence[physics_process.PhysicsProcess],
        *,
        dtime: time.RelativeTime,
        grid: base_grid.Grid,
        static_state: static_fields.PhysicsDriverStaticState,
        exchange: decomposition.ExchangeRuntime,
        backend: model_backends.BackendLike = None,
    ) -> None:
        if mismatched := [
            p.time_control.driver_dtime for p in processes if p.time_control.driver_dtime != dtime
        ]:
            raise ValueError(
                f"the processes' time controls must use the driver's dtime {dtime}, got {mismatched}"
            )
        self.dtime = dtime
        self._grid = grid
        self._exchange = exchange
        self._allocator = model_backends.get_allocator(backend)
        self._process_outputs: dict[physics_process.PhysicsProcess, fw.State] = {
            process: fw.allocate(process.Output, self._grid, self._allocator)
            for process in processes
        }
        # NOTE: temporary, replaced in part 02 (collect): goes with the physics state.
        # Dict of process outputs which are read by other processes through the
        # physics state, e.g. surface outputs read by tmx
        self._computed_by_processes: dict[type[quantity.Quantity], field.Field] = {}
        for process_output in self._process_outputs.values():
            for name, field_ in fw.fields(process_output).items():
                if field_.quantity in physics_state.COMPUTED_BY_PROCESSES:
                    if field_.quantity in self._computed_by_processes:
                        raise ValueError(f"'{name}' is computed by more than one physics process")
                    self._computed_by_processes[field_.quantity] = field_
        self._state_leaves = {
            quantity: name
            for name, quantity in fw.leaf_quantities(physics_state.PhysicsState).items()
        }
        # NOTE: temporary, replaced in parts 07 (tendencies) and 08 (updates), see
        # `_zero_accumulators`.
        # one sum per tendency the processes emit, by quantity, made when first seen
        self.accumulators: dict[type[quantity.Quantity], field.Field] = {}

        cell_range = {
            "horizontal_start": gtx.int32(0),
            "horizontal_end": gtx.int32(self._grid.num_cells),
        }
        edge_range = {
            "horizontal_start": gtx.int32(0),
            "horizontal_end": gtx.int32(self._grid.num_edges),
        }
        k_range = {"vertical_start": gtx.int32(0), "vertical_end": gtx.int32(self._grid.num_levels)}
        khalf_range = {
            "vertical_start": gtx.int32(0),
            "vertical_end": gtx.int32(self._grid.num_levels + 1),
        }

        self._ddqz_z_full = static_state.ddqz_z_full
        self._diagnose_temperature = model_options.setup_program(
            program=compute_temperature.compute_virtual_temperature_and_temperature,
            backend=backend,
            horizontal_sizes=cell_range,
            vertical_sizes=k_range,
            offset_provider={},
        )
        self._compute_surface_and_hydrostatic_pressure = model_options.setup_program(
            program=compute_pressure.compute_surface_and_hydrostatic_pressure,
            backend=backend,
            horizontal_sizes=cell_range,
            vertical_sizes=k_range,
            offset_provider={},
        )
        self._rbf_interpolation = model_options.setup_program(
            program=edge_2_cell_vector_rbf_interpolation,
            backend=backend,
            constant_args={
                "ptr_coeff_1": static_state.rbf_vec_coeff_c1,
                "ptr_coeff_2": static_state.rbf_vec_coeff_c2,
            },
            horizontal_sizes=cell_range,
            vertical_sizes=k_range,
            offset_provider=self._grid.connectivities,
        )
        self._apply_tendency = model_options.setup_program(
            program=generic_math_operations.compute_field_a_plus_coeff_times_field_b_on_cell_k,
            backend=backend,
            horizontal_sizes=cell_range,
            vertical_sizes=k_range,
            offset_provider={},
        )
        self._apply_tendency_w = model_options.setup_program(
            program=generic_math_operations.compute_field_a_plus_coeff_times_field_b_on_cell_khalf,
            backend=backend,
            horizontal_sizes=cell_range,
            vertical_sizes=khalf_range,
            offset_provider={},
        )
        self._apply_tendency_vn = model_options.setup_program(
            program=generic_math_operations.compute_field_a_plus_coeff_times_field_b_on_edge_k,
            backend=backend,
            horizontal_sizes=edge_range,
            vertical_sizes=k_range,
            offset_provider={},
        )
        self._compute_virtual_temperature_tendency = model_options.setup_program(
            program=compute_tendencies.compute_virtual_temperature_tendency,
            backend=backend,
            horizontal_sizes=cell_range,
            vertical_sizes=k_range,
            offset_provider={},
        )
        self._update_exner_and_theta_v = model_options.setup_program(
            program=compute_temperature.update_exner_and_theta_v,
            backend=backend,
            horizontal_sizes=cell_range,
            vertical_sizes=k_range,
            offset_provider={},
        )
        self._update_exner_theta_v_and_perturbed_exner = model_options.setup_program(
            program=compute_temperature.update_exner_theta_v_and_perturbed_exner,
            backend=backend,
            horizontal_sizes=cell_range,
            vertical_sizes=k_range,
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
            horizontal_sizes=edge_range,
            vertical_sizes=k_range,
            offset_provider=self._grid.connectivities,
        )

        # A scan's range is deduced from its single output domain, so the half-level pressure
        # lands on model levels first and compute_surface_and_hydrostatic_pressure copies it up.
        self._pressure_ifc_on_model_levels = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=self._allocator
        )
        self._virtual_temperature_tendency = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=self._allocator
        )
        self._vn_tendency = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, allocator=self._allocator
        )

    def run(self, input: Input, output: Output) -> Output:
        dt_seconds = self.dtime.total_seconds()
        fw.copy_to_output(input, output)
        # NOTE: temporary, replaced in parts 04 (recipes) and 05 (derived inputs), see `_diagnose`.
        state = self._diagnose(input, output)
        # NOTE: temporary, replaced in parts 07 (tendencies) and 08 (updates)
        self._zero_accumulators()
        for process, process_output in self._process_outputs.items():
            process.run(state, process_output)
            self._accumulate_tendencies(process_output)
            self._update_state(state, process_output, dt_seconds)
        self._update_prognostics(state, dt_seconds, output)
        return output

    def diagnostic_fields(self) -> dict[str, field.Field]:
        """
        The processes' diagnostic (non-tendency) output fields, by the standard name of their
        quantity; those without a standard name are not offered.
        """
        fields: dict[str, field.Field] = {}
        for process_output in self._process_outputs.values():
            # NOTE: temporary, replaced in part 07 (tendencies): a tendency is a `TendencyOf`
            # instead of a subclass of the marker base.
            for field_ in fw.fields(process_output, exclude=quantity.Tendency).values():
                name = field_.quantity.standard_name
                if name is None:
                    continue
                if name in fields:
                    raise ValueError(f"'{name}' is an output of more than one physics process")
                fields[name] = field_
        return fields

    @staticmethod
    def _require_tracer[Q: quantity.Quantity](field: field.Field[Q] | None) -> field.Field[Q]:
        """A tracer leaf the physics requires."""
        if field is None:
            raise ValueError("physics requires all moisture species active in the TracerState")
        return field

    # NOTE: temporary, replaced in parts 04 (recipes) and 05 (derived inputs): the
    # diagnostics become recipes, which the processes define on their `Input`, and `_diagnose` goes.
    def _diagnose(self, input: Input, output: Output) -> physics_state.PhysicsState:
        """
        The physics state and the diagnostics derived from `input` into `output` (dyn2phy).

        The fields the processes advance (w, the tracers and the diagnostics) are the ones of
        `output`, which continues `input`; the others are read only and stay the ones of `input`.

        At the end the halos of rho, u, v and w are synced for the horizontal operations of
        the physics, as ICON's interface does before tmx (mo_interface_iconam_aes): diffusion
        skips the cell sync of w under AES forcing.
        """
        # The tracers are optional leaves (a tracer may be inactive in the TracerConfig): fail
        # here rather than feed None to the physics.
        if missing := [name for name in states.TRACERS if getattr(output, name) is None]:
            raise ValueError(
                f"physics requires all moisture species active in the TracerState; missing: {missing}"
            )
        state = physics_state.PhysicsState(
            vn=input.vn,
            w=output.w,
            exner=input.exner,
            theta_v=input.theta_v,
            rho=input.rho,
            qv=self._require_tracer(output.qv),
            qc=self._require_tracer(output.qc),
            qi=self._require_tracer(output.qi),
            qr=self._require_tracer(output.qr),
            qs=self._require_tracer(output.qs),
            qg=self._require_tracer(output.qg),
            temperature=output.temperature,
            virtual_temperature=output.virtual_temperature,
            pressure=output.pressure,
            pressure_ifc=output.pressure_ifc,
            u=output.u,
            v=output.v,
            air_mass=input.air_mass,
            **{self._state_leaves[q]: f for q, f in self._computed_by_processes.items()},
            step_start=input.simulation_current_datetime - self.dtime,
        )
        # 1. virtual temperature and temperature
        self._diagnose_temperature(
            qv=state.qv,
            qc=state.qc,
            qi=state.qi,
            qr=state.qr,
            qs=state.qs,
            qg=state.qg,
            theta_v=state.theta_v,
            exner=state.exner,
            virtual_temperature=state.virtual_temperature,
            temperature=state.temperature,
        )
        # 2. surface pressure at the bottom interface, then the full pressure column
        self._compute_surface_and_hydrostatic_pressure(
            exner=state.exner,
            virtual_temperature=state.virtual_temperature,
            ddqz_z_full=self._ddqz_z_full,
            pressure=state.pressure,
            pressure_ifc_on_model_levels=self._pressure_ifc_on_model_levels,
            pressure_ifc=state.pressure_ifc,
        )
        # 3. cell-centre (u, v) from the edge-normal vn
        self._rbf_interpolation(p_e_in=state.vn, p_u_out=state.u, p_v_out=state.v)
        # 4. halo sync for the horizontal operations of the physics: tmx interpolates w to the
        #    vertices and edges of its owned cells, and diffusion left the halo of w stale.
        # TODO(yiluchen1066): move this exchange to driver.py: it writes the halo of
        # `input.rho`, which the physics only reads. It also duplicates work: tmx syncs u and
        # v itself and the dycore syncs rho at the end of the step; only w is stale here.
        self._exchange.exchange(dims.CellDim, state.rho, state.u, state.v, state.w)
        return state

    # NOTE: temporary, replaced in parts 07 (tendencies) and 08 (updates): 07 keys the
    # accumulators by the tendency's parent quantity instead of the tendency's; 08 replaces
    # `_zero_accumulators`, `_accumulate`, `_update` and `_update_prognostics` with the updates
    # each process declares.
    def _zero_accumulators(self) -> None:
        for accumulator in self.accumulators.values():
            accumulator.ndarray[...] = 0.0  # type: ignore[index]

    def _accumulate_tendencies(self, output: fw.State) -> None:
        """
        Add every `Tendency` leaf of a process output to the sum of its quantity.

        Element-wise with no neighbour access, so a plain array operation rather than a stencil.
        """
        for field_ in fw.fields(output, kind=quantity.Tendency).values():
            if (accumulator := self.accumulators.get(field_.quantity)) is None:
                accumulator = self.accumulators[field_.quantity] = fw.zeros(
                    field_.quantity, self._grid, self._allocator
                )
            accumulator.ndarray[...] += field_.ndarray  # type: ignore[index]

    def _update_state(
        self, state: physics_state.PhysicsState, output: fw.State, dt_seconds: float
    ) -> None:
        """
        Add a process's tendencies to the physics state in place: field += dt * tendency.

        The next process reads the advanced temperature, u, v, w and tracers (ICON AES
        `fc_xxx = 1`); the virtual temperature and the pressure stay at step start. Each
        tendency must be the tendency of one of these fields.
        """
        targets: dict[type[quantity.Quantity], field.Field] = {
            qty.TendencyOfTemperatureOnCellK: state.temperature,
            qty.TendencyOfUOnCellK: state.u,
            qty.TendencyOfVOnCellK: state.v,
            qty.TendencyOfWOnCellKHalf: state.w,
            qty.TendencyOfQvOnCellK: state.qv,
            qty.TendencyOfQcOnCellK: state.qc,
            qty.TendencyOfQiOnCellK: state.qi,
            qty.TendencyOfQrOnCellK: state.qr,
            qty.TendencyOfQsOnCellK: state.qs,
            qty.TendencyOfQgOnCellK: state.qg,
        }
        for name, tendency in fw.fields(output, kind=quantity.Tendency).items():
            if (target := targets.get(tendency.quantity)) is None:
                raise ValueError(
                    f"tendency output '{name}' is not the tendency of an advanced field; "
                    f"expected one of {[q.__name__ for q in targets]}"
                )
            apply = (
                self._apply_tendency_w
                if tendency.quantity is qty.TendencyOfWOnCellKHalf
                else self._apply_tendency
            )
            apply(field_a=target, coeff=dt_seconds, field_b=tendency, output_field=target)

    def _update_prognostics(
        self, state: physics_state.PhysicsState, dt_seconds: float, output: Output
    ) -> None:
        """
        Exner, theta_v, the perturbed exner and vn into `output`, once (phy2dyn).

        The tracers, temperature, u, v and w were already advanced in place; exner and theta_v
        follow from the final temperature and moisture, and vn from the summed wind tendencies.

        The exit halo exchanges of the interface happen here, as in ICON's
        mo_interface_iconam_aes: the summed wind tendencies before their projection, since an
        owned edge reads the sums of a halo cell and the sync of vn only repairs halo edges, and
        at the end what the physics wrote.

        TODO(yiluchen1066): ICON clips negative tracers at the entry and the exit of the
        interface (`iqneg_d2p` / `iqneg_p2d`: set to zero and move the deficit over dt into the
        tendency; default off, the APE experiment sets both to 2). Port it with the two config
        members before the exner update here and in `_diagnose`; the APE archive's config has to
        be regenerated for the switches.
        """
        acc = self.accumulators
        self._compute_virtual_temperature_tendency(
            dtime=dt_seconds,
            qv=state.qv,
            qc=state.qc,
            qi=state.qi,
            qr=state.qr,
            qs=state.qs,
            qg=state.qg,
            temperature=state.temperature,
            virtual_temperature=state.virtual_temperature,
            virtual_temperature_tendency=self._virtual_temperature_tendency,
        )
        if output.perturbed_exner is None:
            self._update_exner_and_theta_v(
                rho=state.rho,
                virtual_temperature=state.virtual_temperature,
                virtual_temperature_tendency=self._virtual_temperature_tendency,
                dtime=dt_seconds,
                exner=output.exner,
                theta_v=output.theta_v,
            )
        else:
            self._update_exner_theta_v_and_perturbed_exner(
                rho=state.rho,
                virtual_temperature=state.virtual_temperature,
                virtual_temperature_tendency=self._virtual_temperature_tendency,
                dtime=dt_seconds,
                exner=output.exner,
                theta_v=output.theta_v,
                perturbed_exner=output.perturbed_exner,
            )

        # one projection of the summed (u, v) tendencies onto vn, when a process produced them
        wind_tendencies = {qty.TendencyOfUOnCellK, qty.TendencyOfVOnCellK} & acc.keys()
        if len(wind_tendencies) == 1:
            (declared,) = wind_tendencies
            raise ValueError(
                f"the horizontal wind tendencies are applied as a pair; got only {declared.__name__}"
            )
        if wind_tendencies:
            u, v = acc[qty.TendencyOfUOnCellK], acc[qty.TendencyOfVOnCellK]
            self._exchange.exchange(dims.CellDim, u, v)
            self._compute_vn_from_uv(u=u, v=v, vn=self._vn_tendency)
            self._apply_tendency_vn(
                field_a=output.vn,
                coeff=dt_seconds,
                field_b=self._vn_tendency,
                output_field=output.vn,
            )

        written_cells: list[field.Field] = [getattr(state, name) for name in states.TRACERS]
        written_cells += [output.w, output.exner, output.theta_v]
        if output.perturbed_exner is not None:
            written_cells.append(output.perturbed_exner)
        self._exchange.exchange(dims.CellDim, *written_cells)
        if wind_tendencies:
            self._exchange.exchange(dims.EdgeDim, output.vn)
