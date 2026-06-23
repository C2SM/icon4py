# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.diagnostic_calculations.stencils import (
    calculate_tendency,
    diagnose_pressure,
    diagnose_surface_pressure,
    diagnose_temperature,
)
from icon4py.model.common.math.stencils import generic_math_operations
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.common.states import factory, prognostic_state as prognostics, tracer_state


#: muphys species keys, in the order of the muphys ``Q`` tuple.
_SPECIES = ("v", "c", "r", "s", "i", "g")


class State:
    """The muphys physics State adapter.

    Bridges the dycore's prognostic state and the muphys Component contract.
    Two independent axes describe each field:

    muphys role
      - input      : fed to muphys via ``as_component_input`` -- dz, rho, q, te, p
      - returned   : muphys updates it; te/q changes come back as tendencies
                     (tend_T -> exner, tend_q -> tracers).
                     rho and p are input-only.
      - internal   : not a muphys fields -- tv, pressure_ifc -- used only
                     to diagnose p and to convert tend_T into an exner increment.
      - diagnostic : a muphys output stored for reporting -- pflx, pr, ps, pi, pg, pre.

    memory ownership
      - reference  : a pointer into the dycore state, no copy -- dz, rho, q.
      - owned      : a buffer allocated once here and overwritten in place each
                     step -- te, p, tv, pressure_ifc, and the scratch buffers.
    """

    def __init__(
        self,
        grid: base_grid.Grid,
        metrics: factory.FieldSource,
        backend: gtx_typing.Backend | None = None,
    ) -> None:
        self._num_cells = grid.num_cells
        self._num_levels = grid.num_levels
        self._backend = backend

        self._diagnose_temperature_program = (
            diagnose_temperature.diagnose_virtual_temperature_and_temperature.with_backend(
                self._backend
            )
        )
        self._diagnose_surface_pressure_program = (
            diagnose_surface_pressure.diagnose_surface_pressure.with_backend(self._backend)
        )
        self._diagnose_pressure_program = diagnose_pressure.diagnose_pressure.with_backend(
            self._backend
        )
        self._apply_tendency_program = (
            generic_math_operations.compute_field_a_plus_coeff_times_field_b_on_cell_k.with_backend(
                self._backend
            )
        )
        self._virtual_temperature_tendency_program = (
            calculate_tendency.calculate_virtual_temperature_tendency.with_backend(self._backend)
        )
        self._exner_tendency_program = calculate_tendency.calculate_exner_tendency.with_backend(
            self._backend
        )

        self.dz = metrics.get(metrics_attributes.DDQZ_Z_FULL)
        self.rho: fa.CellKField[ta.wpfloat] | None = None
        self.q: dict[str, fa.CellKField[ta.wpfloat]] = {}
        self.te = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=backend
        )  # temperature
        self.p = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)  # pressure
        self.tv = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self.pressure_ifc = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=backend
        )

        # INTERNAL
        self._new_te = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self._tv_tendency = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self._exner_tendency = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=backend
        )

        self.pflx: fa.CellKField[ta.wpfloat] | None = None  # total precipitation flux
        self.pr: fa.CellKField[ta.wpfloat] | None = None  # surface rain rate
        self.ps: fa.CellKField[ta.wpfloat] | None = None  # surface snow rate
        self.pi: fa.CellKField[ta.wpfloat] | None = None  # surface ice rate
        self.pg: fa.CellKField[ta.wpfloat] | None = None  # surface graupel rate
        self.pre: fa.CellKField[ta.wpfloat] | None = None  # surface precip energy flux

    def gather_from_prognostic(
        self, prognostic: prognostics.PrognosticState, tracers: tracer_state.TracerState
    ) -> None:
        """
        prepare the input fields for muphys from the prognostic state. This includes:
            - binding the references for rho and q (the muphys input fields that are stored prognostically in the dycore state)
            - diagnosing the muphys input fields that aren't stored prognostically (te, p) from the prognostic state.
        """
        self.rho = prognostic.rho
        self.q = {
            "v": tracers.qv,
            "c": tracers.qc,
            "r": tracers.qr,
            "s": tracers.qs,
            "i": tracers.qi,
            "g": tracers.qg,
        }

        # Diagnose virtual temperature and temperature (te is not stored prognostically).
        self._diagnose_temperature_program(
            qv=self.q["v"],
            qc=self.q["c"],
            qi=self.q["i"],
            qr=self.q["r"],
            qs=self.q["s"],
            qg=self.q["g"],
            theta_v=prognostic.theta_v,
            exner=prognostic.exner,
            virtual_temperature=self.tv,
            temperature=self.te,
            horizontal_start=0,
            horizontal_end=self._num_cells,
            vertical_start=0,
            vertical_end=self._num_levels,
            offset_provider={},
        )

        # Diagnose surface pressure into the surface slot of pressure_ifc, ...
        self._diagnose_surface_pressure_program(
            exner=prognostic.exner,
            virtual_temperature=self.tv,
            ddqz_z_full=self.dz,
            surface_pressure=self.pressure_ifc,
            horizontal_start=0,
            horizontal_end=self._num_cells,
            vertical_start=self._num_levels,
            vertical_end=self._num_levels + 1,
            offset_provider={"Koff": dims.KDim},  # type: ignore[dict-item]
        )
        # diagnose full-level pressure p
        surface_pressure = gtx.as_field(
            (dims.CellDim,), self.pressure_ifc.ndarray[:, -1], allocator=self._backend
        )
        self._diagnose_pressure_program(
            ddqz_z_full=self.dz,
            virtual_temperature=self.tv,
            surface_pressure=surface_pressure,
            pressure=self.p,
            pressure_ifc=self.pressure_ifc,
            horizontal_start=0,
            horizontal_end=self._num_cells,
            vertical_start=0,
            vertical_end=self._num_levels,
            offset_provider={},
        )

    def scatter_to_prognostic(
        self,
        prognostic: prognostics.PrognosticState,
        outputs: dict[str, fa.CellKField[ta.wpfloat]],
        dt: float,
    ) -> None:
        """Outbound translation: apply muphys output (tendencies) back to the prognostic state.

        This will be called after calling the muphys.
        output is got from muphys, and the tendencies in output will be applied to the prognostic state.
        """
        # 1. Apply moisture tendencies to the tracers (in place; self.q was bound in gather).
        for s in _SPECIES:
            tracer = self.q[s]
            self._apply_tendency_program(
                field_a=tracer,
                coeff=dt,
                field_b=outputs[f"tend_q{s}"],
                output_field=tracer,
                offset_provider={},
                horizontal_start=0,
                horizontal_end=self._num_cells,
                vertical_start=0,
                vertical_end=self._num_levels,
            )

        # 2. tend_T -> exner. new_te = te + tend_T*dt
        self._apply_tendency_program(
            field_a=self.te,
            coeff=dt,
            field_b=outputs["tend_temperature"],
            output_field=self._new_te,
            offset_provider={},
            horizontal_start=0,
            horizontal_end=self._num_cells,
            vertical_start=0,
            vertical_end=self._num_levels,
        )

        # dTv/dt from the new temperature and the species just updated in step 1
        self._virtual_temperature_tendency_program(
            dtime=dt,
            qv=self.q["v"],
            qc=self.q["c"],
            qi=self.q["i"],
            qr=self.q["r"],
            qs=self.q["s"],
            qg=self.q["g"],
            temperature=self._new_te,
            virtual_temperature=self.tv,
            virtual_temperature_tendency=self._tv_tendency,
            offset_provider={},
            horizontal_start=0,
            horizontal_end=self._num_cells,
            vertical_start=0,
            vertical_end=self._num_levels,
        )
        # d(exner)/dt from dTv/dt, then exner += d(exner)/dt * dt.
        self._exner_tendency_program(
            dtime=dt,
            virtual_temperature=self.tv,
            virtual_temperature_tendency=self._tv_tendency,
            exner=prognostic.exner,
            exner_tendency=self._exner_tendency,
            offset_provider={},
            horizontal_start=0,
            horizontal_end=self._num_cells,
            vertical_start=0,
            vertical_end=self._num_levels,
        )
        self._apply_tendency_program(
            field_a=prognostic.exner,
            coeff=dt,
            field_b=self._exner_tendency,
            output_field=prognostic.exner,
            offset_provider={},
            horizontal_start=0,
            horizontal_end=self._num_cells,
            vertical_start=0,
            vertical_end=self._num_levels,
        )

        # 3. Store precip diagnostics (references; never applied to prognostic state).
        self.pflx = outputs["pflx"]
        self.pr = outputs["pr"]
        self.ps = outputs["ps"]
        self.pi = outputs["pi"]
        self.pg = outputs["pg"]
        self.pre = outputs["pre"]

    def as_component_input(self) -> dict[str, fa.CellKField[ta.wpfloat]]:
        """
        Translate to the generic Component input dict (the 10 muphys input fields).
        """
        if self.rho is None:
            raise RuntimeError("as_component_input called before gather_from_prognostic")
        inp = {"dz": self.dz, "te": self.te, "p": self.p, "rho": self.rho}
        inp.update({f"q{s}": self.q[s] for s in _SPECIES})
        return inp
