# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.diagnostic_calculations.stencils import (
    calculate_tendency,
    diagnose_pressure,
    diagnose_surface_pressure,
    diagnose_temperature,
)
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.common.states import factory, prognostic_state as prognostics


#: muphys species keys, in the order of the muphys ``Q`` tuple.
_SPECIES = ("v", "c", "r", "s", "i", "g")

#: STUB (scope-4): placeholder species -> tracer-index map.
# TODO (Yilu): this should be identical to the mapping in the warm_bubble_init_conditon branch
_STUB_TRACER_INDEX: dict[str, int] = {"v": 0, "c": 1, "r": 2, "s": 3, "i": 4, "g": 5}


@gtx.field_operator
def _add_scaled_tendency(
    field: fa.CellKField[ta.wpfloat],
    tendency: fa.CellKField[ta.wpfloat],
    dt: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    return field + tendency * dt


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_tendency(  # noqa: PLR0917  # stencil params referenced in domain specs must stay positional
    field: fa.CellKField[ta.wpfloat],
    tendency: fa.CellKField[ta.wpfloat],
    dt: ta.wpfloat,
    result: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """``result = field + tendency * dt`` (Lie-Trotter split-update).

    Pass ``result=field`` for an in-place update, or a distinct buffer to keep
    the original (e.g. computing the new temperature without clobbering ``te``).
    """
    _add_scaled_tendency(
        field,
        tendency,
        dt,
        out=result,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

# TODO (Yilu)
class MuphysState:
    """L1 physics state adapter

    Bridges the dycore's prognostic state and the muphys Component contract.
    Two independent axes describe each field:

    muphys role
      - input      : fed to muphys via ``as_component_input`` -- dz, rho, q, te, p
      - returned   : muphys updates it; te/q changes come back as tendencies
                     (tend_T -> exner, tend_q -> tracers). NOTE rho and p are
                     input-only: muphys returns no updated density or pressure.
      - internal   : not a muphys field at all -- tv, pressure_ifc -- used only
                     to diagnose p and to convert tend_T into an exner increment.
      - diagnostic : a muphys output stored for reporting, never applied to the
                     prognostic state -- pflx, pr, ps, pi, pg, pre.

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
        # muphys INPUT, reference: layer thickness pulled once from the metrics
        # field source (ddqz_z_full is static, so fetch-at-construction is enough).
        self.dz = metrics.get(metrics_attributes.DDQZ_Z_FULL)

        # muphys INPUTS, owned: diagnosed each step
        self.te = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=backend
        )  # temperature
        self.p = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)  # pressure

        # INTERNAL. tv feeds the pressure diagnosis and the
        # tend_T -> exner conversion; pressure_ifc is half-level pressure whose extra
        # surface slot (index num_levels) holds surface pressure -- the
        # DiagnosticState.surface_pressure idiom.
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

        # muphys INPUTS, references into the dycore state, bound each step by
        # gather_from_prognostic (None until then). rho is input-only; the q
        # updates return via tend_q in scatter_to_prognostic.
        self.rho: fa.CellKField[ta.wpfloat] | None = None
        self.q: dict[str, fa.CellKField[ta.wpfloat]] = {}

        # muphys DIAGNOSTIC outputs
        self.pflx: fa.CellKField[ta.wpfloat] | None = None  # total precipitation flux
        self.pr: fa.CellKField[ta.wpfloat] | None = None  # surface rain rate
        self.ps: fa.CellKField[ta.wpfloat] | None = None  # surface snow rate
        self.pi: fa.CellKField[ta.wpfloat] | None = None  # surface ice rate
        self.pg: fa.CellKField[ta.wpfloat] | None = None  # surface graupel rate
        self.pre: fa.CellKField[ta.wpfloat] | None = None  # surface precip energy flux

    def _run(self, program: Any, **kwargs: Any) -> None:
        if self._backend is not None:
            program = program.with_backend(self._backend)
        program(**kwargs)

    def gather_from_prognostic(self, prognostic: prognostics.PrognosticState) -> None:
        """
        prepare the input fields for muphys from the prognostic state. This includes:
            - binding the references for rho and q (the muphys input fields that are stored prognostically in the dycore state)
            - diagnosing the muphys input fields that aren't stored prognostically (te, p) from the prognostic state.

        This will be called before calling the muphys
        """
        self.rho = prognostic.rho
        self.q = {s: prognostic.tracer[_STUB_TRACER_INDEX[s]] for s in _SPECIES}
        # After these two lines, the inputs that muphys reads directly (rho, q) are ready. The remaining work is computing the inputs that aren't stored.

        # Diagnose virtual temperature and temperature (te is not stored prognostically).
        self._run(
            diagnose_temperature.diagnose_virtual_temperature_and_temperature,
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
        self._run(
            diagnose_surface_pressure.diagnose_surface_pressure,
            exner=prognostic.exner,
            virtual_temperature=self.tv,
            ddqz_z_full=self.dz,
            surface_pressure=self.pressure_ifc,
            horizontal_start=0,
            horizontal_end=self._num_cells,
            vertical_start=self._num_levels,
            vertical_end=self._num_levels + 1,
            offset_provider={"Koff": dims.KDim},
        )
        # diagnose full-level pressure p
        surface_pressure = gtx.as_field(
            (dims.CellDim,), self.pressure_ifc.ndarray[:, -1], allocator=self._backend
        )
        self._run(
            diagnose_pressure.diagnose_pressure,
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

        This will be called before calling the muphys.
        output is got from muphys, and the tendencies in output will be applied to the prognostic state.
        """
        bounds = dict(
            horizontal_start=0,
            horizontal_end=self._num_cells,
            vertical_start=0,
            vertical_end=self._num_levels,
        )

        # 1. Apply moisture tendencies to the tracers (in place).
        for s in _SPECIES:
            tracer = prognostic.tracer[_STUB_TRACER_INDEX[s]]
            self._run(
                apply_tendency,
                field=tracer,
                tendency=outputs[f"tend_q{s}"],
                dt=dt,
                result=tracer,
                offset_provider={},
                **bounds,
            )

        # 2. tend_T -> exner. new_te = te + tend_T*dt (kept off te in a scratch buffer).
        self._run(
            apply_tendency,
            field=self.te,
            tendency=outputs["tend_temperature"],
            dt=dt,
            result=self._new_te,
            offset_provider={},
            **bounds,
        )
        # dTv/dt from the new temperature and the species just updated in step 1
        # (read straight from prognostic.tracer to make the data dependency explicit),
        # with the old Tv = self.tv.
        tracers = prognostic.tracer
        self._run(
            calculate_tendency.calculate_virtual_temperature_tendency,
            dtime=dt,
            qv=tracers[_STUB_TRACER_INDEX["v"]],
            qc=tracers[_STUB_TRACER_INDEX["c"]],
            qi=tracers[_STUB_TRACER_INDEX["i"]],
            qr=tracers[_STUB_TRACER_INDEX["r"]],
            qs=tracers[_STUB_TRACER_INDEX["s"]],
            qg=tracers[_STUB_TRACER_INDEX["g"]],
            temperature=self._new_te,
            virtual_temperature=self.tv,
            virtual_temperature_tendency=self._tv_tendency,
            offset_provider={},
            **bounds,
        )
        # d(exner)/dt from dTv/dt, then exner += d(exner)/dt * dt.
        self._run(
            calculate_tendency.calculate_exner_tendency,
            dtime=dt,
            virtual_temperature=self.tv,
            virtual_temperature_tendency=self._tv_tendency,
            exner=prognostic.exner,
            exner_tendency=self._exner_tendency,
            offset_provider={},
            **bounds,
        )
        self._run(
            apply_tendency,
            field=prognostic.exner,
            tendency=self._exner_tendency,
            dt=dt,
            result=prognostic.exner,
            offset_provider={},
            **bounds,
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
