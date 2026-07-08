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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import SPECIES
from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    model_options,
    type_alias as ta,
)
from icon4py.model.common.components.physics_state import PhysicsState
from icon4py.model.common.diagnostic_calculations.stencils import (
    calculate_tendency,
    diagnose_pressure,
    diagnose_surface_pressure,
    diagnose_temperature,
    update_exner_and_theta_v,
)
from icon4py.model.common.math.stencils import generic_math_operations
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.common.states import factory, prognostic_state as prognostics, tracer_state


def _require(field: fa.CellKField[ta.wpfloat] | None, name: str) -> fa.CellKField[ta.wpfloat]:
    """Return ``field``, or raise if it is inactive (``None``).

    muphys needs all six moisture species; ``TracerState`` fields are optional
    (a tracer may be inactive per ``TracerConfig``), so we fail loudly here rather
    than feed ``None`` into the microphysics.
    """
    if field is None:
        raise ValueError(f"muphys requires tracer '{name}' to be active in the TracerState")
    return field


# muphys precip diagnostic field names (pure diagnostics -- never applied to the
# prognostic state): total precip flux, surface rain / snow / ice / graupel rates,
# and surface precip energy flux.
_PRECIP_DIAGNOSTICS: tuple[str, ...] = ("pflx", "pr", "ps", "pi", "pg", "pre")


class State(PhysicsState):
    """The muphys physics State adapter.

    Bridges the dycore's prognostic state and the muphys Component contract.
    Two independent axes describe each field:

    muphys role
      - input      : fed to muphys via ``as_component_input`` -- dz, rho, q, te, p
      - returned   : muphys updates it; te/q changes come back as tendencies. tend_q
                     updates the tracers; tend_T drives the exner + theta_v update
                     (via the exact EOS, mirroring ICON's phy2dyn coupling).
                     rho and p are input-only.
      - internal   : not a muphys fields -- tv, pressure_on_cells_half_levels -- used only
                     to diagnose p and to recompute exner/theta_v from tend_T.
      - diagnostic : a muphys output stored for reporting -- pflx, pr, ps, pi, pg, pre.

    memory ownership
      - reference  : a pointer into the dycore state, no copy -- dz, rho, q.
      - owned      : a buffer allocated once here and overwritten in place each
                     step -- te, p, tv, pressure_on_cells_half_levels, and the scratch buffers.
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

        full_horizontal = {
            "horizontal_start": gtx.int32(0),
            "horizontal_end": gtx.int32(self._num_cells),
        }
        full_vertical = {
            "vertical_start": gtx.int32(0),
            "vertical_end": gtx.int32(self._num_levels),
        }

        self._diagnose_temperature = model_options.setup_program(
            program=diagnose_temperature.diagnose_virtual_temperature_and_temperature,
            backend=self._backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        self._diagnose_surface_pressure = model_options.setup_program(
            program=diagnose_surface_pressure.diagnose_surface_pressure,
            backend=self._backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes={
                "vertical_start": gtx.int32(self._num_levels),
                "vertical_end": gtx.int32(self._num_levels + 1),
            },
            offset_provider={},
        )
        self._diagnose_pressure = model_options.setup_program(
            program=diagnose_pressure.diagnose_pressure,
            backend=self._backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        self._apply_tendency = model_options.setup_program(
            program=generic_math_operations.compute_field_a_plus_coeff_times_field_b_on_cell_k,
            backend=self._backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        self._calculate_virtual_temperature_tendency = model_options.setup_program(
            program=calculate_tendency.calculate_virtual_temperature_tendency,
            backend=self._backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        self._update_exner_and_theta_v = model_options.setup_program(
            program=update_exner_and_theta_v.update_exner_and_theta_v,
            backend=self._backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )

        self.dz = metrics.get(metrics_attributes.DDQZ_Z_FULL)
        self.rho: fa.CellKField[ta.wpfloat] | None = None
        self._tracers: tracer_state.TracerState | None = None
        self.te = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self.p = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self.tv = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self.pressure_on_cells_half_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=backend
        )

        # INTERNAL
        self._new_te = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self._tv_tendency = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self._precip_diagnostics: dict[str, fa.CellKField[ta.wpfloat]] | None = None

    def gather_from_prognostic(
        self, prognostic: prognostics.PrognosticState, tracers: tracer_state.TracerState
    ) -> None:
        """
        prepare the input fields for muphys from the prognostic state. This includes:
            - binding the references for rho and q (the muphys input fields that are stored prognostically in the dycore state)
            - diagnosing the muphys input fields that aren't stored prognostically (te, p) from the prognostic state.
        """
        self.rho = prognostic.rho
        self._tracers = tracers

        # Diagnose virtual temperature and temperature (te is not stored prognostically).
        self._diagnose_temperature(
            qv=_require(tracers.qv, "qv"),
            qc=_require(tracers.qc, "qc"),
            qi=_require(tracers.qi, "qi"),
            qr=_require(tracers.qr, "qr"),
            qs=_require(tracers.qs, "qs"),
            qg=_require(tracers.qg, "qg"),
            theta_v=prognostic.theta_v,
            exner=prognostic.exner,
            virtual_temperature=self.tv,
            temperature=self.te,
        )

        self._diagnose_surface_pressure(
            exner=prognostic.exner,
            virtual_temperature=self.tv,
            ddqz_z_full=self.dz,
            surface_pressure=self.pressure_on_cells_half_levels,
        )
        surface_pressure = gtx.as_field(
            (dims.CellDim,),
            self.pressure_on_cells_half_levels.ndarray[:, -1],
            allocator=self._backend,
        )
        self._diagnose_pressure(
            ddqz_z_full=self.dz,
            virtual_temperature=self.tv,
            surface_pressure=surface_pressure,
            pressure=self.p,
            pressure_ifc=self.pressure_on_cells_half_levels,
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
        assert self._tracers is not None, "gather_from_prognostic must be called first"
        # 1. Apply moisture tendencies to the tracers (in place; tracers were bound in gather).
        for s in SPECIES:
            tracer = _require(getattr(self._tracers, f"q{s}"), f"q{s}")
            self._apply_tendency(
                field_a=tracer,
                coeff=dt,
                field_b=outputs[f"tend_q{s}"],
                output_field=tracer,
            )

        # 2. tend_T -> new temperature: new_te = te + tend_T*dt
        self._apply_tendency(
            field_a=self.te,
            coeff=dt,
            field_b=outputs["tend_temperature"],
            output_field=self._new_te,
        )

        # dTv/dt from the new temperature and the species just updated in step 1
        self._calculate_virtual_temperature_tendency(
            dtime=dt,
            qv=_require(self._tracers.qv, "qv"),
            qc=_require(self._tracers.qc, "qc"),
            qi=_require(self._tracers.qi, "qi"),
            qr=_require(self._tracers.qr, "qr"),
            qs=_require(self._tracers.qs, "qs"),
            qg=_require(self._tracers.qg, "qg"),
            temperature=self._new_te,
            virtual_temperature=self.tv,
            virtual_temperature_tendency=self._tv_tendency,
        )
        # Recompute exner via the exact EOS from the updated virtual temperature and
        # diagnose theta_v = Tv/exner, mirroring ICON's phy2dyn coupling
        # (mo_interface_iconam_aes.f90). rho is unchanged by the fast physics, so the
        # exner/rho/theta_v trio stays EOS-consistent.
        self._update_exner_and_theta_v(
            rho=self.rho,
            virtual_temperature=self.tv,
            virtual_temperature_tendency=self._tv_tendency,
            dtime=dt,
            exner=prognostic.exner,
            theta_v=prognostic.theta_v,
        )

        # 3. Store precip diagnostics (references; never applied to prognostic state).
        self._precip_diagnostics = {name: outputs[name] for name in _PRECIP_DIAGNOSTICS}

    def as_component_input(self) -> dict[str, fa.CellKField[ta.wpfloat]]:
        """
        Translate to the generic Component input dict (the 10 muphys input fields).
        """
        if self.rho is None or self._tracers is None:
            raise RuntimeError("as_component_input called before gather_from_prognostic")
        inp = {"dz": self.dz, "te": self.te, "p": self.p, "rho": self.rho}
        inp.update({f"q{s}": _require(getattr(self._tracers, f"q{s}"), f"q{s}") for s in SPECIES})
        return inp

    @property
    def precip_diagnostics(self) -> dict[str, fa.CellKField[ta.wpfloat]]:
        """muphys precip diagnostics keyed by name, ready for IO / plotting."""
        if self._precip_diagnostics is None:
            raise RuntimeError("precip_diagnostics accessed before scatter_to_prognostic")
        return self._precip_diagnostics
