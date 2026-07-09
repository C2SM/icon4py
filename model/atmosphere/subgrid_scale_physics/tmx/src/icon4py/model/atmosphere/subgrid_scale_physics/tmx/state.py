# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""TmxState — physics-state adapter for the TMX turbulent mixing scheme.

Bridges the dycore's prognostic state and the :class:`TmxComponent` contract.
The class follows the same *gather / as_component_input / scatter* pattern as
``muphys.state.State``.  Only the gather half (plus ``as_component_input``) is
implemented here; ``scatter_to_prognostic`` is deferred to Task 5.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import state_stencils
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_vn_from_uv import (
    compute_vn_from_uv,
)
from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    model_options,
    type_alias as ta,
)
from icon4py.model.common.diagnostic_calculations.stencils import (
    calculate_tendency,
    diagnose_pressure,
    diagnose_surface_pressure,
    diagnose_temperature,
    update_exner_and_theta_v,
)
from icon4py.model.common.interpolation.stencils.edge_2_cell_vector_rbf_interpolation import (
    edge_2_cell_vector_rbf_interpolation,
)
from icon4py.model.common.math.stencils import generic_math_operations
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.common.states import prognostic_state as prognostics, tracer_state


def _require(field: fa.CellKField[ta.wpfloat] | None, name: str) -> fa.CellKField[ta.wpfloat]:
    """Return *field*, or raise if it is inactive (``None``).

    TMX requires all six moisture species; ``TracerState`` fields are optional
    per ``TracerConfig``, so we fail loudly here rather than pass ``None`` into
    the physics.
    """
    if field is None:
        raise ValueError(f"tmx requires tracer '{name}' to be active in the TracerState")
    return field


class TmxState:
    """Physics-state adapter for the TMX turbulent mixing scheme.

    Two independent axes describe each field:

    tmx role
      - input    : fed to the TmxComponent via ``as_component_input``
      - internal : used only to diagnose derived inputs (pressure, T, u, v)
      - seam     : surface-flux buffers owned here; TMX fills them, scatter reads them

    memory ownership
      - reference : a pointer into the dycore state, no copy — rho, w, vn, tracers
      - owned     : a buffer allocated once here and overwritten each step —
                    temperature, virtual_temperature, pressure, pressure_ifc, u, v,
                    air_mass, cv_air, surface-flux buffers, and scatter scratch.
    """

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        ddqz_z_full: fa.CellKField[ta.wpfloat],
        rbf_coeff_c1: gtx.Field,
        rbf_coeff_c2: gtx.Field,
        c_lin_e: gtx.Field,
        primal_normal_cell_x: gtx.Field,
        primal_normal_cell_y: gtx.Field,
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

        # Geometry field — kept as a reference, not owned
        self.ddqz_z_full = ddqz_z_full

        # --- Compiled programs ---
        self._diagnose_temperature = model_options.setup_program(
            program=diagnose_temperature.diagnose_virtual_temperature_and_temperature,
            backend=backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        self._diagnose_surface_pressure = model_options.setup_program(
            program=diagnose_surface_pressure.diagnose_surface_pressure,
            backend=backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes={
                "vertical_start": gtx.int32(self._num_levels),
                "vertical_end": gtx.int32(self._num_levels + 1),
            },
            offset_provider={},
        )
        self._diagnose_pressure = model_options.setup_program(
            program=diagnose_pressure.diagnose_pressure,
            backend=backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        # RBF reconstruction of cell-centre wind from edge-normal vn
        # Needs C2E2C2E connectivity → grid.connectivities
        self._rbf_interpolation = model_options.setup_program(
            program=edge_2_cell_vector_rbf_interpolation,
            backend=backend,
            constant_args={
                "ptr_coeff_1": rbf_coeff_c1,
                "ptr_coeff_2": rbf_coeff_c2,
            },
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider=grid.connectivities,
        )
        self._compute_air_mass = model_options.setup_program(
            program=state_stencils.compute_air_mass,
            backend=backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        self._compute_cv_air = model_options.setup_program(
            program=state_stencils.compute_cv_air,
            backend=backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        # compute_vn_from_uv: used by scatter (Task 5); wired now so the program
        # is compiled with the grid's E2C connectivity at construction time.
        self._compute_vn_from_uv = model_options.setup_program(
            program=compute_vn_from_uv,
            backend=backend,
            constant_args={
                "primal_normal_cell_x": primal_normal_cell_x,
                "primal_normal_cell_y": primal_normal_cell_y,
                "c_lin_e": c_lin_e,
            },
            horizontal_sizes={
                "horizontal_start": gtx.int32(0),
                "horizontal_end": gtx.int32(grid.num_edges),
            },
            vertical_sizes=full_vertical,
            offset_provider=grid.connectivities,
        )
        # Scatter programs (Task 5)
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
                "vertical_end": gtx.int32(self._num_levels + 1),
            },
            offset_provider={},
        )
        self._apply_tendency_vn = model_options.setup_program(
            program=state_stencils.apply_tendency_on_edge_k,
            backend=backend,
            horizontal_sizes={
                "horizontal_start": gtx.int32(0),
                "horizontal_end": gtx.int32(grid.num_edges),
            },
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        self._calculate_virtual_temperature_tendency = model_options.setup_program(
            program=calculate_tendency.calculate_virtual_temperature_tendency,
            backend=backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        self._update_exner_and_theta_v = model_options.setup_program(
            program=update_exner_and_theta_v.update_exner_and_theta_v,
            backend=backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )

        # --- Owned buffers: diagnosed/computed each step ---
        self.temperature = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self.virtual_temperature = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=backend
        )
        self.pressure = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        # Half-level pressure: one extra interface level above top
        self.pressure_ifc = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=backend
        )
        self.u = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self.v = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self.air_mass = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self.cv_air = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)

        # --- Surface-flux buffers: 2-D (CellDim only), allocated once, zero.
        #     TMX fills them during a step; scatter reads them back to the land model.
        self.evapotranspiration = data_alloc.zero_field(grid, dims.CellDim, allocator=backend)
        self.sensible_heat_flux = data_alloc.zero_field(grid, dims.CellDim, allocator=backend)
        self.u_stress = data_alloc.zero_field(grid, dims.CellDim, allocator=backend)
        self.v_stress = data_alloc.zero_field(grid, dims.CellDim, allocator=backend)
        self.q_snocpymlt = data_alloc.zero_field(grid, dims.CellDim, allocator=backend)

        # --- Scratch buffers for scatter (Task 5) ---
        self._new_te = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self._tv_tendency = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self._ddt_vn = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=backend)

        # --- Prognostic references — bound during gather ---
        self._rho: fa.CellKField[ta.wpfloat] | None = None
        self._w: fa.CellKField[ta.wpfloat] | None = None
        self._vn: fa.EdgeKField[ta.wpfloat] | None = None
        self._tracers: tracer_state.TracerState | None = None

    # ------------------------------------------------------------------
    # PhysicsState protocol
    # ------------------------------------------------------------------

    def gather_from_prognostic(
        self,
        prognostic: prognostics.PrognosticState,
        tracers: tracer_state.TracerState,
    ) -> None:
        """Bind prognostic references and diagnose all TMX input fields.

        Steps (mirrors ``muphys.state.State.gather_from_prognostic``):
        1. Bind rho / w / vn / tracer references.
        2. Diagnose virtual temperature and temperature from exner + theta_v.
        3. Diagnose column pressure (surface pressure then full-column scan).
        4. Reconstruct cell-centre (u, v) from edge-normal vn via RBF.
        5. Compute air_mass = rho * dz.
        6. Compute cv_air (moisture-weighted heat capacity per unit area).
        """
        self._rho = prognostic.rho
        self._w = prognostic.w
        self._vn = prognostic.vn
        self._tracers = tracers

        # 1. Diagnose virtual temperature and temperature
        self._diagnose_temperature(
            qv=_require(tracers.qv, "qv"),
            qc=_require(tracers.qc, "qc"),
            qi=_require(tracers.qi, "qi"),
            qr=_require(tracers.qr, "qr"),
            qs=_require(tracers.qs, "qs"),
            qg=_require(tracers.qg, "qg"),
            theta_v=prognostic.theta_v,
            exner=prognostic.exner,
            virtual_temperature=self.virtual_temperature,
            temperature=self.temperature,
        )

        # 2. Surface pressure at the bottom interface (half-level num_levels)
        self._diagnose_surface_pressure(
            exner=prognostic.exner,
            virtual_temperature=self.virtual_temperature,
            ddqz_z_full=self.ddqz_z_full,
            surface_pressure=self.pressure_ifc,
        )
        # Extract the 1-D surface-pressure slice for the full-column scan
        surface_pressure = gtx.as_field(
            (dims.CellDim,),
            self.pressure_ifc.ndarray[:, -1],
            allocator=self._backend,
        )
        self._diagnose_pressure(
            ddqz_z_full=self.ddqz_z_full,
            virtual_temperature=self.virtual_temperature,
            surface_pressure=surface_pressure,
            pressure=self.pressure,
            pressure_ifc=self.pressure_ifc,
        )

        # 3. Reconstruct cell-centre (u, v) from edge-normal vn via RBF
        self._rbf_interpolation(
            p_e_in=prognostic.vn,
            p_u_out=self.u,
            p_v_out=self.v,
        )

        # 4. Air mass = rho * dz
        self._compute_air_mass(
            rho=self._rho,
            ddqz_z_full=self.ddqz_z_full,
            air_mass=self.air_mass,
        )

        # 5. cv_air (moisture-weighted)
        self._compute_cv_air(
            qv=_require(tracers.qv, "qv"),
            qc=_require(tracers.qc, "qc"),
            qi=_require(tracers.qi, "qi"),
            qr=_require(tracers.qr, "qr"),
            qs=_require(tracers.qs, "qs"),
            qg=_require(tracers.qg, "qg"),
            air_mass=self.air_mass,
            cv_air=self.cv_air,
        )

    def scatter_to_prognostic(
        self,
        prognostic: prognostics.PrognosticState,
        outputs: dict[str, fa.CellKField[ta.wpfloat]],
        dtime: datetime.timedelta,
    ) -> None:
        """Outbound translation: apply TMX output tendencies back to the prognostic state.

        Apply order (must match brief):
        1. Moisture tracers: qv/qc/qi += ddt * dt  (qr/qs/qg untouched — TMX does not diffuse them)
        2. ddt_temperature → new_temperature → Tv tendency → update exner + theta_v (muphys path)
        3. Project (ddt_u, ddt_v) → _ddt_vn via compute_vn_from_uv, then vn += dt * _ddt_vn
        4. w += dt * ddt_w  (KDim+1 half-levels)
        5. Store 8 diagnostics as attributes
        """
        assert self._tracers is not None, "gather_from_prognostic must be called first"
        dt = dtime.total_seconds()

        # 1. Moisture tendencies: only qv, qc, qi (TMX does not diffuse qr/qs/qg)
        for name in ("qv", "qc", "qi"):
            tracer = _require(getattr(self._tracers, name), name)
            self._apply_tendency(
                field_a=tracer,
                coeff=ta.wpfloat(dt),
                field_b=outputs[f"ddt_{name}"],
                output_field=tracer,
            )

        # 2. ddt_temperature → exner/theta_v (verbatim muphys scatter step 2)
        # 2a. new_temperature = temperature + ddt_temperature * dt
        self._apply_tendency(
            field_a=self.temperature,
            coeff=ta.wpfloat(dt),
            field_b=outputs["ddt_temperature"],
            output_field=self._new_te,
        )
        # 2b. Tv tendency: uses updated tracers (post step-1) and new temperature
        self._calculate_virtual_temperature_tendency(
            dtime=ta.wpfloat(dt),
            qv=_require(self._tracers.qv, "qv"),
            qc=_require(self._tracers.qc, "qc"),
            qi=_require(self._tracers.qi, "qi"),
            qr=_require(self._tracers.qr, "qr"),
            qs=_require(self._tracers.qs, "qs"),
            qg=_require(self._tracers.qg, "qg"),
            temperature=self._new_te,
            virtual_temperature=self.virtual_temperature,
            virtual_temperature_tendency=self._tv_tendency,
        )
        # 2c. Recompute exner via exact EOS; diagnose theta_v = Tv_new / exner_new
        self._update_exner_and_theta_v(
            rho=self._rho,
            virtual_temperature=self.virtual_temperature,
            virtual_temperature_tendency=self._tv_tendency,
            dtime=ta.wpfloat(dt),
            exner=prognostic.exner,
            theta_v=prognostic.theta_v,
        )

        # 3. Project wind tendencies (ddt_u, ddt_v) onto edge normals, then apply
        self._compute_vn_from_uv(
            u=outputs["ddt_u"],
            v=outputs["ddt_v"],
            vn=self._ddt_vn,
        )
        self._apply_tendency_vn(
            field_a=prognostic.vn,
            coeff=ta.wpfloat(dt),
            field_b=self._ddt_vn,
            output_field=prognostic.vn,
        )

        # 4. w (KDim+1 half-levels)
        self._apply_tendency_w(
            field_a=prognostic.w,
            coeff=ta.wpfloat(dt),
            field_b=outputs["ddt_w"],
            output_field=prognostic.w,
        )

        # 5. Store 8 diagnostics as attributes
        self.km = outputs["km"]
        self.kh = outputs["kh"]
        self.heating = outputs["heating"]
        self.dissip_ke = outputs["dissip_ke"]
        self.cptgz_vi = outputs["cptgz_vi"]
        self.dissip_ke_vi = outputs["dissip_ke_vi"]
        self.int_energy_vi = outputs["int_energy_vi"]
        self.int_energy_vi_tend = outputs["int_energy_vi_tend"]

    def as_component_input(self) -> dict[str, fa.CellKField[ta.wpfloat]]:
        """Return exactly the 21 ``INPUTS_PROPERTIES`` keys mapped to GT4Py fields."""
        if self._rho is None or self._tracers is None:
            raise RuntimeError("as_component_input called before gather_from_prognostic")
        return {
            # Diagnosed thermodynamic fields
            "temperature": self.temperature,
            "virtual_temperature": self.virtual_temperature,
            "pressure": self.pressure,
            "pressure_ifc": self.pressure_ifc,
            # Reconstructed winds
            "u": self.u,
            "v": self.v,
            # Prognostic references (no copy)
            "w": self._w,
            "rho": self._rho,
            # Tracers (no copy)
            "qv": _require(self._tracers.qv, "qv"),
            "qc": _require(self._tracers.qc, "qc"),
            "qi": _require(self._tracers.qi, "qi"),
            "qr": _require(self._tracers.qr, "qr"),
            "qs": _require(self._tracers.qs, "qs"),
            "qg": _require(self._tracers.qg, "qg"),
            # TMX-specific computed fields
            "air_mass": self.air_mass,
            "cv_air": self.cv_air,
            # Surface-flux seam (phase-2; zero until TMX fills them)
            "evapotranspiration": self.evapotranspiration,
            "sensible_heat_flux": self.sensible_heat_flux,
            "u_stress": self.u_stress,
            "v_stress": self.v_stress,
            "q_snocpymlt": self.q_snocpymlt,
        }
