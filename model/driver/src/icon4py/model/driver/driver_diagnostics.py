# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
The standard diagnostics the driver offers as output variables.

:data:`DIAGNOSTIC_VARIABLES` names them, by the standard name of their quantity;
:class:`DiagnosticsComputer` diagnoses the requested ones from the prognostic state and the
active tracers into the driver's ``Diagnostics``, running only the programs those variables
need. ``driver_io`` assembles them with the other output variables.
"""

from collections.abc import Collection
from typing import Final, NamedTuple, Self

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field, type_alias as ta
from icon4py.model.common.framework import quantities as qty, states
from icon4py.model.common.grid import base as grid_base, horizontal as h_grid
from icon4py.model.common.interpolation import interpolation_attributes as intp_attr
from icon4py.model.common.interpolation.stencils import edge_2_cell_vector_rbf_interpolation as rbf
from icon4py.model.common.metrics import metrics_attributes as metrics_attr
from icon4py.model.common.physics.thermodynamics import compute_pressure, compute_temperature
from icon4py.model.common.states import factory
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver import driver_output_sources as output_sources


EASTWARD_WIND: Final = output_sources.standard_name(qty.UOnCellK)
NORTHWARD_WIND: Final = output_sources.standard_name(qty.VOnCellK)
TEMPERATURE: Final = output_sources.standard_name(qty.TemperatureOnCellK)
VIRTUAL_TEMPERATURE: Final = output_sources.standard_name(qty.VirtualTemperatureOnCellK)
PRESSURE: Final = output_sources.standard_name(qty.PressureOnCellK)
SURFACE_PRESSURE: Final = output_sources.standard_name(qty.SurfacePressureOnCell)

DIAGNOSTIC_VARIABLES: Final[tuple[str, ...]] = (
    EASTWARD_WIND,
    NORTHWARD_WIND,
    TEMPERATURE,
    VIRTUAL_TEMPERATURE,
    PRESSURE,
    SURFACE_PRESSURE,
)


class _Programs(NamedTuple):
    """Which programs of the ``DiagnosticsComputer`` a set of variables needs."""

    temperature: bool
    winds: bool
    pressure: bool

    @classmethod
    def needed_by(cls, variables: Collection[str]) -> Self:
        requested = frozenset(variables)
        pressure = bool(requested & {PRESSURE, SURFACE_PRESSURE})
        return cls(
            # the pressure program integrates the virtual temperature hydrostatically
            temperature=pressure or bool(requested & {TEMPERATURE, VIRTUAL_TEMPERATURE}),
            winds=bool(requested & {EASTWARD_WIND, NORTHWARD_WIND}),
            pressure=pressure,
        )


class DiagnosticsComputer:
    """Diagnoses the requested standard diagnostic output fields.

    Three programs: temperature and virtual temperature from ``theta_v``, ``exner`` and
    the active tracers (an inactive species counts as zero, so a dry run gives the
    dry-air temperature); the cell winds by RBF interpolation of ``vn``; and the pressure
    on the model levels and at the surface by vertical integration from the diagnosed
    surface pressure, which needs the virtual temperature. :meth:`compute` runs only the
    programs the variables of the call need (``_Programs``), so a group that writes just
    the winds pays for neither temperature nor pressure.

    The diagnostics are written into the ``Diagnostics`` handed to :meth:`compute`
    (the driver's, allocated once); only the stencil scratch field and the zero stand-in
    for inactive tracers are owned here, and the static inputs of the requested programs
    are read once from the field sources at construction.

    TODO(kotsaloscv): the physics driver diagnoses the same variables into its own
    ``Diagnostics``; share them once the driver groups its model state.
    """

    def __init__(
        self,
        *,
        grid: grid_base.Grid,
        backend: gtx.typing.Backend | None,
        variables: Collection[str],
        metrics: factory.FieldSource,
        interpolation: factory.FieldSource,
    ) -> None:
        requested = frozenset(variables)
        if unknown := requested - set(DIAGNOSTIC_VARIABLES):
            raise ValueError(
                f"Unknown diagnostic output variables: {', '.join(sorted(unknown))}. Known "
                f"variables are: {', '.join(DIAGNOSTIC_VARIABLES)}."
            )
        self._variables = requested
        self._grid = grid
        self._backend = backend
        self._num_levels = grid.num_levels
        self._surface_level = dims.KHalfDim(grid.num_levels)
        cell_domain = h_grid.domain(dims.CellDim)
        self._end_cell_end = grid.end_index(cell_domain(h_grid.Zone.END))
        # ICON's rbf_vec_interpol_cell starts at refinement level 2 (``rl_start = 2``)
        self._start_cell_lateral_boundary_level_2 = grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )

        def _zero_full() -> gtx.Field:
            return data_alloc.zero_field(
                grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
            )

        # stands in for every inactive tracer species (read only, so one buffer serves all)
        self._zero_tracer = _zero_full()
        # scratch output of the pressure program (interface pressure on the model levels)
        self._pressure_ifc_on_model_levels = _zero_full()

        # static inputs, fetched only for the programs some variable needs
        programs = _Programs.needed_by(requested)
        self._ddqz_z_full: gtx.Field | None = (
            metrics.get(metrics_attr.DDQZ_Z_FULL) if programs.pressure else None
        )
        self._rbf_vec_coeff_c1: gtx.Field | None = None
        self._rbf_vec_coeff_c2: gtx.Field | None = None
        if programs.winds:
            self._rbf_vec_coeff_c1 = interpolation.get(intp_attr.RBF_VEC_COEFF_C1)
            self._rbf_vec_coeff_c2 = interpolation.get(intp_attr.RBF_VEC_COEFF_C2)

    @property
    def variables(self) -> frozenset[str]:
        """The diagnostic variables this computer can provide."""
        return self._variables

    def _tracer_or_zero(self, tracer: gtx.Field | None) -> gtx.Field:
        return self._zero_tracer if tracer is None else tracer

    def compute(
        self,
        prognostic_state: states.PrognosticState,
        tracer_state: states.TracerState,
        diagnostic_state: states.Diagnostics,
        variables: Collection[str],
    ) -> dict[str, field.Field]:
        """Diagnose the requested variables (a subset of those given at construction).

        The fields of ``diagnostic_state`` are overwritten in place; the returned fields
        are those fields, read for the requested variables only (the surface pressure is
        the view of the bottom half level of ``pressure_ifc``).
        """
        requested = frozenset(variables)
        if unknown := requested - self._variables:
            raise ValueError(
                f"Diagnostic output variables not configured for this computer: "
                f"{', '.join(sorted(unknown))}."
            )
        backend = self._backend
        num_levels = self._num_levels
        end_cell_end = self._end_cell_end
        programs = _Programs.needed_by(requested)

        if programs.temperature:
            compute_temperature.compute_virtual_temperature_and_temperature.with_backend(backend)(
                qv=self._tracer_or_zero(tracer_state.qv),
                qc=self._tracer_or_zero(tracer_state.qc),
                qi=self._tracer_or_zero(tracer_state.qi),
                qr=self._tracer_or_zero(tracer_state.qr),
                qs=self._tracer_or_zero(tracer_state.qs),
                qg=self._tracer_or_zero(tracer_state.qg),
                theta_v=prognostic_state.theta_v,
                exner=prognostic_state.exner,
                virtual_temperature=diagnostic_state.virtual_temperature,
                temperature=diagnostic_state.temperature,
                horizontal_start=0,
                horizontal_end=end_cell_end,
                vertical_start=0,
                vertical_end=num_levels,
                offset_provider={},
            )

        if programs.winds:
            assert self._rbf_vec_coeff_c1 is not None and self._rbf_vec_coeff_c2 is not None
            rbf.edge_2_cell_vector_rbf_interpolation.with_backend(backend)(
                p_e_in=prognostic_state.vn,
                ptr_coeff_1=self._rbf_vec_coeff_c1,
                ptr_coeff_2=self._rbf_vec_coeff_c2,
                p_u_out=diagnostic_state.u,
                p_v_out=diagnostic_state.v,
                horizontal_start=self._start_cell_lateral_boundary_level_2,
                horizontal_end=end_cell_end,
                vertical_start=0,
                vertical_end=num_levels,
                offset_provider={"C2E2C2E": self._grid.get_connectivity("C2E2C2E")},
            )

        if programs.pressure:
            assert self._ddqz_z_full is not None
            compute_pressure.compute_surface_and_hydrostatic_pressure.with_backend(backend)(
                exner=prognostic_state.exner,
                virtual_temperature=diagnostic_state.virtual_temperature,
                ddqz_z_full=self._ddqz_z_full,
                pressure=diagnostic_state.pressure,
                pressure_ifc_on_model_levels=self._pressure_ifc_on_model_levels,
                pressure_ifc=diagnostic_state.pressure_ifc,
                horizontal_start=0,
                horizontal_end=end_cell_end,
                vertical_start=0,
                vertical_end=num_levels,
                offset_provider={},
            )

        fields = output_sources.by_standard_name(diagnostic_state)
        fields[SURFACE_PRESSURE] = field.Field(
            qty.SurfacePressureOnCell, diagnostic_state.pressure_ifc[self._surface_level]
        )
        return {name: fields[name] for name in DIAGNOSTIC_VARIABLES if name in requested}
