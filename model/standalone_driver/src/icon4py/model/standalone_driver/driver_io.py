# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Driver-side glue between the model state and the ``icon4py.model.common.io`` module.

- assembles the prognostic model state into the ``dict[str, xarray.DataArray]`` consumed
  by ``IOMonitor.store`` (:func:`prognostic_state_to_dataarrays`),
- computes the standard diagnostic output fields (u, v, temperature, virtual temperature,
  pressure) from the prognostic state (:func:`compute_diagnostics`) and assembles them
  (:func:`diagnostic_fields_to_dataarrays`),
- provides a factory that builds an ``IOMonitor`` with a prognostic and (optionally) a
  diagnostic field group (:func:`create_io_monitor`).

The whole module imports the optional ``icon4py-common[io]`` dependencies (xarray,
netCDF4, uxarray, cftime) lazily, so the driver keeps working without them as long as
output is not requested.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib
from typing import TYPE_CHECKING, Any, Final

from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import xarray as xr

    from icon4py.model.common.decomposition import definitions as decomposition_defs
    from icon4py.model.common.grid import base as grid_base, vertical as v_grid
    from icon4py.model.common.io import io as common_io
    from icon4py.model.common.states import prognostic_state as prognostics
    from icon4py.model.standalone_driver import driver_states


#: Output subfolder created *inside* the driver's run/output directory.
OUTPUT_SUBDIR: Final[str] = "io"

#: File-name stubs for the two field groups (a counter + ``.nc`` is appended).
DEFAULT_PROGNOSTIC_FILENAME: Final[str] = "icon4py_prognostics"
DEFAULT_DIAGNOSTIC_FILENAME: Final[str] = "icon4py_diagnostics"


# --------------------------------------------------------------------------------------
# Prognostic fields
# --------------------------------------------------------------------------------------

#: ``state_key -> (prognostic_attr, cf_attributes_key, is_on_interface)``; mirrors the
#: tested reference in ``model/common/tests/common/io/utils.py::model_state``.
_PROGNOSTIC_FIELD_SPECS: Final[dict[str, tuple[str, str, bool]]] = {
    "air_density": ("rho", "air_density", False),
    "exner_function": ("exner", "exner_function", False),
    "theta_v": ("theta_v", "virtual_potential_temperature", False),
    "upward_air_velocity": ("w", "upward_air_velocity", True),
    "normal_velocity": ("vn", "normal_velocity", False),
}

#: Prognostic variables written by default.
DEFAULT_OUTPUT_VARIABLES: Final[list[str]] = list(_PROGNOSTIC_FIELD_SPECS.keys())


def _to_host_data_array(
    field: Any, cf_attrs: dict[str, Any], *, is_on_interface: bool
) -> xr.DataArray:
    """Wrap a gt4py field as a CF/UGRID-annotated host (numpy) ``xarray.DataArray``.

    ``io.utils.to_data_array`` mutates the ``attrs`` dict it is handed (it adds the UGRID
    ``location``/``coordinates``/``mesh`` keys), so callers must pass a fresh copy. The data
    buffer is forced to host via ``data_alloc.as_numpy`` so the path works for GPU backends
    too (netCDF4 cannot consume a device array).
    """
    from icon4py.model.common.io import utils as io_utils  # noqa: PLC0415

    data_array = io_utils.to_data_array(field, cf_attrs, is_on_interface=is_on_interface)
    data_array.data = data_alloc.as_numpy(field)
    return data_array


def prognostic_state_to_dataarrays(
    prognostic_state: prognostics.PrognosticState,
    variables: list[str] | None = None,
) -> dict[str, xr.DataArray]:
    """Assemble a CF/UGRID-annotated model-state dict from a ``PrognosticState``."""
    from icon4py.model.common.states import data as state_data  # noqa: PLC0415

    selected = DEFAULT_OUTPUT_VARIABLES if variables is None else variables

    state: dict[str, xr.DataArray] = {}
    for key in selected:
        try:
            prognostic_attr, cf_key, is_on_interface = _PROGNOSTIC_FIELD_SPECS[key]
        except KeyError as err:
            raise ValueError(
                f"Unknown output variable '{key}'. Known variables are: {DEFAULT_OUTPUT_VARIABLES}."
            ) from err
        field = getattr(prognostic_state, prognostic_attr)
        attrs = dict(
            state_data.PROGNOSTIC_CF_ATTRIBUTES[cf_key]
        )  # fresh copy: to_data_array mutates
        state[key] = _to_host_data_array(field, attrs, is_on_interface=is_on_interface)
    return state


# --------------------------------------------------------------------------------------
# Diagnostic fields
# --------------------------------------------------------------------------------------

#: Diagnostic output variables. All are cell/full-level fields, i.e. they ride the same
#: (proven) write path as ``air_density``. ``surface_pressure`` is intentionally omitted
#: for now: it is a horizontal-only field and the horizontal-only write path is untested.
DEFAULT_DIAGNOSTIC_VARIABLES: Final[list[str]] = [
    "eastward_wind",
    "northward_wind",
    "temperature",
    "virtual_temperature",
    "pressure",
]


@dataclasses.dataclass
class DiagnosticInputs:
    """Static fields needed to compute the diagnostic output fields.

    Fetched once from the driver's field sources and reused every output step.
    """

    ddqz_z_full: Any
    rbf_vec_coeff_c1: Any
    rbf_vec_coeff_c2: Any


def fetch_diagnostic_inputs(
    static_field_factories: driver_states.StaticFieldFactories,
) -> DiagnosticInputs:
    """Retrieve ``ddqz_z_full`` and the cell rbf coefficients from the field sources."""
    from icon4py.model.common.interpolation import (  # noqa: PLC0415
        interpolation_attributes as intp_attr,
    )
    from icon4py.model.common.metrics import metrics_attributes as metrics_attr  # noqa: PLC0415

    metrics = static_field_factories.metrics_field_source
    interpolation = static_field_factories.interpolation_field_source
    return DiagnosticInputs(
        ddqz_z_full=metrics.get(metrics_attr.DDQZ_Z_FULL),
        rbf_vec_coeff_c1=interpolation.get(intp_attr.RBF_VEC_COEFF_C1),
        rbf_vec_coeff_c2=interpolation.get(intp_attr.RBF_VEC_COEFF_C2),
    )


def compute_diagnostics(
    prognostic_state: prognostics.PrognosticState,
    *,
    grid: grid_base.Grid,
    backend: Any,
    inputs: DiagnosticInputs,
) -> dict[str, Any]:
    """Compute the diagnostic output fields from the prognostic state.

    This transcribes the verified reference sequence in
    ``model/common/tests/common/diagnostic_calculations/unit_tests/test_diagnostic_calculations.py``
    (temperature, edge->cell rbf winds, surface pressure, pressure), using the same domain
    bounds and offset providers. The dry-air path (zero hydrometeors) is assumed.

    Returns:
        ``{eastward_wind, northward_wind, temperature, virtual_temperature, pressure}`` as
        gt4py cell/full-level fields.
    """
    from icon4py.model.common import dimension as dims  # noqa: PLC0415
    from icon4py.model.common.diagnostic_calculations.stencils import (  # noqa: PLC0415
        diagnose_pressure,
        diagnose_surface_pressure,
        diagnose_temperature,
    )
    from icon4py.model.common.grid import horizontal as h_grid  # noqa: PLC0415
    from icon4py.model.common.interpolation.stencils import (  # noqa: PLC0415
        edge_2_cell_vector_rbf_interpolation as rbf,
    )

    num_levels = grid.num_levels
    cell_domain = h_grid.domain(dims.CellDim)
    end_cell_end = grid.end_index(cell_domain(h_grid.Zone.END))
    cell_lateral_boundary_level_2 = grid.end_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    def _zero_full() -> Any:
        return data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend)

    def _zero_interface() -> Any:
        return data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, dtype=float, extend={dims.KDim: 1}, allocator=backend
        )

    # dry air: all hydrometeors are zero
    qv, qc, qi, qr, qs, qg = (_zero_full() for _ in range(6))
    temperature = _zero_full()
    virtual_temperature = _zero_full()
    u = _zero_full()
    v = _zero_full()
    pressure = _zero_full()
    pressure_ifc = _zero_interface()
    surface_pressure_k = _zero_interface()

    diagnose_temperature.diagnose_virtual_temperature_and_temperature.with_backend(backend)(
        qv=qv,
        qc=qc,
        qi=qi,
        qr=qr,
        qs=qs,
        qg=qg,
        theta_v=prognostic_state.theta_v,
        exner=prognostic_state.exner,
        virtual_temperature=virtual_temperature,
        temperature=temperature,
        horizontal_start=0,
        horizontal_end=end_cell_end,
        vertical_start=0,
        vertical_end=num_levels,
        offset_provider={},
    )

    rbf.edge_2_cell_vector_rbf_interpolation.with_backend(backend)(
        p_e_in=prognostic_state.vn,
        ptr_coeff_1=inputs.rbf_vec_coeff_c1,
        ptr_coeff_2=inputs.rbf_vec_coeff_c2,
        p_u_out=u,
        p_v_out=v,
        horizontal_start=cell_lateral_boundary_level_2,
        horizontal_end=end_cell_end,
        vertical_start=0,
        vertical_end=num_levels,
        offset_provider={"C2E2C2E": grid.get_connectivity("C2E2C2E")},
    )

    diagnose_surface_pressure.diagnose_surface_pressure.with_backend(backend)(
        exner=prognostic_state.exner,
        virtual_temperature=virtual_temperature,
        ddqz_z_full=inputs.ddqz_z_full,
        surface_pressure=surface_pressure_k,
        horizontal_start=0,
        horizontal_end=end_cell_end,
        vertical_start=num_levels,
        vertical_end=num_levels + 1,
        offset_provider={"Koff": dims.KDim},
    )

    # surface pressure lives at the bottom interface; extract it as a cell field,
    # allocated with the model allocator so the buffer stays on the right device.
    # Typed as Any: gt4py's NDArrayObject protocol does not expose __setitem__, so the
    # in-place buffer fill below would not type-check against the precise Field type.
    surface_pressure: Any = data_alloc.zero_field(
        grid, dims.CellDim, dtype=float, allocator=backend
    )
    surface_pressure.ndarray[:] = surface_pressure_k.ndarray[:, num_levels]
    pressure_ifc.ndarray[:, -1] = surface_pressure.ndarray

    diagnose_pressure.diagnose_pressure.with_backend(backend)(
        inputs.ddqz_z_full,
        virtual_temperature,
        surface_pressure,
        pressure,
        pressure_ifc,
        horizontal_start=0,
        horizontal_end=end_cell_end,
        vertical_start=0,
        vertical_end=num_levels,
        offset_provider={},
    )

    return {
        "eastward_wind": u,
        "northward_wind": v,
        "temperature": temperature,
        "virtual_temperature": virtual_temperature,
        "pressure": pressure,
    }


def diagnostic_fields_to_dataarrays(
    diagnostic_fields: dict[str, Any],
) -> dict[str, xr.DataArray]:
    """Assemble CF/UGRID-annotated DataArrays from computed diagnostic fields."""
    from icon4py.model.common.states import data as state_data  # noqa: PLC0415

    state: dict[str, xr.DataArray] = {}
    for key, field in diagnostic_fields.items():
        attrs = dict(state_data.DIAGNOSTIC_CF_ATTRIBUTES[key])  # fresh copy: to_data_array mutates
        state[key] = _to_host_data_array(field, attrs, is_on_interface=False)
    return state


# --------------------------------------------------------------------------------------
# Monitor factory
# --------------------------------------------------------------------------------------


def create_io_monitor(
    *,
    output_path: pathlib.Path,
    grid_file_path: pathlib.Path,
    grid: grid_base.Grid,
    vertical_grid: v_grid.VerticalGrid,
    start_date: dt.datetime,
    dtime: dt.timedelta,
    variables: list[str] | None = None,
    diagnostic_variables: list[str] | None = None,
    include_diagnostics: bool = True,
    process_properties: decomposition_defs.ProcessProperties | None = None,
) -> common_io.IOMonitor:
    """Build a single-node ``IOMonitor`` with a prognostic and, optionally, a diagnostic group.

    The output interval and start time are derived from the driver's ``start_date`` and
    ``dtime`` so that the monitor's capture logic fires on every model step. The first model
    time seen at the IO hook is ``start_date + dtime`` (the simulation date is advanced before
    integration), so the field groups start there.
    """
    from icon4py.model.common.io import io as common_io  # noqa: PLC0415

    prognostic_vars = DEFAULT_OUTPUT_VARIABLES if variables is None else variables
    diagnostic_vars = (
        DEFAULT_DIAGNOSTIC_VARIABLES if diagnostic_variables is None else diagnostic_variables
    )

    io_output_path = output_path / OUTPUT_SUBDIR
    first_output_time = start_date + dtime
    interval = f"{int(dtime.total_seconds())} SECONDS"

    field_groups = [
        common_io.FieldGroupIOConfig(
            output_interval=interval,
            start_time=first_output_time.isoformat(),
            filename=DEFAULT_PROGNOSTIC_FILENAME,
            variables=prognostic_vars,
            nc_title="ICON4Py standalone driver prognostic output",
            nc_comment="Prognostic fields from the ICON4Py standalone driver.",
        )
    ]
    if include_diagnostics:
        field_groups.append(
            common_io.FieldGroupIOConfig(
                output_interval=interval,
                start_time=first_output_time.isoformat(),
                filename=DEFAULT_DIAGNOSTIC_FILENAME,
                variables=diagnostic_vars,
                nc_title="ICON4Py standalone driver diagnostic output",
                nc_comment="Diagnostic fields from the ICON4Py standalone driver.",
            )
        )

    config = common_io.IOConfig(output_path=str(io_output_path), field_groups=field_groups)
    return common_io.IOMonitor(
        config=config,
        vertical_size=vertical_grid,
        horizontal_size=grid.config.horizontal_config,
        grid_file_name=str(grid_file_path),
        grid_id=grid.id,
    )
