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
- builds an ``IOMonitor`` that writes all requested fields to one file
  (:func:`create_io_monitor`).
"""

import dataclasses
import pathlib
import uuid
from typing import Any, Final

import gt4py.next as gtx
import xarray as xr

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.diagnostic_calculations.stencils import (
    diagnose_pressure,
    diagnose_surface_pressure,
    diagnose_temperature,
)
from icon4py.model.common.grid import base as grid_base, horizontal as h_grid, vertical as v_grid
from icon4py.model.common.interpolation import interpolation_attributes as intp_attr
from icon4py.model.common.interpolation.stencils import edge_2_cell_vector_rbf_interpolation as rbf
from icon4py.model.common.io import io as common_io, utils as io_utils
from icon4py.model.common.metrics import metrics_attributes as metrics_attr
from icon4py.model.common.states import data as state_data, prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import driver_states


#: Output subfolder created *inside* the driver's run/output directory.
OUTPUT_SUBDIR: Final[str] = "output"

#: File-name stub for the output file (a counter + ``.nc`` is appended).
DEFAULT_OUTPUT_FILENAME: Final[str] = "icon4py_output"


# --------------------------------------------------------------------------------------
# Prognostic fields
# --------------------------------------------------------------------------------------


#: Default prognostic output variables, selected by CF name from the
#: ``states.data.PROGNOSTIC_CF_ATTRIBUTES`` catalog (which also holds fields the driver does
#: not output, e.g. ``tangential_velocity``). The metadata, the state attribute
#: (``icon_var_name``) and the vertical placement (``is_on_half_levels``) all come from that
#: catalog; this list only selects which entries to emit.
PROGNOSTIC_VARIABLES: Final[list[str]] = [
    "air_density",
    "exner_function",
    "virtual_potential_temperature",
    "upward_air_velocity",
    "normal_velocity",
]


def prognostic_state_to_dataarrays(
    prognostic_state: prognostics.PrognosticState,
    variables: list[str] | None = None,
) -> dict[str, xr.DataArray]:
    """Assemble a CF/UGRID-annotated model-state dict from a ``PrognosticState``."""
    selected = PROGNOSTIC_VARIABLES if variables is None else variables

    state: dict[str, xr.DataArray] = {}
    for name in selected:
        try:
            metadata = state_data.PROGNOSTIC_CF_ATTRIBUTES[name]
        except KeyError as err:
            raise ValueError(
                f"Unknown prognostic output variable '{name}'. "
                f"Known variables are: {PROGNOSTIC_VARIABLES}."
            ) from err
        field = getattr(prognostic_state, metadata["icon_var_name"])
        state[name] = io_utils.to_data_array(
            field,
            metadata,
            is_on_half_levels=metadata.get("is_on_half_levels", False),
            to_host=True,
        )
    return state


# --------------------------------------------------------------------------------------
# Diagnostic fields
# --------------------------------------------------------------------------------------

#: Diagnostic output variables. All are cell/full-level fields, i.e. they ride the same
#: (proven) write path as ``air_density``. ``surface_pressure`` is intentionally omitted
#: for now: it is a horizontal-only field and the horizontal-only write path is untested.
DIAGNOSTIC_VARIABLES: Final[list[str]] = [
    "eastward_wind",
    "northward_wind",
    "temperature",
    "virtual_temperature",
    "pressure",
]

#: All output variables (prognostic + diagnostic), written together into the same file.
DEFAULT_OUTPUT_VARIABLES: Final[list[str]] = [*PROGNOSTIC_VARIABLES, *DIAGNOSTIC_VARIABLES]


@dataclasses.dataclass
class DiagnosticFields:
    """Static fields needed to compute the diagnostic output fields.

    Fetched once from the driver's field sources and reused every output step. Kept as a
    plain field container (instead of passing the field factories around) so that
    :func:`compute_diagnostics` stays usable with bare fields, e.g. in tests.
    """

    ddqz_z_full: gtx.Field
    rbf_vec_coeff_c1: gtx.Field
    rbf_vec_coeff_c2: gtx.Field


def fetch_diagnostic_fields(
    static_field_factories: driver_states.StaticFieldFactories,
) -> DiagnosticFields:
    """Retrieve ``ddqz_z_full`` and the cell rbf coefficients from the field sources."""
    metrics = static_field_factories.metrics_field_source
    interpolation = static_field_factories.interpolation_field_source
    return DiagnosticFields(
        ddqz_z_full=metrics.get(metrics_attr.DDQZ_Z_FULL),
        rbf_vec_coeff_c1=interpolation.get(intp_attr.RBF_VEC_COEFF_C1),
        rbf_vec_coeff_c2=interpolation.get(intp_attr.RBF_VEC_COEFF_C2),
    )


def compute_diagnostics(
    prognostic_state: prognostics.PrognosticState,
    *,
    grid: grid_base.Grid,
    backend: gtx.typing.Backend | None,
    inputs: DiagnosticFields,
) -> dict[str, gtx.Field]:
    """Compute the diagnostic output fields from the prognostic state.

    Diagnoses temperature and virtual temperature from ``theta_v``/``exner``, the
    cell-center wind components by RBF interpolation of ``vn``, and the pressure field
    by vertical integration from the diagnosed surface pressure. The dry-air path (zero
    hydrometeors) is assumed.

    Returns:
        ``{eastward_wind, northward_wind, temperature, virtual_temperature, pressure}`` as
        gt4py cell/full-level fields.
    """
    num_levels = grid.num_levels
    cell_domain = h_grid.domain(dims.CellDim)
    end_cell_end = grid.end_index(cell_domain(h_grid.Zone.END))
    cell_lateral_boundary_level_2 = grid.end_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    def _zero_full() -> gtx.Field:
        return data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
        )

    def _zero_interface() -> gtx.Field:
        return data_alloc.zero_field(
            grid,
            dims.CellDim,
            dims.KDim,
            dtype=ta.wpfloat,
            extend={dims.KDim: 1},
            allocator=backend,
        )

    # dry air: all hydrometeors are zero
    qv, qc, qi, qr, qs, qg = (_zero_full() for _ in range(6))
    temperature = _zero_full()
    virtual_temperature = _zero_full()
    u = _zero_full()
    v = _zero_full()
    pressure = _zero_full()
    # Typed as Any: gt4py's NDArrayObject protocol does not expose __setitem__, so the
    # in-place buffer fill below would not type-check against the precise Field type.
    pressure_ifc: Any = _zero_interface()
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
        grid, dims.CellDim, dtype=ta.wpfloat, allocator=backend
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
    diagnostic_fields: dict[str, gtx.Field],
) -> dict[str, xr.DataArray]:
    """Assemble CF/UGRID-annotated DataArrays from computed diagnostic fields."""
    state: dict[str, xr.DataArray] = {}
    for name, field in diagnostic_fields.items():
        metadata = state_data.DIAGNOSTIC_CF_ATTRIBUTES[name]
        state[name] = io_utils.to_data_array(
            field,
            metadata,
            is_on_half_levels=metadata.get("is_on_half_levels", False),
            to_host=True,
        )
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
    variables: list[str] | None = None,
    output_interval_steps: int = 1,
    process_props: decomposition_defs.ProcessProperties | None = None,
) -> common_io.IOMonitor:
    """Build a single-node ``IOMonitor`` with one field group holding all output fields.

    Output is written every ``output_interval_steps`` model steps (default: every step),
    so the schedule is independent of the time step length.

    ``process_props`` is currently unused: IO is single-node only. It is kept on the
    signature so the distributed path (per-rank IO setup) can be wired in without a
    signature change.
    """
    del process_props  # reserved for the distributed IO path; unused while single-node
    output_variables = DEFAULT_OUTPUT_VARIABLES if variables is None else variables

    io_output_path = output_path / OUTPUT_SUBDIR
    field_groups = [
        common_io.FieldGroupIOConfig(
            output_interval_steps=output_interval_steps,
            filename=DEFAULT_OUTPUT_FILENAME,
            variables=output_variables,
            nc_title="ICON4Py output",
            nc_comment="Fields computed by ICON4Py.",
        )
    ]

    config = common_io.IOConfig(output_path=str(io_output_path), field_groups=field_groups)
    return common_io.IOMonitor(
        config=config,
        vertical_size=vertical_grid,
        horizontal_size=grid.config.horizontal_config,
        grid_file_name=grid_file_path,
        # Grid.id holds the file's `uuidOfHGrid` as a string; the IO layer wants a UUID.
        grid_id=uuid.UUID(grid.id),
    )
