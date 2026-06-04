# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import logging
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import serialbox  # serialbox4py

from icon4py.model.common import type_alias as ta
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
)
from icon4py.model.common.utils import data_allocation as data_alloc, fortran_config
from icon4py.model.standalone_driver.testcases import (
    gauss3d,
    jablonowski_williamson as jabw,
    utils as testcases_utils,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import geometry as grid_geometry
    from icon4py.model.standalone_driver import driver_states


log = logging.getLogger(__name__)

_SER_DATA_SUBDIR: Final = "ser_data"
_ICON_PYDYCORE_PREFIX: Final = "icon_pydycore"


def _params_from_dict(cls: type, source: dict[str, Any]):
    """Construct a dataclass from a namelist dict.

    Unknown keys are ignored (e.g. topography params mixed into the same nml block).
    Missing keys fall back to the dataclass field defaults.
    Fortran→Python name translation is driven by the required ``_fortran_name_map``
    class variable: ``{fortran_key: python_field_name}``.
    """
    name_map: dict[str, str] = cls._fortran_name_map  # type: ignore[attr-defined]
    known_fields = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in source.items():
        python_name = name_map.get(key, key)
        if python_name in known_fields:
            kwargs[python_name] = value
    return cls(**kwargs)


@dataclasses.dataclass
class FromFileParameters:
    """Parameters for the file-based initial condition."""

    #: Path to the serialised data directory (typically ``<experiment>/ser_data``).
    data_path: pathlib.Path
    #: Number of tracer species stored in the snapshot (0 means no tracers).
    ntracer: int = 0


def _read_prognostics_from_serialbox(
    *,
    data_path: pathlib.Path,
    rank: int,
    grid: icon_grid.IconGrid,
    backend: gtx_typing.Backend | None,
    ntracer: int,
) -> prognostics.PrognosticState:
    """Read prognostic IC fields directly from a serialbox snapshot.

    Opens the serialbox archive at *data_path* for MPI rank *rank*,
    finds the ``prognostics / initial-state`` savepoint, and fills a
    freshly allocated :class:`~icon4py.model.common.states.prognostic_state.PrognosticState`.

    All array manipulation uses only ``numpy`` and the GT4Py field API;
    there is intentionally no dependency on ``icon4py.model.testing``.
    """
    fname = f"{_ICON_PYDYCORE_PREFIX}_rank{rank}"
    ser = serialbox.Serializer(serialbox.OpenModeKind.Read, str(data_path), fname)
    sp = ser.savepoint["prognostics"].id[1].location["initial-state"].as_savepoint()
    log.debug("Reading prognostics initial-state from %s / %s", data_path, fname)

    nc = grid.num_cells
    ne = grid.num_edges
    xp = data_alloc.import_array_ns(backend)

    def read_cell_k(name: str):
        return xp.asarray(np.squeeze(ser.read(name, sp).astype(float))[:nc, :])

    def read_edge_k(name: str):
        return xp.asarray(np.squeeze(ser.read(name, sp).astype(float))[:ne, :])

    state = prognostics.initialize_prognostic_state(grid=grid, allocator=backend, ntracer=ntracer)
    state.rho.ndarray[:, :] = read_cell_k("rho_now")
    state.exner.ndarray[:, :] = read_cell_k("exner_now")
    state.theta_v.ndarray[:, :] = read_cell_k("theta_v_now")
    state.vn.ndarray[:, :] = read_edge_k("vn_now")
    state.w.ndarray[:, :] = read_cell_k("w_now")  # shape (nc, num_levels + 1)

    if ntracer > 0:
        tracers_raw = np.squeeze(ser.read("tracers_now", sp).astype(float))
        for i in range(ntracer):
            state.tracer[i].ndarray[:, :] = xp.asarray(tracers_raw[:nc, :, i])

    return state


def read_from_file(
    *,
    parameters: FromFileParameters,
    grid: icon_grid.IconGrid,
    geometry_field_source: grid_geometry.GridGeometry,  # unused; kept for API parity
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend: gtx_typing.Backend | None,
    lowest_layer_thickness: ta.wpfloat,  # unused; kept for API parity
    model_top_height: ta.wpfloat,  # unused; kept for API parity
    stretch_factor: ta.wpfloat,  # unused; kept for API parity
    damping_height: ta.wpfloat,  # unused; kept for API parity
    exchange: decomposition_defs.ExchangeRuntime,
) -> driver_states.DriverStates:
    """Initialise prognostic state from a serialised ICON initial-condition snapshot.

    Reads ``prognostics / initial-state`` from the serialbox archive located at
    ``parameters.data_path``.  All other physics arguments are accepted but
    ignored so that this function shares the same call signature as the
    analytical initial-condition functions (``jablonowski_williamson``,
    ``gauss3d``, …) and can be stored transparently in
    :attr:`InitialConditionConfig.create`.
    """
    prognostic_state_now = _read_prognostics_from_serialbox(
        data_path=parameters.data_path,
        rank=exchange.my_rank(),
        grid=grid,
        backend=backend,
        ntracer=parameters.ntracer,
    )
    diagnostic_state = diagnostics.initialize_diagnostic_state(grid=grid, allocator=backend)
    return testcases_utils.assemble_driver_states(
        grid=grid,
        allocator=backend,
        backend=backend,
        exchange=exchange,
        interpolation=testcases_utils.extract_interpolation(interpolation_field_source),
        zone_indices_map=testcases_utils.zone_indices(grid),
        metrics_field_source=metrics_field_source,
        prognostic_state_now=prognostic_state_now,
        diagnostic_state=diagnostic_state,
    )


@dataclasses.dataclass
class InitialConditionConfig:
    parameters: (
        jabw.JablonowskiWilliamsonParameters | gauss3d.Gauss3DParameters | FromFileParameters
    )
    create: Callable[..., driver_states.DriverStates]

    @classmethod
    def from_fortran_dict(
        cls,
        atm_dict: dict[str, Any],
        input_dict: dict[str, Any],
        *,
        data_path: pathlib.Path | None = None,
    ) -> InitialConditionConfig | None:
        run_nml = atm_dict.get("run_nml", {})
        if not run_nml.get("ltestcase", False):
            if data_path is None:
                return None
            ntracer = fortran_config.list_to_value(run_nml.get("ntracer", 0))
            return cls(
                parameters=FromFileParameters(
                    data_path=data_path / _SER_DATA_SUBDIR,
                    ntracer=ntracer,
                ),
                create=read_from_file,
            )

        testcase_nml = input_dict.get("nh_testcase_nml", {})
        match testcase_nml.get("nh_test_name"):
            case "jabw" | "jabw_s":
                parameters = _params_from_dict(jabw.JablonowskiWilliamsonParameters, testcase_nml)
                create = jabw.jablonowski_williamson
            case "gauss3D":
                parameters = _params_from_dict(gauss3d.Gauss3DParameters, testcase_nml)
                create = gauss3d.gauss3d
            case name:
                raise ValueError(f"Unknown or missing test case name: {name!r}")

        return cls(parameters=parameters, create=create)
