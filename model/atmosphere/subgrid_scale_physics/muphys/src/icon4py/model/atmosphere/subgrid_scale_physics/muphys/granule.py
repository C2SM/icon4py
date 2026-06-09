# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import datetime
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import Q
from icon4py.model.common import dimension as dims
from icon4py.model.common.diagnostic_calculations.stencils import calculate_tendency
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common import field_type_aliases as fa, type_alias as ta
    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.common.states import model

_SPECIES = ("v", "c", "r", "s", "i", "g")

_INPUTS_PROPERTIES: dict[str, model.FieldMetaData] = {
    "dz": {"standard_name": "layer_thickness", "units": "m"},
    "te": {"standard_name": "air_temperature", "units": "K"},
    "p": {"standard_name": "air_pressure", "units": "Pa"},
    "rho": {"standard_name": "air_density", "units": "kg m-3"},
    "qv": {"standard_name": "specific_humidity", "units": "kg kg-1"},
    "qc": {"standard_name": "specific_cloud_water_content", "units": "kg kg-1"},
    "qr": {"standard_name": "specific_rain_content", "units": "kg kg-1"},
    "qs": {"standard_name": "specific_snow_content", "units": "kg kg-1"},
    "qi": {"standard_name": "specific_cloud_ice_content", "units": "kg kg-1"},
    "qg": {"standard_name": "specific_graupel_content", "units": "kg kg-1"},
}

_OUTPUTS_PROPERTIES: dict[str, model.FieldMetaData] = {
    "tend_temperature": {
        "standard_name": "tendency_of_air_temperature",
        "units": "K s-1",
        "kind": "tendency",
    },
    "tend_qv": {
        "standard_name": "tendency_of_specific_humidity",
        "units": "kg kg-1 s-1",
        "kind": "tendency",
    },
    "tend_qc": {
        "standard_name": "tendency_of_specific_cloud_water_content",
        "units": "kg kg-1 s-1",
        "kind": "tendency",
    },
    "tend_qr": {
        "standard_name": "tendency_of_specific_rain_content",
        "units": "kg kg-1 s-1",
        "kind": "tendency",
    },
    "tend_qs": {
        "standard_name": "tendency_of_specific_snow_content",
        "units": "kg kg-1 s-1",
        "kind": "tendency",
    },
    "tend_qi": {
        "standard_name": "tendency_of_specific_cloud_ice_content",
        "units": "kg kg-1 s-1",
        "kind": "tendency",
    },
    "tend_qg": {
        "standard_name": "tendency_of_specific_graupel_content",
        "units": "kg kg-1 s-1",
        "kind": "tendency",
    },
    "pflx": {"standard_name": "precipitation_flux", "units": "kg m-2 s-1", "kind": "diagnostic"},
    "pr": {"standard_name": "rainfall_flux", "units": "kg m-2 s-1", "kind": "diagnostic"},
    "ps": {"standard_name": "snowfall_flux", "units": "kg m-2 s-1", "kind": "diagnostic"},
    "pi": {"standard_name": "ice_fall_flux", "units": "kg m-2 s-1", "kind": "diagnostic"},
    "pg": {"standard_name": "graupel_fall_flux", "units": "kg m-2 s-1", "kind": "diagnostic"},
    "pre": {
        "standard_name": "precipitation_energy_flux",
        "units": "W m-2",
        "kind": "diagnostic",
    },
}


class MuphysGranule:
    """L4 per-process adapter wrapping the muphys microphysics program."""

    inputs_properties = _INPUTS_PROPERTIES
    outputs_properties = _OUTPUTS_PROPERTIES

    def __init__(
        self,
        grid: base_grid.Grid,
        dt: float,
        qnc: float,
        backend: gtx_typing.Backend | None = None,
        *,
        muphys_step: Callable[..., Any] | None = None,
    ) -> None:
        self._grid = grid
        self._dt = dt
        self._qnc = qnc
        self._backend = backend
        if muphys_step is None:
            # TODO(Yilu): run_full_muphys.setup_muphys(single_program=False) from dt/qnc/grid.
            raise NotImplementedError(
                "MuphysGranule needs an explicit `muphys_step` until scope-4 wires the "
                "muphys setup; inject one for now."
            )
        self._muphys_step = muphys_step

        # output fields should distinct from input fields to avoid in-place updates, since we will need to calculate the tendencies later
        self._t_out = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self._q_out = Q(
            v=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend),
            c=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend),
            r=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend),
            s=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend),
            i=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend),
            g=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend),
        )
        self._pflx = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self._pr = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self._ps = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self._pi = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self._pg = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self._pre = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        # tendency buffers, one per returned tendency.
        self._tendencies: dict[str, fa.CellKField[ta.wpfloat]] = {
            "tend_temperature": data_alloc.zero_field(
                grid, dims.CellDim, dims.KDim, allocator=backend
            ),
            "tend_qv": data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend),
            "tend_qc": data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend),
            "tend_qr": data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend),
            "tend_qs": data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend),
            "tend_qi": data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend),
            "tend_qg": data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend),
        }

    # function to convert muphy's updated-state output into the tendencies
    def _to_tendency(
        self,
        old: fa.CellKField[ta.wpfloat],
        new: fa.CellKField[ta.wpfloat],
        out: fa.CellKField[ta.wpfloat],
    ) -> None:
        """``out = (new - old) / dt`` over the whole column."""
        program = calculate_tendency.calculate_cell_kdim_field_tendency
        if self._backend is not None:
            program = program.with_backend(self._backend)
        program(
            dtime=self._dt,
            old_field=old,
            new_field=new,
            tendency=out,
            horizontal_start=0,
            horizontal_end=self._grid.num_cells,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider={},
        )

    def __call__(
        self, state: dict[str, model.DataField], time_step: datetime.datetime
    ) -> dict[str, model.DataField]:
        """Run muphys, then convert its updated state into tendencies.

        muphys returns updated state (t_out, q_out); this boundary converts it
        to tendencies ``(new - old) / dt`` s. Precip outputs are diagnostics, passed straight through.
        """

        q_in = Q(
            v=state["qv"],
            c=state["qc"],
            r=state["qr"],
            s=state["qs"],
            i=state["qi"],
            g=state["qg"],
        )

        # TODO (Yilu): currently this is a fake step. later on this will be a real call to muphys
        self._muphys_step(
            dz=state["dz"],
            te=state["te"],
            p=state["p"],
            rho=state["rho"],
            q_in=q_in,
            q_out=self._q_out,
            t_out=self._t_out,
            pflx=self._pflx,
            pr=self._pr,
            ps=self._ps,
            pi=self._pi,
            pg=self._pg,
            pre=self._pre,
        )

        self._to_tendency(state["te"], self._t_out, self._tendencies["tend_temperature"])
        for s in _SPECIES:
            self._to_tendency(
                state[f"q{s}"], getattr(self._q_out, s), self._tendencies[f"tend_q{s}"]
            )

        return {
            **self._tendencies,
            "pflx": self._pflx,
            "pr": self._pr,
            "ps": self._ps,
            "pi": self._pi,
            "pg": self._pg,
            "pre": self._pre,
        }
