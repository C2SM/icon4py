# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
The dycore is forced with two sets of tendencies that come from parts of ICON which
ICON4Py does not have:

  - grf_tend_rho, grf_tend_thv, grf_tend_w and grf_tend_vn drive the lateral boundary
    of a limited area domain. ICON computes them as the time difference of two
    consecutive lateral boundary data slices (compute_boundary_tendencies in
    mo_async_latbc_utils.f90) and refreshes them whenever a new slice is read.
  - ddt_exner_phy and ddt_vn_phy are the tendencies of the slow physics (radiation,
    convection, sub-grid scale orography, gravity wave drag), computed once per time
    step by the NWP interface (mo_nh_interface_nwp.f90) and held constant over the
    dynamics substeps.

Without a lateral boundary reader and without physics, both sets can only be
prescribed from the serialized data of the reference run, once per time step.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import TYPE_CHECKING, Any

import gt4py.next as gtx
import serialbox  # type: ignore[import-untyped]

from icon4py.model.common import model_backends, time
from icon4py.model.common.states import nonhydro_states
from icon4py.model.common.utils import data_allocation as data_alloc, fortran_config


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid


log = logging.getLogger(__name__)


@dataclasses.dataclass
class PrescribedTendenciesConfig:
    #: None for the testcases, which are neither limited area nor forced by physics.
    data_path: pathlib.Path | None = None

    @classmethod
    def from_fortran_dict(
        cls,
        *,
        atm_dict: dict[str, Any],
        data_path: pathlib.Path,
    ) -> PrescribedTendenciesConfig:
        run_nml = atm_dict["run_nml"]
        if run_nml["ltestcase"]:
            return cls(data_path=None)
        return cls(data_path=data_path / fortran_config.SER_DATA_SUBDIR)


class SerializedTendencies:

    def __init__(
        self,
        *,
        data_path: pathlib.Path,
        grid: icon_grid.IconGrid,
        backend: gtx_typing.Backend | None,
        rank: int,
    ) -> None:
        self._grid = grid
        self._array_ns = data_alloc.import_array_ns(model_backends.get_allocator(backend))
        self._serializer = serialbox.Serializer(
            serialbox.OpenModeKind.Read, str(data_path), f"icon_pydycore_rank{rank}"
        )

    def update(
        self,
        *,
        diagnostic_state_nh: nonhydro_states.DiagnosticStateNonHydro,
        at_datetime: time.AbsoluteTime,
    ) -> None:
        """
        Read the tendencies of the time step ending at 'at_datetime'.

        The savepoints of a time step are stamped with the date of its end, and the
        tendencies are the same for all its substeps, so the ones of the first predictor
        step are read.
        """
        date = at_datetime.replace(tzinfo=None).isoformat(timespec="milliseconds")
        try:
            savepoint = (
                self._serializer.savepoint["solve-nonhydro-init"]
                .istep[1]
                .date[date]
                .dyn_timestep[1]
                .as_savepoint()
            )
        except serialbox.SerialboxError as err:
            raise ValueError(
                f"there are no serialized tendencies for the time step ending at {date}. "
                f"The serialized data of this experiment covers: {self._available_nonhydro_init_dates()}. "
                "The lateral boundary and slow physics tendencies of a real data run can "
                "only be prescribed within that window."
            ) from err

        log.debug("Reading the prescribed tendencies of %s", date)

        num_cells = self._grid.num_cells
        num_edges = self._grid.num_edges

        # lateral boundary tendencies
        self._fill(diagnostic_state_nh.grf_tend_rho, "grf_tend_rho", savepoint, num_cells)
        self._fill(diagnostic_state_nh.grf_tend_thv, "grf_tend_thv", savepoint, num_cells)
        self._fill(diagnostic_state_nh.grf_tend_w, "grf_tend_w", savepoint, num_cells)
        self._fill(diagnostic_state_nh.grf_tend_vn, "grf_tend_vn", savepoint, num_edges)
        # slow physics tendencies
        self._fill(
            diagnostic_state_nh.exner_tendency_due_to_slow_physics,
            "ddt_exner_phy",
            savepoint,
            num_cells,
        )
        self._fill(
            diagnostic_state_nh.normal_wind_tendency_due_to_slow_physics_process,
            "ddt_vn_phy",
            savepoint,
            num_edges,
        )

    def _fill(self, field: gtx.Field, name: str, savepoint: serialbox.Savepoint, size: int) -> None:
        # the serialized fields are padded to nproma
        buffer = self._array_ns.squeeze(self._serializer.read(name, savepoint).astype(float))
        field.ndarray[:, :] = self._array_ns.asarray(  # type: ignore[index] # NDArrayObject Protocol doesn't support this
            buffer[:size, :]
        )

    def _available_nonhydro_init_dates(self) -> str:
        dates = sorted(
            {
                savepoint.metainfo["date"]
                for savepoint in self._serializer.savepoint_list()
                if savepoint.name == "solve-nonhydro-init"
            }
        )
        return ", ".join(dates) if dates else "none"
