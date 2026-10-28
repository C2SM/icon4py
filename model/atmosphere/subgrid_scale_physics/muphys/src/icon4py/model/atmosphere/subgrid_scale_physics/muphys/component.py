# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import SPECIES, Q
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver.run_full_muphys import (
    setup_muphys,
)
from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    framework as fw,
    model_backends,
    model_options,
    time,
    type_alias as ta,
)
from icon4py.model.common.framework import quantities as qty
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.math.stencils import generic_math_operations
from icon4py.model.common.physics.thermodynamics import compute_tendencies
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    # a type-only import: muphys does not depend on the physics driver
    from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
    from icon4py.model.common.grid import base as base_grid


class Muphys(fw.Component["Muphys.Input", "Muphys.Output"]):
    """
    The muphys microphysics granule as a component.

    The granule and the component disagree in two ways, and this class settles both:

    * The granule overwrites the fields it is given (its `t_out`/`q_out` arguments are the
      same buffers as its `te`/`q_in` inputs), whereas the component reports tendencies. So
      `run` runs it on private copies and derives `(new - old) / dt`, leaving the inputs alone.
    * The granule also produces precipitation fluxes. Those are not tendencies and are never
      applied to the model state; they are reported as they come.

    The tendencies are computed on the prognostic subdomain only: outside it the output
    buffers keep what they hold.
    """

    class Input(fw.State):
        te: fw.Field[qty.TemperatureOnCellK]
        p: fw.Field[qty.PressureOnCellK]
        rho: fw.Field[qty.RhoOnCellK]
        qv: fw.Field[qty.QvOnCellK]
        qc: fw.Field[qty.QcOnCellK]
        qi: fw.Field[qty.QiOnCellK]
        qr: fw.Field[qty.QrOnCellK]
        qs: fw.Field[qty.QsOnCellK]
        qg: fw.Field[qty.QgOnCellK]

    class Output(fw.State):
        tend_temperature: fw.Field[qty.TendencyOfTemperatureOnCellK]
        tend_qv: fw.Field[qty.TendencyOfQvOnCellK]
        tend_qc: fw.Field[qty.TendencyOfQcOnCellK]
        tend_qi: fw.Field[qty.TendencyOfQiOnCellK]
        tend_qr: fw.Field[qty.TendencyOfQrOnCellK]
        tend_qs: fw.Field[qty.TendencyOfQsOnCellK]
        tend_qg: fw.Field[qty.TendencyOfQgOnCellK]
        pflx: fw.Field[qty.PrecipitationFluxOnCellK]
        pr: fw.Field[qty.RainfallFluxOnCellK]
        ps: fw.Field[qty.SnowfallFluxOnCellK]
        pi: fw.Field[qty.IcefallFluxOnCellK]
        pg: fw.Field[qty.GraupelfallFluxOnCellK]
        pre: fw.Field[qty.PrecipitationEnergyFluxOnCellK]

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        dtime: time.RelativeTime,
        qnc: float,
        dz: fa.CellKField[ta.wpfloat],
        backend: model_backends.BackendLike = None,
        compute: Callable[..., Any] | None = None,
    ) -> None:
        allocator = model_backends.get_allocator(backend)
        self._dt_seconds = ta.wpfloat(dtime.total_seconds())
        self._dz = dz
        program_backend = model_options.customize_backend(program=None, backend=backend)

        cell_domain = h_grid.domain(dims.CellDim)
        # the tendencies only on the prognostic subdomain
        prognostic_cells = {
            "horizontal_start": grid.start_index(cell_domain(h_grid.Zone.NUDGING)),
            "horizontal_end": grid.end_index(cell_domain(h_grid.Zone.LOCAL)),
        }
        cells = {"horizontal_start": gtx.int32(0), "horizontal_end": gtx.int32(grid.num_cells)}
        levels = {"vertical_start": gtx.int32(0), "vertical_end": gtx.int32(grid.num_levels)}

        self._calculate_tendency = model_options.setup_program(
            program=compute_tendencies.compute_cell_kdim_field_tendency,
            backend=program_backend,
            horizontal_sizes=prognostic_cells,
            vertical_sizes=levels,
            offset_provider={},
        )
        self._copy_field = model_options.setup_program(
            program=generic_math_operations.copy_field_on_cell_k,
            backend=program_backend,
            horizontal_sizes=cells,
            vertical_sizes=levels,
            offset_provider={},
        )
        if compute is None:
            compute = setup_muphys(
                ncells=grid.num_cells,
                nlev=grid.num_levels,
                dt=self._dt_seconds,
                qnc=qnc,
                backend=backend,
                single_program=False,
            )
        self._compute = compute

        # the private copies the granule overwrites
        self._te = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator)
        self._q = Q(
            *(
                data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator)
                for _ in SPECIES
            )
        )

    def run(self, inputs: Input, outputs: Output) -> Output:
        """
        Run the granule on private copies of the inputs and report the tendencies.

        The temperature and the tracers are copied, because the granule overwrites whatever it
        is handed and the old values are still needed afterwards; the layer thickness, pressure
        and density go in as they are, since the granule only reads them.
        """
        self._copy_field(field=inputs.te, output_field=self._te)
        for s in SPECIES:
            self._copy_field(field=getattr(inputs, f"q{s}"), output_field=getattr(self._q, s))

        self._compute(
            dz=self._dz,
            te=self._te,
            p=inputs.p,
            rho=inputs.rho,
            q_in=self._q,
            q_out=self._q,
            t_out=self._te,
            pflx=outputs.pflx,
            pr=outputs.pr,
            ps=outputs.ps,
            pi=outputs.pi,
            pg=outputs.pg,
            pre=outputs.pre,
        )

        self._calculate_tendency(
            dtime=self._dt_seconds,
            old_field=inputs.te,
            new_field=self._te,
            tendency=outputs.tend_temperature,
        )
        for s in SPECIES:
            self._calculate_tendency(
                dtime=self._dt_seconds,
                old_field=getattr(inputs, f"q{s}"),
                new_field=getattr(self._q, s),
                tendency=getattr(outputs, f"tend_q{s}"),
            )
        return outputs


# NOTE: temporary, components MWE part 02 (collect): replaced by `fw.collect(Muphys.Input, ...)`,
# which picks the leaves by quantity.
def collect_input(entry: physics_state.EntryState) -> Muphys.Input:
    """The muphys input from the physics entry state (no copies)."""
    return Muphys.Input(
        te=entry.temperature,
        p=entry.pressure,
        rho=entry.rho,
        qv=entry.qv,
        qc=entry.qc,
        qi=entry.qi,
        qr=entry.qr,
        qs=entry.qs,
        qg=entry.qg,
    )
