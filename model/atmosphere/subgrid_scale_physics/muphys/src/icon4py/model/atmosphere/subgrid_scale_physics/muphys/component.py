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
    component,
    dimension as dims,
    field,
    field_type_aliases as fa,
    model_backends,
    model_options,
    quantities as qty,
    state,
    time,
    type_alias as ta,
)
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.math.stencils import generic_math_operations
from icon4py.model.common.physics.thermodynamics import compute_tendencies
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    # a type-only import: muphys does not depend on the physics driver
    from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
    from icon4py.model.common.grid import base as base_grid


class Muphys(component.Component["Muphys.Input", "Muphys.Output"]):
    """
    The Component interface to the muphys microphysics.
    """

    class Input(state.State):
        te: field.Field[qty.TemperatureOnCellK]
        p: field.Field[qty.PressureOnCellK]
        rho: field.Field[qty.RhoOnCellK]
        qv: field.Field[qty.QvOnCellK]
        qc: field.Field[qty.QcOnCellK]
        qi: field.Field[qty.QiOnCellK]
        qr: field.Field[qty.QrOnCellK]
        qs: field.Field[qty.QsOnCellK]
        qg: field.Field[qty.QgOnCellK]

    class Output(state.State):
        tend_temperature: field.Field[qty.TendencyOfTemperatureOnCellK]
        tend_qv: field.Field[qty.TendencyOfQvOnCellK]
        tend_qc: field.Field[qty.TendencyOfQcOnCellK]
        tend_qi: field.Field[qty.TendencyOfQiOnCellK]
        tend_qr: field.Field[qty.TendencyOfQrOnCellK]
        tend_qs: field.Field[qty.TendencyOfQsOnCellK]
        tend_qg: field.Field[qty.TendencyOfQgOnCellK]
        pflx: field.Field[qty.PrecipitationFluxOnCellK]
        pr: field.Field[qty.RainfallFluxOnCell]
        ps: field.Field[qty.SnowfallFluxOnCell]
        pi: field.Field[qty.IcefallFluxOnCell]
        pg: field.Field[qty.GraupelfallFluxOnCell]
        pre: field.Field[qty.PrecipitationEnergyFluxOnCell]

    class Full3DPrecip(state.State):
        """The granule's precip fluxes on all levels, of which only the lowest is an output."""

        pr: field.Field[qty.RainfallFluxOnCellK]
        ps: field.Field[qty.SnowfallFluxOnCellK]
        pi: field.Field[qty.IcefallFluxOnCellK]
        pg: field.Field[qty.GraupelfallFluxOnCellK]
        pre: field.Field[qty.PrecipitationEnergyFluxOnCellK]

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        dtime: time.RelativeTime,
        qnc: float,
        dz: fa.CellKField[ta.wpfloat],
        backend_like: model_backends.BackendLike = None,
        compute: Callable[..., Any] | None = None,
    ) -> None:
        allocator = model_backends.get_allocator(backend_like)
        self._dt_seconds = ta.wpfloat(dtime.total_seconds())
        self._dz = dz
        backend = model_options.customize_backend(program=None, backend=backend_like)

        cell_domain = h_grid.domain(dims.CellDim)
        # the tendencies only on the prognostic subdomain
        nudging_to_local_cell_range = {
            "horizontal_start": grid.start_index(cell_domain(h_grid.Zone.NUDGING)),
            "horizontal_end": grid.end_index(cell_domain(h_grid.Zone.LOCAL)),
        }
        cell_range = {"horizontal_start": gtx.int32(0), "horizontal_end": gtx.int32(grid.num_cells)}
        k_range = {"vertical_start": gtx.int32(0), "vertical_end": gtx.int32(grid.num_levels)}

        self._calculate_tendency = model_options.setup_program(
            program=compute_tendencies.compute_cell_kdim_field_tendency,
            backend=backend,
            horizontal_sizes=nudging_to_local_cell_range,
            vertical_sizes=k_range,
            offset_provider={},
        )
        self._copy_field = model_options.setup_program(
            program=generic_math_operations.copy_field_on_cell_k,
            backend=backend,
            horizontal_sizes=cell_range,
            vertical_sizes=k_range,
            offset_provider={},
        )
        if compute is None:
            compute = setup_muphys(
                ncells=grid.num_cells,
                nlev=grid.num_levels,
                dt=self._dt_seconds,
                qnc=qnc,
                backend=backend_like,
                single_program=False,
            )
        self._compute = compute

        # The granule writes the precip fluxes on all levels; the outputs are their lowest
        # level, the flux reaching the ground.
        # TODO (Chia Rui): a workaround until either the gt4py scan operator can return a
        #   CellField or the io supports level selection
        self._precip_3d = state.allocate(self.Full3DPrecip, grid, allocator)

        # the private copies the granule overwrites
        self._te = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator)
        self._q = Q(
            *(
                data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator)
                for _ in SPECIES
            )
        )

    def run(self, input: Input, output: Output) -> Output:
        """
        Run the granule and return the tendencies and the precip diagnostics.

        The temperature and the tracers are copied because muphys overwrites
        them and the original values are needed afterwards. The layer
        thickness, pressure and density go in as they are, since muphys only
        reads them.

        The total precipitation flux `pflx` is returned on all levels; the rain, snow,
        ice, graupel and energy fluxes only at the surface (their lowest level).
        """
        self._copy_field(field=input.te, output_field=self._te)
        for s in SPECIES:
            self._copy_field(field=getattr(input, f"q{s}"), output_field=getattr(self._q, s))

        self._compute(
            dz=self._dz,
            te=self._te,
            p=input.p,
            rho=input.rho,
            q_in=self._q,
            q_out=self._q,
            t_out=self._te,
            pflx=output.pflx,
            pr=self._precip_3d.pr,
            ps=self._precip_3d.ps,
            pi=self._precip_3d.pi,
            pg=self._precip_3d.pg,
            pre=self._precip_3d.pre,
        )

        self._calculate_tendency(
            dtime=self._dt_seconds,
            old_field=input.te,
            new_field=self._te,
            tendency=output.tend_temperature,
        )
        for s in SPECIES:
            self._calculate_tendency(
                dtime=self._dt_seconds,
                old_field=getattr(input, f"q{s}"),
                new_field=getattr(self._q, s),
                tendency=getattr(output, f"tend_q{s}"),
            )

        for name, precip in state.fields(self._precip_3d).items():
            getattr(output, name).ndarray[...] = precip.ndarray[:, -1]
        return output


# NOTE: temporary, replaced in part 02 (collect): `fw.collect(Muphys.Input, ...)`,
# which picks the leaves by quantity.
def collect_input(state: physics_state.PhysicsState) -> Muphys.Input:
    """The muphys input from the physics physics state (no copies)."""
    return Muphys.Input(
        te=state.temperature,
        p=state.pressure,
        rho=state.rho,
        qv=state.qv,
        qc=state.qc,
        qi=state.qi,
        qr=state.qr,
        qs=state.qs,
        qg=state.qg,
    )
