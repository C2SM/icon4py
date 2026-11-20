# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.muphys import data as muphys_data
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import SPECIES, Q
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver.run_full_muphys import (
    setup_muphys,
)
from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    model_options,
    time,
    type_alias as ta,
)
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.math.stencils import generic_math_operations
from icon4py.model.common.physics.thermodynamics import compute_tendencies


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid
    from icon4py.model.common.states import model


class MuphysComponent:
    """Presents the muphys microphysics granule as a model component.

    The granule and the component contract disagree in two ways, and this class
    is where both are settled:

    * The granule **overwrites the fields it is given** (its ``t_out``/``q_out``
      arguments are the same buffers as its ``te``/``q_in`` inputs), whereas a
      component has to report *tendencies*. So ``__call__`` runs it on private
      copies and derives ``(new - old) / dt``, leaving the caller's fields alone.
    * The granule also produces precipitation diagnostics, as (cell, level)
      fields of which only the lowest level, the flux reaching the ground, is
      bound. They are not tendencies and are never applied to the model state.

    Every buffer the granule writes into is allocated once in ``__init__`` and
    reused, so a timestep allocates nothing. The surface precipation buffers
    are the one exception: a caller can swap in its own via ``bind_output_buffers``.
    """

    # TODO (Yilu): inherit the Component protocol once it is formalized (deferred to a separate PR).
    inputs_properties = muphys_data.INPUTS_PROPERTIES
    outputs_properties = muphys_data.OUTPUTS_PROPERTIES

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        dtime: time.RelativeTime,
        qnc: float,
        backend: gtx_typing.Backend | None = None,
        *,
        step: Callable[..., Any] | None = None,
    ) -> None:
        self._ncells = grid.num_cells
        self._nlev = grid.num_levels
        self._dt_seconds = ta.wpfloat(dtime.total_seconds())
        self._qnc = qnc
        self._backend = model_options.customize_backend(program=None, backend=backend)

        full_horizontal_sizes = {
            "horizontal_start": gtx.int32(0),
            "horizontal_end": gtx.int32(self._ncells),
        }

        cell_domain = h_grid.domain(dims.CellDim)
        cell_start = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        cell_end = grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        # Tendencies only on the prognostic subdomain -- the tendency buffers stay zero outside

        prognostic_horizontal_sizes = {
            "horizontal_start": cell_start,
            "horizontal_end": cell_end,
        }
        vertical_sizes = {"vertical_start": gtx.int32(0), "vertical_end": gtx.int32(self._nlev)}

        self._calculate_tendency = model_options.setup_program(
            program=compute_tendencies.compute_cell_kdim_field_tendency,
            backend=self._backend,
            horizontal_sizes=prognostic_horizontal_sizes,
            vertical_sizes=vertical_sizes,
            offset_provider={},
        )
        self._copy_field = model_options.setup_program(
            program=generic_math_operations.copy_field_on_cell_k,
            backend=self._backend,
            horizontal_sizes=full_horizontal_sizes,
            vertical_sizes=vertical_sizes,
            offset_provider={},
        )

        allocator = model_backends.get_allocator(backend)

        if step is None:
            step = setup_muphys(
                ncells=self._ncells,
                nlev=self._nlev,
                dt=self._dt_seconds,
                qnc=qnc,
                backend=backend,
                single_program=False,
            )
        self._step = step

        cell_k_domain = gtx.domain({dims.CellDim: self._ncells, dims.KDim: self._nlev})
        self._allocator = allocator
        self._precip_3d: dict[str, fa.CellKField[ta.wpfloat]] = {
            port: gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator)
            for port in muphys_data.PRECIP_PORTS
        }
        # TODO (Chia Rui): Outer scope only sees this surface precipitation flux. This is a workaround before either gt4py does not return a CellField from the scan operator or io supported level selection
        self._precip_surface: dict[str, fa.CellField[ta.wpfloat]] | None = None

        self._tendencies: dict[str, fa.CellKField[ta.wpfloat]] = {
            "tend_temperature": gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            "tend_qv": gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            "tend_qc": gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            "tend_qr": gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            "tend_qs": gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            "tend_qi": gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            "tend_qg": gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
        }

        self._te_in = gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator)
        self._q_in = Q(
            v=gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            c=gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            r=gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            s=gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            i=gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            g=gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
        )

    def __call__(
        self, state: dict[str, model.DataField], time_step: time.AbsoluteTime
    ) -> dict[str, model.DataField]:
        """Run the granule on private copies of the inputs and report tendencies.

        Three steps. Copy the caller's temperature and tracers into our own
        buffers, because the granule overwrites whatever it is handed and the
        original values are still needed afterwards. Run the granule, which
        updates those copies in place and fills the precip diagnostics. Then
        difference old against new to get ``(new - old) / dt`` per field.

        The caller's fields are never written to. Layer thickness, pressure and
        density go straight in without a copy, since the granule only reads them.

        Returns the tendencies together with the precip diagnostics at the surface:
        the granule writes the precip fluxes as (cell, level) full 3D fields. whose lowest
        level is the flux reaching the ground binded to via 
        ``bind_output_buffers`` those diagnostic entries are the caller's own buffers, filled in place.
        """
        fields = cast("dict[str, fa.CellKField[ta.wpfloat]]", state)

        self._copy_field(field=fields["te"], output_field=self._te_in)
        for s in SPECIES:
            self._copy_field(field=fields[f"q{s}"], output_field=getattr(self._q_in, s))

        self._step(
            dz=fields["dz"],
            te=self._te_in,
            p=fields["p"],
            rho=fields["rho"],
            q_in=self._q_in,
            q_out=self._q_in,
            t_out=self._te_in,
            pflx=self._precip_3d["pflx"],
            pr=self._precip_3d["pr"],
            ps=self._precip_3d["ps"],
            pi=self._precip_3d["pi"],
            pg=self._precip_3d["pg"],
            pre=self._precip_3d["pre"],
        )

        self._calculate_tendency(
            dtime=self._dt_seconds,
            old_field=fields["te"],
            new_field=self._te_in,
            tendency=self._tendencies["tend_temperature"],
        )
        for s in SPECIES:
            self._calculate_tendency(
                dtime=self._dt_seconds,
                old_field=fields[f"q{s}"],
                new_field=getattr(self._q_in, s),
                tendency=self._tendencies[f"tend_q{s}"],
            )

        precip_surface = self._precip_surface_buffers()
        for port in muphys_data.PRECIP_PORTS:
            precip_surface[port].ndarray[...] = self._precip_3d[port].ndarray[:, self._nlev - 1]  # type: ignore[index] # NDArrayObject Protocol doesn't support this

        return cast(
            "dict[str, model.DataField]",
            {**self._tendencies, **precip_surface},
        )

    def _precip_surface_buffers(self) -> dict[str, fa.CellField[ta.wpfloat]]:
        """The surface precip outputs: the caller's if bound, else our own, allocated on first use.

        Allocating on demand rather than in ``__init__`` keeps the driver path
        from ever creating buffers that ``bind_output_buffers`` replaces.
        """
        if self._precip_surface is None:
            cell_domain = gtx.domain({dims.CellDim: self._ncells})
            self._precip_surface = {
                port: gtx.zeros(cell_domain, dtype=ta.wpfloat, allocator=self._allocator)
                for port in muphys_data.PRECIP_PORTS
            }
        return self._precip_surface

    def bind_output_buffers(self, buffers: dict[str, fa.CellField[ta.wpfloat]]) -> None:
        """Redirect the surface precip diagnostics into buffers the caller owns.

        ``__call__`` copies the lowest level of the granule's precip fields into
        these on every step, so binding them once here redirects every write that
        follows.

        ``PhysicsDriver`` calls this once per process while it is being built,
        passing the buffers its ``PhysicsState`` owns. If nobody calls it the
        component allocates its own on the first step, which is what lets it run
        on its own in the granule datatests.
        """
        self._precip_surface = {port: buffers[port] for port in muphys_data.PRECIP_PORTS}
