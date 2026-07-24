# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.muphys import (
    config as muphys_config,
    data as muphys_data,
)
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
from icon4py.model.common.diagnostic_calculations.stencils import calculate_tendency
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.math.stencils import generic_math_operations


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid
    from icon4py.model.common.states import model


class MuphysComponent:
    """Per-process adapter wrapping the muphys microphysics program."""

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
        scheme: muphys_config.MuphysScheme = muphys_config.MuphysScheme.KOKKOS_MUPHYS,
        step: Callable[..., Any] | None = None,
    ) -> None:
        self._ncells = grid.num_cells
        self._nlev = grid.num_levels
        self._dt_seconds = dtime.total_seconds()
        self._qnc = qnc
        self._backend = model_options.customize_backend(program=None, backend=backend)

        cell_domain = h_grid.domain(dims.CellDim)
        # ICON applies physics on the prognostic cells only:
        # grf_bdywidth_c+1 .. min_rlcell_int (NUDGING start .. LOCAL end).
        cell_start = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        cell_end = grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        # Inputs are copied over the full range: the muphys step reads whole fields.
        full_horizontal_sizes = {
            "horizontal_start": gtx.int32(0),
            "horizontal_end": gtx.int32(self._ncells),
        }
        # Tendencies only on the prognostic subdomain -- the tendency buffers stay
        # zero outside, so scattering is a no-op on boundary/halo rows, as in ICON.
        prognostic_horizontal_sizes = {
            "horizontal_start": cell_start,
            "horizontal_end": cell_end,
        }
        vertical_sizes = {"vertical_start": gtx.int32(0), "vertical_end": gtx.int32(self._nlev)}
        self._calculate_tendency = model_options.setup_program(
            program=calculate_tendency.calculate_cell_kdim_field_tendency,
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
            sizes = types.SimpleNamespace(ncells=self._ncells, nlev=self._nlev)
            step = setup_muphys(
                inp=sizes,  # type: ignore[arg-type]  # only .ncells/.nlev are read
                dt=self._dt_seconds,
                qnc=qnc,
                backend=backend,
                single_program=False,
                scheme=scheme,
            )
        self._step = step

        cell_k_domain = gtx.domain({dims.CellDim: self._ncells, dims.KDim: self._nlev})
        self._t_out = gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator)
        self._q_out = Q(
            v=gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            c=gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            r=gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            s=gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            i=gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            g=gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
        )
        self._pflx = gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator)
        self._pr = gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator)
        self._ps = gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator)
        self._pi = gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator)
        self._pg = gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator)
        self._pre = gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator)

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
        """Run muphys, then convert its updated state into tendencies.

        muphys returns updated state (t_out, q_out); this boundary converts it
        to tendencies ``(new - old) / dt`` s. Precip outputs are diagnostics, passed straight through.
        """
        # cast from generic ``DataField`` to bare gt4py fields
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
            q_out=self._q_out,
            t_out=self._t_out,
            pflx=self._pflx,
            pr=self._pr,
            ps=self._ps,
            pi=self._pi,
            pg=self._pg,
            pre=self._pre,
        )

        self._calculate_tendency(
            dtime=self._dt_seconds,
            old_field=fields["te"],
            new_field=self._t_out,
            tendency=self._tendencies["tend_temperature"],
        )
        for s in SPECIES:
            self._calculate_tendency(
                dtime=self._dt_seconds,
                old_field=fields[f"q{s}"],
                new_field=getattr(self._q_out, s),
                tendency=self._tendencies[f"tend_q{s}"],
            )

        return cast(
            "dict[str, model.DataField]",
            {
                **self._tendencies,
                "pflx": self._pflx,
                "pr": self._pr,
                "ps": self._ps,
                "pi": self._pi,
                "pg": self._pg,
                "pre": self._pre,
            },
        )
