# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import datetime
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.muphys import data as muphys_data
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import Q
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver import run_full_muphys
from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    model_options,
    type_alias as ta,
)
from icon4py.model.common.diagnostic_calculations.stencils import calculate_tendency


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.states import model

_SPECIES = ("v", "c", "r", "s", "i", "g")


@gtx.field_operator
def _copy(field: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    return field


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def copy_field(  # noqa: PLR0917  # stencil params referenced in domain specs stay positional
    field: fa.CellKField[ta.wpfloat],
    result: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _copy(
        field,
        out=result,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


class MuphysGranule:
    """L4 per-process adapter wrapping the muphys microphysics program."""

    inputs_properties = muphys_data.INPUTS_PROPERTIES
    outputs_properties = muphys_data.OUTPUTS_PROPERTIES

    def __init__(
        self,
        ncells: int,
        nlev: int,
        dt: float,
        qnc: float,
        backend: gtx_typing.Backend | None = None,
        *,
        muphys_step: Callable[..., Any] | None = None,
    ) -> None:
        self._ncells = ncells
        self._nlev = nlev
        self._dt = dt
        self._qnc = qnc
        self._backend = model_options.customize_backend(None, backend)

        allocator = model_backends.get_allocator(backend)

        if muphys_step is None:
            # TODO (Yilu): what does the comment mean?
            # Build the verified separate-path step (graupel + saturation adjustment).
            # setup_muphys only reads `.ncells`/`.nlev` off `inp`, so a sizes shim suffices.
            sizes = types.SimpleNamespace(ncells=ncells, nlev=nlev)
            muphys_step = run_full_muphys.setup_muphys(
                inp=sizes,  # type: ignore[arg-type]  # only .ncells/.nlev are read
                dt=dt,
                qnc=qnc,
                backend=backend,
                single_program=False,
            )
        self._muphys_step = muphys_step

        domain = gtx.domain({dims.CellDim: ncells, dims.KDim: nlev})
        self._t_out = gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator)
        self._q_out = Q(
            v=gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator),
            c=gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator),
            r=gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator),
            s=gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator),
            i=gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator),
            g=gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator),
        )
        self._pflx = gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator)
        self._pr = gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator)
        self._ps = gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator)
        self._pi = gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator)
        self._pg = gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator)
        self._pre = gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator)

        self._tendencies: dict[str, fa.CellKField[ta.wpfloat]] = {
            "tend_temperature": gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator),
            "tend_qv": gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator),
            "tend_qc": gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator),
            "tend_qr": gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator),
            "tend_qs": gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator),
            "tend_qi": gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator),
            "tend_qg": gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator),
        }

        self._te_in = gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator)
        self._q_in = Q(
            v=gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator),
            c=gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator),
            r=gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator),
            s=gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator),
            i=gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator),
            g=gtx.zeros(domain, dtype=ta.wpfloat, allocator=allocator),
        )

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
            horizontal_end=self._ncells,
            vertical_start=0,
            vertical_end=self._nlev,
            offset_provider={},
        )

    def _copy_into(self, src: fa.CellKField[ta.wpfloat], dst: fa.CellKField[ta.wpfloat]) -> None:
        """Copy ``src`` into the granule-owned buffer ``dst``."""
        program = copy_field
        if self._backend is not None:
            program = program.with_backend(self._backend)
        program(
            field=src,
            result=dst,
            horizontal_start=0,
            horizontal_end=self._ncells,
            vertical_start=0,
            vertical_end=self._nlev,
            offset_provider={},
        )

    def __call__(
        self, state: dict[str, model.DataField], time_step: datetime.datetime
    ) -> dict[str, model.DataField]:
        """Run muphys, then convert its updated state into tendencies.

        muphys returns updated state (t_out, q_out); this boundary converts it
        to tendencies ``(new - old) / dt`` s. Precip outputs are diagnostics, passed straight through.
        """
        # cast from generic ``DataFeild`` to bare gt4py fields
        fields = cast("dict[str, fa.CellKField[ta.wpfloat]]", state)


        self._copy_into(fields["te"], self._te_in)
        for s in _SPECIES:
            self._copy_into(fields[f"q{s}"], getattr(self._q_in, s))

        self._muphys_step(
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

        self._to_tendency(fields["te"], self._t_out, self._tendencies["tend_temperature"])
        for s in _SPECIES:
            self._to_tendency(
                fields[f"q{s}"], getattr(self._q_out, s), self._tendencies[f"tend_q{s}"]
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
