# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
from typing import ClassVar

from gt4py import next as gtx
from gt4py.next import typing as gtx_typing

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import Q
from icon4py.model.common import dimension as dims


@dataclasses.dataclass
class GraupelOutput:
    t: gtx.Field[dims.CellDim, dims.KDim]
    qv: gtx.Field[dims.CellDim, dims.KDim]
    qc: gtx.Field[dims.CellDim, dims.KDim]
    qi: gtx.Field[dims.CellDim, dims.KDim]
    qr: gtx.Field[dims.CellDim, dims.KDim]
    qs: gtx.Field[dims.CellDim, dims.KDim]
    qg: gtx.Field[dims.CellDim, dims.KDim]

    pflx: gtx.Field[dims.CellDim, dims.KDim] | None
    pr: gtx.Field[dims.CellDim, dims.KDim] | None
    ps: gtx.Field[dims.CellDim, dims.KDim] | None
    pi: gtx.Field[dims.CellDim, dims.KDim] | None
    pg: gtx.Field[dims.CellDim, dims.KDim] | None
    pre: gtx.Field[dims.CellDim, dims.KDim] | None

    _surface_fields: ClassVar[list[str]] = ["pr", "ps", "pi", "pg", "pre"]

    @classmethod
    def allocate(
        cls,
        allocator: gtx_typing.Allocator,
        domain: gtx.Domain,
        dtype: gtx.float32 | gtx.float64 = gtx.float64,
        references: dict[str, gtx.Field] | None = None,
    ):
        """
        Returns a GraupelOutput with allocated fields.

        :param domain: Full domain of the Muphys fields.
        :param references: Dictionary of fields that should be re-used instead of allocated.
        """
        # TODO(havogt): maybe this function should become an __init__ with defaults
        if references is None:
            references = {}

        zeros_full = functools.partial(gtx.zeros, domain=domain, allocator=allocator, dtype=dtype)
        surface_domain = gtx.Domain(
            dims=domain.dims,
            ranges=(
                domain.ranges[0],
                gtx.unit_range((domain.ranges[1].stop - 1, domain.ranges[1].stop)),
            ),
        )
        zeros_surface = functools.partial(
            gtx.zeros, domain=surface_domain, allocator=allocator, dtype=dtype
        )
        return cls(
            **{
                field.name: references[field.name]
                if field.name in references
                else zeros_surface()
                if field.name in cls._surface_fields
                else zeros_full()
                for field in dataclasses.fields(cls)
            }
        )

    @property
    def q(self) -> Q:
        return Q(
            v=self.qv,
            c=self.qc,
            r=self.qr,
            s=self.qs,
            i=self.qi,
            g=self.qg,
        )
