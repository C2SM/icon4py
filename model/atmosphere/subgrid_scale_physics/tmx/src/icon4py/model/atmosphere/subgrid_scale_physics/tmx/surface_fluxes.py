# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Surface-flux provider seam for TMX.

The provider fills the surface-flux input buffers of the granule
(:class:`~icon4py.model.atmosphere.subgrid_scale_physics.tmx.tmx_states.TmxSurfaceFluxState`)
once per physics step. This cycle ships only :class:`ZeroFluxProvider` (pure
plumbing); the ocean bulk-flux scheme (prescribed SST, Louis exchange
coefficients — ``mo_tmx_surface.f90`` / ``mo_vdf_sfc.f90``) is a follow-up
implementation behind the same seam.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states


class SurfaceFluxProvider(Protocol):
    """Fills TMX's surface-flux input buffers, once per physics step."""

    def compute(self, *, out: tmx_states.TmxSurfaceFluxState) -> None:
        """Set every field of ``out``.

        Called as the final step of ``State.gather_from_prognostic`` (after
        the thermodynamic diagnostics, before ``Tmx.run`` consumes the
        buffers). Implementations must write all fields on every call — no
        partial updates.
        """
        ...


class ZeroFluxProvider:
    """Zero surface fluxes (the phase-2 seam's plumbing-only implementation).

    Explicitly re-zeros every call instead of no-op'ing: this upholds the
    "fluxes are set each step" contract even if the granule ever mutated the
    buffers in place. The buffers are 2-D, so the cost is negligible.
    """

    def compute(self, *, out: tmx_states.TmxSurfaceFluxState) -> None:
        for field in dataclasses.fields(out):
            getattr(out, field.name).ndarray[...] = 0.0
