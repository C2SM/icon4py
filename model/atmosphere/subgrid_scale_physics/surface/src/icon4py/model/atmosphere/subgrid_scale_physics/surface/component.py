# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The surface scheme as a physics component."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from icon4py.model.atmosphere.subgrid_scale_physics.surface import (
    data as surface_data,
    surface,
    surface_states,
)
from icon4py.model.common import model_backends
from icon4py.model.common.states.surface_flux_state import SurfaceFluxState


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.atmosphere.subgrid_scale_physics.surface import config as surface_config
    from icon4py.model.common import field_type_aliases as fa, time, type_alias as ta
    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.common.states import model


class SurfaceComponent:
    """Presents the surface scheme as a model component.

    The scheme produces no tendencies: its outputs are the surface fluxes and the surface
    temperature, which the physics driver binds to its surface state so that tmx reads them
    there. Without a driver the component writes a flux state of its own.
    """

    # TODO (Yilu): inherit the Component protocol once it is formalized (deferred to a separate PR).
    inputs_properties = surface_data.INPUTS_PROPERTIES
    outputs_properties = surface_data.OUTPUTS_PROPERTIES

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        config: surface_config.SurfaceConfig,
        backend: gtx_typing.Backend | None = None,
    ) -> None:
        self._surface = surface.Surface(grid=grid, config=config, backend=backend)
        # the outputs: replaced by the caller's buffers in `bind_output_buffers`
        self._fluxes = SurfaceFluxState.allocate(
            grid, allocator=model_backends.get_allocator(backend)
        )
        self._surface_temperature: fa.CellField[ta.wpfloat] = self._surface.surface_temperature

    def __call__(
        self, state: dict[str, model.DataField], time_step: time.AbsoluteTime
    ) -> dict[str, model.DataField]:
        """Run one step of the surface scheme and report the fluxes and the surface temperature.

        After ``bind_output_buffers`` the returned fields are the caller's own buffers,
        already filled in place.
        """
        fields = cast("dict[str, fa.CellField[ta.wpfloat]]", state)
        self._surface.run(
            input_state=surface_states.SurfaceInputState(
                surface_pressure=fields["surface_pressure"], temperature=fields["temperature"]
            ),
            out=self._fluxes,
        )
        return cast(
            "dict[str, model.DataField]",
            {
                **{name: getattr(self._fluxes, name) for name in surface_data.FLUX_PORTS},
                "surface_temperature": self._surface_temperature,
            },
        )

    def bind_output_buffers(self, buffers: dict[str, Any]) -> None:
        """Redirect the outputs into buffers the caller owns.

        ``__call__`` hands the flux state to the scheme as its output on every step, so
        replacing it once here redirects every write that follows. The surface temperature is
        constant, so the bound buffer is filled once, here.

        ``PhysicsDriver`` calls this once per process while it is being built, passing the
        fields of its surface state.
        """
        self._fluxes = SurfaceFluxState(**{name: buffers[name] for name in surface_data.FLUX_PORTS})
        self._surface_temperature = buffers["surface_temperature"]
        self._surface_temperature.ndarray[...] = self._surface.surface_temperature.ndarray  # type: ignore[index] # NDArrayObject Protocol doesn't support this
