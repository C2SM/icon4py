# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The tmx turbulent mixing as a physics component."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, cast

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import data as tmx_data, tmx, tmx_states
from icon4py.model.common import model_backends, type_alias as ta
from icon4py.model.common.states import surface_flux_state


if TYPE_CHECKING:
    import gt4py.next as gtx
    import gt4py.next.typing as gtx_typing

    from icon4py.model.atmosphere.subgrid_scale_physics.tmx import config as tmx_config
    from icon4py.model.common import time
    from icon4py.model.common.decomposition import definitions as decomposition
    from icon4py.model.common.grid import base as base_grid, states as grid_states
    from icon4py.model.common.states import model


class TmxComponent:
    """Presents the tmx turbulent mixing as a model component.

    tmx reports the tendencies of temperature, the three moist tracers it diffuses and the
    three wind components, which the physics driver adds to the state, and the turbulent
    viscosity and diffusivity ``km`` and ``kh`` as diagnostics, bound to driver-owned buffers.
    The tmx diagnostic, tendency and new states are allocated once here and reused.
    """

    # TODO (Yilu): inherit the Component protocol once it is formalized (deferred to a separate PR).
    inputs_properties = tmx_data.INPUTS_PROPERTIES
    outputs_properties = tmx_data.OUTPUTS_PROPERTIES

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        config: tmx_config.TmxConfig,
        metric_state: tmx_states.TmxMetricState,
        interpolation_state: tmx_states.TmxInterpolationState,
        edge_params: grid_states.EdgeParams,
        cell_params: grid_states.CellParams,
        dtime: time.RelativeTime,
        backend: gtx_typing.Backend | None = None,
        exchange: decomposition.ExchangeRuntime,
    ) -> None:
        self._dt_seconds = ta.wpfloat(dtime.total_seconds())
        self._tmx = tmx.Tmx(
            grid=grid,
            config=config,
            metric_state=metric_state,
            interpolation_state=interpolation_state,
            edge_params=edge_params,
            cell_params=cell_params,
            backend=backend,
            exchange=exchange,
        )
        allocator = model_backends.get_allocator(backend)
        self._diagnostic_state = tmx_states.TmxDiagnosticState.allocate(grid, allocator=allocator)
        self._tendency_state = tmx_states.TmxTendencyState.allocate(grid, allocator=allocator)
        self._new_state = tmx_states.TmxNewState.allocate(grid, allocator=allocator)

    def __call__(
        self, state: dict[str, model.DataField], time_step: time.AbsoluteTime
    ) -> dict[str, model.DataField]:
        """Run one tmx step and report its tendencies and ``km``, ``kh``.

        The inputs are read only; the new state tmx computes alongside the tendencies stays
        internal, the driver advances the model state from the tendencies.
        """
        fields = cast("dict[str, gtx.Field]", state)
        input_state = tmx_states.TmxInputState(
            temperature=fields["temperature"],
            virtual_temperature=fields["virtual_temperature"],
            pressure=fields["pressure"],
            u=fields["u"],
            v=fields["v"],
            w=fields["w"],
            qv=fields["qv"],
            qc=fields["qc"],
            qi=fields["qi"],
            qr=fields["qr"],
            qs=fields["qs"],
            qg=fields["qg"],
            rho=fields["rho"],
            air_mass=fields["air_mass"],
        )
        self._tmx.run(
            input_state=input_state,
            surface_flux_state=surface_flux_state.SurfaceFluxState(
                **{name: fields[name] for name in tmx_data.FLUX_PORTS}
            ),
            diagnostic_state=self._diagnostic_state,
            tendency_state=self._tendency_state,
            new_state=self._new_state,
            dtime=self._dt_seconds,
        )
        return cast(
            "dict[str, model.DataField]",
            {
                **{
                    f"tend_{name}": getattr(self._tendency_state, f"tend_{name}")
                    for name in tmx_data.TENDENCY_PORTS
                },
                "km": self._diagnostic_state.km,
                "kh": self._diagnostic_state.kh,
            },
        )

    def bind_output_buffers(self, buffers: dict[str, Any]) -> None:
        """Redirect ``km`` and ``kh`` into buffers the caller owns.

        tmx writes its diagnostic state in place on every step, so replacing the two fields
        once here redirects every write that follows. ``PhysicsDriver`` calls this once per
        process while it is being built.
        """
        self._diagnostic_state = dataclasses.replace(
            self._diagnostic_state, km=buffers["km"], kh=buffers["kh"]
        )
