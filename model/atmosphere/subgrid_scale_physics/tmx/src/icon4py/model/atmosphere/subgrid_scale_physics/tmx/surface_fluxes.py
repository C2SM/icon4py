# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Protocol

import gt4py.next as gtx
from gt4py.next import common as gtx_common

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import config as tmx_config
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.physics.thermodynamics.compute_moisture import specific_humidity_on_cells
from icon4py.model.common.physics.thermodynamics.compute_pressure import sat_pres_water_on_cells
from icon4py.model.common.type_alias import wpfloat


if TYPE_CHECKING:
    from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states
    from icon4py.model.common import field_type_aliases as fa, type_alias as ta


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
    """Zero surface fluxes.

    Explicitly re-zeros every call: this upholds the
    "fluxes are set each step" contract even if the granule ever mutated the
    buffers in place. The buffers are 2-D, so the cost is negligible.
    """

    def compute(self, *, out: tmx_states.TmxSurfaceFluxState) -> None:
        for field in dataclasses.fields(out):
            getattr(out, field.name).ndarray[...] = 0.0


@gtx.field_operator
def _compute_prescribed_surface_fluxes(
    surface_temperature: fa.CellField[ta.wpfloat],
    surface_pressure: fa.CellField[ta.wpfloat],
    shflx: ta.wpfloat,
    lhflx: ta.wpfloat,
) -> tuple[fa.CellField[ta.wpfloat], fa.CellField[ta.wpfloat]]:
    """Sensible heat flux and evapotranspiration from the prescribed coefficients."""
    q_sat = specific_humidity_on_cells(
        sat_pres_water_on_cells(surface_temperature), surface_pressure
    )
    density = surface_pressure / (
        PhysicsConstants.rd
        * surface_temperature
        * (wpfloat("1.0") + PhysicsConstants.rv_o_rd_minus_1 * q_sat)
    )
    return -shflx * PhysicsConstants.cvd * density, -lhflx * density


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_prescribed_surface_fluxes(  # noqa: PLR0917 [too-many-positional-arguments]
    surface_temperature: fa.CellField[ta.wpfloat],
    surface_pressure: fa.CellField[ta.wpfloat],
    sensible_heat_flux: fa.CellField[ta.wpfloat],
    evapotranspiration: fa.CellField[ta.wpfloat],
    shflx: ta.wpfloat,
    lhflx: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _compute_prescribed_surface_fluxes(
        surface_temperature,
        surface_pressure,
        shflx,
        lhflx,
        out=(sensible_heat_flux, evapotranspiration),
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class PrescribedFluxProvider:
    """Fixed kinematic surface heat fluxes (``SurfaceType.FIXED_HEAT_FLUXES``).

    Port of the ``isrfc_type == 1`` early-return branch of
    'compute_sfc_fluxes' (mo_tmx_surface.f90:802-812), with the surface
    density it multiplies taken from 'compute_sfc_density'
    (mo_vdf_diag_smag.f90:72-125):

        e_sat   = sat_pres_water(t_sfc)
        q_sat   = specific_humidity(e_sat, p_sfc)
        rho_sfc = p_sfc / (rd * t_sfc * (1 + vtmpc1 * q_sat))

        sensible_heat_flux = -shflx * cvd * rho_sfc
        evapotranspiration = -lhflx * rho_sfc
        u_stress = v_stress = 0

    The momentum stresses stay zero because the Fortran returns before the
    bulk-stress block; ``q_snocpymlt`` is a land-only quantity and stays zero
    as well.

    ``pressure_ifc`` is read afresh on every ``compute`` call, so it must be
    the live buffer the caller updates each step, not a snapshot.
    """

    config: tmx_config.TmxConfig
    pressure_ifc: fa.CellKField[ta.wpfloat]
    surface_temperature: fa.CellField[ta.wpfloat]

    def __post_init__(self) -> None:
        if self.config.surface_type is not tmx_config.SurfaceType.FIXED_HEAT_FLUXES:
            raise NotImplementedError(
                "PrescribedFluxProvider implements "
                f"{tmx_config.SurfaceType.FIXED_HEAT_FLUXES!r}, got {self.config.surface_type!r}."
            )

    def compute(self, *, out: tmx_states.TmxSurfaceFluxState) -> None:
        num_cells = self.surface_temperature.domain[dims.CellDim].unit_range.stop
        # psfc: bottom interface of pressure_ifc (Fortran pres_ifc(:, nlevp1, :)).
        # A view, so the live buffer is read rather than copied on every step.
        surface_pressure = gtx_common._field(
            self.pressure_ifc.ndarray[:, -1], domain={dims.CellDim: (0, num_cells)}
        )
        compute_prescribed_surface_fluxes(
            surface_temperature=self.surface_temperature,
            surface_pressure=surface_pressure,
            sensible_heat_flux=out.sensible_heat_flux,
            evapotranspiration=out.evapotranspiration,
            shflx=self.config.shflx,
            lhflx=self.config.lhflx,
            horizontal_start=0,
            horizontal_end=num_cells,
            offset_provider={},
        )
        out.u_stress.ndarray[...] = 0.0
        out.v_stress.ndarray[...] = 0.0
        out.q_snocpymlt.ndarray[...] = 0.0
