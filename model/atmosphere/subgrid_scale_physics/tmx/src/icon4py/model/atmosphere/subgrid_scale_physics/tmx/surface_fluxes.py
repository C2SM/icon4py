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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import config as tmx_config
from icon4py.model.common import constants
from icon4py.model.common.physics.thermodynamics import compute_moisture, compute_pressure


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
        t_sfc = self.surface_temperature.ndarray
        # psfc: bottom interface of pressure_ifc (Fortran pres_ifc(:, nlevp1, :))
        p_sfc = self.pressure_ifc.ndarray[:, -1]
        q_sat = compute_moisture.specific_humidity(compute_pressure.sat_pres_water(t_sfc), p_sfc)
        rho_sfc = p_sfc / (constants.RD * t_sfc * (1.0 + constants.RV_O_RD_MINUS_1 * q_sat))

        out.sensible_heat_flux.ndarray[...] = -self.config.shflx * constants.CVD * rho_sfc
        out.evapotranspiration.ndarray[...] = -self.config.lhflx * rho_sfc
        out.u_stress.ndarray[...] = 0.0
        out.v_stress.ndarray[...] = 0.0
        out.q_snocpymlt.ndarray[...] = 0.0
