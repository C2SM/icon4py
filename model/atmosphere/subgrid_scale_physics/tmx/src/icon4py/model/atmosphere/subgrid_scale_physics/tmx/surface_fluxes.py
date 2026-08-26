# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Protocol

from icon4py.model.common import constants, thermodynamic_functions


if TYPE_CHECKING:
    from typing import Self

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


# Fortran defaults of the fixed-surface-flux members of 'nh_testcase_nml',
# from the initialization in mo_nh_testcases_nml.f90:280-282. They are needed
# as defaults because the *input* namelist dict carries only the members the
# experiment sets explicitly; unset ones reach the run through this Fortran
# initialization and never appear in the dict.
_ISRFC_TYPE_DEFAULT = 0
_SHFLX_DEFAULT = 0.1
_LHFLX_DEFAULT = 0.0

# 'isrfc_type' value that selects the fixed surface heat fluxes.
FIXED_SURFACE_FLUXES = 1


@dataclasses.dataclass(frozen=True)
class PrescribedFluxConfig:
    """The fixed-surface-flux options of the ICON testcase namelist.

    Mirrors the 'nh_testcase_nml' members read by the tmx surface scheme
    (mo_tmx_surface.f90 uses 'isrfc_type', 'shflx' and 'lhflx' from
    mo_nh_testcases_nml.f90). Defaults mirror the Fortran initialization.
    """

    isrfc_type: int = _ISRFC_TYPE_DEFAULT
    """Surface flux treatment: 0 = interactive surface, 1 = fixed heat fluxes."""
    shflx: float = _SHFLX_DEFAULT
    """Kinematic sensible heat flux at the surface [K m/s]."""
    lhflx: float = _LHFLX_DEFAULT
    """Kinematic latent heat flux at the surface [m/s]."""

    @classmethod
    def from_fortran_dict(cls, input_dict: dict[str, Any]) -> Self:
        """Read the config from the converted *input* namelist dict.

        Unlike the echoed namelists, the input dict holds only the members the
        experiment sets explicitly, so absent members fall back to the Fortran
        defaults above rather than being indexed strictly.
        """
        testcase = input_dict.get("nh_testcase_nml", {})
        return cls(
            isrfc_type=int(testcase.get("isrfc_type", _ISRFC_TYPE_DEFAULT)),
            shflx=float(testcase.get("shflx", _SHFLX_DEFAULT)),
            lhflx=float(testcase.get("lhflx", _LHFLX_DEFAULT)),
        )


class PrescribedFluxProvider:
    """Fixed kinematic surface heat fluxes (``isrfc_type = 1``).

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

    def __init__(
        self,
        *,
        config: PrescribedFluxConfig,
        pressure_ifc: fa.CellKField[ta.wpfloat],
        surface_temperature: fa.CellField[ta.wpfloat],
    ) -> None:
        if config.isrfc_type != FIXED_SURFACE_FLUXES:
            raise NotImplementedError(
                f"PrescribedFluxProvider implements isrfc_type = {FIXED_SURFACE_FLUXES} "
                f"(fixed surface heat fluxes), got {config.isrfc_type}."
            )
        self._config = config
        self._pressure_ifc = pressure_ifc
        self._surface_temperature = surface_temperature

    def compute(self, *, out: tmx_states.TmxSurfaceFluxState) -> None:
        t_sfc = self._surface_temperature.ndarray
        # psfc: bottom interface of pressure_ifc (Fortran pres_ifc(:, nlevp1, :))
        p_sfc = self._pressure_ifc.ndarray[:, -1]
        q_sat = thermodynamic_functions.specific_humidity(
            thermodynamic_functions.sat_pres_water(t_sfc), p_sfc
        )
        rho_sfc = p_sfc / (constants.RD * t_sfc * (1.0 + constants.RV_O_RD_MINUS_1 * q_sat))

        out.sensible_heat_flux.ndarray[...] = -self._config.shflx * constants.CVD * rho_sfc
        out.evapotranspiration.ndarray[...] = -self._config.lhflx * rho_sfc
        out.u_stress.ndarray[...] = 0.0
        out.v_stress.ndarray[...] = 0.0
        out.q_snocpymlt.ndarray[...] = 0.0
