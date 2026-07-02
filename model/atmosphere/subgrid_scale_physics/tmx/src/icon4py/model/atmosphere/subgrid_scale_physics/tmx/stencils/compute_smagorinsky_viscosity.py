# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Final

import gt4py.next as gtx
from gt4py.next import abs, maximum, minimum, power, sqrt, where  # noqa: A004
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import KDim
from icon4py.model.common.type_alias import wpfloat


#: Threshold to avoid division by zero in the Richardson number
#: (``eps_louis`` in ICON's ``mo_tmx_smagorinsky.f90``)
EPS_LOUIS: Final[wpfloat] = wpfloat(1.0e-28)


@gtx.field_operator
def _stability_term_classic(
    mech_prod: fa.CellKField[wpfloat],
    bruvais: fa.CellKField[wpfloat],
    rturb_prandtl: wpfloat,
) -> fa.CellKField[wpfloat]:
    """
    Compute the classic (Lilly 1962) stability correction term for the eddy viscosity:

        stability_term = sqrt(max(0, |S|^2 - N^2 / Pr_t))

    with |S|^2 = 0.5 * mech_prod the square of the strain rate magnitude,
    N^2 = bruvais the Brunt-Vaisala frequency squared, and
    1 / Pr_t = rturb_prandtl the reciprocal turbulent Prandtl number.
    """
    return sqrt(maximum(wpfloat("0.0"), wpfloat("0.5") * mech_prod - rturb_prandtl * bruvais))


@gtx.field_operator
def _stability_term_louis(
    mech_prod: fa.CellKField[wpfloat],
    bruvais: fa.CellKField[wpfloat],
    scaling_factor_louis: fa.CellField[wpfloat],
    rturb_prandtl: wpfloat,
    louis_constant_b: wpfloat,
) -> fa.CellKField[wpfloat]:
    """
    Compute the stability correction term for the eddy viscosity based on the
    stability correction function of Louis (1979):

        Ri = 2 * N^2 / max(eps, mech_prod)
        stability_function = max(1 - Ri / Pr_t,
                                 min(1, (1 / (1 + b * scaling * |Ri|))^4))
        stability_term = sqrt(0.5 * mech_prod * stability_function)
    """
    # Note: has to be defined inside the field operator, module-level closure
    # constants are not supported by the gtfn backend (keep in sync with
    # EPS_LOUIS above).
    eps_louis = wpfloat("1.0e-28")
    ri = wpfloat("2.0") * bruvais / maximum(eps_louis, mech_prod)

    stability_function = maximum(
        wpfloat("1.0") - ri * rturb_prandtl,
        minimum(
            wpfloat("1.0"),
            power(
                wpfloat("1.0")
                / (wpfloat("1.0") + louis_constant_b * scaling_factor_louis * abs(ri)),
                wpfloat("4.0"),
            ),
        ),
    )

    return sqrt(wpfloat("0.5") * mech_prod * stability_function)


@gtx.field_operator
def _compute_smagorinsky_viscosity(
    mech_prod: fa.CellKField[wpfloat],
    bruvais: fa.CellKField[wpfloat],
    rho_ic: fa.CellKField[wpfloat],
    mixing_length_sq: fa.CellKField[wpfloat],
    scaling_factor_louis: fa.CellField[wpfloat],
    fract_land: fa.CellField[wpfloat],
    fract_ice: fa.CellField[wpfloat],
    rturb_prandtl: wpfloat,
    louis_constant_b: wpfloat,
    use_louis: bool,
    use_louis_land: bool,
    use_louis_ice: bool,
    nlev: gtx.int32,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Compute the eddy viscosity and diffusivity at half-level cell centers based on
    the Smagorinsky-Lilly eddy viscosity model.

    Port of ``Smagorinsky_model`` in ICON's ``mo_tmx_smagorinsky.f90``:
    - interior half levels (0 < k < nlev):
        km_ic = rho_ic * mixing_length_sq * stability_term
        kh_ic = km_ic * rturb_prandtl
    - boundary half levels are copies of the adjacent interior rows:
        k = 0 copies k = 1, k = nlev copies k = nlev - 1
      (Fortran 1-based: k = 1 <- k = 2, k = nlevp1 <- k = nlev).

    Depending on the configuration, the classic (Lilly 1962) or the Louis (1979)
    stability correction function is used. If the Louis formulation is enabled but
    excluded over land (``use_louis_land = False``) and/or sea ice
    (``use_louis_ice = False``), cells with more than 50% land fraction and/or more
    than 50% ice fraction fall back to the classic formulation.

    ``use_louis``, ``use_louis_land`` and ``use_louis_ice`` are scalar configuration
    flags; they can be passed as static (compile-time) arguments so that only the
    selected variant is compiled.
    """
    if use_louis:
        stability_classic = _stability_term_classic(
            mech_prod=mech_prod, bruvais=bruvais, rturb_prandtl=rturb_prandtl
        )
        stability_louis = _stability_term_louis(
            mech_prod=mech_prod,
            bruvais=bruvais,
            scaling_factor_louis=scaling_factor_louis,
            rturb_prandtl=rturb_prandtl,
            louis_constant_b=louis_constant_b,
        )
        if use_louis_land:
            if use_louis_ice:
                stability_term = stability_louis
            else:
                stability_term = where(
                    fract_ice > wpfloat("0.5"), stability_classic, stability_louis
                )
        else:
            if use_louis_ice:
                stability_term = where(
                    fract_land > wpfloat("0.5"), stability_classic, stability_louis
                )
            else:
                stability_term = where(
                    (fract_land > wpfloat("0.5")) | (fract_ice > wpfloat("0.5")),
                    stability_classic,
                    stability_louis,
                )
    else:
        stability_term = _stability_term_classic(
            mech_prod=mech_prod, bruvais=bruvais, rturb_prandtl=rturb_prandtl
        )

    km = rho_ic * mixing_length_sq * stability_term
    km_ic = concat_where(dims.KDim == 0, km(KDim + 1), km)
    km_ic = concat_where(dims.KDim == nlev, km(KDim - 1), km_ic)
    kh_ic = km_ic * rturb_prandtl
    return km_ic, kh_ic


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_smagorinsky_viscosity(
    mech_prod: fa.CellKField[wpfloat],
    bruvais: fa.CellKField[wpfloat],
    rho_ic: fa.CellKField[wpfloat],
    mixing_length_sq: fa.CellKField[wpfloat],
    scaling_factor_louis: fa.CellField[wpfloat],
    fract_land: fa.CellField[wpfloat],
    fract_ice: fa.CellField[wpfloat],
    km_ic: fa.CellKField[wpfloat],
    kh_ic: fa.CellKField[wpfloat],
    rturb_prandtl: wpfloat,
    louis_constant_b: wpfloat,
    use_louis: bool,
    use_louis_land: bool,
    use_louis_ice: bool,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_smagorinsky_viscosity(
        mech_prod=mech_prod,
        bruvais=bruvais,
        rho_ic=rho_ic,
        mixing_length_sq=mixing_length_sq,
        scaling_factor_louis=scaling_factor_louis,
        fract_land=fract_land,
        fract_ice=fract_ice,
        rturb_prandtl=rturb_prandtl,
        louis_constant_b=louis_constant_b,
        use_louis=use_louis,
        use_louis_land=use_louis_land,
        use_louis_ice=use_louis_ice,
        nlev=nlev,
        out=(km_ic, kh_ic),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
