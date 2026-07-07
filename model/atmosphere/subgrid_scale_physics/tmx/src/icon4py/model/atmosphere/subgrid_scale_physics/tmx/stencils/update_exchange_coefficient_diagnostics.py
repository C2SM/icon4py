# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import KDim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _update_exchange_coefficient_diagnostics(
    km_ic: fa.CellKField[wpfloat],
    kh_ic: fa.CellKField[wpfloat],
    km_const: wpfloat,
    rturb_prandtl: wpfloat,
    use_km_const: bool,
    nlev: gtx.int32,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Assemble the full-level exchange coefficient diagnostics ``km`` / ``kh``.

    Port of the km/kh loop of 'Update_diagnostics' in mo_vdf.f90 (these fields
    are output-only diagnostics, nothing in tmx reads them):

        km(k) = km_ic(k + 1),  kh(k) = kh_ic(k + 1)     for jk = 1..nlev-1
        km(nlev) = km_const,   kh(nlev) = km_const * rturb_prandtl
                                                        if use_km_const
        km(nlev) = km_sfc,     kh(nlev) = kh_sfc        otherwise

    The surface exchange coefficients ``km_sfc`` / ``kh_sfc`` are aggregated
    from the surface tiles (mo_vdf_diag_smag.f90) and are out of scope of the
    atmosphere-only port: the bottom row is set to zero when ``use_km_const``
    is False.

    The bottom row is selected with 'dims.KDim < nlev - 1' because
    'concat_where(dims.KDim == nlev - 1, ...)' is currently broken in GT4Py
    1.1.11 (GridTools/gt4py#2205). The constant bottom-row values are anchored
    to the K-bounded shifted field (``shifted * 0 + value``) instead of a bare
    ``broadcast``: a broadcast has an unbounded K range, which breaks the
    domain inference of the 'concat_where' branches ("Cannot compute length of
    open 'UnitRange'").

    Domains (Fortran): jk = 1..nlev; the tmx ``t_domain`` cell range
    (``grf_bdywidth_c + 1`` to ``min_rlcell_int``), which maps to the
    horizontal domain ``(h_grid.Zone.NUDGING, h_grid.Zone.LOCAL)``.

    Args:
        km_ic: turbulent viscosity at half-level cell centers (nlev + 1 rows)
            [m^2/s]
        kh_ic: turbulent diffusivity at half-level cell centers (nlev + 1
            rows) [m^2/s]
        km_const: constant exchange coefficient of the ``use_km_const``
            configuration [m^2/s]
        rturb_prandtl: reciprocal turbulent Prandtl number
        use_km_const: True if the constant exchange coefficient is used
        nlev: number of full levels

    Returns:
        turbulent viscosity and diffusivity at full-level cell centers
    """
    shifted_km = km_ic(KDim + 1)
    shifted_kh = kh_ic(KDim + 1)
    if use_km_const:
        km_bottom = shifted_km * wpfloat("0.0") + km_const
        kh_bottom = shifted_kh * wpfloat("0.0") + km_const * rturb_prandtl
    else:
        km_bottom = shifted_km * wpfloat("0.0")
        kh_bottom = shifted_kh * wpfloat("0.0")
    km = concat_where(dims.KDim < nlev - 1, shifted_km, km_bottom)
    kh = concat_where(dims.KDim < nlev - 1, shifted_kh, kh_bottom)
    return km, kh


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_exchange_coefficient_diagnostics(
    km_ic: fa.CellKField[wpfloat],
    kh_ic: fa.CellKField[wpfloat],
    km: fa.CellKField[wpfloat],
    kh: fa.CellKField[wpfloat],
    km_const: wpfloat,
    rturb_prandtl: wpfloat,
    use_km_const: bool,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _update_exchange_coefficient_diagnostics(
        km_ic=km_ic,
        kh_ic=kh_ic,
        km_const=km_const,
        rturb_prandtl=rturb_prandtl,
        use_km_const=use_km_const,
        nlev=nlev,
        out=(km, kh),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
