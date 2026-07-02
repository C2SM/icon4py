# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_smagorinsky_viscosity import (
    EPS_LOUIS,
    compute_smagorinsky_viscosity,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def stability_term_classic_numpy(
    mech_prod: np.ndarray, bruvais: np.ndarray, rturb_prandtl: float
) -> np.ndarray:
    return np.sqrt(np.maximum(0.0, 0.5 * mech_prod - rturb_prandtl * bruvais))


def stability_term_louis_numpy(
    mech_prod: np.ndarray,
    bruvais: np.ndarray,
    scaling_factor_louis: np.ndarray,
    rturb_prandtl: float,
    louis_constant_b: float,
) -> np.ndarray:
    ri = 2.0 * bruvais / np.maximum(EPS_LOUIS, mech_prod)
    stability_function = np.maximum(
        1.0 - ri * rturb_prandtl,
        np.minimum(
            1.0,
            (1.0 / (1.0 + louis_constant_b * scaling_factor_louis[:, np.newaxis] * np.abs(ri)))
            ** 4.0,
        ),
    )
    return np.sqrt(0.5 * mech_prod * stability_function)


def compute_smagorinsky_viscosity_numpy(
    mech_prod: np.ndarray,
    bruvais: np.ndarray,
    rho_ic: np.ndarray,
    mixing_length_sq: np.ndarray,
    *,
    scaling_factor_louis: np.ndarray,
    fract_land: np.ndarray,
    fract_ice: np.ndarray,
    rturb_prandtl: float,
    louis_constant_b: float,
    use_louis: bool,
    use_louis_land: bool,
    use_louis_ice: bool,
) -> tuple[np.ndarray, np.ndarray]:
    nlev = mech_prod.shape[1] - 1

    if use_louis:
        # If the Louis formula is used but not over land and/or sea ice, use the
        # classic formulation for cells with more than 50% land fraction or more
        # than 50% ice fraction.
        classic_mask = ((not use_louis_land) & (fract_land > 0.5)) | (
            (not use_louis_ice) & (fract_ice > 0.5)
        )
        stability_term = np.where(
            classic_mask[:, np.newaxis],
            stability_term_classic_numpy(mech_prod, bruvais, rturb_prandtl),
            stability_term_louis_numpy(
                mech_prod, bruvais, scaling_factor_louis, rturb_prandtl, louis_constant_b
            ),
        )
    else:
        stability_term = stability_term_classic_numpy(mech_prod, bruvais, rturb_prandtl)

    km_ic = np.zeros_like(mech_prod)
    # interior half levels, Fortran jk = 2..nlev (1-based) -> k = 1..nlev-1 (0-based)
    km_ic[:, 1:nlev] = rho_ic[:, 1:nlev] * mixing_length_sq[:, 1:nlev] * stability_term[:, 1:nlev]
    # boundary rows are copies of the adjacent interior rows
    # (Fortran 1-based: k = 1 <- k = 2, k = nlevp1 <- k = nlev)
    km_ic[:, 0] = km_ic[:, 1]
    km_ic[:, nlev] = km_ic[:, nlev - 1]
    kh_ic = km_ic * rturb_prandtl
    return km_ic, kh_ic


def smagorinsky_viscosity_reference(
    connectivities: dict[gtx.Dimension, np.ndarray],
    *,
    mech_prod: np.ndarray,
    bruvais: np.ndarray,
    rho_ic: np.ndarray,
    mixing_length_sq: np.ndarray,
    scaling_factor_louis: np.ndarray,
    fract_land: np.ndarray,
    fract_ice: np.ndarray,
    rturb_prandtl: float,
    louis_constant_b: float,
    use_louis: bool,
    use_louis_land: bool,
    use_louis_ice: bool,
    **kwargs,
) -> dict:
    km_ic, kh_ic = compute_smagorinsky_viscosity_numpy(
        mech_prod,
        bruvais,
        rho_ic,
        mixing_length_sq,
        scaling_factor_louis=scaling_factor_louis,
        fract_land=fract_land,
        fract_ice=fract_ice,
        rturb_prandtl=rturb_prandtl,
        louis_constant_b=louis_constant_b,
        use_louis=use_louis,
        use_louis_land=use_louis_land,
        use_louis_ice=use_louis_ice,
    )
    return dict(km_ic=km_ic, kh_ic=kh_ic)


def smagorinsky_viscosity_input_data(
    grid: base.Grid,
    use_louis: bool,
    use_louis_land: bool,
    use_louis_ice: bool,
) -> dict[str, gtx.Field | state_utils.ScalarType]:
    mech_prod = data_alloc.random_field(
        grid, dims.CellDim, dims.KDim, low=0.0, high=1.0e-2, dtype=wpfloat, extend={dims.KDim: 1}
    )
    bruvais = data_alloc.random_field(
        grid,
        dims.CellDim,
        dims.KDim,
        low=-1.0e-3,
        high=1.0e-3,
        dtype=wpfloat,
        extend={dims.KDim: 1},
    )
    rho_ic = data_alloc.random_field(
        grid, dims.CellDim, dims.KDim, low=0.5, high=1.4, dtype=wpfloat, extend={dims.KDim: 1}
    )
    mixing_length_sq = data_alloc.random_field(
        grid, dims.CellDim, dims.KDim, low=0.0, high=1.0e4, dtype=wpfloat, extend={dims.KDim: 1}
    )
    scaling_factor_louis = data_alloc.random_field(
        grid, dims.CellDim, low=0.5, high=2.0, dtype=wpfloat
    )
    fract_land = data_alloc.random_field(grid, dims.CellDim, low=0.0, high=1.0, dtype=wpfloat)
    fract_ice = data_alloc.random_field(grid, dims.CellDim, low=0.0, high=1.0, dtype=wpfloat)
    km_ic = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=wpfloat, extend={dims.KDim: 1}
    )
    kh_ic = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=wpfloat, extend={dims.KDim: 1}
    )

    return dict(
        mech_prod=mech_prod,
        bruvais=bruvais,
        rho_ic=rho_ic,
        mixing_length_sq=mixing_length_sq,
        scaling_factor_louis=scaling_factor_louis,
        fract_land=fract_land,
        fract_ice=fract_ice,
        km_ic=km_ic,
        kh_ic=kh_ic,
        rturb_prandtl=wpfloat(2.0),
        louis_constant_b=wpfloat(5.3),
        use_louis=use_louis,
        use_louis_land=use_louis_land,
        use_louis_ice=use_louis_ice,
        nlev=gtx.int32(grid.num_levels),
        horizontal_start=0,
        horizontal_end=gtx.int32(grid.num_cells),
        vertical_start=0,
        vertical_end=gtx.int32(grid.num_levels + 1),
    )


#: Static-params variants: prove that the config bools can be passed both as regular
#: runtime scalars ("none") and as static (compile-time) arguments selecting the variant.
STATIC_VARIANTS = {
    "none": (),
    "compile_time_variant": ("use_louis", "use_louis_land", "use_louis_ice"),
}


class TestComputeSmagorinskyViscosityClassic(StencilTest):
    PROGRAM = compute_smagorinsky_viscosity
    OUTPUTS = ("km_ic", "kh_ic")
    STATIC_PARAMS = STATIC_VARIANTS

    reference = staticmethod(smagorinsky_viscosity_reference)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        return smagorinsky_viscosity_input_data(
            grid, use_louis=False, use_louis_land=True, use_louis_ice=True
        )


class TestComputeSmagorinskyViscosityLouis(StencilTest):
    PROGRAM = compute_smagorinsky_viscosity
    OUTPUTS = ("km_ic", "kh_ic")
    STATIC_PARAMS = STATIC_VARIANTS

    reference = staticmethod(smagorinsky_viscosity_reference)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        return smagorinsky_viscosity_input_data(
            grid, use_louis=True, use_louis_land=True, use_louis_ice=True
        )


class TestComputeSmagorinskyViscosityLouisMaskedLandIce(StencilTest):
    PROGRAM = compute_smagorinsky_viscosity
    OUTPUTS = ("km_ic", "kh_ic")
    STATIC_PARAMS = STATIC_VARIANTS

    reference = staticmethod(smagorinsky_viscosity_reference)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        return smagorinsky_viscosity_input_data(
            grid, use_louis=True, use_louis_land=False, use_louis_ice=False
        )


class TestComputeSmagorinskyViscosityLouisMaskedLandOnly(StencilTest):
    PROGRAM = compute_smagorinsky_viscosity
    OUTPUTS = ("km_ic", "kh_ic")
    STATIC_PARAMS = STATIC_VARIANTS

    reference = staticmethod(smagorinsky_viscosity_reference)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        return smagorinsky_viscosity_input_data(
            grid, use_louis=True, use_louis_land=False, use_louis_ice=True
        )
