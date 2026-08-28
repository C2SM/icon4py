# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.diagnostics import (
    compute_strain_rate_diagnostics,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.testing import stencil_tests


def interpolate_to_cell_center_numpy(
    interpolant: np.ndarray, e_bln_c_s: np.ndarray, c2e: np.ndarray
) -> np.ndarray:
    """Edge -> cell average with the bilinear C2E weights, on full levels."""
    return np.sum(np.expand_dims(e_bln_c_s, axis=-1) * interpolant[c2e], axis=1)


def interpolate_shear_to_half_level_cells_numpy(
    shear: np.ndarray, e_bln_c_s: np.ndarray, wgtfac_c: np.ndarray, c2e: np.ndarray
) -> np.ndarray:
    """Reference of ``_interpolate_edge_field_to_cell_half_levels_wp`` (nlev + 1 levels)."""
    shear_c = interpolate_to_cell_center_numpy(shear, e_bln_c_s, c2e)

    # Full -> half level interpolation: half level k mixes full levels k and k - 1.
    # Fortran jk = 2..nlev (1-based) -> k = 1..nlev-1 (0-based); the top and
    # bottom half-level rows are not computed.
    mech_prod = np.zeros_like(wgtfac_c)
    mech_prod[:, 1:-1] = (
        wgtfac_c[:, 1:-1] * shear_c[:, 1:] + (1.0 - wgtfac_c[:, 1:-1]) * shear_c[:, :-1]
    )
    return mech_prod


class TestComputeStrainRateDiagnostics(stencil_tests.StencilTest):
    PROGRAM = compute_strain_rate_diagnostics
    OUTPUTS = ("div_c", "mech_prod")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        shear: np.ndarray,
        div_stress: np.ndarray,
        e_bln_c_s: np.ndarray,
        wgtfac_c: np.ndarray,
        div_c: np.ndarray,
        mech_prod: np.ndarray,
        vertical_end: int,
        cell_start_nudging: int,
        cell_start_lateral_boundary_level_3: int,
        cell_end_halo: int,
        **kwargs: Any,
    ) -> dict:
        nlev = vertical_end
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        c2e = connectivities[dims.C2E]  # (n_cells, 3)

        div_c_full = interpolate_to_cell_center_numpy(div_stress, e_bln_c_s, c2e)
        mech_prod_full = interpolate_shear_to_half_level_cells_numpy(
            shear, e_bln_c_s, wgtfac_c, c2e
        )

        # Each output only covers its own sub-domain; elsewhere the field keeps the
        # value it was allocated with.
        div_c_out = div_c.copy()
        div_c_out[cell_start_nudging:cell_end_halo, 0:nlev] = div_c_full[
            cell_start_nudging:cell_end_halo, 0:nlev
        ]

        mech_prod_out = mech_prod.copy()
        mech_prod_out[cell_start_lateral_boundary_level_3:cell_end_halo, 1:nlev] = mech_prod_full[
            cell_start_lateral_boundary_level_3:cell_end_halo, 1:nlev
        ]

        return dict(div_c=div_c_out, mech_prod=mech_prod_out)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        shear = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        div_stress = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        e_bln_c_s = data_alloc.random_field(dims.CellDim, dims.C2EDim, dtype=ta.wpfloat)
        wgtfac_c = data_alloc.random_field(
            dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        div_c = data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        mech_prod = data_alloc.zero_field(
            dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )

        # Fortran: get_horizontal_divergence_strain_rate_cell (div_c) starts at
        # refin_ctrl grf_bdywidth_c + 1 and interpolate_rate_of_strain_full2half_edge2cell
        # (mech_prod) at refin_ctrl 3; both end at min_rlcell_int - 1.
        cell_domain = h_grid.domain(dims.CellDim)
        cell_start_nudging = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        cell_start_lateral_boundary_level_3 = grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        cell_end_halo = grid.end_index(cell_domain(h_grid.Zone.HALO))
        # A grid without a lateral boundary (the simple grid) starts every cell zone
        # at 0, which would make the two per-output horizontal domains coincide.
        # Pull the div_c start in by one cell so that both outputs are masked with
        # their own bound, as they are on a regional grid where the nudging zone
        # starts well after lateral boundary level 3.
        cell_start_nudging = max(cell_start_nudging, cell_start_lateral_boundary_level_3 + 1)
        assert cell_start_lateral_boundary_level_3 < cell_start_nudging < cell_end_halo

        return dict(
            shear=shear,
            div_stress=div_stress,
            e_bln_c_s=e_bln_c_s,
            wgtfac_c=wgtfac_c,
            div_c=div_c,
            mech_prod=mech_prod,
            vertical_start=gtx.int32(0),
            vertical_start_interior=gtx.int32(1),
            vertical_end=gtx.int32(grid.num_levels),
            cell_start_nudging=gtx.int32(cell_start_nudging),
            cell_start_lateral_boundary_level_3=gtx.int32(cell_start_lateral_boundary_level_3),
            cell_end_halo=gtx.int32(cell_end_halo),
        )
