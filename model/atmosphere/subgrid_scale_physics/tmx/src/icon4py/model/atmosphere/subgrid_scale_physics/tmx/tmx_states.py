# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""States of the tmx turbulent mixing granule."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


@dataclasses.dataclass(frozen=True)
class TmxMetricState:
    """Represents the metric (vertical grid) fields needed by tmx."""

    ddqz_z_full: fa.CellKField[ta.wpfloat]
    """Layer thickness at cell centers on full levels [m]."""
    inv_ddqz_z_full: fa.CellKField[ta.wpfloat]
    """Inverse layer thickness at cell centers on full levels [1/m]."""
    ddqz_z_half: fa.CellKHalfField[ta.wpfloat]
    """Vertical distance between full levels, at cell centers on half levels [m]."""
    inv_ddqz_z_half: fa.CellKHalfField[ta.wpfloat]
    """Inverse vertical distance between full levels, at cell centers on half levels [1/m]."""
    inv_ddqz_z_full_e: fa.EdgeKField[ta.wpfloat]
    """Inverse layer thickness at edge midpoints on full levels [1/m]."""
    wgtfac_c: fa.CellKHalfField[ta.wpfloat]
    """Weighting factor for interpolation from full to half levels at cell centers (half levels)."""
    wgtfac_e: fa.EdgeKHalfField[ta.wpfloat]
    """Weighting factor for interpolation from full to half levels at edge midpoints (half levels)."""
    wgtfacq_c: fa.CellKField[ta.wpfloat]
    """Quadratic extrapolation coefficients to the surface level at cell centers."""
    wgtfacq1_c: fa.CellKField[ta.wpfloat]
    """Quadratic extrapolation coefficients to the model top level at cell centers."""
    wgtfacq_e: fa.EdgeKField[ta.wpfloat]
    """Quadratic extrapolation coefficients to the surface level at edges."""
    wgtfacq1_e: fa.EdgeKField[ta.wpfloat]
    """Quadratic extrapolation coefficients to the model top level at edges."""
    geopot_agl_ifc: fa.CellKHalfField[ta.wpfloat]
    """Geopotential above ground level at cell centers on half levels [m^2/s^2]."""
    height_above_ground: fa.CellKField[ta.wpfloat]
    """Geometric height of the full levels above the surface [m] (``ghf``)."""


@dataclasses.dataclass(frozen=True)
class TmxInterpolationState:
    """Represents the ICON interpolation coefficients needed by tmx."""

    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat]
    """Coefficients for linear interpolation from cell centers to edge midpoints."""
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat]
    """Coefficients for bilinear interpolation from edge midpoints to cell centers."""
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat]
    """Geometric factors for the cell-centered divergence of an edge-normal vector field."""
    cells_aw_verts: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat]
    """Coefficients for area-weighted interpolation from cell centers to vertices."""
    rbf_coeff_v1: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat]
    """RBF coefficients for the zonal wind component at vertices (rbf_vec_coeff_v_1)."""
    rbf_coeff_v2: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat]
    """RBF coefficients for the meridional wind component at vertices (rbf_vec_coeff_v_2)."""
    rbf_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat]
    """RBF coefficients for the tangential wind component at edges (rbf_vec_coeff_e)."""
    rbf_coeff_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], ta.wpfloat]
    """RBF coefficients for the zonal wind component at cell centers (rbf_vec_coeff_c_1)."""
    rbf_coeff_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], ta.wpfloat]
    """RBF coefficients for the meridional wind component at cell centers (rbf_vec_coeff_c_2)."""


def _field_allocators(
    grid: base_grid.Grid, allocator: gtx_typing.Allocator | None
) -> tuple[
    Callable[[gtx.Dimension], gtx.Field],
    Callable[[gtx.Dimension], gtx.Field],
]:
    """Return zero-field factories for full-level and half-level fields."""

    def full(horizontal_dim: gtx.Dimension) -> gtx.Field:
        return data_alloc.zero_field(
            grid, horizontal_dim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        )

    def half(horizontal_dim: gtx.Dimension) -> gtx.Field:
        return data_alloc.zero_field(
            grid, horizontal_dim, dims.KHalfDim, dtype=ta.wpfloat, allocator=allocator
        )

    return full, half


@dataclasses.dataclass(frozen=True)
class TmxInputState:
    """Atmospheric input fields of tmx (``t_vdf_atmo_inputs`` in mo_vdf_atmo_memory.f90)."""

    temperature: fa.CellKField[ta.wpfloat]
    """Air temperature (``ta``) on full levels [K]."""
    virtual_temperature: fa.CellKField[ta.wpfloat]
    """Virtual temperature (``tv``) on full levels [K]."""
    pressure: fa.CellKField[ta.wpfloat]
    """Air pressure (``pa``) on full levels [Pa]."""
    u: fa.CellKField[ta.wpfloat]
    """Zonal wind (``ua``) on full levels [m/s]."""
    v: fa.CellKField[ta.wpfloat]
    """Meridional wind (``va``) on full levels [m/s]."""
    w: fa.CellKHalfField[ta.wpfloat]
    """Vertical wind (``wa``) on half levels [m/s]."""
    rho: fa.CellKField[ta.wpfloat]
    """Air density on full levels [kg/m^3]."""


@dataclasses.dataclass(frozen=True)
class TmxDiagnosticState:
    """Diagnostic fields of tmx (``t_vdf_atmo_diags`` in mo_vdf_atmo_memory.f90)."""

    # cell, full levels
    theta_v: fa.CellKField[ta.wpfloat]
    """Virtual potential temperature at cell centers on full levels [K]."""
    cptgz: fa.CellKField[ta.wpfloat]
    """Dry static energy cp*T + g*z at cell centers on full levels [J/kg]."""
    div_c: fa.CellKField[ta.wpfloat]
    """Horizontal wind divergence at cell centers on full levels [1/s]."""
    km_c: fa.CellKField[ta.wpfloat]
    """Turbulent viscosity at cell centers on full levels [kg/(m s)]."""
    # cell, half levels
    rho_ic: fa.CellKHalfField[ta.wpfloat]
    """Air density at cell centers on half levels [kg/m^3]."""
    bruvais: fa.CellKHalfField[ta.wpfloat]
    """Brunt-Vaisala frequency squared at cell centers on half levels [1/s^2]."""
    mech_prod: fa.CellKHalfField[ta.wpfloat]
    """Mechanical production term of turbulent kinetic energy on half levels [1/s^2]."""
    km_ic: fa.CellKHalfField[ta.wpfloat]
    """Turbulent viscosity at cell centers on half levels [kg/(m s)]."""
    kh_ic: fa.CellKHalfField[ta.wpfloat]
    """Turbulent diffusivity at cell centers on half levels [kg/(m s)]."""
    # edge, full levels
    vn: fa.EdgeKField[ta.wpfloat]
    """Normal wind at edge midpoints on full levels [m/s]."""
    shear: fa.EdgeKField[ta.wpfloat]
    """Horizontal shear production term at edge midpoints on full levels [1/s^2]."""
    div_of_stress: fa.EdgeKField[ta.wpfloat]
    """Divergence of the stress tensor at edge midpoints on full levels [1/s]."""
    # edge, half levels
    vn_ie: fa.EdgeKHalfField[ta.wpfloat]
    """Normal wind at edge midpoints on half levels [m/s]."""
    vt_ie: fa.EdgeKHalfField[ta.wpfloat]
    """Tangential wind at edge midpoints on half levels [m/s]."""
    w_ie: fa.EdgeKHalfField[ta.wpfloat]
    """Vertical wind at edge midpoints on half levels [m/s]."""
    km_ie: fa.EdgeKHalfField[ta.wpfloat]
    """Turbulent viscosity at edge midpoints on half levels [kg/(m s)]."""
    # vertex, full levels
    u_vert: fa.VertexKField[ta.wpfloat]
    """Zonal wind at vertices on full levels [m/s]."""
    v_vert: fa.VertexKField[ta.wpfloat]
    """Meridional wind at vertices on full levels [m/s]."""
    # vertex, half levels
    w_vert: fa.VertexKHalfField[ta.wpfloat]
    """Vertical wind at vertices on half levels [m/s]."""
    km_iv: fa.VertexKHalfField[ta.wpfloat]
    """Turbulent viscosity at vertices on half levels [kg/(m s)]."""

    @classmethod
    def allocate(
        cls, grid: base_grid.Grid, allocator: gtx_typing.Allocator | None = None
    ) -> TmxDiagnosticState:
        """Allocate a diagnostic state with all fields initialized to zero."""
        full, half = _field_allocators(grid, allocator)
        return cls(
            theta_v=full(dims.CellDim),
            cptgz=full(dims.CellDim),
            div_c=full(dims.CellDim),
            km_c=full(dims.CellDim),
            rho_ic=half(dims.CellDim),
            bruvais=half(dims.CellDim),
            mech_prod=half(dims.CellDim),
            km_ic=half(dims.CellDim),
            kh_ic=half(dims.CellDim),
            vn=full(dims.EdgeDim),
            shear=full(dims.EdgeDim),
            div_of_stress=full(dims.EdgeDim),
            vn_ie=half(dims.EdgeDim),
            vt_ie=half(dims.EdgeDim),
            w_ie=half(dims.EdgeDim),
            km_ie=half(dims.EdgeDim),
            u_vert=full(dims.VertexDim),
            v_vert=full(dims.VertexDim),
            w_vert=half(dims.VertexDim),
            km_iv=half(dims.VertexDim),
        )
