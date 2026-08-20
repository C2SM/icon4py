# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    from collections.abc import Callable

    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


"""
State dataclasses of the TMX (AES turbulent mixing) scheme.

Ported from the memory classes in ICON's ``mo_vdf_atmo_memory.f90`` (see also
``mo_vdf.f90`` / ``mo_vdf_atmo.f90``). Full-level fields have ``num_levels``
vertical entries, half-level (interface) fields have ``num_levels + 1``.
"""


@dataclasses.dataclass(frozen=True)
class TmxMetricState:
    """Represents the metric (vertical grid) fields needed by tmx."""

    ddqz_z_full: fa.CellKField[ta.wpfloat]
    """Layer thickness at cell centers on full levels [m]."""
    inv_ddqz_z_full: fa.CellKField[ta.wpfloat]
    """Inverse layer thickness at cell centers on full levels [1/m]."""
    ddqz_z_half: fa.CellKField[ta.wpfloat]
    """Vertical distance between full levels, at cell centers on half levels [m]."""
    inv_ddqz_z_half: fa.CellKField[ta.wpfloat]
    """Inverse vertical distance between full levels, at cell centers on half levels [1/m]."""
    inv_ddqz_z_full_e: fa.EdgeKField[ta.wpfloat]
    """Inverse layer thickness at edge midpoints on full levels [1/m]."""
    inv_ddqz_z_half_e: fa.EdgeKField[ta.wpfloat]
    """Inverse vertical distance between full levels, at edge midpoints on half levels [1/m]."""
    inv_ddqz_z_half_v: fa.VertexKField[ta.wpfloat]
    """Inverse vertical distance between full levels, at vertices on half levels [1/m]."""
    wgtfac_c: fa.CellKField[ta.wpfloat]
    """Weighting factor for interpolation from full to half levels at cell centers (half levels)."""
    wgtfac_e: fa.EdgeKField[ta.wpfloat]
    """Weighting factor for interpolation from full to half levels at edge midpoints (half levels)."""
    wgtfacq_c: fa.CellKField[ta.wpfloat]
    """Extrapolation coefficients to the bottom surface half level at cell centers.

    Rows k = 0..2 used, in Fortran coefficient order: row k multiplies the full
    level nlev - 1 - k (``wgtfacq_c(jc, k+1, jb)`` in mo_vertical_grid.f90)."""
    wgtfacq1_c: fa.CellKField[ta.wpfloat]
    """Extrapolation coefficients to the top half level at cell centers.

    Rows k = 0..2 used, in Fortran coefficient order: row k multiplies the full
    level k (``wgtfacq1_c(jc, k+1, jb)`` in mo_vertical_grid.f90)."""
    wgtfacq_e: fa.EdgeKField[ta.wpfloat]
    """Extrapolation coefficients to the bottom surface half level at edges.

    Rows k = 0..2 used, in Fortran coefficient order: row k multiplies the full
    level nlev - 1 - k (``wgtfacq_e(je, k+1, jb)`` in mo_vertical_grid.f90)."""
    wgtfacq1_e: fa.EdgeKField[ta.wpfloat]
    """Extrapolation coefficients to the top half level at edges.

    Rows k = 0..2 used, in Fortran coefficient order: row k multiplies the full
    level k (``wgtfacq1_e(je, k+1, jb)`` in mo_vertical_grid.f90)."""
    geopot_agl_ifc: fa.CellKField[ta.wpfloat]
    """Geopotential above ground level at cell centers on half levels [m^2/s^2]."""
    z_mc: fa.CellKField[ta.wpfloat]
    """Geometric height at cell centers on full levels [m]."""
    z_ifc: fa.CellKField[ta.wpfloat]
    """Geometric height at cell centers on half levels [m]."""
    edge_cell_length: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat]
    """Distance between the edge midpoint and the circumcenters of the two
    adjacent cells [m] (``t_grid_edges%edge_cell_length`` in mo_model_domain.f90).

    A grid-geometry field carried here because it is not part of the common
    ``grid_states.EdgeParams`` (used by the horizontal w diffusion, Stage E)."""


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


@dataclasses.dataclass(frozen=True)
class TmxInputState:
    """Atmospheric input fields of tmx (``t_vdf_atmo_inputs`` in mo_vdf_atmo_memory.f90)."""

    temperature: fa.CellKField[ta.wpfloat]
    """Air temperature (``ta``) on full levels [K]."""
    virtual_temperature: fa.CellKField[ta.wpfloat]
    """Virtual temperature (``tv``) on full levels [K]."""
    pressure: fa.CellKField[ta.wpfloat]
    """Air pressure (``pa``) on full levels [Pa]."""
    pressure_ifc: fa.CellKField[ta.wpfloat]
    """Air pressure at interfaces (``pa_ifc``) on half levels [Pa]."""
    u: fa.CellKField[ta.wpfloat]
    """Zonal wind (``ua``) on full levels [m/s]."""
    v: fa.CellKField[ta.wpfloat]
    """Meridional wind (``va``) on full levels [m/s]."""
    w: fa.CellKField[ta.wpfloat]
    """Vertical wind (``wa``) on half levels [m/s]."""
    qv: fa.CellKField[ta.wpfloat]
    """Specific humidity on full levels [kg/kg]."""
    qc: fa.CellKField[ta.wpfloat]
    """Cloud water mixing ratio on full levels [kg/kg]."""
    qi: fa.CellKField[ta.wpfloat]
    """Cloud ice mixing ratio on full levels [kg/kg]."""
    qr: fa.CellKField[ta.wpfloat]
    """Rain mixing ratio on full levels [kg/kg]."""
    qs: fa.CellKField[ta.wpfloat]
    """Snow mixing ratio on full levels [kg/kg]."""
    qg: fa.CellKField[ta.wpfloat]
    """Graupel mixing ratio on full levels [kg/kg]."""
    rho: fa.CellKField[ta.wpfloat]
    """Air density on full levels [kg/m^3]."""
    air_mass: fa.CellKField[ta.wpfloat]
    """Air mass per unit area (``mair``) on full levels [kg/m^2]."""
    cv_air: fa.CellKField[ta.wpfloat]
    """Isometric specific heat of moist air (``cvair``) on full levels [J/(kg K)]."""


@dataclasses.dataclass(frozen=True)
class TmxSurfaceFluxState:
    """Surface fluxes provided by the surface scheme (inputs to the atmospheric diffusion)."""

    evapotranspiration: fa.CellField[ta.wpfloat]
    """Surface evapotranspiration flux (``evspsbl``) [kg/(m^2 s)]."""
    sensible_heat_flux: fa.CellField[ta.wpfloat]
    """Surface sensible heat flux (``hfss``) [W/m^2]."""
    u_stress: fa.CellField[ta.wpfloat]
    """Zonal surface wind stress (``tauu``) [N/m^2]."""
    v_stress: fa.CellField[ta.wpfloat]
    """Meridional surface wind stress (``tauv``) [N/m^2]."""
    q_snocpymlt: fa.CellField[ta.wpfloat]
    """Heating used to melt snow on the canopy [W/m^2]."""

    @classmethod
    def allocate(
        cls, grid: base_grid.Grid, allocator: gtx_typing.Allocator | None = None
    ) -> TmxSurfaceFluxState:
        """Allocate a surface flux state with all fields initialized to zero."""
        _, _, surface = _field_allocators(grid, allocator)
        return cls(
            evapotranspiration=surface(dims.CellDim),
            sensible_heat_flux=surface(dims.CellDim),
            u_stress=surface(dims.CellDim),
            v_stress=surface(dims.CellDim),
            q_snocpymlt=surface(dims.CellDim),
        )


@dataclasses.dataclass(frozen=True)
class TmxDiagnosticState:
    """Diagnostic fields of tmx (``t_vdf_atmo_diags`` in mo_vdf_atmo_memory.f90)."""

    # cell, full levels
    theta_v: fa.CellKField[ta.wpfloat]
    """Virtual potential temperature at cell centers on full levels [K]."""
    cptgz: fa.CellKField[ta.wpfloat]
    """Dry static energy cp*T + g*z at cell centers on full levels [J/kg]."""
    ghf: fa.CellKField[ta.wpfloat]
    """Geopotential height above ground at full levels [m]."""
    div_c: fa.CellKField[ta.wpfloat]
    """Horizontal wind divergence at cell centers on full levels [1/s]."""
    km_c: fa.CellKField[ta.wpfloat]
    """Turbulent viscosity at cell centers on full levels [m^2/s]."""
    km: fa.CellKField[ta.wpfloat]
    """Mass-weighted turbulent viscosity on full levels [kg/(m s)]."""
    kh: fa.CellKField[ta.wpfloat]
    """Mass-weighted turbulent diffusivity on full levels [kg/(m s)]."""
    heating: fa.CellKField[ta.wpfloat]
    """Turbulent heating rate at cell centers on full levels [W/m^2]."""
    dissip_ke: fa.CellKField[ta.wpfloat]
    """Kinetic energy dissipation rate at cell centers on full levels [W/m^2]."""
    # cell, half levels
    rho_ic: fa.CellKField[ta.wpfloat]
    """Air density at cell centers on half levels [kg/m^3]."""
    bruvais: fa.CellKField[ta.wpfloat]
    """Brunt-Vaisala frequency squared at cell centers on half levels [1/s^2]."""
    mech_prod: fa.CellKField[ta.wpfloat]
    """Mechanical production term of turbulent kinetic energy on half levels [m^2/s^2]."""
    km_ic: fa.CellKField[ta.wpfloat]
    """Turbulent viscosity at cell centers on half levels [m^2/s]."""
    kh_ic: fa.CellKField[ta.wpfloat]
    """Turbulent diffusivity at cell centers on half levels [m^2/s]."""
    mix_len_sq: fa.CellKField[ta.wpfloat]
    """Squared Smagorinsky mixing length at cell centers on half levels [m^2]."""
    # edge, full levels
    vn: fa.EdgeKField[ta.wpfloat]
    """Normal wind at edge midpoints on full levels [m/s]."""
    shear: fa.EdgeKField[ta.wpfloat]
    """Horizontal shear production term at edge midpoints on full levels [1/s^2]."""
    div_of_stress: fa.EdgeKField[ta.wpfloat]
    """Divergence of the stress tensor at edge midpoints on full levels [1/s]."""
    # edge, half levels
    vn_ie: fa.EdgeKField[ta.wpfloat]
    """Normal wind at edge midpoints on half levels [m/s]."""
    vt_ie: fa.EdgeKField[ta.wpfloat]
    """Tangential wind at edge midpoints on half levels [m/s]."""
    w_ie: fa.EdgeKField[ta.wpfloat]
    """Vertical wind at edge midpoints on half levels [m/s]."""
    km_ie: fa.EdgeKField[ta.wpfloat]
    """Turbulent viscosity at edge midpoints on half levels [m^2/s]."""
    # vertex, full levels
    u_vert: fa.VertexKField[ta.wpfloat]
    """Zonal wind at vertices on full levels [m/s]."""
    v_vert: fa.VertexKField[ta.wpfloat]
    """Meridional wind at vertices on full levels [m/s]."""
    # vertex, half levels
    w_vert: fa.VertexKField[ta.wpfloat]
    """Vertical wind at vertices on half levels [m/s]."""
    km_iv: fa.VertexKField[ta.wpfloat]
    """Turbulent viscosity at vertices on half levels [m^2/s]."""
    # cell, 2D (surface / vertically integrated)
    louis_factor: fa.CellField[ta.wpfloat]
    """Cell-area scaling factor of the Louis constant b (``scaling_factor_louis``)."""
    cptgz_vi: fa.CellField[ta.wpfloat]
    """Vertically integrated dry static energy [J/m^2]."""
    dissip_ke_vi: fa.CellField[ta.wpfloat]
    """Vertically integrated kinetic energy dissipation rate [W/m^2]."""
    int_energy_vi: fa.CellField[ta.wpfloat]
    """Vertically integrated internal energy [J/m^2]."""
    int_energy_vi_tend: fa.CellField[ta.wpfloat]
    """Tendency of the vertically integrated internal energy [W/m^2]."""

    @classmethod
    def allocate(
        cls, grid: base_grid.Grid, allocator: gtx_typing.Allocator | None = None
    ) -> TmxDiagnosticState:
        """Allocate a diagnostic state with all fields initialized to zero."""
        full, half, surface = _field_allocators(grid, allocator)
        return cls(
            theta_v=full(dims.CellDim),
            cptgz=full(dims.CellDim),
            ghf=full(dims.CellDim),
            div_c=full(dims.CellDim),
            km_c=full(dims.CellDim),
            km=full(dims.CellDim),
            kh=full(dims.CellDim),
            heating=full(dims.CellDim),
            dissip_ke=full(dims.CellDim),
            rho_ic=half(dims.CellDim),
            bruvais=half(dims.CellDim),
            mech_prod=half(dims.CellDim),
            km_ic=half(dims.CellDim),
            kh_ic=half(dims.CellDim),
            mix_len_sq=half(dims.CellDim),
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
            louis_factor=surface(dims.CellDim),
            cptgz_vi=surface(dims.CellDim),
            dissip_ke_vi=surface(dims.CellDim),
            int_energy_vi=surface(dims.CellDim),
            int_energy_vi_tend=surface(dims.CellDim),
        )


@dataclasses.dataclass(frozen=True)
class TmxNewState:
    """
    Updated prognostic fields after the tmx diffusion.

    Corresponds to the ``new_states`` of the (state, new_state, tendency)
    triples in mo_vdf.f90: ``new = state + tend * dtime``.
    """

    temperature: fa.CellKField[ta.wpfloat]
    """Updated air temperature on full levels [K]."""
    qv: fa.CellKField[ta.wpfloat]
    """Updated specific humidity on full levels [kg/kg]."""
    qc: fa.CellKField[ta.wpfloat]
    """Updated cloud water mixing ratio on full levels [kg/kg]."""
    qi: fa.CellKField[ta.wpfloat]
    """Updated cloud ice mixing ratio on full levels [kg/kg]."""
    u: fa.CellKField[ta.wpfloat]
    """Updated zonal wind on full levels [m/s]."""
    v: fa.CellKField[ta.wpfloat]
    """Updated meridional wind on full levels [m/s]."""
    w: fa.CellKField[ta.wpfloat]
    """Updated vertical wind on half levels [m/s]."""

    @classmethod
    def allocate(
        cls, grid: base_grid.Grid, allocator: gtx_typing.Allocator | None = None
    ) -> TmxNewState:
        """Allocate a new-state container with all fields initialized to zero."""
        full, half, _ = _field_allocators(grid, allocator)
        return cls(
            temperature=full(dims.CellDim),
            qv=full(dims.CellDim),
            qc=full(dims.CellDim),
            qi=full(dims.CellDim),
            u=full(dims.CellDim),
            v=full(dims.CellDim),
            w=half(dims.CellDim),
        )


@dataclasses.dataclass(frozen=True)
class TmxTendencyState:
    """Tendencies computed by tmx."""

    ddt_temperature: fa.CellKField[ta.wpfloat]
    """Air temperature tendency on full levels [K/s]."""
    ddt_qv: fa.CellKField[ta.wpfloat]
    """Specific humidity tendency on full levels [kg/(kg s)]."""
    ddt_qc: fa.CellKField[ta.wpfloat]
    """Cloud water mixing ratio tendency on full levels [kg/(kg s)]."""
    ddt_qi: fa.CellKField[ta.wpfloat]
    """Cloud ice mixing ratio tendency on full levels [kg/(kg s)]."""
    ddt_u: fa.CellKField[ta.wpfloat]
    """Zonal wind tendency on full levels [m/s^2]."""
    ddt_v: fa.CellKField[ta.wpfloat]
    """Meridional wind tendency on full levels [m/s^2]."""
    ddt_w: fa.CellKField[ta.wpfloat]
    """Vertical wind tendency on half levels [m/s^2]."""

    @classmethod
    def allocate(
        cls, grid: base_grid.Grid, allocator: gtx_typing.Allocator | None = None
    ) -> TmxTendencyState:
        """Allocate a tendency state with all fields initialized to zero."""
        full, half, _ = _field_allocators(grid, allocator)
        return cls(
            ddt_temperature=full(dims.CellDim),
            ddt_qv=full(dims.CellDim),
            ddt_qc=full(dims.CellDim),
            ddt_qi=full(dims.CellDim),
            ddt_u=full(dims.CellDim),
            ddt_v=full(dims.CellDim),
            ddt_w=half(dims.CellDim),
        )


def _field_allocators(
    grid: base_grid.Grid, allocator: gtx_typing.Allocator | None
) -> tuple[
    Callable[[gtx.Dimension], gtx.Field],
    Callable[[gtx.Dimension], gtx.Field],
    Callable[[gtx.Dimension], gtx.Field],
]:
    """Return zero-field factories for full-level, half-level and surface (2D) fields."""

    def full(horizontal_dim: gtx.Dimension) -> gtx.Field:
        return data_alloc.zero_field(
            grid, horizontal_dim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        )

    def half(horizontal_dim: gtx.Dimension) -> gtx.Field:
        return data_alloc.zero_field(
            grid,
            horizontal_dim,
            dims.KDim,
            extend={dims.KDim: 1},
            dtype=ta.wpfloat,
            allocator=allocator,
        )

    def surface(horizontal_dim: gtx.Dimension) -> gtx.Field:
        return data_alloc.zero_field(grid, horizontal_dim, dtype=ta.wpfloat, allocator=allocator)

    return full, half, surface
