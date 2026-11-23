# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""States of the tmx turbulent mixing component."""

from __future__ import annotations

import dataclasses

import gt4py.next as gtx

from icon4py.model.common import (
    dimension as dims,
    field,
    field_type_aliases as fa,
    quantities as qty,
    state,
    type_alias as ta,
)


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
    inv_ddqz_z_half_e: fa.EdgeKHalfField[ta.wpfloat]
    """Inverse vertical distance between full levels, at edge midpoints on half levels [1/m]."""
    inv_ddqz_z_half_v: fa.VertexKHalfField[ta.wpfloat]
    """Inverse vertical distance between full levels, at vertices on half levels [1/m]."""
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
    """Air mass per unit area (`mair`) on full levels [kg/m^2]."""


@dataclasses.dataclass(frozen=True)
class TmxSurfaceFluxState:
    """The surface fluxes tmx reads: the bottom boundary of its vertical diffusion."""

    evapotranspiration: fa.CellField[ta.wpfloat]
    """Surface evapotranspiration flux (``evspsbl``) [kg/(m^2 s)]."""
    u_stress: fa.CellField[ta.wpfloat]
    """Zonal surface wind stress (``tauu``) [N/m^2]."""
    v_stress: fa.CellField[ta.wpfloat]
    """Meridional surface wind stress (``tauv``) [N/m^2]."""
    q_snocpymlt: fa.CellField[ta.wpfloat]
    """Heat used to melt snow on the canopy (``q_snocpymlt``), zero off land [W/m^2]."""
    thermal_energy_flux: fa.CellField[ta.wpfloat]
    """Surface flux of internal energy from thermal exchange (``ufts``) [W/m^2]."""
    vapor_energy_flux: fa.CellField[ta.wpfloat]
    """Surface flux of internal energy from vapor exchange (``ufvs``) [W/m^2]."""


class TmxDiagnosticState(state.State):
    """Diagnostic fields of tmx (``t_vdf_atmo_diags`` in mo_vdf_atmo_memory.f90)."""

    # cell, full levels
    cv_air: field.Field[qty.HeatCapacityOnCellK]
    """
    Isochoric heat capacity of the air per unit area (`cvair`) on full levels [J/(K m^2)].
    Computed here from the tracers and the air mass (``get_cvair`` in mo_aes_phy_diag.f90), where
    ICON passes it to tmx as the input ``cv_air_c``.
    """
    theta_v: field.Field[qty.ThetaVOnCellK]
    """Virtual potential temperature at cell centers on full levels [K]."""
    cptgz: field.Field[qty.DryStaticEnergyOnCellK]
    """Dry static energy cp*T + g*z at cell centers on full levels [J/kg]."""
    div_c: field.Field[qty.DivergenceOnCellK]
    """Horizontal wind divergence at cell centers on full levels [1/s]."""
    km_c: field.Field[qty.TurbulentViscosityOnCellK]
    """Turbulent viscosity at cell centers on full levels [kg/(m s)]."""
    km: field.Field[qty.ExchangeCoefficientForMomentumOnCellK]
    """
    Above the lowest level: turbulent viscosity from the half level below [kg/(m s)].
    At the lowest level: the surface exchange coefficient `km_sfc` [dimensionless], or
    `km_const` [m^2/s] when `use_km_const=True`.
    """
    kh: field.Field[qty.ExchangeCoefficientForHeatOnCellK]
    """
    Above the lowest level: turbulent diffusivity from the half level below [kg/(m s)].
    At the lowest level: the surface exchange coefficient `kh_sfc` [dimensionless], or
    `km_const / turb_prandtl` [m^2/s] when `use_km_const=True`.
    """
    dissip_ke: field.Field[qty.KineticEnergyDissipationOnCellK]
    """Kinetic energy dissipated by the wind diffusion, per layer [W/m^2]."""
    heating: field.Field[qty.TurbulentHeatingOnCellK]
    """Turbulent heating, per layer [W/m^2]."""
    # cell, vertically integrated
    cptgz_vi: field.Field[qty.VerticallyIntegratedDryStaticEnergyOnCell]
    """Vertical integral of the dry static energy of the updated state [J/m^2]."""
    dissip_ke_vi: field.Field[qty.VerticallyIntegratedKineticEnergyDissipationOnCell]
    """Vertical integral of the dissipated kinetic energy [W/m^2]."""
    int_energy_vi: field.Field[qty.VerticallyIntegratedInternalEnergyOnCell]
    """Vertical integral of the internal energy of the updated state [J/m^2]."""
    tend_int_energy_vi: field.Field[qty.TendencyOfVerticallyIntegratedInternalEnergyOnCell]
    """Tendency of the vertically integrated internal energy [W/m^2]."""
    # cell, half levels
    rho_ic: field.Field[qty.RhoOnCellKHalf]
    """Air density at cell centers on half levels [kg/m^3]."""
    bruvais: field.Field[qty.BruntVaisalaFrequencySquaredOnCellKHalf]
    """Brunt-Vaisala frequency squared at cell centers on half levels [1/s^2]."""
    mech_prod: field.Field[qty.MechanicalProductionOnCellKHalf]
    """Mechanical production term of turbulent kinetic energy on half levels [1/s^2]."""
    km_ic: field.Field[qty.TurbulentViscosityOnCellKHalf]
    """Turbulent viscosity at cell centers on half levels [kg/(m s)]."""
    kh_ic: field.Field[qty.TurbulentDiffusivityOnCellKHalf]
    """Turbulent diffusivity at cell centers on half levels [kg/(m s)]."""
    # edge, full levels
    vn: field.Field[qty.VnOnEdgeK]
    """Normal wind at edge midpoints on full levels [m/s]."""
    shear: field.Field[qty.HorizontalShearProductionOnEdgeK]
    """Horizontal shear production term at edge midpoints on full levels [1/s^2]."""
    div_of_stress: field.Field[qty.DivergenceOfStressOnEdgeK]
    """Divergence of the stress tensor at edge midpoints on full levels [1/s]."""
    # edge, half levels
    vn_ie: field.Field[qty.VnOnEdgeKHalf]
    """Normal wind at edge midpoints on half levels [m/s]."""
    vt_ie: field.Field[qty.TangentialWindOnEdgeKHalf]
    """Tangential wind at edge midpoints on half levels [m/s]."""
    w_ie: field.Field[qty.WOnEdgeKHalf]
    """Vertical wind at edge midpoints on half levels [m/s]."""
    km_ie: field.Field[qty.TurbulentViscosityOnEdgeKHalf]
    """Turbulent viscosity at edge midpoints on half levels [kg/(m s)]."""
    # vertex, full levels
    u_vert: field.Field[qty.UOnVertexK]
    """Zonal wind at vertices on full levels [m/s]."""
    v_vert: field.Field[qty.VOnVertexK]
    """Meridional wind at vertices on full levels [m/s]."""
    # vertex, half levels
    w_vert: field.Field[qty.WOnVertexKHalf]
    """Vertical wind at vertices on half levels [m/s]."""
    km_iv: field.Field[qty.TurbulentViscosityOnVertexKHalf]
    """Turbulent viscosity at vertices on half levels [kg/(m s)]."""


class TmxNewState(state.State):
    """Fields updated by tmx: `new = state + tend * dtime`."""

    temperature: field.Field[qty.TemperatureOnCellK]
    """Updated air temperature on full levels [K]."""
    qv: field.Field[qty.QvOnCellK]
    """Updated specific humidity on full levels [kg/kg]."""
    qc: field.Field[qty.QcOnCellK]
    """Updated cloud water mixing ratio on full levels [kg/kg]."""
    qi: field.Field[qty.QiOnCellK]
    """Updated cloud ice mixing ratio on full levels [kg/kg]."""
    u: field.Field[qty.UOnCellK]
    """Updated zonal wind on full levels [m/s]."""
    v: field.Field[qty.VOnCellK]
    """Updated meridional wind on full levels [m/s]."""
    w: field.Field[qty.WOnCellKHalf]
    """Updated vertical wind on half levels [m/s]."""


class TmxTendencyState(state.State):
    """Tendencies computed by tmx."""

    tend_temperature: field.Field[qty.TendencyOfTemperatureOnCellK]
    """Air temperature tendency on full levels [K/s]."""
    tend_qv: field.Field[qty.TendencyOfQvOnCellK]
    """Specific humidity tendency on full levels [kg/(kg s)]."""
    tend_qc: field.Field[qty.TendencyOfQcOnCellK]
    """Cloud water mixing ratio tendency on full levels [kg/(kg s)]."""
    tend_qi: field.Field[qty.TendencyOfQiOnCellK]
    """Cloud ice mixing ratio tendency on full levels [kg/(kg s)]."""
    tend_u: field.Field[qty.TendencyOfUOnCellK]
    """Zonal wind tendency on full levels [m/s^2]."""
    tend_v: field.Field[qty.TendencyOfVOnCellK]
    """Meridional wind tendency on full levels [m/s^2]."""
    tend_w: field.Field[qty.TendencyOfWOnCellKHalf]
    """Vertical wind tendency on half levels [m/s^2]."""
