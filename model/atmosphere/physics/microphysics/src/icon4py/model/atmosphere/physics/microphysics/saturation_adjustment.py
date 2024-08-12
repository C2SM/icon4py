# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Changes.

- Only implemented Tetens (ipsat = 1). Dropped Murphy-Koop.
- Harmonized name of constants
- Only implementend gpu version. Maybe further optimizations possible for CPU (check original code)

TODO:
1. Implement Newtonian iteration! -> Needs fixted-size for loop feature in GT4Py

Comment from FORTRAN version:
- Suggested by U. Blahak: Replace pres_sat_water, pres_sat_ice and spec_humi by
lookup tables in mo_convect_tables. Bit incompatible change!
"""
import dataclasses
from typing import Final

import gt4py.next as gtx
from gt4py.eve.utils import FrozenNamespace
from gt4py.next.ffront.fbuiltins import (
    abs,
    exp,
    maximum,
    where,
)

from icon4py.model.common import constants
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.settings import backend, xp
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
    tracer_state as tracers,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc


# TODO (Chia Rui): Refactor this class when there is consensus in gt4py team about the best way to express compile-time constants
class ConvectTables(FrozenNamespace):
    """Constants used for the computation in saturation adjustment."""

    #: Latent heat of vaporisation for water [J/kg]
    vaporisation_latent_heat: wpfloat = 2.5008e6
    #: Melting temperature of ice/snow [K]
    tmelt: wpfloat = 273.15

    rd: wpfloat = constants.RD
    rv: wpfloat = constants.RV
    cvd: wpfloat = constants.CVD
    cpd: wpfloat = constants.CPD

    #: cpd / cpl - 1
    rcpl: wpfloat = 3.1733
    #: Specific heat capacity of liquid water
    spec_heat_cap_water: wpfloat = (rcpl + 1.0) * cpd

    #: p0 in Tetens formula for saturation water pressure, see eq. 5.33 in COSMO documentation.
    tetens_p0: wpfloat = 610.78
    #: aw in Tetens formula for saturation water pressure
    tetens_aw: wpfloat = 17.269
    #: bw in Tetens formula for saturation water pressure
    tetens_bw: wpfloat = 35.86
    #: numerator in temperature partial derivative of Tetens formula for saturation water pressure (psat tetens_der / (t - tetens_bw)^2)
    tetens_der: wpfloat = tetens_aw * (tmelt - tetens_bw)


# Instantiate the class
conv_table: Final = ConvectTables()


@dataclasses.dataclass(frozen=True)
class SaturationAdjustmentConfig:
    max_iter: int = 10
    tolerance: wpfloat = 1.0e-3


class ConvergenceError(Exception):
    pass


class SaturationAdjustment:
    def __init__(self, config: SaturationAdjustmentConfig, grid: icon_grid.IconGrid):
        self.grid = grid
        self.config = config
        self._allocate_tendencies()

    def _allocate_tendencies(self):
        self._new_temperature1 = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=vpfloat
        )
        self._new_temperature2 = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=vpfloat
        )
        self._subsaturated_mask = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=bool
        )
        self._newton_iteration_mask = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=bool
        )
        self._lwdocvd = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=vpfloat
        )
        # TODO (Chia Rui): remove the tendency terms below when architecture of the entire phyiscs component is ready to use.
        self.temperature_tendency = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=vpfloat
        )
        self.qv_tendency = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=vpfloat
        )
        self.qc_tendency = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=vpfloat
        )

    def run(
        self,
        dtime: wpfloat,
        prognostic_state: prognostics.PrognosticState,
        diagnostic_state: diagnostics.DiagnosticState,
        tracer_state: tracers.TracerState,
    ):
        """
        Adjust saturation at each grid point.

        Synopsis:
        Saturation adjustment condenses/evaporates specific humidity (qv) into/from
        cloud water content (qc) such that a gridpoint is just saturated. Temperature (t)
        is adapted accordingly and pressure adapts itself in ICON.

        Method:
        Saturation adjustment at constant total density (adjustment of T and p accordingly)
        assuming chemical equilibrium of water and vapor. For the heat capacity of
        of the total system (dry air, vapor, and hydrometeors) the value of dry air
        is taken, which is a common approximation and introduces only a small error.

        Originally inspirered from satad_v_3D_gpu of ICON release 2.6.4.
        """

        start_cell_nudging = self.grid.get_start_index(
            CellDim, h_grid.HorizontalMarkerIndex.nudging(CellDim)
        )
        end_cell_local = self.grid.get_end_index(
            CellDim, h_grid.HorizontalMarkerIndex.local(CellDim)
        )

        compute_subsaturated_case_and_initialize_newton_iterations(
            self.config.tolerance,
            diagnostic_state.temperature,
            tracer_state.qv,
            tracer_state.qc,
            prognostic_state.rho,
            self._subsaturated_mask,
            self._lwdocvd,
            self._new_temperature1,
            self._new_temperature2,
            self._newton_iteration_mask,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        # TODO (Chia Rui): refactor this code when break and for loop features are ready in gt4py.
        temperature_list = [self._new_temperature1, self._new_temperature2]
        ncurrent, nnext = 0, 1
        for _ in range(self.config.max_iter):
            if xp.any(
                self._newton_iteration_mask.ndarray[
                    start_cell_nudging:end_cell_local, 0 : self.grid.num_levels
                ]
            ):
                update_temperature_by_newton_iteration(
                    diagnostic_state.temperature,
                    tracer_state.qv,
                    prognostic_state.rho,
                    self._newton_iteration_mask,
                    self._lwdocvd,
                    temperature_list[nnext],
                    temperature_list[ncurrent],
                    horizontal_start=start_cell_nudging,
                    horizontal_end=end_cell_local,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )

                compute_newton_iteration_mask(
                    self.config.tolerance,
                    temperature_list[ncurrent],
                    temperature_list[nnext],
                    self._newton_iteration_mask,
                    horizontal_start=start_cell_nudging,
                    horizontal_end=end_cell_local,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )

                copy_temperature(
                    self._newton_iteration_mask,
                    temperature_list[ncurrent],
                    temperature_list[nnext],
                    horizontal_start=start_cell_nudging,
                    horizontal_end=end_cell_local,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )
                ncurrent = (ncurrent + 1) % 2
                nnext = (nnext + 1) % 2
            else:
                break
        if xp.any(
            self._newton_iteration_mask.ndarray[
                start_cell_nudging:end_cell_local, 0 : self.grid.num_levels
            ]
        ):
            raise ConvergenceError(
                f"Maximum iteration of saturation adjustment ({self.config.max_iter}) is not enough. The max absolute error is {xp.abs(self.new_temperature1.ndarray - self.new_temperature2.ndarray).max()} . Please raise max_iter"
            )
        update_temperature_qv_qc_tendencies(
            dtime,
            diagnostic_state.temperature,
            temperature_list[ncurrent],
            tracer_state.qv,
            tracer_state.qc,
            prognostic_state.rho,
            self._subsaturated_mask,
            self.temperature_tendency,
            self.qv_tendency,
            self.qc_tendency,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )


@gtx.field_operator
def _latent_heat_vaporization(
    t: gtx.Field[[CellDim, KDim], vpfloat],
) -> gtx.Field[[CellDim, KDim], vpfloat]:
    """
    Compute the latent heat of vaporisation with Kirchoff's relations (users can refer to Pruppacher and Klett textbook).
        dL/dT ~= cpv - cpw + v dp/dT
        L ~= (cpv - cpw) (T - T0) - Rv T

    Args:
        t: temperature [K]
    Returns:
        latent heat of vaporization.
    """
    return (
        conv_table.vaporisation_latent_heat
        + (wpfloat("1850.0") - conv_table.spec_heat_cap_water) * (t - conv_table.tmelt)
        - conv_table.rv * t
    )


@gtx.field_operator
def _sat_pres_water(t: gtx.Field[[CellDim, KDim], vpfloat]) -> gtx.Field[[CellDim, KDim], vpfloat]:
    """
    Compute saturation water vapour pressure by the Tetens formula.
        psat = p0 exp( aw (T-T0)/(T-bw)) )  [Tetens formula]

    Args:
        t: temperature [K]
    Returns:
        saturation water vapour pressure.
    """
    return conv_table.tetens_p0 * exp(
        conv_table.tetens_aw * (t - conv_table.tmelt) / (t - conv_table.tetens_bw)
    )


@gtx.field_operator
def _qsat_rho(
    t: gtx.Field[[CellDim, KDim], vpfloat], rho: gtx.Field[[CellDim, KDim], vpfloat]
) -> gtx.Field[[CellDim, KDim], vpfloat]:
    """
    Compute specific humidity at water saturation (with respect to flat surface).
        qsat = Rd/Rv psat/(p - psat) ~= Rd/Rv psat/p = 1/Rv psat/(rho T)
    Tetens formula is used for saturation water pressure (psat).
        psat = p0 exp( aw (T-T0)/(T-bw)) )  [Tetens formula]

    Args:
        t: temperature [K]
        rho: total air density (including hydrometeors) [kg m-3]
    Returns:
        specific humidity at water saturation.
    """
    return _sat_pres_water(t) / (rho * conv_table.rv * t)


@gtx.field_operator
def _dqsatdT_rho(
    t: gtx.Field[[CellDim, KDim], vpfloat], zqsat: gtx.Field[[CellDim, KDim], vpfloat]
) -> gtx.Field[[CellDim, KDim], vpfloat]:
    """
    Compute the partical derivative of the specific humidity at water saturation (qsat) with respect to the temperature at
    constant total density. qsat is approximated as
        qsat = Rd/Rv psat/(p - psat) ~= Rd/Rv psat/p = 1/Rv psat/(rho T)
    Tetens formula is used for saturation water pressure (psat).
        psat = p0 exp( aw (T-T0)/(T-bw)) )  [Tetens formula]
    FInally, the derivative with respect to temperature is
        dpsat/dT = psat (T0-bw)/(T-bw)^2
        dqsat/dT = 1/Rv psat/(rho T) (T0-bw)/(T-bw)^2 - 1/Rv psat/(rho T^2) = qsat ((T0-bw)/(T-bw)^2 - 1/T)

    Args:
        t: temperature [K]
        zqsat: saturated water mixing ratio
    Returns:
        partial derivative of the specific humidity at water saturation.
    """
    beta = conv_table.tetens_der / (t - conv_table.tetens_bw) ** 2 - wpfloat("1.0") / t
    return beta * zqsat


@gtx.field_operator
def _new_temperature_in_newton_iteration(
    temperature: gtx.Field[[CellDim, KDim], vpfloat],
    qv: gtx.Field[[CellDim, KDim], vpfloat],
    rho: gtx.Field[[CellDim, KDim], vpfloat],
    lwdocvd: gtx.Field[[CellDim, KDim], vpfloat],
    new_temperature2: gtx.Field[[CellDim, KDim], vpfloat],
) -> gtx.Field[[CellDim, KDim], vpfloat]:
    """
    Update the temperature in saturation adjustment by Newton iteration. Moist enthalpy and mass are conserved.
    The latent heat is assumed to be constant with its value computed from the initial temperature.
        T + Lv / cpd qv = TH, qv + qc = QTOT
        T = TH - Lv / cpd qsat(T), which is a transcendental function. Newton method is applied to solve it for T.
        f(T) = Lv / cpd qsat(T) + T - TH
        f'(T) = Lv / cpd dqsat(T)/dT + 1
        T_new = T - f(T)/f'(T) = ( TH - Lv / cpd (qsat(T) - T dqsat(T)/dT) ) / (Lv / cpd dqsat(T)/dT + 1)

    Args:
        temperature: initial temperature [K]
        qv: specific humidity
        rho: total air density [kg m-3]
        lwdocvd:
        new_temperature2: temperature at previous iteration [K]
    Returns:
        updated temperature [K]
    """
    ft = new_temperature2 - temperature + lwdocvd * (_qsat_rho(new_temperature2, rho) - qv)
    dft = vpfloat("1.0") + lwdocvd * _dqsatdT_rho(
        new_temperature2, _qsat_rho(new_temperature2, rho)
    )

    return new_temperature2 - ft / dft


@gtx.field_operator
def _update_temperature_by_newton_iteration(
    temperature: gtx.Field[[CellDim, KDim], vpfloat],
    qv: gtx.Field[[CellDim, KDim], vpfloat],
    rho: gtx.Field[[CellDim, KDim], vpfloat],
    newton_iteration_mask: gtx.Field[[CellDim, KDim], bool],
    lwdocvd: gtx.Field[[CellDim, KDim], vpfloat],
    new_temperature2: gtx.Field[[CellDim, KDim], vpfloat],
) -> gtx.Field[[CellDim, KDim], vpfloat]:
    new_temperature1 = where(
        newton_iteration_mask,
        _new_temperature_in_newton_iteration(temperature, qv, rho, lwdocvd, new_temperature2),
        new_temperature2,
    )
    return new_temperature1


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def update_temperature_by_newton_iteration(
    temperature: gtx.Field[[CellDim, KDim], vpfloat],
    qv: gtx.Field[[CellDim, KDim], vpfloat],
    rho: gtx.Field[[CellDim, KDim], vpfloat],
    newton_iteration_mask: gtx.Field[[CellDim, KDim], bool],
    lwdocvd: gtx.Field[[CellDim, KDim], vpfloat],
    new_temperature2: gtx.Field[[CellDim, KDim], vpfloat],
    new_temperature1: gtx.Field[[CellDim, KDim], vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _update_temperature_by_newton_iteration(
        temperature,
        qv,
        rho,
        newton_iteration_mask,
        lwdocvd,
        new_temperature2,
        out=new_temperature1,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _update_temperature_qv_qc_tendencies(
    dtime: wpfloat,
    temperature: gtx.Field[[CellDim, KDim], vpfloat],
    temperature_next: gtx.Field[[CellDim, KDim], vpfloat],
    qv: gtx.Field[[CellDim, KDim], vpfloat],
    qc: gtx.Field[[CellDim, KDim], vpfloat],
    rho: gtx.Field[[CellDim, KDim], vpfloat],
    subsaturated_mask: gtx.Field[[CellDim, KDim], bool],
) -> tuple[
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
]:
    """
    Compute temperature, qv, and qc tendencies from the saturation adjustment.

    Args:
        dtime: time step
        temperature: initial temperature [K]
        temperature_next: temperature updated by saturation adjustment [K]
        qv: initial specific humidity
        qc: initial cloud specific mixing ratio
        rho: total air density [kg m-3]
        subsaturated_mask: a mask where the air is subsaturated even if all cloud particles evaporate
    Returns:
        (updated temperature - initial temperautre) / dtime [K s-1],
        (saturated specific humidity - initial specific humidity) / dtime [s-1],
        (total specific mixing ratio - saturated specific humidity - initial cloud specific mixing ratio) / dtime [s-1],
    """
    zqwmin = wpfloat("1e-20")
    qv_tendency, qc_tendency = where(
        subsaturated_mask,
        (qc / dtime, -qc / dtime),
        (
            (_qsat_rho(temperature_next, rho) - qv) / dtime,
            (maximum(qv + qc - _qsat_rho(temperature_next, rho), zqwmin) - qc) / dtime,
        ),
    )
    return (temperature_next - temperature) / dtime, qv_tendency, qc_tendency


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def update_temperature_qv_qc_tendencies(
    dtime: wpfloat,
    temperature: gtx.Field[[CellDim, KDim], vpfloat],
    temperature_next: gtx.Field[[CellDim, KDim], vpfloat],
    qv: gtx.Field[[CellDim, KDim], vpfloat],
    qc: gtx.Field[[CellDim, KDim], vpfloat],
    rho: gtx.Field[[CellDim, KDim], vpfloat],
    subsaturated_mask: gtx.Field[[CellDim, KDim], bool],
    temperature_tendency: gtx.Field[[CellDim, KDim], vpfloat],
    qv_tendency: gtx.Field[[CellDim, KDim], vpfloat],
    qc_tendency: gtx.Field[[CellDim, KDim], vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _update_temperature_qv_qc_tendencies(
        dtime,
        temperature,
        temperature_next,
        qv,
        qc,
        rho,
        subsaturated_mask,
        out=(temperature_tendency, qv_tendency, qc_tendency),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_subsaturated_case_and_initialize_newton_iterations(
    tolerance: wpfloat,
    temperature: gtx.Field[[CellDim, KDim], vpfloat],
    qv: gtx.Field[[CellDim, KDim], vpfloat],
    qc: gtx.Field[[CellDim, KDim], vpfloat],
    rho: gtx.Field[[CellDim, KDim], vpfloat],
) -> tuple[
    gtx.Field[[CellDim, KDim], bool],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], vpfloat],
    gtx.Field[[CellDim, KDim], bool],
]:
    """
    Preparation for saturation adjustment.
    First obtain the subsaturated case, where the saturation specific humidity is larger than qv + qc. This can be
    derived by computing saturation specific humidity at the temperature after all cloud particles are evaporated.
    If this is the case, the new temperature is simply the temperature after all cloud particles are evaporated, and
    qv_new = qv + qc, qc = 0.
    All the remaining grid cells are marked as newton_iteration_mask for which Newton iteration is required to solve
    for the new temperature.

    Args:
        tolerance: tolerance for convergence in Newton iteration
        temperature: initial temperature [K]
        qv: initial specific humidity
        qc: initial cloud specific mixing ratio
        rho: total air density [kg m-3]
    Returns:
        mask for subsaturated case,
        Lv / cvd,
        current temperature for starting the Newton iteration,
        next temperature for starting the Newton iteration,
        mask for Newton iteration case
    """
    temperature_after_all_qc_evaporated = (
        temperature - _latent_heat_vaporization(temperature) / conv_table.cvd * qc
    )

    # Check, which points will still be subsaturated even after evaporating all cloud water.
    subsaturated_mask = qv + qc <= _qsat_rho(temperature_after_all_qc_evaporated, rho)

    # Remains const. during iteration
    lwdocvd = _latent_heat_vaporization(temperature) / conv_table.cvd

    new_temperature1 = where(
        subsaturated_mask,
        temperature_after_all_qc_evaporated,
        temperature - vpfloat("2.0") * tolerance,
    )
    new_temperature2 = where(subsaturated_mask, temperature_after_all_qc_evaporated, temperature)
    newton_iteration_mask = where(subsaturated_mask, False, True)

    return subsaturated_mask, lwdocvd, new_temperature1, new_temperature2, newton_iteration_mask


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def compute_subsaturated_case_and_initialize_newton_iterations(
    tolerance: wpfloat,
    temperature: gtx.Field[[CellDim, KDim], vpfloat],
    qv: gtx.Field[[CellDim, KDim], vpfloat],
    qc: gtx.Field[[CellDim, KDim], vpfloat],
    rho: gtx.Field[[CellDim, KDim], vpfloat],
    subsaturated_mask: gtx.Field[[CellDim, KDim], bool],
    lwdocvd: gtx.Field[[CellDim, KDim], vpfloat],
    new_temperature1: gtx.Field[[CellDim, KDim], vpfloat],
    new_temperature2: gtx.Field[[CellDim, KDim], vpfloat],
    newton_iteration_mask: gtx.Field[[CellDim, KDim], bool],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_subsaturated_case_and_initialize_newton_iterations(
        tolerance,
        temperature,
        qv,
        qc,
        rho,
        out=(subsaturated_mask, lwdocvd, new_temperature1, new_temperature2, newton_iteration_mask),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_newton_iteration_mask(
    tolerance: wpfloat,
    temperature_current: gtx.Field[[CellDim, KDim], vpfloat],
    temperature_next: gtx.Field[[CellDim, KDim], vpfloat],
) -> gtx.Field[[CellDim, KDim], bool]:
    return where(abs(temperature_current - temperature_next) > tolerance, True, False)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def compute_newton_iteration_mask(
    tolerance: wpfloat,
    temperature_current: gtx.Field[[CellDim, KDim], vpfloat],
    temperature_next: gtx.Field[[CellDim, KDim], vpfloat],
    newton_iteration_mask: gtx.Field[[CellDim, KDim], bool],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_newton_iteration_mask(
        tolerance,
        temperature_current,
        temperature_next,
        out=newton_iteration_mask,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _copy_temperature(
    newton_iteration_mask: gtx.Field[[CellDim, KDim], bool],
    temperature_current: gtx.Field[[CellDim, KDim], vpfloat],
) -> gtx.Field[[CellDim, KDim], vpfloat]:
    return where(newton_iteration_mask, vpfloat("0.0"), temperature_current)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def copy_temperature(
    newton_iteration_mask: gtx.Field[[CellDim, KDim], bool],
    temperature_current: gtx.Field[[CellDim, KDim], vpfloat],
    temperature_next: gtx.Field[[CellDim, KDim], vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _copy_temperature(
        newton_iteration_mask,
        temperature_current,
        out=temperature_next,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
