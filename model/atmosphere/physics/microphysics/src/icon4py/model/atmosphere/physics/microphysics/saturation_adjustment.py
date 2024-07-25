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
import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import (
    abs,
    exp,
    maximum,
    where,
    broadcast,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import xp, backend
from gt4py.next.program_processors.runners.gtfn import (
    run_gtfn_cached,
    run_gtfn_gpu_cached,
    run_gtfn_imperative,
)
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.grid import horizontal as h_grid, vertical as v_grid
from icon4py.model.common.states import prognostic_state as prognostics, diagnostic_state as diagnostics, tracer_state as tracers
from icon4py.model.atmosphere.dycore.state_utils.utils import _allocate
from typing import Final
from gt4py.eve.utils import FrozenNamespace
import dataclasses
from icon4py.model.common import constants
from icon4py.model.common.type_alias import vpfloat, wpfloat


# TODO (Chia Rui): move parameters below to common/constants? But they need to be declared under FrozenNameSpace to be used in the big graupel scan operator.
class ConvectTables(FrozenNamespace):
    """Constants used for the computation in saturation adjustment."""

    #: Latent heat of vaporisation for water [J/kg]
    alv: wpfloat = 2.5008e6
    #: Latent heat of sublimation for water [J/kg]
    als: wpfloat = 2.8345e6
    #: Melting temperature of ice/snow [K]
    tmelt: wpfloat = 273.15

    rd: wpfloat = constants.RD
    rv: wpfloat = constants.RV
    cpd: wpfloat = constants.CPD
    cvd: wpfloat = constants.CVD
    #: cp_d / cp_l - 1
    rcpl: wpfloat = 3.1733
    #: Specific heat capacity of liquid water
    clw: wpfloat = (rcpl + 1.0) * cpd

    c1es: wpfloat = 610.78
    c2es: wpfloat = c1es * rd / rv
    c3les: wpfloat = 17.269
    c3ies: wpfloat = 21.875
    c4les: wpfloat = 35.86
    c4ies: wpfloat = 7.66
    c5les: wpfloat = c3les * (tmelt - c4les)
    c5ies: wpfloat = c3ies * (tmelt - c4ies)
    c5alvcp: wpfloat = c5les * alv / cpd
    c5alscp: wpfloat = c5ies * als / cpd
    alvdcp: wpfloat = alv / cpd
    alsdcp: wpfloat = als / cpd


# Instantiate the class
conv_table: Final = ConvectTables()


@dataclasses.dataclass(frozen=True)
class SaturationAdjustmentConfig:
    max_iter: int = 10
    tolerance: float = 1.e-3


class ConvergenceError(Exception):
    pass


class SaturationAdjustment:

    def __init__(self, config: SaturationAdjustmentConfig, grid: icon_grid.IconGrid):
        self.grid = grid
        self.config = config
        self._allocate_tendencies()

    def _allocate_tendencies(self):
        self.new_temperature1 = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.new_temperature2 = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.temperature_tendency = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.qv_tendency = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.qc_tendency = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)
        self.subsaturated_mask = _allocate(CellDim, KDim, grid=self.grid, dtype=bool)
        self.newton_iteration_mask = _allocate(CellDim, KDim, grid=self.grid, dtype=bool)
        self.lwdocvd = _allocate(CellDim, KDim, grid=self.grid, dtype=vpfloat)

    def run(
        self,
        dtime: float,
        prognostic_state: prognostics.PrognosticState,
        diagnostic_state: diagnostics.DiagnosticState,
        tracer_state: tracers.TracerState
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
        end_cell_local = self.grid.get_end_index(CellDim, h_grid.HorizontalMarkerIndex.local(CellDim))

        compute_subsaturated_case_and_initialize_newton_iterations(
            self.config.tolerance,
            diagnostic_state.temperature,
            tracer_state.qv,
            tracer_state.qc,
            prognostic_state.rho,
            self.subsaturated_mask,
            self.lwdocvd,
            self.new_temperature1,
            self.new_temperature2,
            self.newton_iteration_mask,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        temperature_list = [self.new_temperature1, self.new_temperature2]
        ncurrent, nnext = 0, 1
        for _ in range(self.config.max_iter):
            if xp.any(self.newton_iteration_mask.ndarray[start_cell_nudging:end_cell_local, 0:self.grid.num_levels]):
                compute_temperature_by_newton_iteration(
                    diagnostic_state.temperature,
                    tracer_state.qv,
                    prognostic_state.rho,
                    self.newton_iteration_mask,
                    self.lwdocvd,
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
                    self.newton_iteration_mask,
                    horizontal_start=start_cell_nudging,
                    horizontal_end=end_cell_local,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )

                copy_temperature(
                    self.newton_iteration_mask,
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
        if xp.any(self.newton_iteration_mask.ndarray[start_cell_nudging:end_cell_local, 0:self.grid.num_levels]):
            raise ConvergenceError(f"Maximum iteration of saturation adjustment ({self.config.max_iter}) is not enough. The max absolute error is {xp.abs(self.new_temperature1.ndarray - self.new_temperature2.ndarray).max()} . Please raise max_iter")
        update_temperature_qv_qc_tendencies(
            dtime,
            diagnostic_state.temperature,
            temperature_list[ncurrent],
            tracer_state.qv,
            tracer_state.qc,
            prognostic_state.rho,
            self.subsaturated_mask,
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
    t: gtx.Field[[CellDim, KDim], float]
) -> gtx.Field[[CellDim, KDim], float]:
    """Return latent heat of vaporization.

    Computed as internal energy and taking into account Kirchoff's relations
    """
    # specific heat of water vapor at constant pressure (Landolt-Bornstein)
    #cp_v = 1850.0

    return (
        conv_table.alv
        + (1850.0 - conv_table.clw) * (t - conv_table.tmelt)
        - conv_table.rv * t
    )


@gtx.field_operator
def _sat_pres_water(t: gtx.Field[[CellDim, KDim], float]) -> gtx.Field[[CellDim, KDim], float]:
    """Return saturation water vapour pressure."""
    return conv_table.c1es * exp(
        conv_table.c3les * (t - conv_table.tmelt) / (t - conv_table.c4les)
    )


@gtx.field_operator
def _qsat_rho(
    t: gtx.Field[[CellDim, KDim], float], rho: gtx.Field[[CellDim, KDim], float]
) -> gtx.Field[[CellDim, KDim], float]:
    """Return specific humidity at water saturation (with respect to flat surface)."""
    return _sat_pres_water(t) / (rho * conv_table.rv * t)


@gtx.field_operator
def _dqsatdT_rho(
    t: gtx.Field[[CellDim, KDim], float], zqsat: gtx.Field[[CellDim, KDim], float]
) -> gtx.Field[[CellDim, KDim], float]:
    """
    Return partial derivative of the specific humidity at water saturation.

    Computed with respect to the temperature at constant total density.
    """
    beta = conv_table.c5les / (t - conv_table.c4les) ** 2 - 1.0 / t
    return beta * zqsat


@gtx.field_operator
def _new_temperature_in_newton_iteration(
    temperature: gtx.Field[[CellDim, KDim], float],
    qv: gtx.Field[[CellDim, KDim], float],
    rho: gtx.Field[[CellDim, KDim], float],
    lwdocvd: gtx.Field[[CellDim, KDim], float],
    new_temperature2: gtx.Field[[CellDim, KDim], float]
) -> gtx.Field[[CellDim, KDim], float]:

    ft = new_temperature2 - temperature + lwdocvd * (_qsat_rho(new_temperature2, rho) - qv)
    dft = 1.0 + lwdocvd * _dqsatdT_rho(new_temperature2, _qsat_rho(new_temperature2, rho))

    return new_temperature2 - ft / dft


@gtx.field_operator
def _compute_temperature_by_newton_iteration(
    temperature: gtx.Field[[CellDim, KDim], float],
    qv: gtx.Field[[CellDim, KDim], float],
    rho: gtx.Field[[CellDim, KDim], float],
    newton_iteration_mask: gtx.Field[[CellDim, KDim], bool],
    lwdocvd: gtx.Field[[CellDim, KDim], float],
    new_temperature2: gtx.Field[[CellDim, KDim], float]
) -> gtx.Field[[CellDim, KDim], float]:
    new_temperature1 = where(
        newton_iteration_mask,
        _new_temperature_in_newton_iteration(temperature, qv, rho, lwdocvd, new_temperature2),
        new_temperature2
    )
    return new_temperature1

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def compute_temperature_by_newton_iteration(
    temperature: gtx.Field[[CellDim, KDim], float],
    qv: gtx.Field[[CellDim, KDim], float],
    rho: gtx.Field[[CellDim, KDim], float],
    newton_iteration_mask: gtx.Field[[CellDim, KDim], bool],
    lwdocvd: gtx.Field[[CellDim, KDim], float],
    new_temperature2: gtx.Field[[CellDim, KDim], float],
    new_temperature1: gtx.Field[[CellDim, KDim], float],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_temperature_by_newton_iteration(
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
    dtime: float,
    temperature: gtx.Field[[CellDim, KDim], float],
    temperature_next: gtx.Field[[CellDim, KDim], float],
    qv: gtx.Field[[CellDim, KDim], float],
    qc: gtx.Field[[CellDim, KDim], float],
    rho: gtx.Field[[CellDim, KDim], float],
    subsaturated_mask: gtx.Field[[CellDim, KDim], bool]
) -> tuple[gtx.Field[[CellDim, KDim], float], gtx.Field[[CellDim, KDim], float], gtx.Field[[CellDim, KDim], float]]:
    # Local treshold
    zqwmin = 1e-20
    qv_tendency, qc_tendency = where(
        subsaturated_mask,
        (qc / dtime, -qc / dtime),
        ((_qsat_rho(temperature_next, rho) - qv) / dtime , (maximum(qv + qc - _qsat_rho(temperature_next, rho), zqwmin) - qc) / dtime),
    )
    return (temperature_next - temperature) / dtime, qv_tendency, qc_tendency

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def update_temperature_qv_qc_tendencies(
    dtime: float,
    temperature: gtx.Field[[CellDim, KDim], float],
    temperature_next: gtx.Field[[CellDim, KDim], float],
    qv: gtx.Field[[CellDim, KDim], float],
    qc: gtx.Field[[CellDim, KDim], float],
    rho: gtx.Field[[CellDim, KDim], float],
    subsaturated_mask: gtx.Field[[CellDim, KDim], bool],
    temperature_tendency: gtx.Field[[CellDim, KDim], float],
    qv_tendency: gtx.Field[[CellDim, KDim], float],
    qc_tendency: gtx.Field[[CellDim, KDim], float],
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
    tolerance: float,
    temperature: gtx.Field[[CellDim, KDim], float],
    qv: gtx.Field[[CellDim, KDim], float],
    qc: gtx.Field[[CellDim, KDim], float],
    rho: gtx.Field[[CellDim, KDim], float],
) -> tuple[gtx.Field[[CellDim, KDim], bool], gtx.Field[[CellDim, KDim], float], gtx.Field[[CellDim, KDim], float], gtx.Field[[CellDim, KDim], float], gtx.Field[[CellDim, KDim], bool]]:

    temperature_after_all_qc_evaporated = (
        temperature - _latent_heat_vaporization(temperature) / conv_table.cvd * qc
    )

    # Check, which points will still be subsaturated even after evaporating all cloud water.
    subsaturated_mask = qv + qc <= _qsat_rho(temperature_after_all_qc_evaporated, rho)

    # Remains const. during iteration
    lwdocvd = _latent_heat_vaporization(temperature) / conv_table.cvd

    new_temperature1 = where(subsaturated_mask, temperature_after_all_qc_evaporated, temperature - 2.0 * tolerance)
    new_temperature2 = where(subsaturated_mask, temperature_after_all_qc_evaporated, temperature)
    newton_iteration_mask = where(subsaturated_mask, False, True)

    return subsaturated_mask, lwdocvd, new_temperature1, new_temperature2, newton_iteration_mask

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def compute_subsaturated_case_and_initialize_newton_iterations(
    tolerance: float,
    temperature: gtx.Field[[CellDim, KDim], float],
    qv: gtx.Field[[CellDim, KDim], float],
    qc: gtx.Field[[CellDim, KDim], float],
    rho: gtx.Field[[CellDim, KDim], float],
    subsaturated_mask: gtx.Field[[CellDim, KDim], bool],
    lwdocvd: gtx.Field[[CellDim, KDim], float],
    new_temperature1: gtx.Field[[CellDim, KDim], float],
    new_temperature2: gtx.Field[[CellDim, KDim], float],
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
    tolerance: float,
    temperature_current: gtx.Field[[CellDim, KDim], float],
    temperature_next: gtx.Field[[CellDim, KDim], float],
) -> gtx.Field[[CellDim, KDim], bool]:
    return where(abs(temperature_current - temperature_next) > tolerance,True,False)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def compute_newton_iteration_mask(
    tolerance: float,
    temperature_current: gtx.Field[[CellDim, KDim], float],
    temperature_next: gtx.Field[[CellDim, KDim], float],
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
    temperature_current: gtx.Field[[CellDim, KDim], float],
) -> gtx.Field[[CellDim, KDim], float]:
    return where(newton_iteration_mask, 0.0, temperature_current)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def copy_temperature(
    newton_iteration_mask: gtx.Field[[CellDim, KDim], bool],
    temperature_current: gtx.Field[[CellDim, KDim], float],
    temperature_next: gtx.Field[[CellDim, KDim], float],
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
