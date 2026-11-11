# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""What the physics driver offers its processes each step."""

from typing import Final

from icon4py.model.common import field, framework as fw, time
from icon4py.model.common.framework import quantities as qty


# NOTE: temporary, components MWE part 02 (collect): the processes collect from the physics
# driver's `Input`, its `Diagnostics` and the other processes' `Output` directly, and the
# physics state goes.
class PhysicsState(fw.State):
    """
    The state the physics processes read and advance in one step (ICON's dyn2phy).

    Every field leaf is a view, no copies: onto the driver's input (the prognostics and the
    air mass), onto its output (w, the tracers and the diagnostics), or, for the surface
    fluxes, onto the output of the surface process. `step_start` is derived from the input instead: the end of
    the step being integrated less its length.

    The coupling is sequential, as in ICON AES: after each process the driver adds its
    tendencies to the temperature, u, v, w and the tracers in place, so the next process reads
    the advanced values. The virtual temperature and the pressure stay at step start. Each
    process picks its component's input from here by hand (its `collect_input`); its time
    control reads `step_start`.
    """

    vn: field.Field[qty.VnOnEdgeK]
    w: field.Field[qty.WOnCellKHalf]
    exner: field.Field[qty.ExnerOnCellK]
    theta_v: field.Field[qty.ThetaVOnCellK]
    rho: field.Field[qty.RhoOnCellK]
    qv: field.Field[qty.QvOnCellK]
    qc: field.Field[qty.QcOnCellK]
    qi: field.Field[qty.QiOnCellK]
    qr: field.Field[qty.QrOnCellK]
    qs: field.Field[qty.QsOnCellK]
    qg: field.Field[qty.QgOnCellK]
    temperature: field.Field[qty.TemperatureOnCellK]
    virtual_temperature: field.Field[qty.VirtualTemperatureOnCellK]
    pressure: field.Field[qty.PressureOnCellK]
    pressure_ifc: field.Field[qty.PressureOnCellKHalf]
    u: field.Field[qty.UOnCellK]
    v: field.Field[qty.VOnCellK]
    air_mass: field.Field[qty.AirMassOnCellK] | None = None
    evapotranspiration: field.Field[qty.EvapotranspirationOnCell] | None = None
    sensible_heat_flux: field.Field[qty.SensibleHeatFluxOnCell] | None = None
    latent_heat_flux: field.Field[qty.LatentHeatFluxOnCell] | None = None
    u_stress: field.Field[qty.UStressOnCell] | None = None
    v_stress: field.Field[qty.VStressOnCell] | None = None
    q_snocpymlt: field.Field[qty.SnowCanopyMeltHeatOnCell] | None = None
    thermal_energy_flux: field.Field[qty.ThermalEnergyFluxOnCell] | None = None
    vapor_energy_flux: field.Field[qty.VaporEnergyFluxOnCell] | None = None
    step_start: time.AbsoluteTime


# the leaves of the physics state that a process's output provides (the surface fluxes, which tmx
# reads), one provider each
PROVIDED_BY_PROCESSES: Final = (
    "evapotranspiration",
    "sensible_heat_flux",
    "latent_heat_flux",
    "u_stress",
    "v_stress",
    "q_snocpymlt",
    "thermal_energy_flux",
    "vapor_energy_flux",
)
