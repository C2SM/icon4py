# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""What the physics driver offers its processes each step."""

from icon4py.model.common import field, framework as fw, time
from icon4py.model.common.framework import quantities as qty


# NOTE: temporary, components MWE part 02 (collect): the processes collect from the physics
# driver's `Input` and `Diagnostics` directly, and the entry state goes.
class EntryState(fw.State):
    """
    The prognostics the physics driver received, the diagnostics it derived from them, and the
    start of the step being integrated.

    ICON's dyn2phy: every field leaf is a view onto the driver's input or onto its `diagnostics`,
    no copies; `step_start` is derived from the input instead: the end of the step being
    integrated less its length. Diagnosed once per step and never updated between the
    processes, which all read the same entry state (parallel coupling). Each process picks its
    component's input from here by hand (its `collect_input`); its time control reads
    `step_start`.
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
    step_start: time.AbsoluteTime
