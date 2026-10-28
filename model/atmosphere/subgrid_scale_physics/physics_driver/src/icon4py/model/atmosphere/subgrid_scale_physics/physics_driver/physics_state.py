# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""What the physics driver offers its processes each step."""

import datetime

from icon4py.model.common import framework as fw
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

    vn: fw.Field[qty.VnOnEdgeK]
    w: fw.Field[qty.WOnCellKHalf]
    exner: fw.Field[qty.ExnerOnCellK]
    theta_v: fw.Field[qty.ThetaVOnCellK]
    rho: fw.Field[qty.RhoOnCellK]
    qv: fw.Field[qty.QvOnCellK]
    qc: fw.Field[qty.QcOnCellK]
    qi: fw.Field[qty.QiOnCellK]
    qr: fw.Field[qty.QrOnCellK]
    qs: fw.Field[qty.QsOnCellK]
    qg: fw.Field[qty.QgOnCellK]
    temperature: fw.Field[qty.TemperatureOnCellK]
    virtual_temperature: fw.Field[qty.VirtualTemperatureOnCellK]
    pressure: fw.Field[qty.PressureOnCellK]
    pressure_ifc: fw.Field[qty.PressureOnCellKHalf]
    u: fw.Field[qty.UOnCellK]
    v: fw.Field[qty.VOnCellK]
    step_start: datetime.datetime = datetime.datetime.min
