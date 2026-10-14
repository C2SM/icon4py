# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""What the physics driver offers its processes each step."""

from icon4py.model.common.components import framework as fw, quantities as qty


class PhysicsState(fw.State):
    """
    What the physics processes read: the prognostics, the diagnostics derived from them
    (ICON's dyn2phy) and the provisional updates of the processes run so far.

    The processes are coupled sequentially, as ICON AES with forcing control fc = 1: after each
    process the driver advances `temperature`, the tracers, `u`, `v` and `w` by dt times the
    process's tendencies, so the next process reads them updated (mo_interface_cloud_mig.f90:
    "update physics state for input to the next physics process"). Those leaves are the
    driver's own buffers, reset to the entry values each step; the others are views onto the
    driver's input or its `diagnostics`, never advanced. Each process picks its input from here
    by hand (`collect_input` next to its component).
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
