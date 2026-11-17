# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.common import dimension as dims, quantity


CELL_K = (dims.CellDim, dims.KDim)


class Pressure(quantity.Quantity, standard_name="air_pressure", units="Pa", dims=CELL_K): ...


class Salt(quantity.Quantity, units="1", dims=(dims.CellDim,)): ...


class Vt(quantity.Quantity, units="m s-1", dims=(dims.EdgeDim, dims.KDim), precision="vp"): ...


class TendencyOfPressure(quantity.Tendency, units="Pa s-1", dims=CELL_K): ...
