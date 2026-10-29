# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.common import dimension as dims, quantity


CELL_K = (dims.CellDim, dims.KDim)


class Pressure(quantity.Quantity, dims=CELL_K, standard_name="air_pressure", units="Pa"): ...


class Salt(quantity.Quantity, dims=(dims.CellDim,), units="1"): ...


class Vt(quantity.Quantity, dims=(dims.EdgeDim, dims.KDim), units="m s-1", precision="vp"): ...
