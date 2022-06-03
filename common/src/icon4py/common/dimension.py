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

from functional.ffront.fbuiltins import Dimension, FieldOffset


KDim = Dimension("K")
EdgeDim = Dimension("Edge")
CellDim = Dimension("Cell")
VertexDim = Dimension("Vertex")
C2EDim = Dimension("C2E", True)
V2CDim = Dimension("V2C", True)
V2EDim = Dimension("V2E", True)
C2E2CODim = Dimension("C2E2CO", True)
E2C2EODim = Dimension("E2C2EO", True)
E2C2EDim = Dimension("E2C2E", True)
C2E = FieldOffset("C2E", source=EdgeDim, target=(CellDim, C2EDim))
V2C = FieldOffset("V2C", source=CellDim, target=(VertexDim, V2CDim))
V2E = FieldOffset("V2E", source=EdgeDim, target=(VertexDim, V2EDim))
C2E2CO = FieldOffset("C2E2CO", source=CellDim, target=(CellDim, C2E2CODim))
E2C2EO = FieldOffset("E2C2EO", source=EdgeDim, target=(EdgeDim, E2C2EODim))
E2C2E = FieldOffset("E2C2E", source=EdgeDim, target=(EdgeDim, E2C2EDim))
