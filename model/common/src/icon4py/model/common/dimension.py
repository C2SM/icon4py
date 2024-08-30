# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import TypeAlias, Union

from gt4py.next.common import DimensionKind
from gt4py.next.ffront.fbuiltins import Dimension, FieldOffset


KDim = Dimension("K", kind=DimensionKind.VERTICAL)
KHalfDim = Dimension("KHalf", kind=DimensionKind.VERTICAL)
EdgeDim = Dimension("Edge")
CellDim = Dimension("Cell")
VertexDim = Dimension("Vertex")
global_dimensions = {"CellDim": CellDim, "EdgeDim": EdgeDim, "VertexDim": VertexDim}
CEDim = Dimension("CE")
CECDim = Dimension("CEC")
ECDim = Dimension("EC")
ECVDim = Dimension("ECV")
CECECDim = Dimension("CECEC")
E2CDim = Dimension("E2C", DimensionKind.LOCAL)
E2VDim = Dimension("E2V", DimensionKind.LOCAL)
C2EDim = Dimension("C2E", DimensionKind.LOCAL)
V2CDim = Dimension("V2C", DimensionKind.LOCAL)
C2VDim = Dimension("C2V", DimensionKind.LOCAL)
V2EDim = Dimension("V2E", DimensionKind.LOCAL)
V2E2VDim = Dimension("V2E2V", DimensionKind.LOCAL)
E2C2VDim = Dimension("E2C2V", DimensionKind.LOCAL)
C2E2CODim = Dimension("C2E2CO", DimensionKind.LOCAL)
E2C2EODim = Dimension("E2C2EO", DimensionKind.LOCAL)
E2C2EDim = Dimension("E2C2E", DimensionKind.LOCAL)
C2E2CDim = Dimension("C2E2C", DimensionKind.LOCAL)
C2E2C2EDim = Dimension("C2E2C2E", DimensionKind.LOCAL)
C2E2C2E2CDim = Dimension("C2E2C2E2C", DimensionKind.LOCAL)
E2C = FieldOffset("E2C", source=CellDim, target=(EdgeDim, E2CDim))
C2E = FieldOffset("C2E", source=EdgeDim, target=(CellDim, C2EDim))
V2C = FieldOffset("V2C", source=CellDim, target=(VertexDim, V2CDim))
C2V = FieldOffset("C2V", source=VertexDim, target=(CellDim, C2VDim))
V2E = FieldOffset("V2E", source=EdgeDim, target=(VertexDim, V2EDim))
E2V = FieldOffset("E2V", source=VertexDim, target=(EdgeDim, E2VDim))
C2CE = FieldOffset("C2CE", source=CEDim, target=(CellDim, C2EDim))
C2CEC = FieldOffset("C2CEC", source=CECDim, target=(CellDim, C2E2CDim))
E2EC = FieldOffset("E2EC", source=ECDim, target=(EdgeDim, E2CDim))
E2ECV = FieldOffset("E2ECV", source=ECVDim, target=(EdgeDim, E2C2VDim))
E2C2V = FieldOffset("E2C2V", source=VertexDim, target=(EdgeDim, E2C2VDim))
C2E2CO = FieldOffset("C2E2CO", source=CellDim, target=(CellDim, C2E2CODim))
E2C2EO = FieldOffset("E2C2EO", source=EdgeDim, target=(EdgeDim, E2C2EODim))
E2C2E = FieldOffset("E2C2E", source=EdgeDim, target=(EdgeDim, E2C2EDim))
C2E2C = FieldOffset("C2E2C", source=CellDim, target=(CellDim, C2E2CDim))
C2E2C2E = FieldOffset("C2E2C2E", source=EdgeDim, target=(CellDim, C2E2C2EDim))
C2E2C2E2C = FieldOffset("C2E2C2E2C", source=CellDim, target=(CellDim, C2E2C2E2CDim))
C2CECEC = FieldOffset("C2CECEC", source=CECECDim, target=(CellDim, C2E2C2E2CDim))
V2E2V = FieldOffset("V2E2V", source=VertexDim, target=(VertexDim, V2E2VDim))
Koff = FieldOffset("Koff", source=KDim, target=(KDim,))
KHalfOff = FieldOffset("KHalfOff", source=KHalfDim, target=(KHalfDim,))


VerticalDim: TypeAlias = Union[KDim, KHalfDim]
