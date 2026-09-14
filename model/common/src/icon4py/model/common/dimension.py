# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Iterator

import gt4py.next as gtx
from gt4py.next import common as gtx_common


class KDim(gtx.DimensionIndex, kind=gtx.DimensionKind.VERTICAL):
    tag = "K"


KHalfDim = gtx.flip_staggered(KDim)


class EdgeDim(gtx.DimensionIndex):
    tag = "Edge"


class CellDim(gtx.DimensionIndex):
    tag = "Cell"


class VertexDim(gtx.DimensionIndex):
    tag = "Vertex"


class LsqUnkDim(gtx.DimensionIndex, kind=gtx.DimensionKind.LOCAL):
    tag = "LsqUnk"


class E2CDim(gtx.DimensionIndex, kind=gtx.DimensionKind.LOCAL):
    tag = "E2C"


class E2VDim(gtx.DimensionIndex, kind=gtx.DimensionKind.LOCAL):
    tag = "E2V"


class C2EDim(gtx.DimensionIndex, kind=gtx.DimensionKind.LOCAL):
    tag = "C2E"


class V2CDim(gtx.DimensionIndex, kind=gtx.DimensionKind.LOCAL):
    tag = "V2C"


class C2VDim(gtx.DimensionIndex, kind=gtx.DimensionKind.LOCAL):
    tag = "C2V"


class V2EDim(gtx.DimensionIndex, kind=gtx.DimensionKind.LOCAL):
    tag = "V2E"


class V2E2VDim(gtx.DimensionIndex, kind=gtx.DimensionKind.LOCAL):
    tag = "V2E2V"


class E2C2VDim(gtx.DimensionIndex, kind=gtx.DimensionKind.LOCAL):
    tag = "E2C2V"


class C2E2CODim(gtx.DimensionIndex, kind=gtx.DimensionKind.LOCAL):
    tag = "C2E2CO"


class E2C2EODim(gtx.DimensionIndex, kind=gtx.DimensionKind.LOCAL):
    tag = "E2C2EO"


class E2C2EDim(gtx.DimensionIndex, kind=gtx.DimensionKind.LOCAL):
    tag = "E2C2E"


class C2E2CDim(gtx.DimensionIndex, kind=gtx.DimensionKind.LOCAL):
    tag = "C2E2C"


class C2E2C2EDim(gtx.DimensionIndex, kind=gtx.DimensionKind.LOCAL):
    tag = "C2E2C2E"


class C2E2C2E2CDim(gtx.DimensionIndex, kind=gtx.DimensionKind.LOCAL):
    tag = "C2E2C2E2C"


E2C = gtx.FieldOffset("E2C", source=CellDim, target=(EdgeDim, E2CDim))
C2E = gtx.FieldOffset("C2E", source=EdgeDim, target=(CellDim, C2EDim))
V2C = gtx.FieldOffset("V2C", source=CellDim, target=(VertexDim, V2CDim))
C2V = gtx.FieldOffset("C2V", source=VertexDim, target=(CellDim, C2VDim))
V2E = gtx.FieldOffset("V2E", source=EdgeDim, target=(VertexDim, V2EDim))
E2V = gtx.FieldOffset("E2V", source=VertexDim, target=(EdgeDim, E2VDim))
E2C2V = gtx.FieldOffset("E2C2V", source=VertexDim, target=(EdgeDim, E2C2VDim))
C2E2CO = gtx.FieldOffset("C2E2CO", source=CellDim, target=(CellDim, C2E2CODim))
E2C2EO = gtx.FieldOffset("E2C2EO", source=EdgeDim, target=(EdgeDim, E2C2EODim))
E2C2E = gtx.FieldOffset("E2C2E", source=EdgeDim, target=(EdgeDim, E2C2EDim))
C2E2C = gtx.FieldOffset("C2E2C", source=CellDim, target=(CellDim, C2E2CDim))
C2E2C2E = gtx.FieldOffset("C2E2C2E", source=EdgeDim, target=(CellDim, C2E2C2EDim))
C2E2C2E2C = gtx.FieldOffset("C2E2C2E2C", source=CellDim, target=(CellDim, C2E2C2E2CDim))
V2E2V = gtx.FieldOffset("V2E2V", source=VertexDim, target=(VertexDim, V2E2VDim))
Koff = gtx.FieldOffset("Koff", source=KDim, target=(KDim,))


def horizontal_dims() -> Iterator[gtx.Dimension]:
    return iter(
        tuple(
            d
            for d in globals().values()
            if isinstance(d, gtx_common.DimensionMeta) and d.kind == gtx.DimensionKind.HORIZONTAL
        )
    )


def non_horizontal_dims() -> Iterator[gtx.Dimension]:
    yield from vertical_dims()
    yield from local_dims()


def local_dims() -> Iterator[gtx.Dimension]:
    for d in globals().values():
        if isinstance(d, gtx_common.DimensionMeta) and d.kind == gtx.DimensionKind.LOCAL:
            yield d


def vertical_dims() -> Iterator[gtx.Dimension]:
    return iter(
        tuple(
            d
            for d in globals().values()
            if isinstance(d, gtx_common.DimensionMeta) and d.kind == gtx.DimensionKind.VERTICAL
        )
    )
