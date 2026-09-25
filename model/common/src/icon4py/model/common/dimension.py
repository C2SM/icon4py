# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import typing
from collections.abc import Iterator

import gt4py.next as gtx


class KDim(gtx.DimensionIndex, kind=gtx.DimensionKind.VERTICAL): ...


KHalfDim = gtx.flip_staggered(KDim)


class EdgeDim(gtx.DimensionIndex): ...


class CellDim(gtx.DimensionIndex): ...


class VertexDim(gtx.DimensionIndex): ...


class LsqUnkDim(gtx.LocalDimensionIndex): ...


class E2CDim(gtx.LocalDimensionIndex): ...


class E2VDim(gtx.LocalDimensionIndex): ...


class C2EDim(gtx.LocalDimensionIndex): ...


class V2CDim(gtx.LocalDimensionIndex): ...


class C2VDim(gtx.LocalDimensionIndex): ...


class V2EDim(gtx.LocalDimensionIndex): ...


class V2E2VDim(gtx.LocalDimensionIndex): ...


class E2C2VDim(gtx.LocalDimensionIndex): ...


class C2E2CODim(gtx.LocalDimensionIndex): ...


class E2C2EODim(gtx.LocalDimensionIndex): ...


class E2C2EDim(gtx.LocalDimensionIndex): ...


class C2E2CDim(gtx.LocalDimensionIndex): ...


class C2E2C2EDim(gtx.LocalDimensionIndex): ...


class C2E2C2E2CDim(gtx.LocalDimensionIndex): ...


class E2C(gtx.NeighborConnectivity[EdgeDim, CellDim]):
    Local: typing.TypeAlias = E2CDim  # noqa: UP040


class C2E(gtx.NeighborConnectivity[CellDim, EdgeDim]):
    Local: typing.TypeAlias = C2EDim  # noqa: UP040


class V2C(gtx.NeighborConnectivity[VertexDim, CellDim]):
    Local: typing.TypeAlias = V2CDim  # noqa: UP040


class C2V(gtx.NeighborConnectivity[CellDim, VertexDim]):
    Local: typing.TypeAlias = C2VDim  # noqa: UP040


class V2E(gtx.NeighborConnectivity[VertexDim, EdgeDim]):
    Local: typing.TypeAlias = V2EDim  # noqa: UP040


class E2V(gtx.NeighborConnectivity[EdgeDim, VertexDim]):
    Local: typing.TypeAlias = E2VDim  # noqa: UP040


class E2C2V(gtx.NeighborConnectivity[EdgeDim, VertexDim]):
    Local: typing.TypeAlias = E2C2VDim  # noqa: UP040


class C2E2CO(gtx.NeighborConnectivity[CellDim, CellDim]):
    Local: typing.TypeAlias = C2E2CODim  # noqa: UP040


class E2C2EO(gtx.NeighborConnectivity[EdgeDim, EdgeDim]):
    Local: typing.TypeAlias = E2C2EODim  # noqa: UP040


class E2C2E(gtx.NeighborConnectivity[EdgeDim, EdgeDim]):
    Local: typing.TypeAlias = E2C2EDim  # noqa: UP040


class C2E2C(gtx.NeighborConnectivity[CellDim, CellDim]):
    Local: typing.TypeAlias = C2E2CDim  # noqa: UP040


class C2E2C2E(gtx.NeighborConnectivity[CellDim, EdgeDim]):
    Local: typing.TypeAlias = C2E2C2EDim  # noqa: UP040


class C2E2C2E2C(gtx.NeighborConnectivity[CellDim, CellDim]):
    Local: typing.TypeAlias = C2E2C2E2CDim  # noqa: UP040


class V2E2V(gtx.NeighborConnectivity[VertexDim, VertexDim]):
    Local: typing.TypeAlias = V2E2VDim  # noqa: UP040


def horizontal_dims() -> Iterator[gtx.Dimension]:
    return iter(
        tuple(
            d
            for d in globals().values()
            if isinstance(d, gtx.Dimension) and d.kind == gtx.DimensionKind.HORIZONTAL
        )
    )


def non_horizontal_dims() -> Iterator[gtx.Dimension]:
    yield from vertical_dims()
    yield from local_dims()


def local_dims() -> Iterator[gtx.Dimension]:
    for d in globals().values():
        if isinstance(d, gtx.Dimension) and d.kind == gtx.DimensionKind.LOCAL:
            yield d


def vertical_dims() -> Iterator[gtx.Dimension]:
    return iter(
        tuple(
            d
            for d in globals().values()
            if isinstance(d, gtx.Dimension) and d.kind == gtx.DimensionKind.VERTICAL
        )
    )
