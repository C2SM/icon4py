from functional.ffront.fbuiltins import (
    Dimension,
    FieldOffset,
)

KDim = Dimension("K")
EdgeDim = Dimension("Edge")
CellDim = Dimension("Cell")
C2EDim = Dimension("C2E", True)
C2E = FieldOffset("C2E", source=EdgeDim, target=(CellDim, C2EDim))
C2K = FieldOffset("C2K", source=KDim, target=(CellDim, KDim)) # TODO remove after https://github.com/GridTools/gt4py/pull/777 is available
