import numpy as np
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import (
    Dimension,
    Field,
    float32,
    FieldOffset,
    neighbor_sum,
)
from functional.iterator.embedded import (
    np_as_located_field,
    NeighborTableOffsetProvider,
)


def mo_nh_diffusion_stencil_div_numpy(
    c2e: np.array, vn: np.array, geofac_div: np.array
) -> np.array:
    """Numpy implementation of mo_nh_diffusion_stencil_02.py dusk stencil.

    Note:
        The following is the dusk implementation:


        # stencil without sparse field multiplication
        @stencil
        def mo_nh_diffusion_stencil_02(vn: Field[Edge, K], div: Field[Cell, K]):
            with domain.upward.across[nudging:halo]:
                div = sum_over(Cell > Edge, vn) # computation over all edge values of a cell
    """
    div = np.sum(vn[c2e] * geofac_div, axis=-1)
    return div


def mo_nh_diffusion_stencil_div_gt4py(
    c2e: np.array, vn: np.array, geofac_div: np.array
) -> np.array:
    """GT4PY implementation of mo_nh_diffusion_stencil_02.py dusk stencil."""
    EdgeDim = Dimension("Edge")
    CellDim = Dimension("Cell")
    C2EDim = Dimension("C2E", True)  # special local dim

    C2E = FieldOffset("C2E", source=EdgeDim, target=(CellDim, C2EDim))
    C2E_offset_provider = NeighborTableOffsetProvider(c2e, CellDim, EdgeDim, 3)

    vn_field = np_as_located_field(EdgeDim)(vn)
    geofac_div_field = np_as_located_field(CellDim, C2EDim)(geofac_div)
    out = np_as_located_field(CellDim)(np.zeros(shape=(18,)))

    @field_operator
    def sum_cell_edge_neighbors(
        vn: Field[[EdgeDim], float32], geofac_div: Field[[CellDim, C2EDim], float32]
    ) -> Field[[CellDim], float32]:
        return neighbor_sum(vn(C2E) * geofac_div, axis=C2EDim)

    @program
    def sum_cell_edge_neighbors_program(
        vn: Field[[EdgeDim], float32],
        geofac_div: Field[[CellDim, C2EDim], float32],
        out: Field[[CellDim], float32],
    ):
        sum_cell_edge_neighbors(vn, geofac_div, out=out)

    sum_cell_edge_neighbors_program(
        vn_field, geofac_div_field, out, offset_provider={"C2E": C2E_offset_provider}
    )
    return out
