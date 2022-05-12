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


def mo_nh_diffusion_stencil_z_temp_numpy(
    c2e: np.array, z_nabla2_e: np.array, geofac_div: np.array
) -> np.array:
    """Numpy implementation of mo_nh_diffusion_stencil_02.py dusk stencil.

    Note:
        The following is the dusk implementation:

        @stencil
        def mo_nh_diffusion_stencil_14(z_nabla2_e: Field[Edge, K], geofac_div: Field[Cell > Edge], z_temp: Field[Cell, K]):
            with domain.upward.across[nudging:halo]:
                z_temp = sum_over(Cell > Edge, z_nabla2_e*geofac_div)
    """
    # TODO: add KDimension
    z_temp = np.sum(z_nabla2_e[c2e] * geofac_div, axis=-1)
    return z_temp


def mo_nh_diffusion_stencil_z_temp_gt4py(
    c2e: np.array, z_nabla2_e: np.array, geofac_div: np.array
) -> np.array:
    """GT4PY implementation of mo_nh_diffusion_stencil_02.py dusk stencil."""
    # TODO: add KDimension
    EdgeDim = Dimension("Edge")
    CellDim = Dimension("Cell")
    C2EDim = Dimension("C2E", True)  # special local dim

    C2E = FieldOffset("C2E", source=EdgeDim, target=(CellDim, C2EDim))
    C2E_offset_provider = NeighborTableOffsetProvider(c2e, CellDim, EdgeDim, 3)

    z_nabla2_e_field = np_as_located_field(EdgeDim)(z_nabla2_e)
    geofac_div_field = np_as_located_field(CellDim, C2EDim)(geofac_div)
    out_field = np_as_located_field(CellDim)(np.zeros(shape=(c2e.shape[0],)))

    @field_operator
    def sum_cell_edge_neighbors(
        z_nabla2_e: Field[[EdgeDim], float32],
        geofac_div: Field[[CellDim, C2EDim], float32],
    ) -> Field[[CellDim], float32]:
        return neighbor_sum(z_nabla2_e(C2E) * geofac_div, axis=C2EDim)

    @program
    def exec_stencil(
        z_nabla2_e: Field[[EdgeDim], float32],
        geofac_div: Field[[CellDim, C2EDim], float32],
        out: Field[[CellDim], float32],
    ):
        sum_cell_edge_neighbors(z_nabla2_e, geofac_div, out=out)

    exec_stencil(
        z_nabla2_e_field,
        geofac_div_field,
        out_field,
        offset_provider={"C2E": C2E_offset_provider},
    )
    return out_field
