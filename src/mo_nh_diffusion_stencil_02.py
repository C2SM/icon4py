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
    c2e: np.array,
    vn: np.array,
    geofac_div: np.array,
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
    # TODO: add KDimension
    div = np.sum(vn[c2e] * geofac_div, axis=-1)
    return div


def mo_nh_diffusion_stencil_kh_c_numpy(
    c2e: np.array,
    kh_smag_ec: np.array,
    e_bln_c_s: np.array,
    diff_multfac_smag: np.array,
):
    """

    Note:
        Dusk implementation is the following

        @stencil
        def mo_nh_diffusion_stencil_02(kh_smag_ec: Field[Edge, K], e_bln_c_s: Field[Cell > Edge], diff_multfac_smag: Field[K], kh_c: Field[Cell, K]):
        with domain.upward.across[nudging:halo]:
            kh_c = sum_over(Cell > Edge, kh_smag_ec*e_bln_c_s)/diff_multfac_smag

    """
    # TODO: add KDimension
    mul = kh_smag_ec[c2e] * e_bln_c_s
    summed = np.sum(mul, axis=-1)
    kh_c = summed / diff_multfac_smag
    return kh_c


def mo_nh_diffusion_stencil_div_gt4py(
    c2e: np.array, vn: np.array, geofac_div: np.array
) -> np.array:
    """GT4PY implementation of mo_nh_diffusion_stencil_02.py dusk stencil."""
    # TODO: add KDimension
    EdgeDim = Dimension("Edge")
    CellDim = Dimension("Cell")
    C2EDim = Dimension("C2E", True)  # special local dim

    C2E = FieldOffset("C2E", source=EdgeDim, target=(CellDim, C2EDim))
    C2E_offset_provider = NeighborTableOffsetProvider(c2e, CellDim, EdgeDim, 3)

    vn_field = np_as_located_field(EdgeDim)(vn)
    geofac_div_field = np_as_located_field(CellDim, C2EDim)(geofac_div)
    out_field = np_as_located_field(CellDim)(np.zeros(shape=(c2e.shape[0],)))

    @field_operator
    def sum_cell_edge_neighbors(
        vn: Field[[EdgeDim], float32], geofac_div: Field[[CellDim, C2EDim], float32]
    ) -> Field[[CellDim], float32]:
        return neighbor_sum(vn(C2E) * geofac_div, axis=C2EDim)

    @program
    def exec_stencil(
        vn: Field[[EdgeDim], float32],
        geofac_div: Field[[CellDim, C2EDim], float32],
        out: Field[[CellDim], float32],
    ):
        sum_cell_edge_neighbors(vn, geofac_div, out=out)

    exec_stencil(
        vn_field,
        geofac_div_field,
        out_field,
        offset_provider={"C2E": C2E_offset_provider},
    )
    return out_field


def mo_nh_diffusion_stencil_kh_c_gt4py(
    c2e: np.array,
    kh_smag_ec: np.array,
    e_bln_c_s: np.array,
    diff_multfac_smag: np.array,
):
    """GT4PY implementation of mo_nh_diffusion_stencil_02.py dusk stencil."""
    # TODO: add KDimension
    EdgeDim = Dimension("Edge")
    CellDim = Dimension("Cell")
    C2EDim = Dimension("C2E", True)  # special local dim

    C2E = FieldOffset("C2E", source=EdgeDim, target=(CellDim, C2EDim))
    C2E_offset_provider = NeighborTableOffsetProvider(c2e, CellDim, EdgeDim, 3)

    kh_smag_ec_field = np_as_located_field(EdgeDim)(kh_smag_ec)
    e_bln_c_s_field = np_as_located_field(CellDim, C2EDim)(e_bln_c_s)
    diff_multfac_smag_field = np_as_located_field(CellDim)(diff_multfac_smag)
    out_field = np_as_located_field(CellDim)(np.zeros(shape=(c2e.shape[0],)))

    @field_operator
    def sum_edge_neighbors_and_divide(
        kh_smag_ec: Field[[EdgeDim], float32],
        e_bln_c_s: Field[[CellDim, C2EDim], float32],
        diff_multfac_smag: Field[[CellDim], float32],
    ) -> Field[[CellDim], float32]:
        summed = neighbor_sum(kh_smag_ec(C2E) * e_bln_c_s, axis=C2EDim)
        divided = summed / diff_multfac_smag
        return divided

    @program
    def exec_stencil(
        kh_smag_ec: Field[[EdgeDim], float32],
        e_bln_c_s: Field[[CellDim, C2EDim], float32],
        diff_multfac_smag: Field[[CellDim], float32],
        out: Field[[CellDim], float32],
    ):
        sum_edge_neighbors_and_divide(kh_smag_ec, e_bln_c_s, diff_multfac_smag, out=out)

    exec_stencil(
        kh_smag_ec_field,
        e_bln_c_s_field,
        diff_multfac_smag_field,
        out_field,
        offset_provider={"C2E": C2E_offset_provider},
    )

    return out_field
