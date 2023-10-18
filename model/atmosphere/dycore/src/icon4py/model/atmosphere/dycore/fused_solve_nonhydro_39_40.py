from gt4py.next.ffront.decorator import field_operator, program, GridType
from gt4py.next.ffront.fbuiltins import Field, int32, where

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_39 import _mo_solve_nonhydro_stencil_39
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_40 import _mo_solve_nonhydro_stencil_40
from icon4py.model.common.dimension import CEDim, CellDim, EdgeDim, KDim


@field_operator
def _fused_solve_nonhydro_39_40(
    e_bln_c_s: Field[[CEDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    wgtfacq_c: Field[[CellDim, KDim], float],
    vert_idx: Field[[KDim], int32],
    nlev: int32,
    nflatlev: int32
) -> Field[[CellDim, KDim], float]:
    w_concorr_c = where(
        nflatlev < vert_idx < nlev,
        _mo_solve_nonhydro_stencil_39(e_bln_c_s, z_w_concorr_me, wgtfac_c),
        _mo_solve_nonhydro_stencil_40(e_bln_c_s, z_w_concorr_me, wgtfacq_c)
    )
    return w_concorr_c


@program(grid_type=GridType.UNSTRUCTURED)
def fused_solve_nonhydro_stencil_39_40(
    e_bln_c_s: Field[[CEDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    wgtfacq_c: Field[[CellDim, KDim], float],
    vert_idx: Field[[KDim], int32],
    nlev: int32,
    nflatlev: int32,
    w_concorr_c: Field[[CellDim, KDim], float],
):
    _fused_solve_nonhydro_39_40(e_bln_c_s, z_w_concorr_me, wgtfac_c, wgtfacq_c, vert_idx, nlev, nflatlev, out=w_concorr_c[:, 1:])
