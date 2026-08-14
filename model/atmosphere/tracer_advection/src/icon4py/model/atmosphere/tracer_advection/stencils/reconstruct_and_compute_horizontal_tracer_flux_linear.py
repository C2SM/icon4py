import gt4py.next as gtx
from gt4py.next import astype, where

from icon4py.model.atmosphere.tracer_advection.stencils.reconstruct_linear_coefficients_svd import (
    _reconstruct_linear_coefficients_svd,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C
from icon4py.model.common.type_alias import wpfloat


# reconstruct_linear_coefficients_svd fused into
# compute_horizontal_tracer_flux_from_linear_coefficients_alt, so the three coefficient fields
# are never written to memory. This is the half of Andreas Jocksch's upwind_hflux_miura_cell
# reformulation that carries over to GT4Py; the other half, scattering from the cell to its
# three edges, does not, and is not needed, because the E2C gather below already performs one
# reconstruction per cell and one read per edge.
#
# The trade is that the backend recomputes the upwind cell's reconstruction per edge, so 1.5
# evaluations per cell against three cell-field stores and their gathers. Whether that pays is
# an empirical question -- see the benchmark that compares this against the two-stage form.


@gtx.field_operator
def _reconstruct_and_compute_horizontal_tracer_flux_linear(
    p_cc: fa.CellKField[ta.wpfloat],
    lsq_pseudoinv_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    distv_bary_1: fa.EdgeKField[ta.vpfloat],
    distv_bary_2: fa.EdgeKField[ta.vpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    p_vn: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    z_lsq_coeff_1, z_lsq_coeff_2, z_lsq_coeff_3 = _reconstruct_linear_coefficients_svd(
        p_cc=p_cc,
        lsq_pseudoinv_1=lsq_pseudoinv_1,
        lsq_pseudoinv_2=lsq_pseudoinv_2,
    )

    lvn_pos_inv = p_vn < 0.0

    p_out_e = (
        where(lvn_pos_inv, z_lsq_coeff_1(E2C[1]), z_lsq_coeff_1(E2C[0]))
        + astype(distv_bary_1, wpfloat)
        * where(lvn_pos_inv, z_lsq_coeff_2(E2C[1]), z_lsq_coeff_2(E2C[0]))
        + astype(distv_bary_2, wpfloat)
        * where(lvn_pos_inv, z_lsq_coeff_3(E2C[1]), z_lsq_coeff_3(E2C[0]))
    ) * p_mass_flx_e

    return p_out_e


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def reconstruct_and_compute_horizontal_tracer_flux_linear(
    p_cc: fa.CellKField[ta.wpfloat],
    lsq_pseudoinv_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    distv_bary_1: fa.EdgeKField[ta.vpfloat],
    distv_bary_2: fa.EdgeKField[ta.vpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    p_vn: fa.EdgeKField[ta.wpfloat],
    p_out_e: fa.EdgeKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _reconstruct_and_compute_horizontal_tracer_flux_linear(
        p_cc=p_cc,
        lsq_pseudoinv_1=lsq_pseudoinv_1,
        lsq_pseudoinv_2=lsq_pseudoinv_2,
        distv_bary_1=distv_bary_1,
        distv_bary_2=distv_bary_2,
        p_mass_flx_e=p_mass_flx_e,
        p_vn=p_vn,
        out=p_out_e,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
