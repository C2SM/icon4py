import gt4py.next as gtx
from gt4py.next import astype, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C
from icon4py.model.common.type_alias import wpfloat


# Final flux of the miura3 scheme with a quadratic reconstruction, lsq_high_ord=2
# (mo_advection_hflux.f90 4764-4770): the reconstruction coefficients of the upwind cell
# dotted with the quadrature vector, times the mass flux. The Fortran indexes the cell
# through ptr_ilc/ptr_ibc, which the backtrajectory fills; here the edge gathers it through
# E2C, selected by p_cell_rel_idx_dsl, as in accumulate_weno_candidate_flux_weights.
#
# The quadrature vector holds the departure-region AREA AVERAGES of the monomials
# (prepare_gauss_quadrature_quadratic_miura3), so there is no division by the region area.


@gtx.field_operator
def _compute_horizontal_tracer_flux_from_quadratic_coefficients(
    p_coeff_1: fa.CellKField[ta.wpfloat],
    p_coeff_2: fa.CellKField[ta.wpfloat],
    p_coeff_3: fa.CellKField[ta.wpfloat],
    p_coeff_4: fa.CellKField[ta.wpfloat],
    p_coeff_5: fa.CellKField[ta.wpfloat],
    p_coeff_6: fa.CellKField[ta.wpfloat],
    p_cell_rel_idx_dsl: fa.EdgeKField[gtx.int32],
    p_quad_vector_sum_1: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_2: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_3: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_4: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_5: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_6: fa.EdgeKField[ta.vpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    c_1 = where(p_cell_rel_idx_dsl == 1, p_coeff_1(E2C[1]), p_coeff_1(E2C[0]))
    c_2 = where(p_cell_rel_idx_dsl == 1, p_coeff_2(E2C[1]), p_coeff_2(E2C[0]))
    c_3 = where(p_cell_rel_idx_dsl == 1, p_coeff_3(E2C[1]), p_coeff_3(E2C[0]))
    c_4 = where(p_cell_rel_idx_dsl == 1, p_coeff_4(E2C[1]), p_coeff_4(E2C[0]))
    c_5 = where(p_cell_rel_idx_dsl == 1, p_coeff_5(E2C[1]), p_coeff_5(E2C[0]))
    c_6 = where(p_cell_rel_idx_dsl == 1, p_coeff_6(E2C[1]), p_coeff_6(E2C[0]))

    p_out_e = (
        c_1 * astype(p_quad_vector_sum_1, wpfloat)
        + c_2 * astype(p_quad_vector_sum_2, wpfloat)
        + c_3 * astype(p_quad_vector_sum_3, wpfloat)
        + c_4 * astype(p_quad_vector_sum_4, wpfloat)
        + c_5 * astype(p_quad_vector_sum_5, wpfloat)
        + c_6 * astype(p_quad_vector_sum_6, wpfloat)
    ) * p_mass_flx_e

    return p_out_e


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_horizontal_tracer_flux_from_quadratic_coefficients(
    p_coeff_1: fa.CellKField[ta.wpfloat],
    p_coeff_2: fa.CellKField[ta.wpfloat],
    p_coeff_3: fa.CellKField[ta.wpfloat],
    p_coeff_4: fa.CellKField[ta.wpfloat],
    p_coeff_5: fa.CellKField[ta.wpfloat],
    p_coeff_6: fa.CellKField[ta.wpfloat],
    p_cell_rel_idx_dsl: fa.EdgeKField[gtx.int32],
    p_quad_vector_sum_1: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_2: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_3: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_4: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_5: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_6: fa.EdgeKField[ta.vpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    p_out_e: fa.EdgeKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_horizontal_tracer_flux_from_quadratic_coefficients(
        p_coeff_1=p_coeff_1,
        p_coeff_2=p_coeff_2,
        p_coeff_3=p_coeff_3,
        p_coeff_4=p_coeff_4,
        p_coeff_5=p_coeff_5,
        p_coeff_6=p_coeff_6,
        p_cell_rel_idx_dsl=p_cell_rel_idx_dsl,
        p_quad_vector_sum_1=p_quad_vector_sum_1,
        p_quad_vector_sum_2=p_quad_vector_sum_2,
        p_quad_vector_sum_3=p_quad_vector_sum_3,
        p_quad_vector_sum_4=p_quad_vector_sum_4,
        p_quad_vector_sum_5=p_quad_vector_sum_5,
        p_quad_vector_sum_6=p_quad_vector_sum_6,
        p_mass_flx_e=p_mass_flx_e,
        out=p_out_e,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
