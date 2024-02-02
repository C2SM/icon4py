from typing import Final

from gt4py.next import field_operator, program, GridType, Field
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.ffront.fbuiltins import exp, log

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.type_alias import wpfloat


#: Constants used for the computation of the background reference atmosphere of the nh-model


@field_operator
def compute_z_temp(
    z_mc: Field[[CellDim, KDim], wpfloat],
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
    h_scal_bg: wpfloat,
) -> Field[[CellDim, KDim], wpfloat]:
    denom = t0sl_bg - del_t_bg
    z_temp = denom + del_t_bg * exp(-z_mc / h_scal_bg)
    return z_temp


@field_operator
def compute_z_aux1(
    z_mc: Field[[CellDim, KDim], wpfloat],
    p0sl_bg: wpfloat,
    grav: wpfloat,
    rd: wpfloat,
    h_scal_bg: wpfloat,
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
) -> Field[[CellDim, KDim], wpfloat]:
    denom = t0sl_bg - del_t_bg
    logval = log((exp(z_mc / h_scal_bg) * denom + del_t_bg) / t0sl_bg)
    return p0sl_bg * exp(-grav / rd * h_scal_bg / denom * logval)


@field_operator
def _compute_reference_atmosphere(
    z_mc: Field[[CellDim, KDim], wpfloat],
    p0ref: wpfloat,
    p0sl_bg: wpfloat,
    grav: wpfloat,
    cpd: wpfloat,
    rd: wpfloat,
    h_scal_bg: wpfloat,
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
) -> tuple[
    Field[[CellDim, KDim], wpfloat],
    Field[[CellDim, KDim], wpfloat],
    Field[[CellDim, KDim], wpfloat],
]:
    z_aux1 = compute_z_aux1(
        z_mc=z_mc,
        p0sl_bg=p0sl_bg,
        grav=grav,
        rd=rd,
        h_scal_bg=h_scal_bg,
        t0sl_bg=t0sl_bg,
        del_t_bg=del_t_bg,
    )

    rd_o_cpd = rd / cpd
    exner_ref_mc = (z_aux1 / p0ref) ** rd_o_cpd
    z_temp = compute_z_temp(z_mc=z_mc, del_t_bg=del_t_bg, t0sl_bg=t0sl_bg, h_scal_bg=h_scal_bg)
    rho_ref_mc = z_aux1 / (rd * z_temp)
    theta_ref_mc = z_temp / exner_ref_mc
    return exner_ref_mc, rho_ref_mc, theta_ref_mc


@program(grid_type=GridType.UNSTRUCTURED)
def compute_reference_atmosphere(
    z_mc: Field[[CellDim, KDim], wpfloat],
    p0ref: wpfloat,
    p0sl_bg: wpfloat,
    grav: wpfloat,
    cpd: wpfloat,
    rd: wpfloat,
    h_scal_bg: wpfloat,
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
    exner_ref_mc: Field[[CellDim, KDim], wpfloat],
    rho_ref_mc: Field[[CellDim, KDim], wpfloat],
    theta_ref_mc: Field[[CellDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """
    Calculate rererence atmosphere fields on full levels.

    exner_ref_mc: reference exner pressure on full level mass points
    rho_ref_mc: reference density on full level mass points
    theta_ref_mc: reference potential temperature on full level mass points
    """
    _compute_reference_atmosphere(
        z_mc,
        p0ref,
        p0sl_bg,
        grav,
        cpd,
        rd,
        h_scal_bg,
        t0sl_bg,
        del_t_bg,
        out=(exner_ref_mc, rho_ref_mc, theta_ref_mc),
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )
