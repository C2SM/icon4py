from gt4py.next import Field, field_operator
from gt4py.next.iterator.builtins import exp, log

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.type_alias import wpfloat


z_aux1(CellDim) = p0sl_bg*EXP(-grav/rd*h_scal_bg/(t0sl_bg-del_t_bg) *LOG((EXP(p_nh(jg)%metrics%z_mc(1:nlen,jk,jb)/h_scal_bg)      &
                 &  *(t0sl_bg-del_t_bg) +del_t_bg)/t0sl_bg))

z_aux1 = p0sl_bg * exp(constant) * LOG(exp)


#: Constants used for the computation of the background reference atmosphere of the nh-model
#: defined  in mo_vertical_grid
h_scal_bg = 10000.0
#: scale height [m]
t0sl_bg   = 288.15
#: sea level temperature [K]

del_t_bg  = 75.0
#: difference between sea level temperature and assymptotic statospheric temperature [K]
@field_operator
def compute_reference_atmosphere(p0ref: wpfloat, p0sl_bg: wpfloat, grav:wpfloat, rd:wpfloat, h_scal_bg: wpfloat, t0sl_bg:wpfloat, del_t_bg:wpfloat, z_mc:Field[[CellDim, KDim], wpfloat])->tuple[Field[[CellDim, KDim], wpfloat], Field[[CellDim, KDim], wpfloat]]:
    """ Calculate exner_ref_mc, rho_ref_mc, theta_ref_mc"""


    z_aux1 =  compute_z_aux1(p0sl_bg, grav, rd, h_scal_bg, t0sl_bg, del_t_bg, z_mc)

    rho_ref_mc = func(z_aux1, z_temp)

@field_operator
def compute_z_aux1(p0sl_bg: wpfloat, grav:wpfloat, rd:wpfloat, h_scal_bg: wpfloat, t0sl_bg:wpfloat, del_t_bg:wpfloat, z_mc:Field[[CellDim, KDim], wpfloat]
                ) -> Field[[CellDim, KDim], wpfloat]:
    denom = t0sl_bg-del_t_bg
    z_aux1 = exp((- grav / rd * h_scal_bg) / denom * log(exp(z_mc / h_scal_bg) * (denom + del_t_bg)))
    exner_ref_mc = z_aux1 /
    return z_aux1


def compute_z_temp(t0sl_bg:wpfloat,del_t_bg:wpfloat, h_scal_bg:wpfloat, z_mc:Field[[CellDim, KDim], wpfloat] )-> Field[[CellDim, KDim], wpfloat]:
    denom = t0sl_bg-del_t_bg
    z_temp = denom + del_t_bg * exp(-z_mc / h_scal_bg)
    return z_temp

(z_aux1(1:nlen)/p0ref)**rd_o_cpd
def compute_exner_ref_mc
