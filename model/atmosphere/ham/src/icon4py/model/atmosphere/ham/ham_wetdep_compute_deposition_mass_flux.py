from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import scan_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@scan_operator(axis=KDim, forward=True, init=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
def _ham_wetdep_compute_deposition_mass_flux(
    state            : tuple[wpfloat, \
                             wpfloat, \
                             wpfloat, \
                             wpfloat, \
                             wpfloat, \
                             wpfloat, \
                             wpfloat, \
                             wpfloat, \
                             wpfloat, \
                             wpfloat, \
                             wpfloat, \
                             wpfloat, \
                             wpfloat, \
                             wpfloat],
    ll_mxp           : bool,
    ll_ice           : bool,
    ll_wat           : bool,
    prevap           : wpfloat,
    zdepintic_impc_2d: wpfloat,
    zdepintic_impm_2d: wpfloat,
    zdepintic_impw_2d: wpfloat,
    zdepintic_nucc_2d: wpfloat,
    zdepintic_nucm_2d: wpfloat,
    zdepintic_nucw_2d: wpfloat,
    zdepintic_2d     : wpfloat,
    zdep             : wpfloat,
    zdep_imp         : wpfloat,
    zdep_nuc         : wpfloat
) -> (
    tuple[wpfloat,
          wpfloat,
          wpfloat,
          wpfloat,
          wpfloat,
          wpfloat,
          wpfloat,
          wpfloat,
          wpfloat,
          wpfloat,
          wpfloat,
          wpfloat,
          wpfloat,
          wpfloat]
):

    zdepintic_2d = state[6] + zdep

    zdepintic_nucw_2d = state[5] + (zdep_nuc if ll_wat else 0.0)
    zdepintic_impw_2d = state[2] + (zdep_imp if ll_wat else 0.0)

    zdepintic_nucm_2d = state[4] + (zdep_nuc if ll_mxp else 0.0)
    zdepintic_impm_2d = state[1] + (zdep_imp if ll_mxp else 0.0)

    zdepintic_nucc_2d = state[3] + (zdep_nuc if ll_ice else 0.0)
    zdepintic_impc_2d = state[0] + (zdep_imp if ll_ice else 0.0)

    zdxtevapic_tmp = zdepintic_2d * prevap

    zdxtevapic_nucw_tmp = zdepintic_nucw_2d * prevap
    zdxtevapic_impw_tmp = zdepintic_impw_2d * prevap

    zdxtevapic_nucm_tmp = zdepintic_nucm_2d * prevap
    zdxtevapic_impm_tmp = zdepintic_impm_2d * prevap

    zdxtevapic_nucc_tmp = zdepintic_nucc_2d * prevap
    zdxtevapic_impc_tmp = zdepintic_impc_2d * prevap

    zdepintic_2d = zdepintic_2d - zdxtevapic_tmp

    zdepintic_nucw_2d = zdepintic_nucw_2d - zdxtevapic_nucw_tmp
    zdepintic_impw_2d = zdepintic_impw_2d - zdxtevapic_impw_tmp

    zdepintic_nucm_2d = zdepintic_nucm_2d - zdxtevapic_nucm_tmp
    zdepintic_impm_2d = zdepintic_impm_2d - zdxtevapic_impm_tmp

    zdepintic_nucc_2d = zdepintic_nucc_2d - zdxtevapic_nucc_tmp
    zdepintic_impc_2d = zdepintic_impc_2d - zdxtevapic_impc_tmp

    return ( zdepintic_impc_2d,
             zdepintic_impm_2d,
             zdepintic_impw_2d,
             zdepintic_nucc_2d,
             zdepintic_nucm_2d,
             zdepintic_nucw_2d,
             zdepintic_2d,
             zdxtevapic_tmp,
             zdxtevapic_impc_tmp,
             zdxtevapic_impm_tmp,
             zdxtevapic_impw_tmp,
             zdxtevapic_nucc_tmp,
             zdxtevapic_nucm_tmp,
             zdxtevapic_nucw_tmp )


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def ham_wetdep_compute_deposition_mass_flux(
    ll_mxp           : Field[[CellDim, KDim], bool],
    ll_ice           : Field[[CellDim, KDim], bool],
    ll_wat           : Field[[CellDim, KDim], bool],
    prevap           : Field[[CellDim, KDim], wpfloat],
    zdepintic_impc_2d: Field[[CellDim, KDim], wpfloat],
    zdepintic_impm_2d: Field[[CellDim, KDim], wpfloat],
    zdepintic_impw_2d: Field[[CellDim, KDim], wpfloat],
    zdepintic_nucc_2d: Field[[CellDim, KDim], wpfloat],
    zdepintic_nucm_2d: Field[[CellDim, KDim], wpfloat],
    zdepintic_nucw_2d: Field[[CellDim, KDim], wpfloat],
    zdepintic_2d     : Field[[CellDim, KDim], wpfloat],
    zdep             : Field[[CellDim, KDim], wpfloat],
    zdep_imp         : Field[[CellDim, KDim], wpfloat],
    zdep_nuc         : Field[[CellDim, KDim], wpfloat],
    zdxtevapic       : Field[[CellDim, KDim], wpfloat],
    zdxtevapic_impc  : Field[[CellDim, KDim], wpfloat],
    zdxtevapic_impm  : Field[[CellDim, KDim], wpfloat],
    zdxtevapic_impw  : Field[[CellDim, KDim], wpfloat],
    zdxtevapic_nucc  : Field[[CellDim, KDim], wpfloat],
    zdxtevapic_nucm  : Field[[CellDim, KDim], wpfloat],
    zdxtevapic_nucw  : Field[[CellDim, KDim], wpfloat],
    horizontal_start : int32,
    horizontal_end   : int32,
    vertical_start   : int32,
    vertical_end     : int32
):

    _ham_wetdep_compute_deposition_mass_flux( ll_mxp,
                                              ll_ice,
                                              ll_wat,
                                              prevap,
                                              zdepintic_impc_2d,
                                              zdepintic_impm_2d,
                                              zdepintic_impw_2d,
                                              zdepintic_nucc_2d,
                                              zdepintic_nucm_2d,
                                              zdepintic_nucw_2d,
                                              zdepintic_2d,
                                              zdep,
                                              zdep_imp,
                                              zdep_nuc,
                                              out = ( zdepintic_impc_2d,
                                                      zdepintic_impm_2d,
                                                      zdepintic_impw_2d,
                                                      zdepintic_nucc_2d,
                                                      zdepintic_nucm_2d,
                                                      zdepintic_nucw_2d,
                                                      zdepintic_2d,
                                                      zdxtevapic,
                                                      zdxtevapic_impc,
                                                      zdxtevapic_impm,
                                                      zdxtevapic_impw,
                                                      zdxtevapic_nucc,
                                                      zdxtevapic_nucm,
                                                      zdxtevapic_nucw ),
                                              domain = {
                                                  CellDim: (horizontal_start, horizontal_end),
                                                  KDim: (vertical_start, vertical_end)
                                              }
    )