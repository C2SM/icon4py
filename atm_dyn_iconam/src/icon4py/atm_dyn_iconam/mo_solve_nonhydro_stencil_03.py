
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field

from icon4py.common.dimension import CellDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_03(
) ->Field[[CellDim, KDim], float]:
    z_exner_ex_pr = float(0.0)
    return z_exner_ex_pr


@program
def mo_solve_nonhydro_stencil_03(
z_exner_ex_pr: Field[[CellDim, KDim], float],
) :
    _mo_solve_nonhydro_stencil_03(
        out=z_exner_ex_pr,
    )
