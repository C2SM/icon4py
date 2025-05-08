import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _compute_first_vertical_derivative(
    cell_k_field: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    first_vertical_derivative_of_cell = (cell_k_field - cell_k_field(Koff[1])) * inv_ddqz_z_full
    return first_vertical_derivative_of_cell
