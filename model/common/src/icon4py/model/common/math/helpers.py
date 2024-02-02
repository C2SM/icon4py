from gt4py.next import field_operator, Field

from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.type_alias import wpfloat


@field_operator
def interpolate_height_levels_for_cell_k(
    half_level_field: Field[[CellDim, KDim], wpfloat]
) -> Field[[CellDim, KDim], wpfloat]:
    """
    Calculate the mean value of adjacent interface levels.

    Computes the average of two adjacent interface levels over a cell field for storage
    in the corresponding full levels.
    Args:
        half_level_field: Field[[CellDim, KDim], wpfloat]

    Returns: Field[[CellDim, KDim], wpfloat] full level field

    """
    return 0.5 * (half_level_field + half_level_field(Koff[+1]))
