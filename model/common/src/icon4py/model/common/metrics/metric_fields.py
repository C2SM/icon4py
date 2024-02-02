from gt4py.next import Field, field_operator, int32, program, GridType
from gt4py.next.iterator.builtins import exp, log
from gt4py.next.program_processors.runners import roundtrip, gtfn

from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.type_alias import wpfloat


# TODO (magdalena) move this stencil to some math module
@field_operator
def interpolate_height_levels_for_cell_k(
    half_level_field: Field[[CellDim, KDim], wpfloat]
) -> Field[[CellDim, KDim], wpfloat]:
    return 0.5 * (half_level_field + half_level_field(Koff[+1]))


@program(grid_type=GridType.UNSTRUCTURED)
def compute_z_mc(
    z_ifc: Field[[CellDim, KDim], wpfloat],
    z_mc: Field[[CellDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """Compute the geometric height of full levels from the geometric height of half levels (z_ifc)

    This assumes that the input field z_ifc is defined on half levels (KHalfDim) and the
    returned fields is
    defined on full levels (KDim)
    """
    interpolate_height_levels_for_cell_k(
        z_ifc,
        out=z_mc,
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )
