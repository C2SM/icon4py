import pytest
from functional.common import Field
from functional.ffront.decorator import Program, program, field_operator

from icon4py.common.dimension import KDim, CellDim
from icon4py.pyutils.metadata import get_field_infos


@field_operator
def _add(field1: Field[[CellDim, KDim], float], field2: Field[[CellDim, KDim], float])-> Field[[CellDim, KDim], float]:
    return field1 + field2

@program
def with_domain(a: Field[[CellDim, KDim], float], b: Field[[CellDim, KDim], float], sum: Field[[CellDim, KDim], float],
                vertical_start:int, vertical_end: int, k_start:int, k_end:int):
    _add(a, b, out=sum, domain={CellDim:(k_start, k_end), KDim: (vertical_start, vertical_end)})


@program
def without_domain(a: Field[[CellDim, KDim], float], b: Field[[CellDim, KDim], float], sum: Field[[CellDim, KDim], float]):
    _add(a, b, out=sum)

@program
def with_constant_domain(a: Field[[CellDim, KDim], float], b: Field[[CellDim, KDim], float], sum: Field[[CellDim, KDim], float]):
    _add(a, b, out=sum, domain={CellDim: (0, 3), KDim: (1, 8)})


@pytest.mark.parametrize("program", [with_domain, without_domain, with_constant_domain])
def test_get_field_infos_does_not_contain_domain_args(program):
    field_info = get_field_infos(program)
    assert len(field_info) ==3
    assert field_info["a"].out == False
    assert field_info["a"].inp == True

    assert field_info["b"].out == False
    assert field_info["b"].inp == True

    assert field_info["sum"].out == True
    assert field_info["sum"].inp == False
