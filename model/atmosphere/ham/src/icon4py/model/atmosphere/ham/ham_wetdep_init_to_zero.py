# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32

from icon4py.model.atmosphere.dycore.init_cell_kdim_field_with_zero_wp import (
    _init_cell_kdim_field_with_zero_wp,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _ham_wetdep_init_to_zero() -> Field[[CellDim, KDim], wpfloat]:

    return _init_cell_kdim_field_with_zero_wp()


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def ham_wetdep_init_to_zero(
    zdxtevapic      : Field[[CellDim, KDim], wpfloat],
    zdxtevapbc      : Field[[CellDim, KDim], wpfloat],
    zxtfrac_col     : Field[[CellDim, KDim], wpfloat],
    zxtfrac_colr    : Field[[CellDim, KDim], wpfloat],
    zxtfrac_cols    : Field[[CellDim, KDim], wpfloat],
    zdxtwat         : Field[[CellDim, KDim], wpfloat],
    zdxtice         : Field[[CellDim, KDim], wpfloat],
    zdxtwat_nuc     : Field[[CellDim, KDim], wpfloat],
    zdxtice_nuc     : Field[[CellDim, KDim], wpfloat],
    zdxtwat_imp     : Field[[CellDim, KDim], wpfloat],
    zdxtice_imp     : Field[[CellDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end  : int32,
    vertical_start  : int32,
    vertical_end    : int32
):

    _ham_wetdep_init_to_zero(out=zdxtevapic  , domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)})
    _ham_wetdep_init_to_zero(out=zdxtevapbc  , domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)})
    _ham_wetdep_init_to_zero(out=zxtfrac_col , domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)})
    _ham_wetdep_init_to_zero(out=zxtfrac_colr, domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)})
    _ham_wetdep_init_to_zero(out=zxtfrac_cols, domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)})
    _ham_wetdep_init_to_zero(out=zdxtwat     , domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)})
    _ham_wetdep_init_to_zero(out=zdxtice     , domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)})
    _ham_wetdep_init_to_zero(out=zdxtwat_nuc , domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)})
    _ham_wetdep_init_to_zero(out=zdxtice_nuc , domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)})
    _ham_wetdep_init_to_zero(out=zdxtwat_imp , domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)})
    _ham_wetdep_init_to_zero(out=zdxtice_imp , domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)})
