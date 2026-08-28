# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Upper boundary condition on the turbulent velocity scale: a vanishing vertical gradient.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section
3), the single statement at :1898-1904 (icon commit 26d6b98cce):

    tke(i,1,ntur)=tke(i,2,ntur) !kein TKE-Gradient am Oberrand

    -- "no TKE gradient at the upper boundary"

which is the whole of it. The scientific commentary in that file is by Matthias Raschendorfer
(DWD).

WHY THIS IS ITS OWN PROGRAM rather than a 'concat_where' row merged into
'compute_turbulent_velocity_scale', which is what the boundary-row convention in the package
README would otherwise ask for: the value copied here is the JUST COMPUTED one at half level 2,
not an input. Merging would mean a program reading the field it writes, one row apart. Two
programs make the dependency explicit and cost one launch of a single row.

Its read set is row 1 and its write set is row 0, which are disjoint, so it is safe to pass the
same field as input and output -- and that is exactly what the Fortran does.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _set_turbulent_velocity_scale_at_model_top(
    turbulent_velocity_scale: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """The value one half level below, which on row 0 is the uppermost computed level."""
    return turbulent_velocity_scale(Koff[1])


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def set_turbulent_velocity_scale_at_model_top(
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    turbulent_velocity_scale_with_top: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Copy half level 2 of 'tke' onto half level 1, so that the profile has no gradient there.

    The Fortran loop is horizontal only; the vertical domain is the single row 0, so run this
    with 'vertical_start = 0' and 'vertical_end = 1'. It has to run after
    'compute_turbulent_velocity_scale', whose output is what it copies.

    Args:
        turbulent_velocity_scale: 'tke(:,:,ntur)' [m/s] with half levels 2..ke already updated.
        turbulent_velocity_scale_with_top: Output, the same field with row 0 filled. May be the
            same field as the input.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: 0, the model top.
        vertical_end: 1, one past the model top.
    """
    _set_turbulent_velocity_scale_at_model_top(
        turbulent_velocity_scale=turbulent_velocity_scale,
        out=turbulent_velocity_scale_with_top,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
