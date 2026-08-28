# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The stability- and height-corrected length scale of the separated horizontal shear mode.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section
2a), the second block Guenther Zaengl marks '!GZ: For tuning.' at :1431-1449 of icon commit
26d6b98cce, which is the commit that produced the reference capture:

    x4i = MIN( 1._wp, 0.5e-3_wp*(hhl(i,k)-hhl(i,ke1)) )
    x4  = (3._wp - 2._wp*x4i)*x4i**2   !low-level reduction factor
    hor_scale(i,k) = layr(i)*MIN( 5.0_wp, MAX( 0.01_wp, x4*xri(i,k) ) ) &
                            /MAX(1._wp,0.2_wp*tke(i,k,nvor))

with Zaengl's own comment on the first two lines, "Factor for variable 3D horizontal-vertical
length scale proportional to 1/SQRT(Ri), decreasing to zero in the lowest two kilometer above
ground, from ICON 180206". The scientific commentary in the file is otherwise by Matthias
Raschendorfer (DWD).

Three corrections are applied to 'layr', in this order:

  * 'x4', a smoothstep in the height above ground. '0.5e-3 * dz' is 1 at 2000 m, so 'x4i' ramps
    from 0 at the surface to 1 at 2 km and is clipped there, and '(3 - 2*x4i)*x4i**2' is the
    cubic Hermite step that is flat at both ends. The separated horizontal shear mode is
    suppressed inside the boundary layer, where the vertical shear already accounts for it.
  * 'xri', the stability factor of 'compute_inverse_richardson_number_factor'. The product is
    clipped to [0.01, 5], so the correction spans not quite three decades and is neither zero
    nor unbounded however extreme the stratification.
  * a division by the turbulent velocity where that exceeds 5 m/s ('MAX(1, 0.2*q)'), which
    shrinks the scale in already strongly turbulent air.

'imode_shshear' SELECTS THIS FORM. At any value other than 2 the Fortran skips all of the above
and takes 'hor_scale(i,k) = layr(i)' unchanged (:1451-1463). The reference capture ran
'imode_shshear = 2' -- established from the data, since the switch is not serialized, by
'test_the_capture_corrects_the_shear_length_scale_by_the_richardson_number' -- so the plain
branch has no oracle here and is not ported. 'TurbulenceConfig' freezes the switch at 2 as well.
"""

import gt4py.next as gtx
from gt4py.next import maximum, minimum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_effective_horizontal_shear_length_scale(
    uncorrected_horizontal_shear_length_scale: fa.CellField[wpfloat],
    half_level_height: fa.CellKField[wpfloat],
    surface_height: fa.CellField[wpfloat],
    inverse_richardson_number_factor: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'layr' corrected by the height above ground, the stability and the turbulent velocity.

    'x4i**2' is written as a product: Fortran's integer-literal power is a multiplication while
    GT4Py's '**' becomes 'math.pow', which CUDA evaluates to within 2 ulp rather than exactly.
    See '_compute_mechanical_forcing' for where that was measured.
    """
    height_above_ground = minimum(
        wpfloat("1.0"), wpfloat("0.5e-3") * (half_level_height - surface_height)
    )
    low_level_reduction = (wpfloat("3.0") - wpfloat("2.0") * height_above_ground) * (
        height_above_ground * height_above_ground
    )
    return (
        uncorrected_horizontal_shear_length_scale
        * minimum(
            wpfloat("5.0"),
            maximum(wpfloat("0.01"), low_level_reduction * inverse_richardson_number_factor),
        )
        / maximum(wpfloat("1.0"), wpfloat("0.2") * turbulent_velocity_scale)
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_effective_horizontal_shear_length_scale(
    uncorrected_horizontal_shear_length_scale: fa.CellField[wpfloat],
    half_level_height: fa.CellKField[wpfloat],
    surface_height: fa.CellField[wpfloat],
    inverse_richardson_number_factor: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    effective_horizontal_shear_length_scale: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute 'hor_scale' [m].

    'hhl(i,ke1)' is read at a fixed absolute level, so it is pre-sliced by the caller and passed
    as a cell field: GT4Py offsets are relative and cannot express it.

    Args:
        uncorrected_horizontal_shear_length_scale: 'layr' [m], one value per column.
        half_level_height: 'hhl' [m], half levels.
        surface_height: 'hhl(:,ke1)' [m], the lowest half level, one value per column.
        inverse_richardson_number_factor: 'xri' [-], from
            'compute_inverse_richardson_number_factor'.
        turbulent_velocity_scale: 'tke(:,:,nvor)', q = sqrt(2*TKE) at the previous time level
            [m/s], half levels. Section 3) has not yet updated it at this point.
        effective_horizontal_shear_length_scale: Output, 'hor_scale' [m].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First row; 1, mirroring Fortran 'k=2'.
        vertical_end: End of the rows; 'ke', mirroring Fortran 'k=...,kem' with 'kem = ke'.
            'hor_scale' is declared with 'ke' rows, one fewer than the half levels.
    """
    _compute_effective_horizontal_shear_length_scale(
        uncorrected_horizontal_shear_length_scale=uncorrected_horizontal_shear_length_scale,
        half_level_height=half_level_height,
        surface_height=surface_height,
        inverse_richardson_number_factor=inverse_richardson_number_factor,
        turbulent_velocity_scale=turbulent_velocity_scale,
        out=effective_horizontal_shear_length_scale,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
