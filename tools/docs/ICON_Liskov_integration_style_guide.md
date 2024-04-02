# ICON-Liskov integration style-guide

The idea of the style-guide is to have a unique style, to make reading and maintaining as easy as possible.
The code should look as clean and concise as possible. Also it should be similar to the ACC style-guide: https://gitlab.dkrz.de/icon/wiki/-/wikis/GPU-development/ICON-OpenAcc-style-and-implementation-guide.

## General:

- indentation as original surrounding code.
- no empty line between following `!$DSL`-statements of the same kind.

## Specific DSL statements:

- `!$DSL IMPORTS()` after last `USE` statement and before `IMPLICIT NONE`, one empty line before and after.
- `!$DSL DECLARE` after last variable declaration and before code block, one empty line before and after.
- `!$DSL START STENCIL` as close as possible to the start of the original stencil section, one empty line before, no empty line after.
- `!$DSL END STENCIL` as close as possible to the end of the original stencil section, no empty line before, one empty line after.
- `!$DSL START FUSED STENCIL` should be placed before the `!$DSL START STENCIL` of the first stencil in the fused batch, one empty line before, one empty line after.
- `!$DSL END FUSED STENCIL` should be placed after the `!$DSL END STENCIL` of the last stencil in the fused batch, one empty line before, one empty line after.
- `!$DSL INSERT` one empty line before and after, unless the inserted code is part of an `ACC` data region, or a function call.
- `!$DSL START CREATE` after the `!$ACC CREATE` block, no empty line before and one empty line after.

## Content of DSL statements:

- In `!$DSL START STENCIL` and `!$DSL START FUSED STENCIL`:
  - each argument (`name` included) should appear on a new line and aligned with the beginning of the first argument (generally `name`);
  - any equal sign `=` should be preceeded and followed by a single space;
  - the opening and closing round bracket should be followed and preceeded by a single space, respectively. Ignore this if the statement is on a single line.
- In `!$DSL DECLARE`:
  - each argument (`type` included) should appear on a new line and aligned with the beginning of the first argument;
  - any equal sign `=` should be preceeded and followed by a single space;
  - any comma `,` should be followed by a single space, but no space should be added before;
  - the opening and closing round bracket should be followed and preceeded by a single space, respectively. Ignore this if the statement is on a single line.

## Example

```fortran

...

MODULE mo_nh_diffusion

...

USE mo_vertical_grid, ONLY: nrdmax

!$DSL IMPORTS()

IMPLICIT NONE

...

SUBROUTINE diffusion(p_nh_prog,p_nh_diag,p_nh_metrics,p_patch,p_int,dtime,linit)

    ...

    INTEGER :: i_startblk, i_endblk, i_startidx, i_endidx

    !$DSL INSERT(INTEGER :: start_interior_idx_c, end_interior_idx_c, start_nudging_idx_c, end_halo_1_idx_c)
    !$DSL INSERT(INTEGER :: start_2nd_nudge_line_idx_e, end_interior_idx_e, start_bdydiff_idx_e)

    INTEGER :: rl_start, rl_end

    ...

    REAL(wp) :: r_dtimensubsteps

    !$DSL DECLARE( vn = nproma, p_patch%nlev, p_patch%nblks_e; &
    !$DSL          exner = nproma, p_patch%nlev, p_patch%nblks_c; &
    !$DSL          type = REAL(wp) )
    !$DSL DECLARE( kh_c = nproma, p_patch%nlev; &
    !$DSL          z_nabla2_c = nproma, p_patch%nlev, p_patch%nblks_e;
    !$DSL          type = REAL(vp) )

    !$DSL INSERT(REAL(vp) :: smallest_vpfloat = -HUGE(0._vp))

    !--------------------------------------------------------------------------

    ...

    !$ACC DATA CREATE(div, kh_c, kh_smag_e, kh_smag_ec, u_vert, v_vert, u_cell, v_cell, z_w_v, z_temp) &
    !$ACC   CREATE(z_vn_ie, z_vt_ie) &
    !$DSL INSERT(!$ACC   CREATE(w_old) &)
    !$DSL INSERT(!$ACC   COPYIN(vertical_idx, horizontal_idx) &)
    !$ACC   PRESENT(ividx, ivblk, iecidx, iecblk, icidx, icblk, ieidx, ieblk) &
    !$ACC   IF(i_am_accel_node)

    ...


        ! Computation of wind field deformation
        !$DSL START STENCIL( name = calculate_nabla2_and_smag_coefficients_for_vn; &
        !$DSL                smag_offset = smag_offset; &
        !$DSL                kh_smag_e_rel_tol = 1e-10_wp; &
        !$DSL                kh_smag_ec_rel_tol = 1e-10_wp; &
        !$DSL                z_nabla2_e_rel_tol = 1e-07_wp; &
        !$DSL                vertical_lower = 1; &
        !$DSL                vertical_upper = nlev; &
        !$DSL                horizontal_lower = i_startidx; &
        !$DSL                horizontal_upper = i_endidx )
        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)

!$NEC outerloop_unroll(4)
DO jk = 1, nlev
DO je = i_startidx, i_endidx

            ...

          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP
        !$DSL END STENCIL(name = calculate_nabla2_and_smag_coefficients_for_vn)

      ENDDO ! block jb

        ...

        CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, &
                           i_startidx, i_endidx, rl_start, rl_end)

        !$DSL START FUSED STENCIL( name = calculate_diagnostic_quantities_for_turbulence; &
        !$DSL                      hdef_ic = p_nh_diag%hdef_ic(:,:,1); &
        !$DSL                      div_ic_abs_tol = 1e-18_wp; &
        !$DSL                      vertical_lower = 2; &
        !$DSL                      vertical_upper = nlev; &
        !$DSL                      horizontal_lower = i_startidx; &
        !$DSL                      horizontal_upper = i_endidx )

        !$DSL START STENCIL( name = temporary_fields_for_turbulence_diagnostics; &
        !$DSL                kh_smag_ec = kh_smag_ec(:,:,1); &
        !$DSL                vn = p_nh_prog%vn(:,:,1); &
        !$DSL                diff_multfac_smag = diff_multfac_smag(:); &
        !$DSL                kh_c = kh_c(:,:); &
        !$DSL                div = div(:,:); &
        !$DSL                vertical_lower = 1; &
        !$DSL                vertical_upper = nlev; &
        !$DSL                horizontal_lower = i_startidx; &
        !$DSL                horizontal_upper = i_endidx )
        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
        DO jk = 1, nlev
          DO jc = i_startidx, i_endidx

            ...

          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP
        !$DSL END STENCIL(name = temporary_fields_for_turbulence_diagnostics)

        !$DSL START STENCIL( name = calculate_diagnostics_for_turbulence; 
        !$DSL                div = div; &
        !$DSL                kh_c = kh_c; &
        !$DSL                wgtfac_c = p_nh_metrics%wgtfac_c(:,:,1); &
        !$DSL                vertical_lower = 2; &
        !$DSL                vertical_upper = nlev; &
        !$DSL                horizontal_lower = i_startidx; &
        !$DSL                horizontal_upper = i_endidx )
        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
        DO jk = 2, nlev ! levels 1 and nlevp1 are unused

!DIR$ IVDEP
DO jc = i_startidx, i_endidx

            ...

          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP
        !$DSL END STENCIL(name = calculate_diagnostics_for_turbulence)

        !$DSL END FUSED STENCIL(name = calculate_diagnostic_quantities_for_turbulence)


    ...

```
