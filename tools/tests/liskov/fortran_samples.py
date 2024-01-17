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


NO_DIRECTIVES_STENCIL = """\
    !$ACC PARALLEL LOOP DEFAULT(NONE) GANG VECTOR COLLAPSE(2) ASYNC(1) IF( i_am_accel_node .AND. acc_on )
    DO jk = 1, nlev
    !DIR$ IVDEP
      DO je = i_startidx, i_endidx
        p_nh_prog%vn(je,jk,jb) =   &
        p_nh_prog%vn(je,jk,jb) + &
        z_nabla2_e(je,jk,jb) * &
        p_patch%edges%area_edge(je,jb)*fac_bdydiff_v
      ENDDO
    ENDDO
    !$ACC END PARALLEL LOOP
    """

SINGLE_STENCIL = """\
    !$DSL IMPORTS()

    !$DSL START CREATE()

    !$DSL DECLARE(vn=nproma,p_patch%nlev,p_patch%nblks_e; suffix=dsl)

    !$DSL DECLARE(vn=nproma,p_patch%nlev,p_patch%nblks_e; a=nproma,p_patch%nlev,p_patch%nblks_e; &
    !$DSL         b=nproma,p_patch%nlev,p_patch%nblks_e; type=REAL(vp))

    !$DSL START STENCIL(name=apply_nabla2_to_vn_in_lateral_boundary; &
    !$DSL       z_nabla2_e=z_nabla2_e(:,:,1); area_edge=p_patch%edges%area_edge(:,1); &
    !$DSL       fac_bdydiff_v=fac_bdydiff_v; vn=p_nh_prog%vn(:,:,1); &
    !$DSL       vertical_lower=1; vertical_upper=nlev; &
    !$DSL       horizontal_lower=i_startidx; horizontal_upper=i_endidx; &
    !$DSL       accpresent=True)
    !$OMP DO PRIVATE(je,jk,jb,i_startidx,i_endidx) ICON_OMP_DEFAULT_SCHEDULE
      DO jb = i_startblk,i_endblk

        CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, &
                           i_startidx, i_endidx, start_bdydiff_e, grf_bdywidth_e)

    !$ACC PARALLEL IF( i_am_accel_node .AND. acc_on ) DEFAULT(NONE) ASYNC(1)
    vn_before(:,:,:) = p_nh_prog%vn(:,:,:)
    !$ACC END PARALLEL

    !$ACC PARALLEL LOOP DEFAULT(NONE) GANG VECTOR COLLAPSE(2) ASYNC(1) IF( i_am_accel_node .AND. acc_on )
            DO jk = 1, nlev
    !DIR$ IVDEP
              DO je = i_startidx, i_endidx
                p_nh_prog%vn(je,jk,jb) =   &
                  p_nh_prog%vn(je,jk,jb) + &
                  z_nabla2_e(je,jk,jb) * &
                  p_patch%edges%area_edge(je,jb)*fac_bdydiff_v
              ENDDO
            ENDDO
    !$DSL START PROFILE(name=apply_nabla2_to_vn_in_lateral_boundary)
    !$ACC END PARALLEL LOOP
    !$DSL END PROFILE()
    !$DSL END STENCIL(name=apply_nabla2_to_vn_in_lateral_boundary; noprofile=True)
    !$DSL END CREATE()
    """

SINGLE_STENCIL_WITH_COMMENTS = """\
    ! Use !$DSL statements, they are great. They can be easily commented out by:

    !!$DSL IMPORTS()

    ! $DSL START CREATE()

    !$DSL IMPORTS()

    !$DSL START CREATE()

    !$DSL DECLARE(vn=nproma,p_patch%nlev,p_patch%nblks_e; suffix=dsl)

    !$DSL DECLARE(vn=nproma,p_patch%nlev,p_patch%nblks_e; a=nproma,p_patch%nlev,p_patch%nblks_e; &
    !$DSL         b=nproma,p_patch%nlev,p_patch%nblks_e; type=REAL(vp))

    !$DSL START STENCIL(name=apply_nabla2_to_vn_in_lateral_boundary; &
    !$DSL       z_nabla2_e=z_nabla2_e(:,:,1); area_edge=p_patch%edges%area_edge(:,1); &
    !$DSL       fac_bdydiff_v=fac_bdydiff_v; vn=p_nh_prog%vn(:,:,1); &
    !$DSL       vertical_lower=1; vertical_upper=nlev; &
    !$DSL       horizontal_lower=i_startidx; horizontal_upper=i_endidx; &
    !$DSL       accpresent=True)
    !$OMP DO PRIVATE(je,jk,jb,i_startidx,i_endidx) ICON_OMP_DEFAULT_SCHEDULE
      DO jb = i_startblk,i_endblk

        CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, &
                           i_startidx, i_endidx, start_bdydiff_e, grf_bdywidth_e)

    !$ACC PARALLEL IF( i_am_accel_node .AND. acc_on ) DEFAULT(NONE) ASYNC(1)
    vn_before(:,:,:) = p_nh_prog%vn(:,:,:)
    !$ACC END PARALLEL

    !$ACC PARALLEL LOOP DEFAULT(NONE) GANG VECTOR COLLAPSE(2) ASYNC(1) IF( i_am_accel_node .AND. acc_on )
            DO jk = 1, nlev
    !DIR$ IVDEP
              DO je = i_startidx, i_endidx
                p_nh_prog%vn(je,jk,jb) =   &
                  p_nh_prog%vn(je,jk,jb) + &
                  z_nabla2_e(je,jk,jb) * &
                  p_patch%edges%area_edge(je,jb)*fac_bdydiff_v
              ENDDO
            ENDDO
    !$DSL START PROFILE(name=apply_nabla2_to_vn_in_lateral_boundary)
    !$ACC END PARALLEL LOOP
    !$DSL END PROFILE()
    !$DSL END STENCIL(name=apply_nabla2_to_vn_in_lateral_boundary; noprofile=True)
    !$DSL END CREATE()
    """

MULTIPLE_STENCILS = """\
    !$DSL IMPORTS()

    !$DSL START CREATE()

    !$DSL DECLARE(vn=nproma,p_patch%nlev,p_patch%nblks_e; z_rho_e=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL         z_theta_v_e=nproma,p_patch%nlev,p_patch%nblks_c; z_nabla2_c=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL         z_rth_pr_1=nproma,p_patch%nlev,p_patch%nblks_c; z_rth_pr_2=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL         rho_ic=nproma,p_patch%nlev,p_patch%nblks_c)

    !$DSL START STENCIL(name=mo_solve_nonhydro_stencil_08; wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1); rho=p_nh%prog(nnow)%rho(:,:,1); rho_ref_mc=p_nh%metrics%rho_ref_mc(:,:,1); &
    !$DSL               theta_v=p_nh%prog(nnow)%theta_v(:,:,1); theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1); rho_ic=p_nh%diag%rho_ic(:,:,1); z_rth_pr_1=z_rth_pr(:,:,1,1); &
    !$DSL               z_rth_pr_2=z_rth_pr(:,:,1,2); vertical_lower=2; vertical_upper=nlev; horizontal_lower=i_startidx; horizontal_upper=i_endidx)
              !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(NONE) ASYNC(1)
              !$ACC LOOP GANG VECTOR TILE(32, 4)
              DO jk = 2, nlev
    !DIR$ IVDEP
                DO jc = i_startidx, i_endidx
                  ! density at interface levels for vertical flux divergence computation
                  p_nh%diag%rho_ic(jc,jk,jb) = p_nh%metrics%wgtfac_c(jc,jk,jb) *p_nh%prog(nnow)%rho(jc,jk  ,jb) + &
                                        (1._wp-p_nh%metrics%wgtfac_c(jc,jk,jb))*p_nh%prog(nnow)%rho(jc,jk-1,jb)

                  ! perturbation density and virtual potential temperature at main levels for horizontal flux divergence term
                  ! (needed in the predictor step only)
    #ifdef __SWAPDIM
                  z_rth_pr(jc,jk,jb,1) =  p_nh%prog(nnow)%rho(jc,jk,jb)     - p_nh%metrics%rho_ref_mc(jc,jk,jb)
                  z_rth_pr(jc,jk,jb,2) =  p_nh%prog(nnow)%theta_v(jc,jk,jb) - p_nh%metrics%theta_ref_mc(jc,jk,jb)
    #else
                  z_rth_pr(1,jc,jk,jb) =  p_nh%prog(nnow)%rho(jc,jk,jb)     - p_nh%metrics%rho_ref_mc(jc,jk,jb)
                  z_rth_pr(2,jc,jk,jb) =  p_nh%prog(nnow)%theta_v(jc,jk,jb) - p_nh%metrics%theta_ref_mc(jc,jk,jb)
    #endif
    #ifdef _OPENACC
                ENDDO
              ENDDO
              !$ACC END PARALLEL
    #endif

    !$DSL END STENCIL(name=mo_solve_nonhydro_stencil_08)


    !$DSL START STENCIL(name=apply_nabla2_to_vn_in_lateral_boundary; &
    !$DSL       z_nabla2_e=z_nabla2_e(:,:,1); area_edge=p_patch%edges%area_edge(:,1); &
    !$DSL       fac_bdydiff_v=fac_bdydiff_v; vn=p_nh_prog%vn(:,:,1); vn_abs_tol=1e-21_wp; &
    !$DSL       vertical_lower=1; vertical_upper=nlev; &
    !$DSL       horizontal_lower=i_startidx; horizontal_upper=i_endidx)
    !$ACC PARALLEL LOOP DEFAULT(NONE) GANG VECTOR COLLAPSE(2) ASYNC(1) IF( i_am_accel_node .AND. acc_on )
            DO jk = 1, nlev
    !DIR$ IVDEP
              DO je = i_startidx, i_endidx
                p_nh_prog%vn(je,jk,jb) =   &
                  p_nh_prog%vn(je,jk,jb) + &
                  z_nabla2_e(je,jk,jb) * &
                  p_patch%edges%area_edge(je,jb)*fac_bdydiff_v
              ENDDO
            ENDDO
    !$ACC END PARALLEL LOOP
    !$DSL END STENCIL(name=apply_nabla2_to_vn_in_lateral_boundary)

    !$DSL START STENCIL(name=calculate_nabla2_for_w; &
    !$DSL       w=p_nh_prog%w(:,:,1); geofac_n2s=p_int%geofac_n2s(:,:,1); &
    !$DSL       z_nabla2_c=z_nabla2_c(:,:,1); z_nabla2_c_abs_tol=1e-21_wp; &
    !$DSL       z_nabla2_c_rel_tol=1e-21_wp; &
    !$DSL       vertical_lower=1; vertical_upper=nlev; &
    !$DSL       horizontal_lower=i_startidx; horizontal_upper=i_endidx)
    !$ACC PARALLEL LOOP DEFAULT(NONE) GANG VECTOR COLLAPSE(2) ASYNC(1) IF( i_am_accel_node .AND. acc_on )
    #ifdef __LOOP_EXCHANGE
            DO jc = i_startidx, i_endidx
    !DIR$ IVDEP
    #ifdef _CRAYFTN
    !DIR$ PREFERVECTOR
    #endif
            DO jk = 1, nlev
    #else
            DO jk = 1, nlev
              DO jc = i_startidx, i_endidx
    #endif
            z_nabla2_c(jc,jk,jb) =  &
              p_nh_prog%w(jc,jk,jb)                        *p_int%geofac_n2s(jc,1,jb) + &
              p_nh_prog%w(icidx(jc,jb,1),jk,icblk(jc,jb,1))*p_int%geofac_n2s(jc,2,jb) + &
              p_nh_prog%w(icidx(jc,jb,2),jk,icblk(jc,jb,2))*p_int%geofac_n2s(jc,3,jb) + &
              p_nh_prog%w(icidx(jc,jb,3),jk,icblk(jc,jb,3))*p_int%geofac_n2s(jc,4,jb)
          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP
    !$DSL ENDIF()
    !$DSL END STENCIL(name=calculate_nabla2_for_w; noendif=true)
    !$DSL END CREATE()
    """

DIRECTIVES_SAMPLE = """\
!$DSL IMPORTS()

!$DSL START CREATE()

!$DSL DECLARE(vn=p_patch%vn; vn2=p_patch%vn2)

!$DSL START STENCIL(name=mo_nh_diffusion_06; vn=p_patch%vn; &
!$DSL       a=a; b=c)

!$DSL END STENCIL(name=mo_nh_diffusion_06)

!$DSL START STENCIL(name=mo_nh_diffusion_07; xn=p_patch%xn)

!$DSL END STENCIL(name=mo_nh_diffusion_07)

!$DSL UNKNOWN_DIRECTIVE()
!$DSL END CREATE()
"""

CONSECUTIVE_STENCIL = """\
    !$DSL IMPORTS()

    !$DSL START CREATE()

    !$DSL DECLARE(z_q=nproma,p_patch%nlev; field=nproma,p_patch%nlev; field_to_zero_vp=nproma,p_patch%nlev)

    !$DSL START STENCIL(name=set_cell_kdim_field_to_zero_vp; field_to_zero_vp=z_alpha(:,:); vertical_lower=nlevp1; &
    !$DSL               vertical_upper=nlevp1; horizontal_lower=i_startidx; horizontal_upper=i_endidx; mergecopy=true)

    !$DSL START STENCIL(name=set_cell_kdim_field_to_zero_vp; field_to_zero_vp=z_q(:,:); vertical_lower=1; vertical_upper=1; &
    !$DSL               horizontal_lower=i_startidx; horizontal_upper=i_endidx; mergecopy=true)

        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(NONE) ASYNC(1)
        !$ACC LOOP GANG VECTOR
        DO jc = i_startidx, i_endidx
          z_alpha(jc,nlevp1) = 0.0_wp
          !
          ! Note: z_q is used in the tridiagonal matrix solver for w below.
          !       z_q(1) is always zero, irrespective of w(1)=0 or w(1)/=0
          !       z_q(1)=0 is equivalent to cp(slev)=c(slev)/b(slev) in mo_math_utilities:tdma_solver_vec
          z_q(jc,1) = 0._vp
        ENDDO
        !$ACC END PARALLEL
    !$DSL END PROFILE()
    !$DSL ENDIF()

    !$DSL END STENCIL(name=set_cell_kdim_field_to_zero_vp; noendif=true; noprofile=true)
    !$DSL END STENCIL(name=set_cell_kdim_field_to_zero_vp; noendif=true; noprofile=true)

    !$DSL END CREATE()
"""


SINGLE_FUSED = """\
    !$DSL IMPORTS()

    !$DSL INSERT(INTEGER :: start_interior_idx_c, end_interior_idx_c, start_nudging_idx_c, end_halo_1_idx_c)

    !$DSL DECLARE(kh_smag_e=nproma,p_patch%nlev,p_patch%nblks_e; &
    !$DSL      kh_smag_ec=nproma,p_patch%nlev,p_patch%nblks_e; &
    !$DSL      z_nabla2_e=nproma,p_patch%nlev,p_patch%nblks_e; &
    !$DSL      kh_c=nproma,p_patch%nlev; &
    !$DSL      div=nproma,p_patch%nlev; &
    !$DSL      div_ic=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL      hdef_ic=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL      z_nabla4_e2=nproma,p_patch%nlev; &
    !$DSL      vn=nproma,p_patch%nlev,p_patch%nblks_e; &
    !$DSL      z_nabla2_c=nproma,p_patch%nlev,p_patch%nblks_e; &
    !$DSL      dwdx=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL      dwdy=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL      w=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL      enh_diffu_3d=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL      z_temp=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL      theta_v=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL      exner=nproma,p_patch%nlev,p_patch%nblks_c)

        !$DSL START FUSED STENCIL(name=calculate_diagnostic_quantities_for_turbulence; &
        !$DSL  kh_smag_ec=kh_smag_ec(:,:,1); vn=p_nh_prog%vn(:,:,1); e_bln_c_s=p_int%e_bln_c_s(:,:,1); &
        !$DSL  geofac_div=p_int%geofac_div(:,:,1); diff_multfac_smag=diff_multfac_smag(:); &
        !$DSL  wgtfac_c=p_nh_metrics%wgtfac_c(:,:,1); div_ic=p_nh_diag%div_ic(:,:,1); &
        !$DSL  hdef_ic=p_nh_diag%hdef_ic(:,:,1); &
        !$DSL  div_ic_abs_tol=1e-18_wp; vertical_lower=2; &
        !$DSL  vertical_upper=nlev; horizontal_lower=i_startidx; horizontal_upper=i_endidx)

        !$DSL START STENCIL(name=temporary_fields_for_turbulence_diagnostics; kh_smag_ec=kh_smag_ec(:,:,1); vn=p_nh_prog%vn(:,:,1); &
        !$DSL       e_bln_c_s=p_int%e_bln_c_s(:,:,1); geofac_div=p_int%geofac_div(:,:,1); &
        !$DSL       diff_multfac_smag=diff_multfac_smag(:); kh_c=kh_c(:,:); div=div(:,:); &
        !$DSL       vertical_lower=1; vertical_upper=nlev; horizontal_lower=i_startidx; &
        !$DSL       horizontal_upper=i_endidx)

        !$DSL END STENCIL(name=temporary_fields_for_turbulence_diagnostics)

        !$DSL START STENCIL(name=calculate_diagnostics_for_turbulence; div=div; kh_c=kh_c; wgtfac_c=p_nh_metrics%wgtfac_c(:,:,1); &
        !$DSL               div_ic=p_nh_diag%div_ic(:,:,1); hdef_ic=p_nh_diag%hdef_ic(:,:,1); div_ic_abs_tol=1e-18_wp; &
        !$DSL               vertical_lower=2; vertical_upper=nlev; horizontal_lower=i_startidx; horizontal_upper=i_endidx)

        !$DSL END STENCIL(name=calculate_diagnostics_for_turbulence)

        !$DSL END FUSED STENCIL(name=calculate_diagnostic_quantities_for_turbulence)
    """


MULTIPLE_FUSED = """\
    !$DSL IMPORTS()
    !$DSL START DELETE()
    !$DSL END DELETE()
    !$DSL INSERT(INTEGER :: start_interior_idx_c, end_interior_idx_c, start_nudging_idx_c, end_halo_1_idx_c)

    !$DSL DECLARE(kh_smag_e=nproma,p_patch%nlev,p_patch%nblks_e; &
    !$DSL      kh_smag_ec=nproma,p_patch%nlev,p_patch%nblks_e; &
    !$DSL      z_nabla2_e=nproma,p_patch%nlev,p_patch%nblks_e; &
    !$DSL      kh_c=nproma,p_patch%nlev; &
    !$DSL      div=nproma,p_patch%nlev; &
    !$DSL      div_ic=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL      hdef_ic=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL      z_nabla4_e2=nproma,p_patch%nlev; &
    !$DSL      vn=nproma,p_patch%nlev,p_patch%nblks_e; &
    !$DSL      z_nabla2_c=nproma,p_patch%nlev,p_patch%nblks_e; &
    !$DSL      dwdx=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL      dwdy=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL      w=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL      enh_diffu_3d=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL      z_temp=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL      theta_v=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL      exner=nproma,p_patch%nlev,p_patch%nblks_c)

        !$DSL INSERT(start_2nd_nudge_line_idx_e = i_startidx)
        !$DSL INSERT(end_interior_idx_e = i_endidx)
                 ! Compute nabla4(v)

        !$DSL START FUSED STENCIL(name=apply_diffusion_to_vn; &
        !$DSL     u_vert=u_vert(:,:,1); &
        !$DSL     v_vert=v_vert(:,:,1); &
        !$DSL     primal_normal_vert_v1=p_patch%edges%primal_normal_vert_x(:,:,1); &
        !$DSL     primal_normal_vert_v2=p_patch%edges%primal_normal_vert_y(:,:,1); &
        !$DSL     z_nabla2_e=z_nabla2_e(:,:,1); &
        !$DSL     inv_vert_vert_length=p_patch%edges%inv_vert_vert_length(:,1); &
        !$DSL     inv_primal_edge_length=p_patch%edges%inv_primal_edge_length(:,1); &
        !$DSL     area_edge=p_patch%edges%area_edge(:,1); &
        !$DSL     kh_smag_e=kh_smag_e(:,:,1); &
        !$DSL     diff_multfac_vn=diff_multfac_vn(:); &
        !$DSL     nudgecoeff_e=p_int%nudgecoeff_e(:,1); &
        !$DSL     vn=p_nh_prog%vn(:,:,1); &
        !$DSL     edge=horizontal_idx(:); &
        !$DSL     nudgezone_diff=nudgezone_diff; &
        !$DSL     fac_bdydiff_v=fac_bdydiff_v; &
        !$DSL     start_2nd_nudge_line_idx_e=start_2nd_nudge_line_idx_e-1; &
        !$DSL     limited_area=l_limited_area; &
        !$DSL     vn_rel_tol=1e-11_wp; &
        !$DSL     vertical_lower=1; &
        !$DSL     vertical_upper=nlev; &
        !$DSL     horizontal_lower=start_bdydiff_idx_e; &
        !$DSL     horizontal_upper=end_interior_idx_e)

        !$DSL START STENCIL(name=calculate_nabla4; u_vert=u_vert(:,:,1); v_vert=v_vert(:,:,1); &
        !$DSL       primal_normal_vert_v1=p_patch%edges%primal_normal_vert_x(:,:,1); &
        !$DSL       primal_normal_vert_v2=p_patch%edges%primal_normal_vert_y(:,:,1); &
        !$DSL       z_nabla2_e=z_nabla2_e(:,:,1); inv_vert_vert_length=p_patch%edges%inv_vert_vert_length(:,1); &
        !$DSL       inv_primal_edge_length=p_patch%edges%inv_primal_edge_length(:,1); z_nabla4_e2_abs_tol=1e-27_wp; &
        !$DSL       z_nabla4_e2=z_nabla4_e2(:, :); vertical_lower=1; vertical_upper=nlev; horizontal_lower=i_startidx; &
        !$DSL       horizontal_upper=i_endidx)

        !$DSL END STENCIL(name=calculate_nabla4)

        !$DSL START STENCIL(name=apply_nabla2_and_nabla4_to_vn; nudgezone_diff=nudgezone_diff; area_edge=p_patch%edges%area_edge(:,1); &
        !$DSL       kh_smag_e=kh_smag_e(:,:,1); z_nabla2_e=z_nabla2_e(:,:,1); z_nabla4_e2=z_nabla4_e2(:,:); &
        !$DSL       diff_multfac_vn=diff_multfac_vn(:); nudgecoeff_e=p_int%nudgecoeff_e(:,1); vn=p_nh_prog%vn(:,:,1); vn_rel_tol=1e-11_wp; &
        !$DSL       vertical_lower=1; vertical_upper=nlev; horizontal_lower=i_startidx; horizontal_upper=i_endidx)

        !$DSL END STENCIL(name=apply_nabla2_and_nabla4_to_vn)

          !$DSL START STENCIL(name=apply_nabla2_and_nabla4_global_to_vn; area_edge=p_patch%edges%area_edge(:,1); kh_smag_e=kh_smag_e(:,:,1); &
          !$DSL              z_nabla2_e=z_nabla2_e(:,:,1); z_nabla4_e2=z_nabla4_e2(:,:); diff_multfac_vn=diff_multfac_vn(:); vn=p_nh_prog%vn(:,:,1); &
          !$DSL              vn_rel_tol=1e-10_wp; vertical_lower=1; vertical_upper=nlev; horizontal_lower=i_startidx; horizontal_upper=i_endidx)

          !$DSL END STENCIL(name=apply_nabla2_and_nabla4_global_to_vn)

        !$DSL INSERT(start_bdydiff_idx_e = i_startidx)

        !$DSL START STENCIL(name=apply_nabla2_to_vn_in_lateral_boundary; z_nabla2_e=z_nabla2_e(:,:,1); area_edge=p_patch%edges%area_edge(:,1); &
        !$DSL       fac_bdydiff_v=fac_bdydiff_v; vn=p_nh_prog%vn(:,:,1); vn_abs_tol=1e-14_wp; vertical_lower=1; &
        !$DSL       vertical_upper=nlev; horizontal_lower=i_startidx; horizontal_upper=i_endidx)

        !$DSL END STENCIL(name=apply_nabla2_to_vn_in_lateral_boundary)

    !$DSL END FUSED STENCIL(name=apply_diffusion_to_vn)


        !$DSL INSERT(start_nudging_idx_c = i_startidx)
        !$DSL INSERT(end_halo_1_idx_c = i_endidx)

        !$DSL INSERT(!$ACC PARALLEL IF( i_am_accel_node ) DEFAULT(PRESENT) ASYNC(1))
        !$DSL INSERT(w_old(:,:,:) = p_nh_prog%w(:,:,:))
        !$DSL INSERT(!$ACC END PARALLEL)

        !$DSL START FUSED STENCIL(name=apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence; &
        !$DSL       area=p_patch%cells%area(:,1); geofac_grg_x=p_int%geofac_grg(:,:,1,1); &
        !$DSL       geofac_grg_y=p_int%geofac_grg(:,:,1,2); geofac_n2s=p_int%geofac_n2s(:,:,1); &
        !$DSL       w_old=w_old(:,:,1); w=p_nh_prog%w(:,:,1); diff_multfac_w=diff_multfac_w; &
        !$DSL       diff_multfac_n2w=diff_multfac_n2w(:); k=vertical_idx(:); &
        !$DSL       cell=horizontal_idx(:); nrdmax=nrdmax(jg); interior_idx=start_interior_idx_c-1; &
        !$DSL       halo_idx=end_interior_idx_c; dwdx=p_nh_diag%dwdx(:,:,1); &
        !$DSL       dwdy=p_nh_diag%dwdy(:,:,1); &
        !$DSL       w_rel_tol=1e-09_wp; dwdx_rel_tol=1e-09_wp; dwdy_abs_tol=1e-09_wp; &
        !$DSL       vertical_lower=1; vertical_upper=nlev; horizontal_lower=start_nudging_idx_c; &
        !$DSL       horizontal_upper=end_halo_1_idx_c)

        !$DSL START STENCIL(name=calculate_nabla2_for_w; w=p_nh_prog%w(:,:,1); geofac_n2s=p_int%geofac_n2s(:,:,1); &
        !$DSL       z_nabla2_c=z_nabla2_c(:,:,1); z_nabla2_c_abs_tol=1e-21_wp; vertical_lower=1; vertical_upper=nlev; &
        !$DSL       horizontal_lower=i_startidx; horizontal_upper=i_endidx)

        !$DSL END STENCIL(name=calculate_nabla2_for_w)

          !$DSL START STENCIL(name=calculate_horizontal_gradients_for_turbulence; w=p_nh_prog%w(:,:,1); geofac_grg_x=p_int%geofac_grg(:,:,1,1); geofac_grg_y=p_int%geofac_grg(:,:,1,2); &
          !$DSL       dwdx=p_nh_diag%dwdx(:,:,1); dwdy=p_nh_diag%dwdy(:,:,1); dwdx_rel_tol=1e-09_wp; dwdy_rel_tol=1e-09_wp; vertical_lower=2; &
          !$DSL       vertical_upper=nlev; horizontal_lower=i_startidx; horizontal_upper=i_endidx)

          !$DSL END STENCIL(name=calculate_horizontal_gradients_for_turbulence)

        !$DSL INSERT(start_interior_idx_c = i_startidx)
        !$DSL INSERT(end_interior_idx_c = i_endidx)

        !$DSL START STENCIL(name=apply_nabla2_to_w; diff_multfac_w=diff_multfac_w; area=p_patch%cells%area(:,1); &
        !$DSL       z_nabla2_c=z_nabla2_c(:,:,1); geofac_n2s=p_int%geofac_n2s(:,:,1); w=p_nh_prog%w(:,:,1); &
        !$DSL       w_abs_tol=1e-15_wp; vertical_lower=1; vertical_upper=nlev; horizontal_lower=i_startidx; &
        !$DSL       horizontal_upper=i_endidx)

        !$DSL END STENCIL(name=apply_nabla2_to_w)

                  !$DSL START STENCIL(name=apply_nabla2_to_w_in_upper_damping_layer; w=p_nh_prog%w(:,:,1); diff_multfac_n2w=diff_multfac_n2w(:); &
          !$DSL       cell_area=p_patch%cells%area(:,1); z_nabla2_c=z_nabla2_c(:,:,1); vertical_lower=2; w_abs_tol=1e-16_wp; w_rel_tol=1e-10_wp; &
          !$DSL       vertical_upper=nrdmax(jg); horizontal_lower=i_startidx; horizontal_upper=i_endidx)

          !$DSL END STENCIL(name=apply_nabla2_to_w_in_upper_damping_layer)

        !$DSL END FUSED STENCIL(name=apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence)
    """


FREE_FORM_STENCIL = """\
    !$DSL IMPORTS()

    !$DSL START CREATE()

    !$DSL DECLARE(z_q=nproma,p_patch%nlev; field_to_zero_vp=nproma,p_patch%nlev)

    !$DSL INSERT(some custom fields go here)

    !$DSL START STENCIL(name=set_cell_kdim_field_to_zero_vp; field_to_zero_vp=z_alpha(:,:); vertical_lower=nlevp1; &
    !$DSL               vertical_upper=nlevp1; horizontal_lower=i_startidx; horizontal_upper=i_endidx)

        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(NONE) ASYNC(1)
        !$ACC LOOP GANG VECTOR
        DO jc = i_startidx, i_endidx
          z_alpha(jc,nlevp1) = 0.0_wp
          !
          ! Note: z_q is used in the tridiagonal matrix solver for w below.
          !       z_q(1) is always zero, irrespective of w(1)=0 or w(1)/=0
          !       z_q(1)=0 is equivalent to cp(slev)=c(slev)/b(slev) in mo_math_utilities:tdma_solver_vec
          z_q(jc,1) = 0._vp
        ENDDO
        !$ACC END PARALLEL

    !$DSL INSERT(some custom code goes here)

    !$DSL END STENCIL(name=set_cell_kdim_field_to_zero_vp)

    !$DSL END CREATE()
"""
