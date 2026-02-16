module diffusion
   use, intrinsic :: iso_c_binding
   implicit none

   public :: diffusion_run

   public :: diffusion_init

   interface

      function diffusion_run_wrapper(w, &
                                     w_size_0, &
                                     w_size_1, &
                                     vn, &
                                     vn_size_0, &
                                     vn_size_1, &
                                     exner, &
                                     exner_size_0, &
                                     exner_size_1, &
                                     theta_v, &
                                     theta_v_size_0, &
                                     theta_v_size_1, &
                                     rho, &
                                     rho_size_0, &
                                     rho_size_1, &
                                     hdef_ic, &
                                     hdef_ic_size_0, &
                                     hdef_ic_size_1, &
                                     div_ic, &
                                     div_ic_size_0, &
                                     div_ic_size_1, &
                                     dwdx, &
                                     dwdx_size_0, &
                                     dwdx_size_1, &
                                     dwdy, &
                                     dwdy_size_0, &
                                     dwdy_size_1, &
                                     dtime, &
                                     linit, &
                                     on_gpu) bind(c, name="diffusion_run_wrapper") result(rc)
         import :: c_int, c_double, c_bool, c_ptr
         integer(c_int) :: rc  ! Stores the return code

         type(c_ptr), value, target :: w

         integer(c_int), value :: w_size_0

         integer(c_int), value :: w_size_1

         type(c_ptr), value, target :: vn

         integer(c_int), value :: vn_size_0

         integer(c_int), value :: vn_size_1

         type(c_ptr), value, target :: exner

         integer(c_int), value :: exner_size_0

         integer(c_int), value :: exner_size_1

         type(c_ptr), value, target :: theta_v

         integer(c_int), value :: theta_v_size_0

         integer(c_int), value :: theta_v_size_1

         type(c_ptr), value, target :: rho

         integer(c_int), value :: rho_size_0

         integer(c_int), value :: rho_size_1

         type(c_ptr), value, target :: hdef_ic

         integer(c_int), value :: hdef_ic_size_0

         integer(c_int), value :: hdef_ic_size_1

         type(c_ptr), value, target :: div_ic

         integer(c_int), value :: div_ic_size_0

         integer(c_int), value :: div_ic_size_1

         type(c_ptr), value, target :: dwdx

         integer(c_int), value :: dwdx_size_0

         integer(c_int), value :: dwdx_size_1

         type(c_ptr), value, target :: dwdy

         integer(c_int), value :: dwdy_size_0

         integer(c_int), value :: dwdy_size_1

         real(c_double), value, target :: dtime

         logical(c_int), value, target :: linit

         logical(c_int), value :: on_gpu

      end function diffusion_run_wrapper

      function diffusion_init_wrapper(theta_ref_mc, &
                                      theta_ref_mc_size_0, &
                                      theta_ref_mc_size_1, &
                                      wgtfac_c, &
                                      wgtfac_c_size_0, &
                                      wgtfac_c_size_1, &
                                      e_bln_c_s, &
                                      e_bln_c_s_size_0, &
                                      e_bln_c_s_size_1, &
                                      geofac_div, &
                                      geofac_div_size_0, &
                                      geofac_div_size_1, &
                                      geofac_grg_x, &
                                      geofac_grg_x_size_0, &
                                      geofac_grg_x_size_1, &
                                      geofac_grg_y, &
                                      geofac_grg_y_size_0, &
                                      geofac_grg_y_size_1, &
                                      geofac_n2s, &
                                      geofac_n2s_size_0, &
                                      geofac_n2s_size_1, &
                                      nudgecoeff_e, &
                                      nudgecoeff_e_size_0, &
                                      rbf_coeff_1, &
                                      rbf_coeff_1_size_0, &
                                      rbf_coeff_1_size_1, &
                                      rbf_coeff_2, &
                                      rbf_coeff_2_size_0, &
                                      rbf_coeff_2_size_1, &
                                      zd_cellidx, &
                                      zd_cellidx_size_0, &
                                      zd_vertidx, &
                                      zd_vertidx_size_0, &
                                      zd_vertidx_size_1, &
                                      zd_vertidx_size_2, &
                                      zd_vertidx_size_3, &
                                      zd_intcoef, &
                                      zd_intcoef_size_0, &
                                      zd_intcoef_size_1, &
                                      zd_intcoef_size_2, &
                                      zd_diffcoef, &
                                      zd_diffcoef_size_0, &
                                      ndyn_substeps, &
                                      diffusion_type, &
                                      hdiff_w, &
                                      hdiff_vn, &
                                      zdiffu_t, &
                                      type_t_diffu, &
                                      type_vn_diffu, &
                                      hdiff_efdt_ratio, &
                                      smagorinski_scaling_factor, &
                                      hdiff_temp, &
                                      thslp_zdiffu, &
                                      thhgtd_zdiffu, &
                                      denom_diffu_v, &
                                      nudge_max_coeff, &
                                      itype_sher, &
                                      ltkeshs, &
                                      backend, &
                                      on_gpu) bind(c, name="diffusion_init_wrapper") result(rc)
         import :: c_int, c_double, c_bool, c_ptr
         integer(c_int) :: rc  ! Stores the return code

         type(c_ptr), value, target :: theta_ref_mc

         integer(c_int), value :: theta_ref_mc_size_0

         integer(c_int), value :: theta_ref_mc_size_1

         type(c_ptr), value, target :: wgtfac_c

         integer(c_int), value :: wgtfac_c_size_0

         integer(c_int), value :: wgtfac_c_size_1

         type(c_ptr), value, target :: e_bln_c_s

         integer(c_int), value :: e_bln_c_s_size_0

         integer(c_int), value :: e_bln_c_s_size_1

         type(c_ptr), value, target :: geofac_div

         integer(c_int), value :: geofac_div_size_0

         integer(c_int), value :: geofac_div_size_1

         type(c_ptr), value, target :: geofac_grg_x

         integer(c_int), value :: geofac_grg_x_size_0

         integer(c_int), value :: geofac_grg_x_size_1

         type(c_ptr), value, target :: geofac_grg_y

         integer(c_int), value :: geofac_grg_y_size_0

         integer(c_int), value :: geofac_grg_y_size_1

         type(c_ptr), value, target :: geofac_n2s

         integer(c_int), value :: geofac_n2s_size_0

         integer(c_int), value :: geofac_n2s_size_1

         type(c_ptr), value, target :: nudgecoeff_e

         integer(c_int), value :: nudgecoeff_e_size_0

         type(c_ptr), value, target :: rbf_coeff_1

         integer(c_int), value :: rbf_coeff_1_size_0

         integer(c_int), value :: rbf_coeff_1_size_1

         type(c_ptr), value, target :: rbf_coeff_2

         integer(c_int), value :: rbf_coeff_2_size_0

         integer(c_int), value :: rbf_coeff_2_size_1

         type(c_ptr), value, target :: zd_cellidx

         integer(c_int), value :: zd_cellidx_size_0

         type(c_ptr), value, target :: zd_vertidx

         integer(c_int), value :: zd_vertidx_size_0

         integer(c_int), value :: zd_vertidx_size_1

         integer(c_int), value :: zd_vertidx_size_2

         integer(c_int), value :: zd_vertidx_size_3

         type(c_ptr), value, target :: zd_intcoef

         integer(c_int), value :: zd_intcoef_size_0

         integer(c_int), value :: zd_intcoef_size_1

         integer(c_int), value :: zd_intcoef_size_2

         type(c_ptr), value, target :: zd_diffcoef

         integer(c_int), value :: zd_diffcoef_size_0

         integer(c_int), value, target :: ndyn_substeps

         integer(c_int), value, target :: diffusion_type

         logical(c_int), value, target :: hdiff_w

         logical(c_int), value, target :: hdiff_vn

         logical(c_int), value, target :: zdiffu_t

         integer(c_int), value, target :: type_t_diffu

         integer(c_int), value, target :: type_vn_diffu

         real(c_double), value, target :: hdiff_efdt_ratio

         real(c_double), value, target :: smagorinski_scaling_factor

         logical(c_int), value, target :: hdiff_temp

         real(c_double), value, target :: thslp_zdiffu

         real(c_double), value, target :: thhgtd_zdiffu

         real(c_double), value, target :: denom_diffu_v

         real(c_double), value, target :: nudge_max_coeff

         integer(c_int), value, target :: itype_sher

         logical(c_int), value, target :: ltkeshs

         integer(c_int), value, target :: backend

         logical(c_int), value :: on_gpu

      end function diffusion_init_wrapper

   end interface

contains

   subroutine diffusion_run(w, &
                            vn, &
                            exner, &
                            theta_v, &
                            rho, &
                            hdef_ic, &
                            div_ic, &
                            dwdx, &
                            dwdy, &
                            dtime, &
                            linit, &
                            rc)
      use, intrinsic :: iso_c_binding

      real(c_double), dimension(:, :), target :: w

      real(c_double), dimension(:, :), target :: vn

      real(c_double), dimension(:, :), target :: exner

      real(c_double), dimension(:, :), target :: theta_v

      real(c_double), dimension(:, :), target :: rho

      real(c_double), dimension(:, :), pointer :: hdef_ic

      real(c_double), dimension(:, :), pointer :: div_ic

      real(c_double), dimension(:, :), pointer :: dwdx

      real(c_double), dimension(:, :), pointer :: dwdy

      real(c_double), value, target :: dtime

      logical(c_int), value, target :: linit

      logical(c_int) :: on_gpu

      integer(c_int) :: w_size_0

      integer(c_int) :: w_size_1

      integer(c_int) :: vn_size_0

      integer(c_int) :: vn_size_1

      integer(c_int) :: exner_size_0

      integer(c_int) :: exner_size_1

      integer(c_int) :: theta_v_size_0

      integer(c_int) :: theta_v_size_1

      integer(c_int) :: rho_size_0

      integer(c_int) :: rho_size_1

      integer(c_int) :: hdef_ic_size_0

      integer(c_int) :: hdef_ic_size_1

      integer(c_int) :: div_ic_size_0

      integer(c_int) :: div_ic_size_1

      integer(c_int) :: dwdx_size_0

      integer(c_int) :: dwdx_size_1

      integer(c_int) :: dwdy_size_0

      integer(c_int) :: dwdy_size_1

      integer(c_int) :: rc  ! Stores the return code
      ! ptrs

      type(c_ptr) :: hdef_ic_ptr

      type(c_ptr) :: div_ic_ptr

      type(c_ptr) :: dwdx_ptr

      type(c_ptr) :: dwdy_ptr

      hdef_ic_ptr = c_null_ptr

      div_ic_ptr = c_null_ptr

      dwdx_ptr = c_null_ptr

      dwdy_ptr = c_null_ptr

      !$acc host_data use_device(w)
      !$acc host_data use_device(vn)
      !$acc host_data use_device(exner)
      !$acc host_data use_device(theta_v)
      !$acc host_data use_device(rho)
      !$acc host_data use_device(hdef_ic) if(associated(hdef_ic))
      !$acc host_data use_device(div_ic) if(associated(div_ic))
      !$acc host_data use_device(dwdx) if(associated(dwdx))
      !$acc host_data use_device(dwdy) if(associated(dwdy))

#ifdef _OPENACC
      on_gpu = .True.
#else
      on_gpu = .False.
#endif

      w_size_0 = SIZE(w, 1)
      w_size_1 = SIZE(w, 2)

      vn_size_0 = SIZE(vn, 1)
      vn_size_1 = SIZE(vn, 2)

      exner_size_0 = SIZE(exner, 1)
      exner_size_1 = SIZE(exner, 2)

      theta_v_size_0 = SIZE(theta_v, 1)
      theta_v_size_1 = SIZE(theta_v, 2)

      rho_size_0 = SIZE(rho, 1)
      rho_size_1 = SIZE(rho, 2)

      if (associated(hdef_ic)) then
         hdef_ic_ptr = c_loc(hdef_ic)
         hdef_ic_size_0 = SIZE(hdef_ic, 1)
         hdef_ic_size_1 = SIZE(hdef_ic, 2)
      end if

      if (associated(div_ic)) then
         div_ic_ptr = c_loc(div_ic)
         div_ic_size_0 = SIZE(div_ic, 1)
         div_ic_size_1 = SIZE(div_ic, 2)
      end if

      if (associated(dwdx)) then
         dwdx_ptr = c_loc(dwdx)
         dwdx_size_0 = SIZE(dwdx, 1)
         dwdx_size_1 = SIZE(dwdx, 2)
      end if

      if (associated(dwdy)) then
         dwdy_ptr = c_loc(dwdy)
         dwdy_size_0 = SIZE(dwdy, 1)
         dwdy_size_1 = SIZE(dwdy, 2)
      end if

      rc = diffusion_run_wrapper(w=c_loc(w), &
                                 w_size_0=w_size_0, &
                                 w_size_1=w_size_1, &
                                 vn=c_loc(vn), &
                                 vn_size_0=vn_size_0, &
                                 vn_size_1=vn_size_1, &
                                 exner=c_loc(exner), &
                                 exner_size_0=exner_size_0, &
                                 exner_size_1=exner_size_1, &
                                 theta_v=c_loc(theta_v), &
                                 theta_v_size_0=theta_v_size_0, &
                                 theta_v_size_1=theta_v_size_1, &
                                 rho=c_loc(rho), &
                                 rho_size_0=rho_size_0, &
                                 rho_size_1=rho_size_1, &
                                 hdef_ic=hdef_ic_ptr, &
                                 hdef_ic_size_0=hdef_ic_size_0, &
                                 hdef_ic_size_1=hdef_ic_size_1, &
                                 div_ic=div_ic_ptr, &
                                 div_ic_size_0=div_ic_size_0, &
                                 div_ic_size_1=div_ic_size_1, &
                                 dwdx=dwdx_ptr, &
                                 dwdx_size_0=dwdx_size_0, &
                                 dwdx_size_1=dwdx_size_1, &
                                 dwdy=dwdy_ptr, &
                                 dwdy_size_0=dwdy_size_0, &
                                 dwdy_size_1=dwdy_size_1, &
                                 dtime=dtime, &
                                 linit=linit, &
                                 on_gpu=on_gpu)
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
   end subroutine diffusion_run

   subroutine diffusion_init(theta_ref_mc, &
                             wgtfac_c, &
                             e_bln_c_s, &
                             geofac_div, &
                             geofac_grg_x, &
                             geofac_grg_y, &
                             geofac_n2s, &
                             nudgecoeff_e, &
                             rbf_coeff_1, &
                             rbf_coeff_2, &
                             zd_cellidx, &
                             zd_vertidx, &
                             zd_intcoef, &
                             zd_diffcoef, &
                             ndyn_substeps, &
                             diffusion_type, &
                             hdiff_w, &
                             hdiff_vn, &
                             zdiffu_t, &
                             type_t_diffu, &
                             type_vn_diffu, &
                             hdiff_efdt_ratio, &
                             smagorinski_scaling_factor, &
                             hdiff_temp, &
                             thslp_zdiffu, &
                             thhgtd_zdiffu, &
                             denom_diffu_v, &
                             nudge_max_coeff, &
                             itype_sher, &
                             ltkeshs, &
                             backend, &
                             rc)
      use, intrinsic :: iso_c_binding

      real(c_double), dimension(:, :), target :: theta_ref_mc

      real(c_double), dimension(:, :), target :: wgtfac_c

      real(c_double), dimension(:, :), target :: e_bln_c_s

      real(c_double), dimension(:, :), target :: geofac_div

      real(c_double), dimension(:, :), target :: geofac_grg_x

      real(c_double), dimension(:, :), target :: geofac_grg_y

      real(c_double), dimension(:, :), target :: geofac_n2s

      real(c_double), dimension(:), target :: nudgecoeff_e

      real(c_double), dimension(:, :), target :: rbf_coeff_1

      real(c_double), dimension(:, :), target :: rbf_coeff_2

      integer(c_int), dimension(:), pointer :: zd_cellidx

      integer(c_int), dimension(:, :, :, :), pointer :: zd_vertidx

      real(c_double), dimension(:, :, :), pointer :: zd_intcoef

      real(c_double), dimension(:), pointer :: zd_diffcoef

      integer(c_int), value, target :: ndyn_substeps

      integer(c_int), value, target :: diffusion_type

      logical(c_int), value, target :: hdiff_w

      logical(c_int), value, target :: hdiff_vn

      logical(c_int), value, target :: zdiffu_t

      integer(c_int), value, target :: type_t_diffu

      integer(c_int), value, target :: type_vn_diffu

      real(c_double), value, target :: hdiff_efdt_ratio

      real(c_double), value, target :: smagorinski_scaling_factor

      logical(c_int), value, target :: hdiff_temp

      real(c_double), value, target :: thslp_zdiffu

      real(c_double), value, target :: thhgtd_zdiffu

      real(c_double), value, target :: denom_diffu_v

      real(c_double), value, target :: nudge_max_coeff

      integer(c_int), value, target :: itype_sher

      logical(c_int), value, target :: ltkeshs

      integer(c_int), value, target :: backend

      logical(c_int) :: on_gpu

      integer(c_int) :: theta_ref_mc_size_0

      integer(c_int) :: theta_ref_mc_size_1

      integer(c_int) :: wgtfac_c_size_0

      integer(c_int) :: wgtfac_c_size_1

      integer(c_int) :: e_bln_c_s_size_0

      integer(c_int) :: e_bln_c_s_size_1

      integer(c_int) :: geofac_div_size_0

      integer(c_int) :: geofac_div_size_1

      integer(c_int) :: geofac_grg_x_size_0

      integer(c_int) :: geofac_grg_x_size_1

      integer(c_int) :: geofac_grg_y_size_0

      integer(c_int) :: geofac_grg_y_size_1

      integer(c_int) :: geofac_n2s_size_0

      integer(c_int) :: geofac_n2s_size_1

      integer(c_int) :: nudgecoeff_e_size_0

      integer(c_int) :: rbf_coeff_1_size_0

      integer(c_int) :: rbf_coeff_1_size_1

      integer(c_int) :: rbf_coeff_2_size_0

      integer(c_int) :: rbf_coeff_2_size_1

      integer(c_int) :: zd_cellidx_size_0

      integer(c_int) :: zd_vertidx_size_0

      integer(c_int) :: zd_vertidx_size_1

      integer(c_int) :: zd_vertidx_size_2

      integer(c_int) :: zd_vertidx_size_3

      integer(c_int) :: zd_intcoef_size_0

      integer(c_int) :: zd_intcoef_size_1

      integer(c_int) :: zd_intcoef_size_2

      integer(c_int) :: zd_diffcoef_size_0

      integer(c_int) :: rc  ! Stores the return code
      ! ptrs

      type(c_ptr) :: zd_cellidx_ptr

      type(c_ptr) :: zd_vertidx_ptr

      type(c_ptr) :: zd_intcoef_ptr

      type(c_ptr) :: zd_diffcoef_ptr

      zd_cellidx_ptr = c_null_ptr

      zd_vertidx_ptr = c_null_ptr

      zd_intcoef_ptr = c_null_ptr

      zd_diffcoef_ptr = c_null_ptr

      !$acc host_data use_device(theta_ref_mc)
      !$acc host_data use_device(wgtfac_c)
      !$acc host_data use_device(e_bln_c_s)
      !$acc host_data use_device(geofac_div)
      !$acc host_data use_device(geofac_grg_x)
      !$acc host_data use_device(geofac_grg_y)
      !$acc host_data use_device(geofac_n2s)
      !$acc host_data use_device(nudgecoeff_e)
      !$acc host_data use_device(rbf_coeff_1)
      !$acc host_data use_device(rbf_coeff_2)
      !$acc host_data use_device(zd_cellidx) if(associated(zd_cellidx))
      !$acc host_data use_device(zd_vertidx) if(associated(zd_vertidx))
      !$acc host_data use_device(zd_intcoef) if(associated(zd_intcoef))
      !$acc host_data use_device(zd_diffcoef) if(associated(zd_diffcoef))

#ifdef _OPENACC
      on_gpu = .True.
#else
      on_gpu = .False.
#endif

      theta_ref_mc_size_0 = SIZE(theta_ref_mc, 1)
      theta_ref_mc_size_1 = SIZE(theta_ref_mc, 2)

      wgtfac_c_size_0 = SIZE(wgtfac_c, 1)
      wgtfac_c_size_1 = SIZE(wgtfac_c, 2)

      e_bln_c_s_size_0 = SIZE(e_bln_c_s, 1)
      e_bln_c_s_size_1 = SIZE(e_bln_c_s, 2)

      geofac_div_size_0 = SIZE(geofac_div, 1)
      geofac_div_size_1 = SIZE(geofac_div, 2)

      geofac_grg_x_size_0 = SIZE(geofac_grg_x, 1)
      geofac_grg_x_size_1 = SIZE(geofac_grg_x, 2)

      geofac_grg_y_size_0 = SIZE(geofac_grg_y, 1)
      geofac_grg_y_size_1 = SIZE(geofac_grg_y, 2)

      geofac_n2s_size_0 = SIZE(geofac_n2s, 1)
      geofac_n2s_size_1 = SIZE(geofac_n2s, 2)

      nudgecoeff_e_size_0 = SIZE(nudgecoeff_e, 1)

      rbf_coeff_1_size_0 = SIZE(rbf_coeff_1, 1)
      rbf_coeff_1_size_1 = SIZE(rbf_coeff_1, 2)

      rbf_coeff_2_size_0 = SIZE(rbf_coeff_2, 1)
      rbf_coeff_2_size_1 = SIZE(rbf_coeff_2, 2)

      if (associated(zd_cellidx)) then
         zd_cellidx_ptr = c_loc(zd_cellidx)
         zd_cellidx_size_0 = SIZE(zd_cellidx, 1)
      end if

      if (associated(zd_vertidx)) then
         zd_vertidx_ptr = c_loc(zd_vertidx)
         zd_vertidx_size_0 = SIZE(zd_vertidx, 1)
         zd_vertidx_size_1 = SIZE(zd_vertidx, 2)
         zd_vertidx_size_2 = SIZE(zd_vertidx, 3)
         zd_vertidx_size_3 = SIZE(zd_vertidx, 4)
      end if

      if (associated(zd_intcoef)) then
         zd_intcoef_ptr = c_loc(zd_intcoef)
         zd_intcoef_size_0 = SIZE(zd_intcoef, 1)
         zd_intcoef_size_1 = SIZE(zd_intcoef, 2)
         zd_intcoef_size_2 = SIZE(zd_intcoef, 3)
      end if

      if (associated(zd_diffcoef)) then
         zd_diffcoef_ptr = c_loc(zd_diffcoef)
         zd_diffcoef_size_0 = SIZE(zd_diffcoef, 1)
      end if

      rc = diffusion_init_wrapper(theta_ref_mc=c_loc(theta_ref_mc), &
                                  theta_ref_mc_size_0=theta_ref_mc_size_0, &
                                  theta_ref_mc_size_1=theta_ref_mc_size_1, &
                                  wgtfac_c=c_loc(wgtfac_c), &
                                  wgtfac_c_size_0=wgtfac_c_size_0, &
                                  wgtfac_c_size_1=wgtfac_c_size_1, &
                                  e_bln_c_s=c_loc(e_bln_c_s), &
                                  e_bln_c_s_size_0=e_bln_c_s_size_0, &
                                  e_bln_c_s_size_1=e_bln_c_s_size_1, &
                                  geofac_div=c_loc(geofac_div), &
                                  geofac_div_size_0=geofac_div_size_0, &
                                  geofac_div_size_1=geofac_div_size_1, &
                                  geofac_grg_x=c_loc(geofac_grg_x), &
                                  geofac_grg_x_size_0=geofac_grg_x_size_0, &
                                  geofac_grg_x_size_1=geofac_grg_x_size_1, &
                                  geofac_grg_y=c_loc(geofac_grg_y), &
                                  geofac_grg_y_size_0=geofac_grg_y_size_0, &
                                  geofac_grg_y_size_1=geofac_grg_y_size_1, &
                                  geofac_n2s=c_loc(geofac_n2s), &
                                  geofac_n2s_size_0=geofac_n2s_size_0, &
                                  geofac_n2s_size_1=geofac_n2s_size_1, &
                                  nudgecoeff_e=c_loc(nudgecoeff_e), &
                                  nudgecoeff_e_size_0=nudgecoeff_e_size_0, &
                                  rbf_coeff_1=c_loc(rbf_coeff_1), &
                                  rbf_coeff_1_size_0=rbf_coeff_1_size_0, &
                                  rbf_coeff_1_size_1=rbf_coeff_1_size_1, &
                                  rbf_coeff_2=c_loc(rbf_coeff_2), &
                                  rbf_coeff_2_size_0=rbf_coeff_2_size_0, &
                                  rbf_coeff_2_size_1=rbf_coeff_2_size_1, &
                                  zd_cellidx=zd_cellidx_ptr, &
                                  zd_cellidx_size_0=zd_cellidx_size_0, &
                                  zd_vertidx=zd_vertidx_ptr, &
                                  zd_vertidx_size_0=zd_vertidx_size_0, &
                                  zd_vertidx_size_1=zd_vertidx_size_1, &
                                  zd_vertidx_size_2=zd_vertidx_size_2, &
                                  zd_vertidx_size_3=zd_vertidx_size_3, &
                                  zd_intcoef=zd_intcoef_ptr, &
                                  zd_intcoef_size_0=zd_intcoef_size_0, &
                                  zd_intcoef_size_1=zd_intcoef_size_1, &
                                  zd_intcoef_size_2=zd_intcoef_size_2, &
                                  zd_diffcoef=zd_diffcoef_ptr, &
                                  zd_diffcoef_size_0=zd_diffcoef_size_0, &
                                  ndyn_substeps=ndyn_substeps, &
                                  diffusion_type=diffusion_type, &
                                  hdiff_w=hdiff_w, &
                                  hdiff_vn=hdiff_vn, &
                                  zdiffu_t=zdiffu_t, &
                                  type_t_diffu=type_t_diffu, &
                                  type_vn_diffu=type_vn_diffu, &
                                  hdiff_efdt_ratio=hdiff_efdt_ratio, &
                                  smagorinski_scaling_factor=smagorinski_scaling_factor, &
                                  hdiff_temp=hdiff_temp, &
                                  thslp_zdiffu=thslp_zdiffu, &
                                  thhgtd_zdiffu=thhgtd_zdiffu, &
                                  denom_diffu_v=denom_diffu_v, &
                                  nudge_max_coeff=nudge_max_coeff, &
                                  itype_sher=itype_sher, &
                                  ltkeshs=ltkeshs, &
                                  backend=backend, &
                                  on_gpu=on_gpu)
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
   end subroutine diffusion_init

end module