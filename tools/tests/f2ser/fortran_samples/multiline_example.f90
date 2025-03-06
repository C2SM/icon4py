! ICON4Py - ICON inspired code in Python and GT4Py
!
! Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
! All rights reserved.
!
! Please, refer to the LICENSE file in the root directory.
! SPDX-License-Identifier: BSD-3-Clause

MODULE mo_graupel_granule

  USE, INTRINSIC :: iso_fortran_env, ONLY: wp => real64
  USE gscp_graupel, ONLY: graupel

  IMPLICIT NONE

  PRIVATE :: graupel_parameters, params, mma,mmb
  PUBLIC :: graupel_init, graupel_run

  TYPE graupel_parameters
    ! gscp_data
    INTEGER :: iautocon
    INTEGER :: isnow_n0temp
    REAL(wp) :: ccsrim
    REAL(wp) :: ccsagg
    REAL(wp) :: ccsdep
    REAL(wp) :: ccsvel
    REAL(wp) :: ccsvxp
    REAL(wp) :: ccslam
    REAL(wp) :: ccslxp
    REAL(wp) :: ccsaxp
    REAL(wp) :: ccsdxp
    REAL(wp) :: ccshi1
    REAL(wp) :: ccdvtp
    REAL(wp) :: ccidep
    REAL(wp) :: ccswxp
    REAL(wp) :: zconst
    REAL(wp) :: zcev
    REAL(wp) :: zbev
    REAL(wp) :: zcevxp
    REAL(wp) :: zbevxp
    REAL(wp) :: zvzxp
    REAL(wp) :: zvz0r
    REAL(wp) :: v0snow
    REAL(wp) :: x13o8
    REAL(wp) :: x1o2
    REAL(wp) :: x27o16
    REAL(wp) :: x3o4
    REAL(wp) :: x7o4
    REAL(wp) :: x7o8
    REAL(wp) :: zbvi
    REAL(wp) :: zcac
    REAL(wp) :: zccau
    REAL(wp) :: zciau
    REAL(wp) :: zcicri
    REAL(wp) :: zcrcri
    REAL(wp) :: zcrfrz
    REAL(wp) :: zcrfrz1
    REAL(wp) :: zcrfrz2
    REAL(wp) :: zeps
    REAL(wp) :: zkcac
    REAL(wp) :: zkphi1
    REAL(wp) :: zkphi2
    REAL(wp) :: zkphi3
    REAL(wp) :: zmi0
    REAL(wp) :: zmimax
    REAL(wp) :: zmsmin
    REAL(wp) :: zn0s0
    REAL(wp) :: zn0s1
    REAL(wp) :: zn0s2
    REAL(wp) :: znimax_thom
    REAL(wp) :: zqmin
    REAL(wp) :: zrho0
    REAL(wp) :: zthet
    REAL(wp) :: zthn
    REAL(wp) :: ztmix
    REAL(wp) :: ztrfrz
    REAL(wp) :: zvz0i
    REAL(wp) :: icesedi_exp
    REAL(wp) :: zams
    REAL(wp) :: dist_cldtop_ref
    REAL(wp) :: reduce_dep_ref
    REAL(wp) :: tmin_iceautoconv
    REAL(wp) :: zceff_fac
    REAL(wp) :: zceff_min
    REAL(wp) :: v_sedi_rain_min
    REAL(wp) :: v_sedi_snow_min
    REAL(wp) :: v_sedi_graupel_min

     ! mo_physical constants
    REAL(wp) :: r_v
    REAL(wp) :: lh_v
    REAL(wp) :: lh_s
    REAL(wp) :: cpdr
    REAL(wp) :: cvdr
    REAL(wp) :: b3
    REAL(wp) :: t0
  END TYPE graupel_parameters



  REAL(wp):: mma(10),mmb(10)
  TYPE(graupel_parameters) :: params

CONTAINS

  SUBROUTINE graupel_init( &
    ccsrim,    ccsagg,    ccsdep,    ccsvel,    ccsvxp,    ccslam, &
    ccslxp,    ccsaxp,    ccsdxp,    ccshi1,    ccdvtp,    ccidep, &
    ccswxp,    zconst,    zcev,      zbev,      zcevxp,    zbevxp, &
    zvzxp,     zvz0r,                                              &
    v0snow,                                                        &
    x13o8,     x1o2,      x27o16,    x3o4,      x7o4,      x7o8,   &
    zbvi,      zcac,      zccau,     zciau,     zcicri,            &
    zcrcri,    zcrfrz,    zcrfrz1,   zcrfrz2,   zeps,      zkcac,  &
    zkphi1,    zkphi2,    zkphi3,    zmi0,      zmimax,    zmsmin, &
    zn0s0,     zn0s1,     zn0s2,     znimax_thom,          zqmin,  &
    zrho0,     zthet,     zthn,      ztmix,     ztrfrz,            &
    zvz0i,     icesedi_exp,          zams,              &
    iautocon,  isnow_n0temp, dist_cldtop_ref,   reduce_dep_ref,    &
    tmin_iceautoconv,     zceff_fac, zceff_min,                    &
    mma_driver, mmb_driver, v_sedi_rain_min, v_sedi_snow_min, v_sedi_graupel_min, &
    r_v , & !> gas constant for water vapour
    lh_v, & !! latent heat of vapourization
    lh_s, & !! latent heat of sublimation
    cpdr, & !! (spec. heat of dry air at constant press)^-1
    cvdr, & !! (spec. heat of dry air at const vol)^-1
    b3, &   !! melting temperature of ice/snow
    t0)   !! melting temperature of ice/snow

    INTEGER , INTENT(IN) :: iautocon,isnow_n0temp
    REAL(wp), INTENT(IN) :: ccsrim, &
        ccsagg,    ccsdep,    ccsvel,    ccsvxp,    ccslam, &
      ccslxp,    ccsaxp,    ccsdxp,    ccshi1,    ccdvtp,    ccidep, &
      ccswxp,    zconst,    zcev,      zbev,      zcevxp,    zbevxp, &
      zvzxp,     zvz0r,                                              &
      v0snow,                                                        &
      x13o8,     x1o2,      x27o16,    x3o4,      x7o4,      x7o8,   &
      zbvi,      zcac,      zccau,     zciau,     zcicri,            &
      zcrcri,    zcrfrz,    zcrfrz1,   zcrfrz2,   zeps,      zkcac,  &
      zkphi1,    zkphi2,    zkphi3,    zmi0,      zmimax,    zmsmin, &
      zn0s0,     zn0s1,     zn0s2,     znimax_thom,          zqmin,  &
      zrho0,     zthet,     zthn,      ztmix,     ztrfrz,            &
      zvz0i,     icesedi_exp,          zams,              &
      dist_cldtop_ref,   reduce_dep_ref,    &
      tmin_iceautoconv,     zceff_fac, zceff_min,                    &
      mma_driver(10), mmb_driver(10), v_sedi_rain_min, v_sedi_snow_min, v_sedi_graupel_min, &
      r_v , & !> gas constant for water vapour
      lh_v, & !! latent heat of vapourization
      lh_s, & !! latent heat of sublimation
      cpdr, & !! (spec. heat of dry air at constant press)^-1
      cvdr, & !! (spec. heat of dry air at const vol)^-1
      b3, &   !! melting temperature of ice/snow
      t0   !! melting temperature of ice/snow

     ! gscp_data
     params%ccsrim = ccsrim
     params%ccsagg = ccsagg
     params%ccsdep = ccsdep
     params%ccsvel = ccsvel
     params%ccsvxp = ccsvxp
     params%ccslam = ccslam
     params%ccslxp = ccslxp
     params%ccsaxp = ccsaxp
     params%ccsdxp = ccsdxp
     params%ccshi1 = ccshi1
     params%ccdvtp = ccdvtp
     params%ccidep = ccidep
     params%ccswxp = ccswxp
     params%zconst = zconst
     params%zcev = zcev
     params%zbev = zbev
     params%zcevxp = zcevxp
     params%zbevxp = zbevxp
     params%zvzxp = zvzxp
     params%zvz0r = zvz0r
     params%v0snow = v0snow
     params%x13o8 = x13o8
     params%x1o2 = x1o2
     params%x27o16 = x27o16
     params%x3o4 = x3o4
     params%x7o4 = x7o4
     params%x7o8 = x7o8
     params%zbvi = zbvi
     params%zcac = zcac
     params%zccau = zccau
     params%zciau = zciau
     params%zcicri = zcicri
     params%zcrcri = zcrcri
     params%zcrfrz = zcrfrz
     params%zcrfrz1 = zcrfrz1
     params%zcrfrz2 = zcrfrz2
     params%zeps = zeps
     params%zkcac = zkcac
     params%zkphi1 = zkphi1
     params%zkphi2 = zkphi2
     params%zkphi3 = zkphi3
     params%zmi0 = zmi0
     params%zmimax = zmimax
     params%zmsmin = zmsmin
     params%zn0s0 = zn0s0
     params%zn0s1 = zn0s1
     params%zn0s2 = zn0s2
     params%znimax_thom = znimax_thom
     params%zqmin = zqmin
     params%zrho0 = zrho0
     params%zthet = zthet
     params%zthn = zthn
     params%ztmix = ztmix
     params%ztrfrz = ztrfrz
     params%zvz0i = zvz0i
     params%icesedi_exp = icesedi_exp
     params%zams = zams
     params%iautocon = iautocon
     params%isnow_n0temp = isnow_n0temp
     params%dist_cldtop_ref = dist_cldtop_ref
     params%reduce_dep_ref = reduce_dep_ref
     params%tmin_iceautoconv = tmin_iceautoconv
     params%zceff_fac = zceff_fac
     params%zceff_min = zceff_min
     params%v_sedi_rain_min = v_sedi_rain_min
     params%v_sedi_snow_min = v_sedi_snow_min
     params%v_sedi_graupel_min = v_sedi_graupel_min

     ! mo_physical constants
     params%r_v = r_v
     params%lh_v = lh_v
     params%lh_s = lh_s
     params%cpdr = cpdr
     params%cvdr = cvdr
     params%b3 = b3
     params%t0 = t0



     mma = mma_driver
     mmb = mmb_driver

     !$ACC ENTER DATA COPYIN(mma, mmb)

  END SUBROUTINE graupel_init


  SUBROUTINE graupel_run(             &
    nvec,ke,                           & !> array dimensions
    ivstart,ivend, kstart,             & !! optional start/end indicies
    idbg,                              & !! optional debug level
    zdt, dz,                           & !! numerics parameters
    t,p,rho,qv,qc,qi,qr,qs,qg,qnc,     & !! prognostic variables
    qi0,qc0,                           & !! cloud ice/water threshold for autoconversion
    & b1, &
    & b2w, &
    & b4w, &
    prr_gsp,prs_gsp,pri_gsp,prg_gsp,   & !! surface precipitation rates
    qrsflux,                           & !  total precipitation flux
    l_cv,                              &
    ithermo_water,                     & !  water thermodynamics
    ldass_lhn,                         &
    ldiag_ttend,     ldiag_qtend     , &
    ddt_tend_t     , ddt_tend_qv     , &
    ddt_tend_qc    , ddt_tend_qi     , & !> ddt_tend_xx are tendencies
    ddt_tend_qr    , ddt_tend_qs)!!    necessary for dynamics

    INTEGER, INTENT(IN) :: nvec          ,    & !> number of horizontal points
      ke                     !! number of grid points in vertical direction

    INTEGER, INTENT(IN) ::  ivstart   ,    & !> optional start index for horizontal direction
      ivend     ,    & !! optional end index   for horizontal direction
      kstart    ,    & !! optional start index for the vertical index
      idbg             !! optional debug level

    REAL(KIND=wp), INTENT(IN) :: zdt             ,    & !> time step for integration of microphysics     (  s  )
      qi0,qc0,& !> cloud ice/water threshold for autoconversion
      b1,b2w,b4w

    REAL(KIND=wp), DIMENSION(:,:), INTENT(IN) :: dz              ,    & !> layer thickness of full levels                (  m  )
      rho             ,    & !! density of moist air                          (kg/m3)
      p                      !! pressure                                      ( Pa  )

    LOGICAL, INTENT(IN):: l_cv, &                   !! if true, cv is used instead of cp
      ldass_lhn

    INTEGER, INTENT(IN):: ithermo_water          !! water thermodynamics

    LOGICAL, INTENT(IN):: ldiag_ttend,         & ! if true, temperature tendency shall be diagnosed
      ldiag_qtend            ! if true, moisture tendencies shall be diagnosed

    REAL(KIND=wp), DIMENSION(:,:), INTENT(INOUT) ::  t               ,    & !> temperature                                   (  K  )
      qv              ,    & !! specific water vapor content                  (kg/kg)
      qc              ,    & !! specific cloud water content                  (kg/kg)
      qi              ,    & !! specific cloud ice   content                  (kg/kg)
      qr              ,    & !! specific rain content                         (kg/kg)
      qs              ,    & !! specific snow content                         (kg/kg)
      qg                     !! specific graupel content                      (kg/kg)

    REAL(KIND=wp), INTENT(INOUT) :: qrsflux(:,:)       ! total precipitation flux (nudg)

    REAL(KIND=wp), DIMENSION(:), INTENT(INOUT) ::  prr_gsp,             & !> precipitation rate of rain, grid-scale        (kg/(m2*s))
      prs_gsp,             & !! precipitation rate of snow, grid-scale        (kg/(m2*s))
      prg_gsp,             & !! precipitation rate of graupel, grid-scale     (kg/(m2*s))
      qnc                    !! cloud number concentration

    REAL(KIND=wp), DIMENSION(:), INTENT(INOUT)::   pri_gsp                !! precipitation rate of ice, grid-scale        (kg/(m2*s))

    REAL(KIND=wp), DIMENSION(:,:), INTENT(OUT)::   ddt_tend_t      , & !> tendency T                                       ( 1/s )
      ddt_tend_qv     , & !! tendency qv                                      ( 1/s )
      ddt_tend_qc     , & !! tendency qc                                      ( 1/s )
      ddt_tend_qi     , & !! tendency qi                                      ( 1/s )
      ddt_tend_qr     , & !! tendency qr                                      ( 1/s )
      ddt_tend_qs         !! tendency qs                                      ( 1/s )

          CALL graupel (                                     &
            & nvec   =nvec                            ,    & !> in:  actual array size
            & ke     =ke                              ,    & !< in:  actual array size
            & ivstart=ivstart                        ,    & !< in:  start index of calculation
            & ivend  =ivend                          ,    & !< in:  end index of calculation
            & kstart =kstart                  ,    & !< in:  vertical start index
            & zdt    =zdt                     ,    & !< in:  timestep
            & qi0    =qi0        ,    &
            & qc0    =qc0        ,    &
            & b1     = b1, &
            & b2w    = b2w, &
            & b4w    = b4w, &
            & dz     =dz     ,    & !< in:  vertical layer thickness
            & t      =t           ,    & !< in:  temp,tracer,...
            & p      =p           ,    & !< in:  full level pres
            & rho    =rho          ,    & !< in:  density
            & qv     =qv    ,    & !< in:  spec. humidity
            & qc     =qc    ,    & !< in:  cloud water
            & qi     =qi    ,    & !< in:  cloud ice
            & qr     =qr    ,    & !< in:  rain water
            & qs     =qs    ,    & !< in:  snow
            & qg     =qg    ,    & !< in:  graupel
            & qnc    = qnc                            ,    & !< cloud number concentration
            & prr_gsp=prr_gsp     ,    & !< out: precipitation rate of rain
            & prs_gsp=prs_gsp     ,    & !< out: precipitation rate of snow
            & pri_gsp=pri_gsp      ,    & !< out: precipitation rate of cloud ice
            & prg_gsp=prg_gsp  ,    & !< out: precipitation rate of graupel
            & qrsflux= qrsflux       ,    & !< out: precipitation flux
            & ldiag_ttend = ldiag_ttend                 ,    & !< in:  if temp. tendency shall be diagnosed
            & ldiag_qtend = ldiag_qtend                 ,    & !< in:  if moisture tendencies shall be diagnosed
            & ddt_tend_t  = ddt_tend_t                  ,    & !< out: tendency temperature
            & ddt_tend_qv = ddt_tend_qv                 ,    & !< out: tendency QV
            & ddt_tend_qc = ddt_tend_qc                 ,    & !< out: tendency QC
            & ddt_tend_qi = ddt_tend_qi                 ,    & !< out: tendency QI
            & ddt_tend_qr = ddt_tend_qr                 ,    & !< out: tendency QR
            & ddt_tend_qs = ddt_tend_qs                 ,    & !< out: tendency QS
            & idbg=idbg                          ,    &
            & l_cv=l_cv                               ,    &
            & ldass_lhn = ldass_lhn                     ,    &
            & ithermo_water=ithermo_water, &!< in: latent heat choice
            & ccsrim = params%ccsrim, &
            & ccsagg = params%ccsagg, &
            & ccsdep = params%ccsdep, &
            & ccsvel = params%ccsvel, &
            & ccsvxp = params%ccsvxp, &
            & ccslam = params%ccslam, &
            & ccslxp = params%ccslxp, &
            & ccsaxp = params%ccsaxp, &
            & ccsdxp = params%ccsdxp, &
            & ccshi1 = params%ccshi1, &
            & ccdvtp = params%ccdvtp, &
            & ccidep = params%ccidep, &
            & ccswxp = params%ccswxp, &
            & zconst = params%zconst, &
            & zcev = params%zcev, &
            & zbev = params%zbev, &
            & zcevxp = params%zcevxp, &
            & zbevxp = params%zbevxp, &
            & zvzxp = params%zvzxp, &
            & zvz0r = params%zvz0r, &
            & v0snow = params%v0snow, &
            & x13o8 = params%x13o8, &
            & x1o2 = params%x1o2, &
            & x27o16 = params%x27o16, &
            & x3o4 = params%x3o4, &
            & x7o4 = params%x7o4, &
            & x7o8 = params%x7o8, &
            & zbvi = params%zbvi, &
            & zcac = params%zcac, &
            & zccau = params%zccau, &
            & zciau = params%zciau, &
            & zcicri = params%zcicri, &
            & zcrcri = params%zcrcri, &
            & zcrfrz = params%zcrfrz, &
            & zcrfrz1 = params%zcrfrz1, &
            & zcrfrz2 = params%zcrfrz2, &
            & zeps = params%zeps, &
            & zkcac = params%zkcac, &
            & zkphi1 = params%zkphi1, &
            & zkphi2 = params%zkphi2, &
            & zkphi3 = params%zkphi3, &
            & zmi0 = params%zmi0, &
            & zmimax = params%zmimax, &
            & zmsmin = params%zmsmin, &
            & zn0s0 = params%zn0s0, &
            & zn0s1 = params%zn0s1, &
            & zn0s2 = params%zn0s2, &
            & znimax_thom = params%znimax_thom, &
            & zqmin = params%zqmin, &
            & zrho0 = params%zrho0, &
            & zthet = params%zthet, &
            & zthn = params%zthn, &
            & ztmix = params%ztmix, &
            & ztrfrz = params%ztrfrz, &
            & zvz0i = params%zvz0i, &
            & icesedi_exp = params%icesedi_exp, &
            & zams = params%zams, &
            & iautocon = params%iautocon, &
            & isnow_n0temp = params%isnow_n0temp, &
            & dist_cldtop_ref = params%dist_cldtop_ref, &
            & reduce_dep_ref = params%reduce_dep_ref, &
            & tmin_iceautoconv = params%tmin_iceautoconv, &
            & zceff_fac = params%zceff_fac, &
            & zceff_min = params%zceff_min, &
            & mma = mma, &
            & mmb = mmb, &
            & v_sedi_rain_min = params%v_sedi_rain_min, &
            & v_sedi_snow_min = params%v_sedi_snow_min, &
            & v_sedi_graupel_min = params%v_sedi_graupel_min, &
            & r_v = params%r_v, &
            & lh_v = params%lh_v, &
            & lh_s = params%lh_s, &
            & cpdr = params%cpdr, &
            & cvdr = params%cvdr, &
            & b3 = params%b3, &
              t0 = params%t0)

  END SUBROUTINE graupel_run

END MODULE mo_graupel_granule

