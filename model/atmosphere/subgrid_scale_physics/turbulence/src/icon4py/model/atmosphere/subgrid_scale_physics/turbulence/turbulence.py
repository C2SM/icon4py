# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the NWP 1D turbulence granule (Raschendorfer scheme).

`TurbulenceConfig` mirrors 'turbdiff_nml' of mo_turbdiff_nml.f90 and, beyond it, those members of
't_turbdiff_config' (mo_turbdiff_config.f90) that select a formulation but are not reachable from
a namelist. Every default is the compiled-in default of mo_turbdiff_config.f90.

Carrying parameters the implementation refuses is deliberate. The granule interface is the
contract and has to be expressible from Fortran, C and Python alike, so it accepts anything ICON
can be configured to do; `FROZEN_SWITCHES` is where the implementation says which of those
formulations were ported (port spec D5/D6). Twenty-nine switches select alternatives that were
not ported and are refused with a 'NotImplementedError' that names the one supported value, says
what it means, and points at the Fortran scheme. Eight more do vary operationally across the DWD
and MeteoSwiss setups -- 'itype_sher', 'icldm_turb', 'imode_tkesso', 'imode_charpar',
'frcsmot', 'a_hshr', 'ltkesso' and 'ltkeshs' -- and are supported over the range those setups
need. Which values occur was verified by grepping all 648 configurations under 'icon/run/',
not assumed.

Every other formulation switch is accounted for too, because what D6 rules out is silence, not
acceptance. Eight of them -- 'imode_pat_len', 'imode_snowsmot', 'lconst_z0', 'ldiff_qi',
'ldiff_qs', 'loutsso', 'loutnst' and 'loutbms' -- are accepted at any value because no statement
of the ported scheme reads them: each is either consumed by ICON code that is out of scope (port
spec D1: port the scheme, not the interface) and reaches the granule only through a field the
caller has already filled, or gates an output argument the ICON interfaces never pass. The doc
comment of each field names the line that consumes it.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Final

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence import turbulence_options as options
from icon4py.model.common import constants


__all__ = [
    "FORTRAN_NAMELIST_GROUP",
    "FROZEN_SWITCHES",
    "FrozenSwitch",
    "TurbulenceConfig",
    "TurbulenceParams",
]


#: Group of the echoed ICON namelists that `TurbulenceConfig.from_fortran_dict` reads.
FORTRAN_NAMELIST_GROUP: Final[str] = "turbdiff_nml"


@dataclasses.dataclass(frozen=True)
class FrozenSwitch:
    """A switch whose alternative formulations were not ported.

    The scheme offers several formulations for the same physical step. Only the one every
    operational configuration selects is implemented, so any other setting must fail loudly
    instead of being silently ignored.
    """

    #: Name of the switch, spelled as in 'mo_turbdiff_config.f90'.
    name: str
    #: The single value the granule implements -- the compiled-in Fortran default.
    supported_value: int | bool
    #: What that value means physically, translated from the Fortran declaration.
    meaning: str

    def check(self, value: int | bool) -> None:
        """Raise unless `value` is the one formulation that was ported."""
        if value != self.supported_value:
            raise NotImplementedError(
                f"Only {self.name} = {self.supported_value} ({self.meaning}) is implemented; "
                f"got {value}. Set {self.name} = {self.supported_value} or use the Fortran scheme."
            )


#: The reject list: switches frozen at their compiled-in default. The trailing comment of each
#: entry is the declaration in 'icon/src/configure_model/mo_turbdiff_config.f90' the default and
#: the meaning were read from. Twelve are reachable from 'turbdiff_nml' (mo_turbdiff_nml.f90:
#: 56-71); the other seventeen can only change by editing Fortran. Across all 648 configurations
#: under 'icon/run/' only three settings differ from a value frozen here -- 'imode_frcsmot = 0',
#: 'icldm_tran = -1' and 'lfreeslip = .TRUE.' -- and the entries below record where.
FROZEN_SWITCHES: Final[tuple[FrozenSwitch, ...]] = (
    FrozenSwitch("imode_turb", 1, "prognostic TKE equation"),  # :299
    FrozenSwitch("imode_tran", 0, "diagnostic TKE equation in the transfer scheme"),  # :298
    FrozenSwitch(
        "imode_stbcalc",
        1,
        "stability function always evaluated for unstable stratification, "
        "using a restricted 'gama' in terms of the previous forcing",
    ),  # :318
    FrozenSwitch("imode_tkediff", 2, "implicit TKE diffusion in terms of TKE = q^2 / 2"),  # :383
    FrozenSwitch(
        "imode_trancnf",
        2,
        "1st ConSAT transfer-scheme configuration: estimated friction velocity, no laminar "
        "correction of the profile functions, liquid-water potential temperature interpolated "
        "to the surface level, no upper bound on the TKE forcing",
    ),  # :361
    FrozenSwitch(
        "imode_adshear",
        2,
        "additional shear from scale interaction also entering the stability functions",
    ),  # :386
    FrozenSwitch(
        "imode_tkemini",
        1,
        "lower limits of the diffusion coefficients treated as corrections of the stability "
        "length, without further adaptation of the turbulence model",
    ),  # :372
    FrozenSwitch(
        "imode_suradap", 0, "no adaptation of the surface-layer diffusion coefficients"
    ),  # :378
    FrozenSwitch(
        "imode_vel_min",
        2,
        "minimal turbulent velocity scale with a stability-dependent correction",
    ),  # :223
    FrozenSwitch(
        "imode_tkvmini",
        2,
        "minimal turbulent diffusion coefficients with a stability-dependent correction",
    ),  # :129
    FrozenSwitch(
        "itype_wcld", 2, "water cloud diagnosis by a statistical saturation adjustment"
    ),  # :309
    FrozenSwitch(
        "ilow_def_cond", 2, "zero surface value as the default lower boundary condition"
    ),  # :322
    FrozenSwitch(
        "imode_lamdiff",
        1,
        "laminar diffusion entering only as a limit on the surface-level diffusion coefficients "
        "when the resistance lengths of the roughness layer are computed",
    ),  # :369
    FrozenSwitch(
        "imode_nsf_wind",
        1,
        "near-surface wind taken as the magnitude of the grid-scale averaged wind vector",
    ),  # :168
    FrozenSwitch(
        "imode_stadlim",
        2,
        "statistical saturation adjustment limited by a relative limit on the standard deviation "
        "of the local supersaturation and an upper limit on cloud water",
    ),  # :357
    FrozenSwitch(
        "imode_shshear",
        2,
        "separated horizontal shear with a Richardson-number dependent length-scale correction "
        "and the trace constraint on the 2D strain tensor",
    ),  # :339
    #: The only other value under 'icon/run/' is 0 in 'checksuite.nwp/
    #: nwpexp.run_ICON_03_R19B7N8-ID2_ID1_lam', where 'frcsmot = 0.' switches the smoothing off
    #: altogether, so the setting is inert there. That configuration is refused anyway, on its
    #: 'imode_tkesso = 3'.
    FrozenSwitch(
        "imode_frcsmot",
        2,
        "vertical smoothing of the TKE forcing terms confined to the tropics by 'trop_mask'",
    ),  # :140
    #: The only other value under 'icon/run/' is -1 in 'checksuite.nwp/
    #: nwpexp.run_ICON_02_R2B13_lam', which sets 'icldm_turb = -1' in the same breath and is
    #: therefore already refused by the supported range of 'icldm_turb'.
    FrozenSwitch(
        "icldm_tran",
        2,
        "turbulent sub-grid condensation considered in the transfer scheme as well, as for "
        "'icldm_turb = 2'",
    ),  # :303
    FrozenSwitch(
        "itype_2m_diag",
        1,
        "2 m temperature and dew point diagnosed over the fictive roughness of a SYNOP lawn, "
        "from a purely logarithmic profile",
    ),  # :353
    FrozenSwitch(
        "lexpcor",
        False,
        "no explicit warm-cloud correction of the implicitly calculated turbulent diffusion",
    ),  # :279
    FrozenSwitch(
        "ltmpcor", False, "minor turbulent sources omitted from the enthalpy budget"
    ),  # :276
    FrozenSwitch(
        "lcpfluc", False, "fluctuations of the heat capacity of air not considered"
    ),  # :277
    FrozenSwitch(
        "lcirflx", False, "no non-turbulent fluxes from near-surface circulations"
    ),  # :281
    FrozenSwitch("ltkecon", False, "no convective buoyancy production in the TKE equation"),  # :266
    FrozenSwitch(
        "ltkenst", True, "production by near-surface thermals kept in the TKE equation"
    ),  # :268
    FrozenSwitch(
        "loutshs",
        True,
        "separated horizontal shear production written to the 'tket_hshr' output",
    ),  # :271
    FrozenSwitch(
        "lsflcnd",
        True,
        "surface flux density used as the lower boundary condition of the vertical diffusion, "
        "rather than a surface concentration",
    ),  # :280
    #: Set '.TRUE.' by two idealized configurations, 'exp.exclaim_nh_weisman_klemp_sb' and
    #: 'checksuite.nwp/nwpexp.run_ICON_02_R2B13_lam'; both are refused. Free slip is a
    #: formulation of its own -- it zeroes the surface momentum flux (turb_utilities.f90:2492)
    #: and replaces the near-surface diagnostics (turb_transfer.f90:1992,:2097) -- not a tuning.
    FrozenSwitch(
        "lfreeslip", False, "no free-slip lower boundary; the surface stays coupled"
    ),  # :284
    #: Not a namelist switch and not a member of 't_turbdiff_config': a dummy argument of
    #: 'turbdiff' (turb_diffusion.f90:439) that ICON hardcodes to .FALSE. at both call sites,
    #: mo_nwp_turbdiff_interface.f90:584 and mo_nwp_phy_init.f90:1732, "not yet arranged for ICON".
    FrozenSwitch("l3dturb", False, "no 3D turbulent diffusion"),
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class TurbulenceConfig:
    """Configuration of the turbulence granule.

    Field names are the Fortran names, so that the mapping to 'turbdiff_nml' stays one to one and
    a namelist can be read without a translation table. Defaults are the compiled-in defaults of
    mo_turbdiff_config.f90; the trailing comment on each group gives the section of that file.
    """

    # 1. Numerical parameters (mo_turbdiff_config.f90:124-153)

    #: Implicit weight near the surface (maximal value).
    impl_s: float = 1.20
    #: Implicit weight near the top of the atmosphere (maximal value).
    impl_t: float = 0.75
    #: Mode of calculating the minimal turbulent diffusion coefficients.
    imode_tkvmini: int = 2
    #: Minimal diffusion coefficient for vertical scalar (heat) transport [m^2/s].
    tkhmin: float = 0.75
    #: Minimal diffusion coefficient for vertical momentum transport [m^2/s].
    tkmmin: float = 0.75
    #: Stratospheric value of `tkhmin` [m^2/s].
    tkhmin_strat: float = 0.75
    #: Stratospheric value of `tkmmin` [m^2/s].
    tkmmin_strat: float = 4.00
    #: Smoothing factor for direct time-step iterations.
    ditsmot: float = 0.00
    #: Where the vertical smoothing of the TKE source terms applies if `frcsmot` > 0:
    #: 1 globally, 2 in the tropics only.
    imode_frcsmot: int = 2
    #: Vertical smoothing factor for the TKE forcing. Operationally 0.0 (MCH) or 0.2 (DWD).
    frcsmot: float = 0.00
    #: Time smoothing factor for TKE and the diffusion coefficients.
    tkesmot: float = 0.15
    #: Security factor for the TKE forcing (<= 1).
    frcsecu: float = 1.00
    #: Security factor in the TKE equation (in [0, 1]).
    tkesecu: float = 1.00
    #: Security factor in the stability function (in ]0, 1]).
    stbsecu: float = 0.01
    #: Relative security factor for the profile functions (in ]0, 1[).
    prfsecu: float = 0.50
    #: Relative limit of accuracy for the comparison of numbers.
    epsi: float = 1.0e-6
    #: Number of initialization iterations (>= 0).
    it_end: int = 1

    # 2. Physical properties of the lower boundary (mo_turbdiff_config.f90:155-190)

    #: Scaling factor of the laminar boundary layer for heat.
    rlam_heat: float = 10.0
    #: Scaling factor of the laminar boundary layer for momentum. Should stay at 0 with the
    #: current formulation.
    rlam_mom: float = 0.0
    #: Vapour/heat ratio of the laminar scaling factors over land.
    rat_lam: float = 1.0
    #: Sea/land ratio of the laminar scaling factors for heat and vapour.
    rat_sea: float = 0.8
    #: Glacier/land ratio of the laminar scaling factors for heat and vapour.
    rat_glac: float = 3.0
    #: Ratio of canopy height over sai*z0m.
    rat_can: float = 1.0
    #: Mode of the local wind definition at near-surface levels, related to `rsur_sher`.
    imode_nsf_wind: int = 1
    #: Fraction of the additional surface shear forcing that is transmitted upwards.
    rsur_sher: float = 0.0
    #: Mode of estimating the Charnock parameter. Operationally 2 (DWD) or 3 (MCH).
    imode_charpar: options.CharnockParameterType = options.CharnockParameterType.WIND_DEPENDENT
    #: Charnock parameter.
    alpha0: float = 0.0123
    #: Upper limit of the velocity-dependent Charnock parameter.
    alpha0_max: float = 0.0335
    #: Additive ensemble perturbation of the Charnock parameter.
    alpha0_pert: float = 0.0
    #: Parameter scaling the molecular roughness of water waves.
    alpha1: float = 0.7500

    # 3. Stand-ins for external parameter fields (mo_turbdiff_config.f90:192-207)

    #: Surface area density of the roughness elements over land.
    c_lnd: float = 2.0
    #: Surface area density of the waves over sea.
    c_sea: float = 1.5
    #: Surface area density of the evaporative soil surface.
    c_soil: float = 1.0
    #: Surface area density of stems and branches over the plant-covered surface (deactivated).
    c_stm: float = 0.0
    #: Exponent yielding the effective surface area.
    e_surf: float = 1.0
    #: Apply a horizontally homogeneous roughness length (idealized testcases only). Read only
    #: by 'mo_nwp_phy_init.f90:1450,:1872', which seeds the 'gz0' field the granule receives;
    #: 'mo_nml_crosscheck.f90:426' only warns about it. No statement of the scheme tests it, so
    #: it cannot change what the granule computes.
    lconst_z0: bool = False
    #: The horizontally homogeneous roughness length used if `lconst_z0` [m].
    const_z0: float = 0.001

    # 4. Stand-ins for dynamical fields (mo_turbdiff_config.f90:209-213)

    #: Roughness length of a typical synoptic station [m].
    z0m_dia: float = 0.2
    #: Roughness length of sea ice [m].
    z0_ice: float = 0.001

    # 5. Turbulent diffusion parameters (mo_turbdiff_config.f90:215-254)

    #: Asymptotic maximal turbulent distance [m].
    tur_len: float = 500.0
    #: Effective global length scale of subscale surface patterns over land [m].
    pat_len: float = 100.0
    #: Minimal turbulent length scale [m].
    len_min: float = 1.0e-6
    #: Mode of calculating the minimal turbulent velocity scale in the surface layer.
    imode_vel_min: int = 2
    #: Minimal velocity scale [m/s].
    vel_min: float = 0.01
    #: Maximal velocity scale [m/s].
    vel_max: float = 30.0
    #: Von Karman constant.
    akt: float = 0.4
    #: Length-scale factor for the pressure destruction of turbulent scalar (heat) transport.
    a_heat: float = 0.74
    #: Length-scale factor for the pressure destruction of turbulent momentum transport.
    a_mom: float = 0.92
    #: Length-scale factor for the dissipation of turbulent temperature variance.
    d_heat: float = 10.1
    #: Length-scale factor for the dissipation of turbulent momentum variance.
    d_mom: float = 16.6
    #: Length-scale factor for the turbulent transport of TKE.
    c_diff: float = 0.20
    #: Length-scale factor for the stability correction of the integral turbulent length scale.
    a_stab: float = 0.00
    #: Length-scale factor for separate horizontal shear circulations, related to `ltkeshs`.
    #: Operationally 1.25 (RUC) or 2.0.
    a_hshr: float = 1.00
    #: Cloud cover at saturation.
    clc_diag: float = 0.5
    #: Critical value for the normalized supersaturation.
    q_crit: float = 1.6
    #: Shape factor applied to the saturation fraction at the moist correction (0 <= c_scld).
    c_scld: float = 1.0

    # 6. Switches (mo_turbdiff_config.f90:260-284)

    #: Consider mechanical SSO-wake production in the TKE equation. Both values are supported:
    #: '.FALSE.' switches the SSO source term off, which ICON expresses by forcing
    #: 'imode_tkesso = 0' (mo_turbdiff_nml.f90:158); `__post_init__` mirrors that assignment.
    ltkesso: bool = True
    #: Consider convective buoyancy production in the TKE equation.
    ltkecon: bool = False
    #: Consider separated horizontal shear production in the TKE equation.
    ltkeshs: bool = True
    #: Consider production by near-surface thermals in the TKE equation.
    ltkenst: bool = True
    #: Consider mechanical SSO-wake production of TKE for output. Gates only the write of
    #: 'tket_sso' (turb_diffusion.f90:956), which the ICON interfaces never pass, so 'loutmcsso'
    #: is false either way and the switch cannot change what the granule computes.
    loutsso: bool = True
    #: Consider separated horizontal shear production of TKE for output.
    loutshs: bool = True
    #: Consider production by near-surface thermals of TKE for output. Gates only the write of
    #: 'tket_nstc' (turb_diffusion.f90:954), never passed either, so 'loutthcrc' is false and the
    #: switch cannot change what the granule computes.
    loutnst: bool = False
    #: Consider buoyancy and shear TKE production for additional output. Gates the writes of
    #: 'tket_buoy', 'tket_fshr' and 'tket_gshr' (turb_diffusion.f90:1684), none of which is
    #: passed, and the fill of the scratch array 'ftm' (:1363). That fill is unobservable too:
    #: 'ftm' is read at :1677 only under 'rsur_sher > 0', and then only at the one level the
    #: 'ELSEIF' branch writes as well, and inside 'solve_turb_budgets'
    #: (turb_utilities.f90:1285,:1459) only under 'lssintact', which is false while
    #: 'imode_adshear' is frozen at 2. So it cannot change what the granule computes.
    loutbms: bool = False
    #: Consider minor turbulent sources in the enthalpy budget.
    ltmpcor: bool = False
    #: Consider fluctuations of the heat capacity of air.
    lcpfluc: bool = False
    #: Explicit warm-cloud correction of the implicitly calculated turbulent diffusion.
    lexpcor: bool = False
    #: Lower flux condition for the vertical diffusion calculation.
    lsflcnd: bool = True
    #: Consider non-turbulent fluxes related to near-surface circulations.
    lcirflx: bool = False
    #: Turbulent diffusion of cloud ice active. Read only by
    #: 'mo_nwp_turbdiff_interface.f90:353', which decides whether 'qi' joins the 'ptr(:)' list
    #: that reaches the granule as `TurbulenceInputState.tracers`. vertdiff diffuses the tracers
    #: it is handed and never tests the switch, so it cannot change what the granule computes --
    #: assembling the tracer list is the caller's job, the interface being out of scope (D1).
    ldiff_qi: bool = False
    #: Turbulent diffusion of snow active. As `ldiff_qi`, at
    #: 'mo_nwp_turbdiff_interface.f90:386', and equally unable to change what the granule
    #: computes.
    ldiff_qs: bool = False
    #: Free-slip lower boundary condition (idealized runs only).
    lfreeslip: bool = False
    #: Run with 3D turbulent diffusion. Not a namelist switch: a dummy argument of 'turbdiff'
    #: that ICON hardcodes to .FALSE. (mo_nwp_turbdiff_interface.f90:584).
    l3dturb: bool = False

    # 7. Selectors (mo_turbdiff_config.f90:290-389)

    #: Mode of the TKE equation in the transfer scheme.
    imode_tran: int = 0
    #: Mode of the TKE equation in the turbulence scheme.
    imode_turb: int = 1
    #: Mode of cloud representation in the transfer parameterization.
    icldm_tran: int = 2
    #: Mode of cloud representation in the turbulence parameterization. Operationally 1 (DWD
    #: global) or 2.
    icldm_turb: options.CloudRepresentationType = options.CloudRepresentationType.SUBGRID_SCALE
    #: Type of water cloud diagnosis within the turbulence scheme.
    itype_wcld: int = 2
    #: Type of mean shear production for TKE. Operationally 1, 2 or 3; the default 0, which
    #: 'mo_nml_crosscheck.f90:329' forces on runs without dynamics, is supported as well because
    #: it is the same code path with the horizontal shear correction left out.
    itype_sher: options.ShearProductionType = options.ShearProductionType.VERTICAL_ONLY
    #: Mode of calculating the stability function, related to `stbsecu`.
    imode_stbcalc: int = 1
    #: Type of the default condition at the lower boundary.
    ilow_def_cond: int = 2
    #: Mode of determining the length scale of the surface patterns used for the circulation
    #: term. Read only by 'mo_nwp_phy_init.f90:1570', which computes the 'l_pat' field the
    #: granule receives, so it cannot change what the granule computes.
    imode_pat_len: int = 2
    #: Mode of calculating the separated horizontal shear, related to `ltkeshs` and `a_hshr`.
    imode_shshear: int = 2
    #: Mode of calculating the SSO source term for TKE production, related to `ltkesso`.
    #: Operationally 1 (DWD global) or 2.
    imode_tkesso: options.SsoTkeProductionType = options.SsoTkeProductionType.ORIGINAL
    #: Mode of treating the aerodynamic surface smoothing by snow. Read only by
    #: 'mo_nwp_turbtrans_interface.f90:336', which smooths the 'gz0_t' and 'sai_t' fields before
    #: the call, so it cannot change what the granule computes.
    imode_snowsmot: int = 1
    #: Type of the 2m diagnostics for temperature and dewpoint, related to `z0m_dia`.
    itype_2m_diag: int = 1
    #: Mode of limiting the statistical saturation adjustment in 'turb_cloud'.
    imode_stadlim: int = 2
    #: Mode of configuring the transfer scheme.
    imode_trancnf: int = 2
    #: Mode of considering laminar diffusion within the surface layer.
    imode_lamdiff: int = 1
    #: Mode of adapting TKE and the turbulence model to the lower limits of the diffusion
    #: coefficients.
    imode_tkemini: int = 1
    #: Mode of adapting the diffusion coefficient at the lowest half levels.
    imode_suradap: int = 0
    #: Mode of implicit TKE diffusion, related to `c_diff`.
    imode_tkediff: int = 2
    #: Mode of considering additional shear by scale interaction.
    imode_adshear: int = 2

    def __post_init__(self) -> None:
        for name, option in (
            ("itype_sher", options.ShearProductionType),
            ("icldm_turb", options.CloudRepresentationType),
            ("imode_tkesso", options.SsoTkeProductionType),
            ("imode_charpar", options.CharnockParameterType),
        ):
            object.__setattr__(self, name, option(getattr(self, name)))

        # Consistency rule of 'mo_turbdiff_nml.f90:158'. ICON derives 'imode_tkesso' from
        # 'ltkesso' rather than rejecting the pair, because with the SSO source term off the mode
        # is meaningless: 'turb_diffusion.f90:1573' reads 'imode_tkesso' only inside
        # 'IF (ltkemcsso)'. Mirrored as the same assignment and not as a crosscheck, because
        # 'ltkesso = .FALSE.' is a configuration ICON runs -- 'exp.exclaim_nh_weisman_klemp_sb'
        # and 'checksuite.rcnl.dwd.de/exp.run_ICON-SCM_01_BOMEX.run' set it -- and refusing it
        # would refuse a live setup. Omitting an additional TKE source term is not a formulation
        # the port lacks: 'ltkecon' is frozen at '.FALSE.', so the same path is already the
        # ported one for the convective term.
        if not self.ltkesso:
            object.__setattr__(self, "imode_tkesso", options.SsoTkeProductionType.OFF)

        self._validate()

    @classmethod
    def from_fortran_dict(cls, nml: dict[str, Any], **overrides: Any) -> TurbulenceConfig:
        """Construct the configuration from the echoed ICON namelists.

        Reads the 'turbdiff_nml' group; parameters it does not carry keep their Fortran
        default. Every one of its 45 entries is a field here, so an entry that is not means the
        Fortran namelist has changed. That is raised rather than ignored: silently dropping a
        tuning parameter an operational setup relies on would change the answer without a trace.
        """
        group = nml[FORTRAN_NAMELIST_GROUP]
        known = {field.name for field in dataclasses.fields(cls)}
        settings = {}
        for name, value in group.items():
            if name not in known:
                raise ValueError(
                    f"Unknown entry '{name}' in namelist group '{FORTRAN_NAMELIST_GROUP}': "
                    f"'TurbulenceConfig' has no such parameter, so the Fortran namelist and this "
                    f"configuration have drifted apart."
                )
            settings[name] = _as_scalar(name, value)
        return cls(**{**settings, **overrides})

    def _validate(self) -> None:
        """Refuse the formulations that were not ported."""
        for switch in FROZEN_SWITCHES:
            switch.check(getattr(self, switch.name))

        _check_supported(
            "icldm_turb",
            self.icldm_turb,
            (
                options.CloudRepresentationType.GRID_SCALE,
                options.CloudRepresentationType.SUBGRID_SCALE,
            ),
            "grid-scale and sub-grid condensation",
        )
        # Only meaningful while the SSO source term is on; '__post_init__' has already forced
        # 'OFF' otherwise. So 'imode_tkesso = 0' is reachable only through that assignment, and
        # writing it next to 'ltkesso = .TRUE.' -- a pair the Fortran accepts and then silently
        # treats as no SSO term at all -- is refused instead of reproduced.
        if self.ltkesso:
            _check_supported(
                "imode_tkesso",
                self.imode_tkesso,
                (
                    options.SsoTkeProductionType.ORIGINAL,
                    options.SsoTkeProductionType.RICHARDSON_REDUCED,
                ),
                "the original and the Richardson-reduced SSO source term",
            )
        _check_supported(
            "imode_charpar",
            self.imode_charpar,
            (
                options.CharnockParameterType.WIND_DEPENDENT,
                options.CharnockParameterType.WIND_DEPENDENT_CYCLONE_REDUCED,
            ),
            "the wind-dependent Charnock parameter, with and without the cyclone reduction",
        )

        if not 0.0 <= self.frcsmot <= 1.0:
            raise ValueError(
                f"Invalid argument 'frcsmot': should be a smoothing fraction in [0, 1], "
                f"got {self.frcsmot}."
            )
        if self.a_hshr < 0.0:
            raise ValueError(
                f"Invalid argument 'a_hshr': should be a non-negative length-scale factor, "
                f"got {self.a_hshr}."
            )
        # Consistency check of mo_nml_crosscheck.f90:432, where ICON aborts on the same mismatch.
        if self.ltkeshs != (self.a_hshr > 0.0):
            raise ValueError(
                f"Invalid combination of 'ltkeshs' and 'a_hshr': separated horizontal shear "
                f"production is {'on' if self.ltkeshs else 'off'}, so 'a_hshr' must be "
                f"{'positive' if self.ltkeshs else 'zero'}, got {self.a_hshr}."
            )


@dataclasses.dataclass(frozen=True)
class TurbulenceParams:
    """Derived quantities of the turbulence model that depend only on the configuration.

    Port of the parameter block of 'turb_setup' (turb_utilities.f90:400-435), which derives the
    closure constants of the Raschendorfer stability functions from the length-scale factors.
    """

    config: dataclasses.InitVar[TurbulenceConfig]

    #: Adiabatic temperature gradient g/cp_d [K/m].
    tet_g: Final[float] = dataclasses.field(init=False)
    #: Cube root of the momentum dissipation length-scale factor.
    c_tke: Final[float] = dataclasses.field(init=False)
    #: Momentum closure constant of the stability functions.
    c_m: Final[float] = dataclasses.field(init=False)
    #: Scalar closure constant of the stability functions. May be treated as an independent
    #: parameter, but is zero in the Fortran.
    c_h: Final[float] = dataclasses.field(init=False)
    #: 1 - `c_m`.
    b_m: Final[float] = dataclasses.field(init=False)
    #: 1 - `c_h`.
    b_h: Final[float] = dataclasses.field(init=False)
    #: Reciprocal scalar pressure-destruction length-scale factor.
    d_1: Final[float] = dataclasses.field(init=False)
    #: Reciprocal momentum pressure-destruction length-scale factor.
    d_2: Final[float] = dataclasses.field(init=False)
    #: Auxiliary closure constant 9*a_heat.
    d_3: Final[float] = dataclasses.field(init=False)
    #: Auxiliary closure constant 6*a_mom.
    d_4: Final[float] = dataclasses.field(init=False)
    #: Auxiliary closure constant 3*(d_heat + `d_4`).
    d_5: Final[float] = dataclasses.field(init=False)
    #: Auxiliary closure constant `d_3` + 3*`d_4`.
    d_6: Final[float] = dataclasses.field(init=False)
    #: One minus the critical flux Richardson number, '1 - Rf_c'. The scheme tests against
    #: '1 - rim', not against 'rim' (turb_utilities.f90:535).
    rim: Final[float] = dataclasses.field(init=False)
    #: Auxiliary closure constant of the stability functions.
    a_3: Final[float] = dataclasses.field(init=False)
    #: Auxiliary closure constant of the stability functions.
    a_5: Final[float] = dataclasses.field(init=False)
    #: Auxiliary closure constant of the stability functions.
    a_6: Final[float] = dataclasses.field(init=False)
    #: Stability function for momentum at neutral stratification.
    sm_0: Final[float] = dataclasses.field(init=False)
    #: Stability function for scalars at neutral stratification.
    sh_0: Final[float] = dataclasses.field(init=False)
    #: cp_d/cp_v - 1, or zero when the heat-capacity fluctuations are switched off. ICON's
    #: rcpv is defined that way round (mo_physical_constants.f90:144), asymmetrically with
    #: rcpl below; cp_v/cp_d - 1 is a different constant, ICON's vtmpc2.
    tur_rcpv: Final[float] = dataclasses.field(init=False)
    #: cp_l/cp_d - 1, or zero when the heat-capacity fluctuations are switched off.
    tur_rcpl: Final[float] = dataclasses.field(init=False)

    def __post_init__(self, config: TurbulenceConfig) -> None:
        a_h, a_m = config.a_heat, config.a_mom
        d_h, d_m = config.d_heat, config.d_mom

        c_tke = d_m ** (1.0 / 3.0)
        c_m = 1.0 - 1.0 / (a_m * c_tke) - 6.0 * a_m / d_m
        c_h = 0.0
        d_1, d_2 = 1.0 / a_h, 1.0 / a_m
        d_3, d_4 = 9.0 * a_h, 6.0 * a_m
        d_5, d_6 = 3.0 * (d_h + d_4), d_3 + 3.0 * d_4
        b_m, b_h = 1.0 - c_m, 1.0 - c_h

        derived = {
            "tet_g": constants.GRAV_O_CPD,
            "c_tke": c_tke,
            "c_m": c_m,
            "c_h": c_h,
            "b_m": b_m,
            "b_h": b_h,
            "d_1": d_1,
            "d_2": d_2,
            "d_3": d_3,
            "d_4": d_4,
            "d_5": d_5,
            "d_6": d_6,
            "rim": 1.0 / (1.0 + (d_m - d_4) / d_5),
            "a_3": d_3 / (d_2 * d_m),
            "a_5": d_5 / (d_1 * d_m),
            "a_6": d_6 / (d_2 * d_m),
            "sm_0": (b_m - d_4 / d_m) / d_2,
            "sh_0": (b_h - d_4 / d_m) / d_1,
            "tur_rcpv": (constants.CPD / constants.CPV - 1.0) if config.lcpfluc else 0.0,
            "tur_rcpl": (constants.CPL / constants.CPD - 1.0) if config.lcpfluc else 0.0,
        }
        for name, value in derived.items():
            object.__setattr__(self, name, value)


def _as_scalar(name: str, value: Any) -> Any:
    """Unwrap the single-element arrays that the ICON namelist echo produces for scalars."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                f"Invalid namelist entry '{name}': expected a single value, got {len(value)}. "
                f"'turbdiff_nml' declares no domain-specific settings (mo_turbdiff_nml.f90:51)."
            )
        return value[0]
    return value


def _check_supported(name: str, value: int, supported: tuple[int, ...], meaning: str) -> None:
    """Raise unless an operationally varying switch is set to one of the ported formulations."""
    if value not in supported:
        values = " or ".join(str(int(option)) for option in supported)
        raise NotImplementedError(
            f"Only {name} = {values} ({meaning}) is implemented; got {int(value)}. "
            f"Set {name} to {values} or use the Fortran scheme."
        )
