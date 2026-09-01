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
'frcsmot', 'a_hshr', 'ltkesso' and 'ltkeshs' -- and are accepted over the range those setups
need. Which values occur was verified by grepping all 648 configurations under 'icon/run/',
not assumed.

WHAT 'TurbulenceConfig' ACCEPTS IS NOT WHAT THE GRANULE RUNS. `Turbulence` refuses four further
configurations at construction because the assembled stencils cannot express them, each being a
guarded Fortran block fused into an unguarded expression: see
`Turbulence._validate_the_configuration_the_stencils_can_express`. The widest of the four gaps is
'itype_sher', where the config accepts all four Fortran values and the granule runs only '2'; that
field's doc comment says so. Read the granule's refusal, not the config's acceptance, as the
contract.

Every other formulation switch is accounted for too, because what D6 rules out is silence, not
acceptance. Eight of them -- 'imode_pat_len', 'imode_snowsmot', 'lconst_z0', 'ldiff_qi',
'ldiff_qs', 'loutsso', 'loutnst' and 'loutbms' -- are accepted at any value because no statement
of the ported scheme reads them: each is either consumed by ICON code that is out of scope (port
spec D1: port the scheme, not the interface) and reaches the granule only through a field the
caller has already filled, or gates an output argument the ICON interfaces never pass. The doc
comment of each field names the line that consumes it.

`Turbulence` is the granule itself: it owns the working set and runs the ported stencils in the
Fortran's order. Both stages are here -- `run_turbdiff`, then `run_vertdiff`, composed by `run`,
which is the unit 'mo_nwp_turbdiff_interface.f90' substitutes. 'turbtran' is not; see the class
docstring for why that is a separate phase rather than a missing third line.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Final, NamedTuple

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
from gt4py.next import common as gtx_common

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence import (
    turbulence_options as options,
    turbulence_states as states,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.add_three_dimensional_shear_complements import (
    add_three_dimensional_shear_complements,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.add_virtual_diffusion_increment_to_tke_profile import (
    add_virtual_diffusion_increment_to_tke_profile,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.calc_impl_vert_diff import (
    calc_impl_vert_diff,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_and_apply_diffusion_tendency import (
    compute_and_apply_diffusion_tendency,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_and_apply_potential_temperature_diffusion_tendency import (
    compute_and_apply_potential_temperature_diffusion_tendency,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_conserved_variables_and_factors_at_main_levels import (
    compute_conserved_variables_and_factors_at_main_levels,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_conserved_variables_and_factors_at_the_surface import (
    compute_conserved_variables_and_factors_at_the_surface,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_current_potential_temperature_profile import (
    compute_current_potential_temperature_profile,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_current_profile import (
    compute_current_profile,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_diffusion_inversion_factor import (
    compute_diffusion_inversion_factor,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_effective_diffusion_coefficients import (
    compute_effective_diffusion_coefficients,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_explicit_tke_flux_density import (
    compute_explicit_tke_flux_density,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_half_level_interpolation_weight import (
    compute_half_level_interpolation_weight,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_horizontal_wind_including_the_zero_level import (
    compute_horizontal_wind_including_the_zero_level,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_implicit_part_of_tke_diffusion_momentum import (
    compute_implicit_part_of_tke_diffusion_momentum,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_inverted_diffusion_momentum import (
    compute_inverted_diffusion_momentum,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_layer_depth import (
    compute_layer_depth,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_stability_lengths_from_diffusion_coefficients import (
    compute_stability_lengths_from_diffusion_coefficients,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_surface_profile_value_from_flux_gradient import (
    compute_surface_profile_value_from_flux_gradient,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_tke_diffusion_right_hand_side import (
    compute_tke_diffusion_right_hand_side,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_tke_forcing_functions import (
    compute_tke_forcing_functions,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_total_mechanical_forcing import (
    compute_total_mechanical_forcing,
    compute_total_mechanical_forcing_without_richardson_reduction,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_turbulent_length_scale import (
    compute_turbulent_length_scale,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_turbulent_velocity_scale_tendency import (
    compute_turbulent_velocity_scale_tendency,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_vertical_gradients_of_conserved_variables import (
    compute_vertical_gradients_of_conserved_variables,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_virtual_tke_profile import (
    compute_virtual_tke_profile,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.interpolate_supersaturation_deviation_to_main_levels import (
    interpolate_supersaturation_deviation_to_main_levels,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.interpolate_variables_onto_half_levels import (
    interpolate_variables_onto_half_levels,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.invert_diffusion_momentum_at_the_surface_flux_level import (
    invert_diffusion_momentum_at_the_surface_flux_level,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.prep_impl_vert_diff import (
    prep_impl_vert_diff,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.prepare_the_tke_diffusion import (
    prepare_the_tke_diffusion,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.prepare_the_vertical_diffusion_matrix import (
    prepare_the_vertical_diffusion_matrix,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.set_turbulent_velocity_scale_at_model_top import (
    set_turbulent_velocity_scale_at_model_top,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.smooth_tke_forcing_vertically import (
    smooth_tke_forcing_vertically,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.solve_tke_diffusion_equation import (
    solve_tke_diffusion_equation,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.solve_turb_budgets import (
    solve_turb_budgets,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.subtract_implicit_part_of_tke_diffusion_momentum import (
    subtract_implicit_part_of_tke_diffusion_momentum,
)
from icon4py.model.common import (
    constants,
    dimension as dims,
    model_backends,
    model_options,
    type_alias as ta,
)
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, vertical as v_grid
from icon4py.model.common.utils import data_allocation as data_alloc


__all__ = [
    "FORTRAN_NAMELIST_GROUP",
    "FROZEN_SWITCHES",
    "DiffusedVariable",
    "FrozenSwitch",
    "Turbulence",
    "TurbulenceConfig",
    "TurbulenceParams",
]


#: Reciprocal of the gravitational acceleration [s2/m], ICON's 'edgrav'
#: (mo_physical_constants.f90). The turbulent length scale is seeded with 'gz0 * edgrav'.
_INVERSE_GRAVITY: Final[float] = 1.0 / constants.GRAV

#: Height above which the implicit weight of the TKE diffusion is ramped up towards 'impl_s'
#: (mo_nwp_phy_init.f90:781-795, :1541-1547) [m].
_IMPLICIT_WEIGHT_RAMP_HEIGHT: Final[float] = 1500.0


#: Group of the echoed ICON namelists that `TurbulenceConfig.from_fortran_dict` reads.
FORTRAN_NAMELIST_GROUP: Final[str] = "turbdiff_nml"


class DiffusedVariable(NamedTuple):
    """One of the five first-order variables 'vertdiff' diffuses, and the three fields it owns.

    'vertdiff' builds an array of these itself -- 'dvar(nmvar+ndtr)' of TYPE 'modvar'
    (turb_vertdiff.f90:451-460) -- with '%av' the profile, '%at' the tendency and '%sv' the
    surface value. This is that array, minus the two components the ported call site does not
    reach: '%kstart' is 1 for every variable ('kstart_cloud = 1' in this capture) and '%sv' is
    replaced by the two booleans, since the granule's surface values come from
    `TurbulenceSurfaceState` and the two flux-density variables are exactly those with
    'lsfli = .TRUE.'.

    Everything else a variable needs is workspace shared with the other four.

    Attributes:
        profile: 'dvar(n)%av', the variable on the main levels, in its own units. Read-only.
        tendency: 'dvar(n)%at', accumulated onto.
        right_hand_side: 'zvari(:,:,m)', which is one of the granule's five '_gradient_*'
            fields: the storage 'turbdiff' left its vertical gradients in and 'vertdiff'
            overwrites. See `Turbulence._diffuse_one_variable`.
        has_a_prescribed_surface_flux: 'lsfli(n)'. True for the temperature and the water
            vapour, whose lower boundary condition is the surface flux density 'turbtran'
            produced rather than a concentration.
        is_potential_temperature: 'n == tem'. The temperature is diffused as 'T/pi' and its
            tendency converted back, which is the one variable-specific arithmetic of the stage.
    """

    profile: gtx.Field
    tendency: gtx.Field
    right_hand_side: gtx.Field
    has_a_prescribed_surface_flux: bool = False
    is_potential_temperature: bool = False


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
    #: Vertical smoothing factor for the TKE forcing, in [0, 1]. Operationally 0.0 (MCH) or 0.2
    #: (DWD, and the Fortran default: 28 configurations under 'icon/run/' set it). Above zero it
    #: runs 'smooth_tke_forcing_vertically', which is the ONE stencil of this package with no
    #: ICON reference behind it -- the capture is a Swiss LAM domain where 'trop_mask' is
    #: identically zero, so no reference run made from it exercises the smoothing at any value of
    #: this parameter. See that stencil's module docstring for what stands in place of one.
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
    #: 'mo_nml_crosscheck.f90:329' forces on runs without dynamics, is accepted here as well
    #: because it is the same code path with the horizontal shear correction left out.
    #: THE GRANULE IS NARROWER THAN THIS FIELD. 'Turbulence' runs 'itype_sher = 2' and nothing
    #: else, refusing 0, 1 and 3 at construction
    #: ('_validate_the_configuration_the_stencils_can_express'), because
    #: 'add_three_dimensional_shear_complements' carries the 'IF (itype_sher == 2)' block of
    #: turb_diffusion.f90:1330 unguarded -- so the alternative is a wrong number, not a missing
    #: term. The reference capture exercises no other value, which is why: the other three have
    #: no serialized oracle. The compile-time static-parameter mechanism that would let one
    #: build serve several values ('program.compile(...)' / 'StencilTest.STATIC_PARAMS', as the
    #: dycore stencil tests use it) is used nowhere in this package.
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


class Turbulence:
    """The NWP 1D turbulence granule: the ported stencils, wired up and run in Fortran order.

    One instance owns one grid, one configuration and one working set. `run` executes the two
    stages 'mo_nwp_turbdiff_interface.f90' calls, in its order: `run_turbdiff`
    ('SUBROUTINE turbdiff', turb_diffusion.f90:281-2604) and then `run_vertdiff`
    ('SUBROUTINE vertdiff', turb_vertdiff.f90:117-935). `run_turbtran` does not exist yet --
    see "What is not here" below.

    NO 'lini' ANYWHERE. The Fortran threads 'iini'/'lini' through one entry point and branches
    on it inside, which is how Fortran avoids duplicating a ninety-argument list. It is not the
    same computation: the initialisation call of 'mo_nwp_phy_init.f90:1662' runs 'turbtran' on a
    two-level slab ('ke=2, ke1=3'). On the call site this granule replaces,
    'mo_nwp_turbdiff_interface.f90:582', 'iini' is the literal 0, so 'lini' is false by
    construction and `run_turbdiff` is unconditional. If the cold start is ever needed it is a
    method of its own, not a flag.

    FOUR SECTIONS ARE NOT CALLED because they write nothing in any configuration this granule
    accepts, which was measured over the whole 'nproma' slab at all four serialized dates and is
    asserted by 'test_turbdiff_section_{1c,2b,5,7}.py':

        1c  'IF (lini)' and 'IF (ltkeadapt)'   -- the cold start, and 'imode_tkemini == 2'
        2b  the vertically resolved canopy     -- 'c_big'/'c_sml'/'r_air' absent, 'kcm = ke+1'
        5   'IF (ltmpcor)' and 'IF (ldocirflx)' -- both frozen '.FALSE.' in `FROZEN_SWITCHES`
        7   'IF (ldocirflx)'                    -- 'lcirflx', frozen '.FALSE.'

    THE WORKING SET IS ALLOCATED ONCE, in `_allocate_local_fields`, which is where the Fortran's
    per-call '!$ACC DATA' scaffolding disappears; it was measured at 8.7% of the scheme's GPU
    cost (port spec 4.1).

    LOCAL FIELDS ARE NAMED AFTER THE FORTRAN STORAGE THEY ARE, not after the quantity they hold,
    because 'turbdiff' reuses six of its working arrays for unrelated quantities as it proceeds
    and that reuse is load-bearing: section 9) hands the solver a row of 'frh' that section 6)
    wrote and section 9) never touches, so a port that gave every quantity a fresh buffer would
    quietly change the answer. The argument name at each call site says which role is meant, the
    field name says which storage it lives in, and `_allocate_local_fields` carries the role
    table -- the same discipline the savepoint reader uses
    ('model/testing/serialbox.py::IconTurbdiffSectionSavepoint').

    THE TWO STAGES SHARE MORE THAN THEIR ARGUMENTS. 'vertdiff' reads the diffusion
    coefficients, the transfer velocities and the half-level density 'turbdiff' produced, it
    replaces the surface row of that density, and it overwrites 'zvari(:,:,1..5)' -- the five
    '_gradient_*' fields -- with its own right-hand sides. `run` is where that composition is
    stated; a stage test cannot see any of it.

    WHAT IS NOT HERE.

    * `run_turbtran` -- 'SUBROUTINE turbtran' (turb_transfer.f90), phase 3 of the plan. ICON
      calls it from a different interface, once per surface tile and before the surface
      scheme, so it is not a missing third line of `run`. Nothing is stubbed for it on
      purpose: an empty method that returns successfully is indistinguishable from a working
      one at the call site.
    """

    def __init__(
        self,
        *,
        grid: icon_grid.IconGrid,
        config: TurbulenceConfig,
        params: TurbulenceParams,
        vertical_grid: v_grid.VerticalGrid,
        metric_state: states.TurbulenceMetricState,
        backend: gtx_typing.Backend
        | model_backends.DeviceType
        | model_backends.BackendDescriptor
        | None,
    ) -> None:
        """Configure the granule and build its working set.

        Args:
            grid: The horizontal grid; supplies the cell count and the domain zones.
            config: The turbulence configuration, already validated by its own '__post_init__'.
            params: The closure constants derived from `config`.
            vertical_grid: The vertical grid; 'vct_a' is what the implicit weight of the TKE
                diffusion is built from, exactly as 'mo_nwp_phy_init.f90:1541-1547' builds it.
            metric_state: The vertical geometry and the horizontal masks. Held by reference
                and read at every call, which 'TurbulenceMetricState.dp0' relies on: it is the
                one member ICON recomputes each step.
            backend: The GT4Py backend, or a descriptor of one.
        """
        self._grid = grid
        self._vertical_grid = vertical_grid
        self._config = config
        self._params = params
        self._metric_state = metric_state
        self._backend = backend
        self._allocator = model_backends.get_allocator(backend)
        self._nlev = gtx.int32(grid.num_levels)

        self._validate_the_configuration_the_stencils_can_express()
        self._determine_derived_switches()
        self._determine_horizontal_domains()
        self._allocate_local_fields(self._allocator)
        self._setup_turbdiff_programs()
        self._setup_vertdiff_programs()

    # ------------------------------------------------------------------ configuration ---

    def _validate_the_configuration_the_stencils_can_express(self) -> None:
        """Refuse the configurations the ported stencils cannot represent.

        `TurbulenceConfig` states which formulations the PORT supports; this states which of
        those the assembled 'turbdiff' can actually run, which is narrower in four places. Each
        of the four is a stencil that fuses a guarded Fortran block into an unguarded expression
        -- correct only while the guard holds -- so the alternative is not a missing term but a
        wrong number.

        'imode_tkesso' used to be a fifth. It is not any more: mode 1 has its own program
        ('compute_total_mechanical_forcing_without_richardson_reduction') and
        '_setup_turbdiff_programs' selects it, so both values 'TurbulenceConfig' accepts run
        here. Mode 1 is validated against ICON only where the reduction factor is exactly 1;
        that stencil's module docstring says so.
        """
        if (
            self._config.itype_sher
            is not options.ShearProductionType.VERTICAL_AND_VERTICAL_VELOCITY
        ):
            raise NotImplementedError(
                f"Only itype_sher = 2 (mean shear including the vertical wind) is implemented in "
                f"'run_turbdiff'; got {int(self._config.itype_sher)}. "
                f"'add_three_dimensional_shear_complements' contains the "
                f"'IF (itype_sher == 2)' block of turb_diffusion.f90:1330 unguarded."
            )
        if not self._config.ltkeshs:
            raise NotImplementedError(
                "Only ltkeshs = True (separated horizontal shear production) is implemented in "
                "'run_turbdiff'; 'add_three_dimensional_shear_complements' adds that term "
                "without a guard (turb_diffusion.f90:1531-1536)."
            )
        if not self._config.ltkesso:
            raise NotImplementedError(
                "Only ltkesso = True (mechanical SSO-wake production) is implemented in "
                "'run_turbdiff'; 'add_three_dimensional_shear_complements' adds that term "
                "without a guard (turb_diffusion.f90:1572-1596)."
            )
        if self._config.c_diff <= 0.0:
            raise NotImplementedError(
                f"Only c_diff > 0 (the TKE diffusion runs) is implemented in 'run_turbdiff'; got "
                f"{self._config.c_diff}. At c_diff = 0 'turbdiff' skips sections 6) and 8) to "
                f"10) and zeroes 'tketens' instead (turb_diffusion.f90:2541), a branch the "
                f"reference capture does not exercise."
            )

    def _determine_derived_switches(self) -> None:
        """The host-side quantities 'turbdiff' derives from the namelist at :944-966.

        'lcircterm' decides a PAIR of programs, not one: section 8)'s virtual TKE profile and
        section 9)'s subtraction of it. Without the circulation term the Fortran aliases
        'cur_prof' onto 'sav_prof' (:2390) and neither program runs -- there is no virtual
        profile to build and none to remove. No stencil can enforce that, which is why the
        choice is made once here and reaches section 9) as the field
        '_current_virtual_profile' rather than as a second flag.
        """
        #: 'lcircterm' (:945, :959): the raw circulation term is an additional TKE source.
        self._circulation_term_is_active = self._config.pat_len > 0.0 and self._config.ltkenst

        #: 'c_diff_llim' (:2135-2139). Section 8) divides by the diffusion momentum section 6)
        #: builds with this factor, so the factor is limited away from zero while that division
        #: can happen; 'fakt' below undoes the limit again for the pure TKE diffusion.
        self._tke_diffusion_factor = (
            max(self._config.epsi, self._config.c_diff)
            if self._circulation_term_is_active
            else self._config.c_diff
        )
        #: 'fakt' (:2361), 'c_diff / c_diff_llim'.
        self._tke_diffusion_limit_correction = self._config.c_diff / self._tke_diffusion_factor

    def _determine_horizontal_domains(self) -> None:
        """The column window 'turbdiff' computes, as ICON's interface chooses it.

        'mo_nwp_turbdiff_interface.f90' calls with 'rl_start = grf_bdywidth_c + 1' and
        'rl_end = min_rlcell_int', which is the nudging zone through the last prognostic cell.
        Verified against the capture: the two indices are exactly 'ivstart' and 'ivend' of every
        turbulence savepoint (2424 and 10700 for exp.mch_icon-ch2_small).
        """
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))

    # --------------------------------------------------------------------- working set ---

    def _allocate_local_fields(self, allocator: gtx_typing.Allocator | None) -> None:
        """Allocate the whole working set once, and derive what depends only on the grid.

        ONCE, not per call: this is where the Fortran's per-call '!$ACC DATA' scaffolding
        disappears, measured at 8.7% of the scheme's GPU cost (port spec 4.1). The three
        methods below split it by stage and none of them is reachable from a 'run_*' method.
        'vertdiff's half of the set has its own role table, in
        `_allocate_the_vertdiff_working_set`; what follows is 'turbdiff's.

        THE ROLE TABLE. Six of these fields are one Fortran storage each and change meaning as
        the routine proceeds; the argument name at the call site says which role is meant.

            field         Fortran     roles, in order
            ------------  ----------  ----------------------------------------------------------
            _zaux_1       zaux(:,:,1) Exner factor on half levels (0)  -> 'upd_prof' (9)
            _zaux_2       zaux(:,:,2) 'r_cpd' (0)                      -> 'sav_prof' (6)
            _zaux_3       zaux(:,:,3) 'dQsat/dT' on half levels (0)    -> 'expl_mom' (6)
            _zaux_4       zaux(:,:,4) 'g_tet_l' (0)                    -> 'impl_mom' (9)
            _zaux_5       zaux(:,:,5) 'g_h2o' (0)                      -> 'invs_mom' (9)
            _frh          frh         thermal forcing (1b) -> CKE flux density (6)
                                      -> 'invs_fac' (9)
            _frm          frm         mechanical forcing (1b, 2a)
                                      -> CKE flux at main levels (6)
            _hlp          hlp         interpolation weight (0) -> inverse layer depth (1a)
                                      -> SSO wake production (2a) -> virtual TKE profile (8)
            _dicke        dicke       layer depth (0) -> discretisation momentum (1a)
            _len_scale    len_scale   turbulent master length scale (0)
                                      -> right-hand side of the TKE solve (9)
            _rcld         rcld        cloud cover on half levels (0) -> SDSS (3)

        WHERE THE PORT NEEDS TWO FIELDS FOR ONE FORTRAN STORAGE. Four of the Fortran's
        in-place rewrites read a NEIGHBOURING level of the array they write, and a GT4Py program
        computes its whole domain from the values it is given, so in place is not the same
        computation:

        * 'zvari(:,:,1..5)' -- section 1a) replaces the quasi-conserved variables by their
          vertical differences and reads level 'k-1' doing it. Hence '_conserved_*' and
          '_gradient_*'.
        * 'zvari(:,:,0)' -- section 3)'s circulation acceleration is pointwise, but keeping it
          apart from the half-level pressure it overwrites costs one field and removes the
          ordering constraint. Hence '_half_level_pressure' and '_circulation_acceleration'.
        * the four quantities 'bound_level_interp' interpolates in place at :1066-1073, which
          read main levels 'k-1' and 'k'. Hence '_*_on_main_levels' beside '_zaux_3', '_zaux_4',
          '_zaux_5' and '_rcld'.
        * 'frm' and 'frh' under the optional vertical smoothing of section 2c), which reads both
          neighbours of every row it writes -- the Fortran's own 'sav_tend' is what makes its
          in-place sweep legitimate. Hence '_smoothed_mechanical_forcing' and
          '_smoothed_thermal_forcing' beside '_frm' and '_frh'.

        The one in-place rewrite the port KEEPS is section 9)'s
        'subtract_implicit_part_of_tke_diffusion_momentum', which is pointwise and whose row
        'nlev' must survive from section 6): see `run_turbdiff`.
        """
        self._allocate_the_turbdiff_working_set(allocator)
        self._allocate_the_vertdiff_working_set(allocator)
        self._derive_what_depends_only_on_the_grid(allocator)

    def _field_shapes(
        self, allocator: gtx_typing.Allocator | None
    ) -> tuple[Callable[[], gtx.Field], Callable[[], gtx.Field], Callable[[], gtx.Field]]:
        """The three shapes every field of the working set has.

        A method rather than three closures inside one allocation routine because the working
        set is allocated in three parts and each of them builds fields of more than one shape.
        """

        def half() -> gtx.Field:
            """A field on the 'nlev + 1' half levels."""
            return data_alloc.zero_field(
                self._grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
            )

        def main() -> gtx.Field:
            """A field on the 'nlev' main levels."""
            return data_alloc.zero_field(self._grid, dims.CellDim, dims.KDim, allocator=allocator)

        def surface() -> gtx.Field:
            """One value per column."""
            return data_alloc.zero_field(self._grid, dims.CellDim, allocator=allocator)

        return half, main, surface

    def _allocate_the_turbdiff_working_set(self, allocator: gtx_typing.Allocator | None) -> None:
        """What 'run_turbdiff' computes into; the role table is in `_allocate_local_fields`."""
        half, main, surface = self._field_shapes(allocator)

        # -- the quasi-conserved variables and their vertical gradients, 'zvari(:,:,1..5)'
        self._conserved_zonal_wind = half()
        self._conserved_meridional_wind = half()
        self._conserved_liquid_water_potential_temperature = half()
        self._conserved_total_water = half()
        self._conserved_liquid_water = half()
        self._gradient_zonal_wind = half()
        self._gradient_meridional_wind = half()
        self._gradient_liquid_water_potential_temperature = half()
        self._gradient_total_water = half()
        self._gradient_liquid_water = half()

        # -- 'zvari(:,:,0)': half-level pressure, then the circulation acceleration
        self._half_level_pressure = half()
        self._circulation_acceleration = half()

        # -- the thermodynamic factors at MAIN levels, which 'bound_level_interp' consumes.
        # Allocated over the half levels because that is the shape of the Fortran storage they
        # share with their interpolated selves; only rows 0..nlev-1 are ever written or read.
        self._cloud_cover_on_main_levels = half()
        self._dqsat_dt_on_main_levels = half()
        self._buoyancy_factor_tet_l_on_main_levels = half()
        self._buoyancy_factor_h2o_g_on_main_levels = half()

        # -- the eleven reused storages (see the role table above)
        self._zaux_1 = half()
        self._zaux_2 = half()
        self._zaux_3 = half()
        self._zaux_4 = half()
        self._zaux_5 = half()
        self._frh = half()
        self._frm = half()
        self._hlp = half()
        self._dicke = half()
        self._len_scale = half()
        self._rcld = half()

        # -- single-role intermediates
        #: 'ftm': the mechanical forcing by the mean flow alone. The Fortran keeps it in 'frm'
        #: and only saves it under 'lssintact .OR. loutbms'; the port needs it as a field
        #: because 'xri' is formed from it before the non-turbulent terms are added.
        self._mean_shear_forcing = half()
        #: 'xri' = 1/Ri**(2/3), main levels.
        self._inverse_richardson_number_factor = main()
        #: 'hor_scale', main levels.
        self._effective_horizontal_shear_length_scale = main()
        #: 'layr', one value per column.
        self._uncorrected_horizontal_shear_length_scale = surface()
        #: 'lays(:,1)' and 'lays(:,2)', the two surface transfer ratios.
        self._surface_transfer_ratio_for_momentum = surface()
        self._surface_transfer_ratio_for_scalars = surface()
        #: 'frm' and 'frh' after the optional vertical smoothing of section 2c). The Fortran
        #: smooths in place; 'smooth_tke_forcing_vertically' reads both neighbours of every row
        #: it writes, so the port needs a second field per profile. Sections 3) and 4) read
        #: these, sections 6) and 9) go on reusing '_frm'/'_frh' for their unrelated roles --
        #: exactly as the Fortran reuses the two storages.
        self._smoothed_mechanical_forcing, self._smoothed_thermal_forcing = half(), half()
        #: The stability lengths section 2c) makes out of the diffusion coefficients and
        #: section 3) replaces. In the Fortran both live in 'tkvm'/'tkvh'.
        self._stability_length_for_momentum = half()
        self._stability_length_for_scalars = half()
        self._updated_stability_length_for_momentum = half()
        self._updated_stability_length_for_scalars = half()
        #: The diffusion coefficients as section 3) leaves them, before section 4)'s lower
        #: limits. Also 'tkvm'/'tkvh' in the Fortran.
        self._diffusion_coefficient_for_momentum = half()
        self._diffusion_coefficient_for_scalars = half()
        #: The explicit TKE flux density of section 9). The Fortran writes it into the
        #: 'len_scale' storage and immediately overwrites it with the right-hand side, so it
        #: reaches no savepoint and needs a field of its own here.
        self._explicit_tke_flux_density = half()

        # -- absolute-level slices, refilled by 'run_turbdiff' at the point in the sequence
        # where the Fortran reads the row: see the '_extract_level' call sites there.
        self._diffusion_coefficient_for_momentum_at_the_surface = surface()
        self._diffusion_coefficient_for_scalars_at_the_surface = surface()
        self._liquid_water_potential_temperature_above_the_surface = surface()
        self._total_water_above_the_surface = surface()

    def _allocate_the_vertdiff_working_set(self, allocator: gtx_typing.Allocator | None) -> None:
        """What 'run_vertdiff' computes into, named after the Fortran storage each field is.

        'vertdiff' declares its own '!$ACC CREATE' locals (turb_vertdiff.f90:363-380) and hands
        them to 'vert_grad_diff' under the names below. THEY ARE NOT 'turbdiff's, even though
        six of them are spelled the same: 'zaux', 'frh', 'frm', 'dicke', 'hlp' and 'len_scale'
        are arrays of a different subroutine, and the two exit hooks serialize them separately.
        So they are separate fields here too, and after `run` either stage can still be asked
        what it left behind.

            field                          Fortran                 what it holds
            -----------------------------  ----------------------  --------------------------
            _surface_exner_factor          eprs(:,ke1:ke1)         '(p_s/p0)**(R_d/c_pd)'
            _discretisation_momentum       zaux(:,:,1) disc_mom    'rho*dz/dt'
            _diffusion_momentum            zaux(:,:,2) expl_mom    'rho*K/dz', then its
                                                                   explicit part alone
            _implicit_diffusion_momentum   zaux(:,:,3) impl_mom    its implicit part
            _inverted_diffusion_momentum   zaux(:,:,4) invs_mom    the LU diagonal
            _diffusion_depth               zaux(:,:,5) diff_dep    the layer separation
            _inversion_factor              frh         invs_fac    the LU sub-diagonal
            _current_profile               hlp         cur_prof    the profile being diffused
            _diffusion_increment           dicke       dif_tend    its diffusion tendency

        THREE OF THESE ARE REUSED WITHIN THE STAGE, and the reuse is ICON's rather than a
        saving of the port's:

        * '_implicit_diffusion_momentum' is ONE storage for both variable types. The scalar
          type's surface-FLUX condition writes one row less than the momentum type's
          surface-CONCENTRATION condition, so its surface row still holds the momentum type's
          value when 'vertdiff' returns -- which is what the exit savepoint has, and what a
          port with one buffer per type would get wrong.
        * '_diffusion_momentum' and '_diffusion_depth' are rewritten per variable type and only
          the second type's survives, again as in ICON.

        WHERE THE PORT NEEDS A FIELD ICON DOES NOT HAVE:

        * 'eff_flux' -- 'zvari(:,:,m)' -- is the explicit flux density and then the right-hand
          side built from it, and 'calc_impl_vert_diff:2988' reads flux level 'k+1' while
          writing row 'k'. In place is therefore not the same computation, and the explicit
          flux needs a field beside the 'zvari' one.
        * 'upd_prof' IS the 'dicke' storage: 'vert_grad_diff:2663' turns the solved profile into
          a tendency in place. That statement is pointwise, so aliasing would be legitimate;
          they are kept apart because the solved profile is the only quantity of the solve that
          a test can compare, and nothing else preserves it.

        THE FIVE RIGHT-HAND SIDES ARE NOT ALLOCATED HERE. They live in 'zvari(:,:,1..5)', which
        the ICON interface passes to BOTH stages, so they are the five '_gradient_*' fields
        'turbdiff' wrote -- see `run_vertdiff`.
        """
        half, _main, _surface = self._field_shapes(allocator)

        self._surface_exner_factor = half()
        self._discretisation_momentum = half()
        self._diffusion_momentum = half()
        self._implicit_diffusion_momentum = half()
        self._inverted_diffusion_momentum = half()
        self._diffusion_depth = half()
        self._inversion_factor = half()
        self._current_profile = half()
        self._diffusion_increment = half()
        self._explicit_flux_density = half()
        self._updated_profile = half()

    def _derive_what_depends_only_on_the_grid(self, allocator: gtx_typing.Allocator | None) -> None:
        """The part of the working set that is fixed once the grid and the configuration are."""
        _half, _main, surface = self._field_shapes(allocator)

        # -- what depends only on the grid and the configuration
        #
        # 'hhl(:,ke1)' -- the surface height -- AS A ONE-DIMENSIONAL VIEW OF THE CALLER'S ARRAY.
        # GT4Py cannot slice a vertical level out of a field: an offset is always relative to the
        # row being computed, so a value read at a fixed 'k' has to reach a stencil as a cell
        # field somebody prepared. 'gtx_common._field' is how the dycore wraps such a row back up
        # as a field (solve_nonhydro.py:885-899), and it is GT4Py INTERNAL API -- leading
        # underscore and all. If it is ever renamed, this granule and the dycore break together.
        #
        # A VIEW AND NOT A COPY, which is what makes this correct when the caller's field is
        # wider than the grid. Through the ICON bindings 'hhl' is a '(:,:,jb)' slice of
        # '(nproma, nlev+1, nblks_c)', so its first extent is 'nproma', while a field the granule
        # allocates is 'grid.num_cells' wide -- and no ICON configuration makes the two equal,
        # because 'icon4py_init' requires 'nproma >= n_patch_edges' and edges always outnumber
        # cells. Copying the row into a grid-sized field is what raised
        # "operands could not be broadcast together with shapes (nproma,) (num_cells,)" the first
        # time this granule ran inside ICON.
        #
        # The slice stays a view: py2fgen builds its arrays 'order="F"'
        # ('py2fgen/_conversion.py:51'), so fixing the TRAILING index gives a stride-1 vector and
        # 'ndarray[:num_cells, nlev]' is a contiguous prefix of it. Nothing is materialised, so
        # the row cannot go stale and costs no kernel -- which a 'concat_where' over a one-level
        # domain would.
        #
        # The domain has to be given because a bare array carries none.
        self._surface_height = gtx_common._field(
            self._metric_state.hhl.ndarray[: self._grid.num_cells, self._nlev],
            domain={dims.CellDim: (0, self._grid.num_cells)},
        )
        self._surface_liquid_water = surface()  # 'liqs', zero at 'ilow_def_cond == 2'
        self._horizontal_length_scale_limit = surface()
        self._minimal_tke_forcing = surface()
        self._compute_the_turb_setup_scales()
        self._implicit_weight = self._build_the_implicit_weight(allocator)

        #: Which profile section 9) diffuses. With the circulation term it is section 8)'s
        #: virtual profile in the 'hlp' scratch; without it, 'turb_diffusion.f90:2390' points
        #: 'cur_prof' at 'sav_prof' itself and section 8) does not run.
        self._current_virtual_profile = (
            self._hlp if self._circulation_term_is_active else self._zaux_2
        )

    def _compute_the_turb_setup_scales(self) -> None:
        """'l_scal' and 'fc_min' of SUB 'turb_setup' (turb_utilities.f90:335-338).

            l_scal(i) = MIN( z1d2*l_hori(i), tur_len )
            fc_min(i) = (vel_min/MAX( l_hori(i), tur_len ))**2

        'turb_setup' is not one of the numbered sections and is not ported as stencils, but
        'turbdiff' reads both of these and they depend on nothing that changes with time --
        'l_hori' is a metric field and the two parameters are namelist constants. So they are
        derived once, here, in the array namespace of the backend rather than in a program.

        The square is written as a product, as the Fortran's integer-exponent '**2' is: see the
        package README on why 'x**2' is not 'x*x' on the GPU.
        """
        xp = data_alloc.import_array_ns(self._allocator)
        # Clamped to 'grid.num_cells', NOT to 'l_hori's own length: 'l_hori' is caller-supplied
        # and is 'nproma' wide when the caller is ICON, whereas the two fields written from it
        # here are the granule's own and are 'grid.num_cells' wide. 'num_cells' is the only
        # length the three agree on. What lies beyond it is ICON's block padding -- no grid point
        # is there, ICON leaves it undefined, and the column window the scheme computes
        # ('_determine_horizontal_domains') never reaches it. Unlike the surface height above,
        # these two are genuinely new data and not a row of somebody else's array, so they are
        # computed and stored rather than viewed.
        num_cells = self._grid.num_cells
        l_hori = self._metric_state.l_hori.ndarray[:num_cells]
        self._horizontal_length_scale_limit.ndarray[...] = xp.minimum(
            0.5 * l_hori, self._config.tur_len
        )
        velocity_scale = self._config.vel_min / xp.maximum(l_hori, self._config.tur_len)
        self._minimal_tke_forcing.ndarray[...] = velocity_scale * velocity_scale

    def _build_the_implicit_weight(self, allocator: gtx_typing.Allocator | None) -> gtx.Field:
        """'tdc%impl_weight', the implicit weight of each flux level of the TKE diffusion.

        Not a field of the turbulence scheme and not serialized: ICON fills it once at model
        initialisation (mo_nwp_phy_init.f90:1541-1547) and never changes it, "using an over
        implicit value (impl_s) near surface, reduced to in general slightly off-centered value
        (impl_t) in about 1500 m height". Reproduced here from the same reference vertical
        coordinate 'vct_a' that ICON takes 'k1500m' from (:781-795), on the host, because it is
        a one-off over 'nlev' entries.
        """
        vct_a = self._vertical_grid.vct_a.asnumpy()
        nlev = int(self._nlev)
        ramp_level = 1
        for level in range(nlev, 0, -1):  # Fortran 'DO jk = nlev,1,-1', one-based
            if (
                vct_a[level - 1] >= _IMPLICIT_WEIGHT_RAMP_HEIGHT
                and vct_a[level] < _IMPLICIT_WEIGHT_RAMP_HEIGHT
            ):
                ramp_level = level
        weight = [self._config.impl_t] * (nlev + 1)
        for level in range(ramp_level + 1, nlev + 1):
            weight[level - 1] = self._config.impl_t + (
                self._config.impl_s - self._config.impl_t
            ) * (level - ramp_level) / float(nlev - ramp_level)
        weight[nlev] = self._config.impl_s
        array_ns = data_alloc.import_array_ns(allocator)
        return gtx.as_field(
            (dims.KDim,), array_ns.asarray(weight, dtype=ta.wpfloat), allocator=allocator
        )

    # ------------------------------------------------------------------ program setup ---

    def _program(
        self,
        program: gtx_typing.Program,
        *,
        constant_args: dict | None = None,
        levels: tuple[int, int] | None = None,
        shifted: bool = False,
    ) -> Callable[..., None]:
        """Bind one stencil to this granule's domain, its constants and its offset provider.

        'setup_program' turns every scalar in 'constant_args' into a compile-time constant of
        the generated code. THE ROW INDICES OF THE BOUNDARY-ROW PROGRAMS ARE DELIBERATELY NOT
        AMONG THEM -- 'nlev' and 'uppermost_diffused_level' are passed at call time -- because
        making one of those static while the domain bounds are static too miscompiles on
        'dace_cpu': the concat_where replacement pass asks a one-dimensional producer for its
        vertical offset and dies with

            gt4py/next/program_processors/runners/dace/transformations/concat_where_mapper.py
            :1043 in _replace_single_read: prod_offset = prod_offsets[dim][0]
            dace/subsets.py:755 in __getitem__: IndexError: list index out of range

        on the cell field that 'compute_vertical_gradients_of_conserved_variables' selects for
        its surface row. Measured 2026-08-28: static 'nlev' with runtime domain bounds compiles,
        runtime 'nlev' with static bounds compiles, both static does not. The three programs
        whose boundary branch reads a cell field are the ones exposed, but the rule is applied
        to all six that take a row index, because which branch is one-dimensional is a property
        of a stencil that may change.

        Args:
            program: The GT4Py program.
            constant_args: Fields and scalars that do not change between calls; the scalars are
                inlined into the generated code by 'setup_program'.
            levels: The half-open vertical domain, or None for a program with no vertical axis.
            shifted: Whether the program reads a neighbouring vertical level.
        """
        return model_options.setup_program(
            program=program,
            backend=self._backend,
            constant_args=constant_args,
            horizontal_sizes={
                "horizontal_start": self._start_cell,
                "horizontal_end": self._end_cell,
            },
            vertical_sizes=None
            if levels is None
            else {
                "vertical_start": gtx.int32(levels[0]),
                "vertical_end": gtx.int32(levels[1]),
            },
            offset_provider={dims.Koff.value: dims.KDim} if shifted else {},
        )

    def _setup_turbdiff_programs(self) -> None:
        """Compile every stencil of 'turbdiff' with its domain and its constants bound.

        The vertical domain of each program is stated HERE and nowhere else, so that the
        translation of Fortran one-based inclusive 'DO k = a, b' into a zero-based half-open
        GT4Py domain happens once per program. The comment on each line is the Fortran loop.
        """
        nlev = int(self._nlev)
        config, params = self._config, self._params
        metric = self._metric_state

        cloud_diagnosis = {
            "cloud_cover_shape_factor": config.c_scld,
            "critical_normalized_supersaturation": config.q_crit,
            "cloud_cover_at_saturation": config.clc_diag,
            "relative_accuracy_limit": config.epsi,
        }

        # -- section 0) conserved variables, cloud cover, thermodynamic factors, length scales
        self._compute_conserved_variables_and_factors_at_main_levels = self._program(
            compute_conserved_variables_and_factors_at_main_levels,
            constant_args=cloud_diagnosis,
            levels=(0, nlev),  # 'k_st=1, k_en=ke'
        )
        self._compute_conserved_variables_and_factors_at_the_surface = self._program(
            compute_conserved_variables_and_factors_at_the_surface,
            constant_args=cloud_diagnosis,
            levels=(nlev, nlev + 1),  # 'k_st=ke1, k_en=ke1'
        )
        self._compute_layer_depth = self._program(
            compute_layer_depth,
            constant_args={"half_level_height": metric.hhl},
            levels=(0, nlev),  # 'DO k=1,ke'
            shifted=True,
        )
        self._compute_horizontal_wind_including_the_zero_level = self._program(
            compute_horizontal_wind_including_the_zero_level,
            levels=(0, nlev + 1),  # 'DO k=1,ke' plus the separate 'ke1' row
            shifted=True,
        )
        self._compute_half_level_interpolation_weight = self._program(
            compute_half_level_interpolation_weight,
            constant_args={"layer_pressure_thickness": metric.dp0},
            levels=(1, nlev),  # 'bound_level_interp(..., k_st=2, k_en=ke)'
            shifted=True,
        )
        self._interpolate_variables_onto_half_levels = self._program(
            interpolate_variables_onto_half_levels,
            levels=(1, nlev),  # 'bound_level_interp(..., k_st=2, k_en=ke)'
            shifted=True,
        )
        self._compute_turbulent_length_scale = self._program(
            compute_turbulent_length_scale,
            constant_args={
                "horizontal_length_scale_limit": self._horizontal_length_scale_limit,
                "von_karman_constant": config.akt,
                "minimal_length_scale": config.len_min,
            },
            levels=(0, nlev + 1),  # 'DO k=kcm-1,1,-1' then 'DO k=ke1,1,-1'
            shifted=True,
        )

        # -- section 1a) vertical gradients
        # One program, three statements: 'lays' with no vertical axis, 'hlp'/'dicke' over
        # 'DO k=ke,2,-1', and the five gradients over that range plus the separate 'ke1' row.
        # The bound below is the gradients'; the 'hlp'/'dicke' statement stops one row earlier.
        self._compute_vertical_gradients_of_conserved_variables = self._program(
            compute_vertical_gradients_of_conserved_variables,
            constant_args={"hhl": metric.hhl},
            levels=(1, nlev + 1),  # 'DO k=ke,2,-1' plus the separate 'ke1' row
            shifted=True,
        )

        # -- section 1b) the two basic TKE forcing functions
        # One program, two statements: 'frh' over the whole range bound here, 'frm' over one row
        # less. The 'kem = ke' bound is inside the stencil now, next to the Fortran that sets it.
        self._compute_tke_forcing_functions = self._program(
            compute_tke_forcing_functions,
            constant_args={"min_forcing": self._minimal_tke_forcing},
            levels=(1, nlev + 1),  # 'DO k=2,ke1'; the shear statement stops at 'kem = ke'
        )

        # -- section 2a) the three-dimensional shear complements, one program of six statements
        # and then one of two. Five statements run over 'DO k=2,kem'; the SSO wake production is
        # on MAIN levels and starts one row higher, which the stencil writes as
        # 'vertical_start - 1'. Nothing here reads a neighbouring level, so no offset provider.
        self._add_three_dimensional_shear_complements = self._program(
            add_three_dimensional_shear_complements,
            constant_args={
                "min_forcing": self._minimal_tke_forcing,
                "horizontal_mesh_size": metric.l_hori,
                "horizontal_shear_length_factor": config.a_hshr,
                "karman_constant": config.akt,
                "half_level_height": metric.hhl,
                "surface_height": self._surface_height,
                "neutral_momentum_stability_function": params.sm_0,
            },
            levels=(1, nlev),  # 'DO k=2,kem' with 'kem = ke'
        )
        # 'imode_tkesso' picks the program that finishes 'frm', and this is the only place the
        # mode is read. Mode 1 (turb_diffusion.f90:1587) adds the SSO source without the
        # Richardson reduction and never touches 'xri', so its program does not take the field;
        # binding 'xri' here rather than at the call site keeps 'run_turbdiff' free of the
        # branch. 'setup_program' inlines only SCALARS as compile-time constants -- a field in
        # 'constant_args' is bound by identity, and this one is allocated once and rewritten in
        # place at every call, so binding it is the same as passing it.
        #
        # THE MERGE PLAN WANTED THIS STATEMENT INSIDE THE PROGRAM ABOVE, switched off by an
        # empty vertical domain when mode 1 supplies 'frm' instead. It cannot be: the statement
        # reads 'hlp' and 'dp0' at 'Koff[-1]', and an empty vertical domain is a no-op only for
        # a POINTWISE statement -- on 'embedded' a shifted one raises 'IndexOutOfBounds',
        # because gt4py normalises the empty range to '(0, 0)' and bounds-checks it against the
        # shifted operand's domain, which starts at row 1. Measured 2026-09-01; the merged
        # stencil's docstring carries the table. So section 2a) is three programs.
        richardson_reduction = (
            {}
            if config.imode_tkesso is options.SsoTkeProductionType.ORIGINAL
            else {"inverse_richardson_number_factor": self._inverse_richardson_number_factor}
        )
        self._compute_total_mechanical_forcing = self._program(
            compute_total_mechanical_forcing_without_richardson_reduction
            if config.imode_tkesso is options.SsoTkeProductionType.ORIGINAL
            else compute_total_mechanical_forcing,
            constant_args={"layer_pressure_thickness": metric.dp0, **richardson_reduction},
            levels=(1, nlev),  # 'DO k=2,kem'
            shifted=True,
        )

        # -- section 2c) final preparations
        #: 'vert_smooth' (turb_utilities.f90:3098), compiled only when it will run: 'setup_program'
        #: compiles eagerly, and at 'frcsmot = 0' -- what the three MCH production experiments set
        #: -- the routine is the identity and 'run_turbdiff' skips it.
        self._smooth_tke_forcing_vertically = (
            self._program(
                smooth_tke_forcing_vertically,
                constant_args={
                    "smoothing_mask": metric.trop_mask,
                    "smoothing_weight": config.frcsmot,
                },
                # The whole column: 'vert_smooth' writes 'k_tp+1 .. k_sf-1' and the port copies
                # the two rows outside that through, because it cannot smooth in place.
                levels=(0, nlev + 1),
                shifted=True,
            )
            if config.frcsmot > 0.0
            else None
        )
        self._compute_stability_lengths_from_diffusion_coefficients = self._program(
            compute_stability_lengths_from_diffusion_coefficients,
            levels=(1, nlev),  # 'DO k=2,ke'
        )

        # -- section 3) the turbulent budgets ('solve_turb_budgets'), one program of six
        # statements. The bound pair is the section's own 'DO k=k_st,k_en'; the model-top row
        # is 'vertical_start - 1' and the circulation acceleration runs to 'vertical_end + 1'.
        self._solve_turb_budgets = self._program(
            solve_turb_budgets,
            constant_args={
                "horizontal_grid_scale": metric.l_hori,
                "a_h": config.a_heat,
                "a_m": config.a_mom,
                "b_h": params.b_h,
                "b_m": params.b_m,
                "d_h": config.d_heat,
                "d_m": config.d_mom,
                "d_1": params.d_1,
                "d_2": params.d_2,
                "d_3": params.d_3,
                "d_4": params.d_4,
                "d_5": params.d_5,
                "d_6": params.d_6,
                "rim": params.rim,
                "frcsecu": config.frcsecu,
                "stbsecu": config.stbsecu,
                "tkesecu": config.tkesecu,
                "tkesmot": config.tkesmot,
                "vel_min": config.vel_min,
                "gravitational_acceleration": constants.GRAV,
                "molecular_diffusivity_for_scalars": constants.MOLECULAR_DIFFUSIVITY_FOR_SCALARS,
            },
            levels=(1, nlev),  # 'DO k=k_st,k_en' with 'k_st=2, k_en=kem'
        )
        # NOT a statement of 'solve_turb_budgets', although it is one of the same section's
        # Fortran statements: it would read through 'Koff' the very parameter it writes, and
        # DaCe drops such a statement silently. Its module docstring carries the measurement.
        self._set_turbulent_velocity_scale_at_model_top = self._program(
            set_turbulent_velocity_scale_at_model_top,
            levels=(0, 1),  # 'tke(:,1) = tke(:,2)'
            shifted=True,
        )

        # -- section 4) lower limits of the diffusion coefficients
        self._compute_effective_diffusion_coefficients = self._program(
            compute_effective_diffusion_coefficients,
            constant_args={
                "height_of_half_levels": metric.hhl,
                "surface_height": self._surface_height,
                "tropics_mask": metric.trop_mask,
                "inner_tropics_mask": metric.innertrop_mask,
                "minimum_coefficient_for_momentum": max(
                    constants.MOLECULAR_DIFFUSIVITY_FOR_MOMENTUM, config.tkmmin
                ),
                "minimum_coefficient_for_scalars": max(
                    constants.MOLECULAR_DIFFUSIVITY_FOR_SCALARS, config.tkhmin
                ),
                "stratospheric_minimum_for_momentum": config.tkmmin_strat,
                "stratospheric_minimum_for_scalars": config.tkhmin_strat,
            },
            levels=(1, nlev),  # 'DO k=2,ke'
        )

        # -- section 6) preparations for the TKE diffusion
        # One program, four statements: 'sav_prof' and 'frh' over 'DO k=2,ke1', 'expl_mom' and
        # 'frm' over 'DO k=3,ke1'. The bound below is the half levels'; the two flux-level
        # statements start at 'vertical_start + 1'.
        self._prepare_the_tke_diffusion = self._program(
            prepare_the_tke_diffusion,
            constant_args={
                "half_level_height": metric.hhl,
                "tke_diffusion_factor": self._tke_diffusion_factor,
            },
            levels=(1, nlev + 1),  # 'DO k=2,ke1'; the flux levels start one row lower
            shifted=True,
        )

        # -- section 8) the circulation term as an extra TKE flux density
        self._compute_virtual_tke_profile = self._program(
            compute_virtual_tke_profile,
            constant_args={"tke_diffusion_limit_correction": self._tke_diffusion_limit_correction},
            levels=(1, nlev + 1),  # 'cur_prof(:,2)' then 'DO k=3,ke1'
            shifted=True,
        )

        # -- section 9) the semi-implicit TKE diffusion
        self._compute_implicit_part_of_tke_diffusion_momentum = self._program(
            compute_implicit_part_of_tke_diffusion_momentum,
            constant_args={"implicit_weight": self._implicit_weight},
            levels=(2, nlev + 1),
        )
        self._subtract_implicit_part_of_tke_diffusion_momentum = self._program(
            subtract_implicit_part_of_tke_diffusion_momentum,
            levels=(2, nlev),
        )
        self._compute_inverted_diffusion_momentum = self._program(
            compute_inverted_diffusion_momentum,
            levels=(1, nlev),  # 'k_tp+1' to 'ke'
            shifted=True,
        )
        self._compute_diffusion_inversion_factor = self._program(
            compute_diffusion_inversion_factor,
            levels=(2, nlev),
            shifted=True,
        )
        self._compute_explicit_tke_flux_density = self._program(
            compute_explicit_tke_flux_density,
            levels=(2, nlev + 1),
            shifted=True,
        )
        self._compute_tke_diffusion_right_hand_side = self._program(
            compute_tke_diffusion_right_hand_side,
            levels=(1, nlev + 1),
            shifted=True,
        )
        self._solve_tke_diffusion_equation = self._program(
            solve_tke_diffusion_equation,
            levels=(1, nlev),
            shifted=True,
        )
        self._add_virtual_diffusion_increment_to_tke_profile = self._program(
            add_virtual_diffusion_increment_to_tke_profile,
            levels=(1, nlev),
        )

        # -- section 10) the q tendency
        self._compute_turbulent_velocity_scale_tendency = self._program(
            compute_turbulent_velocity_scale_tendency,
            levels=(1, nlev + 1),  # 'DO k=2,ke' plus the surface row
        )

        # -- section 11) the SDSS back onto main levels
        self._interpolate_supersaturation_deviation_to_main_levels = self._program(
            interpolate_supersaturation_deviation_to_main_levels,
            levels=(0, nlev - 1),  # 'rcld(:,1)' then 'DO k=2,ke-1'
            shifted=True,
        )

    def _setup_vertdiff_programs(self) -> None:
        """Compile every stencil of 'vertdiff' with its domain and its constants bound.

        As `_setup_turbdiff_programs`, and for the same reason: the translation of the
        Fortran's one-based inclusive loop bounds into zero-based half-open GT4Py domains
        happens once per program, here, and the comment on each line is the Fortran loop.

        'vertdiff' runs two VARIABLE TYPES through one matrix each -- 'mom' for the two wind
        components and 'sca' for temperature, water vapour and cloud water
        (turb_vertdiff.f90:503-504) -- and they differ by exactly one row, because the momentum
        type takes a surface-CONCENTRATION condition and the scalar type a surface-FLUX
        condition ('tdc%lsflcnd', frozen '.TRUE.'). The Fortran writes that as 'k_sf+1-m' with
        'm = 1' or 'm = 2'; a GT4Py domain is fixed at compile time, so the two programs whose
        range depends on it are bound twice, once per type, and named for the type.
        """
        nlev = int(self._nlev)
        metric = self._metric_state

        # -- the parts of the matrix that depend on neither the variable type nor the variable:
        # one program of six statements. The bound pair is the discretisation momentum's own
        # 'disc_mom(i,k_hi)' then 'DO k=k_hi+1,k_lw'; the diffusion depth starts one row lower
        # and the four surface-row statements sit on 'vertical_end'.
        #
        # 'zvari(:,ke1,m) = flux/(rhon*tkv*...)' is at turb_vertdiff.f90:614-634, which the
        # Fortran runs inside the variable loop for 'tem' and again for 'vap'. Both rows are
        # written here, before the loop: the two 'zvari' components are distinct, and nothing
        # between this point and each variable's own use of its row writes either of them.
        self._prepare_the_vertical_diffusion_matrix = self._program(
            prepare_the_vertical_diffusion_matrix,
            constant_args={"half_level_height": metric.hhl},
            levels=(0, nlev),  # 'disc_mom(i,k_hi)' then 'DO k=k_hi+1,k_lw'
            shifted=True,
        )

        # -- once per variable type ('vert_grad_diff' and 'prep_impl_vert_diff'): one program of
        # six statements, ONE binding for both types. Every range that depends on the lower
        # boundary condition is expressed on 'elimination_end', which the caller passes -- 'nlev'
        # for the momentum type ('m = 1') and 'nlev - 1' for the scalar type ('m = 2'). The two
        # programs that follow it cannot be statements of it: both read 'invs_mom' through
        # 'Koff[-1]' and the first WRITES it, which DaCe drops. See the stencil's docstring.
        self._prep_impl_vert_diff = self._program(
            prep_impl_vert_diff,
            constant_args={"implicit_weight": self._implicit_weight},
            levels=(1, nlev),  # 'DO k=k_hi+1,k_lw'
            shifted=True,
        )
        self._invert_diffusion_momentum_at_the_surface_flux_level = self._program(
            invert_diffusion_momentum_at_the_surface_flux_level,
            levels=(nlev - 1, nlev),  # 'DO k=k_sf-m+1,k_sf-1', one row at 'm = 2' and none at 1
            shifted=True,
        )
        self._compute_diffusion_inversion_factor_of_a_variable_type = self._program(
            compute_diffusion_inversion_factor,
            levels=(1, nlev),  # the union of the two elimination ranges
            shifted=True,
        )

        # -- once per variable ('calc_impl_vert_diff' and the two loops around it)
        self._compute_current_profile = self._program(
            compute_current_profile,
            levels=(0, nlev),  # 'DO k=k_st_up,ke' plus the 'ke1' boundary value
        )
        self._compute_current_potential_temperature_profile = self._program(
            compute_current_potential_temperature_profile,
            levels=(0, nlev),
        )
        self._compute_surface_profile_value_from_flux_gradient = self._program(
            compute_surface_profile_value_from_flux_gradient,
            levels=(nlev, nlev + 1),  # 'cur_prof(i,k_sf)' under 'lsfgrduse'
            shifted=True,
        )
        # 'calc_impl_vert_diff' -- one program of five statements, one binding for both types.
        # The bound pair is the diffused main levels 'k_tp+1'..'k_sf-1'; the flux statements run
        # one row lower and one row deeper, and the implicit surface coupling is switched off for
        # a variable type with a surface-flux condition by an EMPTY domain.
        self._calc_impl_vert_diff = self._program(
            calc_impl_vert_diff,
            levels=(0, nlev),  # 'eff_flux(i,k_tp+1)' then 'DO k=k_tp+2,k_sf-1'
            shifted=True,
        )
        self._compute_and_apply_diffusion_tendency = self._program(
            compute_and_apply_diffusion_tendency,
            levels=(0, nlev),  # 'DO k=k_hi,k_lw' and 'DO k=k_st_pp,ke'
        )
        self._compute_and_apply_potential_temperature_diffusion_tendency = self._program(
            compute_and_apply_potential_temperature_diffusion_tendency,
            levels=(0, nlev),
        )

    def _smooth_the_tke_forcing(self) -> tuple[gtx.Field, gtx.Field]:
        """The optional vertical smoothing of section 2c), turb_diffusion.f90:1720-1738.

        A method rather than five lines of `run_turbdiff` because it is the one place in the
        routine where a block of the Fortran changes WHICH FIELD the following sections read,
        and because the Fortran's two guards are not both reproduced. ICON tests
        "frcsmot > z0" and then, at 'imode_frcsmot = 2', "ANY(trop_mask > z0)". Only the first
        is here: the second is a host-side reduction over the block, and skipping it costs
        nothing but a kernel launch, since where 'trop_mask' vanishes so does
        'versmot = frcsmot*trop_mask' and the smoothing is the identity there.

        NOT VALIDATED AGAINST ICON DATA. 'smooth_tke_forcing_vertically' is the one stencil of
        this granule with no reference capture behind it -- the capture's domain cannot exercise
        it at any 'frcsmot' -- so at 'frcsmot > 0' the profiles that reach sections 3) and 4)
        are unchecked. Its module docstring says why, and what stands in place of a reference.

        'disc_mom' is 'dicke' as section 1a) leaves it -- 'rho_n*dz/dt' on rows 1..nlev-1 --
        which is the array the Fortran passes (turb_diffusion.f90:1729, 1734). It weights each
        neighbour in the three-point average, which is what makes the smoothing conservative.

        Returns:
            'frm' and 'frh' as the following sections must read them: the smoothed profiles
            when the smoothing runs, the working fields themselves when it does not.
        """
        if self._smooth_tke_forcing_vertically is None:
            return self._frm, self._frh

        self._smooth_tke_forcing_vertically(
            tke_forcing=self._frm,
            discretisation_momentum=self._dicke,
            nlev=self._nlev,
            smoothed_tke_forcing=self._smoothed_mechanical_forcing,
        )
        self._smooth_tke_forcing_vertically(
            tke_forcing=self._frh,
            discretisation_momentum=self._dicke,
            nlev=self._nlev,
            smoothed_tke_forcing=self._smoothed_thermal_forcing,
        )
        return self._smoothed_mechanical_forcing, self._smoothed_thermal_forcing

    # ------------------------------------------------------------------------- the stage ---

    def run_turbdiff(
        self,
        *,
        input_state: states.TurbulenceInputState,
        surface_state: states.TurbulenceSurfaceState,
        diagnostic_state: states.TurbulenceDiagnosticState,
        tendency_state: states.TurbulenceTendencyState,
        dt_tke: float,
    ) -> None:
        """Run 'SUBROUTINE turbdiff' once: the atmospheric TKE closure and its q-diffusion.

        Reads `input_state` and `surface_state`, reads and writes `diagnostic_state`, writes
        `tendency_state`. `input_state` is never written (ADR-0001); the turbulent velocity the
        scheme produces goes to 'diagnostic_state.updated_tke', which is the 'ntur' time level
        of the Fortran's 'tke(:,:,ntim)' while 'input_state.tke' is the 'nvor' one.

        What is written, and what is left alone:

            diagnostic_state.updated_tke   rows 0..nlev-1 by section 3); row 'nlev' copied from
                                           the input, as 'turbdiff' leaves 'turbtran's value
            diagnostic_state.tkvm, tkvh    rows 1..nlev-1 by section 4)
            diagnostic_state.rhon          rows 1..nlev-1 and 'nlev' by section 0)
            diagnostic_state.rcld          rows 0..nlev-2 by section 11); rows nlev-1 and nlev
                                           copied from the half-level storage, over the column
                                           window and not the full width
            tendency_state.ddt_tke         rows 1..nlev by section 10); row 0 keeps the
                                           advection tendency it arrived with, which section 3)
                                           read as 'tvt'
            tendency_state.tket_hshr       rows 1..nlev-1 by section 2a)

        'diagnostic_state.tfm', 'tfh' and 'tfv' are NOT written. The Fortran would overwrite
        them in sections 3) and 4) under 'lsrfshear' and in section 2c) under "rsur_sher > 0",
        and both are false for every configuration this granule accepts ('rsur_sher = 0' and
        the frozen 'imode_suradap = 0'). Measured over the capture: all three keep the values
        'turbtran' left, at every one of the fifteen section savepoints.

        Args:
            input_state: The atmospheric column and the external forcings. Read-only.
            surface_state: The grid-mean surface state. Read-only.
            diagnostic_state: The turbulence diagnostics; read and written.
            tendency_state: Where the tendencies go. 'ddt_tke' is read on entry as the
                advection tendency, exactly as the Fortran's 'INTENT(INOUT) tketens' is.
            dt_tke: The time step of the TKE equation [s], ICON's 'dt_tke'.
        """
        # 'num_cells' is the horizontal length the raw-array copies below are clamped to; see
        # the section comment above '_extract_level' for why it is the grid's and not an array's
        # own. Bound in the same statement as 'nlev' because this method sits exactly on ruff's
        # statement limit.
        nlev, num_cells = int(self._nlev), self._grid.num_cells
        inverse_dt_tke = 1.0 / dt_tke  # 'fr_tke = z1/dt_tke' (turb_utilities.f90:317)

        # 'lays' divides by the surface diffusion coefficients section 4) has not yet raised,
        # so the two rows are taken before anything writes them.
        _extract_level(
            diagnostic_state.tkvm,
            nlev,
            self._diffusion_coefficient_for_momentum_at_the_surface,
            num_cells,
        )
        _extract_level(
            diagnostic_state.tkvh,
            nlev,
            self._diffusion_coefficient_for_scalars_at_the_surface,
            num_cells,
        )

        # -- 0) conserved variables, cloud cover, thermodynamic factors, length scales -------

        self._compute_conserved_variables_and_factors_at_main_levels(
            temperature=input_state.t,
            specific_humidity=input_state.qv,
            cloud_water=input_state.qc,
            pressure=input_state.prs,
            exner_factor=input_state.epr,
            supersaturation_deviation=diagnostic_state.rcld,
            liquid_water_potential_temperature=self._conserved_liquid_water_potential_temperature,
            total_water=self._conserved_total_water,
            liquid_water=self._conserved_liquid_water,
            cloud_cover=self._cloud_cover_on_main_levels,
            specific_heat_ratio=self._zaux_2,
            dqsat_dt=self._dqsat_dt_on_main_levels,
            buoyancy_factor_tet_l=self._buoyancy_factor_tet_l_on_main_levels,
            buoyancy_factor_h2o_g=self._buoyancy_factor_h2o_g_on_main_levels,
        )

        # The second 'adjust_satur_equil' call interpolates the lowest main level down to the
        # zero level, so it needs that row as a cell field; and the four surface values it
        # starts from are what 'turb_setup' put into 'zvari(:,ke1,:)' -- 'ps', 't_g', 'qv_s' and,
        # at the frozen 'ilow_def_cond = 2', zero (turb_utilities.f90:341-350).
        _extract_level(
            self._conserved_liquid_water_potential_temperature,
            nlev - 1,
            self._liquid_water_potential_temperature_above_the_surface,
            num_cells,
        )
        _extract_level(
            self._conserved_total_water,
            nlev - 1,
            self._total_water_above_the_surface,
            num_cells,
        )
        self._compute_conserved_variables_and_factors_at_the_surface(
            surface_pressure=surface_state.ps,
            surface_temperature=surface_state.t_g,
            surface_specific_humidity=surface_state.qv_s,
            surface_liquid_water=self._surface_liquid_water,
            liquid_water_potential_temperature_above=self._liquid_water_potential_temperature_above_the_surface,
            total_water_above=self._total_water_above_the_surface,
            laminar_reduction_factor_for_scalars=diagnostic_state.tfh,
            supersaturation_deviation=diagnostic_state.rcld,
            exner_factor=self._zaux_1,
            liquid_water_potential_temperature=self._conserved_liquid_water_potential_temperature,
            total_water=self._conserved_total_water,
            liquid_water=self._conserved_liquid_water,
            cloud_cover=self._rcld,
            air_density=diagnostic_state.rhon,
            specific_heat_ratio=self._zaux_2,
            dqsat_dt=self._zaux_3,
            buoyancy_factor_tet_l=self._zaux_4,
            buoyancy_factor_h2o_g=self._zaux_5,
        )

        self._compute_layer_depth(layer_depth=self._dicke)
        self._compute_horizontal_wind_including_the_zero_level(
            zonal_wind=input_state.u,
            meridional_wind=input_state.v,
            laminar_reduction_factor_for_momentum=diagnostic_state.tfm,
            nlev=self._nlev,
            zonal_wind_on_conserved_variable_levels=self._conserved_zonal_wind,
            meridional_wind_on_conserved_variable_levels=self._conserved_meridional_wind,
        )

        self._compute_half_level_interpolation_weight(interpolation_weight=self._hlp)
        self._interpolate_variables_onto_half_levels(
            cloud_cover=self._cloud_cover_on_main_levels,
            exner_factor=input_state.epr,
            dqsat_dt=self._dqsat_dt_on_main_levels,
            buoyancy_factor_tet_l=self._buoyancy_factor_tet_l_on_main_levels,
            buoyancy_factor_h2o_g=self._buoyancy_factor_h2o_g_on_main_levels,
            pressure=input_state.prs,
            air_density=input_state.rhoh,
            interpolation_weight=self._hlp,
            cloud_cover_on_half_levels=self._rcld,
            exner_factor_on_half_levels=self._zaux_1,
            dqsat_dt_on_half_levels=self._zaux_3,
            buoyancy_factor_tet_l_on_half_levels=self._zaux_4,
            buoyancy_factor_h2o_g_on_half_levels=self._zaux_5,
            pressure_on_half_levels=self._half_level_pressure,
            air_density_on_half_levels=diagnostic_state.rhon,
        )
        # Four of the seven interpolations are IN PLACE in the Fortran, over rows 1..nlev-1 of
        # the storage that already held the main-level values. Row 0 is therefore the main-level
        # value there, and the port has to put it back because it interpolates out of place.
        _copy_level(self._cloud_cover_on_main_levels, 0, self._rcld, num_cells)
        _copy_level(self._dqsat_dt_on_main_levels, 0, self._zaux_3, num_cells)
        _copy_level(self._buoyancy_factor_tet_l_on_main_levels, 0, self._zaux_4, num_cells)
        _copy_level(self._buoyancy_factor_h2o_g_on_main_levels, 0, self._zaux_5, num_cells)
        # 'prss' is a pointer into 'zvari(:,:,0)' whose surface row 'turb_setup' filled with the
        # surface pressure (turb_utilities.f90:341); section 3) reads it there.
        _set_level(surface_state.ps, nlev, self._half_level_pressure, num_cells)

        self._compute_turbulent_length_scale(
            layer_depth=self._dicke,
            roughness_length_times_gravity=diagnostic_state.gz0,
            nlev=self._nlev,
            turbulent_length_scale=self._len_scale,
        )

        # -- 1a) the vertical gradients -------------------------------------------------------

        # The gradients replace the variables in the Fortran's own storage, so the model top --
        # which the difference quotient never writes -- carries the variable's value into the
        # 'zvari' the routine returns. Row 0 is disjoint from every row the program below writes,
        # so this runs before it rather than between two of its statements, as it used to.
        for variable, gradient in (
            (self._conserved_zonal_wind, self._gradient_zonal_wind),
            (self._conserved_meridional_wind, self._gradient_meridional_wind),
            (
                self._conserved_liquid_water_potential_temperature,
                self._gradient_liquid_water_potential_temperature,
            ),
            (self._conserved_total_water, self._gradient_total_water),
            (self._conserved_liquid_water, self._gradient_liquid_water),
        ):
            _copy_level(variable, 0, gradient, num_cells)
        # 'hlp' stops being the interpolation weight here and 'dicke' stops being the layer
        # depth; the length scale in section 0) was the last reader of both.
        self._compute_vertical_gradients_of_conserved_variables(
            tvm=diagnostic_state.tvm,
            tvh=diagnostic_state.tvh,
            tkvm_at_surface=self._diffusion_coefficient_for_momentum_at_the_surface,
            tkvh_at_surface=self._diffusion_coefficient_for_scalars_at_the_surface,
            tfm=diagnostic_state.tfm,
            tfh=diagnostic_state.tfh,
            rhon=diagnostic_state.rhon,
            inverse_tke_time_step=inverse_dt_tke,
            zonal_wind=self._conserved_zonal_wind,
            meridional_wind=self._conserved_meridional_wind,
            liquid_water_potential_temperature=self._conserved_liquid_water_potential_temperature,
            total_water=self._conserved_total_water,
            liquid_water=self._conserved_liquid_water,
            nlev=self._nlev,
            surface_transfer_ratio_for_momentum=self._surface_transfer_ratio_for_momentum,
            surface_transfer_ratio_for_scalars=self._surface_transfer_ratio_for_scalars,
            inverse_layer_depth=self._hlp,
            tke_discretisation_momentum=self._dicke,
            zonal_wind_gradient=self._gradient_zonal_wind,
            meridional_wind_gradient=self._gradient_meridional_wind,
            liquid_water_potential_temperature_gradient=self._gradient_liquid_water_potential_temperature,
            total_water_gradient=self._gradient_total_water,
            liquid_water_gradient=self._gradient_liquid_water,
        )

        # -- 1b) the two basic TKE forcing functions ------------------------------------------

        # Every row section 1b) writes into 'frm' is overwritten again by section 2a) at
        # 'itype_sher = 2', which is the only value this granule accepts. The shear statement is
        # kept because the Fortran keeps it: 'frm' is INTENT(OUT)-like scratch and a future
        # 'itype_sher < 2' would need exactly this value.
        self._compute_tke_forcing_functions(
            buoyancy_factor_tet_l=self._zaux_4,
            buoyancy_factor_h2o_g=self._zaux_5,
            vertical_gradient_tet_l=self._gradient_liquid_water_potential_temperature,
            vertical_gradient_h2o_g=self._gradient_total_water,
            vertical_gradient_u=self._gradient_zonal_wind,
            vertical_gradient_v=self._gradient_meridional_wind,
            thermal_forcing=self._frh,
            mechanical_forcing=self._frm,
        )

        # -- 1c) DEAD: 'IF (lini)' and 'IF (ltkeadapt)', neither reachable here ----------------

        # -- 2a) the three-dimensional shear complements --------------------------------------

        # Six statements, three of which read a field an earlier one wrote. That is the aliasing
        # shape every backend orders correctly; the stencil's docstring carries the other two.
        #
        # 'tket_hshr' is written here and read back by the total forcing below. The Fortran keeps
        # the separated-shear source in 'hlp' and copies it out under 'loutshs', which
        # 'FROZEN_SWITCHES' fixes '.TRUE.'; the port writes the output slot directly, because
        # 'hlp' is about to become the SSO term.
        self._add_three_dimensional_shear_complements(
            vertical_gradient_u=self._gradient_zonal_wind,
            vertical_gradient_v=self._gradient_meridional_wind,
            dwdx=input_state.dwdx,
            dwdy=input_state.dwdy,
            horizontal_divergence=input_state.hdiv,
            horizontal_deformation_square=input_state.hdef2,
            thermal_forcing=self._frh,
            turbulent_velocity_scale=input_state.tke,
            sso_tendency_u=input_state.ut_sso,
            sso_tendency_v=input_state.vt_sso,
            wind_u=input_state.u,
            wind_v=input_state.v,
            mean_shear_forcing=self._mean_shear_forcing,
            inverse_richardson_number_factor=self._inverse_richardson_number_factor,
            uncorrected_horizontal_shear_length_scale=self._uncorrected_horizontal_shear_length_scale,
            effective_horizontal_shear_length_scale=self._effective_horizontal_shear_length_scale,
            separated_horizontal_shear_tke_source=tendency_state.tket_hshr,
            sso_wake_energy_production=self._hlp,
        )
        # 'inverse_richardson_number_factor' is bound in '_setup_turbdiff_programs', because
        # 'imode_tkesso = 1' selects a program that does not take it.
        self._compute_total_mechanical_forcing(
            mean_shear_forcing=self._mean_shear_forcing,
            separated_horizontal_shear_tke_source=tendency_state.tket_hshr,
            sso_wake_energy_production=self._hlp,
            momentum_diffusion_coefficient=diagnostic_state.tkvm,
            mechanical_forcing=self._frm,
        )

        # -- 2b) DEAD: the vertically resolved canopy, 'kcm = ke+1' ---------------------------

        # -- 2c) final preparations ------------------------------------------------------------

        mechanical_forcing, thermal_forcing = self._smooth_the_tke_forcing()

        self._compute_stability_lengths_from_diffusion_coefficients(
            diffusion_coefficient_for_momentum=diagnostic_state.tkvm,
            diffusion_coefficient_for_scalars=diagnostic_state.tkvh,
            turbulent_velocity_scale=input_state.tke,
            stability_length_for_momentum=self._stability_length_for_momentum,
            stability_length_for_scalars=self._stability_length_for_scalars,
        )

        # -- 3) the turbulent budgets ('solve_turb_budgets') -----------------------------------

        # The surface half level is 'turbtran's and 'turbdiff' does not touch it; with the two
        # TKE time levels as two fields it has to be carried across explicitly.
        _copy_level(input_state.tke, nlev, diagnostic_state.updated_tke, num_cells)
        # 'self._rcld' IS PASSED TWICE, as the cloud cover the circulation term reads and as the
        # SDSS the next statement writes -- one Fortran storage with two meanings, which the
        # granule reproduces. The two are separate program parameters, so GT4Py cannot see the
        # aliasing; what keeps them apart is the order of the statements inside
        # 'solve_turb_budgets', asserted by
        # 'test_the_sdss_is_written_after_the_circulation_term_reads_the_cloud_cover'.
        # 'diagnostic_state.updated_tke' is likewise both written by the first statement and read
        # by the second, one row up, which is what used to make the model-top row its own program.
        self._solve_turb_budgets(
            master_length_scale=self._len_scale,
            stability_length_for_momentum=self._stability_length_for_momentum,
            stability_length_for_scalars=self._stability_length_for_scalars,
            mechanical_forcing=mechanical_forcing,
            thermal_forcing=thermal_forcing,
            previous_velocity_scale=input_state.tke,
            transport_tendency=tendency_state.ddt_tke,
            cloud_cover=self._rcld,
            half_level_pressure=self._half_level_pressure,
            air_density=diagnostic_state.rhon,
            exner_factor=self._zaux_1,
            saturation_humidity_derivative=self._zaux_3,
            gradient_of_liquid_water_potential_temperature=self._gradient_liquid_water_potential_temperature,
            gradient_of_total_water=self._gradient_total_water,
            pattern_length_scale=surface_state.l_pat,
            tke_time_step=dt_tke,
            inverse_tke_time_step=inverse_dt_tke,
            turbulent_velocity_scale=diagnostic_state.updated_tke,
            updated_stability_length_for_momentum=self._updated_stability_length_for_momentum,
            updated_stability_length_for_scalars=self._updated_stability_length_for_scalars,
            circulation_acceleration=self._circulation_acceleration,
            supersaturation_standard_deviation=self._rcld,
            diffusion_coefficient_for_momentum=self._diffusion_coefficient_for_momentum,
            diffusion_coefficient_for_scalars=self._diffusion_coefficient_for_scalars,
        )
        # In place on purpose, and out of the program above on purpose: the read set is row 1
        # and the write set is row 0, and one field bound to TWO parameters is the form DaCe
        # compiles correctly. One parameter read and written by one statement is not.
        self._set_turbulent_velocity_scale_at_model_top(
            turbulent_velocity_scale=diagnostic_state.updated_tke,
            turbulent_velocity_scale_with_top=diagnostic_state.updated_tke,
        )

        # -- 4) lower limits of the diffusion coefficients --------------------------------------

        self._compute_effective_diffusion_coefficients(
            diffusion_coefficient_for_momentum=self._diffusion_coefficient_for_momentum,
            diffusion_coefficient_for_scalars=self._diffusion_coefficient_for_scalars,
            inverse_richardson_number=self._inverse_richardson_number_factor,
            roughness_length_times_gravity=diagnostic_state.gz0,
            pattern_length_scale=surface_state.l_pat,
            surface_reduction_for_momentum=diagnostic_state.tkred_sfc,
            surface_reduction_for_scalars=diagnostic_state.tkred_sfc_h,
            effective_diffusion_coefficient_for_momentum=diagnostic_state.tkvm,
            effective_diffusion_coefficient_for_scalars=diagnostic_state.tkvh,
        )

        # -- 5) DEAD: 'IF (ltmpcor)' and 'IF (ldocirflx)', both frozen '.FALSE.' -----------------

        # -- 6) preparations for the TKE diffusion ------------------------------------------------

        self._prepare_the_tke_diffusion(
            turbulent_velocity_scale=diagnostic_state.updated_tke,
            mixing_length=self._len_scale,
            air_density_at_main_levels=input_state.rhoh,
            air_density=diagnostic_state.rhon,
            scalar_diffusion_coefficient=diagnostic_state.tkvh,
            circulation_acceleration=self._circulation_acceleration,
            saved_tke_profile=self._zaux_2,
            explicit_diffusion_momentum=self._zaux_3,
            cke_flux_density=self._frh,
            cke_flux_at_main_levels=self._frm,
        )

        # -- 7) DEAD: 'IF (ldocirflx)', i.e. 'lcirflx', frozen '.FALSE.' --------------------------

        # -- 8) the circulation term as an additional TKE flux density -----------------------------

        # Section 8) and section 9)'s 'add_virtual_diffusion_increment_to_tke_profile' are a
        # PAIR. Without the circulation term there is no virtual profile to build and none to
        # remove: 'turb_diffusion.f90:2390' points 'cur_prof' at 'sav_prof' itself and both
        # programs are skipped. Only the branch taken here has reference data.
        if self._circulation_term_is_active:
            self._compute_virtual_tke_profile(
                saved_tke_profile=self._zaux_2,
                cke_flux_at_main_levels=self._frm,
                explicit_diffusion_momentum=self._zaux_3,
                virtual_tke_profile=self._hlp,
            )

        # -- 9) the semi-implicit vertical diffusion of the TKE -------------------------------------

        self._compute_implicit_part_of_tke_diffusion_momentum(
            diffusion_momentum=self._zaux_3,
            implicit_diffusion_momentum=self._zaux_4,
        )
        # IN PLACE, as the Fortran is. The subtraction is pointwise and covers one flux level
        # less than the implicit part, so the surface row has to keep the value section 6) put
        # there -- which it does only if this writes the storage it reads.
        self._subtract_implicit_part_of_tke_diffusion_momentum(
            diffusion_momentum=self._zaux_3,
            implicit_diffusion_momentum=self._zaux_4,
            explicit_diffusion_momentum=self._zaux_3,
        )
        self._compute_inverted_diffusion_momentum(
            discretisation_momentum=self._dicke,
            implicit_diffusion_momentum=self._zaux_4,
            inverted_diffusion_momentum=self._zaux_5,
        )
        self._compute_diffusion_inversion_factor(
            inverted_diffusion_momentum=self._zaux_5,
            implicit_diffusion_momentum=self._zaux_4,
            inversion_factor=self._frh,
        )
        self._compute_explicit_tke_flux_density(
            explicit_diffusion_momentum=self._zaux_3,
            implicit_diffusion_momentum=self._zaux_4,
            current_tke_profile=self._current_virtual_profile,
            nlev=self._nlev,
            explicit_tke_flux_density=self._explicit_tke_flux_density,
        )
        # 'len_scale' stops being the master length scale here; section 6) was its last reader.
        self._compute_tke_diffusion_right_hand_side(
            discretisation_momentum=self._dicke,
            current_tke_profile=self._current_virtual_profile,
            explicit_tke_flux_density=self._explicit_tke_flux_density,
            uppermost_diffused_level=gtx.int32(1),
            nlev=self._nlev,
            right_hand_side=self._len_scale,
        )
        self._solve_tke_diffusion_equation(
            right_hand_side=self._len_scale,
            implicit_diffusion_momentum=self._zaux_4,
            inverted_diffusion_momentum=self._zaux_5,
            inversion_factor=self._frh,
            updated_tke_profile=self._zaux_1,
        )
        if self._circulation_term_is_active:
            self._add_virtual_diffusion_increment_to_tke_profile(
                saved_tke_profile=self._zaux_2,
                updated_virtual_profile=self._zaux_1,
                current_virtual_profile=self._hlp,
                updated_tke_profile=self._zaux_1,
            )

        # -- 10) the q tendency of the TKE diffusion -------------------------------------------------

        self._compute_turbulent_velocity_scale_tendency(
            updated_tke_profile=self._zaux_1,
            turbulent_velocity_scale=diagnostic_state.updated_tke,
            inverse_tke_time_step=inverse_dt_tke,
            nlev=self._nlev,
            turbulent_velocity_scale_tendency=tendency_state.ddt_tke,
        )

        # -- 11) the SDSS back onto main levels --------------------------------------------------------

        # The Fortran averages 'rcld' onto main levels in its own storage, so the two rows the
        # averaging does not reach keep their half-level values. Out of place here, because the
        # average reads the half level below the row it writes.
        #
        # CLAMPED TO THE COLUMN WINDOW, not to 'num_cells' like the other raw copies. This is the
        # only one whose SOURCE is a granule-internal field and whose TARGET is the caller's, so
        # it is the only one that can carry a column the scheme never computed into ICON's state.
        # 'self._rcld' is zero outside 'ivstart:ivend'; ICON's 'rcld' there holds the lateral
        # boundary and halo values that 'turbdiff' deliberately leaves alone, and overwriting
        # them with zeros showed up as 'max_rel_err = 1.0' on the surface half level in an
        # ICON4PY_MODE_VERIFY run while every stencil-written output agreed to 1e-6.
        _copy_levels(
            self._rcld,
            slice(nlev - 1, nlev + 1),
            diagnostic_state.rcld,
            slice(int(self._start_cell), int(self._end_cell)),
        )
        self._interpolate_supersaturation_deviation_to_main_levels(
            supersaturation_deviation_on_half_levels=self._rcld,
            supersaturation_deviation_on_main_levels=diagnostic_state.rcld,
        )

    # ------------------------------------------------------------------ the second stage ---

    def _prepare_the_diffusion_matrix(
        self,
        *,
        input_state: states.TurbulenceInputState,
        surface_state: states.TurbulenceSurfaceState,
        diagnostic_state: states.TurbulenceDiagnosticState,
        reciprocal_time_step: float,
    ) -> None:
        """The one program of 'vertdiff' that neither variable type nor variable can change.

        'rhon' is the one field of the granule that both stages write: 'turbdiff' fills rows
        1..nlev of it and 'vertdiff' replaces the surface row with the ideal-gas density of the
        ground, which is a different quantity from the Prandtl-layer boundary value 'turbdiff'
        left there. The Fortran says so itself at turb_vertdiff.f90:544-547 and the two
        savepoints differ measurably in exactly that row.

        The two prescribed surface gradients go into 'zvari(:,ke1,tet_l)' and
        'zvari(:,ke1,h2o_g)' -- the granule's gradient fields -- because that is the storage the
        Fortran uses, and because their consumer reads them from there.
        """
        # 'diffusion_coefficient' is the SCALAR type's, whichever type is running: the two
        # variables with a prescribed surface flux are both scalars, so 'vtyp(ivtype)%tkv' is
        # 'tkvh'. The two gradient statements read the 'rhon' surface row and the 'eprs' the
        # first two statements of the same program wrote.
        self._prepare_the_vertical_diffusion_matrix(
            surface_pressure=surface_state.ps,
            surface_specific_humidity=surface_state.qv_s,
            surface_temperature=surface_state.t_g,
            air_density_at_main_levels=input_state.rhoh,
            reciprocal_time_step=reciprocal_time_step,
            diffusion_coefficient=diagnostic_state.tkvh,
            sensible_heat_flux=diagnostic_state.shfl_s,
            water_vapour_flux=diagnostic_state.qvfl_s,
            air_density=diagnostic_state.rhon,
            surface_exner_factor=self._surface_exner_factor,
            discretisation_momentum=self._discretisation_momentum,
            diffusion_depth=self._diffusion_depth,
            surface_temperature_gradient=self._gradient_liquid_water_potential_temperature,
            surface_vapour_gradient=self._gradient_total_water,
        )

    def _factorise_one_variable_type(
        self,
        *,
        diffusion_coefficient: gtx.Field,
        transfer_velocity: gtx.Field,
        air_density: gtx.Field,
        surface_flux_condition: bool,
    ) -> None:
        """Build and LU-factorise the tridiagonal matrix of one variable type.

        'vert_grad_diff:2461-2478' and 'prep_impl_vert_diff:2764-2858', once for 'mom' and once
        for 'sca'. One matrix serves every variable of its type, which is the whole reason
        'vertdiff' loops over types on the outside and variables on the inside. Three programs
        since the stencil merge: 'prep_impl_vert_diff' holds six of the eight Fortran statements.

        THE TWO TYPES DIFFER BY ONE ROW AND NOTHING ELSE. Under a surface-FLUX condition
        ('lsflucond', the scalar type) the implicit part stops above the surface row and the
        elimination stops one row above that, so the row it stopped at is finished by a program
        of its own -- the Fortran's third loop, 'DO k=k_sf-m+1,k_sf-1', which is empty at
        'm = 1'. Getting that boundary wrong changes 'u_tens' and 'v_tens' with nothing
        upstream of them disagreeing. That program and the inversion factor after it are the two
        Fortran statements the merge could NOT absorb: both read 'invs_mom' through 'Koff[-1]'
        and the first writes it, and DaCe silently drops a statement whose 'out=' names the same
        parameter as a shifted input.

        'diffusion_coefficient' is read at the surface row as well as inside, so the row must be
        the one 'turbtran' produced and 'turbdiff' left alone -- section 4) writes rows
        1..nlev-1 only.

        Args:
            diffusion_coefficient: 'vtyp(ivtype)%tkv', i.e. 'tkvm' or 'tkvh' [m2/s].
            transfer_velocity: 'vtyp(ivtype)%tsv', i.e. 'tvm' or 'tvh' [m/s].
            air_density: 'rhon' on half levels, surface row included [kg/m3].
            surface_flux_condition: 'lsflucond'; false for momentum, 'tdc%lsflcnd' for scalars.
        """
        # 'elimination_end' is the one row the two types differ by: 'k_sf - m' zero-based, so
        # 'nlev' under a surface-concentration condition and 'nlev - 1' under a flux condition.
        # The implicit part runs one row further than it. The subtraction inside is in place and
        # covers one row less than the split, so the surface row keeps the WHOLE diffusion
        # momentum -- which is what makes the surface row of the explicit flux the explicit
        # surface flux.
        self._prep_impl_vert_diff(
            diffusion_coefficient=diffusion_coefficient,
            air_density=air_density,
            surface_transfer_velocity=transfer_velocity,
            discretisation_momentum=self._discretisation_momentum,
            elimination_end=gtx.int32(
                int(self._nlev) - 1 if surface_flux_condition else int(self._nlev)
            ),
            diffusion_momentum=self._diffusion_momentum,
            diffusion_depth=self._diffusion_depth,
            implicit_diffusion_momentum=self._implicit_diffusion_momentum,
            inverted_diffusion_momentum=self._inverted_diffusion_momentum,
        )
        if surface_flux_condition:
            self._invert_diffusion_momentum_at_the_surface_flux_level(
                discretisation_momentum=self._discretisation_momentum,
                implicit_diffusion_momentum=self._implicit_diffusion_momentum,
                inverted_diffusion_momentum_above=self._inverted_diffusion_momentum,
                inverted_diffusion_momentum=self._inverted_diffusion_momentum,
            )
        self._compute_diffusion_inversion_factor_of_a_variable_type(
            inverted_diffusion_momentum=self._inverted_diffusion_momentum,
            implicit_diffusion_momentum=self._implicit_diffusion_momentum,
            inversion_factor=self._inversion_factor,
        )

    def _diffuse_one_variable(
        self,
        *,
        variable: DiffusedVariable,
        exner_factor: gtx.Field,
        surface_flux_condition: bool,
        reciprocal_time_step: float,
    ) -> None:
        """Diffuse one first-order variable through the matrix its type left standing.

        turb_vertdiff.f90:646-799 around 'calc_impl_vert_diff'. Four programs since the stencil
        merge, of which two are conditional; every field but 'variable.right_hand_side' is
        workspace shared with the other four variables, so the order here is the Fortran's and
        not a preference.

        THE RIGHT-HAND SIDE IS 'zvari(:,:,m)' AND THAT IS A CROSS-STAGE ALIAS. The ICON
        interface passes one 'zvari' to both stages (mo_nwp_turbdiff_interface.f90:652, :735),
        'turbdiff' leaves the vertical gradients of the quasi-conserved variables in it, and
        'vertdiff' overwrites component 'm' with the right-hand side of variable 'm'. At
        'ldogrdcor = .FALSE.' -- what 'ldoexpcor' and 'ldocirflx' being false forces here --
        nothing reads the gradients back, so the overwrite is total and the port reproduces it
        by handing each variable the '_gradient_*' field of its own component.

        THE SURFACE ROW OF THAT FIELD HAS THREE OCCUPANTS IN SUCCESSION, for 't' and 'qv':
        'turbdiff's gradient, then the prescribed surface gradient
        '_prepare_the_diffusion_matrix' wrote, then the explicit surface flux copied here. The
        second is read by 'compute_surface_profile_value_from_flux_gradient' before the third
        replaces it, which is why the copy is where it is.

        Args:
            variable: Which variable, and the three fields that are its own.
            exner_factor: 'epr' on main levels; read only for the temperature.
            surface_flux_condition: 'lsflucond' of the variable's type.
            reciprocal_time_step: 'fr_var = 1/dt_var' [1/s].
        """
        if variable.is_potential_temperature:
            self._compute_current_potential_temperature_profile(
                temperature=variable.profile,
                exner_factor=exner_factor,
                current_profile=self._current_profile,
            )
        else:
            self._compute_current_profile(
                variable=variable.profile, current_profile=self._current_profile
            )
        if variable.has_a_prescribed_surface_flux:
            self._compute_surface_profile_value_from_flux_gradient(
                current_profile_above=self._current_profile,
                diffusion_depth=self._diffusion_depth,
                surface_gradient=variable.right_hand_side,
                current_profile=self._current_profile,
            )
        # The implicit surface coupling is the Fortran's 'IF (.NOT.lsflucond)', expressed as a
        # domain: 'nlev' when it applies, 'nlev + 1' -- an empty range -- when it does not.
        self._calc_impl_vert_diff(
            explicit_diffusion_momentum=self._diffusion_momentum,
            current_profile=self._current_profile,
            implicit_diffusion_momentum=self._implicit_diffusion_momentum,
            discretisation_momentum=self._discretisation_momentum,
            inverted_diffusion_momentum=self._inverted_diffusion_momentum,
            inversion_factor=self._inversion_factor,
            surface_addition_start=gtx.int32(
                int(self._nlev) + 1 if surface_flux_condition else int(self._nlev)
            ),
            explicit_flux_density=self._explicit_flux_density,
            right_hand_side=variable.right_hand_side,
            updated_profile=self._updated_profile,
        )
        # 'eff_flux' becomes the right-hand side in place in the Fortran, over rows 0..nlev-1
        # only, so its surface row keeps the explicit flux. The port computes out of place --
        # the right-hand side reads flux level 'k+1' while writing row 'k' -- so the row that
        # survives in the Fortran has to be carried across here. It runs AFTER the program
        # rather than in the middle of it, which is exactly equivalent: nothing in there reads
        # the right-hand side's surface row, the solve covering 'k_tp+1..k_sf-1' only.
        _copy_level(
            self._explicit_flux_density,
            int(self._nlev),
            variable.right_hand_side,
            self._grid.num_cells,
        )
        # IN PLACE on the tendency, as 'vert_grad_diff:2668' is: the accumulation is pointwise
        # and ICON has one array. 'tendency_state' is an output container, so ADR-0001 is not
        # in question -- what may not be written is 'input_state', and nothing here does.
        if variable.is_potential_temperature:
            self._compute_and_apply_potential_temperature_diffusion_tendency(
                updated_profile=self._updated_profile,
                current_profile=self._current_profile,
                exner_factor=exner_factor,
                temperature_tendency_before=variable.tendency,
                reciprocal_time_step=reciprocal_time_step,
                diffusion_tendency=self._diffusion_increment,
                temperature_tendency=variable.tendency,
            )
        else:
            self._compute_and_apply_diffusion_tendency(
                updated_profile=self._updated_profile,
                current_profile=self._current_profile,
                variable_tendency_before=variable.tendency,
                reciprocal_time_step=reciprocal_time_step,
                diffusion_tendency=self._diffusion_increment,
                variable_tendency=variable.tendency,
            )

    def run_vertdiff(
        self,
        *,
        input_state: states.TurbulenceInputState,
        surface_state: states.TurbulenceSurfaceState,
        diagnostic_state: states.TurbulenceDiagnosticState,
        tendency_state: states.TurbulenceTendencyState,
        dt_var: float,
    ) -> None:
        """Run 'SUBROUTINE vertdiff' once: the implicit vertical diffusion of u, v, T, qv, qc.

        Reads `input_state` and `surface_state`, reads and writes `diagnostic_state.rhon`,
        accumulates into `tendency_state`. `input_state` is never written (ADR-0001), and
        neither is ICON's: 'u_tens'..'qc_tens' are mandatory 'TARGET, INTENT(INOUT)'
        (turb_vertdiff.f90:293-302), the accumulation at ':783-806' is unconditional, and the
        optional in-place incrementation of the prognostic variables went upstream in
        '597f090cf2'. 'u'..'qc' keep 'INTENT(INOUT)' only because ':451-460' pointer-associates
        them; they are on no left-hand side in the file. ICON adds the tendencies to the state
        itself, once, at 'mo_nwp_turbdiff_interface.f90:910-960'; the interface passes all five
        tendency arrays at ':724-728'.

        What is written, and what is left alone:

            tendency_state.ddt_u, ddt_v      accumulated over rows 0..nlev-1
            tendency_state.ddt_t             as above, through the Exner factor
            tendency_state.ddt_qv, ddt_qc    as above
            diagnostic_state.rhon            row 'nlev' only, replacing what 'turbdiff' left
            the five '_gradient_*' fields    'zvari(:,:,1..5)', overwritten by the right-hand
                                             sides; see `_diffuse_one_variable`

        'shfl_s' and 'qvfl_s' are READ and not written. The Fortran would recompute them at
        turb_vertdiff.f90:850-895 from the effective implicit fluxes, but only under
        '.NOT.(lsfluse .AND. tdc%lsflcnd)', and the interface passes 'lsfluse = tdc%lsflcnd'
        with 'lsflcnd' frozen '.TRUE.', so both come out byte-identical. 'umfl_s' and 'vmfl_s'
        are not passed by the interface at all.

        THE ORDER IS THE FORTRAN'S: both wind components through the momentum matrix, then
        temperature, water vapour and cloud water through the scalar one. It is not a
        preference. One matrix and one profile workspace serve all five variables, so the
        variables of a type must run between that type's factorisation and the next one; and
        the surface row of the implicit momentum still holds the momentum type's value when the
        stage returns, because the scalar type never writes it.

        WHAT IS NOT REACHED, in the configuration the ICON interface passes: 'itndcon = 0' (no
        explicit-tendency handling), 'ldogrdcor = .FALSE.' (no gradient correction, so 'zvari'
        is written and never read), 'l3dflxout = .FALSE.' (no effective-flux integration),
        'ndtr = 0' (no passive tracers), 'kcm = ke1' (no canopy volume correction) and
        'lprecnd = .FALSE.' (no preconditioning). Each is asserted against the entry savepoint
        by 'test_vertdiff_runs_in_the_configuration_this_port_assumes'.

        THE PASSIVE TRACERS ARE REFUSED HERE, and this is the innermost of three refusals of
        the same thing. 'vertdiff' is the only one of the two stages that has them at all --
        'ptr(:)' and 'ndtr' are dummy arguments of 'vertdiff' (turb_vertdiff.f90:135) and
        appear nowhere in 'turbdiff' -- so this method, and not `run` and not `run_turbdiff`,
        is where a tracer tuple would be dropped. It reads neither
        `TurbulenceInputState.tracers` nor `TurbulenceTendencyState.ddt_tracers`; the
        containers declare them because the Fortran interface has them, and a caller that
        filled them and got a successful return would be missing a physical process with
        nothing to say so. The other two refusals of the same condition:

            ICON      'check_supported_configuration' (mo_icon4py_turbulence.f90), which fires
                      first on the blue line and is the only one that can name the ICON
                      namelist switches that produced the tracers.
            wrapper   'turbulence_init' ('icon4py.bindings.turbulence_wrapper'), which refuses
                      'nturb_tracer_tot /= 0' at the C boundary, where the tuples have no flat
                      representation.

        Both of those guard a path INTO the granule. This one guards the granule itself, so it
        is the one a green-line driver, a standalone experiment or a second wrapper -- anything
        that builds the state containers directly -- still runs into.

        ALL THREE COME OUT TOGETHER when the tracers are implemented; none of them is a
        placeholder for a partial fix. The arity is not the obstacle: 'ndtr' is constant for a
        run, and `Turbulence.__init__` runs '_setup_vertdiff_programs' after it, so a tuple of
        exactly 'ndtr' fields can be allocated and the scans compiled for that width -- the
        tuple width being fixed at compile time is not in conflict with 'ndtr' being a runtime
        number, because the compile happens later. Diffusing them is therefore a change to the
        BODY of this method -- 'ndtr' further `DiffusedVariable` entries through the scalar
        matrix, which is what the Fortran does too: 'ndiff = nmvar + ndtr'
        (turb_vertdiff.f90:423), the tracers are entries 'liq+1..liq+ndtr' of the same 'dvar'
        list (:489-504), and one loop diffuses all of them (:566-838). What is still open is
        on the wrapper's side only: py2fgen renders a fixed argument list, so
        'turbulence_run' cannot take 'ndtr' field pointers, and the clean answer there is one
        rank-3 '(cells, levels, ndtr)' array per direction sliced into a tuple on the Python
        side.

        Args:
            input_state: The atmospheric column. Read-only. 'tracers' must be empty: it is not
                diffused here and is refused rather than ignored, see above.
            surface_state: The grid-mean surface state. Read-only.
            diagnostic_state: The turbulence diagnostics; 'rhon' is read and written, the
                diffusion coefficients, the transfer velocities and the two surface flux
                densities are read.
            tendency_state: Where the tendencies go, accumulated onto what is already there,
                exactly as the Fortran's 'INTENT(INOUT)' '*_tens' arguments are.
                'ddt_tracers' must be empty, as 'input_state.tracers'.
            dt_var: The time step of the diffusion equation [s], ICON's 'dt_var'. The interface
                passes 'tcall_turb_jg' for this and for 'dt_tke' alike.

        Raises:
            NotImplementedError: If either tracer tuple is non-empty.
        """
        # Before anything is computed, and before anything else is read: the condition depends
        # on neither the grid nor the configuration, and the alternative to refusing it is a
        # forecast that is quietly missing the diffusion of every tracer it was handed.
        if input_state.tracers or tendency_state.ddt_tracers:
            raise NotImplementedError(
                f"'run_vertdiff' does not diffuse tracers: 'input_state.tracers' and "
                f"'tendency_state.ddt_tracers' are read nowhere in the granule, so they would "
                f"be silently ignored. Got {len(input_state.tracers)} tracers and "
                f"{len(tendency_state.ddt_tracers)} tracer tendencies. Pass empty tuples, or "
                f"use the Fortran scheme."
            )

        reciprocal_time_step = 1.0 / dt_var  # 'fakt = z1/dt_var' (turb_vertdiff.f90:513)
        self._prepare_the_diffusion_matrix(
            input_state=input_state,
            surface_state=surface_state,
            diagnostic_state=diagnostic_state,
            reciprocal_time_step=reciprocal_time_step,
        )
        momentum = (
            DiffusedVariable(
                profile=input_state.u,
                tendency=tendency_state.ddt_u,
                right_hand_side=self._gradient_zonal_wind,
            ),
            DiffusedVariable(
                profile=input_state.v,
                tendency=tendency_state.ddt_v,
                right_hand_side=self._gradient_meridional_wind,
            ),
        )
        scalars = (
            DiffusedVariable(
                profile=input_state.t,
                tendency=tendency_state.ddt_t,
                right_hand_side=self._gradient_liquid_water_potential_temperature,
                has_a_prescribed_surface_flux=True,
                is_potential_temperature=True,
            ),
            DiffusedVariable(
                profile=input_state.qv,
                tendency=tendency_state.ddt_qv,
                right_hand_side=self._gradient_total_water,
                has_a_prescribed_surface_flux=True,
            ),
            DiffusedVariable(
                profile=input_state.qc,
                tendency=tendency_state.ddt_qc,
                right_hand_side=self._gradient_liquid_water,
            ),
        )
        for coefficient, velocity, variables, surface_flux_condition in (
            (diagnostic_state.tkvm, diagnostic_state.tvm, momentum, False),
            (diagnostic_state.tkvh, diagnostic_state.tvh, scalars, self._config.lsflcnd),
        ):
            self._factorise_one_variable_type(
                diffusion_coefficient=coefficient,
                transfer_velocity=velocity,
                air_density=diagnostic_state.rhon,
                surface_flux_condition=surface_flux_condition,
            )
            for variable in variables:
                self._diffuse_one_variable(
                    variable=variable,
                    exner_factor=input_state.epr,
                    surface_flux_condition=surface_flux_condition,
                    reciprocal_time_step=reciprocal_time_step,
                )

    # ---------------------------------------------------------------------- the granule ---

    def run(
        self,
        *,
        input_state: states.TurbulenceInputState,
        surface_state: states.TurbulenceSurfaceState,
        diagnostic_state: states.TurbulenceDiagnosticState,
        tendency_state: states.TurbulenceTendencyState,
        dt_var: float,
        dt_tke: float,
    ) -> None:
        """Run the atmospheric turbulence of one time step: 'turbdiff', then 'vertdiff'.

        THIS IS THE UNIT THE ICON INTERFACE SUBSTITUTES. 'mo_nwp_turbdiff_interface.f90' calls
        'turbdiff' at :576 and 'vertdiff' at :672 with nothing between them but a timer, and
        the two calls share 'rhon', 'zvari', 'tkvm', 'tkvh', 'tvm' and 'tvh'. Verified against
        the capture: every field the two stages have in common is bit-identical at
        'turbdiff-exit' and at 'vertdiff-entry', on all four dates.

        'run_turbtran' is not part of this. ICON calls it from a different interface
        ('mo_nwp_turbtrans_interface.f90'), once per surface tile, before the surface scheme;
        it is phase 3 of the port and it is not a third line of this method.

        NO 'lini' HERE EITHER. See the class docstring: the initialisation is a different
        computation reached from a different call site, and it will be a method of its own.

        THE TRACER REFUSAL IS NOT REPEATED HERE. It belongs to `run_vertdiff`, the stage that
        has 'ptr(:)' and the stage that would drop it, and this method reaches it by
        delegation. The cost is that a call with a non-empty tracer tuple runs 'turbdiff'
        before being refused, which is a diagnostic on an already-fatal path; what it buys is
        one refusal rather than two, and one that `run_vertdiff` called directly -- as the
        second-stage tests call it -- runs into as well.

        Args:
            input_state: The atmospheric column and the external forcings. Read-only. Its
                'tracers' must be empty; see `run_vertdiff`.
            surface_state: The grid-mean surface state. Read-only.
            diagnostic_state: The turbulence diagnostics; read and written by both stages.
            tendency_state: Where the tendencies go. 'ddt_tke' is read on entry as the
                advection tendency and overwritten; the other five are accumulated onto. Its
                'ddt_tracers' must be empty; see `run_vertdiff`.
            dt_var: The time step of the vertical diffusion [s].
            dt_tke: The time step of the TKE equation [s]. ICON passes 'tcall_turb_jg' for
                this and for 'dt_var' alike, but the Fortran keeps them apart and so does this.

        Raises:
            NotImplementedError: If either tracer tuple is non-empty; raised by `run_vertdiff`.
        """
        self.run_turbdiff(
            input_state=input_state,
            surface_state=surface_state,
            diagnostic_state=diagnostic_state,
            tendency_state=tendency_state,
            dt_tke=dt_tke,
        )
        self.run_vertdiff(
            input_state=input_state,
            surface_state=surface_state,
            diagnostic_state=diagnostic_state,
            tendency_state=tendency_state,
            dt_var=dt_var,
        )


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


# --------------------------------------------------------------- absolute vertical levels ---
#
# The four helpers below address the whole vertical axis rather than a relative offset, which
# is something GT4Py deliberately cannot express: an offset is always relative to the row being
# computed, so a value the Fortran reads at a fixed 'k' -- 'tkvm(:,ke1)', 'zvari(:,ke,tet_l)' --
# has to reach a stencil as a cell field the caller prepared. 'tests/turbulence/utils.py'
# solves the same problem with 'surface_row', by way of the host; these write into fields the
# granule allocated once, so nothing leaves the device.
#
# They are also what carries the Fortran's in-place storage reuse across the places where the
# port has to compute out of place. `Turbulence._allocate_local_fields` lists which those are;
# each call site below says which Fortran statement it stands in for.
#
# NONE OF THEM IS CLAMPED TO EITHER ARRAY'S OWN LENGTH. A field the granule allocated is
# exactly 'num_cells' wide; a field the caller supplied is 'nproma' wide when the caller is
# ICON, because it is a '(:,:,jb)' slice of '(nproma, nlev, nblks)' -- and no ICON configuration
# makes the two equal, since 'icon4py_init' requires 'nproma >= n_patch_edges' and edges always
# outnumber cells. Either kind appears on either side of these copies, so a shared length has to
# be passed in, and an unclamped assignment raised a broadcast error the first time this granule
# ran inside ICON. The rows past 'num_cells' are ICON's block padding: no grid point is there,
# ICON leaves them undefined, and the column window the scheme computes never reaches them -- so
# they must be neither read nor written.
#
# THREE OF THEM TAKE 'num_cells' AND '_copy_levels' TAKES THE COLUMN WINDOW, which is narrower.
# 'num_cells' is wide enough wherever the write lands in a field the granule owns, because
# nothing outside the window is ever read back out of one. It is NOT wide enough for the one
# copy whose source is a granule field and whose target is the caller's: the columns between the
# window and 'num_cells' are real grid points -- the lateral boundary and the halo -- which ICON
# owns and which 'turbdiff' deliberately leaves alone, and the granule has nothing to put there
# but the untouched rows of its own working field.
#
# Copies, not views, because these carry values between two fields that both already
# exist; the one place a view is right is 'hhl(:,ke1)', in
# `Turbulence._derive_what_depends_only_on_the_grid`.


def _extract_level(source: gtx.Field, level: int, target: gtx.Field, num_cells: int) -> None:
    """Copy one vertical level of a (Cell, K) field into a cell field."""
    target.ndarray[:num_cells] = source.ndarray[:num_cells, level]


def _set_level(source: gtx.Field, level: int, target: gtx.Field, num_cells: int) -> None:
    """Write a cell field into one vertical level of a (Cell, K) field."""
    target.ndarray[:num_cells, level] = source.ndarray[:num_cells]


def _copy_level(source: gtx.Field, level: int, target: gtx.Field, num_cells: int) -> None:
    """Copy one vertical level from one (Cell, K) field to another."""
    target.ndarray[:num_cells, level] = source.ndarray[:num_cells, level]


def _copy_levels(source: gtx.Field, levels: slice, target: gtx.Field, columns: slice) -> None:
    """Copy a range of vertical levels from one (Cell, K) field to another, over 'columns'.

    Takes the column window rather than 'num_cells', unlike the three helpers above: its target
    is a field the caller owns and its source is one the granule computes only inside that
    window, so writing the full width would hand ICON the working field's untouched rows.
    """
    target.ndarray[columns, levels] = source.ndarray[columns, levels]
