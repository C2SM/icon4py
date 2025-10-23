import dataclasses

from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.common.config import reader as config_reader
from icon4py.model.common import constants

@dataclasses.dataclass
class NonHydrostaticConfig:
    """
    Contains necessary parameter to configure a nonhydro run.

    Encapsulates namelist parameters and derived parameters.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    """

    # number of dynamics substep -> TODO (@halungge): should this really be here?
    n_dyn_substep: int = 5


    itime_scheme: dycore_states.TimeSteppingScheme = dycore_states.TimeSteppingScheme.MOST_EFFICIENT

    #: Miura scheme for advection of rho and theta
    iadv_rhotheta: dycore_states.RhoThetaAdvectionType = dycore_states.RhoThetaAdvectionType.MIURA

    #: Use truly horizontal pressure-gradient computation to ensure numerical
    #: stability without heavy orography smoothing
    igradp_method: dycore_states.HorizontalPressureDiscretizationType = dycore_states.HorizontalPressureDiscretizationType.TAYLOR_HYDRO

    #: type of Rayleigh damping
    rayleigh_type: constants.RayleighType = constants.RayleighType.KLEMP

    #: Rayleigh coefficient
    #: used for calculation of rayleigh_w, rayleigh_vn in mo_vertical_grid.f90
    rayleigh_coeff: float = 0.05


    #: type of divergence damping
    divdamp_type: dycore_states.DivergenceDampingType = dycore_states.DivergenceDampingType.THREE_DIMENSIONAL

    #: order of divergence damping
    divdamp_order: dycore_states.DivergenceDampingOrder = dycore_states.DivergenceDampingOrder.COMBINED  # the ICON default is 4,

    #: Lower and upper bound of transition zone between 2D and 3D divergence damping in case of divdamp_type = 32 [m]
    divdamp_trans_start: float = 12500.0
    divdamp_trans_end: float = 17500.0
    #: scaling factor for divergence damping

    #: Declared as divdamp_fac in ICON. It is a scaling factor for fourth order divergence damping between
    #: heights of fourth_order_divdamp_z and fourth_order_divdamp_z2.
    fourth_order_divdamp_factor: float = 0.0025

    #: Declared as divdamp_fac2 in ICON. It is a scaling factor for fourth order divergence damping between
    #: heights of fourth_order_divdamp_z and fourth_order_divdamp_z2. Divergence damping factor reaches
    #: fourth_order_divdamp_factor2 at fourth_order_divdamp_z2.
    fourth_order_divdamp_factor2: float = 0.004

    #: Declared as divdamp_fac3 in ICON. It is a scaling factor to determine the quadratic vertical
    #: profile of fourth order divergence damping factor between heights of fourth_order_divdamp_z2
    #: and fourth_order_divdamp_z4.
    fourth_order_divdamp_factor3: float = 0.004

    #: Declared as divdamp_fac4 in ICON. It is a scaling factor to determine the quadratic vertical
    #: profile of fourth order divergence damping factor between heights of fourth_order_divdamp_z2
    #: and fourth_order_divdamp_z4. Divergence damping factor reaches fourth_order_divdamp_factor4
    #: at fourth_order_divdamp_z4.
    fourth_order_divdamp_factor4: float = 0.004

    #: Declared as divdamp_z in ICON. The upper limit in height where divergence damping factor is a constant.
    fourth_order_divdamp_z: float = 32500.0

    #: Declared as divdamp_z2 in ICON. The upper limit in height above fourth_order_divdamp_z where divergence
    #: damping factor decreases as a linear function of height.
    fourth_order_divdamp_z2: float = 40000.0

    #: Declared as divdamp_z3 in ICON. Am intermediate height between fourth_order_divdamp_z2 and
    #: fourth_order_divdamp_z4 where divergence damping factor decreases quadratically with height.
    fourth_order_divdamp_z3: float = 60000.0

    #: Declared as divdamp_z4 in ICON. The upper limit in height where divergence damping factor decreases
    #: quadratically with height.
    fourth_order_divdamp_z4: float = 80000.0

    #: off-centering for density and potential temperature at interface levels.
    #: Specifying a negative value here reduces the amount of vertical
    #: wind off-centering needed for stability of sound waves.
    rhotheta_offctr: float = -0.1

    #: off-centering of velocity advection in corrector step
    veladv_offctr: float = 0.25

    #: parameters from other namelists:

    #: from mo_initicon_nml.f90/ mo_initicon_config.f90
    #: whether or not incremental analysis update is active
    is_iau_active: bool = False
    #: IAU weight for dynamics fields
    iau_wgt_dyn: float = 0.0

    #: from mo_run_nml.f90
    #: use vertical nesting # TODO (halungge) not supported in icon4py remove!!
    l_vert_nested: bool = dataclasses.field(default=False,metadata={"omegaconf_ignore": True})

    #: from mo_interpol_nml.f90

    #: Parameter describing the lateral boundary nudging in limited area mode.
    #:
    #: Maximal value of the nudging coefficients used cell row bordering the boundary interpolation zone,
    #: from there nudging coefficients decay exponentially with `nudge_efold_width` in units of cell rows.
    #: Called 'nudge_max_coeff' in mo_interpol_nml.f90.
    #: Note: The user can pass the ICON namelist parameter `nudge_max_coeff` as `_nudge_max_coeff` or
    #: the properly scaled one as `max_nudging_coefficient`,
    #: see the comment in mo_interpol_nml.f90
    _nudge_max_coeff: float | None = dataclasses.field(default=None, metadata={"omegaconf_ignore": True})  # default is set in __init__
    max_nudging_coefficient: float | None = None  # default is set in __init__





    def __post_init__(self):
        #: TODO: This code is duplicated in `diffusion.py`, clean this up when implementing proper configuration handling.
        if self._nudge_max_coeff is not None and self.max_nudging_coefficient is not None:
            raise ValueError(
                "Cannot set both '_max_nudging_coefficient' and 'scaled_max_nudging_coefficient'."
            )
        elif self.max_nudging_coefficient is not None:
            self.max_nudging_coefficient: float = self.max_nudging_coefficient
        elif self._nudge_max_coeff is not None:
            self.max_nudging_coefficient: float = (
                constants.DEFAULT_DYNAMICS_TO_PHYSICS_TIMESTEP_RATIO * self._nudge_max_coeff
            )
        else:  # default value in ICON
            self.max_nudging_coefficient: float = (
                constants.DEFAULT_DYNAMICS_TO_PHYSICS_TIMESTEP_RATIO * 0.02
            )

        self._validate()

    def _validate(self):
        """Apply consistency checks and validation on configuration parameters."""

        if self.l_vert_nested:
            raise NotImplementedError("Vertical nesting support not implemented")

        if self.igradp_method != dycore_states.HorizontalPressureDiscretizationType.TAYLOR_HYDRO:
            raise NotImplementedError("igradp_method can only be 3")

        if self.itime_scheme != dycore_states.TimeSteppingScheme.MOST_EFFICIENT:
            raise NotImplementedError("itime_scheme can only be 4")

        if self.iadv_rhotheta != dycore_states.RhoThetaAdvectionType.MIURA:
            raise NotImplementedError("iadv_rhotheta can only be 2 (Miura scheme)")

        if self.divdamp_order != dycore_states.DivergenceDampingOrder.COMBINED:
            raise NotImplementedError("divdamp_order can only be 24")

        if self.divdamp_type == dycore_states.DivergenceDampingType.TWO_DIMENSIONAL:
            raise NotImplementedError(
                "`DivergenceDampingType.TWO_DIMENSIONAL` (2) is not yet implemented"
            )



def init_config() -> config_reader.ConfigReader:
    return config_reader.ConfigReader(NonHydrostaticConfig())
