import dataclasses
import functools
from typing import TypeVar, Generic

from icon4py.model.atmosphere.diffusion.diffusion import TurbulenceShearForcingType, DiffusionType
from icon4py.model.common import constants


@dataclasses.dataclass
class DiffusionConfig:
    """
    Contains necessary parameter to configure a diffusion run.

    Encapsulates namelist parameters and derived parameters.
    Values should be read from configuration.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    """

    # parameters from namelist diffusion_nml
    diffusion_type: DiffusionType = DiffusionType.SMAGORINSKY_4TH_ORDER

    #: If True, apply diffusion on the vertical wind field
    #: Called 'lhdiff_w' in mo_diffusion_nml.f90
    apply_to_vertical_wind:bool = True

    #: True apply diffusion on the horizontal wind field, is ONLY used in mo_nh_stepping.f90
    #: Called 'lhdiff_vn' in mo_diffusion_nml.f90
    apply_to_horizontal_wind:bool = True

    #:  If True, apply horizontal diffusion to temperature field
    #: Called 'lhdiff_temp' in mo_diffusion_nml.f90
    apply_to_temperature:bool = True

    #: Options for discretizing the Smagorinsky momentum diffusion
    #: Called 'itype_vn_diffu' in mo_diffusion_nml.f90
    type_vn_diffu: int = 1
    #: If True, compute 3D Smagorinsky diffusion coefficient
    #: Called 'lsmag_3d' in mo_diffusion_nml.f90
    compute_3d_smag_coeff: bool = False

    #: Options for discretizing the Smagorinsky temperature diffusion
    #: Called 'itype_t_diffu' in mo_diffusion_nml.f90
    type_t_diffu: int = 2

    #: Ratio of e-folding time to (2*)time step
    #: Called 'hdiff_efdt_ratio' in mo_diffusion_nml.f90
    hdiff_efdt_ratio: float = 36.0

    #: Ratio of e-folding time to time step for w diffusion (NH only)
    #: Called 'hdiff_w_efdt_ratio' in mo_diffusion_nml.f90.
    hdiff_w_efdt_ratio: float = 15.0

    #: Scaling factor for Smagorinsky diffusion at height hdiff_smag_z and below
    #: Called 'hdiff_smag_fac' in mo_diffusion_nml.f90
    smagorinski_scaling_factor: float = 0.015

    ## parameters from other namelists


    # TODO (@halungge): interpolation??
    # from mo_nonhydrostatic_nml.f90
    n_substeps: int = 5

    #: If True, apply truly horizontal temperature diffusion over steep slopes
    #: Called 'l_zdiffu_t' in mo_nonhydrostatic_nml.f90
    apply_zdiffusion_t: bool = True

    #:slope threshold (temperature diffusion): is used to build up an index list for application of truly horizontal diffusion in mo_vertical_grid.f90
    thslp_zdiffu: float = 0.025
    #: threshold [m] for height difference between adjacent grid points, defaults to 200m (temperature diffusion)
    thhgtd_zdiffu: float = 200.0

    # namelist mo_gridref_nml.f90

    #: Denominator for temperature boundary diffusion
    #: Called 'denom_diffu_t' in mo_gridref_nml.f90
    temperature_boundary_diffusion_denom: float = 135.0

    #: Denominator for velocity boundary diffusion
    #: Called 'denom_diffu_v' in mo_gridref_nml.f90
    velocity_boundary_diffusion_denom: float = 200.0

    # parameters from namelist: mo_interpol_nml.f90

    #: Parameter describing the lateral boundary nudging in limited area mode.
    #:
    #: Maximal value of the nudging coefficients used cell row bordering the boundary interpolation zone,
    #: from there nudging coefficients decay exponentially with `nudge_efold_width` in units of cell rows.
    #: Called 'nudge_max_coeff' in mo_interpol_nml.f90.
    #: Note: The user can pass the ICON namelist paramter `nudge_max_coeff` as `_nudge_max_coeff` or
    #: the properly scaled one as `max_nudging_coefficient`,
    #: see the comment in mo_interpol_nml.f90
    _nudge_max_coeff: float | None = None
    max_nudging_coefficient: float | None = 0.1

    #: Exponential decay rate (in units of cell rows) of the lateral boundary nudging coefficients
    #: Called 'nudge_efold_width' in mo_interpol_nml.f90
    nudging_decay_rate: float = 2.0

    #: Type of shear forcing used in turbulence
    #: Called 'itype_shear' in mo_turbdiff_nml.f90
    shear_type: TurbulenceShearForcingType = TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND

    #: Consider separate horizontal shear production in TKE-equation.
    #: Called 'ltkeshs' in mo_turbdiff_nml.f90
    ltkeshs: bool = True





    def __post_init__(self):
        #: TODO: This code is duplicated in `solve_nonhydro.py`, clean this up when implementing proper configuration handling.
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

    def _validate(self):
        """Apply consistency checks and validation on configuration parameters."""
        if self.diffusion_type != 5:
            raise NotImplementedError(
                "Only diffusion type 5 = `Smagorinsky diffusion with fourth-order background "
                "diffusion` is implemented"
            )

        if self.diffusion_type < 0:
            self.apply_to_temperature = False
            self.apply_to_horizontal_wind = False
            self.apply_to_vertical_wind = False

        if self.shear_type not in (
            TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND,
            TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND,
        ):
            raise NotImplementedError(
                f"Turbulence Shear only {TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND} "
                f"and {TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND} "
                f"implemented"
            )

    @functools.cached_property
    def substep_as_float(self):
        return float(self.ndyn_substeps)




