"""Moist warm-bubble initial condition on a torus (ICON test case ``aes_bubble``)."""

from __future__ import annotations

import dataclasses
import logging
import math
import typing
from typing import TYPE_CHECKING

from icon4py.model.common import constants as phy_const
from icon4py.model.common.config import options as common_conf_opt
from icon4py.model.common.grid import geometry_attributes as geometry_meta, icon as icon_grid
from icon4py.model.common.initial_condition.analytical import utils as testcases_utils
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.physics.thermodynamics import compute_moisture, compute_pressure
from icon4py.model.common.states import prognostic_state as prognostics, tracer_states
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    from icon4py.model.common.initial_condition.config import ConfigContext
    from icon4py.model.common.states import static_fields

log = logging.getLogger(__name__)

# Conversion from the full width at half maximum of a Gaussian to its standard deviation.
_FWHM_TO_SIGMA: typing.Final[float] = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


@dataclasses.dataclass
class AesBubbleConfig:
    # The default values are from init_aes_bubble_config in mo_aes_bubble_config.f90; the
    # name of the corresponding member of aes_bubble_config is given in brackets.
    surface_pressure: typing.Annotated[
        float, common_conf_opt.ConfigOption(description="Surface pressure [Pa] (psfc).")
    ] = 101325.0
    minimum_temperature: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Lower bound of the background temperature profile [K] (t_am)."
        ),
    ] = 180.0
    surface_temperature: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(description="Background surface temperature [K] (t0)."),
    ] = 303.15
    boundary_layer_lapse_rate: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Temperature lapse rate below boundary_layer_height [K/m] (gamma0)."
        ),
    ] = 0.009
    boundary_layer_height: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Height at which the lapse rate changes [m] (z0).",
        ),
    ] = 3000.0
    free_atmosphere_lapse_rate: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Temperature lapse rate above boundary_layer_height [K/m] (gamma1)."
        ),
    ] = 0.00001
    bubble_temperature_amplitude: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Temperature perturbation at the centre of the bubble [K] (t_perturb)."
        ),
    ] = 3.0
    background_relative_humidity: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Relative humidity outside of the bubble (relhum_bg).",
        ),
    ] = 0.7
    bubble_relative_humidity: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Relative humidity at the centre of the bubble (relhum_mx).",
        ),
    ] = 0.95
    bubble_horizontal_width: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Full width at half maximum of the horizontal Gaussian profile [m] (hw_x)."
        ),
    ] = 12500.0
    bubble_vertical_width: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description=(
                "Full width at half maximum of the vertical Gaussian profile, which is "
                "centred at the ground [m] (hw_z)."
            )
        ),
    ] = 500.0
    bubble_center_x: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="X offset of the bubble centre from the domain centre [m] (x_center)."
        ),
    ] = 0.0
    gaussian_in_y: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description=(
                "Make the bubble Gaussian in y as well (a 3D bubble) instead of a ridge "
                "that is uniform in y (lgaussxy)."
            )
        ),
    ] = False

    def __post_init__(self) -> None:
        # Checks of eval_aes_bubble_config (mo_aes_bubble_config.f90). The widths must
        # be strictly positive here, as a zero width divides by zero in the Gaussian.
        checks = {
            "minimum_temperature": self.minimum_temperature > 0.0,
            "surface_pressure": self.surface_pressure > 0.0,
            "boundary_layer_lapse_rate": self.boundary_layer_lapse_rate >= 0.0,
            "free_atmosphere_lapse_rate": self.free_atmosphere_lapse_rate >= 0.0,
            "boundary_layer_height": self.boundary_layer_height > 0.0,
            "bubble_temperature_amplitude": self.bubble_temperature_amplitude >= 0.0,
            "background_relative_humidity": 0.0 <= self.background_relative_humidity <= 1.0,
            "bubble_relative_humidity": 0.0 <= self.bubble_relative_humidity <= 1.0,
            "bubble_horizontal_width": self.bubble_horizontal_width > 0.0,
            "bubble_vertical_width": self.bubble_vertical_width > 0.0,
        }
        if invalid := [name for name, is_valid in checks.items() if not is_valid]:
            raise ValueError(
                f"Invalid 'AesBubbleConfig' values for: {', '.join(invalid)} (see the field descriptions)."
            )


def _gaussian(x: data_alloc.NDArray, center: float, sigma: float) -> data_alloc.NDArray:
    """Gaussian with unit amplitude, ``gaussian(mu, sigma, 1, x)`` in mo_aes_bubble.f90."""
    array_ns = data_alloc.array_namespace(x)
    normalized = (x - center) / sigma
    return array_ns.exp(-0.5 * normalized * normalized)


def compute_first_guess_profiles(
    *,
    ic_config: AesBubbleConfig,
    z_mc: data_alloc.NDArray,
    horizontal_weight: data_alloc.NDArray,
) -> tuple[data_alloc.NDArray, ...]:
    """
    Background state with the bubble, integrated hydrostatically from the surface.

    The temperature and relative humidity profiles are prescribed: a piecewise linear
    temperature profile, bounded below by ``minimum_temperature``, plus a perturbation
    that is Gaussian in the horizontal (``horizontal_weight``) and in height above
    ground, which also blends the relative humidity from its background to its bubble
    value. The pressure is integrated from ``surface_pressure`` with the layer-mean
    temperature. This is the first part of init_aes_bubble (mo_aes_bubble.f90), before
    hydro_adjust_iterative.

    Args:
        ic_config: configuration of the test case
        z_mc: height of the full levels [m], levels ordered top to bottom
        horizontal_weight: horizontal Gaussian weight of each cell, in [0, 1]

    Returns:
        temperature [K], relative humidity, exner, theta_v [K], rho [kg/m3] and
        qv [kg/kg], all on (cells, levels)
    """
    array_ns = data_alloc.array_namespace(z_mc)
    num_levels = z_mc.shape[1]
    sigma_z = ic_config.bubble_vertical_width * _FWHM_TO_SIGMA
    t0 = ic_config.surface_temperature
    z0 = ic_config.boundary_layer_height
    t_perturb = ic_config.bubble_temperature_amplitude
    rh_bg = ic_config.background_relative_humidity
    rh_bubble = ic_config.bubble_relative_humidity

    temperature = array_ns.zeros_like(z_mc)
    relative_humidity = array_ns.zeros_like(z_mc)
    exner = array_ns.zeros_like(z_mc)
    theta_v = array_ns.zeros_like(z_mc)
    rho = array_ns.zeros_like(z_mc)
    qv = array_ns.zeros_like(z_mc)

    # Temperature and pressure at the lower boundary of the current layer.
    temperature_below = t0 + t_perturb * horizontal_weight
    pressure = array_ns.full_like(horizontal_weight, ic_config.surface_pressure)
    for k in range(num_levels - 1, -1, -1):
        z = z_mc[:, k]
        dz = z if k == num_levels - 1 else z - z_mc[:, k + 1]
        bubble_weight = horizontal_weight * _gaussian(z, 0.0, sigma_z)
        background_temperature = array_ns.where(
            z <= z0,
            t0 - ic_config.boundary_layer_lapse_rate * z,
            t0
            - ic_config.boundary_layer_lapse_rate * z0
            - ic_config.free_atmosphere_lapse_rate * (z - z0),
        )
        temperature[:, k] = (
            array_ns.maximum(background_temperature, ic_config.minimum_temperature)
            + t_perturb * bubble_weight
        )
        saturation_vapor_pressure = compute_pressure.sat_pres_water(temperature[:, k])
        vapor_pressure = saturation_vapor_pressure * ((rh_bubble - rh_bg) * bubble_weight + rh_bg)
        layer_mean_temperature = 0.5 * (temperature_below + temperature[:, k])
        pressure = pressure * array_ns.exp(
            -phy_const.GRAV / phy_const.RD / layer_mean_temperature * dz
        )
        temperature_below = temperature[:, k]
        virtual_temperature = temperature[:, k] / (
            1.0 - (vapor_pressure / pressure) * (1.0 - phy_const.RD_O_RV)
        )
        relative_humidity[:, k] = vapor_pressure / saturation_vapor_pressure
        exner[:, k] = (pressure / phy_const.P0REF) ** phy_const.RD_O_CPD
        rho[:, k] = pressure / phy_const.RD / virtual_temperature
        theta_v[:, k] = virtual_temperature / exner[:, k]
        qv[:, k] = compute_moisture.specific_humidity(vapor_pressure, pressure)

    return temperature, relative_humidity, exner, theta_v, rho, qv


def aes_bubble(
    *,
    config: ConfigContext,
    grid: icon_grid.IconGrid,
    static_fields: static_fields.StaticFieldFactories,
    prognostic_state_now: prognostics.PrognosticState,
    tracer_state_now: tracer_states.TracerState,
) -> None:
    """
    Initial condition for the moist warm-bubble test case on a torus (``aes_bubble``).

    An atmosphere at rest with a piecewise linear temperature profile is perturbed by
    a warm and moist bubble at the ground, which is Gaussian in height and in x (and
    optionally in y). The state is first integrated hydrostatically from the surface
    and then balanced to the discretization of the dynamical core, which requires the
    water-vapour tracer ``qv`` to be active; the other tracers stay zero.

    Port of init_aes_bubble in mo_aes_bubble.f90.
    """
    ic_config = config.initial_condition
    assert isinstance(ic_config, AesBubbleConfig)
    if grid.geometry_type != icon_grid.GeometryType.TORUS:
        raise NotImplementedError(
            f"The 'aes_bubble' initial condition requires a torus grid, got '{grid.geometry_type}'."
        )
    if tracer_state_now.qv is None:
        raise ValueError(
            "The 'aes_bubble' initial condition requires the 'qv' tracer to be active."
        )

    geometry = static_fields.geometry
    metrics = static_fields.metrics
    cell_x = geometry.get_full_precision(geometry_meta.CELL_CENTER_X).ndarray
    cell_y = geometry.get_full_precision(geometry_meta.CELL_CENTER_Y).ndarray
    z_mc = metrics.get_full_precision(metrics_attributes.Z_MC).ndarray

    # As in ICON, the bubble is placed relative to the domain centre of the grid file, and
    # the Gaussian in y has the same width as in x. The Gaussians are not periodic.
    domain_center = grid.grid_params.domain_center
    assert domain_center is not None
    sigma_x = ic_config.bubble_horizontal_width * _FWHM_TO_SIGMA
    horizontal_weight = _gaussian(cell_x, domain_center[0] + ic_config.bubble_center_x, sigma_x)
    if ic_config.gaussian_in_y:
        horizontal_weight = horizontal_weight * _gaussian(cell_y, domain_center[1], sigma_x)

    temperature, relative_humidity, exner, theta_v, rho, qv = compute_first_guess_profiles(
        ic_config=ic_config, z_mc=z_mc, horizontal_weight=horizontal_weight
    )
    log.info("AES bubble first-guess profiles completed.")

    testcases_utils.hydrostatic_adjustment_iterative_ndarray(
        temperature=temperature,
        relative_humidity=relative_humidity,
        wgtfac_c=metrics.get_full_precision(metrics_attributes.WGTFAC_C).ndarray,
        ddqz_z_half=metrics.get_full_precision(metrics_attributes.DDQZ_Z_HALF).ndarray,
        exner_ref_mc=metrics.get_full_precision(metrics_attributes.EXNER_REF_MC).ndarray,
        d_exner_dz_ref_ic=metrics.get_full_precision(metrics_attributes.D_EXNER_DZ_REF_IC).ndarray,
        theta_ref_mc=metrics.get_full_precision(metrics_attributes.THETA_REF_MC).ndarray,
        exner=exner,
        theta_v=theta_v,
        rho=rho,
        qv=qv,
        num_levels=grid.num_levels,
    )
    log.info("AES bubble hydrostatic adjustment completed.")

    prognostic_state_now.exner.ndarray[:, :] = exner
    prognostic_state_now.theta_v.ndarray[:, :] = theta_v
    prognostic_state_now.rho.ndarray[:, :] = rho
    prognostic_state_now.vn.ndarray[:, :] = 0.0
    prognostic_state_now.w.ndarray[:, :] = 0.0
    tracer_state_now.qv.ndarray[:, :] = qv
    # ICON sets all tracers to zero before initializing qv.
    for tracer in tracer_state_now.active_fields():
        if tracer.name != "qv":
            tracer.field.ndarray[:, :] = 0.0
