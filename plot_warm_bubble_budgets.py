#!/usr/bin/env python
"""Mass and energy budgets of the warm bubble experiment as time series.

Mirrors the "Tracking mass" and "Tracking energy" analyses of the ICON bubble test
(https://easy.gems.dkrz.de/simulations/ICON/bubble.html), using icon4py output:

  left : change of the column-integrated total air mass  M = int rho dz  (rho is the
         density of moist air, i.e. including vapour and condensates), of the total
         water  W = int q_t rho dz, and the precipitation that left the domain through
         the surface; per output interval in mass_energy_budgets.png, and accumulated
         since the first output in mass_total_energy_budgets.png;
  middle: total energy, total energy plus the energy carried out of the domain by
         precipitation (which should be ~0 if energy is conserved), and that surface
         energy flux;
  right: internal, potential and kinetic energy;
         energy panels as tendencies (W m-2) in mass_energy_budgets.png, and as changes
         since the first output (J m-2) in mass_total_energy_budgets.png.

All quantities are horizontal (cell-area weighted) means of vertical integrals, i.e.
per unit surface area. The internal energy uses the same definition as icon4py's
physics (common/physics/thermodynamics/compute_energy.py):
    e_i = cv T - q_liq lvc - q_ice lsc,
    cv  = cvd (1 - q_t) + cvv q_v + cpl q_liq + ci q_ice,
with q_liq = qc + qr and q_ice = qi + qs + qg. Temperature is diagnosed from theta_v
and exner like icon4py's 'compute_virtual_temperature_and_temperature'.

Inputs (from the driver's output directory):
  - the 3D group (default 'winds') with air_density, exner_function,
    virtual_potential_temperature, eastward/northward/upward wind and qv..qg
  - optionally the 'precipitation' group with rainfall/snowfall/graupelfall/icefall
    fluxes and precipitation_energy_flux; their lowest level is the flux through the
    surface. Without it, only the uncorrected curves are drawn. The surface flux is
    integrated over each output interval from the precipitation samples, so it is
    exact only if the precipitation group is written every time step.

Usage (inside the uenv, with the icon4py venv activated):
    python plot_warm_bubble_budgets.py output_sample
    python plot_warm_bubble_budgets.py output_sample --csv
    python plot_warm_bubble_budgets.py output_sample --every 10   # every 10th output
"""

import argparse
import pathlib

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from icon4py.model.common import constants as c  # noqa: E402
from plot_warm_bubble import open_output  # noqa: E402


SURFACE_WATER_FLUXES = ("rainfall_flux", "snowfall_flux", "graupelfall_flux", "icefall_flux")
SURFACE_ENERGY_FLUX = "precipitation_energy_flux"


def set_fonts(size: float) -> None:
    """Base font size; titles, labels, ticks and legends scale with it."""
    plt.rcParams.update({
        "font.size": size,
        "axes.titlesize": size,
        "axes.labelsize": size,
        "xtick.labelsize": 0.9 * size,
        "ytick.labelsize": 0.9 * size,
        "legend.fontsize": 0.8 * size,
        "lines.linewidth": 2.0,
        "lines.markersize": 8.0,
    })


def layer_thickness(data: xr.Dataset) -> np.ndarray:
    """Thickness (m) of the full levels from the half-level heights (top to bottom)."""
    h = data["height"].values
    return h[:-1] - h[1:]


def full_level_heights(data: xr.Dataset) -> np.ndarray:
    h = data["height"].values
    return 0.5 * (h[:-1] + h[1:])


def column_budgets(data: xr.Dataset, it: int, dz: np.ndarray, z: np.ndarray, weights: np.ndarray) -> dict:
    """Area-weighted means of the vertical integrals at output time 'it' (per m2)."""

    def get(name: str) -> np.ndarray:
        return data[name].isel(time=it).values  # (level, cell)

    rho = get("air_density")
    # if no tracers, set them to zero arrays of the right shape
    qv, qc, qi, qr, qs, qg = (get(q) if q in data else np.zeros_like(rho) for q in ("qv", "qc", "qi", "qr", "qs", "qg"))
    q_liq = qc + qr
    q_ice = qi + qs + qg
    q_tot = qv + q_liq + q_ice

    
    # temperature as in icon4py's compute_virtual_temperature_and_temperature
    t_v = get("virtual_potential_temperature") * get("exner_function")
    t = t_v / (1.0 + c.RV_O_RD_MINUS_1 * qv - (q_liq + q_ice))

    w_half = get("upward_air_velocity")
    w = 0.5 * (w_half[:-1] + w_half[1:])
    u = get("eastward_wind")
    v = get("northward_wind")

    m = rho * dz[:, None]  # air mass per area of each layer (kg m-2)
    c_v = c.CVD * (1.0 - q_tot) + c.CVV * qv + c.CPL * q_liq + c.SPECIFIC_HEAT_CAPACITY_ICE_THERMO * q_ice
    e_int = (c_v * t - q_liq * c.LATENT_HEAT_FOR_VAPORISATION_INVARIANT - q_ice * c.LATENT_HEAT_FOR_SUBLIMATION_INVARIANT) * m
    e_pot = c.GRAV * z[:, None] * m
    e_kin = 0.5 * (u**2 + v**2 + w**2) * m

    def area_mean(column_field: np.ndarray) -> float:
        return float(np.sum(column_field.sum(axis=0) * weights))

    print(f"max absolute kinetic energy: {np.max(np.abs(area_mean(e_kin))):.3e} J m-2")

    return {
        "mass": area_mean(m),
        "water": area_mean(q_tot * m),
        "internal_energy": area_mean(e_int),
        "potential_energy": area_mean(e_pot),
        "kinetic_energy": area_mean(e_kin),
    }


def surface_outflow(precip: xr.Dataset, t_out: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Water (kg m-2) and energy (J m-2) leaving through the surface in each output interval.

    The precipitation samples at times t with t_out[i-1] < t <= t_out[i] are each
    taken to represent the flux over their own sampling interval.
    """
    t_p = precip["time"].values
    dt_p = np.diff(t_p, prepend=t_p[0] - (t_p[1] - t_p[0]) if t_p.size > 1 else t_p[0])
    lowest = {"level": precip.sizes["level"] - 1}
    water_names = [n for n in SURFACE_WATER_FLUXES if n in precip]
    water_flux = sum(precip[n].isel(**lowest).values for n in water_names) @ weights  # (time,)
    energy_flux = (
        precip[SURFACE_ENERGY_FLUX].isel(**lowest).values @ weights
        if SURFACE_ENERGY_FLUX in precip
        else np.full(t_p.size, np.nan)
    )
    water = np.empty(t_out.size - 1)
    energy = np.empty(t_out.size - 1)
    for i in range(1, t_out.size):
        print(f"surface outflow: output time {i}/{t_out.size - 1}")
        sel = (t_p > t_out[i - 1]) & (t_p <= t_out[i])
        water[i - 1] = np.sum(water_flux[sel] * dt_p[sel])
        energy[i - 1] = np.sum(energy_flux[sel] * dt_p[sel])
    return water, energy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir", type=pathlib.Path, help="driver output directory (e.g. output_sample)")
    p.add_argument("--group", default="winds", help="basename of the field group with the 3D fields")
    p.add_argument("--precip-group", default="precipitation", help="basename of the precipitation group")
    p.add_argument("--plot-dir", type=pathlib.Path, default=None, help="default: <output_dir>/plots")
    p.add_argument("--end-time", type=float, default=None, help="default: until the last output time; if set, only outputs up to this time (min) are read")
    p.add_argument(
        "--every", type=int, default=1, metavar="N",
        help="read only every N-th output time of the 3D group (default: 1, all); the "
        "surface fluxes are still integrated over all precipitation samples",
    )
    p.add_argument("--csv", action="store_true", help="also write the time series as CSV")
    args = p.parse_args()
    if args.every < 1:
        p.error("--every must be >= 1")

    set_fonts(14.0)

    data, grid = open_output(args.output_dir, args.group)
    n_all = data.sizes["time"]
    if args.end_time is not None:
        t_all = data["time"].values
        data = data.isel(time=np.flatnonzero((t_all - t_all[0]) / 60.0 <= args.end_time))
    if args.every > 1:
        data = data.isel(time=slice(None, None, args.every))
    print(f"using {data.sizes['time']} of {n_all} output times (every {args.every})")
    area = grid["cell_area"].values
    weights = area / area.sum()
    dz = layer_thickness(data)
    z = full_level_heights(data)
    t_out = data["time"].values
    t_min = (t_out - t_out[0]) / 60.0
    dt_out = np.diff(t_out)

    series = {k: np.empty(t_out.size) for k in ("mass", "water", "internal_energy", "potential_energy", "kinetic_energy")}
    for it in range(t_out.size):
        for k, val in column_budgets(data, it, dz, z, weights).items():
            series[k][it] = val
        if it % 20 == 0 or it == t_out.size - 1:
            print(f"read output time {it + 1}/{t_out.size}")

    try:
        precip, _ = open_output(args.output_dir, args.precip_group)
        sfc_water, sfc_energy = surface_outflow(precip, t_out, weights)
    except FileNotFoundError:
        print(f"no '{args.precip_group}' output found: plotting without surface fluxes")
        sfc_water = sfc_energy = None

    d_mass = np.diff(series["mass"])
    d_water = np.diff(series["water"])
    d_ie = np.diff(series["internal_energy"]) / dt_out
    d_pe = np.diff(series["potential_energy"]) / dt_out
    d_ke = np.diff(series["kinetic_energy"]) / dt_out
    d_te = d_ie + d_pe + d_ke
    ie = series["internal_energy"]
    pe = series["potential_energy"]
    ke = series["kinetic_energy"]
    te = ie + pe + ke
    t_mid = 0.5 * (t_min[1:] + t_min[:-1])

    def plot_budgets(
        input_t_mass,
        input_t_energy,
        input_mass,
        input_water,
        input_ie,
        input_pe,
        input_ke,
        input_te,
        output_file,
        input_sfc_water=None,
        input_sfc_energy=None,
        cumulative=False,
    ):
        """Mass panel: total air mass M = int rho dz (rho includes vapour and condensates),
        total water W = int q_t rho dz and the surface precipitation (kg m-2).
        Energy panel: total energy E, E plus the energy carried out through the surface by
        precipitation, and that surface energy flux. Components panel: internal, potential
        and kinetic energy.
        By default, changes per output interval and energy tendencies (W m-2).
        With 'cumulative', changes since the first output, with the surface precipitation and
        energy flux accumulated, and E(t) - E(t0) (J m-2)."""
        fig, (ax_m, ax_e, ax_c) = plt.subplots(1, 3, figsize=(16, 3.8), constrained_layout=True)
        if cumulative:
            # m_label = r"$M - M(t_0)$, $M = \int \rho \, \mathrm{d}z$"
            # w_label = r"$W - W(t_0)$, $W = \int q_\mathrm{t} \rho \, \mathrm{d}z$"
            # p_label = r"$\int_{t_0}^{t} P_\mathrm{sfc} \, \mathrm{d}t'$"
            # m_ylabel = r"mass change since $t_0$ / kg m$^{-2}$"
            # e_label = {k: rf"$E_\mathrm{{{k}}} - E_\mathrm{{{k}}}(t_0)$" for k in ("I", "P", "K")}
            # te_label = r"$E - E(t_0)$"
            # te_sfc_label = r"$E - E(t_0) + \int_{t_0}^{t} F_{E,\mathrm{sfc}} \, \mathrm{d}t'$"
            # f_label = r"$\int_{t_0}^{t} F_{E,\mathrm{sfc}} \, \mathrm{d}t'$"
            # e_ylabel = r"energy change since $t_0$ / J m$^{-2}$"
            m_label = r"total $\Delta M$"
            w_label = r"atmospheric $\Delta M$"
            p_label = r"Cumulative precipitation flux"
            m_ylabel = r"mass change (kg m$^{-2}$)"
            e_label = {k: rf"$\Delta E_\mathrm{{{k}}}$" for k in ("I", "P", "K")}
            te_label = r"atmospheric $\Delta E$"
            te_sfc_label = r"total $\Delta E$"
            f_label = r"Cumulative surface energy flux"
            e_ylabel = r"energy change (J m$^{-2}$)"
        else:
            m_label = r"$\Delta M$, $M = \int \rho \, \mathrm{d}z$"
            w_label = r"$\Delta W$, $W = \int q_\mathrm{t} \rho \, \mathrm{d}z$"
            p_label = r"$P_\mathrm{sfc} \, \Delta t$"
            m_ylabel = r"mass change per output interval / kg m$^{-2}$"
            e_label = {k: rf"$\mathrm{{d}}E_\mathrm{{{k}}}/\mathrm{{d}}t$" for k in ("I", "P", "K")}
            te_label = r"$\mathrm{d}E/\mathrm{d}t$"
            te_sfc_label = r"$\mathrm{d}E/\mathrm{d}t + F_{E,\mathrm{sfc}}$"
            f_label = r"$F_{E,\mathrm{sfc}}$"
            e_ylabel = r"power density / W m$^{-2}$"

        ax_m.plot(input_t_mass, input_mass, "k-", lw=1.5, label=m_label)
        ax_m.plot(input_t_mass, input_water, "k--", label=w_label)
        if sfc_water is not None:
            ax_m.plot(input_t_mass, input_sfc_water, color="dodgerblue", lw=1.2, label=p_label)
        ax_m.set_ylabel(m_ylabel)
        ax_m.set_title("mass")

        ax_e.plot(input_t_energy, input_te, "k-", label=te_label)
        if sfc_energy is not None and np.all(np.isfinite(sfc_energy)):
            ax_e.plot(
                input_t_energy, input_te + input_sfc_energy, color="darkorange",
                label=te_sfc_label,
            )
            ax_e.plot(input_t_energy, input_sfc_energy, ":", color="darkorange", label=f_label)
        ax_e.set_ylabel(e_ylabel)
        ax_e.set_title("energy")

        ax_c.plot(input_t_energy, input_ie, color="crimson", label=e_label["I"])
        ax_c.plot(input_t_energy, input_pe, color="dodgerblue", label=e_label["P"])
        ax_c.plot(input_t_energy, input_ke, color="green", label=e_label["K"])
        ax_c.set_ylabel(e_ylabel)
        ax_c.set_title("energy components")

        for ax, ncol in ((ax_m, 1), (ax_e, 1), (ax_c, 1)):
            ax.axhline(0.0, color="0.7", lw=0.6, zorder=0)
            ax.set_xlabel("time / min")
            ax.legend(ncol=ncol, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
            if args.end_time is not None:
                ax.set_xlim(right=args.end_time)

        fig.savefig(output_file, dpi=150)
        print(f"figure: {output_file}")

    plot_dir = args.plot_dir or args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    # water (kg m-2) and energy (J m-2) that have left through the surface since the first
    # output, to add to W(t) - W(t0) and E(t) - E(t0)
    if sfc_water is not None:
        cum_sfc_water = np.concatenate(([0.0], np.cumsum(sfc_water)))
        cum_sfc_energy = np.concatenate(([0.0], np.cumsum(sfc_energy)))
    else:
        cum_sfc_water = cum_sfc_energy = None
    mass = series["mass"]
    water = series["water"]
    plot_budgets(
        input_t_mass=t_mid,
        input_t_energy=t_mid,
        input_mass=d_mass,
        input_water=d_water,
        input_ie=d_ie,
        input_pe=d_pe,
        input_ke=d_ke,
        input_te=d_te,
        input_sfc_water=sfc_water,
        input_sfc_energy=sfc_energy / dt_out if sfc_energy is not None else None,
        output_file=plot_dir / "mass_energy_budgets.png",
    )
    plot_budgets(
        input_t_mass=t_min,
        input_t_energy=t_min,
        input_mass=mass - mass[0],
        input_water=water - water[0],
        input_ie=ie - ie[0],
        input_pe=pe - pe[0],
        input_ke=ke - ke[0],
        input_te=(te - te[0]), #  / te[0]
        input_sfc_water=cum_sfc_water,
        input_sfc_energy=cum_sfc_energy, #  / te[0]
        output_file=plot_dir / "mass_total_energy_budgets.png",
        cumulative=True,
    )

    total_e = series["internal_energy"] + series["potential_energy"] + series["kinetic_energy"]
    print(f"relative change over the run: air mass {series['mass'][-1] / series['mass'][0] - 1:+.3e}, "
          f"total energy {total_e[-1] / total_e[0] - 1:+.3e}")
    print(f"mean |dE/dt| = {np.mean(np.abs(d_te)):.3f} W m-2"
          + (f", mean |dE/dt + F_E,sfc| = {np.mean(np.abs(d_te + sfc_energy / dt_out)):.3f} W m-2"
             if sfc_energy is not None and np.all(np.isfinite(sfc_energy)) else ""))

    if args.csv:
        csv = plot_dir / "mass_energy_budgets.csv"
        columns = {
            "time_min": t_mid, "d_mass": d_mass, "d_water": d_water,
            "dIE_dt": d_ie, "dPE_dt": d_pe, "dKE_dt": d_ke, "dE_dt": d_te,
        }
        if sfc_water is not None:
            columns["surface_precip"] = sfc_water
            columns["surface_energy_flux"] = sfc_energy / dt_out
        np.savetxt(csv, np.column_stack(list(columns.values())), delimiter=",",
                   header=",".join(columns), comments="")
        print(f"csv: {csv}")


if __name__ == "__main__":
    main()
