#!/usr/bin/env python
"""Time evolution of the domain maximum, the domain mean and the difference of two runs.

For every selected variable of a field group (e.g. atmospheric_state) one figure with
three panels is written:

  max         domain maximum of run A and run B as a function of time
  mean        domain mean of run A and run B (horizontally cell-area weighted, all
              levels with equal weight)
  difference  max |A - B|, mean |A - B| (same weighting as above) and RMS(A - B) on a
              log axis; times at which the two runs agree bit for bit cannot be shown on
              the log axis and are counted in the panel title instead, and a variable
              that is identical at all times is labelled so

NaN values: max and mean of a run are taken over its finite values, and the difference
statistics over the cells where both runs are finite. Cells that are NaN in only one of
the runs are counted separately ("NaN mismatch"), marked as red ticks in the difference
panel and listed in the table, and they prevent a variable from being called identical.

In addition, summary.png shows max |A - B| / max(|A|, |B|) for all variables in one
panel (normalised by the largest magnitude of either run over the compared period), and a
table of the largest difference over the compared period is printed.

Output times are matched by their time value (to 1 ms), so only times present in both
runs are compared; the time axis is the time since the first compared output. The two
runs must have been written on the same grid: the cell areas and centres in the UGRID
files next to the data are compared and the script stops if they differ. Rank-block
output of the "distributed" IO mode is reordered by its global_index_cell coordinate
first, and a group that rolled over into several files (<group>_0001, _0002, ...) is
read as one time series.

RUN_A and RUN_B are either a driver output directory holding <group>_NNNN.nc / .zarr, a
directory one level above it (e.g. output_std, containing output_sample/), or a data
file itself.

Output (in --plot-dir, default ./plots_diff_<A>_vs_<B>):
  diff_<variable>.png   per variable
  summary.png           relative max |A - B| of all variables
  diff_<group>.csv      with --csv, all time series

Usage (inside the uenv, with the icon4py venv activated):
    python plot_output_difference.py output_sequential output_parallel
    python plot_output_difference.py output_sequential output_parallel --vars qv qc air_density
    python plot_output_difference.py output_std output_parallel --group precipitation --csv
"""

import argparse
import pathlib
import re
import warnings

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


TIME_BLOCK = 1 #32  # number of output times read at once (bounds the memory use)
TIME_MATCH_TOLERANCE_S = 1e-3
TIME_UNIT_SECONDS = {"seconds": 1.0, "second": 1.0, "s": 1.0, "minutes": 60.0, "minute": 60.0, "hours": 3600.0, "hour": 3600.0, "days": 86400.0, "day": 86400.0}
STAT_KEYS = ("max_a", "max_b", "mean_a", "mean_b", "maxabs_a", "maxabs_b", "maxabs_diff", "meanabs_diff", "rms_diff", "nan_a", "nan_b", "nan_mismatch")
CSV_KEYS = ("max_a", "max_b", "mean_a", "mean_b", "maxabs_diff", "meanabs_diff", "rms_diff", "nan_mismatch")


# --------------------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------------------
def find_data_files(path: pathlib.Path, group: str) -> list[pathlib.Path]:
    """Data file(s) of the field group: the path itself, or <group>_NNNN.* in it or one level down."""
    if path.is_file() or path.suffix == ".zarr":
        return [path]
    if not path.is_dir():
        raise SystemExit(f"error: {path} does not exist")
    name = re.compile(rf"^{re.escape(group)}_\d+\.(nc|zarr)$")
    for directory in [path, *sorted(p for p in path.iterdir() if p.is_dir() and p.suffix != ".zarr")]:
        files = sorted(p for p in directory.iterdir() if name.match(p.name))
        if files:
            if len({p.suffix for p in files}) > 1:
                raise SystemExit(f"error: both .nc and .zarr files of '{group}' in {directory}, pass the file to use")
            return files
    raise SystemExit(f"error: no '{group}_NNNN.nc' or '{group}_NNNN.zarr' in {path} or its subdirectories")


def reorder_rank_blocks(data: xr.Dataset) -> xr.Dataset:
    """Bring rank-block output (IO mode "distributed") into the global cell order of the grid."""
    if "global_index_cell" not in data:
        return data
    index = data["global_index_cell"].values
    owned = np.flatnonzero(index >= 0)  # -1 marks padding
    order = owned[np.argsort(index[owned], kind="stable")]
    if not np.array_equal(index[order], np.arange(order.size)):
        raise SystemExit("error: global_index_cell is not a permutation of the global cells")
    print(f"note: rank-block layout, reordering {order.size} cells ({index.size - order.size} padding entries dropped)")
    return data.isel(cell=order)


def open_data(files: list[pathlib.Path], every: int) -> xr.Dataset:
    engine = "zarr" if files[0].suffix == ".zarr" else None
    if len(files) == 1:
        data = xr.open_dataset(files[0], engine=engine, decode_times=False)
    else:
        print(f"note: reading {len(files)} files {files[0].name} ... {files[-1].name} as one time series")
        data = xr.open_mfdataset(
            files, engine=engine, decode_times=False, combine="nested", concat_dim="time",
            data_vars="minimal", coords="minimal", compat="override", join="override",
        )
    if every > 1:
        data = data.isel(time=slice(None, None, every))
    return reorder_rank_blocks(data)


def open_grid(data_file: pathlib.Path) -> xr.Dataset | None:
    grids = sorted(data_file.parent.glob("*_ugrid.nc"))
    return xr.open_dataset(grids[0]) if grids else None


def check_same_grid(grid_a: xr.Dataset | None, grid_b: xr.Dataset | None, ncell: int) -> np.ndarray:
    """Normalised horizontal weights (cell areas); stop if the two grids differ."""
    if grid_a is None or grid_b is None:
        print("note: no UGRID file next to one of the runs, using equal cell weights and not checking the grid")
        return np.full(ncell, 1.0 / ncell)
    for name in ("cell_area", "cell_circumcenter_cartesian_x", "cell_circumcenter_cartesian_y", "lon_cell_centre", "lat_cell_centre"):
        if name in grid_a and name in grid_b:
            a, b = grid_a[name].values, grid_b[name].values
            if a.shape != b.shape or not np.allclose(a, b, rtol=1e-12, atol=0.0):
                raise SystemExit(f"error: the grids of the two runs differ in '{name}', a cell-wise difference is meaningless")
    area = grid_a["cell_area"].values if "cell_area" in grid_a else np.ones(ncell)
    if area.size != ncell:
        raise SystemExit(f"error: grid has {area.size} cells, the data {ncell}")
    return area / area.sum()


def time_in_seconds(data: xr.Dataset) -> np.ndarray:
    units = data["time"].attrs.get("units", "seconds")
    factor = TIME_UNIT_SECONDS.get(units.split()[0].lower())
    if factor is None:
        raise SystemExit(f"error: unsupported time units '{units}'")
    return data["time"].values.astype(np.float64) * factor


def common_times(time_a: np.ndarray, time_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one pairs of indices into time_a and time_b of the times (s) present in both runs."""
    ia, ib = [], []
    used = np.zeros(time_b.size, dtype=bool)
    for i, t in enumerate(time_a):
        j = int(np.argmin(np.abs(time_b - t)))
        if abs(time_b[j] - t) <= TIME_MATCH_TOLERANCE_S and not used[j]:
            used[j] = True
            ia.append(i)
            ib.append(j)
    if not ia:
        raise SystemExit("error: the two runs have no output time in common (are the start dates the same?)")
    if len(ia) < max(time_a.size, time_b.size):
        print(f"note: comparing the {len(ia)} common output times (A has {time_a.size}, B has {time_b.size})")
    return np.asarray(ia), np.asarray(ib)


def start_label(data: xr.Dataset, t0: float) -> str:
    """Readable date of the first compared output, if the time units allow it."""
    units = data["time"].attrs.get("units", "")
    if "since" not in units:
        return "first compared output"
    try:
        factor = TIME_UNIT_SECONDS[units.split()[0].lower()]
        date = xr.coding.times.decode_cf_datetime(np.array([t0 / factor]), units, data["time"].attrs.get("calendar", "standard"))[0]
        return str(date)[:19].replace("T", " ")
    except Exception:
        return "first compared output"


def select_variables(data_a: xr.Dataset, data_b: xr.Dataset, requested: list[str] | None) -> list[str]:
    def timed(ds: xr.Dataset) -> list[str]:
        return [v for v in ds.data_vars if "time" in ds[v].dims and "cell" in ds[v].dims]

    in_a, in_b = timed(data_a), timed(data_b)
    available = [v for v in in_a if v in set(in_b)]
    if requested is None:
        only_a = [v for v in in_a if v not in set(in_b)]
        only_b = [v for v in in_b if v not in set(in_a)]
        if only_a:
            print(f"note: only in A, not compared: {only_a}")
        if only_b:
            print(f"note: only in B, not compared: {only_b}")
        return available
    missing = [v for v in requested if v not in available]
    if missing:
        raise SystemExit(f"error: {missing} not in both runs; available: {available}")
    return requested


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------
def weighted_mean(x: np.ndarray, valid: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """(time,) area-weighted mean over the valid cells of every level, averaged over the levels."""
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # levels / times without any valid cell -> NaN
        # valid = valid & (np.abs(x) > 0.0)
        per_level = (np.where(valid, x, np.nan) @ weight) / (valid @ weight)
        return np.nanmean(per_level, axis=1)
        # valid = valid | (np.abs(x) > 1.e-12)
        # per_level = (np.where(valid, x, np.nan) @ weight) / (valid @ weight)
        # mean = np.nanmean(np.where(np.abs(per_level) > 1.e-12, per_level, np.nan), axis=1)
        # nan_mask = np.isnan(mean)
        # mean[nan_mask] = 0.0
        # return mean


def masked_max(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(time,) maximum over the valid entries, NaN at times without any."""
    out = np.where(valid, x, -np.inf).max(axis=(1, 2))
    return np.where(np.isneginf(out) & ~valid.any(axis=(1, 2)), np.nan, out)


def time_series(
    da_a: xr.DataArray, da_b: xr.DataArray, ia: np.ndarray, ib: np.ndarray, weight: np.ndarray
) -> dict[str, np.ndarray]:
    """Domain statistics of A, B and A - B at every common output time."""
    other_dims = [d for d in da_a.dims if d not in ("time", "cell")]
    da_a = da_a.transpose("time", *other_dims, "cell")
    da_b = da_b.transpose("time", *other_dims, "cell")
    nt = ia.size
    nlev = int(np.prod([da_a.sizes[d] for d in other_dims])) if other_dims else 1
    out = {k: np.empty(nt) for k in STAT_KEYS}
    for start in range(0, nt, TIME_BLOCK):
        print(f"    output times {start + 1} ... {min(start + TIME_BLOCK, nt)} of {nt}")
        sl = slice(start, start + TIME_BLOCK)
        a = np.asarray(da_a.isel(time=ia[sl]).values, dtype=np.float64).reshape(-1, nlev, weight.size)
        b = np.asarray(da_b.isel(time=ib[sl]).values, dtype=np.float64).reshape(-1, nlev, weight.size)
        fin_a, fin_b = np.isfinite(a), np.isfinite(b)
        both = fin_a & fin_b
        d = np.where(both, a - b, 0.0)
        out["max_a"][sl] = masked_max(a, fin_a)
        out["max_b"][sl] = masked_max(b, fin_b)
        out["mean_a"][sl] = weighted_mean(a, fin_a, weight)
        out["mean_b"][sl] = weighted_mean(b, fin_b, weight)
        out["maxabs_a"][sl] = masked_max(np.abs(a), fin_a)
        out["maxabs_b"][sl] = masked_max(np.abs(b), fin_b)
        out["maxabs_diff"][sl] = np.abs(d).max(axis=(1, 2))
        out["meanabs_diff"][sl] = weighted_mean(np.abs(d), both, weight)
        out["rms_diff"][sl] = np.sqrt(weighted_mean(d**2, both, weight))
        out["nan_a"][sl] = (~fin_a).sum(axis=(1, 2))
        out["nan_b"][sl] = (~fin_b).sum(axis=(1, 2))
        # a cell where exactly one run is non-finite, or both are but differently (NaN vs inf)
        mismatch = (fin_a != fin_b) | (~fin_a & ~fin_b & ~((a == b) | (np.isnan(a) & np.isnan(b))))
        out["nan_mismatch"][sl] = mismatch.sum(axis=(1, 2))
    return out


def is_identical(s: dict[str, np.ndarray]) -> bool:
    return bool(np.all(s["maxabs_diff"] == 0.0) and np.all(s["nan_mismatch"] == 0))


def magnitude(s: dict[str, np.ndarray]) -> float:
    """Largest magnitude of either run over the compared period (0 if both are zero)."""
    values = np.concatenate([s["maxabs_a"], s["maxabs_b"]])
    values = values[np.isfinite(values)]
    return float(values.max()) if values.size else 0.0


# --------------------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------------------
def positive(x: np.ndarray) -> np.ma.MaskedArray:
    """Mask the non-positive and non-finite values, which a log axis cannot show."""
    return np.ma.masked_where(~np.isfinite(x) | (x <= 0.0), x)


def mark_nan_mismatch(ax: plt.Axes, t_min: np.ndarray, s: dict[str, np.ndarray]) -> str:
    bad = s["nan_mismatch"] > 0
    if not bad.any():
        return ""
    for t in t_min[bad]:
        ax.axvline(t, color="red", alpha=0.25, linewidth=1.0)
    return f"NaN in only one run at {int(bad.sum())} times (red), first at {t_min[bad][0]:.1f} min"


def plot_variable(
    name: str, units: str, t_min: np.ndarray, xlabel: str, s: dict[str, np.ndarray], *, labels: tuple[str, str], path: pathlib.Path
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, constrained_layout=True)
    unit = f" ({units})" if units else ""
    marker = {"marker": ".", "markersize": 3} if t_min.size <= 400 else {}

    ax = axes[0]
    ax.plot(t_min, s["max_a"], color="C0", label=labels[0])
    ax.plot(t_min, s["max_b"], color="C1", linestyle="--", label=labels[1])
    ax.set_ylabel(f"max{unit}")
    ax.legend(loc="best", fontsize="small")

    ax = axes[1]
    ax.plot(t_min, s["mean_a"], color="C0", label=labels[0])
    ax.plot(t_min, s["mean_b"], color="C1", linestyle="--", label=labels[1])
    ax.set_ylabel(f"mean{unit}")

    ax = axes[2]
    notes = []
    if is_identical(s):
        ax.text(0.5, 0.5, "bitwise identical at all compared times", transform=ax.transAxes, ha="center", va="center")
        ax.set_yticks([])
    else:
        if np.any(s["maxabs_diff"] > 0.0):
            ax.semilogy(t_min, positive(s["maxabs_diff"]), color="C3", label="max |A - B|", **marker)
            ax.semilogy(t_min, positive(s["rms_diff"]), color="C2", label="RMS(A - B)", **marker)
            ax.semilogy(t_min, positive(s["meanabs_diff"]), color="C4", linestyle="--", label="mean |A - B|", **marker)
            ax.legend(loc="best", fontsize="small")
        else:
            ax.set_yticks([])
            notes.append("finite values identical")
        n_identical = int(np.count_nonzero((s["maxabs_diff"] == 0.0) & (s["nan_mismatch"] == 0)))
        if n_identical:
            notes.append(f"identical at {n_identical} of {t_min.size} times (not shown)")
        notes.append(mark_nan_mismatch(ax, t_min, s))
        ax.set_title("; ".join(n for n in notes if n), fontsize="small")
    ax.set_ylabel(f"difference{unit}")
    ax.set_xlabel(xlabel)

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{name}:  A = {labels[0]},  B = {labels[1]}")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_summary(
    t_min: np.ndarray, xlabel: str, stats: dict[str, dict[str, np.ndarray]], labels: tuple[str, str], path: pathlib.Path
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    identical, nan_only = [], []
    marker = {"marker": ".", "markersize": 3} if t_min.size <= 400 else {}
    n = 0
    for name, s in stats.items():
        if is_identical(s):
            identical.append(name)
            continue
        if np.any(s["nan_mismatch"] > 0):
            nan_only.append(name)
        if not np.any(s["maxabs_diff"] > 0.0):
            continue
        rel = s["maxabs_diff"] / magnitude(s)  # magnitude > 0 since the runs differ somewhere
        ax.semilogy(t_min, positive(rel), color=f"C{n % 10}", linestyle="-" if n < 10 else "--", label=name, **marker)
        n += 1
    if ax.get_lines():
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small")
    text = []
    if identical:
        text.append("bitwise identical: " + ", ".join(identical))
    if nan_only:
        text.append("NaN in only one run: " + ", ".join(nan_only))
    if text:
        ax.text(0.01, 0.99, "\n".join(text), transform=ax.transAxes, va="top", fontsize="small", wrap=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("max |A - B| / max over time of max(|A|, |B|)")
    ax.set_title(f"A = {labels[0]},  B = {labels[1]}")
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_csv(t_s: np.ndarray, stats: dict[str, dict[str, np.ndarray]], path: pathlib.Path) -> None:
    columns = ["time_s"]
    values = [t_s]
    for name, s in stats.items():
        for k in CSV_KEYS:
            columns.append(f"{name}:{k}")
            values.append(s[k])
    np.savetxt(path, np.column_stack(values), delimiter=",", header=",".join(columns), comments="", fmt="%.10e")


def print_table(t_min: np.ndarray, stats: dict[str, dict[str, np.ndarray]]) -> None:
    width = max([len(n) for n in stats] + [8])
    print(f"\n{'variable':<{width}}  {'max|A-B|':>11}  {'at t (min)':>10}  {'max|A|,|B|':>11}  {'relative':>9}  {'NaN mismatch':>12}")
    for name, s in stats.items():
        scale = magnitude(s)
        nan_cells = int(s["nan_mismatch"].sum())
        nan_text = f"{nan_cells} @{t_min[np.argmax(s['nan_mismatch'] > 0)]:.1f}min" if nan_cells else "-"
        if is_identical(s):
            print(f"{name:<{width}}  {'identical':>11}  {'-':>10}  {scale:11.3e}  {'-':>9}  {'-':>12}")
            continue
        k = int(np.argmax(s["maxabs_diff"]))
        if s["maxabs_diff"][k] == 0.0:
            print(f"{name:<{width}}  {0.0:11.3e}  {'-':>10}  {scale:11.3e}  {'-':>9}  {nan_text:>12}")
            continue
        rel = s["maxabs_diff"][k] / scale
        print(f"{name:<{width}}  {s['maxabs_diff'][k]:11.3e}  {t_min[k]:10.1f}  {scale:11.3e}  {rel:9.2e}  {nan_text:>12}")


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def run_label(path: pathlib.Path) -> str:
    path = path.resolve()
    if path.is_file() or path.suffix == ".zarr":
        path = path.parent
    return path.parent.name if path.name.startswith("output_sample") else path.name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_a", type=pathlib.Path, help="run A: output directory or data file")
    p.add_argument("run_b", type=pathlib.Path, help="run B: output directory or data file")
    p.add_argument("--group", default="atmospheric_state", help="basename of the field group (default: atmospheric_state)")
    p.add_argument("--vars", nargs="+", default=None, help="variables to compare (default: all with time and cell dimensions)")
    p.add_argument("--labels", nargs=2, default=None, metavar=("A", "B"), help="legend labels (default: directory names)")
    p.add_argument("--plot-dir", type=pathlib.Path, default=None, help="default: ./plots_diff_<A>_vs_<B>")
    p.add_argument("--csv", action="store_true", help="also write all time series as CSV")
    p.add_argument(
        "--every", type=int, default=1, metavar="N",
        help="read only every N-th output time of the 3D group (default: 1, all); the "
        "surface fluxes are still integrated over all precipitation samples",
    )
    args = p.parse_args()

    files_a, files_b = find_data_files(args.run_a, args.group), find_data_files(args.run_b, args.group)
    print(f"A: {files_a[0]}" + (f" (+{len(files_a) - 1} files)" if len(files_a) > 1 else ""))
    print(f"B: {files_b[0]}" + (f" (+{len(files_b) - 1} files)" if len(files_b) > 1 else ""))
    labels = tuple(args.labels) if args.labels else (run_label(files_a[0]), run_label(files_b[0]))
    if labels[0] == labels[1]:
        labels = (f"{labels[0]} (A)", f"{labels[1]} (B)")
    data_a, data_b = open_data(files_a, args.every), open_data(files_b, args.every)
    if data_a.sizes["cell"] != data_b.sizes["cell"]:
        raise SystemExit(f"error: {data_a.sizes['cell']} cells in A, {data_b.sizes['cell']} in B")
    weight = check_same_grid(open_grid(files_a[0]), open_grid(files_b[0]), data_a.sizes["cell"])

    units_a, units_b = data_a["time"].attrs.get("units", ""), data_b["time"].attrs.get("units", "")
    if units_a != units_b:
        print(f"warning: different time units '{units_a}' (A) and '{units_b}' (B), comparing the raw values in seconds")
    time_a, time_b = time_in_seconds(data_a), time_in_seconds(data_b)
    ia, ib = common_times(time_a, time_b)
    t_s = time_a[ia]
    t_min = (t_s - t_s[0]) / 60.0
    xlabel = f"time since {start_label(data_a, t_s[0])} (min)"
    variables = select_variables(data_a, data_b, args.vars)

    plot_dir = args.plot_dir or pathlib.Path(f"plots_diff_{labels[0]}_vs_{labels[1]}".replace(" ", "_"))
    plot_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict[str, np.ndarray]] = {}
    for name in variables:
        da_a, da_b = data_a[name], data_b[name]
        if da_a.dims != da_b.dims or da_a.shape[1:] != da_b.shape[1:]:
            print(f"skip {name}: shapes differ {da_a.dims}{da_a.shape} vs {da_b.dims}{da_b.shape}")
            continue
        print(f"  {name}")
        stats[name] = time_series(da_a, da_b, ia, ib, weight)
        plot_variable(name, da_a.attrs.get("units", ""), t_min, xlabel, stats[name], labels=labels, path=plot_dir / f"diff_{name}.png")

    if not stats:
        raise SystemExit("error: no variable could be compared")
    plot_summary(t_min, xlabel, stats, labels, plot_dir / "summary.png")
    if args.csv:
        write_csv(t_s, stats, plot_dir / f"diff_{args.group}.csv")
    print_table(t_min, stats)
    print(f"\nplots written to {plot_dir}/")


if __name__ == "__main__":
    main()
