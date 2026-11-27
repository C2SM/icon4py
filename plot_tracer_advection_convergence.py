#!/usr/bin/env python
"""Plot the convergence of the tracer advection tests from the errors they wrote.

The tests in model/driver/tests/driver/validation_tests/test_tracer_advection_convergence.py
write <test>_<experiment_case>.json into $ICON4PY_CONVERGENCE_OUTPUT_DIR (see
run_convergence_tests.sh). Each file holds the relative L1 and Linf errors of the tracer qv
against the analytical solution at every resolution, the grid spacing (mean edge length for
the horizontal tests, layer thickness for the vertical ones) and the acceptable ranges of
the convergence order the test asserts.

For every file this writes <name>.png: both errors on log-log axes (like
plot_utils.plot_convergence), the least-squares order p of each norm in the legend, and
dashed reference slopes at the expected orders (the middle of the asserted ranges),
anchored at the coarsest point of each norm. convergence_overview.png puts all cases side by
side, and a table of the orders and whether they pass the test's checks is printed.

Usage:
    python plot_tracer_advection_convergence.py convergence_<jobid>/errors
    python plot_tracer_advection_convergence.py convergence_*/errors/horizontal_*.json --out-dir plots
"""

import argparse
import json
import pathlib

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


STD_TOL = 0.4  # _STD_TOL of the test
NORMS = (("l1", "$L_1$", "tab:blue", "o"), ("linf", r"$L_\infty$", "tab:red", "s"))
XLABEL = {"horizontal": "mean edge length $h$ (m)", "vertical": r"layer thickness $\Delta z$ (m)"}


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


def read_records(paths: list[pathlib.Path]) -> list[dict]:
    files: list[pathlib.Path] = []
    for p in paths:
        files.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    if not files:
        raise SystemExit(f"error: no .json files in {' '.join(map(str, paths))}")
    records = []
    for f in files:
        rec = json.loads(f.read_text())
        rec["path"] = f
        records.append(rec)
    return sorted(records, key=lambda r: (r["test"], r["experiment_case"]))


def fit(h: np.ndarray, err: np.ndarray) -> tuple[float, float] | None:
    """Least-squares slope and its standard error of log(err) over log(h), as in the test."""
    if h.size < 2:
        return None
    r = linregress(np.log(h), np.log(err))
    return float(r.slope), float(r.stderr)


def draw(ax, rec: dict) -> None:
    h = np.asarray(rec["grid_spacing"], dtype=float)
    ax.set_axisbelow(True)
    for norm, label, colour, marker in NORMS:
        err = np.asarray(rec[f"error_{norm}"], dtype=float)
        p = fit(h, err)
        text = label if p is None else f"{label}: $p = {p[0]:.2f}$"
        ax.plot(h, err, marker=marker, color=colour, label=text)
        # expected order: middle of the asserted range, anchored at the coarsest point
        lo, hi = rec[f"{norm}_acceptable_range"]
        order = 0.5 * (lo + hi)
        i0 = int(np.argmax(h))
        ax.axline(
            (h[i0], err[i0]),
            (h[i0] / 2, err[i0] * 0.5**order),
            ls="--",
            c=colour,
            lw=1.0,
            alpha=0.6,
            label=rf"{label} expected $p={order:g}$ ($\pm{0.5 * (hi - lo):g}$)",
            zorder=1.9,
        )
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend()
    ax.set_xlabel(XLABEL.get(rec["test"], "$h$"))
    ax.set_ylabel("relative error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{rec['experiment_case']}") #  ({rec['backend']})
    for x, res in zip(h, rec.get("resolutions", []), strict=False):
        ax.annotate(res.replace("torus_100000mx100000m_", ""), (x, ax.get_ylim()[0]), fontsize=0.65 * plt.rcParams["font.size"], color="0.4",
                    rotation=90, ha="right", va="bottom", xytext=(-2, 2), textcoords="offset points")


def check(rec: dict) -> list[str]:
    """One table row per norm: order, its standard error and whether the test's asserts pass."""
    h = np.asarray(rec["grid_spacing"], dtype=float)
    rows = []
    for norm, _, _, _ in NORMS:
        err = np.asarray(rec[f"error_{norm}"], dtype=float)
        lo, hi = rec[f"{norm}_acceptable_range"]
        p = fit(h, err)
        if p is None:
            status, ptxt = "too few points", "-"
        else:
            ok = lo <= p[0] <= hi and p[1] <= STD_TOL
            status = "PASS" if ok else "FAIL"
            ptxt = f"{p[0]:6.3f} ± {p[1]:.3f}"
        errs = " ".join(f"{e:.3e}" for e in err)
        rows.append(f"{rec['experiment_case']:42s} {norm:4s} {ptxt:>16s}  [{lo:g}, {hi:g}]  {status:5s}  {errs}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", type=pathlib.Path, nargs="+", help="directories with the .json files, or the files")
    p.add_argument("--out-dir", type=pathlib.Path, default=None, help="default: <first input dir>/../plots")
    p.add_argument("--font-size", type=float, default=15.0, help="base font size (pt)")
    args = p.parse_args()
    set_fonts(args.font_size)

    records = read_records(args.inputs)
    first = args.inputs[0] if args.inputs[0].is_dir() else args.inputs[0].parent
    out_dir = args.out_dir or first.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for rec in records:
        fig, ax = plt.subplots(figsize=(11, 7))
        draw(ax, rec)
        out = out_dir / f"{rec['path'].stem}.png"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"plot: {out}")

    ncol = min(2, len(records))
    nrow = int(np.ceil(len(records) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9 * ncol, 6.5 * nrow), squeeze=False)
    for ax, rec in zip(axes.flat, records, strict=False):
        draw(ax, rec)
    for ax in axes.flat[len(records):]:
        ax.set_visible(False)
    fig.tight_layout()
    out = out_dir / "convergence_overview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"plot: {out}\n")

    print(f"{'case':42s} {'norm':4s} {'order ± stderr':>16s}  range    check  errors (coarse -> fine)")
    for rec in records:
        for row in check(rec):
            print(row)


if __name__ == "__main__":
    main()
