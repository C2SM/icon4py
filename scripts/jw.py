import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import netCDF4 as nc4
import numpy as np
import xarray as xr
import pickle
import colormaps as cmaps
import cmcrameri.cm as cmc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
# import xesmf
from scipy.interpolate import griddata
import cartopy.crs as ccrs
from matplotlib.colors import TwoSlopeNorm
from scale_tools import amps_library as amps, helper_functions as helper
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from concurrent.futures import ThreadPoolExecutor
from scipy.interpolate import RectSphereBivariateSpline


# DATA_DIR = "/capstor/store/cscs/userlab/cwd01/cong/jw_data/r2b4/"
EXPERIMENT = ("r2b4","r2b5","r2b6","r2b7","r2b8","r2b8_ref")
EXPERIMENT_LABEL = ("r2b4 (220 km)","r2b5 (110 km)","r2b6 (55 km)","r2b7 (27 km)","r2b8 (13 km)","r2b8-ref (13 km)")
LINE_COLOR = ["#2b6dad", "#e2764e", "#4c9c6e", "#000000", "#d62728", "#9467bd"]
BAROCLINIC_EXPERIMENT = "r2b6_2nodes/" # output_2026-04-06_8h_59m_30s output_2026-04-05_22h_45m_24s
# DATA_DIR = "/capstor/store/cscs/userlab/cwd01/cong/icon4py_data/multi_8_nodes_gpu/"
# DATA_DIR = "/iopsstor/scratch/cscs/vogtha/icon4py_jw/icon4py/icon4py_r2b8_32ranks/"
# /iopsstor/scratch/cscs/vogtha/icon4py_jw/icon4py/icon4py_r2b8_32ranks/output_2026-04-06_8h_59m_30s
# DATA_DIR = "/capstor/store/cscs/userlab/cwd01/cong/icon4py_data/single_node_gpu/"
DATA_DIR = "/capstor/store/cscs/userlab/cwd01/cong/icon4py_data/new_data/"
BAROCLINIC_EXPERIMENT_FOR_VERIFICATION = "r2b6_1node/"
# DATA_DIR_FOR_VERIFICATION = "/iopsstor/scratch/cscs/vogtha/icon4py_jw/icon4py/icon4py_r2b8_32ranks/"
# DATA_DIR_FOR_VERIFICATION = "/capstor/store/cscs/userlab/cwd01/cong/icon4py_data/single_node_gpu/"
# DATA_DIR_FOR_VERIFICATION = "/capstor/store/cscs/userlab/cwd01/cong/icon4py_data/multi_8_nodes_cpu/"
DATA_DIR_FOR_VERIFICATION = "/capstor/store/cscs/userlab/cwd01/cong/icon4py_data/new_data/"
PLOT_DIR = "/capstor/scratch/cscs/cong/analysis/plotting/src/global_plot/"
DATA_FILENAME = "model_data"
EXTENSION_NAME = ".pkl"
DO_DIFF = True

def setup_plot_style():
    """论文级绘图风格"""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "DejaVu Sans",
        "axes.labelsize": 13,
        "axes.titlesize": 16,
        "axes.linewidth": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.frameon": True,
    })

def setup_contour_style():
    """论文级绘图风格"""
    plt.rcParams.update({
        "font.size": 14,
        "font.family": "DejaVu Sans",
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "axes.linewidth": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.frameon": True,
    })


def lineplot(time, input_data, ylog: bool, ylabel: str, filename: str):
    plt.close()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    
    for exp_i, experiment in enumerate(EXPERIMENT):
        sns.lineplot(
            data=pd.DataFrame(data={"Time": time[exp_i], "L2 error": input_data[exp_i]}),
            x="Time",
            y="L2 error",
            color=LINE_COLOR[exp_i],
            label=EXPERIMENT_LABEL[exp_i],
            linewidth=1.8,
            ax=ax,
            zorder=1,
        )
    ax.spines['bottom'].set_alpha(0.8)
    ax.spines['left'].set_alpha(0.8)
    ax.spines['top'].set_alpha(0.8)
    ax.spines['right'].set_alpha(0.8)
    if ylog:
        ax.set_ylim(1.e-5, 10)
        ax.set_yticks([1.e-5, 1.e-4, 1.e-3, 1.e-2, 1.e-1, 1, 10])
        ax.set_yscale("log")
    else:    
        ax.set_ylim(-5, 105)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlim(-0.7, 14.7)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax.set_xlabel("Time (day)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.0)
    # ax.set_title("Surface pressure error evolution")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)


def contourplot(lon2d, lat2d, input_data, cmap, colors_boundary, colors_tick, title, output_filename):
    plt.close()
    fig, ax = plt.subplots(figsize=(6, 4.2))

    cp = ax.contourf(lon2d, lat2d, input_data, cmap=cmap, levels=colors_boundary, vmin=colors_boundary.min(), vmax=colors_boundary.max(), extend='both')
    ax.contour(lon2d, lat2d, input_data, colors='black', levels=colors_boundary, vmin=colors_boundary.min(), vmax=colors_boundary.max(), linewidths=0.3)
    cb = plt.colorbar(
        cp,
        ax=ax,
        orientation="horizontal",
        pad=0.18,
        ticks=colors_tick,
    )
    ax.set_title(title)
    ax.set_xlim(45, 360)
    ax.set_ylim(0, 90)
    ax.set_xticks([45, 90, 180, 270, 360])
    ax.set_yticks([0, 30, 60, 90])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.savefig(output_filename, format="png", dpi=300)
    plt.clf()


def full_contourplot(lon2d, lat2d, input_data, cmap, colors_boundary, colors_tick, title, output_filename):
    plt.close()
    fig, ax = plt.subplots(figsize=(6, 4.2))

    cp = ax.contourf(lon2d, lat2d, input_data, cmap=cmap, levels=colors_boundary, vmin=colors_boundary.min(), vmax=colors_boundary.max(), extend='both')
    cb = plt.colorbar(
        cp,
        ax=ax,
        orientation="horizontal",
        pad=0.18,
        ticks=colors_tick,
    )
    ax.set_title(title)
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.savefig(output_filename, format="png", dpi=300)
    plt.clf()


def get_filename(day, hour):
    return f"{DATA_FILENAME}_day{day:d}_hour{hour:d}{EXTENSION_NAME}"


def plot_error():
    def _create_list_of_lists():
        return [[] for _ in range(len(EXPERIMENT))]
    error = _create_list_of_lists()
    error_u = _create_list_of_lists()
    error_v = _create_list_of_lists()
    error_t = _create_list_of_lists()
    
    time = [[] for _ in range(len(EXPERIMENT))]

    def compute_error(exp_i, experiment):
        print("Reading", experiment)
        data_dir = DATA_DIR + experiment + "/"
        # initial condition
        with open(os.path.join(data_dir, get_filename(1, 0)), "rb") as f:
            initial_ds = pickle.load(f)

        num_levels = initial_ds["u"].shape[1]
        num_cells = initial_ds["cell_area"].shape[0]
        expanded_cell_area = np.repeat(np.expand_dims(initial_ds["cell_area"], axis=-1), num_levels, axis=1)
        print("numbers of levels and cells: ", num_levels, num_cells)
        print("expanded shapes: ", expanded_cell_area.shape, initial_ds["dz"].shape)

        for day in range(1,15):
            for hour in range(0,1):
                # fname = get_filename(day, hour)
                # print(fname)
                with open(os.path.join(data_dir, get_filename(day, hour)), "rb") as f:
                    ds = pickle.load(f)
                time[exp_i].append(day+float(hour/24) - 1)
                error[exp_i].append(np.sqrt(np.sum(np.abs(ds["sfc_pressure"] - initial_ds["sfc_pressure"])**2 * initial_ds["cell_area"]) / np.sum(initial_ds["cell_area"])) )
                error_u[exp_i].append(np.sqrt(np.sum(np.abs(ds["u"] - initial_ds["u"])**2 * expanded_cell_area * initial_ds["dz"]) / np.sum(expanded_cell_area * initial_ds["dz"])) )
                error_v[exp_i].append(np.sqrt(np.sum(np.abs(ds["v"] - initial_ds["v"])**2 * expanded_cell_area * initial_ds["dz"]) / np.sum(expanded_cell_area * initial_ds["dz"])) )
                error_t[exp_i].append(np.sqrt(np.sum(np.abs(ds["temperature"] - initial_ds["temperature"])**2 * expanded_cell_area * initial_ds["dz"]) / np.sum(expanded_cell_area * initial_ds["dz"])) )
        print("Finish reading", experiment)

    with ThreadPoolExecutor(max_workers=len(EXPERIMENT)) as executor:
        # Submit tasks to run in parallel
        finishers = []
        for exp_i, experiment in enumerate(EXPERIMENT):
            finishers.append(executor.submit(compute_error, exp_i, experiment))
        for finisher in finishers:
            finisher.result()

    setup_plot_style()
    lineplot(time, error, False, "L2 error of surface pressure (Pa)", "sfc_pressure_error.png")
    lineplot(time, error_u, True, "L2 error of U (m s$^{-1}$)", "u_error.png")
    lineplot(time, error_v, True, "L2 error of V (m s$^{-1}$)", "v_error.png")
    lineplot(time, error_t, True, "L2 error of T (K)", "t_error.png")


def plot_baroclinic_case():
    grid_filename = (
        "/capstor/store/cscs/userlab/cwd01/cong/grids/icon_grid_0002_R02B06_G.nc"
        # "/capstor/store/cscs/userlab/cwd01/cong/grids/icon_grid_0010_R02B04_G.nc"
    )

    grid = xr.open_dataset(grid_filename, engine="netcdf4")

    def _read_for_2dplot(day) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        hour = 0
        data_dir = DATA_DIR + BAROCLINIC_EXPERIMENT
        with open(os.path.join(data_dir, get_filename(day, hour)), "rb") as f:
            ds = pickle.load(f)

        lon_reg = np.linspace(0,360,360)
        lat_reg = np.linspace(-90,90,180)

        lon2d, lat2d = np.meshgrid(lon_reg, lat_reg)

        rebased_lon = np.where(grid.clon.values < 0.0, 360.0*math.pi/180.0 + grid.clon.values, grid.clon.values)
        print(rebased_lon.shape, rebased_lon[1])
        if np.nanmin(ds["sfc_pressure"]) > 2000.0:
            rescale = 100.0
        else:
            rescale = 1.0
        sp_interp = griddata(
            (rebased_lon*180.0/math.pi, grid.clat.values*180.0/math.pi),
            ds["sfc_pressure"]/rescale,
            (lon2d, lat2d),
            method="linear"
        )

        target_pres = 85000.0
        num_levels = ds["temperature"].shape[1]
        num_cells = ds["temperature"].shape[0]
        target_levels, intp_coeff = np.full(num_cells, fill_value=-1, dtype=int), np.zeros(num_cells, dtype=float)
        for k in range(num_levels-1):
            mask = np.where((target_pres >= ds["pressure"][:,k]) & (target_pres < ds["pressure"][:,k+1]), True, False)
            target_levels = np.where(mask, k, target_levels)
            intp_coeff = np.where(mask, (target_pres - ds["pressure"][:,k]) / (ds["pressure"][:,k+1] - ds["pressure"][:,k]), intp_coeff)
        assert np.all(target_levels >= 0), "Some cells do not have valid target levels"
        # assert target_levels.shape == ds["sfc_pressure"].shape, f"target_levels shape {target_levels.shape} does not match sfc_pressure shape {ds['sfc_pressure'].shape}"
        
        temperature_850hpa = (1.0 - intp_coeff) * ds["temperature"][np.arange(num_cells), target_levels] + intp_coeff * ds["temperature"][np.arange(num_cells), target_levels + 1]
        print("temperature_850hpa min and max: ", np.nanmin(temperature_850hpa), np.nanmax(temperature_850hpa))
        print("temperature at 850hpa shape: ", temperature_850hpa.shape)
        temp850_interp = griddata(
            (rebased_lon*180.0/math.pi, grid.clat.values*180.0/math.pi),
            temperature_850hpa,
            (lon2d, lat2d),
            method="linear"
        )
        
        u_interp = griddata(
            (rebased_lon*180.0/math.pi, grid.clat.values*180.0/math.pi),
            ds["u"][:,num_levels-1],
            (lon2d, lat2d),
            method="linear"
        )
        v_interp = griddata(
            (rebased_lon*180.0/math.pi, grid.clat.values*180.0/math.pi),
            ds["v"][:,num_levels-1],
            (lon2d, lat2d),
            method="linear"
        )
        
        return lon2d, lat2d, sp_interp, temp850_interp, u_interp, v_interp

    setup_contour_style()

    lon2d, lat2d, sp_interp, temp850_interp, u_interp, v_interp = _read_for_2dplot(11)
    sp_colors = [(236, 0, 140), (166, 34, 142), (32, 65, 154), (0, 133, 204), (0, 174, 239), (0, 170, 79), (200, 218, 43), (255, 242, 0), (249, 158, 27), (237, 28, 36)]
    temp850_colors = [(166, 34, 142), (32, 65, 154), (0, 133, 204), (0, 174, 239), (0, 170, 79), (200, 218, 43), (255, 242, 0), (249, 158, 27), (237, 28, 36)]
    sp_colors_norm = [(r/255, g/255, b/255) for r, g, b in sp_colors]
    temp850_colors_norm = [(r/255, g/255, b/255) for r, g, b in temp850_colors]
    sp_cmap = LinearSegmentedColormap.from_list("sp_list", sp_colors_norm, N=len(sp_colors))
    temp850_cmap = LinearSegmentedColormap.from_list("temp850_list", temp850_colors_norm, N=len(temp850_colors))
    # colors_tick = [992,994,996,998,1000,1002,1004,1006]
    sp_colors_boundary = np.linspace(930,1030,11)
    sp_colors_tick = [940, 960, 980, 1000, 1020]
    temp850_colors_boundary = np.linspace(220,310,10)
    temp850_colors_tick = [220, 240, 260, 280, 300]
    u_colors_boundary = np.linspace(-50, 50,100)
    u_colors_tick = [-50, -40, -20, 0, 20, 40, 50]
    v_colors_boundary = np.linspace(-50, 50, 100)
    v_colors_tick = [-50, -40, -20, 0, 20, 40, 50]

    contourplot(lon2d, lat2d, sp_interp, sp_cmap, sp_colors_boundary, sp_colors_tick, "Surface pressure (hPa) at day 10", os.path.join(PLOT_DIR, "sfc_pressure_day10.png"))
    contourplot(lon2d, lat2d, temp850_interp, temp850_cmap, temp850_colors_boundary, temp850_colors_tick, "Temperature at 850hPa (K) at day 10", os.path.join(PLOT_DIR, "temperature_850hpa_day10.png"))
    full_contourplot(lon2d, lat2d, u_interp, plt.get_cmap("jet"), u_colors_boundary, u_colors_tick, "Surface U (m s-1) at day 10", os.path.join(PLOT_DIR, "sfc_u_day10.png"))
    full_contourplot(lon2d, lat2d, v_interp, plt.get_cmap("jet"), v_colors_boundary, v_colors_tick, "Surface V (m s-1) at day 10", os.path.join(PLOT_DIR, "sfc_v_day10.png"))
    

    lon2d, lat2d, sp_interp, temp850_interp, u_interp, v_interp = _read_for_2dplot(7)
    sp_colors = [(236, 0, 140), (32, 65, 154), (0, 170, 79), (200, 218, 43), (255, 242, 0), (249, 158, 27), (237, 28, 36)]
    temp850_colors = [(166, 34, 142), (32, 65, 154), (0, 133, 204), (0, 174, 239), (0, 170, 79), (200, 218, 43), (255, 242, 0), (249, 158, 27), (237, 28, 36)]
    sp_colors_norm = [(r/255, g/255, b/255) for r, g, b in sp_colors]
    temp850_colors_norm = [(r/255, g/255, b/255) for r, g, b in temp850_colors]
    sp_cmap = LinearSegmentedColormap.from_list("sp_list", sp_colors_norm, N=len(sp_colors))
    temp850_cmap = LinearSegmentedColormap.from_list("temp850_list", temp850_colors_norm, N=len(temp850_colors))
    sp_colors_boundary = np.linspace(992,1006,8)
    sp_colors_tick = [992,996,1000,1004]
    temp850_colors_boundary = np.linspace(220,310,10)
    temp850_colors_tick = [220, 240, 260, 280, 300]

    contourplot(lon2d, lat2d, sp_interp, sp_cmap, sp_colors_boundary, sp_colors_tick, "Surface pressure (hPa) at day 6", os.path.join(PLOT_DIR, "sfc_pressure_day6.png"))
    contourplot(lon2d, lat2d, temp850_interp, temp850_cmap, temp850_colors_boundary, temp850_colors_tick, "Temperature at 850hPa (K) at day 6", os.path.join(PLOT_DIR, "temperature_850hpa_day6.png"))
    full_contourplot(lon2d, lat2d, u_interp, plt.get_cmap("jet"), u_colors_boundary, u_colors_tick, "Surface U (m s-1) at day 6", os.path.join(PLOT_DIR, "sfc_u_day6.png"))
    full_contourplot(lon2d, lat2d, v_interp, plt.get_cmap("jet"), v_colors_boundary, v_colors_tick, "Surface V (m s-1) at day 6", os.path.join(PLOT_DIR, "sfc_v_day6.png"))
    
    with open(os.path.join(DATA_DIR + BAROCLINIC_EXPERIMENT, get_filename(15, 0)), "rb") as f:
        ds_target = pickle.load(f)
    with open(os.path.join(DATA_DIR_FOR_VERIFICATION + BAROCLINIC_EXPERIMENT_FOR_VERIFICATION, get_filename(15, 0)), "rb") as f:
        ds_reference = pickle.load(f)
    u_target, v_target = ds_target["u"], ds_target["v"]
    u_reference, v_reference = ds_reference["u"], ds_reference["v"]
    temperature_target, pressure_target = ds_target["temperature"], ds_target["pressure"]
    temperature_reference, pressure_reference = ds_reference["temperature"], ds_reference["pressure"]
    print("15-day max errors for u and v: ", np.abs(u_target - u_reference).max(), np.abs(v_target - v_reference).max())
    print("15-day max errors for temperature and pressure: ", np.abs(temperature_target - temperature_reference).max(), np.abs(pressure_target - pressure_reference).max())



def plot_static_case():
    grid_filename = (
        "/capstor/store/cscs/userlab/cwd01/cong/grids/icon_grid_0002_R02B06_G.nc"
        # "/capstor/store/cscs/userlab/cwd01/cong/grids/icon_grid_0004_R02B07_G.nc"
        # "/capstor/store/cscs/userlab/cwd01/cong/grids/icon_grid_0033_R02B08_G.nc"
        # "/capstor/store/cscs/userlab/cwd01/cong/grids/icon_grid_0010_R02B04_G.nc"
    )

    grid = xr.open_dataset(grid_filename, engine="netcdf4")

    def _read_for_2dplot(day, hour) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        if DO_DIFF:
            data_dir1 = DATA_DIR + BAROCLINIC_EXPERIMENT
            with open(os.path.join(data_dir1, get_filename(day, hour)), "rb") as f:
                ds1 = pickle.load(f)
            data_dir2 = DATA_DIR_FOR_VERIFICATION + BAROCLINIC_EXPERIMENT_FOR_VERIFICATION
            with open(os.path.join(data_dir2, get_filename(day, hour)), "rb") as f:
                ds2 = pickle.load(f)

            lon_reg = np.linspace(0,360,360)
            lat_reg = np.linspace(-90,90,180)

            lon2d, lat2d = np.meshgrid(lon_reg, lat_reg)

            rebased_lon = np.where(grid.clon.values < 0.0, 360.0*math.pi/180.0 + grid.clon.values, grid.clon.values)
            print(rebased_lon.shape, rebased_lon[1])
            sp_interp = griddata(
                (rebased_lon*180.0/math.pi, grid.clat.values*180.0/math.pi),
                ds1["sfc_pressure"] - ds2["sfc_pressure"],
                (lon2d, lat2d),
                method="linear"
            )

            num_levels = ds1["v"].shape[1]
            u_interp = griddata(
                (rebased_lon*180.0/math.pi, grid.clat.values*180.0/math.pi),
                ds1["u"][:,num_levels-1] - ds2["u"][:,num_levels-1],
                (lon2d, lat2d),
                method="linear"
            )
            v_interp = griddata(
                (rebased_lon*180.0/math.pi, grid.clat.values*180.0/math.pi),
                ds1["v"][:,num_levels-1] - ds2["v"][:,num_levels-1],
                (lon2d, lat2d),
                method="linear"
            )
            print("vn min and max: ", np.nanmin(ds1["vn"][:,:] - ds2["vn"][:,:]), np.nanmax(ds1["vn"][:,:] - ds2["vn"][:,:]))
            print("w min and max: ", np.nanmin(ds1["w"][:,:] - ds2["w"][:,:]), np.nanmax(ds1["w"][:,:] - ds2["w"][:,:]))
            print("sfc u min and max: ", np.nanmin(ds1["u"][:,num_levels-1] - ds2["u"][:,num_levels-1]), np.nanmax(ds1["u"][:,num_levels-1] - ds2["u"][:,num_levels-1]))
            print("sfc pres min and max: ", np.nanmin(ds1["sfc_pressure"] - ds2["sfc_pressure"]), np.nanmax(ds1["sfc_pressure"] - ds2["sfc_pressure"]))
        else:
            data_dir = DATA_DIR + BAROCLINIC_EXPERIMENT
            with open(os.path.join(data_dir, get_filename(day, hour)), "rb") as f:
                ds = pickle.load(f)

            lon_reg = np.linspace(0,360,360)
            lat_reg = np.linspace(-90,90,180)

            lon2d, lat2d = np.meshgrid(lon_reg, lat_reg)

            rebased_lon = np.where(grid.clon.values < 0.0, 360.0*math.pi/180.0 + grid.clon.values, grid.clon.values)
            print(rebased_lon.shape, rebased_lon[1])
            if np.nanmin(ds["sfc_pressure"]) > 2000.0:
                rescale = 100.0
            else:
                rescale = 1.0
            sp_interp = griddata(
                (rebased_lon*180.0/math.pi, grid.clat.values*180.0/math.pi),
                ds["sfc_pressure"]/rescale,
                (lon2d, lat2d),
                method="linear"
            )

            num_levels = ds["v"].shape[1]
            u_interp = griddata(
                (rebased_lon*180.0/math.pi, grid.clat.values*180.0/math.pi),
                ds["u"][:,num_levels-1],
                (lon2d, lat2d),
                method="linear"
            )
            v_interp = griddata(
                (rebased_lon*180.0/math.pi, grid.clat.values*180.0/math.pi),
                ds["v"][:,num_levels-1],
                (lon2d, lat2d),
                method="linear"
            )
        
        
        return lon2d, lat2d, sp_interp, u_interp, v_interp

    setup_contour_style()

    sp_colors = [(236, 0, 140), (166, 34, 142), (32, 65, 154), (0, 133, 204), (0, 174, 239), (0, 170, 79), (200, 218, 43), (255, 242, 0), (249, 158, 27), (237, 28, 36)]
    sp_colors_norm = [(r/255, g/255, b/255) for r, g, b in sp_colors]
    sp_cmap = LinearSegmentedColormap.from_list("sp_list", sp_colors_norm, N=len(sp_colors))
    def _do_plot(day: int, hour: int):
        lon2d, lat2d, sp_interp, u_interp, v_interp = _read_for_2dplot(day, hour)
        minvalue, maxvalue = np.nanmin(sp_interp), np.nanmax(sp_interp) + 1.e-11
        sp_colors_boundary = np.linspace(minvalue, maxvalue,100)
        sp_colors_tick = [minvalue, 0.75 * minvalue + 0.25 * maxvalue, 0.5 * (minvalue + maxvalue), 0.25 * minvalue + 0.75 * maxvalue, maxvalue]
        minvalue, maxvalue = np.nanmin(u_interp), np.nanmax(u_interp) + 1.e-11
        u_colors_boundary = np.linspace(minvalue, maxvalue,100)
        u_colors_tick = [minvalue, 0.75 * minvalue + 0.25 * maxvalue, 0.5 * (minvalue + maxvalue), 0.25 * minvalue + 0.75 * maxvalue, maxvalue]
        minvalue, maxvalue = np.nanmin(v_interp), np.nanmax(v_interp) + 1.e-11
        v_colors_boundary = np.linspace(minvalue, maxvalue,100)
        v_colors_tick = [minvalue, 0.75 * minvalue + 0.25 * maxvalue, 0.5 * (minvalue + maxvalue), 0.25 * minvalue + 0.75 * maxvalue, maxvalue]

        full_contourplot(lon2d, lat2d, sp_interp, plt.get_cmap("jet"), sp_colors_boundary, sp_colors_tick, f"Surface pressure diff (hPa) at day {day} hour {hour}", os.path.join(PLOT_DIR, f"sfc_pressure_day{day}_hour{hour}.png"))
        full_contourplot(lon2d, lat2d, u_interp, plt.get_cmap("jet"), u_colors_boundary, u_colors_tick, f"Surface U diff (m s-1) at day {day} hour {hour}", os.path.join(PLOT_DIR, f"sfc_u_day{day}_hour{hour}.png"))
        full_contourplot(lon2d, lat2d, v_interp, plt.get_cmap("jet"), v_colors_boundary, v_colors_tick, f"Surface V diff (m s-1) at day {day} hour {hour}", os.path.join(PLOT_DIR, f"sfc_v_day{day}_hour{hour}.png"))
    
    for i in range(24):
        _do_plot(1, i)
    
    for i in range(2,16):
        _do_plot(i, 0)
    
# plot_error()
plot_static_case()
# plot_baroclinic_case()


# In terms of wind speed, I only pickle u and v  and compare them, u and v   are quantities derived from vn  by rbf interpolaiton. I am adding 