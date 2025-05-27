# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os


current_folder = os.path.dirname(os.path.realpath(__file__))
os.environ["HDF5_PLUGIN_PATH"] = os.environ.get(
    "HDF5_PLUGIN_PATH", os.path.join(current_folder, "EBCC/src/build/lib")
)

import dask
import h5py
import humanize
import numcodecs_observers
import numpy as np
import pandas as pd
import utils
import xarray as xr
from EBCC.filter_wrapper import EBCC_Filter
from EBCC.src.zarr_filter import EBCCZarrFilter
from numcodecs_combinators.stack import CodecStack
from numcodecs_observers.bytesize import BytesizeObserver
from numcodecs_observers.walltime import WalltimeObserver
from numcodecs_wasm import WasmCodecInstructionCounterObserver
from scipy.spatial.distance import euclidean

from ..data_compression.similarity_metrics import calc_dwt_dist, calc_paa_dist


dask.config.set(array__chunk_size="4MiB")

ds = xr.open_dataset("tigge_pl_t_q_dx=2_2024_08_02.nc")
print(f"ds.nbytes = {humanize.naturalsize(ds.nbytes, binary=True)}")
utils.plot_data(ds, title_prefix="Uncompressed ")


################################################################################
# Run EBCC compressor (h5py)
################################################################################
for name, da in ds.items():
    print("EBCC Compressor (h5py) for variable:", name)
    try:
        os.remove("test.hdf5")
    except:
        pass
    f = h5py.File("test.hdf5", "a")

    data = ds[name].squeeze()
    data_32 = np.array(data, dtype=np.float32)

    ebcc_filter = EBCC_Filter(
        base_cr=100,
        height=data.shape[0],
        width=data.shape[1],
        data_dim=len(data.shape),
        residual_opt=("relative_error_target", 0.009),
    )

    f.create_dataset(f"{name}_compressed", shape=data.shape, **ebcc_filter)

    f[f"{name}_compressed"][:] = data_32
    uncompressed = f[f"{name}_compressed"][:]

    data_range = np.max(data) - np.min(data)
    max_error = np.max(np.abs(data - uncompressed))
    if data_range > 0:
        rel_error = max_error / data_range
        print("achieved max relative error:", rel_error.values)
    else:
        print("achieved max absolute error:", max_error.values)

    original_size = data.nbytes
    compressed_size = os.path.getsize("test.hdf5")

    ### Similarity measure
    euclid_dist_h5py = euclidean(np.concatenate(data), np.concatenate(uncompressed))
    paa_dist_h5py = calc_paa_dist(
        data.values, uncompressed, n_segments=round(data.shape[0] * 0.7)
    )  # number of segments should be low for data compression and higher for similarity measure
    dwt_dist_h5py = calc_dwt_dist(
        data.values, uncompressed, n_levels=3
    )  # decomposition levels 2-4 are meant for denoising
    best_dist = (
        (paa_dist_h5py, "paa")
        if (abs(paa_dist_h5py - euclid_dist_h5py) < abs(dwt_dist_h5py - euclid_dist_h5py))
        else (dwt_dist_h5py, "dwt")
    )
    print("Best similarity measure achieved with:", best_dist[1], "method")

    print(f"achieved compression ratio of {original_size/compressed_size}")
    print(f"achieved Euclidean distance of {euclid_dist_h5py}")
    print(f"achieved paa distance of {paa_dist_h5py}")
    print(f"achieved DWT distance of {dwt_dist_h5py}")

try:
    os.remove("test.hdf5")
except:
    pass


################################################################################
# Run EBCC compressor (Zarr)
################################################################################
for name, da in ds.items():
    print("EBCC Compressor (Zarr) for variable:", name)

    data = ds[name].squeeze()
    data_32 = np.array(data, dtype=np.float32)

    ebcc_filter = EBCC_Filter(
        base_cr=100,
        height=data.shape[0],
        width=data.shape[1],
        data_dim=len(data.shape),
        residual_opt=("relative_error_target", 0.009),
    )
    zarr_filter = EBCCZarrFilter(ebcc_filter.hdf_filter_opts)

    encoded = zarr_filter.encode(data_32.tobytes())
    decoded = np.frombuffer(zarr_filter.decode(encoded), dtype=np.float32).reshape(data.shape)
    uncompressed = decoded

    data_range = np.max(data) - np.min(data)
    max_error = np.max(np.abs(data - uncompressed))
    if data_range > 0:
        rel_error = max_error / data_range
        print("achieved max relative error:", rel_error.values)
    else:
        print("achieved max absolute error:", max_error.values)

    paa_dist_zarr = calc_paa_dist(data.values, uncompressed, n_segments=round(data.shape[0] * 0.7))
    dwt_dist_zarr = calc_dwt_dist(data.values, uncompressed, n_levels=3)
    euclid_dist_zarr = euclidean(np.concatenate(data.values), np.concatenate(uncompressed))

    print(f"achieved compression ratio of {len(data.values.tobytes())/len(encoded)}")
    print(f"achieved Euclidean distance of {euclid_dist_zarr}")
    print(f"achieved PAA distance of {paa_dist_zarr}")
    print(f"achieved DWT distance of {dwt_dist_zarr}")


################################################################################
# Run a Linear Quantization compressor
################################################################################
from numcodecs_wasm_linear_quantize import LinearQuantize
from numcodecs_wasm_zlib import Zlib


ds_linquant = {}
metrics_total_linquant = {}

for name, da in ds.items():
    linquant_metrics = dict(
        nbytes=BytesizeObserver(),
        instructions=WasmCodecInstructionCounterObserver(),
        timings=WalltimeObserver(),
    )

    linquant_compressor = CodecStack(
        LinearQuantize(bits=4, dtype=str(da.dtype)),
        Zlib(level=4),
    )

    with numcodecs_observers.observe(
        linquant_compressor,
        observers=linquant_metrics.values(),
    ) as linquant_compressor_:
        ds_linquant[name] = linquant_compressor_.encode_decode_data_array(da).compute()

    print(f"{da.long_name}" + ":")
    linquant_metrics = utils.format_compression_metrics(linquant_compressor, **linquant_metrics)
    print(linquant_metrics)

    metrics_total_linquant[name] = linquant_metrics.loc["Summary"]

linquant_compressor = str(linquant_compressor)

utils.plot_data(
    ds_linquant,
    title_prefix="Compressed ",
    title_postfix=f"\n{linquant_compressor}",
)

ds_linquant_error = {}
errors_linquant = {}
paa_dist_linquant = {}
dwt_dist_linquant = {}
euclid_dist_linquant = {}

for name, da in ds.items():
    with xr.set_options(keep_attrs=True):
        ds_linquant_error[name] = ds_linquant[name] - da
    errors_linquant[name] = utils.compute_relative_errors(ds_linquant[name], da)
    paa_dist_linquant[name] = calc_paa_dist(
        ds_linquant[name].values[0], da.values[0], n_segments=round(da.values[0].shape[0] * 0.7)
    )
    dwt_dist_linquant[name] = calc_dwt_dist(ds_linquant[name].values[0], da.values[0], n_levels=3)
    euclid_dist_linquant[name] = euclidean(
        np.concatenate(ds_linquant[name].values[0]), np.concatenate(da.values[0])
    )

utils.plot_data(
    ds_linquant_error,
    title_prefix="Compression Error for ",
    title_postfix=f"\n{linquant_compressor}",
    error=True,
)


################################################################################
# Run the BitRound compressor
################################################################################
from numcodecs_wasm_bit_round import BitRound
from numcodecs_wasm_zlib import Zlib


ds_bitround = {}
metrics_total_bitround = {}

bitround_compressor = CodecStack(
    BitRound(keepbits=6),
    Zlib(level=9),
)

for name, da in ds.items():
    bitround_metrics = dict(
        nbytes=BytesizeObserver(),
        instructions=WasmCodecInstructionCounterObserver(),
        timings=WalltimeObserver(),
    )

    with numcodecs_observers.observe(
        bitround_compressor,
        observers=bitround_metrics.values(),
    ) as bitround_compressor_:
        ds_bitround[name] = bitround_compressor_.encode_decode_data_array(da).compute()

    print(f"{da.long_name}" + ":")
    bitround_metrics = utils.format_compression_metrics(bitround_compressor, **bitround_metrics)
    print(bitround_metrics)

    metrics_total_bitround[name] = bitround_metrics.loc["Summary"]

bitround_compressor = str(bitround_compressor)

ds_bitround_error = {}
errors_bitround = {}
paa_dist_bitround = {}
dwt_dist_bitround = {}
euclid_dist_bitround = {}

for name, da in ds.items():
    with xr.set_options(keep_attrs=True):
        ds_bitround_error[name] = ds_bitround[name] - da
    errors_bitround[name] = utils.compute_relative_errors(ds_bitround[name], da)
    paa_dist_bitround[name] = calc_paa_dist(
        ds_bitround[name].values[0], da.values[0], n_segments=round(da.values[0].shape[0] * 0.7)
    )
    dwt_dist_bitround[name] = calc_dwt_dist(ds_bitround[name].values[0], da.values[0], n_levels=3)
    euclid_dist_bitround[name] = euclidean(
        np.concatenate(ds_bitround[name].values[0]), np.concatenate(da.values[0])
    )


################################################################################
# Run the transform-based ZFP compressor
################################################################################
from numcodecs_wasm_asinh import Asinh
from numcodecs_wasm_zfp import Zfp


ds_zfp = {}
metrics_total_zfp = {}

zfp_compressor = CodecStack(
    Asinh(linear_width=1.0),
    Zfp(mode="fixed-accuracy", tolerance=1e-3),
)

for name, da in ds.items():
    zfp_metrics = dict(
        nbytes=BytesizeObserver(),
        instructions=WasmCodecInstructionCounterObserver(),
        timings=WalltimeObserver(),
    )

    with numcodecs_observers.observe(
        zfp_compressor,
        observers=zfp_metrics.values(),
    ) as zfp_compressor_:
        ds_zfp[name] = zfp_compressor_.encode_decode_data_array(da).compute()

    print(f"{da.long_name}" + ":")
    zfp_metrics = utils.format_compression_metrics(zfp_compressor, **zfp_metrics)
    print(zfp_metrics)

    metrics_total_zfp[name] = zfp_metrics.loc["Summary"]

zfp_compressor = str(zfp_compressor)

ds_zfp_error = {}
errors_zfp = {}
paa_dist_zfp = {}
dwt_dist_zfp = {}
euclid_dist_zfp = {}

for name, da in ds.items():
    with xr.set_options(keep_attrs=True):
        ds_zfp_error[name] = ds_zfp[name] - da
    errors_zfp[name] = utils.compute_relative_errors(ds_zfp[name], da)
    paa_dist_zfp[name] = calc_paa_dist(
        ds_zfp[name].values[0], da.values[0], n_segments=round(da.values[0].shape[0] * 0.7)
    )
    dwt_dist_zfp[name] = calc_dwt_dist(ds_zfp[name].values[0], da.values[0], n_levels=3)
    euclid_dist_zfp[name] = euclidean(
        np.concatenate(ds_zfp[name].values[0]), np.concatenate(da.values[0])
    )


################################################################################
# Run the prediction-based SZ3 compressor
################################################################################
from numcodecs_wasm_sz3 import Sz3


ds_sz3 = {}
metrics_total_sz3 = {}

sz3_compressor = CodecStack(Sz3(eb_mode="rel", eb_rel=1e-3))

for name, da in ds.items():
    sz3_metrics = dict(
        nbytes=BytesizeObserver(),
        instructions=WasmCodecInstructionCounterObserver(),
        timings=WalltimeObserver(),
    )

    with numcodecs_observers.observe(
        sz3_compressor,
        observers=sz3_metrics.values(),
    ) as sz3_compressor_:
        ds_sz3[name] = sz3_compressor_.encode_decode_data_array(da).compute()

    print(f"{da.long_name}" + ":")
    sz3_metrics = utils.format_compression_metrics(sz3_compressor, **sz3_metrics)
    print(sz3_metrics)

    metrics_total_sz3[name] = sz3_metrics.loc["Summary"]

sz3_compressor = str(sz3_compressor)

ds_sz3_error = {}
errors_sz3 = {}
paa_dist_sz3 = {}
dwt_dist_sz3 = {}
euclid_dist_sz3 = {}

for name, da in ds.items():
    with xr.set_options(keep_attrs=True):
        ds_sz3_error[name] = ds_sz3[name] - da
    errors_sz3[name] = utils.compute_relative_errors(ds_sz3[name], da)
    paa_dist_sz3[name] = calc_paa_dist(
        ds_sz3[name].values[0], da.values[0], n_segments=round(da.values[0].shape[0] * 0.7)
    )
    dwt_dist_sz3[name] = calc_dwt_dist(ds_sz3[name].values[0], da.values[0], n_levels=3)
    euclid_dist_sz3[name] = euclidean(
        np.concatenate(ds_sz3[name].values[0]), np.concatenate(da.values[0])
    )

################################################################################
# Overview
################################################################################
data = []

compressors = {
    linquant_compressor: (
        errors_linquant,
        metrics_total_linquant,
        paa_dist_linquant,
        dwt_dist_linquant,
        euclid_dist_linquant,
    ),
    bitround_compressor: (
        errors_bitround,
        metrics_total_bitround,
        paa_dist_bitround,
        dwt_dist_bitround,
        euclid_dist_bitround,
    ),
    zfp_compressor: (errors_zfp, metrics_total_zfp, paa_dist_zfp, dwt_dist_zfp, euclid_dist_zfp),
    sz3_compressor: (errors_sz3, metrics_total_sz3, paa_dist_sz3, dwt_dist_sz3, euclid_dist_sz3),
}

for compressor_name, (errors, stats, paa_dist, dwt_dist, euclid_dist) in compressors.items():
    for variable, error_data in errors.items():
        row = {
            "Compressor": compressor_name,
            "Variable": variable,
            "Compression Ratio [raw B / enc B]": stats[variable][
                "compression ratio [raw B / enc B]"
            ],
            "Euclidean distance": euclid_dist[variable],
            "PAA distance": paa_dist[variable],
            "DWT distance": dwt_dist[variable],
            "L1 Error": error_data.get("Relative_Error_L1", None),
            "L2 Error": error_data.get("Relative_Error_L2", None),
            "Linf Error": error_data.get("Relative_Error_Linf", None),
            "Encode Instructions [# / raw B]": stats[variable]["encode instructions [#/B]"],
            "Decode Instructions [# / raw B]": stats[variable]["decode instructions [#/B]"],
            "Encode Throughout [raw GB / s]": stats[variable]["encode throughout [raw GB/s]"],
            "Decode Throughout [raw GB / s]": stats[variable]["decode throughout [raw GB/s]"],
        }
        data.append(row)


df = pd.DataFrame(data).set_index(["Compressor", "Variable"])
print(df)
