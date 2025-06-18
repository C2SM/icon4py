# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import sys
import math
import traceback
from collections import OrderedDict
from collections.abc import Sequence
import click
import humanize
import numpy as np
import xarray as xr
import yaml
import pywt
import zarr
import dask as da
import numcodecs
import numcodecs.zarr3
import zfpy


def open_netcdf(netcdf_file: str, field_to_compress: str):
    ds = xr.open_dataset(netcdf_file)

    if field_to_compress not in ds.data_vars:
        click.echo(f"Field {field_to_compress} not found in NetCDF file.")
        click.echo(f"Available fields in the dataset: {list(ds.data_vars.keys())}.")
        click.echo("Aborting...")
        sys.exit(1)

    click.echo(f"netcdf_file.nbytes = {humanize.naturalsize(ds.nbytes, binary=True)}")
    click.echo(
        f"field_to_compress.nbytes = {humanize.naturalsize(ds[field_to_compress].nbytes, binary=True)}"
    )

    return ds


def open_zarr_zipstore(zarr_zipstore_file: str):
    store = zarr.storage.ZipStore(zarr_zipstore_file, read_only=True)
    return zarr.open_group(store, mode='r')


def compress_with_zarr(data, netcdf_file, field_to_compress, filters, compressors, serializer='auto', echo=True):
    store = zarr.storage.ZipStore(f"{netcdf_file}.zarr.zip", mode='w')
    z = zarr.create_array(
        store=store,
        name=field_to_compress,
        data=data,
        chunks="auto",
        filters=filters,
        compressors=compressors,
        serializer=serializer,
        )
    
    info_array = z.info_complete()
    compression_ratio = info_array._count_bytes / info_array._count_bytes_stored
    click.echo(80* "-") if echo else None
    click.echo(info_array) if echo else None
    
    pprint_, errors = compute_relative_errors(z[:], data)
    click.echo(80* "-") if echo else None
    click.echo(pprint_) if echo else None
    click.echo(80* "-") if echo else None

    dwt_dist = calc_dwt_dist(z[:], data)
    click.echo("DWT Distance: ", dwt_dist) if echo else None
    click.echo(80* "-") if echo else None

    store.close()

    return compression_ratio, errors, dwt_dist


def ordered_yaml_loader():
    class OrderedLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return OrderedDict(loader.construct_pairs(node))

    OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)

    return OrderedLoader


def get_filter_parameters(parameters_file: str, filter_name: str):
    with open(parameters_file, "r") as f:
        params = yaml.load(f, Loader=ordered_yaml_loader())

    try:
        filter_config = params[filter_name]["params"]
        return tuple(filter_config.values())

    except Exception:
        click.echo("An unexpected error occurred:", err=True)
        traceback.print_exc(file=sys.stderr)


def compute_relative_errors(da_compressed, da):
    da_error = da_compressed - da

    norm_L1_error = np.abs(da_error).sum().values
    norm_L2_error = np.sqrt((da_error**2).sum().values)
    norm_Linf_error = np.abs(da_error).max().values

    norm_L1_original = np.abs(da).sum().values
    norm_L2_original = np.sqrt((da**2).sum().values)
    norm_Linf_original = np.abs(da).max().values

    relative_error_L1 = norm_L1_error / norm_L1_original
    relative_error_L2 = norm_L2_error / norm_L2_original
    relative_error_Linf = norm_Linf_error / norm_Linf_original

    errors = {
        "Relative_Error_L1": relative_error_L1,
        "Relative_Error_L2": relative_error_L2,
        "Relative_Error_Linf": relative_error_Linf,
    }
    errors_ = {k: f"{v:.3e}" for k, v in errors.items()}

    return "\n".join(f"{k:20s}: {v}" for k, v in errors_.items()), errors


def calc_dwt_dist(input_1, input_2, n_levels=4, wavelet="haar"):
    dwt_data_1 = pywt.wavedec(input_1, wavelet=wavelet, level=n_levels)
    dwt_data_2 = pywt.wavedec(input_2, wavelet=wavelet, level=n_levels)
    distances = [np.linalg.norm(c1 - c2) for c1, c2 in zip(dwt_data_1, dwt_data_2)]
    dwt_distance = np.sqrt(sum(d**2 for d in distances))
    return dwt_distance


def compressor_space(da):
    # https://numcodecs.readthedocs.io/en/stable/zarr3.html#compressors-bytes-to-bytes-codecs
    
    # TODO: take care of integer data types
    compressor_space = []
    
    _COMPRESSORS = [numcodecs.zarr3.Blosc, numcodecs.zarr3.LZ4, numcodecs.zarr3.Zstd, numcodecs.zarr3.Zlib, numcodecs.zarr3.GZip, numcodecs.zarr3.BZ2, numcodecs.zarr3.LZMA]
    for compressor in _COMPRESSORS:
        if compressor == numcodecs.zarr3.Blosc:
            for cname in numcodecs.blosc.list_compressors():
                for clevel in inclusive_range(1,9,4):
                    for shuffle in inclusive_range(0,2):
                        compressor_space.append(numcodecs.zarr3.Blosc(cname=cname, clevel=clevel, shuffle=shuffle))
        elif compressor == numcodecs.zarr3.LZ4:
            # The larger the acceleration value, the faster the algorithm, but also the lesser the compression
            for acceleration in inclusive_range(0, 16, 6):
                compressor_space.append(numcodecs.zarr3.LZ4(acceleration=acceleration))
        elif compressor == numcodecs.zarr3.Zstd:
            for level in inclusive_range(-7, 22, 10):
                compressor_space.append(numcodecs.zarr3.Zstd(level=level))
        elif compressor == numcodecs.zarr3.Zlib:
            for level in inclusive_range(1,9,4):
                compressor_space.append(numcodecs.zarr3.Zlib(level=level))
        elif compressor == numcodecs.zarr3.GZip:
            for level in inclusive_range(1,9,4):
                compressor_space.append(numcodecs.zarr3.GZip(level=level))
        elif compressor == numcodecs.zarr3.BZ2:
            for level in inclusive_range(1,9,4):
                compressor_space.append(numcodecs.zarr3.BZ2(level=level))
        elif compressor == numcodecs.zarr3.LZMA:
            # https://docs.python.org/3/library/lzma.html
            for preset in inclusive_range(1,9,4):
                compressor_space.append(numcodecs.zarr3.LZMA(preset=preset))

    return compressor_space


def filter_space(da):
    # https://numcodecs.readthedocs.io/en/stable/zarr3.html#filters-array-to-array-codecs
    
    # TODO: take care of integer data types
    filter_space = []
    
    _FILTERS = [numcodecs.zarr3.Delta, numcodecs.zarr3.BitRound, numcodecs.zarr3.FixedScaleOffset, numcodecs.zarr3.Quantize]
    for filter in _FILTERS:
        if filter == numcodecs.zarr3.Delta:
            filter_space.append(numcodecs.zarr3.Delta(dtype=str(da.dtype)))
        elif filter == numcodecs.zarr3.BitRound:
            # If keepbits is equal to the maximum allowed for the data type, this is equivalent to no transform.
            for keepbits in valid_keepbits_for_bitround(da, step=9):
                filter_space.append(numcodecs.zarr3.BitRound(keepbits=keepbits))
        elif filter == numcodecs.zarr3.FixedScaleOffset:
            # TODO: scale should be 1/current_scale (check for division by zero)
            # Setting o=mean(x) and s=std(x) normalizes that data
            filter_space.append(numcodecs.zarr3.FixedScaleOffset(offset=da.mean(skipna=True).compute().item(), scale=da.std(skipna=True).compute().item(), dtype=str(da.dtype)))
            # Setting o=min(x) and s=max(x)−min(x)standardizes the data
            filter_space.append(
                numcodecs.zarr3.FixedScaleOffset(offset=da.min(skipna=True).compute().item(), scale=da.max(skipna=True).compute().item()-da.min(skipna=True).compute().item(), dtype=str(da.dtype)))
        elif filter == numcodecs.zarr3.Quantize:
            # Same as BitRound
            for digits in valid_keepbits_for_bitround(da, step=9):
                filter_space.append(numcodecs.zarr3.Quantize(digits=digits, dtype=str(da.dtype)))

    return filter_space


def serializer_space(da):
    # https://numcodecs.readthedocs.io/en/stable/zarr3.html#serializers-array-to-bytes-codecs
    
    # TODO: take care of integer data types
    serializer_space = []
    
    _SERIALIZERS = [numcodecs.zarr3.PCodec, numcodecs.zarr3.ZFPY]
    for serializer in _SERIALIZERS:
        if serializer == numcodecs.zarr3.PCodec:
            for level in inclusive_range(0, 12, 4):  # where 12 take the longest and compresses the most
                for mode_spec in ["auto", "classic"]:
                    for delta_spec in ["auto", "none", "try_consecutive", "try_lookback"]:
                        for delta_encoding_order in inclusive_range(0,7,4):
                            serializer_space.append(numcodecs.zarr3.PCodec(
                                    level=level,
                                    mode_spec=mode_spec,
                                    delta_spec=delta_spec,
                                    delta_encoding_order=delta_encoding_order if delta_spec in ["try_consecutive", "auto"] else None
                                )
                            )
        elif serializer == numcodecs.zarr3.ZFPY:
            # https://github.com/zarr-developers/numcodecs/blob/main/numcodecs/zfpy.py
            for mode in [zfpy.mode_fixed_accuracy,
                         zfpy.mode_fixed_rate,
                         zfpy.mode_fixed_precision]:                
                for compress_param_num in range(3):
                    if mode == zfpy.mode_fixed_accuracy:
                        serializer_space.append(numcodecs.zarr3.ZFPY(
                            mode=mode, tolerance=compute_fixed_accuracy_param(compress_param_num)
                        ))
                    elif mode == zfpy.mode_fixed_precision:
                        serializer_space.append(numcodecs.zarr3.ZFPY(
                            mode=mode, precision=compute_fixed_precision_param(compress_param_num)
                        ))
                    elif mode == zfpy.mode_fixed_rate:
                        serializer_space.append(numcodecs.zarr3.ZFPY(
                            mode=mode, rate=compute_fixed_rate_param(compress_param_num)
                        ))

    return serializer_space


def valid_keepbits_for_bitround(xr_dataarray, step=1):
    dtype = xr_dataarray.dtype
    if np.issubdtype(dtype, np.float64):
        return inclusive_range(1, 52, step)  # float64 mantissa is 52 bits
    elif np.issubdtype(dtype, np.float32):
        return inclusive_range(1, 23, step)  # float32 mantissa is 23 bits
    else:
        raise TypeError(f"Unsupported dtype '{dtype}'. BitRound only supports float32 and float64.")


def compute_fixed_precision_param(param: int) -> int:
    # https://github.com/LLNL/zfp/tree/develop/tests/python
    return 1 << (param + 3)

def compute_fixed_rate_param(param: int) -> int:
    # https://github.com/LLNL/zfp/tree/develop/tests/python
    return 1 << (param + 3)

def compute_fixed_accuracy_param(param: int) -> float:
    # https://github.com/LLNL/zfp/tree/develop/tests/python
    return math.ldexp(1.0, -(1 << param))


def inclusive_range(start, end, step=1):
    if step == 0:
        raise ValueError("step must not be zero")

    values = []
    i = start
    if step > 0:
        while i <= end:
            values.append(i)
            i += step
        if values[-1] != end:
            values.append(end)
    else:
        while i >= end:
            values.append(i)
            i += step
        if values[-1] != end:
            values.append(end)

    return values


def normalize(val, min_val, max_val):
    if max_val == min_val:
        return 0.0  # avoid division by zero
    return (val - min_val) / (max_val - min_val)


def format_compression_metrics(
    codecs: Sequence["numcodecs.abc.Codec"],
    *,
    nbytes: "numcodecs_observers.bytesize.BytesizeObserver",
    instructions: "None | numcodecs_wasm.WasmCodecInstructionCounterObserver" = None,
    timings: "None | numcodecs_observers.walltime.WalltimeObserver" = None,
):
    import pandas as pd
    from numcodecs_observers.hash import HashableCodec

    codecs = tuple(codecs)

    encoded_bytes = {c: sum(e.post for e in es) for c, es in nbytes.encode_sizes.items()}
    decoded_bytes = {c: sum(d.post for d in ds) for c, ds in nbytes.decode_sizes.items()}

    table = pd.DataFrame(
        {
            "Codec": [str(c) for c in codecs] + ["Summary"],
            "compression ratio [raw B / enc B]": [
                round(decoded_bytes[HashableCodec(c)] / encoded_bytes[HashableCodec(c)], 2)
                for c in codecs
            ]
            + (
                [
                    round(
                        decoded_bytes[HashableCodec(codecs[0])]
                        / encoded_bytes[HashableCodec(codecs[-1])],
                        2,
                    )
                ]
                if len(codecs) > 0
                else [1.0]
            ),
        }
    ).set_index(["Codec"])

    if instructions is not None:
        table["encode instructions [#/B]"] = [
            (
                round(
                    sum(instructions.encode_instructions[HashableCodec(c)])
                    / decoded_bytes[HashableCodec(c)],
                    1,
                )
                if HashableCodec(c) in instructions.encode_instructions
                else "<unknown>"
            )
            for c in codecs
        ] + (
            [
                round(
                    sum(sum(instructions.encode_instructions[HashableCodec(c)]) for c in codecs)
                    / decoded_bytes[HashableCodec(codecs[0])],
                    1,
                )
                if all(HashableCodec(c) in instructions.encode_instructions for c in codecs)
                else "<unknown>"
            ]
            if len(codecs) > 0
            else [0.0]
        )

        table["decode instructions [#/B]"] = [
            (
                round(
                    sum(instructions.decode_instructions[HashableCodec(c)])
                    / decoded_bytes[HashableCodec(c)],
                    1,
                )
                if HashableCodec(c) in instructions.decode_instructions
                else "<unknown>"
            )
            for c in codecs
        ] + (
            [
                round(
                    sum(sum(instructions.decode_instructions[HashableCodec(c)]) for c in codecs)
                    / decoded_bytes[HashableCodec(codecs[0])],
                    1,
                )
                if all(HashableCodec(c) in instructions.decode_instructions for c in codecs)
                else "<unknown>"
            ]
            if len(codecs) > 0
            else [0.0]
        )

    if timings is not None:
        table["encode throughout [raw GB/s]"] = [
            round(
                1e-9
                * decoded_bytes[HashableCodec(c)]
                / sum(timings.encode_times[HashableCodec(c)]),
                2,
            )
            for c in codecs
        ] + (
            [
                round(
                    1e-9
                    * decoded_bytes[HashableCodec(codecs[0])]
                    / sum(sum(timings.encode_times[HashableCodec(c)]) for c in codecs),
                    2,
                )
            ]
            if len(codecs) > 0
            else [0.0]
        )

        table["decode throughout [raw GB/s]"] = [
            round(
                1e-9
                * decoded_bytes[HashableCodec(c)]
                / sum(timings.decode_times[HashableCodec(c)]),
                2,
            )
            for c in codecs
        ] + (
            [
                round(
                    1e-9
                    * decoded_bytes[HashableCodec(codecs[0])]
                    / sum(sum(timings.decode_times[HashableCodec(c)]) for c in codecs),
                    2,
                )
            ]
            if len(codecs) > 0
            else [0.0]
        )

    return table
