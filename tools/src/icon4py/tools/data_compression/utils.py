# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import earthkit
import numpy as np


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

    return {
        "Relative_Error_L1": relative_error_L1,
        "Relative_Error_L2": relative_error_L2,
        "Relative_Error_Linf": relative_error_Linf,
    }


def plot_data(data: dict, title_prefix="", title_postfix="", error=False):
    return

    UNITS = dict(t="degC")
    DIVERGENCE_POINT = dict(t=0.0)

    for name, data in data.items():
        units = None if error else UNITS.get(name, None)
        divergence_point = 0.0 if error else DIVERGENCE_POINT.get(name, None)

        source = earthkit.plots.sources.XarraySource(data)

        # compute the default style that earthkit.maps would apply
        style = copy.deepcopy(
            earthkit.plots.styles.auto.guess_style(
                source,
                units=units or source.units,
            )
        )

        # modify the style levels to get a smoother colourbar
        style._levels = earthkit.plots.styles.levels.Levels(
            earthkit.plots.styles.levels.auto_range(
                style.convert_units(data.values, source.units),
                divergence_point=divergence_point,
                n_levels=256,
            )
        )
        style._legend_kwargs["ticks"] = earthkit.plots.styles.levels.auto_range(
            style.convert_units(data.values, source.units),
            divergence_point=divergence_point,
            n_levels=10,
        )

        # force the colourmap to coolwarm for error plots
        style._colors = "coolwarm" if error else style._colors

        # quickplot with the modified style
        chart = earthkit.plots.quickplot(
            data,
            units=units,
            style=style,
        )

        chart.title(f"{title_prefix}{{variable_name}} on {{time:%d.%m.%Y at %H:%M}}{title_postfix}")

        chart.show()


def open_dataset(path: Path, **kwargs) -> "xarray.Dataset":
    import xarray as xr

    path = Path(path)

    if path.suffix == ".grib" or kwargs.get("engine", None) == "cfgrib":
        if "engine" not in kwargs:
            kwargs["engine"] = "cfgrib"
        if "backend_kwargs" not in kwargs:
            kwargs["backend_kwargs"] = dict()
        if "indexpath" not in kwargs["backend_kwargs"]:
            # cfgrib creates index files right next to the data file,
            #  which may be in a read-only file system
            kwargs["backend_kwargs"]["indexpath"] = ""

    if "".join(path.suffixes).endswith(".zarr.zip"):
        if "engine" not in kwargs:
            kwargs["engine"] = "zarr"

    if "chunks" not in kwargs:
        kwargs["chunks"] = "auto"

    if "cache" not in kwargs:
        kwargs["cache"] = False

    ds = xr.open_dataset(str(path), **kwargs)
    ds.attrs["path"] = str(path)

    return ds


async def mount_user_local_file() -> Path:
    import ipyfilite

    uploader = ipyfilite.FileUploadLite()
    await uploader.request()
    uploader.close()

    return uploader.value[0].path


def mount_http_file(url: str, name: Optional[str] = None) -> Path:
    import ipyfilite

    if name is None:
        name = _get_name_from_url(url)

    http_file = ipyfilite.HTTPFileIO(name=name, url=url)

    return http_file.path


def _get_name_from_url(url: str) -> str:
    from urllib.parse import unquote as urlunquote, urlparse

    return urlunquote(Path(urlparse(url).path).name)


async def download_dataset_as_zarr(
    ds: "xarray.Dataset",
    name: str,
    *,
    filters: None | dict[str, "None | zarr.abc.codec.ArrayArrayCodec"],
    serializer: "None | zarr.abc.codec.ArrayBytesCodec",
    compressors: None | dict[str, "None | zarr.abc.codec.BytesBytesCodec"],
    zip_compression: int = 0,
):
    import ipyfilite
    import zarr

    name_suffix = "".join(Path(name).suffixes)

    # Ensure that the file path is easily recognisable as a zipped zarr file
    if name_suffix.endswith(".zarr.zip"):
        pass
    elif name_suffix.endswith(".zarr"):
        name = f"{name}.zip"
    elif name_suffix.endswith(".zip"):
        name = f"{Path(name).stem}.zarr.zip"
    else:
        name = f"{name}.zarr.zip"

    async with ipyfilite.FileDownloadPathLite(name) as path:
        store = zarr.storage.MemoryStore()
        chunk_store = zarr.storage.ZipStore(
            str(path),
            compression=zip_compression,
            allowZip64=True,
            mode="x",
        )

        filters = filters if isinstance(filters, dict) else {var: filters for var in ds}
        serializer = serializer if isinstance(serializer, dict) else {var: serializer for var in ds}
        compressors = (
            compressors if isinstance(compressors, dict) else {var: compressors for var in ds}
        )

        encoding = dict()
        for var in ds:
            encoding[var] = dict(
                filters=filters[var],
                compressors=compressors[var],
                **({"serializer": serializer[var]} if serializer[var] is not None else {}),
            )

        ds.to_zarr(store=store, mode="w-", encoding=encoding)

        async for key in store.list():
            await chunk_store.set(
                key, await store.get(key, zarr.core.buffer.core.default_buffer_prototype())
            )

        store.close()
        chunk_store.close()


@asynccontextmanager
async def file_download_path(name: str) -> Path:
    import ipyfilite

    try:
        async with ipyfilite.FileDownloadPathLite(name) as path:
            yield path
    finally:
        pass


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


def kerchunk_autochunk(kc: dict, *, chunk_size: int) -> dict:
    import json

    import kerchunk
    import numpy as np
    import sympy

    kc_new = kc

    # iterate over all variables
    for k, v in kc["refs"].items():
        if Path(k).name == ".zarray":
            v = json.loads(v)

            # we cannot chunk compressed arrays
            if v["compressor"] is not None:
                continue

            chunks = v["chunks"]

            # calculate the size of the chunk
            nbytes_chunk = np.dtype(v["dtype"]).itemsize * int(np.prod(chunks, dtype=np.uint64))

            # skip to the next variable if the chunk is already small enough
            if nbytes_chunk <= chunk_size:
                continue

            for i, c in enumerate(chunks):
                # factorize the remaining chunk size
                factors = []
                for f, c in sympy.factorint(c).items():
                    for _ in range(c):
                        factors.append(f)
                factors.sort()

                if len(factors) == 0:
                    continue

                # find the smallest factor that would reduce the chunk size
                #  below the limit, or the total factor
                factor = 1
                for f in factors:
                    factor *= f
                    if (nbytes_chunk // factor) <= chunk_size:
                        break

                # use kerchunk to apply the new chunking
                kc_new = kerchunk.utils.subchunk(
                    kc_new,
                    Path(k).parts[0],
                    factor,
                )

                chunks[i] = chunks[i] // factor
                nbytes_chunk = nbytes_chunk // factor

                # break if the chunk is now small enough
                if nbytes_chunk <= chunk_size:
                    break

    return kc_new
