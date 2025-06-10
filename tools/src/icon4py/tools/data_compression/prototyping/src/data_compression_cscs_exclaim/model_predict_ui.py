# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import streamlit as st
import xarray as xr
from data_compression_cscs_exclaim import utils
from data_compression_cscs_exclaim.cli import calc_dwt_dist
from numcodecs_combinators.stack import CodecStack
from numcodecs_wasm_asinh import Asinh
from numcodecs_wasm_bit_round import BitRound
from numcodecs_wasm_linear_quantize import LinearQuantize
from numcodecs_wasm_sz3 import Sz3
from numcodecs_wasm_zfp import Zfp
from numcodecs_wasm_zlib import Zlib


#  prototyping % streamlit run ./src/data_compression_cscs_exclaim/model_predict_ui.py streamlit --server.maxUploadSize=1200
#  prototyping % data_compression_cscs_exclaim models_evaluation netCDF_files/tigge_pl_t_q_dx=2_2024_08_02.nc t parameters.yaml

# Page title
st.title("Upload a file and evaluate compressors")

uploaded_file = st.file_uploader("Choose a netcdf file")

# Dropdown compressors options
options = [
    "linear_quantization_zlib_compressors",
    "bitround_zlib_compressors",
    "zfp_asinh_compressors",
    "sz3_eb_compressors",
]
selected_option_compressor = st.multiselect("Choose compression method(s)", options)

if uploaded_file is not None and uploaded_file.name.endswith(".nc"):
    if len(selected_option_compressor) > 0:
        netcdf_file = uploaded_file
        netcdf_file_xr = xr.open_dataset(netcdf_file)
        options = [var for var in netcdf_file_xr.data_vars]
        # Automatically update session state
        if "selected_column" not in st.session_state:
            st.session_state.selected_column = options[0]

        field_to_compress = st.selectbox(
            "Choose a field:",
            options=options,
            index=options.index(st.session_state.selected_column),
            key="selected_column",
        )

        st.write("Selected column:", field_to_compress)

        ds = utils.open_netcdf(netcdf_file, field_to_compress)
        ds_coords = dict(ds.coords).keys()
        lat_key = [key for key in ds_coords if key.startswith("lat")][0]
        lon_key = [key for key in ds_coords if key.startswith("lon")][0]
        lat_upper = round(ds[field_to_compress][lat_key].shape[0] * 0.2)
        lon_upper = round(ds[field_to_compress][lon_key].shape[0] * 0.2)
        ds = ds.isel(latitude=slice(0, lat_upper), longitude=slice(0, lon_upper))
        dwt_dists = {}
        edit_params = st.checkbox("Adjust parameters")

        # Perform action based on dropdown
        if "linear_quantization_zlib_compressors" in selected_option_compressor:
            linear_quantization_bits, zlib_level = utils.get_filter_parameters(
                "parameters.yaml", "linear_quantization_zlib_compressors"
            )
            ds_linquant = {}
            zlib_level_lq = zlib_level
            if edit_params:
                st.markdown(
                    f"<h1 style='font-size:{15}px; '>Linear Quantization compressor parameters</h1>",
                    unsafe_allow_html=True,
                )
                linear_quantization_bits = st.number_input(
                    "bits", min_value=1, max_value=64, value=4
                )
                zlib_level_lq = st.number_input("zlib_level_lq", min_value=0, max_value=9, value=9)
            linquant_compressor = CodecStack(
                LinearQuantize(
                    bits=linear_quantization_bits, dtype=str(ds[field_to_compress].dtype)
                ),
                Zlib(level=zlib_level_lq),
            )

            ds_linquant[field_to_compress] = linquant_compressor.encode_decode_data_array(
                ds[field_to_compress]
            ).compute()
            lquant_dwt_distances = calc_dwt_dist(
                ds_linquant[field_to_compress], ds[field_to_compress], n_levels=4
            )
            dwt_dists["lquant_dwt_distances"] = lquant_dwt_distances
            st.write("Linear Quantization DWT disctance: " + f"{lquant_dwt_distances}")

        if "bitround_zlib_compressors" in selected_option_compressor:
            bitround_bits, zlib_level = utils.get_filter_parameters(
                "parameters.yaml", "bitround_zlib_compressors"
            )
            ds_bitround = {}
            zlib_level_br = zlib_level
            if edit_params:
                st.markdown(
                    f"<h1 style='font-size:{15}px; '>Bit Round compressor parameters</h1>",
                    unsafe_allow_html=True,
                )
                bitround_bits = st.number_input("bitround_bits", min_value=1, max_value=52, value=6)
                zlib_level_br = st.number_input("zlib_level_br", min_value=0, max_value=9, value=9)
            bitround_compressor = CodecStack(
                BitRound(keepbits=bitround_bits),
                Zlib(level=zlib_level_br),
            )
            ds_bitround[field_to_compress] = bitround_compressor.encode_decode_data_array(
                ds[field_to_compress]
            ).compute()
            bitround_dwt_distances = calc_dwt_dist(
                ds_bitround[field_to_compress], ds[field_to_compress], n_levels=4
            )
            dwt_dists["bitround_dwt_distances"] = bitround_dwt_distances
            st.write("Bit Round DWT disctance: " + f"{bitround_dwt_distances}")

        if "zfp_asinh_compressors" in selected_option_compressor:
            asinh_linear_width, zfp_mode, zfp_tolerance = utils.get_filter_parameters(
                "parameters.yaml", "zfp_asinh_compressors"
            )
            ds_zfp = {}
            if edit_params:
                st.markdown(
                    f"<h1 style='font-size:{15}px; '>Zfp compressor parameters</h1>",
                    unsafe_allow_html=True,
                )
                asinh_linear_width = st.number_input(
                    "asinh_linear_width", min_value=1, max_value=100, value=9
                )
                zfp_tolerance = st.number_input(
                    "zfp_tolerance", min_value=0.0, max_value=1.0, value=0.001, format="%.4f"
                )
                # zfp_mode = st.selectbox(
                #     "zfp_mode",
                #     options=["fixed-accuracy", "fixed-precision", "fixed-rate", "expert", "reversible"],
                #     index=options.index(st.session_state.selected_column),
                #     key="zfp_mode_selected",
                # )
            zfp_compressor = CodecStack(
                Asinh(linear_width=asinh_linear_width),
                Zfp(mode=zfp_mode, tolerance=zfp_tolerance),
            )
            ds_zfp[field_to_compress] = zfp_compressor.encode_decode_data_array(
                ds[field_to_compress]
            ).compute()
            zfp_dwt_distances = calc_dwt_dist(
                ds_zfp[field_to_compress], ds[field_to_compress], n_levels=4
            )
            dwt_dists["zfp_dwt_distances"] = zfp_dwt_distances
            st.write("Zfp DWT disctance: " + f"{zfp_dwt_distances}")

        if "sz3_eb_compressors" in selected_option_compressor:
            sz3_eb_mode, sz3_eb_rel = utils.get_filter_parameters(
                "parameters.yaml", "sz3_eb_compressors"
            )
            ds_sz3 = {}
            if edit_params:
                st.markdown(
                    f"<h1 style='font-size:{15}px; '>Sz3 compressor parameters</h1>",
                    unsafe_allow_html=True,
                )
                sz3_eb_rel = st.number_input(
                    "sz3_eb_rel", min_value=0.0, max_value=1.0, value=0.001, format="%.4f"
                )
                # sz3_eb_mode = st.selectbox(
                #     "sz3_eb_mode",
                #     options=["rel", "abs", "abs-and-rel", "abs-or-rel", "psnr", "l2"],
                #     index=options.index(st.session_state.selected_column),
                #     key="zfp_mode_selected",
                # )
            sz3_compressor = CodecStack(Sz3(eb_mode=sz3_eb_mode, eb_rel=sz3_eb_rel))
            ds_sz3[field_to_compress] = sz3_compressor.encode_decode_data_array(
                ds[field_to_compress]
            ).compute()
            sz3_dwt_distances = calc_dwt_dist(
                ds_sz3[field_to_compress], ds[field_to_compress], n_levels=4
            )
            dwt_dists["sz3_eb_compressors"] = sz3_dwt_distances
            st.write("SZ3 DWT disctance: " + f"{sz3_dwt_distances}")
    else:
        st.warning("Please select at least one compressor.")
else:
    st.warning("Please upload a netcdf file.")

# Submit button
if st.button("Submit"):
    min_key = min(dwt_dists, key=dwt_dists.get)
    st.write("Best compression method: " + f"{min_key}")
