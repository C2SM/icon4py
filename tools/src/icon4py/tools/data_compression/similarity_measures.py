# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
import warnings


warnings.filterwarnings("ignore")

current_folder = os.path.dirname(os.path.realpath(__file__))

import time

import numpy as np
import xarray as xr
from scipy import stats
from scipy.spatial.distance import euclidean


# scp -r santis:/capstor/store/cscs/userlab/cwp03/ppothapa/Dyamond_PostProcessed/exclaim_uncoupled_R02B10L120_pretest_post_script/out1/tot_prec_20200121T000000Z.nc ./icon4py/tools/src/icon4py/tools/data_compression/data
# scp -r santis:/capstor/store/cscs/userlab/cwp03/ppothapa/Dyamond_PostProcessed/exclaim_uncoupled_R02B10L120_pretest_post_script/out1/tot_prec_20200430T000000Z.nc ./icon4py/tools/src/icon4py/tools/data_compression/data

ds_1 = xr.open_dataset(
    "data/tot_prec_20200121T000000Z.nc", chunks={}
)  # chunks specification to avoid loading all file at the beginning
ds_2 = xr.open_dataset("./data/tot_prec_20200430T000000Z.nc", chunks={})
data_1 = ds_1["tot_prec"][0, :].values
data_2 = ds_2["tot_prec"][0, :].values

eucl_start_time = time.time()
orig_distances = euclidean(data_1, data_2)
eucl_end_time = time.time()
eucl_total_time = eucl_end_time - eucl_start_time
print(f"Euclidean distance: {orig_distances}")
print(f"Euclidean distance total time: {eucl_total_time}")


### Pearson Correlation
pearson_corr = stats.pearsonr(data_1, data_2).statistic.mean()
print(f"Pearson correlation: {pearson_corr}")


### PAA - Piecewise Aggregate Approximation
from tslearn.piecewise import PiecewiseAggregateApproximation


paa_start_time = time.time()
paa = PiecewiseAggregateApproximation(n_segments=round(data_1.shape[0] * 1e-6))
paa_data = paa.fit_transform([data_1, data_2])
paa_distance = paa.distance_paa(paa_data[0], paa_data[1])
paa_end_time = time.time()
paa_total_time = paa_end_time - paa_start_time
print(f"PAA distance: {paa_distance}")
print(f"PAA distance total time: {paa_total_time}")

paa_reconstructed = paa.inverse_transform(paa_data)[:, :, 0]
paa_pearson = stats.pearsonr([data_1, data_2], paa_reconstructed).statistic.mean()
print(f"PAA Pearson correlation: {paa_pearson}")


### DWT - Discrete Wavelet Transform
import pywt


dwt_start_time = time.time()
dwt_data_1 = pywt.wavedec(data_1, wavelet="haar", level=3)
dwt_data_2 = pywt.wavedec(data_2, wavelet="haar", level=3)
flat_coeffs1 = np.concatenate(dwt_data_1)
flat_coeffs2 = np.concatenate(dwt_data_2)
dwt_distance = euclidean(flat_coeffs1, flat_coeffs2)
dwt_end_time = time.time()
dwt_total_time = dwt_end_time - dwt_start_time
print(f"DWT distance: {dwt_distance}")
print(f"DWT distance total time: {dwt_total_time}")

dwt_reconstructed_1 = pywt.waverec(dwt_data_1, wavelet="haar")
dwt_reconstructed_2 = pywt.waverec(dwt_data_2, wavelet="haar")
dwt_reconstructed = [dwt_reconstructed_1, dwt_reconstructed_2]
dwt_pearson = stats.pearsonr(
    [data_1, data_2], [dwt_reconstructed_1, dwt_reconstructed_2]
).statistic.mean()
print(f"DWT Pearson correlation: {dwt_pearson}")
