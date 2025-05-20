# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os


current_folder = os.path.dirname(os.path.realpath(__file__))

import time

import numpy as np
import xarray as xr
from scipy import stats
from scipy.spatial.distance import euclidean


ds = xr.open_dataset("tigge_pl_t_q_dx=2_2024_08_02.nc")
data_1 = np.asarray(ds.to_array()["longitude"].values[:90])
data_2 = np.asarray(ds.to_array()["longitude"].values[90:])
orig_distances = euclidean(data_1, data_2)


### Pearson Correlation
pearson_corr = stats.pearsonr(data_1, data_2).statistic.mean()


### PAA - Piecewise Aggregate Approximation
from tslearn.piecewise import PiecewiseAggregateApproximation


paa_start_time = time.time()
paa = PiecewiseAggregateApproximation(
    n_segments=8
)  # TODO: decide on ideal number of segments later
paa_data = paa.fit_transform([data_1, data_2])
paa_distance = paa.distance_paa(paa_data[0], paa_data[1])
paa_end_time = time.time()
paa_total_time = paa_end_time - paa_start_time

paa_reconstructed = paa.inverse_transform(paa_data)[:, :, 0]
paa_pearson = stats.pearsonr([data_1, data_2], paa_reconstructed).statistic.mean()


### DWT - Discrete Wavelet Transform
import pywt


dwt_start_time = time.time()
dwt_data_1 = pywt.wavedec(
    data_1, wavelet="haar", level=3
)  # TODO: decide on ideal number of decomposition levels later
dwt_data_2 = pywt.wavedec(data_2, wavelet="haar", level=3)
flat_coeffs1 = np.concatenate(dwt_data_1)
flat_coeffs2 = np.concatenate(dwt_data_2)
dwt_distance = euclidean(flat_coeffs1, flat_coeffs2)
dwt_end_time = time.time()
dwt_total_time = dwt_end_time - dwt_start_time

dwt_reconstructed_1 = pywt.waverec(dwt_data_1, wavelet="haar")
dwt_reconstructed_2 = pywt.waverec(dwt_data_2, wavelet="haar")
dwt_reconstructed = [dwt_reconstructed_1, dwt_reconstructed_2]
dwt_pearson = stats.pearsonr(
    [data_1, data_2], [dwt_reconstructed_1, dwt_reconstructed_2]
).statistic.mean()
