# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pywt
from scipy import stats
from tslearn.piecewise import PiecewiseAggregateApproximation


def calc_pearson_corr(input_1, input_2):
    pearson_corr = stats.pearsonr(input_1, input_2).statistic.mean()
    return pearson_corr


def calc_paa_dist(input_1, input_2, n_segments):
    paa = PiecewiseAggregateApproximation(n_segments=n_segments)
    paa_data = paa.fit_transform([input_1, input_2])
    paa_distance = paa.distance_paa(paa_data[0], paa_data[1])
    return paa_distance


def calc_dwt_dist(input_1, input_2, n_levels, wavelet="haar"):
    dwt_data_1 = pywt.wavedec(input_1, wavelet=wavelet, level=n_levels)
    dwt_data_2 = pywt.wavedec(input_2, wavelet=wavelet, level=n_levels)
    distances = [np.linalg.norm(c1 - c2) for c1, c2 in zip(dwt_data_1, dwt_data_2)]
    dwt_distance = np.sqrt(sum(d**2 for d in distances))
    return dwt_distance
