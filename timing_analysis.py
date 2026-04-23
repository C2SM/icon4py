import re
import sys

keywords = (
    "zd_intcoef",
    "zd_diffcoef",
    "geometrical_factor_for_green_gauss_gradient_x",
    "cell_to_vertex_interpolation_factor_by_area_weighting",
    "lsq_interpolation_coefficient",
    "rbf_interpolation_coefficient_cell_1",
    "e2c2v",
)
search_prefix = "WARNING TIMING: "


def read_log_and_extract_timing(file_name: str) -> list[float | None]:
    timing_list = []
    with open(file_name, "r") as f:
        log_content = f.read()
        for keyword in keywords:
            result = re.search(search_prefix + keyword + r" took ([0-9.]+)s", log_content)
            if result:
                timing = result[0].split(" ")[-1]
                timing_list.append(float(timing[:-1]))
                print(f"{keyword}: {timing[:-1]} s")
            else:
                timing_list.append(None)
                print(f"{keyword}: not found in log file")
    return timing_list

file_name1 = sys.argv[1]
file_name2 = sys.argv[2]

ref_timing = read_log_and_extract_timing(file_name1)
vec_timing = read_log_and_extract_timing(file_name2)

for keyword, ref, vec in zip(keywords, ref_timing, vec_timing):
    if ref is not None:
        if vec is not None:
            print(f"{keyword}: reference timing = \x1b[31m{ref:.3f} s\x1b[39m, vectorized timing = \x1b[32m{vec:.3f} s\x1b[39m, speedup = \x1b[36m{ref/vec:.2f}x \x1b[39m")
        else:
            minimum = 0.01
            print(f"{keyword}: reference timing = \x1b[31m{ref:.3f} s\x1b[39m, vectorized timing <= \x1b[32m{minimum:.3f} s\x1b[39m, speedup >= \x1b[36m{ref/minimum:.2f}x \x1b[39m")
    else:
        print(f"{keyword}: timing information missing for comparison")