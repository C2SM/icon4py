import re

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

ref_timing = read_log_and_extract_timing("log_file_ref.txt")
vec_timing = read_log_and_extract_timing("log_file3.txt")

for keyword, ref, vec in zip(keywords, ref_timing, vec_timing):
    if ref is not None and vec is not None:
        print(f"{keyword}: reference timing = {ref:.3f} s, vectorized timing = {vec:.3f} s, speedup = {ref/vec:.2f}x")
    else:
        print(f"{keyword}: timing information missing for comparison")