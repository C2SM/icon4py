import pytest
from statistics import mean, median

def test_mean_performance(benchmark):
    # Precompute some data useful for the benchmark but that should not be
    # included in the benchmark time
    data = [1, 2, 3, 4, 5]

    # Benchmark the execution of the function
    benchmark(lambda: mean(data))


def test_mean_and_median_performance(benchmark):
    # Precompute some data useful for the benchmark but that should not be
    # included in the benchmark time
    data = [1, 2, 3, 4, 5]

    # Benchmark the execution of the function:
    # The `@benchmark` decorator will automatically call the function and
    # measure its execution
    @benchmark
    def bench():
        mean(data)
        median(data)
