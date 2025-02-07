# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import benchmarks.setup_tests as setup_tests

class BenchmarkMetaclass(type):
    def __dir__(cls):
        return list(setup_tests.BENCHMARKS.keys())
    def __getattr__(cls, name):
        if not name.startswith(setup_tests.PREFIX):
            raise AttributeError
        setattr(cls, name, setup_tests.BENCHMARKS[name])
        return getattr(cls, name)

class Benchmarks(metaclass=BenchmarkMetaclass):
    def __getattr__(self, name):
        if not name.startswith(setup_tests.PREFIX):
            raise AttributeError
        setattr(type(self), name, setup_tests.BENCHMARKS[name])
        return getattr(self, name)
