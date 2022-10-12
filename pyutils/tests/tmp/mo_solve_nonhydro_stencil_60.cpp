#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_solve_nonhydro_stencil_60.hpp"
#include <gridtools/common/array.hpp>
#include <gridtools/fn/backend/gpu.hpp>
#include <gridtools/fn/cartesian.hpp>
#include <gridtools/stencil/global_parameter.hpp>
#define GRIDTOOLS_DAWN_NO_INCLUDE
#include "driver-includes/math.hpp"
#include <chrono>
#define BLOCK_SIZE 128
#define LEVELS_PER_THREAD 1
namespace {
template <int... sizes>
using block_sizes_t = gridtools::meta::zip<
    gridtools::meta::iseq_to_list<
        std::make_integer_sequence<int, sizeof...(sizes)>,
        gridtools::meta::list, gridtools::integral_constant>,
    gridtools::meta::list<gridtools::integral_constant<int, sizes>...>>;

using fn_backend_t =
    gridtools::fn::backend::gpu<block_sizes_t<BLOCK_SIZE, LEVELS_PER_THREAD>>;
} // namespace
using namespace gridtools::dawn;
#define nproma 50000

template <int N> struct neighbor_table_fortran {
  const int *raw_ptr_fortran;
  __device__ friend inline constexpr gridtools::array<int, N>
  neighbor_table_neighbors(neighbor_table_fortran const &table, int index) {
    gridtools::array<int, N> ret{};
    for (int i = 0; i < N; i++) {
      ret[i] = table.raw_ptr_fortran[index + nproma * i];
    }
    return ret;
  }
};

template <int N> struct neighbor_table_4new_sparse {
  __device__ friend inline constexpr gridtools::array<int, N>
  neighbor_table_neighbors(neighbor_table_4new_sparse const &, int index) {
    gridtools::array<int, N> ret{};
    for (int i = 0; i < N; i++) {
      ret[i] = index + nproma * i;
    }
    return ret;
  }
};

template <class Ptr, class StrideMap>
auto get_sid(Ptr ptr, StrideMap const &strideMap) {
  using namespace gridtools;
  using namespace fn;
  return sid::synthetic()
      .set<sid::property::origin>(sid::host_device::simple_ptr_holder<Ptr>(ptr))
      .template set<sid::property::strides>(strideMap)
      .template set<sid::property::strides_kind, sid::unknown_kind>();
}

namespace dawn_generated {
namespace cuda_ico {

class mo_solve_nonhydro_stencil_60 {
public:
  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int VertexStride;
    int EdgeStride;
    int CellStride;

    GpuTriMesh() {}

    GpuTriMesh(const dawn::GlobalGpuTriMesh *mesh) {
      NumVertices = mesh->NumVertices;
      NumCells = mesh->NumCells;
      NumEdges = mesh->NumEdges;
      VertexStride = mesh->VertexStride;
      CellStride = mesh->CellStride;
      EdgeStride = mesh->EdgeStride;
    }
  };

private:
  double *exner_;
  double *ddt_exner_phy_;
  double *exner_dyn_incr_;
  double ndyn_substeps_var_;
  double dtime_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int exner_dyn_incr_kSize_;

  dim3 grid(int kSize, int elSize, bool kparallel) {
    if (kparallel) {
      int dK = (kSize + LEVELS_PER_THREAD - 1) / LEVELS_PER_THREAD;
      return dim3((elSize + BLOCK_SIZE - 1) / BLOCK_SIZE, dK, 1);
    } else {
      return dim3((elSize + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    }
  }

public:
  static const GpuTriMesh &getMesh() { return mesh_; }

  static cudaStream_t getStream() { return stream_; }

  static int getKSize() { return kSize_; }

  static int get_exner_dyn_incr_KSize() { return exner_dyn_incr_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int exner_dyn_incr_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    exner_dyn_incr_kSize_ = exner_dyn_incr_kSize;
  }

  mo_solve_nonhydro_stencil_60() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_solve_nonhydro_stencil_60 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto exner_sid = get_sid(
        exner_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto ddt_exner_phy_sid = get_sid(
        ddt_exner_phy_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto exner_dyn_incr_sid = get_sid(
        exner_dyn_incr_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    gridtools::stencil::global_parameter ndyn_substeps_var_gp{
        ndyn_substeps_var_};
    gridtools::stencil::global_parameter dtime_gp{dtime_};
    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    auto connectivities = gridtools::hymap::keys<>::make_values();
    generated::mo_solve_nonhydro_stencil_60(connectivities)(
        cuda_backend, exner_sid, ddt_exner_phy_sid, exner_dyn_incr_sid,
        ndyn_substeps_var_gp, dtime_gp, horizontalStart, horizontalEnd,
        verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *exner, double *ddt_exner_phy,
                     double *exner_dyn_incr, double ndyn_substeps_var,
                     double dtime) {
    exner_ = exner;
    ddt_exner_phy_ = ddt_exner_phy;
    exner_dyn_incr_ = exner_dyn_incr;
    ndyn_substeps_var_ = ndyn_substeps_var;
    dtime_ = dtime;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_solve_nonhydro_stencil_60(
    double *exner, double *ddt_exner_phy, double *exner_dyn_incr,
    double ndyn_substeps_var, double dtime, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_60 s;
  s.copy_pointers(exner, ddt_exner_phy, exner_dyn_incr, ndyn_substeps_var,
                  dtime);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_solve_nonhydro_stencil_60(const double *exner_dyn_incr_dsl,
                                         const double *exner_dyn_incr,
                                         const double exner_dyn_incr_rel_tol,
                                         const double exner_dyn_incr_abs_tol,
                                         const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_60::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_60::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_60::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int exner_dyn_incr_kSize = dawn_generated::cuda_ico::
      mo_solve_nonhydro_stencil_60::get_exner_dyn_incr_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * exner_dyn_incr_kSize, exner_dyn_incr_dsl,
      exner_dyn_incr, "exner_dyn_incr", exner_dyn_incr_rel_tol,
      exner_dyn_incr_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_exner_dyn_incr(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_60", "exner_dyn_incr");
  serialiser_exner_dyn_incr.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), exner_dyn_incr_kSize,
                          (mesh.CellStride), exner_dyn_incr,
                          "mo_solve_nonhydro_stencil_60", "exner_dyn_incr",
                          iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), exner_dyn_incr_kSize,
                          (mesh.CellStride), exner_dyn_incr_dsl,
                          "mo_solve_nonhydro_stencil_60", "exner_dyn_incr_dsl",
                          iteration);
    std::cout << "[DSL] serializing exner_dyn_incr as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_solve_nonhydro_stencil_60", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_solve_nonhydro_stencil_60(
    double *exner, double *ddt_exner_phy, double *exner_dyn_incr,
    double ndyn_substeps_var, double dtime, double *exner_dyn_incr_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double exner_dyn_incr_rel_tol,
    const double exner_dyn_incr_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_solve_nonhydro_stencil_60 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_solve_nonhydro_stencil_60(exner, ddt_exner_phy, exner_dyn_incr_before,
                                   ndyn_substeps_var, dtime, verticalStart,
                                   verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_solve_nonhydro_stencil_60 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_solve_nonhydro_stencil_60...\n"
            << std::flush;
  verify_mo_solve_nonhydro_stencil_60(exner_dyn_incr_before, exner_dyn_incr,
                                      exner_dyn_incr_rel_tol,
                                      exner_dyn_incr_abs_tol, iteration);

  iteration++;
}

void setup_mo_solve_nonhydro_stencil_60(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int exner_dyn_incr_k_size) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_60::setup(
      mesh, k_size, stream, exner_dyn_incr_k_size);
}

void free_mo_solve_nonhydro_stencil_60() {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_60::free();
}
}
