#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_velocity_advection_stencil_14.hpp"
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

class mo_velocity_advection_stencil_14 {
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
  double *ddqz_z_half_;
  double *z_w_con_c_;
  double *cfl_clipping_;
  double *pre_levelmask_;
  double *vcfl_;
  double cfl_w_limit_;
  double dtime_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int z_w_con_c_kSize_;
  inline static int cfl_clipping_kSize_;
  inline static int pre_levelmask_kSize_;
  inline static int vcfl_kSize_;

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

  static int get_z_w_con_c_KSize() { return z_w_con_c_kSize_; }

  static int get_cfl_clipping_KSize() { return cfl_clipping_kSize_; }

  static int get_pre_levelmask_KSize() { return pre_levelmask_kSize_; }

  static int get_vcfl_KSize() { return vcfl_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int z_w_con_c_kSize,
                    const int cfl_clipping_kSize, const int pre_levelmask_kSize,
                    const int vcfl_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    z_w_con_c_kSize_ = z_w_con_c_kSize;
    cfl_clipping_kSize_ = cfl_clipping_kSize;
    pre_levelmask_kSize_ = pre_levelmask_kSize;
    vcfl_kSize_ = vcfl_kSize;
  }

  mo_velocity_advection_stencil_14() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_velocity_advection_stencil_14 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto ddqz_z_half_sid = get_sid(
        ddqz_z_half_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto z_w_con_c_sid = get_sid(
        z_w_con_c_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto cfl_clipping_sid = get_sid(
        cfl_clipping_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto pre_levelmask_sid = get_sid(
        pre_levelmask_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto vcfl_sid = get_sid(
        vcfl_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    gridtools::stencil::global_parameter cfl_w_limit_gp{cfl_w_limit_};
    gridtools::stencil::global_parameter dtime_gp{dtime_};
    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    auto connectivities = gridtools::hymap::keys<>::make_values();
    generated::mo_velocity_advection_stencil_14(connectivities)(
        cuda_backend, ddqz_z_half_sid, z_w_con_c_sid, cfl_clipping_sid,
        pre_levelmask_sid, vcfl_sid, cfl_w_limit_gp, dtime_gp, horizontalStart,
        horizontalEnd, verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *ddqz_z_half, double *z_w_con_c,
                     double *cfl_clipping, double *pre_levelmask, double *vcfl,
                     double cfl_w_limit, double dtime) {
    ddqz_z_half_ = ddqz_z_half;
    z_w_con_c_ = z_w_con_c;
    cfl_clipping_ = cfl_clipping;
    pre_levelmask_ = pre_levelmask;
    vcfl_ = vcfl;
    cfl_w_limit_ = cfl_w_limit;
    dtime_ = dtime;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_velocity_advection_stencil_14(
    double *ddqz_z_half, double *z_w_con_c, double *cfl_clipping,
    double *pre_levelmask, double *vcfl, double cfl_w_limit, double dtime,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_velocity_advection_stencil_14 s;
  s.copy_pointers(ddqz_z_half, z_w_con_c, cfl_clipping, pre_levelmask, vcfl,
                  cfl_w_limit, dtime);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_velocity_advection_stencil_14(
    const double *z_w_con_c_dsl, const double *z_w_con_c,
    const double *cfl_clipping_dsl, const double *cfl_clipping,
    const double *pre_levelmask_dsl, const double *pre_levelmask,
    const double *vcfl_dsl, const double *vcfl, const double z_w_con_c_rel_tol,
    const double z_w_con_c_abs_tol, const double cfl_clipping_rel_tol,
    const double cfl_clipping_abs_tol, const double pre_levelmask_rel_tol,
    const double pre_levelmask_abs_tol, const double vcfl_rel_tol,
    const double vcfl_abs_tol, const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_velocity_advection_stencil_14::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_velocity_advection_stencil_14::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_velocity_advection_stencil_14::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int z_w_con_c_kSize = dawn_generated::cuda_ico::
      mo_velocity_advection_stencil_14::get_z_w_con_c_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * z_w_con_c_kSize, z_w_con_c_dsl, z_w_con_c,
      "z_w_con_c", z_w_con_c_rel_tol, z_w_con_c_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_w_con_c(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_velocity_advection_stencil_14", "z_w_con_c");
  serialiser_z_w_con_c.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(
        0, (mesh.NumCells - 1), z_w_con_c_kSize, (mesh.CellStride), z_w_con_c,
        "mo_velocity_advection_stencil_14", "z_w_con_c", iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), z_w_con_c_kSize,
                          (mesh.CellStride), z_w_con_c_dsl,
                          "mo_velocity_advection_stencil_14", "z_w_con_c_dsl",
                          iteration);
    std::cout << "[DSL] serializing z_w_con_c as error is high.\n"
              << std::flush;
#endif
  }
  int cfl_clipping_kSize = dawn_generated::cuda_ico::
      mo_velocity_advection_stencil_14::get_cfl_clipping_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * cfl_clipping_kSize, cfl_clipping_dsl,
      cfl_clipping, "cfl_clipping", cfl_clipping_rel_tol, cfl_clipping_abs_tol,
      iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_cfl_clipping(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_velocity_advection_stencil_14", "cfl_clipping");
  serialiser_cfl_clipping.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), cfl_clipping_kSize,
                          (mesh.CellStride), cfl_clipping,
                          "mo_velocity_advection_stencil_14", "cfl_clipping",
                          iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), cfl_clipping_kSize,
                          (mesh.CellStride), cfl_clipping_dsl,
                          "mo_velocity_advection_stencil_14",
                          "cfl_clipping_dsl", iteration);
    std::cout << "[DSL] serializing cfl_clipping as error is high.\n"
              << std::flush;
#endif
  }
  int pre_levelmask_kSize = dawn_generated::cuda_ico::
      mo_velocity_advection_stencil_14::get_pre_levelmask_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * pre_levelmask_kSize, pre_levelmask_dsl,
      pre_levelmask, "pre_levelmask", pre_levelmask_rel_tol,
      pre_levelmask_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_pre_levelmask(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_velocity_advection_stencil_14", "pre_levelmask");
  serialiser_pre_levelmask.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), pre_levelmask_kSize,
                          (mesh.CellStride), pre_levelmask,
                          "mo_velocity_advection_stencil_14", "pre_levelmask",
                          iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), pre_levelmask_kSize,
                          (mesh.CellStride), pre_levelmask_dsl,
                          "mo_velocity_advection_stencil_14",
                          "pre_levelmask_dsl", iteration);
    std::cout << "[DSL] serializing pre_levelmask as error is high.\n"
              << std::flush;
#endif
  }
  int vcfl_kSize = dawn_generated::cuda_ico::mo_velocity_advection_stencil_14::
      get_vcfl_KSize();
  stencilMetrics =
      ::dawn::verify_field(stream, (mesh.CellStride) * vcfl_kSize, vcfl_dsl,
                           vcfl, "vcfl", vcfl_rel_tol, vcfl_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_vcfl(stencilMetrics,
                                    metricsNameFromEnvVar("SLURM_JOB_ID"),
                                    "mo_velocity_advection_stencil_14", "vcfl");
  serialiser_vcfl.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), vcfl_kSize, (mesh.CellStride),
                          vcfl, "mo_velocity_advection_stencil_14", "vcfl",
                          iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), vcfl_kSize, (mesh.CellStride),
                          vcfl_dsl, "mo_velocity_advection_stencil_14",
                          "vcfl_dsl", iteration);
    std::cout << "[DSL] serializing vcfl as error is high.\n" << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_velocity_advection_stencil_14", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_velocity_advection_stencil_14(
    double *ddqz_z_half, double *z_w_con_c, double *cfl_clipping,
    double *pre_levelmask, double *vcfl, double cfl_w_limit, double dtime,
    double *z_w_con_c_before, double *cfl_clipping_before,
    double *pre_levelmask_before, double *vcfl_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double z_w_con_c_rel_tol, const double z_w_con_c_abs_tol,
    const double cfl_clipping_rel_tol, const double cfl_clipping_abs_tol,
    const double pre_levelmask_rel_tol, const double pre_levelmask_abs_tol,
    const double vcfl_rel_tol, const double vcfl_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_velocity_advection_stencil_14 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_velocity_advection_stencil_14(
      ddqz_z_half, z_w_con_c_before, cfl_clipping_before, pre_levelmask_before,
      vcfl_before, cfl_w_limit, dtime, verticalStart, verticalEnd,
      horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_velocity_advection_stencil_14 run time: " << time
            << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_velocity_advection_stencil_14...\n"
            << std::flush;
  verify_mo_velocity_advection_stencil_14(
      z_w_con_c_before, z_w_con_c, cfl_clipping_before, cfl_clipping,
      pre_levelmask_before, pre_levelmask, vcfl_before, vcfl, z_w_con_c_rel_tol,
      z_w_con_c_abs_tol, cfl_clipping_rel_tol, cfl_clipping_abs_tol,
      pre_levelmask_rel_tol, pre_levelmask_abs_tol, vcfl_rel_tol, vcfl_abs_tol,
      iteration);

  iteration++;
}

void setup_mo_velocity_advection_stencil_14(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int z_w_con_c_k_size,
                                            const int cfl_clipping_k_size,
                                            const int pre_levelmask_k_size,
                                            const int vcfl_k_size) {
  dawn_generated::cuda_ico::mo_velocity_advection_stencil_14::setup(
      mesh, k_size, stream, z_w_con_c_k_size, cfl_clipping_k_size,
      pre_levelmask_k_size, vcfl_k_size);
}

void free_mo_velocity_advection_stencil_14() {
  dawn_generated::cuda_ico::mo_velocity_advection_stencil_14::free();
}
}
