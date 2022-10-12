#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_nh_diffusion_stencil_16.hpp"
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

class mo_nh_diffusion_stencil_16 {
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
  double *z_temp_;
  double *area_;
  double *theta_v_;
  double *exner_;
  double rd_o_cvd_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int theta_v_kSize_;
  inline static int exner_kSize_;

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

  static int get_theta_v_KSize() { return theta_v_kSize_; }

  static int get_exner_KSize() { return exner_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int theta_v_kSize,
                    const int exner_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    theta_v_kSize_ = theta_v_kSize;
    exner_kSize_ = exner_kSize;
  }

  mo_nh_diffusion_stencil_16() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_nh_diffusion_stencil_16 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto z_temp_sid = get_sid(
        z_temp_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto area_sid = get_sid(
        area_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto theta_v_sid = get_sid(
        theta_v_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto exner_sid = get_sid(
        exner_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    gridtools::stencil::global_parameter rd_o_cvd_gp{rd_o_cvd_};
    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    auto connectivities = gridtools::hymap::keys<>::make_values();
    generated::mo_nh_diffusion_stencil_16(connectivities)(
        cuda_backend, z_temp_sid, area_sid, theta_v_sid, exner_sid, rd_o_cvd_gp,
        horizontalStart, horizontalEnd, verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *z_temp, double *area, double *theta_v,
                     double *exner, double rd_o_cvd) {
    z_temp_ = z_temp;
    area_ = area;
    theta_v_ = theta_v;
    exner_ = exner;
    rd_o_cvd_ = rd_o_cvd;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_nh_diffusion_stencil_16(double *z_temp, double *area,
                                    double *theta_v, double *exner,
                                    double rd_o_cvd, const int verticalStart,
                                    const int verticalEnd,
                                    const int horizontalStart,
                                    const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_16 s;
  s.copy_pointers(z_temp, area, theta_v, exner, rd_o_cvd);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_nh_diffusion_stencil_16(
    const double *theta_v_dsl, const double *theta_v, const double *exner_dsl,
    const double *exner, const double theta_v_rel_tol,
    const double theta_v_abs_tol, const double exner_rel_tol,
    const double exner_abs_tol, const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_16::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_16::getStream();
  int kSize = dawn_generated::cuda_ico::mo_nh_diffusion_stencil_16::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int theta_v_kSize =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_16::get_theta_v_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * theta_v_kSize, theta_v_dsl, theta_v,
      "theta_v", theta_v_rel_tol, theta_v_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_theta_v(stencilMetrics,
                                       metricsNameFromEnvVar("SLURM_JOB_ID"),
                                       "mo_nh_diffusion_stencil_16", "theta_v");
  serialiser_theta_v.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), theta_v_kSize,
                          (mesh.CellStride), theta_v,
                          "mo_nh_diffusion_stencil_16", "theta_v", iteration);
    serialize_dense_cells(
        0, (mesh.NumCells - 1), theta_v_kSize, (mesh.CellStride), theta_v_dsl,
        "mo_nh_diffusion_stencil_16", "theta_v_dsl", iteration);
    std::cout << "[DSL] serializing theta_v as error is high.\n" << std::flush;
#endif
  }
  int exner_kSize =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_16::get_exner_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * exner_kSize, exner_dsl, exner, "exner",
      exner_rel_tol, exner_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_exner(stencilMetrics,
                                     metricsNameFromEnvVar("SLURM_JOB_ID"),
                                     "mo_nh_diffusion_stencil_16", "exner");
  serialiser_exner.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), exner_kSize,
                          (mesh.CellStride), exner,
                          "mo_nh_diffusion_stencil_16", "exner", iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), exner_kSize,
                          (mesh.CellStride), exner_dsl,
                          "mo_nh_diffusion_stencil_16", "exner_dsl", iteration);
    std::cout << "[DSL] serializing exner as error is high.\n" << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_nh_diffusion_stencil_16", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_nh_diffusion_stencil_16(
    double *z_temp, double *area, double *theta_v, double *exner,
    double rd_o_cvd, double *theta_v_before, double *exner_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double theta_v_rel_tol,
    const double theta_v_abs_tol, const double exner_rel_tol,
    const double exner_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_nh_diffusion_stencil_16 (" << iteration
            << ") ...\n"
            << std::flush;
  run_mo_nh_diffusion_stencil_16(z_temp, area, theta_v_before, exner_before,
                                 rd_o_cvd, verticalStart, verticalEnd,
                                 horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_nh_diffusion_stencil_16 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_nh_diffusion_stencil_16...\n"
            << std::flush;
  verify_mo_nh_diffusion_stencil_16(theta_v_before, theta_v, exner_before,
                                    exner, theta_v_rel_tol, theta_v_abs_tol,
                                    exner_rel_tol, exner_abs_tol, iteration);

  iteration++;
}

void setup_mo_nh_diffusion_stencil_16(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream,
                                      const int theta_v_k_size,
                                      const int exner_k_size) {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_16::setup(
      mesh, k_size, stream, theta_v_k_size, exner_k_size);
}

void free_mo_nh_diffusion_stencil_16() {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_16::free();
}
}
