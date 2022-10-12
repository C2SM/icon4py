#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_solve_nonhydro_stencil_32.hpp"
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

class mo_solve_nonhydro_stencil_32 {
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
  double *z_rho_e_;
  double *z_vn_avg_;
  double *ddqz_z_full_e_;
  double *z_theta_v_e_;
  double *mass_fl_e_;
  double *z_theta_v_fl_e_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int mass_fl_e_kSize_;
  inline static int z_theta_v_fl_e_kSize_;

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

  static int get_mass_fl_e_KSize() { return mass_fl_e_kSize_; }

  static int get_z_theta_v_fl_e_KSize() { return z_theta_v_fl_e_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int mass_fl_e_kSize,
                    const int z_theta_v_fl_e_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    mass_fl_e_kSize_ = mass_fl_e_kSize;
    z_theta_v_fl_e_kSize_ = z_theta_v_fl_e_kSize;
  }

  mo_solve_nonhydro_stencil_32() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_solve_nonhydro_stencil_32 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto z_rho_e_sid = get_sid(
        z_rho_e_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto z_vn_avg_sid = get_sid(
        z_vn_avg_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto ddqz_z_full_e_sid = get_sid(
        ddqz_z_full_e_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto z_theta_v_e_sid = get_sid(
        z_theta_v_e_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto mass_fl_e_sid = get_sid(
        mass_fl_e_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto z_theta_v_fl_e_sid = get_sid(
        z_theta_v_fl_e_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    auto connectivities = gridtools::hymap::keys<>::make_values();
    generated::mo_solve_nonhydro_stencil_32(connectivities)(
        cuda_backend, z_rho_e_sid, z_vn_avg_sid, ddqz_z_full_e_sid,
        z_theta_v_e_sid, mass_fl_e_sid, z_theta_v_fl_e_sid, horizontalStart,
        horizontalEnd, verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *z_rho_e, double *z_vn_avg, double *ddqz_z_full_e,
                     double *z_theta_v_e, double *mass_fl_e,
                     double *z_theta_v_fl_e) {
    z_rho_e_ = z_rho_e;
    z_vn_avg_ = z_vn_avg;
    ddqz_z_full_e_ = ddqz_z_full_e;
    z_theta_v_e_ = z_theta_v_e;
    mass_fl_e_ = mass_fl_e;
    z_theta_v_fl_e_ = z_theta_v_fl_e;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_solve_nonhydro_stencil_32(
    double *z_rho_e, double *z_vn_avg, double *ddqz_z_full_e,
    double *z_theta_v_e, double *mass_fl_e, double *z_theta_v_fl_e,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_32 s;
  s.copy_pointers(z_rho_e, z_vn_avg, ddqz_z_full_e, z_theta_v_e, mass_fl_e,
                  z_theta_v_fl_e);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_solve_nonhydro_stencil_32(
    const double *mass_fl_e_dsl, const double *mass_fl_e,
    const double *z_theta_v_fl_e_dsl, const double *z_theta_v_fl_e,
    const double mass_fl_e_rel_tol, const double mass_fl_e_abs_tol,
    const double z_theta_v_fl_e_rel_tol, const double z_theta_v_fl_e_abs_tol,
    const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_32::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_32::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_32::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int mass_fl_e_kSize = dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_32::
      get_mass_fl_e_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.EdgeStride) * mass_fl_e_kSize, mass_fl_e_dsl, mass_fl_e,
      "mass_fl_e", mass_fl_e_rel_tol, mass_fl_e_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_mass_fl_e(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_32", "mass_fl_e");
  serialiser_mass_fl_e.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(
        0, (mesh.NumEdges - 1), mass_fl_e_kSize, (mesh.EdgeStride), mass_fl_e,
        "mo_solve_nonhydro_stencil_32", "mass_fl_e", iteration);
    serialize_dense_edges(0, (mesh.NumEdges - 1), mass_fl_e_kSize,
                          (mesh.EdgeStride), mass_fl_e_dsl,
                          "mo_solve_nonhydro_stencil_32", "mass_fl_e_dsl",
                          iteration);
    std::cout << "[DSL] serializing mass_fl_e as error is high.\n"
              << std::flush;
#endif
  }
  int z_theta_v_fl_e_kSize = dawn_generated::cuda_ico::
      mo_solve_nonhydro_stencil_32::get_z_theta_v_fl_e_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.EdgeStride) * z_theta_v_fl_e_kSize, z_theta_v_fl_e_dsl,
      z_theta_v_fl_e, "z_theta_v_fl_e", z_theta_v_fl_e_rel_tol,
      z_theta_v_fl_e_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_theta_v_fl_e(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_32", "z_theta_v_fl_e");
  serialiser_z_theta_v_fl_e.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(0, (mesh.NumEdges - 1), z_theta_v_fl_e_kSize,
                          (mesh.EdgeStride), z_theta_v_fl_e,
                          "mo_solve_nonhydro_stencil_32", "z_theta_v_fl_e",
                          iteration);
    serialize_dense_edges(0, (mesh.NumEdges - 1), z_theta_v_fl_e_kSize,
                          (mesh.EdgeStride), z_theta_v_fl_e_dsl,
                          "mo_solve_nonhydro_stencil_32", "z_theta_v_fl_e_dsl",
                          iteration);
    std::cout << "[DSL] serializing z_theta_v_fl_e as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_solve_nonhydro_stencil_32", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_solve_nonhydro_stencil_32(
    double *z_rho_e, double *z_vn_avg, double *ddqz_z_full_e,
    double *z_theta_v_e, double *mass_fl_e, double *z_theta_v_fl_e,
    double *mass_fl_e_before, double *z_theta_v_fl_e_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double mass_fl_e_rel_tol,
    const double mass_fl_e_abs_tol, const double z_theta_v_fl_e_rel_tol,
    const double z_theta_v_fl_e_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_solve_nonhydro_stencil_32 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_solve_nonhydro_stencil_32(z_rho_e, z_vn_avg, ddqz_z_full_e,
                                   z_theta_v_e, mass_fl_e_before,
                                   z_theta_v_fl_e_before, verticalStart,
                                   verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_solve_nonhydro_stencil_32 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_solve_nonhydro_stencil_32...\n"
            << std::flush;
  verify_mo_solve_nonhydro_stencil_32(
      mass_fl_e_before, mass_fl_e, z_theta_v_fl_e_before, z_theta_v_fl_e,
      mass_fl_e_rel_tol, mass_fl_e_abs_tol, z_theta_v_fl_e_rel_tol,
      z_theta_v_fl_e_abs_tol, iteration);

  iteration++;
}

void setup_mo_solve_nonhydro_stencil_32(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int mass_fl_e_k_size,
                                        const int z_theta_v_fl_e_k_size) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_32::setup(
      mesh, k_size, stream, mass_fl_e_k_size, z_theta_v_fl_e_k_size);
}

void free_mo_solve_nonhydro_stencil_32() {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_32::free();
}
}
