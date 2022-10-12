#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_nh_diffusion_stencil_12.hpp"
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

class mo_nh_diffusion_stencil_12 {
public:
  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int VertexStride;
    int EdgeStride;
    int CellStride;
    int *ecTable;

    GpuTriMesh() {}

    GpuTriMesh(const dawn::GlobalGpuTriMesh *mesh) {
      NumVertices = mesh->NumVertices;
      NumCells = mesh->NumCells;
      NumEdges = mesh->NumEdges;
      VertexStride = mesh->VertexStride;
      CellStride = mesh->CellStride;
      EdgeStride = mesh->EdgeStride;
      ecTable = mesh->NeighborTables.at(
          std::tuple<std::vector<dawn::LocationType>, bool>{
              {dawn::LocationType::Edges, dawn::LocationType::Cells}, 0});
    }
  };

private:
  double *kh_smag_e_;
  double *enh_diffu_3d_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int kh_smag_e_kSize_;

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

  static int get_kh_smag_e_KSize() { return kh_smag_e_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int kh_smag_e_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    kh_smag_e_kSize_ = kh_smag_e_kSize;
  }

  mo_nh_diffusion_stencil_12() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_nh_diffusion_stencil_12 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto kh_smag_e_sid = get_sid(
        kh_smag_e_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto enh_diffu_3d_sid = get_sid(
        enh_diffu_3d_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    neighbor_table_fortran<2> ec_ptr{.raw_ptr_fortran = mesh_.ecTable};
    auto connectivities =
        gridtools::hymap::keys<generated::E2C_t>::make_values(ec_ptr);
    generated::mo_nh_diffusion_stencil_12(connectivities)(
        cuda_backend, kh_smag_e_sid, enh_diffu_3d_sid, horizontalStart,
        horizontalEnd, verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *kh_smag_e, double *enh_diffu_3d) {
    kh_smag_e_ = kh_smag_e;
    enh_diffu_3d_ = enh_diffu_3d;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_nh_diffusion_stencil_12(double *kh_smag_e, double *enh_diffu_3d,
                                    const int verticalStart,
                                    const int verticalEnd,
                                    const int horizontalStart,
                                    const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_12 s;
  s.copy_pointers(kh_smag_e, enh_diffu_3d);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_nh_diffusion_stencil_12(const double *kh_smag_e_dsl,
                                       const double *kh_smag_e,
                                       const double kh_smag_e_rel_tol,
                                       const double kh_smag_e_abs_tol,
                                       const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_12::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_12::getStream();
  int kSize = dawn_generated::cuda_ico::mo_nh_diffusion_stencil_12::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int kh_smag_e_kSize = dawn_generated::cuda_ico::mo_nh_diffusion_stencil_12::
      get_kh_smag_e_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.EdgeStride) * kh_smag_e_kSize, kh_smag_e_dsl, kh_smag_e,
      "kh_smag_e", kh_smag_e_rel_tol, kh_smag_e_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_kh_smag_e(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_nh_diffusion_stencil_12", "kh_smag_e");
  serialiser_kh_smag_e.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(0, (mesh.NumEdges - 1), kh_smag_e_kSize,
                          (mesh.EdgeStride), kh_smag_e,
                          "mo_nh_diffusion_stencil_12", "kh_smag_e", iteration);
    serialize_dense_edges(0, (mesh.NumEdges - 1), kh_smag_e_kSize,
                          (mesh.EdgeStride), kh_smag_e_dsl,
                          "mo_nh_diffusion_stencil_12", "kh_smag_e_dsl",
                          iteration);
    std::cout << "[DSL] serializing kh_smag_e as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_nh_diffusion_stencil_12", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_nh_diffusion_stencil_12(
    double *kh_smag_e, double *enh_diffu_3d, double *kh_smag_e_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double kh_smag_e_rel_tol,
    const double kh_smag_e_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_nh_diffusion_stencil_12 (" << iteration
            << ") ...\n"
            << std::flush;
  run_mo_nh_diffusion_stencil_12(kh_smag_e_before, enh_diffu_3d, verticalStart,
                                 verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_nh_diffusion_stencil_12 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_nh_diffusion_stencil_12...\n"
            << std::flush;
  verify_mo_nh_diffusion_stencil_12(kh_smag_e_before, kh_smag_e,
                                    kh_smag_e_rel_tol, kh_smag_e_abs_tol,
                                    iteration);

  iteration++;
}

void setup_mo_nh_diffusion_stencil_12(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream,
                                      const int kh_smag_e_k_size) {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_12::setup(
      mesh, k_size, stream, kh_smag_e_k_size);
}

void free_mo_nh_diffusion_stencil_12() {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_12::free();
}
}
