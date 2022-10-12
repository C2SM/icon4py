#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_solve_nonhydro_stencil_17.hpp"
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

class mo_solve_nonhydro_stencil_17 {
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
  double *hmask_dd3d_;
  double *scalfac_dd3d_;
  double *inv_dual_edge_length_;
  double *z_dwdz_dd_;
  double *z_graddiv_vn_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int z_graddiv_vn_kSize_;

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

  static int get_z_graddiv_vn_KSize() { return z_graddiv_vn_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int z_graddiv_vn_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    z_graddiv_vn_kSize_ = z_graddiv_vn_kSize;
  }

  mo_solve_nonhydro_stencil_17() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_solve_nonhydro_stencil_17 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto hmask_dd3d_sid = get_sid(
        hmask_dd3d_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto scalfac_dd3d_sid = get_sid(
        scalfac_dd3d_,
        gridtools::hymap::keys<unstructured::dim::vertical>::make_values(1));

    auto inv_dual_edge_length_sid = get_sid(
        inv_dual_edge_length_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto z_dwdz_dd_sid = get_sid(
        z_dwdz_dd_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto z_graddiv_vn_sid = get_sid(
        z_graddiv_vn_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    neighbor_table_fortran<2> ec_ptr{.raw_ptr_fortran = mesh_.ecTable};
    auto connectivities =
        gridtools::hymap::keys<generated::E2C_t>::make_values(ec_ptr);
    generated::mo_solve_nonhydro_stencil_17(connectivities)(
        cuda_backend, hmask_dd3d_sid, scalfac_dd3d_sid,
        inv_dual_edge_length_sid, z_dwdz_dd_sid, z_graddiv_vn_sid,
        horizontalStart, horizontalEnd, verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *hmask_dd3d, double *scalfac_dd3d,
                     double *inv_dual_edge_length, double *z_dwdz_dd,
                     double *z_graddiv_vn) {
    hmask_dd3d_ = hmask_dd3d;
    scalfac_dd3d_ = scalfac_dd3d;
    inv_dual_edge_length_ = inv_dual_edge_length;
    z_dwdz_dd_ = z_dwdz_dd;
    z_graddiv_vn_ = z_graddiv_vn;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_solve_nonhydro_stencil_17(
    double *hmask_dd3d, double *scalfac_dd3d, double *inv_dual_edge_length,
    double *z_dwdz_dd, double *z_graddiv_vn, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_17 s;
  s.copy_pointers(hmask_dd3d, scalfac_dd3d, inv_dual_edge_length, z_dwdz_dd,
                  z_graddiv_vn);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_solve_nonhydro_stencil_17(const double *z_graddiv_vn_dsl,
                                         const double *z_graddiv_vn,
                                         const double z_graddiv_vn_rel_tol,
                                         const double z_graddiv_vn_abs_tol,
                                         const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_17::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_17::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_17::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int z_graddiv_vn_kSize = dawn_generated::cuda_ico::
      mo_solve_nonhydro_stencil_17::get_z_graddiv_vn_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.EdgeStride) * z_graddiv_vn_kSize, z_graddiv_vn_dsl,
      z_graddiv_vn, "z_graddiv_vn", z_graddiv_vn_rel_tol, z_graddiv_vn_abs_tol,
      iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_graddiv_vn(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_17", "z_graddiv_vn");
  serialiser_z_graddiv_vn.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(0, (mesh.NumEdges - 1), z_graddiv_vn_kSize,
                          (mesh.EdgeStride), z_graddiv_vn,
                          "mo_solve_nonhydro_stencil_17", "z_graddiv_vn",
                          iteration);
    serialize_dense_edges(0, (mesh.NumEdges - 1), z_graddiv_vn_kSize,
                          (mesh.EdgeStride), z_graddiv_vn_dsl,
                          "mo_solve_nonhydro_stencil_17", "z_graddiv_vn_dsl",
                          iteration);
    std::cout << "[DSL] serializing z_graddiv_vn as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_solve_nonhydro_stencil_17", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_solve_nonhydro_stencil_17(
    double *hmask_dd3d, double *scalfac_dd3d, double *inv_dual_edge_length,
    double *z_dwdz_dd, double *z_graddiv_vn, double *z_graddiv_vn_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double z_graddiv_vn_rel_tol,
    const double z_graddiv_vn_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_solve_nonhydro_stencil_17 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_solve_nonhydro_stencil_17(hmask_dd3d, scalfac_dd3d,
                                   inv_dual_edge_length, z_dwdz_dd,
                                   z_graddiv_vn_before, verticalStart,
                                   verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_solve_nonhydro_stencil_17 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_solve_nonhydro_stencil_17...\n"
            << std::flush;
  verify_mo_solve_nonhydro_stencil_17(z_graddiv_vn_before, z_graddiv_vn,
                                      z_graddiv_vn_rel_tol,
                                      z_graddiv_vn_abs_tol, iteration);

  iteration++;
}

void setup_mo_solve_nonhydro_stencil_17(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_graddiv_vn_k_size) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_17::setup(
      mesh, k_size, stream, z_graddiv_vn_k_size);
}

void free_mo_solve_nonhydro_stencil_17() {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_17::free();
}
}
