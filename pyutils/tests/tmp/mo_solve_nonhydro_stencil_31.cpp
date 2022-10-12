#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_solve_nonhydro_stencil_31.hpp"
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

class mo_solve_nonhydro_stencil_31 {
public:
  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int VertexStride;
    int EdgeStride;
    int CellStride;
    int *eceoTable;

    GpuTriMesh() {}

    GpuTriMesh(const dawn::GlobalGpuTriMesh *mesh) {
      NumVertices = mesh->NumVertices;
      NumCells = mesh->NumCells;
      NumEdges = mesh->NumEdges;
      VertexStride = mesh->VertexStride;
      CellStride = mesh->CellStride;
      EdgeStride = mesh->EdgeStride;
      eceoTable = mesh->NeighborTables.at(
          std::tuple<std::vector<dawn::LocationType>, bool>{
              {dawn::LocationType::Edges, dawn::LocationType::Cells,
               dawn::LocationType::Edges},
              1});
    }
  };

private:
  double *e_flx_avg_;
  double *vn_;
  double *z_vn_avg_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int z_vn_avg_kSize_;

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

  static int get_z_vn_avg_KSize() { return z_vn_avg_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int z_vn_avg_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    z_vn_avg_kSize_ = z_vn_avg_kSize;
  }

  mo_solve_nonhydro_stencil_31() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_solve_nonhydro_stencil_31 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto vn_sid = get_sid(
        vn_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto z_vn_avg_sid = get_sid(
        z_vn_avg_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    neighbor_table_fortran<5> eceo_ptr{.raw_ptr_fortran = mesh_.eceoTable};
    auto connectivities =
        gridtools::hymap::keys<generated::E2C2EO_t>::make_values(eceo_ptr);
    double *e_flx_avg_0 = &e_flx_avg_[0 * mesh_.EdgeStride];
    double *e_flx_avg_1 = &e_flx_avg_[1 * mesh_.EdgeStride];
    double *e_flx_avg_2 = &e_flx_avg_[2 * mesh_.EdgeStride];
    double *e_flx_avg_3 = &e_flx_avg_[3 * mesh_.EdgeStride];
    double *e_flx_avg_4 = &e_flx_avg_[4 * mesh_.EdgeStride];
    auto e_flx_avg_sid_0 = get_sid(
        e_flx_avg_0,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto e_flx_avg_sid_1 = get_sid(
        e_flx_avg_1,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto e_flx_avg_sid_2 = get_sid(
        e_flx_avg_2,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto e_flx_avg_sid_3 = get_sid(
        e_flx_avg_3,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto e_flx_avg_sid_4 = get_sid(
        e_flx_avg_4,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto e_flx_avg_sid_comp = sid::composite::keys<
        integral_constant<int, 0>, integral_constant<int, 1>,
        integral_constant<int, 2>, integral_constant<int, 3>,
        integral_constant<int, 4>>::make_values(e_flx_avg_sid_0,
                                                e_flx_avg_sid_1,
                                                e_flx_avg_sid_2,
                                                e_flx_avg_sid_3,
                                                e_flx_avg_sid_4);
    generated::mo_solve_nonhydro_stencil_31(connectivities)(
        cuda_backend, e_flx_avg_sid_comp, vn_sid, z_vn_avg_sid, horizontalStart,
        horizontalEnd, verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *e_flx_avg, double *vn, double *z_vn_avg) {
    e_flx_avg_ = e_flx_avg;
    vn_ = vn;
    z_vn_avg_ = z_vn_avg;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_solve_nonhydro_stencil_31(double *e_flx_avg, double *vn,
                                      double *z_vn_avg, const int verticalStart,
                                      const int verticalEnd,
                                      const int horizontalStart,
                                      const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_31 s;
  s.copy_pointers(e_flx_avg, vn, z_vn_avg);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_solve_nonhydro_stencil_31(const double *z_vn_avg_dsl,
                                         const double *z_vn_avg,
                                         const double z_vn_avg_rel_tol,
                                         const double z_vn_avg_abs_tol,
                                         const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_31::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_31::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_31::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int z_vn_avg_kSize = dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_31::
      get_z_vn_avg_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.EdgeStride) * z_vn_avg_kSize, z_vn_avg_dsl, z_vn_avg,
      "z_vn_avg", z_vn_avg_rel_tol, z_vn_avg_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_vn_avg(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_31", "z_vn_avg");
  serialiser_z_vn_avg.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(
        0, (mesh.NumEdges - 1), z_vn_avg_kSize, (mesh.EdgeStride), z_vn_avg,
        "mo_solve_nonhydro_stencil_31", "z_vn_avg", iteration);
    serialize_dense_edges(
        0, (mesh.NumEdges - 1), z_vn_avg_kSize, (mesh.EdgeStride), z_vn_avg_dsl,
        "mo_solve_nonhydro_stencil_31", "z_vn_avg_dsl", iteration);
    std::cout << "[DSL] serializing z_vn_avg as error is high.\n" << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_solve_nonhydro_stencil_31", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_solve_nonhydro_stencil_31(
    double *e_flx_avg, double *vn, double *z_vn_avg, double *z_vn_avg_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double z_vn_avg_rel_tol,
    const double z_vn_avg_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_solve_nonhydro_stencil_31 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_solve_nonhydro_stencil_31(e_flx_avg, vn, z_vn_avg_before,
                                   verticalStart, verticalEnd, horizontalStart,
                                   horizontalEnd);

  std::cout << "[DSL] mo_solve_nonhydro_stencil_31 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_solve_nonhydro_stencil_31...\n"
            << std::flush;
  verify_mo_solve_nonhydro_stencil_31(
      z_vn_avg_before, z_vn_avg, z_vn_avg_rel_tol, z_vn_avg_abs_tol, iteration);

  iteration++;
}

void setup_mo_solve_nonhydro_stencil_31(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_vn_avg_k_size) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_31::setup(
      mesh, k_size, stream, z_vn_avg_k_size);
}

void free_mo_solve_nonhydro_stencil_31() {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_31::free();
}
}
