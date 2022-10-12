#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_nh_diffusion_stencil_08.hpp"
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

class mo_nh_diffusion_stencil_08 {
public:
  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int VertexStride;
    int EdgeStride;
    int CellStride;
    int *cecoTable;

    GpuTriMesh() {}

    GpuTriMesh(const dawn::GlobalGpuTriMesh *mesh) {
      NumVertices = mesh->NumVertices;
      NumCells = mesh->NumCells;
      NumEdges = mesh->NumEdges;
      VertexStride = mesh->VertexStride;
      CellStride = mesh->CellStride;
      EdgeStride = mesh->EdgeStride;
      cecoTable = mesh->NeighborTables.at(
          std::tuple<std::vector<dawn::LocationType>, bool>{
              {dawn::LocationType::Cells, dawn::LocationType::Edges,
               dawn::LocationType::Cells},
              1});
    }
  };

private:
  double *w_;
  double *geofac_grg_x_;
  double *geofac_grg_y_;
  double *dwdx_;
  double *dwdy_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int dwdx_kSize_;
  inline static int dwdy_kSize_;

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

  static int get_dwdx_KSize() { return dwdx_kSize_; }

  static int get_dwdy_KSize() { return dwdy_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int dwdx_kSize,
                    const int dwdy_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    dwdx_kSize_ = dwdx_kSize;
    dwdy_kSize_ = dwdy_kSize;
  }

  mo_nh_diffusion_stencil_08() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_nh_diffusion_stencil_08 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto w_sid = get_sid(
        w_, gridtools::hymap::keys<
                unstructured::dim::horizontal,
                unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto dwdx_sid = get_sid(
        dwdx_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto dwdy_sid = get_sid(
        dwdy_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    neighbor_table_fortran<4> ceco_ptr{.raw_ptr_fortran = mesh_.cecoTable};
    auto connectivities =
        gridtools::hymap::keys<generated::C2E2CO_t>::make_values(ceco_ptr);
    double *geofac_grg_x_0 = &geofac_grg_x_[0 * mesh_.CellStride];
    double *geofac_grg_x_1 = &geofac_grg_x_[1 * mesh_.CellStride];
    double *geofac_grg_x_2 = &geofac_grg_x_[2 * mesh_.CellStride];
    double *geofac_grg_x_3 = &geofac_grg_x_[3 * mesh_.CellStride];
    auto geofac_grg_x_sid_0 = get_sid(
        geofac_grg_x_0,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grg_x_sid_1 = get_sid(
        geofac_grg_x_1,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grg_x_sid_2 = get_sid(
        geofac_grg_x_2,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grg_x_sid_3 = get_sid(
        geofac_grg_x_3,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grg_x_sid_comp = sid::composite::keys<
        integral_constant<int, 0>, integral_constant<int, 1>,
        integral_constant<int, 2>,
        integral_constant<int, 3>>::make_values(geofac_grg_x_sid_0,
                                                geofac_grg_x_sid_1,
                                                geofac_grg_x_sid_2,
                                                geofac_grg_x_sid_3);
    double *geofac_grg_y_0 = &geofac_grg_y_[0 * mesh_.CellStride];
    double *geofac_grg_y_1 = &geofac_grg_y_[1 * mesh_.CellStride];
    double *geofac_grg_y_2 = &geofac_grg_y_[2 * mesh_.CellStride];
    double *geofac_grg_y_3 = &geofac_grg_y_[3 * mesh_.CellStride];
    auto geofac_grg_y_sid_0 = get_sid(
        geofac_grg_y_0,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grg_y_sid_1 = get_sid(
        geofac_grg_y_1,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grg_y_sid_2 = get_sid(
        geofac_grg_y_2,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grg_y_sid_3 = get_sid(
        geofac_grg_y_3,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grg_y_sid_comp = sid::composite::keys<
        integral_constant<int, 0>, integral_constant<int, 1>,
        integral_constant<int, 2>,
        integral_constant<int, 3>>::make_values(geofac_grg_y_sid_0,
                                                geofac_grg_y_sid_1,
                                                geofac_grg_y_sid_2,
                                                geofac_grg_y_sid_3);
    generated::mo_nh_diffusion_stencil_08(connectivities)(
        cuda_backend, w_sid, geofac_grg_x_sid_comp, geofac_grg_y_sid_comp,
        dwdx_sid, dwdy_sid, horizontalStart, horizontalEnd, verticalStart,
        verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *w, double *geofac_grg_x, double *geofac_grg_y,
                     double *dwdx, double *dwdy) {
    w_ = w;
    geofac_grg_x_ = geofac_grg_x;
    geofac_grg_y_ = geofac_grg_y;
    dwdx_ = dwdx;
    dwdy_ = dwdy;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_nh_diffusion_stencil_08(double *w, double *geofac_grg_x,
                                    double *geofac_grg_y, double *dwdx,
                                    double *dwdy, const int verticalStart,
                                    const int verticalEnd,
                                    const int horizontalStart,
                                    const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_08 s;
  s.copy_pointers(w, geofac_grg_x, geofac_grg_y, dwdx, dwdy);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_nh_diffusion_stencil_08(
    const double *dwdx_dsl, const double *dwdx, const double *dwdy_dsl,
    const double *dwdy, const double dwdx_rel_tol, const double dwdx_abs_tol,
    const double dwdy_rel_tol, const double dwdy_abs_tol, const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_08::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_08::getStream();
  int kSize = dawn_generated::cuda_ico::mo_nh_diffusion_stencil_08::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int dwdx_kSize =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_08::get_dwdx_KSize();
  stencilMetrics =
      ::dawn::verify_field(stream, (mesh.CellStride) * dwdx_kSize, dwdx_dsl,
                           dwdx, "dwdx", dwdx_rel_tol, dwdx_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_dwdx(stencilMetrics,
                                    metricsNameFromEnvVar("SLURM_JOB_ID"),
                                    "mo_nh_diffusion_stencil_08", "dwdx");
  serialiser_dwdx.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), dwdx_kSize, (mesh.CellStride),
                          dwdx, "mo_nh_diffusion_stencil_08", "dwdx",
                          iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), dwdx_kSize, (mesh.CellStride),
                          dwdx_dsl, "mo_nh_diffusion_stencil_08", "dwdx_dsl",
                          iteration);
    std::cout << "[DSL] serializing dwdx as error is high.\n" << std::flush;
#endif
  }
  int dwdy_kSize =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_08::get_dwdy_KSize();
  stencilMetrics =
      ::dawn::verify_field(stream, (mesh.CellStride) * dwdy_kSize, dwdy_dsl,
                           dwdy, "dwdy", dwdy_rel_tol, dwdy_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_dwdy(stencilMetrics,
                                    metricsNameFromEnvVar("SLURM_JOB_ID"),
                                    "mo_nh_diffusion_stencil_08", "dwdy");
  serialiser_dwdy.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), dwdy_kSize, (mesh.CellStride),
                          dwdy, "mo_nh_diffusion_stencil_08", "dwdy",
                          iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), dwdy_kSize, (mesh.CellStride),
                          dwdy_dsl, "mo_nh_diffusion_stencil_08", "dwdy_dsl",
                          iteration);
    std::cout << "[DSL] serializing dwdy as error is high.\n" << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_nh_diffusion_stencil_08", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_nh_diffusion_stencil_08(
    double *w, double *geofac_grg_x, double *geofac_grg_y, double *dwdx,
    double *dwdy, double *dwdx_before, double *dwdy_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double dwdx_rel_tol,
    const double dwdx_abs_tol, const double dwdy_rel_tol,
    const double dwdy_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_nh_diffusion_stencil_08 (" << iteration
            << ") ...\n"
            << std::flush;
  run_mo_nh_diffusion_stencil_08(w, geofac_grg_x, geofac_grg_y, dwdx_before,
                                 dwdy_before, verticalStart, verticalEnd,
                                 horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_nh_diffusion_stencil_08 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_nh_diffusion_stencil_08...\n"
            << std::flush;
  verify_mo_nh_diffusion_stencil_08(dwdx_before, dwdx, dwdy_before, dwdy,
                                    dwdx_rel_tol, dwdx_abs_tol, dwdy_rel_tol,
                                    dwdy_abs_tol, iteration);

  iteration++;
}

void setup_mo_nh_diffusion_stencil_08(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream,
                                      const int dwdx_k_size,
                                      const int dwdy_k_size) {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_08::setup(
      mesh, k_size, stream, dwdx_k_size, dwdy_k_size);
}

void free_mo_nh_diffusion_stencil_08() {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_08::free();
}
}
