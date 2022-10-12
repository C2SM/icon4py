#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_velocity_advection_stencil_07.hpp"
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

class mo_velocity_advection_stencil_07 {
public:
  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int VertexStride;
    int EdgeStride;
    int CellStride;
    int *ecTable;
    int *evTable;

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
      evTable = mesh->NeighborTables.at(
          std::tuple<std::vector<dawn::LocationType>, bool>{
              {dawn::LocationType::Edges, dawn::LocationType::Vertices}, 0});
    }
  };

private:
  double *vn_ie_;
  double *inv_dual_edge_length_;
  double *w_;
  double *z_vt_ie_;
  double *inv_primal_edge_length_;
  double *tangent_orientation_;
  double *z_w_v_;
  double *z_v_grad_w_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int z_v_grad_w_kSize_;

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

  static int get_z_v_grad_w_KSize() { return z_v_grad_w_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int z_v_grad_w_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    z_v_grad_w_kSize_ = z_v_grad_w_kSize;
  }

  mo_velocity_advection_stencil_07() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_velocity_advection_stencil_07 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto vn_ie_sid = get_sid(
        vn_ie_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto inv_dual_edge_length_sid = get_sid(
        inv_dual_edge_length_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto w_sid = get_sid(
        w_, gridtools::hymap::keys<
                unstructured::dim::horizontal,
                unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto z_vt_ie_sid = get_sid(
        z_vt_ie_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto inv_primal_edge_length_sid = get_sid(
        inv_primal_edge_length_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto tangent_orientation_sid = get_sid(
        tangent_orientation_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto z_w_v_sid = get_sid(
        z_w_v_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.VertexStride));

    auto z_v_grad_w_sid = get_sid(
        z_v_grad_w_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    neighbor_table_fortran<2> ec_ptr{.raw_ptr_fortran = mesh_.ecTable};
    neighbor_table_fortran<2> ev_ptr{.raw_ptr_fortran = mesh_.evTable};
    auto connectivities =
        gridtools::hymap::keys<generated::E2C_t, generated::E2V_t>::make_values(
            ec_ptr, ev_ptr);
    generated::mo_velocity_advection_stencil_07(connectivities)(
        cuda_backend, vn_ie_sid, inv_dual_edge_length_sid, w_sid, z_vt_ie_sid,
        inv_primal_edge_length_sid, tangent_orientation_sid, z_w_v_sid,
        z_v_grad_w_sid, horizontalStart, horizontalEnd, verticalStart,
        verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *vn_ie, double *inv_dual_edge_length, double *w,
                     double *z_vt_ie, double *inv_primal_edge_length,
                     double *tangent_orientation, double *z_w_v,
                     double *z_v_grad_w) {
    vn_ie_ = vn_ie;
    inv_dual_edge_length_ = inv_dual_edge_length;
    w_ = w;
    z_vt_ie_ = z_vt_ie;
    inv_primal_edge_length_ = inv_primal_edge_length;
    tangent_orientation_ = tangent_orientation;
    z_w_v_ = z_w_v;
    z_v_grad_w_ = z_v_grad_w;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_velocity_advection_stencil_07(
    double *vn_ie, double *inv_dual_edge_length, double *w, double *z_vt_ie,
    double *inv_primal_edge_length, double *tangent_orientation, double *z_w_v,
    double *z_v_grad_w, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_velocity_advection_stencil_07 s;
  s.copy_pointers(vn_ie, inv_dual_edge_length, w, z_vt_ie,
                  inv_primal_edge_length, tangent_orientation, z_w_v,
                  z_v_grad_w);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_velocity_advection_stencil_07(const double *z_v_grad_w_dsl,
                                             const double *z_v_grad_w,
                                             const double z_v_grad_w_rel_tol,
                                             const double z_v_grad_w_abs_tol,
                                             const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_velocity_advection_stencil_07::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_velocity_advection_stencil_07::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_velocity_advection_stencil_07::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int z_v_grad_w_kSize = dawn_generated::cuda_ico::
      mo_velocity_advection_stencil_07::get_z_v_grad_w_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.EdgeStride) * z_v_grad_w_kSize, z_v_grad_w_dsl, z_v_grad_w,
      "z_v_grad_w", z_v_grad_w_rel_tol, z_v_grad_w_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_v_grad_w(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_velocity_advection_stencil_07", "z_v_grad_w");
  serialiser_z_v_grad_w.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(
        0, (mesh.NumEdges - 1), z_v_grad_w_kSize, (mesh.EdgeStride), z_v_grad_w,
        "mo_velocity_advection_stencil_07", "z_v_grad_w", iteration);
    serialize_dense_edges(0, (mesh.NumEdges - 1), z_v_grad_w_kSize,
                          (mesh.EdgeStride), z_v_grad_w_dsl,
                          "mo_velocity_advection_stencil_07", "z_v_grad_w_dsl",
                          iteration);
    std::cout << "[DSL] serializing z_v_grad_w as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_velocity_advection_stencil_07", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_velocity_advection_stencil_07(
    double *vn_ie, double *inv_dual_edge_length, double *w, double *z_vt_ie,
    double *inv_primal_edge_length, double *tangent_orientation, double *z_w_v,
    double *z_v_grad_w, double *z_v_grad_w_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double z_v_grad_w_rel_tol, const double z_v_grad_w_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_velocity_advection_stencil_07 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_velocity_advection_stencil_07(
      vn_ie, inv_dual_edge_length, w, z_vt_ie, inv_primal_edge_length,
      tangent_orientation, z_w_v, z_v_grad_w_before, verticalStart, verticalEnd,
      horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_velocity_advection_stencil_07 run time: " << time
            << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_velocity_advection_stencil_07...\n"
            << std::flush;
  verify_mo_velocity_advection_stencil_07(z_v_grad_w_before, z_v_grad_w,
                                          z_v_grad_w_rel_tol,
                                          z_v_grad_w_abs_tol, iteration);

  iteration++;
}

void setup_mo_velocity_advection_stencil_07(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int z_v_grad_w_k_size) {
  dawn_generated::cuda_ico::mo_velocity_advection_stencil_07::setup(
      mesh, k_size, stream, z_v_grad_w_k_size);
}

void free_mo_velocity_advection_stencil_07() {
  dawn_generated::cuda_ico::mo_velocity_advection_stencil_07::free();
}
}
