#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_nh_diffusion_stencil_04.hpp"
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

class mo_nh_diffusion_stencil_04 {
public:
  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int VertexStride;
    int EdgeStride;
    int CellStride;
    int *ecvTable;

    GpuTriMesh() {}

    GpuTriMesh(const dawn::GlobalGpuTriMesh *mesh) {
      NumVertices = mesh->NumVertices;
      NumCells = mesh->NumCells;
      NumEdges = mesh->NumEdges;
      VertexStride = mesh->VertexStride;
      CellStride = mesh->CellStride;
      EdgeStride = mesh->EdgeStride;
      ecvTable = mesh->NeighborTables.at(
          std::tuple<std::vector<dawn::LocationType>, bool>{
              {dawn::LocationType::Edges, dawn::LocationType::Cells,
               dawn::LocationType::Vertices},
              0});
    }
  };

private:
  double *u_vert_;
  double *v_vert_;
  double *primal_normal_vert_v1_;
  double *primal_normal_vert_v2_;
  double *z_nabla2_e_;
  double *inv_vert_vert_length_;
  double *inv_primal_edge_length_;
  double *z_nabla4_e2_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int z_nabla4_e2_kSize_;

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

  static int get_z_nabla4_e2_KSize() { return z_nabla4_e2_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int z_nabla4_e2_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    z_nabla4_e2_kSize_ = z_nabla4_e2_kSize;
  }

  mo_nh_diffusion_stencil_04() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_nh_diffusion_stencil_04 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto u_vert_sid = get_sid(
        u_vert_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.VertexStride));

    auto v_vert_sid = get_sid(
        v_vert_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.VertexStride));

    auto primal_normal_vert_v1_sid = get_sid(
        primal_normal_vert_v1_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto primal_normal_vert_v2_sid = get_sid(
        primal_normal_vert_v2_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto z_nabla2_e_sid = get_sid(
        z_nabla2_e_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto inv_vert_vert_length_sid = get_sid(
        inv_vert_vert_length_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto inv_primal_edge_length_sid = get_sid(
        inv_primal_edge_length_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto z_nabla4_e2_sid = get_sid(
        z_nabla4_e2_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    neighbor_table_fortran<4> ecv_ptr{.raw_ptr_fortran = mesh_.ecvTable};
    neighbor_table_4new_sparse<4> e2ecv_ptr{};
    auto connectivities =
        gridtools::hymap::keys<generated::E2C2V_t,
                               generated::E2ECV_t>::make_values(ecv_ptr,
                                                                e2ecv_ptr);
    generated::mo_nh_diffusion_stencil_04(connectivities)(
        cuda_backend, u_vert_sid, v_vert_sid, primal_normal_vert_v1_sid,
        primal_normal_vert_v2_sid, z_nabla2_e_sid, inv_vert_vert_length_sid,
        inv_primal_edge_length_sid, z_nabla4_e2_sid, horizontalStart,
        horizontalEnd, verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *u_vert, double *v_vert,
                     double *primal_normal_vert_v1,
                     double *primal_normal_vert_v2, double *z_nabla2_e,
                     double *inv_vert_vert_length,
                     double *inv_primal_edge_length, double *z_nabla4_e2) {
    u_vert_ = u_vert;
    v_vert_ = v_vert;
    primal_normal_vert_v1_ = primal_normal_vert_v1;
    primal_normal_vert_v2_ = primal_normal_vert_v2;
    z_nabla2_e_ = z_nabla2_e;
    inv_vert_vert_length_ = inv_vert_vert_length;
    inv_primal_edge_length_ = inv_primal_edge_length;
    z_nabla4_e2_ = z_nabla4_e2;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_nh_diffusion_stencil_04(
    double *u_vert, double *v_vert, double *primal_normal_vert_v1,
    double *primal_normal_vert_v2, double *z_nabla2_e,
    double *inv_vert_vert_length, double *inv_primal_edge_length,
    double *z_nabla4_e2, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_04 s;
  s.copy_pointers(u_vert, v_vert, primal_normal_vert_v1, primal_normal_vert_v2,
                  z_nabla2_e, inv_vert_vert_length, inv_primal_edge_length,
                  z_nabla4_e2);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_nh_diffusion_stencil_04(const double *z_nabla4_e2_dsl,
                                       const double *z_nabla4_e2,
                                       const double z_nabla4_e2_rel_tol,
                                       const double z_nabla4_e2_abs_tol,
                                       const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_04::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_04::getStream();
  int kSize = dawn_generated::cuda_ico::mo_nh_diffusion_stencil_04::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int z_nabla4_e2_kSize = dawn_generated::cuda_ico::mo_nh_diffusion_stencil_04::
      get_z_nabla4_e2_KSize();
  stencilMetrics =
      ::dawn::verify_field(stream, (mesh.EdgeStride) * z_nabla4_e2_kSize,
                           z_nabla4_e2_dsl, z_nabla4_e2, "z_nabla4_e2",
                           z_nabla4_e2_rel_tol, z_nabla4_e2_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_nabla4_e2(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_nh_diffusion_stencil_04", "z_nabla4_e2");
  serialiser_z_nabla4_e2.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(
        0, (mesh.NumEdges - 1), z_nabla4_e2_kSize, (mesh.EdgeStride),
        z_nabla4_e2, "mo_nh_diffusion_stencil_04", "z_nabla4_e2", iteration);
    serialize_dense_edges(0, (mesh.NumEdges - 1), z_nabla4_e2_kSize,
                          (mesh.EdgeStride), z_nabla4_e2_dsl,
                          "mo_nh_diffusion_stencil_04", "z_nabla4_e2_dsl",
                          iteration);
    std::cout << "[DSL] serializing z_nabla4_e2 as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_nh_diffusion_stencil_04", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_nh_diffusion_stencil_04(
    double *u_vert, double *v_vert, double *primal_normal_vert_v1,
    double *primal_normal_vert_v2, double *z_nabla2_e,
    double *inv_vert_vert_length, double *inv_primal_edge_length,
    double *z_nabla4_e2, double *z_nabla4_e2_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double z_nabla4_e2_rel_tol, const double z_nabla4_e2_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_nh_diffusion_stencil_04 (" << iteration
            << ") ...\n"
            << std::flush;
  run_mo_nh_diffusion_stencil_04(
      u_vert, v_vert, primal_normal_vert_v1, primal_normal_vert_v2, z_nabla2_e,
      inv_vert_vert_length, inv_primal_edge_length, z_nabla4_e2_before,
      verticalStart, verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_nh_diffusion_stencil_04 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_nh_diffusion_stencil_04...\n"
            << std::flush;
  verify_mo_nh_diffusion_stencil_04(z_nabla4_e2_before, z_nabla4_e2,
                                    z_nabla4_e2_rel_tol, z_nabla4_e2_abs_tol,
                                    iteration);

  iteration++;
}

void setup_mo_nh_diffusion_stencil_04(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream,
                                      const int z_nabla4_e2_k_size) {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_04::setup(
      mesh, k_size, stream, z_nabla4_e2_k_size);
}

void free_mo_nh_diffusion_stencil_04() {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_04::free();
}
}
