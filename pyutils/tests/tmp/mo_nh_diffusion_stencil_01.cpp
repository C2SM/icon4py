#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_nh_diffusion_stencil_01.hpp"
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

class mo_nh_diffusion_stencil_01 {
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
  double *diff_multfac_smag_;
  double *tangent_orientation_;
  double *inv_primal_edge_length_;
  double *inv_vert_vert_length_;
  double *u_vert_;
  double *v_vert_;
  double *primal_normal_vert_x_;
  double *primal_normal_vert_y_;
  double *dual_normal_vert_x_;
  double *dual_normal_vert_y_;
  double *vn_;
  double *smag_limit_;
  double *kh_smag_e_;
  double *kh_smag_ec_;
  double *z_nabla2_e_;
  double smag_offset_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int kh_smag_e_kSize_;
  inline static int kh_smag_ec_kSize_;
  inline static int z_nabla2_e_kSize_;

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

  static int get_kh_smag_ec_KSize() { return kh_smag_ec_kSize_; }

  static int get_z_nabla2_e_KSize() { return z_nabla2_e_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int kh_smag_e_kSize,
                    const int kh_smag_ec_kSize, const int z_nabla2_e_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    kh_smag_e_kSize_ = kh_smag_e_kSize;
    kh_smag_ec_kSize_ = kh_smag_ec_kSize;
    z_nabla2_e_kSize_ = z_nabla2_e_kSize;
  }

  mo_nh_diffusion_stencil_01() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_nh_diffusion_stencil_01 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto diff_multfac_smag_sid = get_sid(
        diff_multfac_smag_,
        gridtools::hymap::keys<unstructured::dim::vertical>::make_values(1));

    auto tangent_orientation_sid = get_sid(
        tangent_orientation_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto inv_primal_edge_length_sid = get_sid(
        inv_primal_edge_length_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto inv_vert_vert_length_sid = get_sid(
        inv_vert_vert_length_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

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

    auto primal_normal_vert_x_sid = get_sid(
        primal_normal_vert_x_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto primal_normal_vert_y_sid = get_sid(
        primal_normal_vert_y_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto dual_normal_vert_x_sid = get_sid(
        dual_normal_vert_x_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto dual_normal_vert_y_sid = get_sid(
        dual_normal_vert_y_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto vn_sid = get_sid(
        vn_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto smag_limit_sid = get_sid(
        smag_limit_,
        gridtools::hymap::keys<unstructured::dim::vertical>::make_values(1));

    auto kh_smag_e_sid = get_sid(
        kh_smag_e_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto kh_smag_ec_sid = get_sid(
        kh_smag_ec_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto z_nabla2_e_sid = get_sid(
        z_nabla2_e_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    gridtools::stencil::global_parameter smag_offset_gp{smag_offset_};
    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    neighbor_table_fortran<4> ecv_ptr{.raw_ptr_fortran = mesh_.ecvTable};
    neighbor_table_4new_sparse<4> e2ecv_ptr{};
    auto connectivities =
        gridtools::hymap::keys<generated::E2C2V_t,
                               generated::E2ECV_t>::make_values(ecv_ptr,
                                                                e2ecv_ptr);
    generated::mo_nh_diffusion_stencil_01(connectivities)(
        cuda_backend, diff_multfac_smag_sid, tangent_orientation_sid,
        inv_primal_edge_length_sid, inv_vert_vert_length_sid, u_vert_sid,
        v_vert_sid, primal_normal_vert_x_sid, primal_normal_vert_y_sid,
        dual_normal_vert_x_sid, dual_normal_vert_y_sid, vn_sid, smag_limit_sid,
        kh_smag_e_sid, kh_smag_ec_sid, z_nabla2_e_sid, smag_offset_gp,
        horizontalStart, horizontalEnd, verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *diff_multfac_smag, double *tangent_orientation,
                     double *inv_primal_edge_length,
                     double *inv_vert_vert_length, double *u_vert,
                     double *v_vert, double *primal_normal_vert_x,
                     double *primal_normal_vert_y, double *dual_normal_vert_x,
                     double *dual_normal_vert_y, double *vn, double *smag_limit,
                     double *kh_smag_e, double *kh_smag_ec, double *z_nabla2_e,
                     double smag_offset) {
    diff_multfac_smag_ = diff_multfac_smag;
    tangent_orientation_ = tangent_orientation;
    inv_primal_edge_length_ = inv_primal_edge_length;
    inv_vert_vert_length_ = inv_vert_vert_length;
    u_vert_ = u_vert;
    v_vert_ = v_vert;
    primal_normal_vert_x_ = primal_normal_vert_x;
    primal_normal_vert_y_ = primal_normal_vert_y;
    dual_normal_vert_x_ = dual_normal_vert_x;
    dual_normal_vert_y_ = dual_normal_vert_y;
    vn_ = vn;
    smag_limit_ = smag_limit;
    kh_smag_e_ = kh_smag_e;
    kh_smag_ec_ = kh_smag_ec;
    z_nabla2_e_ = z_nabla2_e;
    smag_offset_ = smag_offset;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_nh_diffusion_stencil_01(
    double *diff_multfac_smag, double *tangent_orientation,
    double *inv_primal_edge_length, double *inv_vert_vert_length,
    double *u_vert, double *v_vert, double *primal_normal_vert_x,
    double *primal_normal_vert_y, double *dual_normal_vert_x,
    double *dual_normal_vert_y, double *vn, double *smag_limit,
    double *kh_smag_e, double *kh_smag_ec, double *z_nabla2_e,
    double smag_offset, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_01 s;
  s.copy_pointers(diff_multfac_smag, tangent_orientation,
                  inv_primal_edge_length, inv_vert_vert_length, u_vert, v_vert,
                  primal_normal_vert_x, primal_normal_vert_y,
                  dual_normal_vert_x, dual_normal_vert_y, vn, smag_limit,
                  kh_smag_e, kh_smag_ec, z_nabla2_e, smag_offset);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_nh_diffusion_stencil_01(
    const double *kh_smag_e_dsl, const double *kh_smag_e,
    const double *kh_smag_ec_dsl, const double *kh_smag_ec,
    const double *z_nabla2_e_dsl, const double *z_nabla2_e,
    const double kh_smag_e_rel_tol, const double kh_smag_e_abs_tol,
    const double kh_smag_ec_rel_tol, const double kh_smag_ec_abs_tol,
    const double z_nabla2_e_rel_tol, const double z_nabla2_e_abs_tol,
    const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_01::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_01::getStream();
  int kSize = dawn_generated::cuda_ico::mo_nh_diffusion_stencil_01::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int kh_smag_e_kSize = dawn_generated::cuda_ico::mo_nh_diffusion_stencil_01::
      get_kh_smag_e_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.EdgeStride) * kh_smag_e_kSize, kh_smag_e_dsl, kh_smag_e,
      "kh_smag_e", kh_smag_e_rel_tol, kh_smag_e_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_kh_smag_e(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_nh_diffusion_stencil_01", "kh_smag_e");
  serialiser_kh_smag_e.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(0, (mesh.NumEdges - 1), kh_smag_e_kSize,
                          (mesh.EdgeStride), kh_smag_e,
                          "mo_nh_diffusion_stencil_01", "kh_smag_e", iteration);
    serialize_dense_edges(0, (mesh.NumEdges - 1), kh_smag_e_kSize,
                          (mesh.EdgeStride), kh_smag_e_dsl,
                          "mo_nh_diffusion_stencil_01", "kh_smag_e_dsl",
                          iteration);
    std::cout << "[DSL] serializing kh_smag_e as error is high.\n"
              << std::flush;
#endif
  }
  int kh_smag_ec_kSize = dawn_generated::cuda_ico::mo_nh_diffusion_stencil_01::
      get_kh_smag_ec_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.EdgeStride) * kh_smag_ec_kSize, kh_smag_ec_dsl, kh_smag_ec,
      "kh_smag_ec", kh_smag_ec_rel_tol, kh_smag_ec_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_kh_smag_ec(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_nh_diffusion_stencil_01", "kh_smag_ec");
  serialiser_kh_smag_ec.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(
        0, (mesh.NumEdges - 1), kh_smag_ec_kSize, (mesh.EdgeStride), kh_smag_ec,
        "mo_nh_diffusion_stencil_01", "kh_smag_ec", iteration);
    serialize_dense_edges(0, (mesh.NumEdges - 1), kh_smag_ec_kSize,
                          (mesh.EdgeStride), kh_smag_ec_dsl,
                          "mo_nh_diffusion_stencil_01", "kh_smag_ec_dsl",
                          iteration);
    std::cout << "[DSL] serializing kh_smag_ec as error is high.\n"
              << std::flush;
#endif
  }
  int z_nabla2_e_kSize = dawn_generated::cuda_ico::mo_nh_diffusion_stencil_01::
      get_z_nabla2_e_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.EdgeStride) * z_nabla2_e_kSize, z_nabla2_e_dsl, z_nabla2_e,
      "z_nabla2_e", z_nabla2_e_rel_tol, z_nabla2_e_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_nabla2_e(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_nh_diffusion_stencil_01", "z_nabla2_e");
  serialiser_z_nabla2_e.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(
        0, (mesh.NumEdges - 1), z_nabla2_e_kSize, (mesh.EdgeStride), z_nabla2_e,
        "mo_nh_diffusion_stencil_01", "z_nabla2_e", iteration);
    serialize_dense_edges(0, (mesh.NumEdges - 1), z_nabla2_e_kSize,
                          (mesh.EdgeStride), z_nabla2_e_dsl,
                          "mo_nh_diffusion_stencil_01", "z_nabla2_e_dsl",
                          iteration);
    std::cout << "[DSL] serializing z_nabla2_e as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_nh_diffusion_stencil_01", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_nh_diffusion_stencil_01(
    double *diff_multfac_smag, double *tangent_orientation,
    double *inv_primal_edge_length, double *inv_vert_vert_length,
    double *u_vert, double *v_vert, double *primal_normal_vert_x,
    double *primal_normal_vert_y, double *dual_normal_vert_x,
    double *dual_normal_vert_y, double *vn, double *smag_limit,
    double *kh_smag_e, double *kh_smag_ec, double *z_nabla2_e,
    double smag_offset, double *kh_smag_e_before, double *kh_smag_ec_before,
    double *z_nabla2_e_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double kh_smag_e_rel_tol, const double kh_smag_e_abs_tol,
    const double kh_smag_ec_rel_tol, const double kh_smag_ec_abs_tol,
    const double z_nabla2_e_rel_tol, const double z_nabla2_e_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_nh_diffusion_stencil_01 (" << iteration
            << ") ...\n"
            << std::flush;
  run_mo_nh_diffusion_stencil_01(
      diff_multfac_smag, tangent_orientation, inv_primal_edge_length,
      inv_vert_vert_length, u_vert, v_vert, primal_normal_vert_x,
      primal_normal_vert_y, dual_normal_vert_x, dual_normal_vert_y, vn,
      smag_limit, kh_smag_e_before, kh_smag_ec_before, z_nabla2_e_before,
      smag_offset, verticalStart, verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_nh_diffusion_stencil_01 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_nh_diffusion_stencil_01...\n"
            << std::flush;
  verify_mo_nh_diffusion_stencil_01(
      kh_smag_e_before, kh_smag_e, kh_smag_ec_before, kh_smag_ec,
      z_nabla2_e_before, z_nabla2_e, kh_smag_e_rel_tol, kh_smag_e_abs_tol,
      kh_smag_ec_rel_tol, kh_smag_ec_abs_tol, z_nabla2_e_rel_tol,
      z_nabla2_e_abs_tol, iteration);

  iteration++;
}

void setup_mo_nh_diffusion_stencil_01(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream,
                                      const int kh_smag_e_k_size,
                                      const int kh_smag_ec_k_size,
                                      const int z_nabla2_e_k_size) {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_01::setup(
      mesh, k_size, stream, kh_smag_e_k_size, kh_smag_ec_k_size,
      z_nabla2_e_k_size);
}

void free_mo_nh_diffusion_stencil_01() {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_01::free();
}
}
