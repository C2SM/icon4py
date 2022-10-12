#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_velocity_advection_stencil_20.hpp"
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

class mo_velocity_advection_stencil_20 {
public:
  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int VertexStride;
    int EdgeStride;
    int CellStride;
    int *ecTable;
    int *eceoTable;
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
      eceoTable = mesh->NeighborTables.at(
          std::tuple<std::vector<dawn::LocationType>, bool>{
              {dawn::LocationType::Edges, dawn::LocationType::Cells,
               dawn::LocationType::Edges},
              1});
      evTable = mesh->NeighborTables.at(
          std::tuple<std::vector<dawn::LocationType>, bool>{
              {dawn::LocationType::Edges, dawn::LocationType::Vertices}, 0});
    }
  };

private:
  int *levelmask_;
  double *c_lin_e_;
  double *z_w_con_c_full_;
  double *ddqz_z_full_e_;
  double *area_edge_;
  double *tangent_orientation_;
  double *inv_primal_edge_length_;
  double *zeta_;
  double *geofac_grdiv_;
  double *vn_;
  double *ddt_vn_adv_;
  double cfl_w_limit_;
  double scalfac_exdiff_;
  double d_time_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int ddt_vn_adv_kSize_;

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

  static int get_ddt_vn_adv_KSize() { return ddt_vn_adv_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int ddt_vn_adv_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    ddt_vn_adv_kSize_ = ddt_vn_adv_kSize;
  }

  mo_velocity_advection_stencil_20() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_velocity_advection_stencil_20 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto levelmask_sid = get_sid(
        levelmask_,
        gridtools::hymap::keys<unstructured::dim::vertical>::make_values(1));

    auto z_w_con_c_full_sid = get_sid(
        z_w_con_c_full_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto ddqz_z_full_e_sid = get_sid(
        ddqz_z_full_e_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto area_edge_sid = get_sid(
        area_edge_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto tangent_orientation_sid = get_sid(
        tangent_orientation_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto inv_primal_edge_length_sid = get_sid(
        inv_primal_edge_length_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto zeta_sid = get_sid(
        zeta_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.VertexStride));

    auto vn_sid = get_sid(
        vn_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto ddt_vn_adv_sid = get_sid(
        ddt_vn_adv_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    gridtools::stencil::global_parameter cfl_w_limit_gp{cfl_w_limit_};
    gridtools::stencil::global_parameter scalfac_exdiff_gp{scalfac_exdiff_};
    gridtools::stencil::global_parameter d_time_gp{d_time_};
    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    neighbor_table_fortran<2> ec_ptr{.raw_ptr_fortran = mesh_.ecTable};
    neighbor_table_fortran<5> eceo_ptr{.raw_ptr_fortran = mesh_.eceoTable};
    neighbor_table_fortran<2> ev_ptr{.raw_ptr_fortran = mesh_.evTable};
    auto connectivities =
        gridtools::hymap::keys<generated::E2C_t, generated::E2C2EO_t,
                               generated::E2V_t>::make_values(ec_ptr, eceo_ptr,
                                                              ev_ptr);
    double *c_lin_e_0 = &c_lin_e_[0 * mesh_.EdgeStride];
    double *c_lin_e_1 = &c_lin_e_[1 * mesh_.EdgeStride];
    auto c_lin_e_sid_0 = get_sid(
        c_lin_e_0,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto c_lin_e_sid_1 = get_sid(
        c_lin_e_1,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto c_lin_e_sid_comp = sid::composite::
        keys<integral_constant<int, 0>, integral_constant<int, 1>>::make_values(
            c_lin_e_sid_0, c_lin_e_sid_1);
    double *geofac_grdiv_0 = &geofac_grdiv_[0 * mesh_.EdgeStride];
    double *geofac_grdiv_1 = &geofac_grdiv_[1 * mesh_.EdgeStride];
    double *geofac_grdiv_2 = &geofac_grdiv_[2 * mesh_.EdgeStride];
    double *geofac_grdiv_3 = &geofac_grdiv_[3 * mesh_.EdgeStride];
    double *geofac_grdiv_4 = &geofac_grdiv_[4 * mesh_.EdgeStride];
    auto geofac_grdiv_sid_0 = get_sid(
        geofac_grdiv_0,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grdiv_sid_1 = get_sid(
        geofac_grdiv_1,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grdiv_sid_2 = get_sid(
        geofac_grdiv_2,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grdiv_sid_3 = get_sid(
        geofac_grdiv_3,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grdiv_sid_4 = get_sid(
        geofac_grdiv_4,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grdiv_sid_comp = sid::composite::keys<
        integral_constant<int, 0>, integral_constant<int, 1>,
        integral_constant<int, 2>, integral_constant<int, 3>,
        integral_constant<int, 4>>::make_values(geofac_grdiv_sid_0,
                                                geofac_grdiv_sid_1,
                                                geofac_grdiv_sid_2,
                                                geofac_grdiv_sid_3,
                                                geofac_grdiv_sid_4);
    generated::mo_velocity_advection_stencil_20(connectivities)(
        cuda_backend, levelmask_sid, c_lin_e_sid_comp, z_w_con_c_full_sid,
        ddqz_z_full_e_sid, area_edge_sid, tangent_orientation_sid,
        inv_primal_edge_length_sid, zeta_sid, geofac_grdiv_sid_comp, vn_sid,
        ddt_vn_adv_sid, cfl_w_limit_gp, scalfac_exdiff_gp, d_time_gp,
        horizontalStart, horizontalEnd, verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(int *levelmask, double *c_lin_e, double *z_w_con_c_full,
                     double *ddqz_z_full_e, double *area_edge,
                     double *tangent_orientation,
                     double *inv_primal_edge_length, double *zeta,
                     double *geofac_grdiv, double *vn, double *ddt_vn_adv,
                     double cfl_w_limit, double scalfac_exdiff, double d_time) {
    levelmask_ = levelmask;
    c_lin_e_ = c_lin_e;
    z_w_con_c_full_ = z_w_con_c_full;
    ddqz_z_full_e_ = ddqz_z_full_e;
    area_edge_ = area_edge;
    tangent_orientation_ = tangent_orientation;
    inv_primal_edge_length_ = inv_primal_edge_length;
    zeta_ = zeta;
    geofac_grdiv_ = geofac_grdiv;
    vn_ = vn;
    ddt_vn_adv_ = ddt_vn_adv;
    cfl_w_limit_ = cfl_w_limit;
    scalfac_exdiff_ = scalfac_exdiff;
    d_time_ = d_time;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_velocity_advection_stencil_20(
    int *levelmask, double *c_lin_e, double *z_w_con_c_full,
    double *ddqz_z_full_e, double *area_edge, double *tangent_orientation,
    double *inv_primal_edge_length, double *zeta, double *geofac_grdiv,
    double *vn, double *ddt_vn_adv, double cfl_w_limit, double scalfac_exdiff,
    double d_time, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_velocity_advection_stencil_20 s;
  s.copy_pointers(levelmask, c_lin_e, z_w_con_c_full, ddqz_z_full_e, area_edge,
                  tangent_orientation, inv_primal_edge_length, zeta,
                  geofac_grdiv, vn, ddt_vn_adv, cfl_w_limit, scalfac_exdiff,
                  d_time);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_velocity_advection_stencil_20(const double *ddt_vn_adv_dsl,
                                             const double *ddt_vn_adv,
                                             const double ddt_vn_adv_rel_tol,
                                             const double ddt_vn_adv_abs_tol,
                                             const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_velocity_advection_stencil_20::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_velocity_advection_stencil_20::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_velocity_advection_stencil_20::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int ddt_vn_adv_kSize = dawn_generated::cuda_ico::
      mo_velocity_advection_stencil_20::get_ddt_vn_adv_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.EdgeStride) * ddt_vn_adv_kSize, ddt_vn_adv_dsl, ddt_vn_adv,
      "ddt_vn_adv", ddt_vn_adv_rel_tol, ddt_vn_adv_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_ddt_vn_adv(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_velocity_advection_stencil_20", "ddt_vn_adv");
  serialiser_ddt_vn_adv.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(
        0, (mesh.NumEdges - 1), ddt_vn_adv_kSize, (mesh.EdgeStride), ddt_vn_adv,
        "mo_velocity_advection_stencil_20", "ddt_vn_adv", iteration);
    serialize_dense_edges(0, (mesh.NumEdges - 1), ddt_vn_adv_kSize,
                          (mesh.EdgeStride), ddt_vn_adv_dsl,
                          "mo_velocity_advection_stencil_20", "ddt_vn_adv_dsl",
                          iteration);
    std::cout << "[DSL] serializing ddt_vn_adv as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_velocity_advection_stencil_20", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_velocity_advection_stencil_20(
    int *levelmask, double *c_lin_e, double *z_w_con_c_full,
    double *ddqz_z_full_e, double *area_edge, double *tangent_orientation,
    double *inv_primal_edge_length, double *zeta, double *geofac_grdiv,
    double *vn, double *ddt_vn_adv, double cfl_w_limit, double scalfac_exdiff,
    double d_time, double *ddt_vn_adv_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double ddt_vn_adv_rel_tol, const double ddt_vn_adv_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_velocity_advection_stencil_20 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_velocity_advection_stencil_20(
      levelmask, c_lin_e, z_w_con_c_full, ddqz_z_full_e, area_edge,
      tangent_orientation, inv_primal_edge_length, zeta, geofac_grdiv, vn,
      ddt_vn_adv_before, cfl_w_limit, scalfac_exdiff, d_time, verticalStart,
      verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_velocity_advection_stencil_20 run time: " << time
            << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_velocity_advection_stencil_20...\n"
            << std::flush;
  verify_mo_velocity_advection_stencil_20(ddt_vn_adv_before, ddt_vn_adv,
                                          ddt_vn_adv_rel_tol,
                                          ddt_vn_adv_abs_tol, iteration);

  iteration++;
}

void setup_mo_velocity_advection_stencil_20(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int ddt_vn_adv_k_size) {
  dawn_generated::cuda_ico::mo_velocity_advection_stencil_20::setup(
      mesh, k_size, stream, ddt_vn_adv_k_size);
}

void free_mo_velocity_advection_stencil_20() {
  dawn_generated::cuda_ico::mo_velocity_advection_stencil_20::free();
}
}
