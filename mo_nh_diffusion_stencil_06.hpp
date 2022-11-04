
#include <cmath>
#include <gridtools/fn/backend/naive.hpp>
#include <gridtools/fn/unstructured.hpp>

namespace generated {

namespace gtfn = ::gridtools::fn;

namespace {
using namespace ::gridtools::literals;

using horizontal_t = gtfn::unstructured::dim::horizontal;
constexpr inline horizontal_t horizontal{};

using K_t = gtfn::unstructured::dim::vertical;
constexpr inline K_t K{};

struct _fun_1 {
  constexpr auto operator()() const {
    return [](auto const &z_nabla2_e, auto const &area_edge, auto const &vn,
              auto const &fac_bdydiff_v) {
      return (gtfn::deref(vn) +
              ((gtfn::deref(z_nabla2_e) * gtfn::deref(area_edge)) *
               gtfn::deref(fac_bdydiff_v)));
    };
  }
};

inline auto mo_nh_diffusion_stencil_06 = [](auto... connectivities__) {
  return
      [connectivities__...](auto backend, auto &&z_nabla2_e, auto &&area_edge,
                            auto &&vn, auto &&fac_bdydiff_v,
                            auto &&horizontal_start, auto &&horizontal_end,
                            auto &&vertical_start, auto &&vertical_end) {
        auto tmp_alloc__ = gtfn::backend::tmp_allocator(backend);

        make_backend(backend,
                     gtfn::unstructured_domain(
                         ::gridtools::tuple((horizontal_end - horizontal_start),
                                            (vertical_end - vertical_start)),
                         ::gridtools::tuple(horizontal_start, vertical_start),
                         connectivities__...))
            .stencil_executor()()
            .arg(vn)
            .arg(z_nabla2_e)
            .arg(area_edge)
            .arg(vn)
            .arg(fac_bdydiff_v)
            .assign(0_c, _fun_1(), 1_c, 2_c, 3_c, 4_c)
            .execute();
      };
};
} // namespace
} // namespace generated
