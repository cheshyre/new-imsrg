/// @file
/// Bindings of the basis module: Orbit, Channel, ModelSpace (design/18 section 3).

#include <string>
#include <vector>

#include "bindings.hpp"
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "imsrg/base/error.hpp"
#include "imsrg/basis/orbit.hpp"

namespace imsrg::py {
namespace {

/// A two-body channel (J, parity, Tz) with its index in the model space.
struct PyChannel {
  std::int32_t index = 0;
  std::int32_t two_j = 0;
  std::int32_t parity = 0;  // 0 even, 1 odd
  std::int32_t two_tz = 0;
  std::int32_t num_kets = 0;
};

using IntArray2 = nb::ndarray<nb::numpy, std::int32_t, nb::ndim<2>>;

/// Copies (p, q) pairs of `kets` into an (n, 2) int32 numpy array.
IntArray2 ket_array(std::span<const basis::Ket> kets) {
  auto* data = new std::int32_t[kets.size() * 2 + 1];
  for (std::size_t i = 0; i < kets.size(); ++i) {
    data[2 * i] = kets[i].p.value();
    data[2 * i + 1] = kets[i].q.value();
  }
  nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<std::int32_t*>(p); });
  return IntArray2(data, {kets.size(), 2}, owner);
}

PyChannel make_channel(const basis::TwoBodyChannels& tb, ChannelIndex ch) {
  const auto& c = tb.channel(ch);
  return {ch.value(), c.two_j.twice, static_cast<std::int32_t>(c.parity), c.two_tz.twice,
          c.num_kets};
}

}  // namespace

void bind_basis(nb::module_& m) {
  nb::class_<basis::Orbit>(m, "Orbit", "A single-particle orbit (n, l, 2j, 2tz); tz2 = -1 proton.")
      .def(
          "__init__",
          [](basis::Orbit* self, int n, int l, int two_j, int two_tz) {
            new (self) basis::Orbit{n, l, TwoJ{two_j}, TwoTz{two_tz}};
          },
          nb::arg("n"), nb::arg("l"), nb::arg("two_j"), nb::arg("two_tz"))
      .def_prop_ro("n", [](const basis::Orbit& o) { return o.n; })
      .def_prop_ro("l", [](const basis::Orbit& o) { return o.l; })
      .def_prop_ro("two_j", [](const basis::Orbit& o) { return o.j2.twice; })
      .def_prop_ro("two_tz", [](const basis::Orbit& o) { return o.tz2.twice; })
      .def_prop_ro("e", &basis::Orbit::e)
      .def_prop_ro("is_proton", &basis::Orbit::is_proton)
      .def_prop_ro("label", [](const basis::Orbit& o) { return basis::orbit_label(o); })
      .def("__eq__", [](const basis::Orbit& a, const basis::Orbit& b) { return a == b; })
      .def("__hash__",
           [](const basis::Orbit& o) {
             return (((o.n * 31 + o.l) * 31 + o.j2.twice) * 31) + o.tz2.twice;
           })
      .def("__repr__",
           [](const basis::Orbit& o) { return "Orbit(" + basis::orbit_label(o) + ")"; });

  nb::class_<PyChannel>(m, "Channel", "A two-body channel (2J, parity, 2Tz).")
      .def_ro("index", &PyChannel::index)
      .def_ro("two_j", &PyChannel::two_j)
      .def_ro("parity", &PyChannel::parity, "0 even, 1 odd")
      .def_ro("two_tz", &PyChannel::two_tz)
      .def_ro("num_kets", &PyChannel::num_kets)
      .def("__repr__", [](const PyChannel& c) {
        return "Channel(index=" + std::to_string(c.index) + ", two_j=" + std::to_string(c.two_j) +
               ", parity=" + std::to_string(c.parity) + ", two_tz=" + std::to_string(c.two_tz) +
               ", num_kets=" + std::to_string(c.num_kets) + ")";
      });

  nb::class_<PyModelSpace>(m, "ModelSpace",
                           "Immutable model space: orbits, reference, valence space, 2B channels.")
      .def(
          "__init__",
          [](PyModelSpace* self, int emax, const std::string& reference,
             const std::string& valence_space, double hw, int lmax, int emax_unocc,
             int lmax_unocc) {
            ensure_runtime();
            basis::BasisTruncation t;
            t.emax = emax;
            t.lmax = lmax;
            t.emax_unocc = emax_unocc;
            t.lmax_unocc = lmax_unocc;
            new (self) PyModelSpace{basis::ModelSpace::from_names(t, hw, reference, valence_space)};
          },
          nb::arg("emax"), nb::arg("reference") = "default", nb::arg("valence_space") = "default",
          nb::arg("hw") = 0.0, nb::arg("lmax") = -1, nb::arg("emax_unocc") = -1,
          nb::arg("lmax_unocc") = -1,
          "Builds the model space as the imsrg++ driver does (emax, reference nucleus name, "
          "valence space name, hbar omega in MeV).")
      .def_prop_ro("emax", [](const PyModelSpace& s) { return s.ms->basis().emax(); })
      .def_prop_ro("lmax", [](const PyModelSpace& s) { return s.ms->basis().lmax(); })
      .def_prop_ro("hbar_omega", [](const PyModelSpace& s) { return s.ms->basis().hbar_omega(); })
      .def_prop_ro("reference_name", [](const PyModelSpace& s) { return s.ms->reference().name(); })
      .def_prop_ro("num_orbits", [](const PyModelSpace& s) { return s.ms->num_orbits(); })
      .def_prop_ro("num_kets", [](const PyModelSpace& s) { return s.ms->two_body().num_kets(); })
      .def_prop_ro("num_channels",
                   [](const PyModelSpace& s) { return s.ms->two_body().num_channels(); })
      .def(
          "orbits",
          [](const PyModelSpace& s) {
            const auto o = s.ms->basis().orbits();
            return std::vector<basis::Orbit>(o.begin(), o.end());
          },
          "List of Orbit in index order.")
      .def("occupations",
           [](const PyModelSpace& s) {
             const auto o = s.ms->reference().occupations();
             return std::vector<double>(o.begin(), o.end());
           })
      .def(
          "kets", [](const PyModelSpace& s) { return ket_array(s.ms->two_body().kets()); },
          "All two-body kets as an (num_kets, 2) int32 array of orbit indices (p <= q).")
      .def(
          "channels",
          [](const PyModelSpace& s) {
            const auto& tb = s.ms->two_body();
            std::vector<PyChannel> out;
            for (std::int32_t i = 0; i < tb.num_channels(); ++i)
              out.push_back(make_channel(tb, ChannelIndex{i}));
            return out;
          },
          "List of Channel in index order.")
      .def(
          "channel",
          [](const PyModelSpace& s, int index) {
            const auto& tb = s.ms->two_body();
            if (index < 0 || index >= tb.num_channels())
              throw imsrg::InputError("channel index out of range");
            return make_channel(tb, ChannelIndex{index});
          },
          nb::arg("index"))
      .def(
          "find_channel",
          [](const PyModelSpace& s, int two_j, int parity, int two_tz) -> std::optional<PyChannel> {
            const auto& tb = s.ms->two_body();
            const auto ch =
                tb.find_channel(TwoJ{two_j}, static_cast<Parity>(parity), TwoTz{two_tz});
            if (!ch) return std::nullopt;
            return make_channel(tb, *ch);
          },
          nb::arg("two_j"), nb::arg("parity"), nb::arg("two_tz"))
      .def(
          "channel_kets",
          [](const PyModelSpace& s, int index) {
            const auto& tb = s.ms->two_body();
            if (index < 0 || index >= tb.num_channels())
              throw imsrg::InputError("channel index out of range");
            std::vector<basis::Ket> kets;
            for (const KetIndex k : tb.channel_kets(ChannelIndex{index})) kets.push_back(tb.ket(k));
            return ket_array(kets);
          },
          nb::arg("index"), "Kets of a channel in local order, as an (n, 2) int32 array.")
      .def("__eq__", [](const PyModelSpace& a, const PyModelSpace& b) { return a.ms == b.ms; })
      .def("__hash__", [](const PyModelSpace& s) { return std::hash<const void*>{}(s.ms.get()); });
}

}  // namespace imsrg::py
