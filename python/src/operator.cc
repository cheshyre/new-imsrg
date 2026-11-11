/// @file
/// Bindings of the operator module: OperatorSpec, Operator (numpy views, arithmetic, norms),
/// normal ordering, random operators, OperatorDump I/O (design/18 sections 3 and 4).

#include <string>
#include <vector>

#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bindings.hpp"
#include "imsrg/base/error.hpp"
#include "imsrg/operator/arithmetic.hpp"
#include "imsrg/operator/normal_ordering.hpp"
#include "imsrg/operator/norms.hpp"
#include "imsrg/operator/operator_dump.hpp"
#include "imsrg/operator/random_operator.hpp"

namespace imsrg::py {
namespace {

/// Wraps a column-major host matrix as a zero-copy numpy view kept alive by `owner`.
/// Element strides are {1, ld}, i.e. the array is Fortran-ordered.
template <class T>
nb::object matrix_view(T* data, std::int64_t rows, std::int64_t cols, std::int64_t ld,
                       nb::handle owner) {
  using Arr = nb::ndarray<nb::numpy, T, nb::ndim<2>>;
  const std::size_t shape[2] = {static_cast<std::size_t>(rows), static_cast<std::size_t>(cols)};
  const std::int64_t strides[2] = {1, ld};
  return nb::cast(Arr(data, 2, shape, owner, strides));
}

PyOperator& as_op(nb::handle h) { return nb::cast<PyOperator&>(h); }

}  // namespace

void bind_operator(nb::module_& m) {
  nb::enum_<Parity>(m, "Parity").value("Even", Parity::Even).value("Odd", Parity::Odd);
  nb::enum_<Hermiticity>(m, "Hermiticity")
      .value("Hermitian", Hermiticity::Hermitian)
      .value("AntiHermitian", Hermiticity::AntiHermitian)
      .value("NonHermitian", Hermiticity::NonHermitian);

  nb::class_<op::OperatorSpec>(m, "OperatorSpec", "Symmetry and rank labels of an operator.")
      .def(
          "__init__",
          [](op::OperatorSpec* self, int particle_rank, int rank_J, Parity parity, int delta_Tz,
             Hermiticity herm, int delta_particle_number) {
            op::OperatorSpec s;
            s.particle_rank = particle_rank;
            s.rank_J = rank_J;
            s.parity = parity;
            s.delta_Tz = delta_Tz;
            s.hermiticity = herm;
            s.delta_particle_number = delta_particle_number;
            new (self) op::OperatorSpec(s);
          },
          nb::arg("particle_rank") = 2, nb::arg("rank_J") = 0, nb::arg("parity") = Parity::Even,
          nb::arg("delta_Tz") = 0, nb::arg("hermiticity") = Hermiticity::Hermitian,
          nb::arg("delta_particle_number") = 0)
      .def_static("scalar", &op::OperatorSpec::scalar, nb::arg("particle_rank") = 2,
                  nb::arg("hermiticity") = Hermiticity::Hermitian)
      .def_rw("particle_rank", &op::OperatorSpec::particle_rank)
      .def_rw("rank_J", &op::OperatorSpec::rank_J)
      .def_rw("parity", &op::OperatorSpec::parity)
      .def_rw("delta_Tz", &op::OperatorSpec::delta_Tz)
      .def_rw("hermiticity", &op::OperatorSpec::hermiticity)
      .def_rw("delta_particle_number", &op::OperatorSpec::delta_particle_number)
      .def_prop_ro("is_scalar", &op::OperatorSpec::is_scalar)
      .def("__repr__", [](const op::OperatorSpec& s) {
        return "OperatorSpec(particle_rank=" + std::to_string(s.particle_rank) +
               ", rank_J=" + std::to_string(s.rank_J) +
               ", parity=" + (s.parity == Parity::Even ? "Even" : "Odd") +
               ", delta_Tz=" + std::to_string(s.delta_Tz) + ")";
      });

  nb::class_<PyOperator>(m, "Operator",
                         "Zero-, one- and two-body operator on a ModelSpace (value semantics).")
      .def(
          "__init__",
          [](PyOperator* self, const PyModelSpace& ms, const op::OperatorSpec& spec) {
            ensure_runtime();
            new (self) PyOperator(op::Operator(ms.ms, spec));
          },
          nb::arg("model_space"), nb::arg("spec") = op::OperatorSpec{})
      .def_prop_ro("model_space",
                   [](const PyOperator& o) { return PyModelSpace{o.model_space_ptr()}; })
      .def_prop_ro("spec", [](const PyOperator& o) { return o.spec(); })
      .def_prop_rw(
          "zero_body", [](const PyOperator& o) { return o.zero_body(); },
          [](PyOperator& o, double v) { o.zero_body() = v; }, "Zero-body value (MeV).")
      .def_prop_ro(
          "one_body",
          [](nb::object self) {
            auto& o = as_op(self);
            o.sync_to_host();
            auto v = std::as_const(o).one_body().host_view();
            return matrix_view<const double>(v.data, v.rows, v.cols, v.ld, self);
          },
          "Read-only zero-copy numpy view (column-major) of the one-body matrix (host mirror).")
      .def(
          "one_body_view",
          [](nb::object self, bool writable) {
            auto& o = as_op(self);
            o.sync_to_host();
            if (!writable) {
              auto v = std::as_const(o).one_body().host_view();
              return matrix_view<const double>(v.data, v.rows, v.cols, v.ld, self);
            }
            auto v = o.one_body().host_view();
            return matrix_view<double>(v.data, v.rows, v.cols, v.ld, self);
          },
          nb::arg("writable") = false,
          "Zero-copy view of the one-body matrix. After writing through a writable view call "
          "sync_to_device().")
      .def(
          "two_body_block",
          [](nb::object self, int channel, bool writable) {
            auto& o = as_op(self);
            const auto& tb = o.model_space_ptr()->two_body();
            if (channel < 0 || channel >= tb.num_channels())
              throw imsrg::InputError("channel index out of range");
            o.sync_to_host();
            const ChannelIndex ch{channel};
            if (!writable) {
              auto v = std::as_const(o).two_body().host_block(ch);
              return matrix_view<const double>(v.data, v.rows, v.cols, v.ld, self);
            }
            auto v = o.two_body().host_block(ch);
            return matrix_view<double>(v.data, v.rows, v.cols, v.ld, self);
          },
          nb::arg("channel"), nb::arg("writable") = false,
          "Zero-copy numpy view of the two-body block of a channel (column-major).")
      .def("sync_to_host", [](PyOperator& o) { o.sync_to_host(); })
      .def("sync_to_device", [](PyOperator& o) { o.sync_to_device(); },
           "Copies host edits (writable views) to the device data.")
      .def("set_zero", [](PyOperator& o) { o.set_zero(); })
      .def("copy", [](const PyOperator& o) { return PyOperator(static_cast<const op::Operator&>(o)); })
      .def("__copy__", [](const PyOperator& o) { return PyOperator(static_cast<const op::Operator&>(o)); })
      .def("__add__",
           [](const PyOperator& a, const PyOperator& b) {
             nb::gil_scoped_release r;
             return PyOperator(op::operator+(a, b));
           })
      .def("__sub__",
           [](const PyOperator& a, const PyOperator& b) {
             nb::gil_scoped_release r;
             return PyOperator(op::operator-(a, b));
           })
      .def("__neg__", [](const PyOperator& a) { return PyOperator(op::operator-(a)); })
      .def("__mul__", [](const PyOperator& a, double c) { return PyOperator(op::operator*(a, c)); })
      .def("__rmul__", [](const PyOperator& a, double c) { return PyOperator(op::operator*(c, a)); })
      .def("__truediv__",
           [](const PyOperator& a, double c) { return PyOperator(op::operator/(a, c)); })
      .def("__iadd__",
           [](nb::object self, const PyOperator& b) {
             {
               nb::gil_scoped_release r;
               op::operator+=(as_op(self), b);
             }
             return self;
           })
      .def("__isub__",
           [](nb::object self, const PyOperator& b) {
             {
               nb::gil_scoped_release r;
               op::operator-=(as_op(self), b);
             }
             return self;
           })
      .def("__imul__",
           [](nb::object self, double c) {
             op::operator*=(as_op(self), c);
             return self;
           })
      .def("__itruediv__",
           [](nb::object self, double c) {
             op::operator/=(as_op(self), c);
             return self;
           })
      .def("norm", [](const PyOperator& o) { return op::norm(o); }, "Frobenius-type total norm.")
      .def("one_body_norm", [](const PyOperator& o) { return op::one_body_norm(o); })
      .def("two_body_norm", [](const PyOperator& o) { return op::two_body_norm(o); })
      .def("three_body_norm", [](const PyOperator& o) { return op::three_body_norm(o); })
      .def("normal_ordered",
           [](const PyOperator& o) {
             nb::gil_scoped_release r;
             return PyOperator(op::normal_order(o));
           },
           "Normal ordering w.r.t. the model-space reference (two-body approximation).")
      .def("undo_normal_order",
           [](const PyOperator& o) {
             nb::gil_scoped_release r;
             return PyOperator(op::undo_normal_order(o));
           })
      .def("__repr__", [](const PyOperator& o) {
        return "Operator(zero_body=" + std::to_string(o.zero_body()) + ", norb=" +
               std::to_string(o.model_space_ptr()->num_orbits()) + ")";
      });

  m.def(
      "random_operator",
      [](const PyModelSpace& ms, const op::OperatorSpec& spec, std::uint64_t seed,
         double zero_body_scale, double one_body_scale, double two_body_scale) {
        ensure_runtime();
        op::RandomOperatorOptions opt;
        opt.seed = seed;
        opt.zero_body_scale = zero_body_scale;
        opt.one_body_scale = one_body_scale;
        opt.two_body_scale = two_body_scale;
        return PyOperator(op::random_operator(ms.ms, spec, opt));
      },
      nb::arg("model_space"), nb::arg("spec") = op::OperatorSpec{}, nb::arg("seed") = 1,
      nb::arg("zero_body_scale") = 0.0, nb::arg("one_body_scale") = 1.0,
      nb::arg("two_body_scale") = 1.0, "Seeded, platform-reproducible random operator.");

  m.def(
      "read_operator_dump_header",
      [](const std::filesystem::path& path) {
        const auto h = op::read_operator_dump_header(path);
        nb::dict d;
        d["source"] = h.source;
        d["emax"] = h.emax;
        d["lmax"] = h.lmax;
        d["hw"] = h.hw;
        d["reference"] = h.reference;
        d["spec"] = h.spec;
        std::vector<basis::Orbit> orbits;
        std::vector<double> occ;
        for (const auto& o : h.orbits) {
          orbits.push_back(o.orbit);
          occ.push_back(o.occupation);
        }
        d["orbits"] = orbits;
        d["occupations"] = occ;
        return d;
      },
      nb::arg("path"), "Header of an OperatorDump file as a dict.");

  m.def(
      "read_operator_dump",
      [](const std::filesystem::path& path, std::optional<PyModelSpace> ms,
         bool check_occupations) {
        ensure_runtime();
        nb::gil_scoped_release r;
        std::shared_ptr<const basis::ModelSpace> space;
        if (ms) {
          space = ms->ms;
        } else {
          space = op::model_space_for_dump(op::read_operator_dump_header(path));
        }
        op::OperatorDumpReadOptions opt;
        opt.check_occupations = check_occupations;
        return PyOperator(op::read_operator_dump(path, space, opt));
      },
      nb::arg("path"), nb::arg("model_space") = nb::none(), nb::arg("check_occupations") = true,
      "Reads an OperatorDump v1; builds the model space from the header if none is given.");

  m.def(
      "write_operator_dump",
      [](const std::filesystem::path& path, const PyOperator& o, const std::string& source,
         const std::string& reference) {
        nb::gil_scoped_release r;
        op::OperatorDumpWriteOptions opt;
        opt.source = source;
        opt.reference = reference;
        op::write_operator_dump(path, o, opt);
      },
      nb::arg("path"), nb::arg("operator"), nb::arg("source") = "imsrg",
      nb::arg("reference") = "");
}

}  // namespace imsrg::py
