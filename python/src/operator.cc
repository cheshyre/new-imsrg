// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

/// @file
/// Bindings of the operator module: OperatorSpec, Operator (numpy views,
/// arithmetic, norms), normal ordering, random operators, OperatorDump I/O
/// (design/18 sections 3 and 4).

#include <memory>
#include <string>
#include <vector>

#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "imsrg/base/error.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/normal_ordering.h"
#include "imsrg/operator/norms.h"
#include "imsrg/operator/operator_dump.h"
#include "imsrg/operator/random_operator.h"
#include "python/src/bindings.h"

namespace imsrg::py {

namespace nb = nanobind;
namespace {

/// Wraps a column-major host matrix as a zero-copy numpy view kept alive by
/// `owner`. Element strides are {1, ld}, i.e. the array is Fortran-ordered.
template <class T>
nb::object NumpyMatrixView(T* data, std::int64_t rows, std::int64_t cols,
                           std::int64_t ld, nb::handle owner) {
  using Arr = nb::ndarray<nb::numpy, T, nb::ndim<2>>;
  const std::size_t shape[2] = {static_cast<std::size_t>(rows),
                                static_cast<std::size_t>(cols)};
  const std::int64_t strides[2] = {1, ld};
  return nb::cast(Arr(data, 2, shape, owner, strides));
}

PyOperator& AsOp(nb::handle h) {
  auto& o = nb::cast<PyOperator&>(h);
  Alive(o);
  return o;
}

/// Pushes `o`'s host edits to the device, applies `f` to the C++ operator and
/// wraps the result as a library result (device current).
template <class F>
PyOperator Apply(const PyOperator& o, F&& f) {
  const op::Operator& in = Input(o);
  return Result(f(in));
}

}  // namespace

void BindOperator(nb::module_& m) {
  nb::enum_<Parity>(m, "Parity")
      .value("Even", Parity::kEven)
      .value("Odd", Parity::kOdd);
  nb::enum_<Residency>(m, "Residency",
                       "Which side of an Operator holds the current data "
                       "(design/18 section 2.2).")
      .value("HostCurrent", Residency::kHostCurrent)
      .value("DeviceCurrent", Residency::kDeviceCurrent)
      .value("Both", Residency::kBoth);
  nb::enum_<ResidencyPolicy>(m, "ResidencyPolicy",
                             "Whether a stale host view raises.")
      .value("Auto", ResidencyPolicy::kAuto)
      .value("Strict", ResidencyPolicy::kStrict)
      .value("Implicit", ResidencyPolicy::kImplicit);
  m.def("set_residency_policy", &SetResidencyPolicy, nb::arg("policy"),
        "Sets the process-wide residency policy: Auto (default: strict on "
        "GPU builds, implicit syncing where the host mirror aliases the "
        "device data), Strict (a host view of a DeviceCurrent operator raises "
        "StaleHostError; use it to write scripts that are portable to GPU "
        "nodes) or Implicit (never raises).");
  m.def("residency_policy", &GetResidencyPolicy,
        "The process-wide residency policy.");
  m.def("residency_enforced", &ResidencyEnforced,
        "True if a host view of a DeviceCurrent operator raises "
        "StaleHostError at the moment (policy Strict, or Auto on a GPU "
        "build).");

  nb::enum_<Hermiticity>(m, "Hermiticity")
      .value("Hermitian", Hermiticity::kHermitian)
      .value("AntiHermitian", Hermiticity::kAntiHermitian)
      .value("NonHermitian", Hermiticity::kNonHermitian);

  nb::class_<op::OperatorSpec>(m, "OperatorSpec",
                               "Symmetry and rank labels of an operator.")
      .def(
          "__init__",
          [](op::OperatorSpec* self, int particle_rank, int rank_J,
             Parity parity, int delta_Tz, Hermiticity herm,
             int delta_particle_number) {
            op::OperatorSpec s;
            s.particle_rank = particle_rank;
            s.rank_J = rank_J;
            s.parity = parity;
            s.delta_Tz = delta_Tz;
            s.hermiticity = herm;
            s.delta_particle_number = delta_particle_number;
            new (self) op::OperatorSpec(s);
          },
          nb::arg("particle_rank") = 2, nb::arg("rank_J") = 0,
          nb::arg("parity") = Parity::kEven, nb::arg("delta_Tz") = 0,
          nb::arg("hermiticity") = Hermiticity::kHermitian,
          nb::arg("delta_particle_number") = 0)
      .def_static("scalar", &op::OperatorSpec::Scalar,
                  nb::arg("particle_rank") = 2,
                  nb::arg("hermiticity") = Hermiticity::kHermitian)
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
               ", parity=" + (s.parity == Parity::kEven ? "Even" : "Odd") +
               ", delta_Tz=" + std::to_string(s.delta_Tz) + ")";
      });

  nb::class_<PyOperator>(
      m, "Operator",
      "Zero-, one- and two-body operator on a ModelSpace (value semantics).")
      .def(
          "__init__",
          [](PyOperator* self, const PyModelSpace& ms,
             const op::OperatorSpec& spec) {
            EnsureRuntime();
            new (self) PyOperator(op::Operator(ms.ms, spec), Residency::kBoth);
          },
          nb::arg("model_space"), nb::arg("spec") = op::OperatorSpec{})
      .def_prop_ro(
          "model_space",
          [](const PyOperator& o) { return PyModelSpace{o.model_space_ptr()}; })
      .def_prop_ro("spec", [](const PyOperator& o) { return o.spec(); })
      .def_prop_ro(
          "residency", [](const PyOperator& o) { return o.residency; },
          "Which side holds the current data: Residency.HostCurrent, "
          "DeviceCurrent or Both (design/18 section 2.2).")
      .def_prop_rw(
          "zero_body", [](const PyOperator& o) { return Alive(o).zero_body(); },
          [](PyOperator& o, double v) {
            Alive(o);
            o.zero_body() = v;
          },
          "Zero-body value (MeV).")
      .def_prop_ro(
          "one_body",
          [](nb::object self) {
            auto& o = AsOp(self);
            PrepareHostAccess(o, false);
            auto v = std::as_const(o).one_body().host_view();
            return NumpyMatrixView<const double>(v.data, v.rows, v.cols, v.ld,
                                                 MakeViewGuard(self));
          },
          "Read-only zero-copy numpy view (column-major) of the one-body "
          "matrix (host mirror). Raises StaleHostError if the host mirror is "
          "stale and residency is enforced: call sync_to_host() first.")
      .def(
          "one_body_view",
          [](nb::object self, bool writable) {
            auto& o = AsOp(self);
            PrepareHostAccess(o, writable);
            if (!writable) {
              auto v = std::as_const(o).one_body().host_view();
              return NumpyMatrixView<const double>(v.data, v.rows, v.cols, v.ld,
                                                   MakeViewGuard(self));
            }
            auto v = o.one_body().host_view();
            return NumpyMatrixView<double>(v.data, v.rows, v.cols, v.ld,
                                           MakeViewGuard(self));
          },
          nb::arg("writable") = false,
          "Zero-copy view of the one-body matrix. A writable view marks the "
          "device data stale (residency HostCurrent); compute calls push the "
          "edits to the device by themselves, or call sync_to_device().")
      .def(
          "two_body_block",
          [](nb::object self, int channel, bool writable) {
            auto& o = AsOp(self);
            const auto& tb = o.model_space_ptr()->two_body();
            if (channel < 0 || channel >= tb.num_channels())
              throw imsrg::InputError("channel index out of range");
            PrepareHostAccess(o, writable);
            const ChannelIndex ch{channel};
            if (!writable) {
              auto v = std::as_const(o).two_body().host_block(ch);
              return NumpyMatrixView<const double>(v.data, v.rows, v.cols, v.ld,
                                                   MakeViewGuard(self));
            }
            auto v = o.two_body().host_block(ch);
            return NumpyMatrixView<double>(v.data, v.rows, v.cols, v.ld,
                                           MakeViewGuard(self));
          },
          nb::arg("channel"), nb::arg("writable") = false,
          "Zero-copy numpy view of the two-body block of a channel "
          "(column-major); residency rules as for one_body_view.")
      .def(
          "sync_to_host",
          [](nb::object self) {
            auto& o = AsOp(self);
            if (o.residency == Residency::kDeviceCurrent) {
              o.SyncToHost();
              o.residency = Residency::kBoth;
            }
            return self;
          },
          "Refreshes the host mirror from the device data "
          "(DeviceCurrent -> Both); a no-op if the host mirror holds edits "
          "(HostCurrent). Returns the operator.")
      .def(
          "sync_to_device",
          [](nb::object self) {
            auto& o = AsOp(self);
            if (o.residency == Residency::kHostCurrent) {
              o.SyncToDevice();
              o.residency = Residency::kBoth;
            }
            return self;
          },
          "Copies host edits (writable views) to the device data "
          "(HostCurrent -> Both). Returns the operator.")
      .def(
          "set_zero",
          [](PyOperator& o) {
            Alive(o);
            o.set_zero();
            o.residency = Residency::kDeviceCurrent;
          },
          "Sets all parts to zero.")
      .def(
          "copy",
          [](const PyOperator& o) {
            Input(o);
            return PyOperator(o);
          },
          "Deep copy; the copy inherits the residency state.")
      .def("__copy__",
           [](const PyOperator& o) {
             Input(o);
             return PyOperator(o);
           })
      .def("__deepcopy__",
           [](const PyOperator& o, nb::handle) {
             Input(o);
             return PyOperator(o);
           })
      .def("__add__",
           [](const PyOperator& a, const PyOperator& b) {
             const op::Operator& x = Input(a);
             const op::Operator& y = Input(b);
             nb::gil_scoped_release r;
             return Result(op::operator+(x, y));
           })
      .def("__sub__",
           [](const PyOperator& a, const PyOperator& b) {
             const op::Operator& x = Input(a);
             const op::Operator& y = Input(b);
             nb::gil_scoped_release r;
             return Result(op::operator-(x, y));
           })
      .def("__neg__",
           [](const PyOperator& a) { return Result(op::operator-(Input(a))); })
      .def("__mul__",
           [](const PyOperator& a, double c) {
             return Result(op::operator*(Input(a), c));
           })
      .def("__rmul__",
           [](const PyOperator& a, double c) {
             return Result(op::operator*(c, Input(a)));
           })
      .def("__truediv__",
           [](const PyOperator& a, double c) {
             return Result(op::operator/(Input(a), c));
           })
      .def("__iadd__",
           [](nb::object self, const PyOperator& b) {
             auto& a = AsOp(self);
             Input(a);
             const op::Operator& y = Input(b);
             {
               nb::gil_scoped_release r;
               op::operator+=(a, y);
             }
             a.residency = Residency::kDeviceCurrent;
             return self;
           })
      .def("__isub__",
           [](nb::object self, const PyOperator& b) {
             auto& a = AsOp(self);
             Input(a);
             const op::Operator& y = Input(b);
             {
               nb::gil_scoped_release r;
               op::operator-=(a, y);
             }
             a.residency = Residency::kDeviceCurrent;
             return self;
           })
      .def("__imul__",
           [](nb::object self, double c) {
             auto& a = AsOp(self);
             Input(a);
             op::operator*=(a, c);
             a.residency = Residency::kDeviceCurrent;
             return self;
           })
      .def("__itruediv__",
           [](nb::object self, double c) {
             auto& a = AsOp(self);
             Input(a);
             op::operator/=(a, c);
             a.residency = Residency::kDeviceCurrent;
             return self;
           })
      .def(
          "norm", [](const PyOperator& o) { return op::Norm(Input(o)); },
          "Frobenius-type total norm (a device reduction; no host sync "
          "needed).")
      .def("one_body_norm",
           [](const PyOperator& o) { return op::OneBodyNorm(Input(o)); })
      .def("two_body_norm",
           [](const PyOperator& o) { return op::TwoBodyNorm(Input(o)); })
      .def("three_body_norm",
           [](const PyOperator& o) { return op::ThreeBodyNorm(Input(o)); })
      .def(
          "normal_ordered",
          [](const PyOperator& o) {
            const op::Operator& x = Input(o);
            nb::gil_scoped_release r;
            return Result(op::NormalOrder(x));
          },
          "Normal ordering w.r.t. the model-space reference (two-body "
          "approximation).")
      .def("undo_normal_order",
           [](const PyOperator& o) {
             const op::Operator& x = Input(o);
             nb::gil_scoped_release r;
             return Result(op::UndoNormalOrder(x));
           })
      .def("__repr__", [](const PyOperator& o) {
        if (o.released) return std::string("Operator(<released>)");
        return "Operator(zero_body=" + std::to_string(o.zero_body()) +
               ", norb=" + std::to_string(o.model_space_ptr()->num_orbits()) +
               ")";
      });

  m.def(
      "random_operator",
      [](const PyModelSpace& ms, const op::OperatorSpec& spec,
         std::uint64_t seed, double zero_body_scale, double one_body_scale,
         double two_body_scale) {
        EnsureRuntime();
        op::RandomOperatorOptions opt;
        opt.seed = seed;
        opt.zero_body_scale = zero_body_scale;
        opt.one_body_scale = one_body_scale;
        opt.two_body_scale = two_body_scale;
        return Result(op::RandomOperator(ms.ms, spec, opt));
      },
      nb::arg("model_space"), nb::arg("spec") = op::OperatorSpec{},
      nb::arg("seed") = 1, nb::arg("zero_body_scale") = 0.0,
      nb::arg("one_body_scale") = 1.0, nb::arg("two_body_scale") = 1.0,
      "Seeded, platform-reproducible random operator.");

  m.def(
      "read_operator_dump_header",
      [](const std::filesystem::path& path) {
        const auto h = op::ReadOperatorDumpHeader(path);
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
        EnsureRuntime();
        nb::gil_scoped_release r;
        std::shared_ptr<const basis::ModelSpace> space;
        if (ms) {
          space = ms->ms;
        } else {
          space = op::ModelSpaceForDump(op::ReadOperatorDumpHeader(path));
        }
        op::OperatorDumpReadOptions opt;
        opt.check_occupations = check_occupations;
        return Result(op::ReadOperatorDump(path, space, opt));
      },
      nb::arg("path"), nb::arg("model_space") = nb::none(),
      nb::arg("check_occupations") = true,
      "Reads an OperatorDump v1; builds the model space from the header if "
      "none is given.");

  m.def(
      "write_operator_dump",
      [](const std::filesystem::path& path, const PyOperator& o,
         const std::string& source, const std::string& reference) {
        const op::Operator& x = Input(o);
        nb::gil_scoped_release r;
        op::OperatorDumpWriteOptions opt;
        opt.source = source;
        opt.reference = reference;
        op::WriteOperatorDump(path, x, opt);
      },
      nb::arg("path"), nb::arg("operator"), nb::arg("source") = "imsrg",
      nb::arg("reference") = "");
}

}  // namespace imsrg::py
