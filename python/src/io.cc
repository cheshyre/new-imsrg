// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

/// @file
/// Bindings of the io module: the me2j and NO2B readers and the tokyo (.snt)
/// and NuShellX (.int, .sp) writers (design/18 section 2.3). All of them
/// release the GIL.

#include <algorithm>
#include <filesystem>
#include <string>
#include <utility>

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>

#include "imsrg/io/me2j.h"
#include "imsrg/io/no2b.h"
#include "imsrg/io/valence_writers.h"
#include "imsrg/operator/no2b.h"
#include "python/src/bindings.h"

namespace imsrg::py {

namespace nb = nanobind;

namespace {

/// Writers read the host mirror: refresh it unless it holds edits.
const op::Operator& ForWriting(const PyOperator& o) {
  Alive(o);
  auto& m = const_cast<PyOperator&>(o);
  if (m.residency() == Residency::kDeviceCurrent) m.SyncToHost();
  return o;
}

}  // namespace

void BindIo(nb::module_& m) {
  nb::enum_<io::No2bPrecision>(m, "No2bPrecision",
                               "Value type of a binary NO2B file.")
      .value("Single", io::No2bPrecision::kSingle)
      .value("Half", io::No2bPrecision::kHalf);
  nb::enum_<io::ValenceWriteMode>(m, "ValenceWriteMode",
                                  "How two-body elements are written.")
      .value("Hamiltonian", io::ValenceWriteMode::kHamiltonian)
      .value("IsospinAveragedOperator",
             io::ValenceWriteMode::kIsospinAveragedOperator);

  nb::class_<op::No2bThreeBody>(
      m, "No2bThreeBody",
      "NO2B three-nucleon matrix elements in isospin coupling (single "
      "precision), the 3N input of Hartree-Fock and normal ordering.")
      .def(
          "__init__",
          [](op::No2bThreeBody* self, int emax, int e2max, int e3max,
             int lmax) {
            new (self)
                op::No2bThreeBody(op::No2bTruncation{emax, e2max, e3max, lmax});
          },
          nb::arg("emax"), nb::arg("e2max"), nb::arg("e3max"), nb::arg("lmax"),
          "Zero-filled storage for the truncations.")
      .def_prop_ro(
          "emax",
          [](const op::No2bThreeBody& v) { return v.truncation().emax; })
      .def_prop_ro(
          "e2max",
          [](const op::No2bThreeBody& v) { return v.truncation().e2max; })
      .def_prop_ro(
          "e3max",
          [](const op::No2bThreeBody& v) { return v.truncation().e3max; })
      .def_prop_ro(
          "lmax",
          [](const op::No2bThreeBody& v) { return v.truncation().lmax; })
      .def_prop_ro("num_channels", &op::No2bThreeBody::num_channels)
      .def_prop_ro("size", &op::No2bThreeBody::size,
                   "Number of stored values.");

  nb::class_<io::ValenceWriteOptions>(
      m, "ValenceWriteOptions",
      "Header information and mode of the valence-space writers (comment "
      "lines only, except `mode` and `hbar_omega` of the .snt file).")
      .def(
          "__init__",
          [](io::ValenceWriteOptions* self, io::ValenceWriteMode mode,
             std::string file_2n, std::string file_3n, int e1max, int e2max,
             int e3max, double hbar_omega, int a_ref, int z_ref,
             double a_target, std::string version) {
            io::ValenceWriteOptions o;
            o.mode = mode;
            o.file_2n = std::move(file_2n);
            o.file_3n = std::move(file_3n);
            o.e1max = e1max;
            o.e2max = e2max;
            o.e3max = e3max;
            o.hbar_omega = hbar_omega;
            o.a_ref = a_ref;
            o.z_ref = z_ref;
            o.a_target = a_target;
            o.version = std::move(version);
            new (self) io::ValenceWriteOptions(std::move(o));
          },
          nb::kw_only(), nb::arg("mode") = io::ValenceWriteMode::kHamiltonian,
          nb::arg("file_2n") = "", nb::arg("file_3n") = "",
          nb::arg("e1max") = 0, nb::arg("e2max") = 0, nb::arg("e3max") = 0,
          nb::arg("hbar_omega") = 0.0, nb::arg("a_ref") = 0,
          nb::arg("z_ref") = 0, nb::arg("a_target") = 0.0,
          nb::arg("version") = "")
      .def_rw("mode", &io::ValenceWriteOptions::mode)
      .def_rw("file_2n", &io::ValenceWriteOptions::file_2n)
      .def_rw("file_3n", &io::ValenceWriteOptions::file_3n)
      .def_rw("e1max", &io::ValenceWriteOptions::e1max)
      .def_rw("e2max", &io::ValenceWriteOptions::e2max)
      .def_rw("e3max", &io::ValenceWriteOptions::e3max)
      .def_rw("hbar_omega", &io::ValenceWriteOptions::hbar_omega)
      .def_rw("a_ref", &io::ValenceWriteOptions::a_ref)
      .def_rw("z_ref", &io::ValenceWriteOptions::z_ref)
      .def_rw("a_target", &io::ValenceWriteOptions::a_target)
      .def_rw("version", &io::ValenceWriteOptions::version);

  m.def(
      "me2j_value_count",
      [](int e1max, int e2max, int lmax, int emax_read) {
        return io::Me2jValueCount(io::Me2jTruncation{e1max, e2max, lmax},
                                  emax_read);
      },
      nb::arg("file_e1max"), nb::arg("file_e2max") = -1,
      nb::arg("file_lmax") = -1, nb::arg("emax_read") = -1,
      "Number of floats of a me2j file of the given truncation (or of the "
      "prefix a model space with emax = emax_read needs).");
  m.def(
      "no2b_value_count",
      [](int e1max, int e2max, int e3max, int lmax) {
        return io::No2bValueCount(
            io::No2bFileTruncation{e1max, e2max, e3max, lmax});
      },
      nb::arg("file_e1max"), nb::arg("file_e2max"), nb::arg("file_e3max"),
      nb::arg("file_lmax") = -1,
      "Number of values of a whole NO2B file of the given truncation.");

  m.def(
      "read_me2j",
      [](const std::filesystem::path& path, const PyModelSpace& ms,
         int file_e1max, int file_e2max, int file_lmax, bool read_whole_file) {
        EnsureRuntime();
        io::Me2jReadOptions opt;
        opt.file = {file_e1max, file_e2max, file_lmax};
        opt.read_whole_file = read_whole_file;
        GilFree r;
        return GuardRead(path, [&] {
          return Result(io::ReadMe2j(path, opt, ms.model_space()));
        });
      },
      nb::arg("path"), nb::arg("model_space"), nb::kw_only(),
      nb::arg("file_e1max"), nb::arg("file_e2max") = -1,
      nb::arg("file_lmax") = -1, nb::arg("read_whole_file") = false,
      "Reads a Darmstadt me2j file (plain or gzip) into a new Hermitian "
      "two-body operator on `model_space` (zero- and one-body parts zero: "
      "add the kinetic term separately). The truncation of the file must be "
      "given; file_e2max < 1 means 2*file_e1max and file_lmax < 1 means "
      "file_e1max. Values are in MeV in the normalized antisymmetrized "
      "convention. Raises InputError for unreadable or inconsistent files.");
  m.def(
      "read_no2b",
      [](const std::filesystem::path& path, const PyModelSpace& ms,
         int file_e1max, int file_e2max, int file_e3max, int e3max,
         int file_lmax, io::No2bPrecision precision, int lmax) {
        const auto& b = ms.model_space()->basis();
        const int emax = b.emax();
        const int l = lmax >= 0 ? lmax : (b.lmax() >= 0 ? b.lmax() : emax);
        op::No2bThreeBody v3(
            op::No2bTruncation{emax, 2 * emax, e3max, std::min(l, emax)});
        io::No2bReadOptions opt;
        opt.file = {file_e1max, file_e2max, file_e3max, file_lmax};
        opt.precision = precision;
        GilFree r;
        GuardRead(path, [&] {
          io::ReadNo2b(path, opt, v3);
          return 0;
        });
        return v3;
      },
      nb::arg("path"), nb::arg("model_space"), nb::kw_only(),
      nb::arg("file_e1max"), nb::arg("file_e2max"), nb::arg("file_e3max"),
      nb::arg("e3max"), nb::arg("file_lmax") = -1,
      nb::arg("precision") = io::No2bPrecision::kSingle, nb::arg("lmax") = -1,
      "Reads a NO2B three-nucleon file into a No2bThreeBody whose "
      "truncations are those of the model space (emax, e2max = 2 emax, lmax "
      "= min(lmax or emax, emax)) and `e3max`. A file name containing "
      "'stream.bin' is binary (raw little-endian values of `precision`), "
      "otherwise text (plain or gzip).");

  m.def(
      "write_tokyo",
      [](const std::filesystem::path& path, const PyOperator& o,
         const io::ValenceWriteOptions& opt) {
        const op::Operator& x = ForWriting(o);
        GilFree r;
        io::WriteTokyo(path, x, opt);
      },
      nb::arg("path"), nb::arg("operator"),
      nb::arg("options") = io::ValenceWriteOptions{},
      "Writes the valence-space interaction of the operator in the "
      "KSHELL/tokyo .snt format.");
  m.def(
      "write_nushellx_int",
      [](const std::filesystem::path& path, const PyOperator& o,
         const io::ValenceWriteOptions& opt) {
        const op::Operator& x = ForWriting(o);
        GilFree r;
        io::WriteNushellxInt(path, x, opt);
      },
      nb::arg("path"), nb::arg("operator"),
      nb::arg("options") = io::ValenceWriteOptions{},
      "Writes the valence-space interaction in the NuShellX .int format.");
  m.def(
      "write_nushellx_sps",
      [](const std::filesystem::path& path, const PyOperator& o) {
        const op::Operator& x = ForWriting(o);
        GilFree r;
        io::WriteNushellxSps(path, x);
      },
      nb::arg("path"), nb::arg("operator"),
      "Writes the valence model space in the NuShellX .sp format.");
}

}  // namespace imsrg::py
