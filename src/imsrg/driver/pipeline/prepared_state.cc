// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/driver/pipeline/prepared_state.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "imsrg/base/error.h"
#include "imsrg/base/log.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/driver/config/writer.h"
#include "imsrg/driver/pipeline/checkpoint.h"
#include "imsrg/io/operator_binary.h"
#include "imsrg/linalg/matrix_view.h"

namespace imsrg::driver::pipeline {
namespace {

namespace fs = std::filesystem;

constexpr int kVersion = 1;
constexpr const char* kManifest = "prepared.toml";

/// Exact text form of a double (hexfloat), quoted.
std::string Hex(double x) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "\"%a\"", x);
  return buf;
}

std::string HexArray(const std::vector<double>& v) {
  std::string s = "[";
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i > 0) s += ", ";
    s += Hex(v[i]);
  }
  return s + "]";
}

std::vector<double> ReferenceOccupations(const basis::ModelSpace& ms) {
  const auto occ = ms.reference().occupations();
  return {occ.begin(), occ.end()};
}

void WriteMatrix(const fs::path& path, const linalg::Matrix<double>& c) {
  io::WriteFileAtomically(path, [&](std::ostream& out) {
    const auto r = static_cast<std::uint64_t>(c.rows());
    const auto cc = static_cast<std::uint64_t>(c.cols());
    out.write(reinterpret_cast<const char*>(&r), 8);
    out.write(reinterpret_cast<const char*>(&cc), 8);
    for (std::int64_t j = 0; j < c.cols(); ++j) {
      for (std::int64_t i = 0; i < c.rows(); ++i) {
        const double x = c(i, j);
        out.write(reinterpret_cast<const char*>(&x), 8);
      }
    }
  });
}

linalg::Matrix<double> ReadMatrix(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw InputError("prepared state: cannot read '" + path.string() + "'");
  std::uint64_t rows = 0;
  std::uint64_t cols = 0;
  in.read(reinterpret_cast<char*>(&rows), 8);
  in.read(reinterpret_cast<char*>(&cols), 8);
  if (!in || rows > (1U << 20) || cols > (1U << 20)) {
    throw InputError("prepared state: '" + path.string() + "': bad header");
  }
  linalg::Matrix<double> c(static_cast<std::int64_t>(rows),
                           static_cast<std::int64_t>(cols));
  for (std::uint64_t j = 0; j < cols; ++j) {
    for (std::uint64_t i = 0; i < rows; ++i) {
      double x = 0.0;
      in.read(reinterpret_cast<char*>(&x), 8);
      c(static_cast<std::int64_t>(i), static_cast<std::int64_t>(j)) = x;
    }
  }
  if (!in) {
    throw InputError("prepared state: '" + path.string() + "' is truncated");
  }
  return c;
}

/// Writes the scalars, occupations and C of an HF/NAT solution under `key`.
void WriteSolution(std::ostream& out, const fs::path& dir,
                   const std::string& key, const hf::HFResult& r) {
  out << key << " = true\n";
  out << key << "_e_hf = " << Hex(r.e_hf) << "\n";
  out << key << "_e1 = " << Hex(r.e1) << "\n";
  out << key << "_e2 = " << Hex(r.e2) << "\n";
  out << key << "_e3 = " << Hex(r.e3) << "\n";
  out << key << "_iterations = " << r.iterations << "\n";
  out << key << "_converged = " << (r.converged ? "true" : "false") << "\n";
  out << key << "_occupations = " << HexArray(r.occupations) << "\n";
  const std::string file = "c_" + key + ".bin";
  WriteMatrix(dir / file, r.C);
  out << key << "_c = \"" << file << "\"\n";
}

/// The `key = value` lines of a manifest (a TOML subset: one key per line,
/// strings, integers, booleans and one-line arrays of strings).
class Manifest {
 public:
  explicit Manifest(const fs::path& path) : path_(path) {
    std::ifstream in(path);
    if (!in) {
      throw InputError("prepared state: cannot read '" + path.string() + "'");
    }
    for (std::string line; std::getline(in, line);) {
      if (line.empty() || line[0] == '#') continue;
      const auto eq = line.find(" = ");
      if (eq == std::string::npos) {
        throw InputError("prepared state '" + path.string() + "': bad line '" +
                         line + "'");
      }
      values_[line.substr(0, eq)] = line.substr(eq + 3);
    }
  }

  bool has(const std::string& key) const { return values_.count(key) > 0; }

  std::string String(const std::string& key) const {
    const std::string& v = Raw(key);
    if (v.size() < 2 || v.front() != '"' || v.back() != '"') {
      Bad(key, v);
    }
    return v.substr(1, v.size() - 2);
  }
  std::int64_t Int(const std::string& key) const {
    const std::string& v = Raw(key);
    char* end = nullptr;
    const std::int64_t x = std::strtoll(v.c_str(), &end, 10);
    if (end == v.c_str() || *end != '\0') Bad(key, v);
    return x;
  }
  bool Bool(const std::string& key) const {
    const std::string& v = Raw(key);
    if (v != "true" && v != "false") Bad(key, v);
    return v == "true";
  }
  double Double(const std::string& key) const {
    return ParseDouble(key, String(key));
  }
  std::vector<double> Doubles(const std::string& key) const {
    const std::string& v = Raw(key);
    if (v.size() < 2 || v.front() != '[' || v.back() != ']') Bad(key, v);
    std::vector<double> out;
    std::istringstream ss(v.substr(1, v.size() - 2));
    for (std::string item; std::getline(ss, item, ',');) {
      const auto a = item.find('"');
      const auto b = item.rfind('"');
      if (a == std::string::npos || b <= a) Bad(key, v);
      out.push_back(ParseDouble(key, item.substr(a + 1, b - a - 1)));
    }
    return out;
  }

 private:
  const std::string& Raw(const std::string& key) const {
    const auto it = values_.find(key);
    if (it == values_.end()) {
      throw InputError("prepared state '" + path_.string() + "': no key '" +
                       key + "'");
    }
    return it->second;
  }
  double ParseDouble(const std::string& key, const std::string& s) const {
    char* end = nullptr;
    const double x = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0') Bad(key, s);
    return x;
  }
  [[noreturn]] void Bad(const std::string& key, const std::string& v) const {
    throw InputError("prepared state '" + path_.string() + "': bad value '" +
                     v + "' of " + key);
  }

  fs::path path_;
  std::map<std::string, std::string> values_;
};

hf::HFResult ReadSolution(const Manifest& m, const fs::path& dir,
                          const std::string& key) {
  hf::HFResult r;
  r.e_hf = m.Double(key + "_e_hf");
  r.e1 = m.Double(key + "_e1");
  r.e2 = m.Double(key + "_e2");
  r.e3 = m.Double(key + "_e3");
  r.iterations = static_cast<std::int32_t>(m.Int(key + "_iterations"));
  r.converged = m.Bool(key + "_converged");
  r.occupations = m.Doubles(key + "_occupations");
  r.C = ReadMatrix(dir / m.String(key + "_c"));
  return r;
}

/// The model space `ms_bare` with occupations `occ`: `ms_bare` itself if they
/// are its own (as `hf::NormalOrderedHamiltonian` keeps the space when HF did
/// not re-fill), else a copy with a new reference named `name`.
std::shared_ptr<const basis::ModelSpace> WithOccupations(
    const std::shared_ptr<const basis::ModelSpace>& ms_bare,
    std::vector<double> occ, const std::string& name, const fs::path& path) {
  const basis::SingleParticleBasis& bb = ms_bare->basis();
  if (static_cast<std::int32_t>(occ.size()) != bb.num_orbits()) {
    throw InputError("prepared state '" + path.string() +
                     "': " + std::to_string(occ.size()) +
                     " occupations, the basis has " +
                     std::to_string(bb.num_orbits()) + " orbits");
  }
  if (occ == ReferenceOccupations(*ms_bare)) return ms_bare;
  return ms_bare->WithReference(
      basis::Reference::FromOccupations(bb, std::move(occ), name));
}

}  // namespace

std::uint64_t PreparationHash(const RunConfig& config) {
  // Only the physics inputs of H_0 (WE-A-4: keys added elsewhere in the schema
  // must not invalidate stored prepared states).
  static constexpr std::array<std::string_view, 10> kTables = {
      "basis",
      "reference",
      "valence",
      "hamiltonian",
      "interaction.two_body",
      "interaction.three_body",
      "hartree_fock",
      "natural_orbitals",
      "imsrg.space",
      "imsrg3"};
  // run.compat_enable / run.compat_disable (debug keys) are deliberately not
  // hashed: no compat option changes H_0 or the flow of the scalar
  // Hamiltonian; fork_step_defaults enters through the hashed
  // imsrg.step.domega.
  static constexpr std::array<std::string_view, 1> kRunKeys = {"compat"};
  return ConfigTablesHash(config, kTables, kRunKeys);
}

fs::path PreparedStateDirectory(const fs::path& checkpoint_dir) {
  return checkpoint_dir / "prepared";
}

bool HasPreparedState(const fs::path& checkpoint_dir) {
  std::error_code ec;
  return fs::exists(PreparedStateDirectory(checkpoint_dir) / kManifest, ec);
}

void WritePreparedState(const fs::path& checkpoint_dir,
                        const PreparedHamiltonian& ph,
                        const RunConfig& config) {
  const fs::path final_dir = PreparedStateDirectory(checkpoint_dir);
  fs::path tmp = final_dir;
  tmp += ".tmp";
  fs::remove_all(tmp);
  fs::create_directories(tmp);
  WriteStoredOperator(tmp / "hno", ph.hno, OperatorFileKind::kNative,
                      "H_0 (normal ordered, IMSRG model space)");
  std::ostringstream m;
  m << "# imsrg prepared Hamiltonian (design/00 D75): read by "
       "checkpoint.restart and checkpoint.prepared_from\n";
  m << "version = " << kVersion << "\n";
  m << "preparation_hash = \"" << PreparationHash(config) << "\"\n";
  m << "basis_kind = \"" << ToString(config.basis.kind) << "\"\n";
  m << "mass_number = " << ph.mass_number << "\n";
  m << "proton_number = " << ph.proton_number << "\n";
  m << "e_reference = " << Hex(ph.e_reference) << "\n";
  m << "e_mp2 = " << Hex(ph.e_mp2) << "\n";
  m << "hno = \"hno\"\n";
  const basis::ModelSpace& ms = ph.hno.model_space();
  m << "hno_reference = \"" << ms.reference().name() << "\"\n";
  m << "hno_occupations = " << HexArray(ReferenceOccupations(ms)) << "\n";
  if (ph.ms_normal_ordering) {
    const basis::ModelSpace& no = *ph.ms_normal_ordering;
    m << "normal_ordering_reference = \"" << no.reference().name() << "\"\n";
    m << "normal_ordering_occupations = " << HexArray(ReferenceOccupations(no))
      << "\n";
  }
  if (ph.hf) WriteSolution(m, tmp, "hf", *ph.hf);
  if (ph.nat) WriteSolution(m, tmp, "nat", *ph.nat);
  m << "natural_occupations = " << HexArray(ph.nat_occupations) << "\n";
  io::WriteFileAtomically(tmp / kManifest,
                          [&](std::ostream& out) { out << m.str(); });
  fs::remove_all(final_dir);
  fs::rename(tmp, final_dir);
  log::Info("prepared Hamiltonian written to {}", final_dir.string());
}

PreparedHamiltonian ReadPreparedState(const fs::path& checkpoint_dir,
                                      const RunConfig& config) {
  const fs::path dir = PreparedStateDirectory(checkpoint_dir);
  const Manifest m(dir / kManifest);
  if (m.Int("version") != kVersion) {
    throw InputError("prepared state '" + dir.string() +
                     "': unsupported version " +
                     std::to_string(m.Int("version")));
  }
  const std::string stored_hash = m.String("preparation_hash");
  const std::string hash = std::to_string(PreparationHash(config));
  if (stored_hash != hash) {
    throw InputError(
        "prepared state '" + dir.string() +
        "' was written for another Hamiltonian (preparation hash " +
        stored_hash + ", this configuration " + hash +
        "): basis, interaction, hamiltonian, reference, valence, "
        "hartree_fock, natural_orbitals, imsrg.space, imsrg3 and "
        "run.compat must be unchanged");
  }
  PreparedHamiltonian ph;
  ph.ms_bare = MakeModelSpace(config);
  ph.mass_number = static_cast<int>(m.Int("mass_number"));
  ph.proton_number = static_cast<int>(m.Int("proton_number"));
  ph.e_reference = m.Double("e_reference");
  ph.e_mp2 = m.Double("e_mp2");
  std::shared_ptr<const basis::ModelSpace> ms_hno;
  const std::vector<double> hno_occ = m.Doubles("hno_occupations");
  if (m.has("normal_ordering_occupations")) {
    if (!config.imsrg.space) {
      throw InputError("prepared state '" + dir.string() +
                       "' has a normal-ordering space but no imsrg.space");
    }
    ph.ms_normal_ordering =
        WithOccupations(ph.ms_bare, m.Doubles("normal_ordering_occupations"),
                        m.String("normal_ordering_reference"), dir);
    ms_hno = MakeImsrgModelSpace(config, *ph.ms_normal_ordering);
    if (ReferenceOccupations(*ms_hno) != hno_occ) {
      throw InputError("prepared state '" + dir.string() +
                       "': the IMSRG-space occupations do not match");
    }
  } else {
    ms_hno =
        WithOccupations(ph.ms_bare, hno_occ, m.String("hno_reference"), dir);
  }
  ph.hno =
      ReadStoredOperator(dir / m.String("hno"), ms_hno,
                         OperatorFileKind::kNative, Hermiticity::kHermitian);
  if (m.has("hf") && m.Bool("hf")) ph.hf = ReadSolution(m, dir, "hf");
  if (m.has("nat") && m.Bool("nat")) ph.nat = ReadSolution(m, dir, "nat");
  ph.nat_occupations = m.Doubles("natural_occupations");
  return ph;
}

}  // namespace imsrg::driver::pipeline
