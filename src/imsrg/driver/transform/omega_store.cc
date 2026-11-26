// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/driver/transform/omega_store.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/basis/orbit.h"
#include "imsrg/driver/pipeline/checkpoint.h"

namespace imsrg::driver::transform {

namespace fs = std::filesystem;

namespace {

std::string Trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\r");
  if (b == std::string::npos) return "";
  const auto e = s.find_last_not_of(" \t\r");
  return s.substr(b, e - b + 1);
}

std::string Unquote(const std::string& s) {
  if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
    return s.substr(1, s.size() - 2);
  }
  return s;
}

/// The quoted strings of a line (`["a", "b"],`).
std::vector<std::string> QuotedStrings(const std::string& line) {
  std::vector<std::string> out;
  std::size_t p = 0;
  while ((p = line.find('"', p)) != std::string::npos) {
    const auto q = line.find('"', p + 1);
    if (q == std::string::npos) break;
    out.push_back(line.substr(p + 1, q - p - 1));
    p = q + 1;
  }
  return out;
}

/// `name` without the extension of a stored operator file.
fs::path StemOf(const std::string& name, const fs::path& dir) {
  for (const std::string ext : {".opb", ".me1j", ".me2jp.bin"}) {
    if (name.size() > ext.size() &&
        name.compare(name.size() - ext.size(), ext.size(), ext) == 0) {
      return dir / name.substr(0, name.size() - ext.size());
    }
  }
  throw InputError("Omega manifest: unknown operator file '" + name + "'");
}

}  // namespace

OmegaManifest ReadOmegaManifest(const fs::path& path) {
  std::ifstream in(path);
  if (!in) throw InputError("cannot read the Omega manifest '" + path.string() + "'");
  const fs::path dir = path.parent_path();
  std::map<std::string, std::string> kv;
  OmegaManifest m;
  std::string line;
  bool in_files = false;
  while (std::getline(in, line)) {
    const std::string t = Trim(line);
    if (t.empty() || t[0] == '#') continue;
    if (in_files) {
      if (t == "]") {
        in_files = false;
        continue;
      }
      const auto names = QuotedStrings(t);
      if (names.empty()) {
        throw InputError("Omega manifest '" + path.string() +
                         "': bad files entry '" + t + "'");
      }
      m.omega_stems.push_back(StemOf(names.front(), dir));
      continue;
    }
    const auto eq = t.find('=');
    if (eq == std::string::npos) {
      throw InputError("Omega manifest '" + path.string() + "': bad line '" +
                       t + "'");
    }
    const std::string key = Trim(t.substr(0, eq));
    const std::string value = Trim(t.substr(eq + 1));
    if (key == "files") {
      if (value != "[") {
        throw InputError("Omega manifest: 'files = [' expected");
      }
      in_files = true;
      continue;
    }
    kv[key] = Unquote(value);
  }
  const auto need = [&](const std::string& key) -> const std::string& {
    const auto it = kv.find(key);
    if (it == kv.end()) {
      throw InputError("Omega manifest '" + path.string() + "': key '" + key +
                       "' missing");
    }
    return it->second;
  };
  const std::string& format = need("format");
  if (format == "native") {
    m.format = OperatorFileKind::kNative;
  } else if (format == "me2jp") {
    m.format = OperatorFileKind::kMe2jp;
  } else {
    throw InputError("Omega manifest: unknown format '" + format + "'");
  }
  m.emax = std::stoi(need("emax"));
  m.e2max = std::stoi(need("e2max"));
  m.hw = std::stod(need("hw"));
  m.layout_hash = std::stoull(need("layout_hash"));
  m.occupations = dir / need("occupations");
  if (kv.count("normal_ordering_occupations")) {
    m.normal_ordering_occupations = dir / kv["normal_ordering_occupations"];
  }
  if (kv.count("normal_ordering_emax")) {
    m.normal_ordering_emax = std::stoi(kv["normal_ordering_emax"]);
  }
  if (kv.count("basis_transformation")) {
    m.basis_transformation = dir / kv["basis_transformation"];
  }
  const auto n = static_cast<std::size_t>(std::stoul(need("num_omegas")));
  if (n != m.omega_stems.size() || n == 0) {
    throw InputError("Omega manifest '" + path.string() + "': num_omegas = " +
                     std::to_string(n) + " but " +
                     std::to_string(m.omega_stems.size()) + " files");
  }
  return m;
}

std::vector<OccupationRow> ReadOccupations(const fs::path& path) {
  std::ifstream in(path);
  if (!in) throw InputError("cannot read '" + path.string() + "'");
  std::vector<OccupationRow> rows;
  std::string line;
  while (std::getline(in, line)) {
    const std::string t = Trim(line);
    if (t.empty() || t[0] == '#') continue;
    std::istringstream s(t);
    OccupationRow r;
    std::string occ;
    if (!(s >> r.n >> r.l >> r.j2 >> r.tz2 >> occ)) {
      throw InputError("'" + path.string() + "': bad line '" + t + "'");
    }
    char* end = nullptr;
    r.occupation = std::strtod(occ.c_str(), &end);
    if (end == occ.c_str() || *end != '\0') {
      throw InputError("'" + path.string() + "': bad number '" + occ + "'");
    }
    rows.push_back(r);
  }
  return rows;
}

std::vector<double> OccupationsInBasis(const std::vector<OccupationRow>& rows,
                                       const basis::SingleParticleBasis& basis) {
  std::vector<double> occ(static_cast<std::size_t>(basis.num_orbits()), 0.0);
  for (const OccupationRow& r : rows) {
    const basis::Orbit o{r.n, r.l, TwoJ(r.j2), TwoTz(r.tz2)};
    const auto a = basis.find(o);
    if (!a) {
      throw InputError("occupation file: orbit (n l 2j 2tz) = " +
                       std::to_string(r.n) + " " + std::to_string(r.l) + " " +
                       std::to_string(r.j2) + " " + std::to_string(r.tz2) +
                       " is not in the basis");
    }
    occ[a->ToSize()] = r.occupation;
  }
  return occ;
}

linalg::Matrix<double> ReadBasisTransformation(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw InputError("cannot read '" + path.string() + "'");
  std::uint64_t rows = 0;
  std::uint64_t cols = 0;
  in.read(reinterpret_cast<char*>(&rows), 8);
  in.read(reinterpret_cast<char*>(&cols), 8);
  if (!in || rows > (1U << 20) || cols > (1U << 20)) {
    throw InputError("'" + path.string() + "': bad header");
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
  if (!in) throw InputError("'" + path.string() + "' is truncated");
  return c;
}

SharedOmegaStack::SharedOmegaStack(const OmegaManifest& manifest,
                                   std::shared_ptr<const basis::ModelSpace> ms,
                                   const comm::Communicator& node)
    : ms_(std::move(ms)),
      spec_(op::OperatorSpec::Scalar(2, Hermiticity::kAntiHermitian)),
      num_(manifest.omega_stems.size()) {
  IMSRG_CHECK(ms_ != nullptr, "SharedOmegaStack: null model space");
  {
    // Buffer sizes from an empty operator of the stored kind (same on every
    // rank, so the window size needs no communication).
    const op::Operator probe(ms_, spec_);
    n1_ = probe.one_body().buffer().host_span().size();
    n2_ = probe.two_body().matrix().buffer().host_span().size();
  }
  const std::size_t per = 1 + n1_ + n2_;
  memory_ = node.AllocateShared(num_ * per * sizeof(double));
  // The reader's exception must reach every rank: rank 0 reports success or
  // the message with a broadcast after filling the window.
  std::string error;
  if (node.is_root()) {
    try {
      auto* base = reinterpret_cast<double*>(memory_.data().data());
      for (std::size_t k = 0; k < num_; ++k) {
        op::Operator o =
            pipeline::ReadStoredOperator(manifest.omega_stems[k], ms_,
                                         manifest.format, spec_.hermiticity);
        if (!o.spec().SameSymmetry(spec_) ||
            o.spec().particle_rank != spec_.particle_rank) {
          throw InputError("Omega file '" + manifest.omega_stems[k].string() +
                           "' is not a scalar two-body operator");
        }
        o.SyncToHost();
        const auto one = o.one_body().buffer().host_span();
        const auto two = o.two_body().matrix().buffer().host_span();
        IMSRG_CHECK(one.size() == n1_ && two.size() == n2_,
                    "SharedOmegaStack: buffer size mismatch");
        double* dst = base + k * per;
        dst[0] = o.zero_body();
        std::memcpy(dst + 1, one.data(), n1_ * sizeof(double));
        std::memcpy(dst + 1 + n1_, two.data(), n2_ * sizeof(double));
      }
    } catch (const std::exception& e) {
      error = e.what();
      if (error.empty()) error = "unknown error";
    }
  }
  node.Broadcast(error, 0);
  node.Barrier();
  if (!error.empty()) throw InputError("reading the Omegas: " + error);
}

op::Operator SharedOmegaStack::Omega(std::size_t k) const {
  IMSRG_CHECK(k < num_, "SharedOmegaStack::Omega: index out of range");
  op::Operator o(ms_, spec_);
  const auto* src = reinterpret_cast<const double*>(memory_.data().data()) +
                    k * (1 + n1_ + n2_);
  o.zero_body() = src[0];
  auto one = o.one_body().buffer().host_span();
  auto two = o.two_body().matrix().buffer().host_span();
  std::memcpy(one.data(), src + 1, n1_ * sizeof(double));
  std::memcpy(two.data(), src + 1 + n1_, n2_ * sizeof(double));
  o.SyncToDevice();
  return o;
}

}  // namespace imsrg::driver::transform
