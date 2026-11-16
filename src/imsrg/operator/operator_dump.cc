// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/operator/operator_dump.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <istream>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/basis/orbit.h"
#include "imsrg/basis/partition.h"
#include "imsrg/basis/reference.h"
#include "imsrg/basis/single_particle_basis.h"
#include "imsrg/basis/two_body_channels.h"

namespace imsrg::op {

namespace {

constexpr std::string_view kVersionLine = "# imsrg-operator-dump v1";

/// `%.17g` formatting of a double (round-trips exactly).
std::string Fmt17(double x) {
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.17g", x);
  return buf;
}

[[noreturn]] void FailAt(std::string_view name, int line,
                         const std::string& msg) {
  throw InputError(std::string(name) + ":" + std::to_string(line) + ": " + msg);
}

/// Whitespace-separated fields of one record line.
class Fields {
 public:
  Fields(std::string_view line, std::string_view name, int lineno)
      : name_(name), lineno_(lineno) {
    std::size_t i = 0;
    while (i < line.size()) {
      while (i < line.size() &&
             (line[i] == ' ' || line[i] == '\t' || line[i] == '\r'))
        ++i;
      std::size_t j = i;
      while (j < line.size() && line[j] != ' ' && line[j] != '\t' &&
             line[j] != '\r')
        ++j;
      if (j > i) tokens_.emplace_back(line.substr(i, j - i));
      i = j;
    }
  }
  [[noreturn]] void Fail(const std::string& msg) const {
    FailAt(name_, lineno_, msg);
  }
  bool empty() const { return pos_ >= tokens_.size(); }
  std::string NextToken(const char* what) {
    if (empty()) Fail(std::string("missing field ") + what);
    return tokens_[pos_++];
  }
  std::int32_t NextInt(const char* what) {
    const std::string tok = NextToken(what);
    std::int32_t v = 0;
    const auto [ptr, ec] =
        std::from_chars(tok.data(), tok.data() + tok.size(), v);
    if (ec != std::errc() || ptr != tok.data() + tok.size()) {
      Fail(std::string("cannot parse ") + what + " '" + tok + "'");
    }
    return v;
  }
  double NextDouble(const char* what) {
    const std::string tok = NextToken(what);
    char* end = nullptr;
    const double v = std::strtod(tok.c_str(), &end);
    if (end == tok.c_str() || *end != '\0') {
      Fail(std::string("cannot parse ") + what + " '" + tok + "'");
    }
    return v;
  }
  void ExpectKey(const std::string& key) {
    const std::string tok = NextToken(key.c_str());
    if (tok != key) Fail("expected key '" + key + "', got '" + tok + "'");
  }
  void ExpectEnd() const {
    if (!empty()) Fail("unexpected trailing field '" + tokens_[pos_] + "'");
  }
  int lineno() const { return lineno_; }

 private:
  std::string_view name_;
  int lineno_;
  std::vector<std::string> tokens_;
  std::size_t pos_ = 0;
};

/// Line reader that skips comments (except line 1) and blank lines and tracks
/// line numbers.
class LineReader {
 public:
  LineReader(std::istream& in, std::string_view name) : in_(in), name_(name) {}

  [[noreturn]] void Fail(const std::string& msg) const {
    FailAt(name_, lineno_ + 1, msg);
  }
  /// Next significant line; false at end of file.
  bool Next(std::string& line) {
    while (std::getline(in_, line)) {
      ++lineno_;
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (lineno_ > 1 && (line.empty() || line[0] == '#')) continue;
      return true;
    }
    return false;
  }
  /// Next record, which must start with `keyword`; the keyword is consumed.
  Fields Expect(const std::string& keyword) {
    std::string line;
    if (!Next(line)) Fail("unexpected end of file, expected '" + keyword + "'");
    Fields f(line, name_, lineno_);
    const std::string kw = f.empty() ? std::string() : f.NextToken("keyword");
    if (kw != keyword) f.Fail("expected '" + keyword + "', got '" + kw + "'");
    return f;
  }
  /// A data line inside `section`.
  Fields data(const std::string& section) {
    std::string line;
    if (!Next(line)) Fail("unexpected end of file in " + section);
    return Fields(line, name_, lineno_);
  }
  void CheckVersion() {
    std::string line;
    if (!Next(line))
      Fail("empty file, expected '" + std::string(kVersionLine) + "'");
    if (line != kVersionLine) {
      FailAt(name_, lineno_,
             "missing or unsupported version line (expected '" +
                 std::string(kVersionLine) + "')");
    }
  }
  int lineno() const { return lineno_; }

 private:
  std::istream& in_;
  std::string_view name_;
  int lineno_ = 0;
};

/// Header plus the line numbers needed for later validation.
struct ParsedHeader {
  OperatorDumpHeader header;
  int basis_line = 0;
  std::vector<int> orbit_lines;
  int spec_line = 0;
};

std::int32_t ReadCount(Fields& f, const char* what) {
  const auto n = f.NextInt(what);
  if (n < 0) f.Fail(std::string("negative ") + what);
  return n;
}

ParsedHeader ParseHeader(LineReader& r) {
  ParsedHeader h;
  OperatorDumpHeader& hd = h.header;
  r.CheckVersion();
  {
    std::string line;
    if (!r.Next(line)) r.Fail("unexpected end of file, expected 'source'");
    if (line != "source" && line.rfind("source ", 0) != 0) {
      r.Fail("expected 'source', got '" + line.substr(0, line.find(' ')) + "'");
    }
    hd.source = line.size() > 7 ? line.substr(7) : "";
  }
  {
    auto f = r.Expect("basis");
    h.basis_line = f.lineno();
    f.ExpectKey("emax");
    hd.emax = f.NextInt("emax");
    f.ExpectKey("lmax");
    hd.lmax = f.NextInt("lmax");
    f.ExpectKey("hw");
    hd.hw = f.NextDouble("hw");
    f.ExpectEnd();
  }
  {
    auto f = r.Expect("reference");
    hd.reference = f.NextToken("reference");
    f.ExpectEnd();
  }
  std::int32_t norb = 0;
  {
    auto f = r.Expect("norbits");
    norb = ReadCount(f, "norbits");
    f.ExpectEnd();
  }
  for (std::int32_t i = 0; i < norb; ++i) {
    auto f = r.Expect("orbit");
    const auto idx = f.NextInt("index");
    basis::Orbit o;
    o.n = f.NextInt("n");
    o.l = f.NextInt("l");
    o.j2 = TwoJ{f.NextInt("2j")};
    o.tz2 = TwoTz{f.NextInt("2tz")};
    const double occ = f.NextDouble("occupation");
    f.ExpectEnd();
    if (idx != i) f.Fail("orbit indices must be 0..N-1 in order");
    if (!basis::IsValid(o)) f.Fail("invalid orbit quantum numbers");
    hd.orbits.push_back({o, occ});
    h.orbit_lines.push_back(f.lineno());
  }
  {
    auto f = r.Expect("spec");
    h.spec_line = f.lineno();
    f.ExpectKey("particle_rank");
    hd.spec.particle_rank = f.NextInt("particle_rank");
    f.ExpectKey("rank_J");
    hd.spec.rank_J = f.NextInt("rank_J");
    f.ExpectKey("parity");
    const auto par = f.NextInt("parity");
    if (par != 0 && par != 1) f.Fail("parity must be 0 or 1");
    hd.spec.parity = par == 0 ? Parity::kEven : Parity::kOdd;
    f.ExpectKey("delta_Tz");
    hd.spec.delta_Tz = f.NextInt("delta_Tz");
    f.ExpectKey("hermiticity");
    const std::string hc = f.NextToken("hermiticity");
    f.ExpectEnd();
    if (hc.size() != 1 || (hc[0] != 'H' && hc[0] != 'A' && hc[0] != 'N')) {
      f.Fail("bad hermiticity '" + hc + "' (expected H, A or N)");
    }
    hd.spec.hermiticity = HermiticityFromCode(hc[0]);
  }
  return h;
}

/// A located ket side: channel label, channel index, local index and exchange
/// phase.
struct Side {
  TwoJ two_j{0};
  Parity parity = Parity::kEven;
  TwoTz two_tz{0};
  std::optional<ChannelIndex> ch;
  std::optional<LocalKetIndex> idx;
  int phase = 1;
};

Side LocateSide(const basis::TwoBodyChannels& tb, OrbitIndex a, OrbitIndex b,
                TwoJ two_j) {
  const auto orbits = tb.orbits();
  const basis::Orbit& oa = orbits[a.ToSize()];
  const basis::Orbit& ob = orbits[b.ToSize()];
  Side s;
  s.two_j = two_j;
  s.parity = oa.parity() * ob.parity();
  s.two_tz = TwoTz{oa.tz2.twice + ob.tz2.twice};
  s.ch = tb.find_channel(two_j, s.parity, s.two_tz);
  if (!s.ch) return s;
  const bool swap = a.value() > b.value();
  const OrbitIndex p = swap ? b : a;
  const OrbitIndex q = swap ? a : b;
  s.idx = tb.local_index(*s.ch, tb.ket_index(p, q));
  if (swap) s.phase = basis::ExchangePhase(oa.j2, ob.j2, two_j);
  return s;
}

/// Accumulates max|m| and max|m - sgn m^T| of a column-major n x n block at
/// `data`.
void AccumulateSymmetry(const double* data, std::int64_t n, double sgn,
                        double& vmax, double& dev) {
  for (std::int64_t j = 0; j < n; ++j) {
    for (std::int64_t i = 0; i < n; ++i) {
      const double v = data[i + j * n];
      vmax = std::max(vmax, std::abs(v));
      dev = std::max(dev, std::abs(v - sgn * data[j + i * n]));
    }
  }
}

/// The records of the three-body section that lie above the diagonal of a
/// Hermitian or anti-Hermitian part (the packed storage keeps the lower
/// triangle only): their values per stored element, to check the declared
/// symmetry after reading.
class ThreeBodyRecords {
 public:
  /// Prepares the storage for `t` (packed parts only).
  void Init(const ThreeBody& t) {
    if (t.packed()) values_.assign(t.size(), 0.0);
  }
  /// Records the file value `v` of the element (i, j), i < j, of block `ch`.
  void Set(std::size_t index, double v) { values_[index] = v; }

  /// Adds max|V| and the largest violation of the declared symmetry to `vmax`
  /// and `dev`: `lower` are the stored values of `t` (packed: lower triangle
  /// as read; the upper-triangle records must equal h times them, and the
  /// anti-Hermitian diagonal must be zero as read).
  void Accumulate(const ThreeBody& t, const std::vector<double>& lower,
                  double& vmax, double& dev) const {
    if (!t.packed()) {
      for (const double v : lower) vmax = std::max(vmax, std::abs(v));
      return;
    }
    const double h = HermiticitySign(t.hermiticity());
    const auto offsets = t.block_offsets();
    for (std::size_t ch = 0; ch + 1 < offsets.size(); ++ch) {
      const std::int64_t n =
          t.dim(Channel3Index{static_cast<std::int32_t>(ch)});
      for (std::int64_t j = 0; j < n; ++j) {
        for (std::int64_t i = j; i < n; ++i) {
          const std::size_t k = static_cast<std::size_t>(
              offsets[ch] + ThreeBody::PackedIndex(n, i, j));
          const double lo = lower[k];
          vmax = std::max(vmax, std::max(std::abs(lo), std::abs(values_[k])));
          // i == j: the lower record stands for both; h = -1 requires zero.
          dev = std::max(dev, i == j ? std::abs(lo - h * lo)
                                     : std::abs(values_[k] - h * lo));
        }
      }
    }
  }

 private:
  std::vector<double> values_;
};

/// Reads one `a b c Jab d e f Jde 2J value` record and stores it in `t`.
/// Both labels of the record must be canonical kept kets of the same channel
/// (`two_j`, parity, 2Tz); errors carry the line of `f`. Lower-triangle (and
/// dense) elements are written to the host mirror, upper-triangle ones to
/// `upper`.
void ReadThreeBodyRecord(Fields& f, const basis::ModelSpace& ms,
                         const OperatorDumpHeader& hd,
                         const std::vector<OrbitIndex>& map, ThreeBody& t,
                         ThreeBodyRecords& upper) {
  const auto norb = static_cast<std::int32_t>(hd.orbits.size());
  std::int32_t x[8];
  static const char* const kLabels[8] = {"a", "b", "c", "Jab",
                                         "d", "e", "f", "Jde"};
  for (int k = 0; k < 8; ++k) x[k] = f.NextInt(kLabels[k]);
  const std::int32_t two_j = f.NextInt("2J");
  const double v = f.NextDouble("value");
  f.ExpectEnd();
  for (const int k : {0, 1, 2, 4, 5, 6}) {
    if (x[k] < 0 || x[k] >= norb) f.Fail("orbit index out of range");
  }
  if (x[3] < 0 || x[7] < 0) f.Fail("negative J");
  if (two_j <= 0 || two_j % 2 == 0) f.Fail("2J must be a positive odd number");
  if (x[0] > x[1] || x[1] > x[2] || x[4] > x[5] || x[5] > x[6]) {
    f.Fail("three-body entries need a <= b <= c and d <= e <= f");
  }
  const basis::ThreeBodySpace& space = t.space();
  const auto& sp = ms.basis();
  struct KetLoc {
    Channel3Index ch;
    basis::KeptRow3Index row;
  };
  const auto locate = [&](int k0, const char* what) {
    OrbitIndex o[3];
    Parity par = Parity::kEven;
    std::int32_t tz = 0;
    for (int m = 0; m < 3; ++m) {
      o[m] = map[static_cast<std::size_t>(x[k0 + m])];
      const basis::Orbit& orb = sp.orbit(o[m]);
      par = par * orb.parity();
      tz += orb.tz2.twice;
    }
    if (o[0].value() > o[1].value() || o[1].value() > o[2].value()) {
      f.Fail(std::string(what) +
             " ket is not canonical in the model space's orbit order");
    }
    const auto ch = ms.three_body()->find_channel(TwoJ{two_j}, par, TwoTz{tz});
    if (!ch) {
      f.Fail(std::string(what) +
             " ket: 2J, parity or 2Tz does not belong to a three-body channel");
    }
    const auto row =
        space.find_row(*ch, o[0], o[1], o[2], TwoJ::FromInteger(x[k0 + 3]));
    if (!row) {
      f.Fail(std::string(what) +
             " ket |(a b) Jab, c; 2J> is not a kept ket of the three-body "
             "space (not coupled, cut, or outside its channel)");
    }
    return KetLoc{*ch, *row};
  };
  const KetLoc bra = locate(0, "bra");
  const KetLoc ket = locate(4, "ket");
  if (bra.ch != ket.ch) {
    f.Fail("bra and ket are in different three-body channels");
  }
  const std::int64_t n = t.dim(bra.ch);
  const std::int64_t i = bra.row.value();
  const std::int64_t j = ket.row.value();
  const std::int64_t off = t.block_offsets()[bra.ch.ToSize()];
  auto host = t.host_span();
  if (!t.packed()) {
    host[static_cast<std::size_t>(off + i + j * n)] = v;
  } else if (i >= j) {
    host[static_cast<std::size_t>(off + ThreeBody::PackedIndex(n, i, j))] = v;
  } else {
    upper.Set(static_cast<std::size_t>(off + ThreeBody::PackedIndex(n, j, i)),
              v);
  }
}

/// Writes the `three_body` section: channel by channel, bra row then ket row,
/// every nonzero element of the full matrix.
void WriteThreeBody(std::ostream& out, const ThreeBody& t) {
  const basis::ThreeBodySpace& space = t.space();
  const std::vector<double> data = t.buffer().ToVector();
  const auto offsets = t.block_offsets();
  const int h = HermiticitySign(t.hermiticity());
  std::ostringstream body;
  std::int64_t count = 0;
  for (std::int32_t c = 0; c < space.num_channels(); ++c) {
    const basis::KeptChannel3& kc = space.channel(Channel3Index{c});
    const std::int64_t n = kc.dim;
    const std::int64_t off = offsets[static_cast<std::size_t>(c)];
    const auto label = [&](std::int64_t row) {
      const std::int64_t flat = kc.row_offset + row;
      const basis::OrbitTriple& tr = space.triple(space.row_triple(flat));
      return std::to_string(tr.a.value()) + " " + std::to_string(tr.b.value()) +
             " " + std::to_string(tr.c.value()) + " " +
             std::to_string(space.row_two_jab(flat).twice / 2);
    };
    for (std::int64_t i = 0; i < n; ++i) {
      for (std::int64_t j = 0; j < n; ++j) {
        double v;
        if (!t.packed()) {
          v = data[static_cast<std::size_t>(off + i + j * n)];
        } else if (i >= j) {
          v = i == j && h < 0 ? 0.0
                              : data[static_cast<std::size_t>(
                                    off + ThreeBody::PackedIndex(n, i, j))];
        } else {
          v = h * data[static_cast<std::size_t>(
                      off + ThreeBody::PackedIndex(n, j, i))];
        }
        if (v == 0.0) continue;
        body << label(i) << " " << label(j) << " " << kc.two_j.twice << " "
             << Fmt17(v) << "\n";
        ++count;
      }
    }
  }
  out << "three_body " << count << "\n" << body.str();
}

}  // namespace

OperatorDumpHeader ReadOperatorDumpHeader(const std::filesystem::path& path) {
  std::ifstream in(path);
  const std::string name = path.string();
  if (!in) throw InputError(name + ":0: cannot open file");
  LineReader r(in, name);
  return ParseHeader(r).header;
}

std::shared_ptr<const basis::ModelSpace> ModelSpaceForDump(
    const OperatorDumpHeader& header) {
  auto b = basis::SingleParticleBasis::HarmonicOscillator(
      basis::BasisTruncation{.emax = header.emax, .lmax = header.lmax},
      header.hw);
  std::vector<double> occ(static_cast<std::size_t>(b.num_orbits()), 0.0);
  for (const auto& d : header.orbits) {
    const auto k = b.find(d.orbit);
    if (!k) {
      throw InputError(
          "dump orbit " + basis::OrbitLabel(d.orbit) +
          " is not in the harmonic-oscillator basis of the dump header");
    }
    occ[k->ToSize()] = d.occupation;
  }
  std::vector<basis::OrbitClass> classes(occ.size());
  for (std::size_t i = 0; i < occ.size(); ++i) {
    classes[i] = occ[i] >= basis::kOccupationCut ? basis::OrbitClass::kCore
                                                 : basis::OrbitClass::kQSpace;
  }
  auto ref =
      basis::Reference::FromOccupations(b, std::move(occ), header.reference);
  auto part = basis::Partition::FromClasses(b, std::move(classes));
  return basis::ModelSpace::Create(std::move(b), std::move(ref),
                                   std::move(part));
}

Operator ReadOperatorDump(std::istream& in, std::string_view name,
                          std::shared_ptr<const basis::ModelSpace> ms,
                          const OperatorDumpReadOptions& options) {
  IMSRG_CHECK(ms != nullptr);
  LineReader r(in, name);
  const ParsedHeader ph = ParseHeader(r);
  const OperatorDumpHeader& hd = ph.header;

  const double hw_ms = ms->basis().hbar_omega();
  if (std::abs(hd.hw - hw_ms) >
      1.0e-12 * std::max(std::abs(hd.hw), std::abs(hw_ms))) {
    FailAt(name, ph.basis_line,
           "hw " + Fmt17(hd.hw) + " differs from the model space's " +
               Fmt17(hw_ms));
  }
  const std::int32_t norb = static_cast<std::int32_t>(hd.orbits.size());
  std::vector<OrbitIndex> map(hd.orbits.size());
  std::vector<bool> mapped(static_cast<std::size_t>(ms->num_orbits()), false);
  for (std::int32_t i = 0; i < norb; ++i) {
    const auto& d = hd.orbits[static_cast<std::size_t>(i)];
    const int line = ph.orbit_lines[static_cast<std::size_t>(i)];
    const auto k = ms->basis().find(d.orbit);
    if (!k)
      FailAt(name, line,
             "orbit " + basis::OrbitLabel(d.orbit) + " not in the model space");
    if (options.check_occupations &&
        std::abs(ms->reference().occupation(*k) - d.occupation) >
            options.occupation_tolerance) {
      FailAt(name, line,
             "occupation " + Fmt17(d.occupation) + " of orbit " +
                 basis::OrbitLabel(d.orbit) +
                 " differs from the model-space reference");
    }
    if (mapped[k->ToSize()]) {
      FailAt(name, line,
             "orbit " + basis::OrbitLabel(d.orbit) + " is listed twice");
    }
    mapped[k->ToSize()] = true;
    map[static_cast<std::size_t>(i)] = *k;
  }

  // NotImplementedError propagates; an InputError gets the spec line.
  std::optional<Operator> opt;
  try {
    opt.emplace(ms, hd.spec);
  } catch (const InputError& e) {
    FailAt(name, ph.spec_line, e.what());
  }
  Operator& op = *opt;

  {
    auto f = r.Expect("zero_body");
    op.zero_body() = f.NextDouble("zero_body");
    f.ExpectEnd();
  }
  std::int32_t n1 = 0;
  {
    auto f = r.Expect("one_body");
    n1 = ReadCount(f, "count");
    f.ExpectEnd();
  }
  for (std::int32_t i = 0; i < n1; ++i) {
    auto f = r.data("one_body");
    const auto a = f.NextInt("a");
    const auto b = f.NextInt("b");
    const double v = f.NextDouble("value");
    f.ExpectEnd();
    if (a < 0 || b < 0 || a >= norb || b >= norb)
      f.Fail("orbit index out of range");
    {
      // A scalar one-body operator only connects orbits with equal (l, 2j,
      // tz2); the norms and the normal ordering ignore other entries, so they
      // must not be accepted silently.
      const auto& oa = hd.orbits[static_cast<std::size_t>(a)].orbit;
      const auto& ob = hd.orbits[static_cast<std::size_t>(b)].orbit;
      if (hd.spec.is_scalar() &&
          (oa.l != ob.l || oa.j2 != ob.j2 || oa.tz2 != ob.tz2)) {
        f.Fail(
            "one-body entry between orbits of different (l, 2j, tz2) in a "
            "scalar operator");
      }
    }
    op.one_body().set_element(map[static_cast<std::size_t>(a)],
                              map[static_cast<std::size_t>(b)], v);
  }
  std::int32_t n2 = 0;
  {
    auto f = r.Expect("two_body");
    n2 = ReadCount(f, "count");
    f.ExpectEnd();
  }
  const auto& tbc = ms->two_body();
  for (std::int32_t i = 0; i < n2; ++i) {
    auto f = r.data("two_body");
    std::int32_t x[4];
    const char* const labels[4] = {"p", "q", "r", "s"};
    for (int k = 0; k < 4; ++k) x[k] = f.NextInt(labels[k]);
    const auto jb = f.NextInt("J_bra");
    const auto jk = f.NextInt("J_ket");
    const double v = f.NextDouble("value");
    f.ExpectEnd();
    for (int k = 0; k < 4; ++k) {
      if (x[k] < 0 || x[k] >= norb) f.Fail("orbit index out of range");
    }
    if (jb < 0 || jk < 0) f.Fail("negative J");
    if (x[0] > x[1] || x[2] > x[3])
      f.Fail("two-body entries need p <= q and r <= s");
    if (!op.two_body().allocated())
      f.Fail("two-body entry in an operator with particle_rank 1");
    const auto m = [&](int k) { return map[static_cast<std::size_t>(x[k])]; };
    const Side bra = LocateSide(tbc, m(0), m(1), TwoJ::FromInteger(jb));
    const Side ket = LocateSide(tbc, m(2), m(3), TwoJ::FromInteger(jk));
    if (!bra.idx) f.Fail("bra ket |p q; J> is not present in its channel");
    if (!ket.idx) f.Fail("ket |r s; J> is not present in its channel");
    if (bra.ch != ket.ch) {
      f.Fail(
          "bra and ket are in different channels (not allowed for a scalar "
          "operator)");
    }
    op.two_body().set(*bra.ch, *ket.ch, *bra.idx, *ket.idx,
                      bra.phase * ket.phase * v);
  }
  ThreeBodyRecords upper;  // upper-triangle records, for the symmetry check
  {
    auto f = r.Expect("three_body");
    const std::string w = f.NextToken("count");
    f.ExpectEnd();
    std::int32_t n3 = 0;
    if (w != "none") {
      Fields cf(w, name, f.lineno());
      n3 = ReadCount(cf, "count");
    }
    if (n3 > 0 && !op.three_body()) {
      f.Fail("three_body entries in an operator with particle_rank " +
             std::to_string(hd.spec.particle_rank) + " (need 3)");
    }
    if (op.three_body()) upper.Init(*op.three_body());
    for (std::int32_t i = 0; i < n3; ++i) {
      auto g = r.data("three_body");
      ReadThreeBodyRecord(g, *ms, hd, map, *op.three_body(), upper);
    }
  }
  {
    auto f = r.Expect("end");
    f.ExpectEnd();
  }
  op.SyncToDevice();
  const std::vector<double> three_body_values =
      op.three_body() ? std::as_const(op).three_body()->buffer().ToVector()
                      : std::vector<double>();

  // Hermiticity check, tolerance 1e-12 * max|V| (design/15).
  if (hd.spec.hermiticity != Hermiticity::kNonHermitian) {
    const double sgn =
        hd.spec.hermiticity == Hermiticity::kHermitian ? 1.0 : -1.0;
    double vmax = 0.0;
    double dev = 0.0;
    const auto ob = std::as_const(op).one_body().buffer().ToVector();
    AccumulateSymmetry(ob.data(), norb == 0 ? 0 : ms->num_orbits(), sgn, vmax,
                       dev);
    if (op.two_body().allocated()) {
      const auto& mat = std::as_const(op).two_body().matrix();
      const auto tb = mat.buffer().ToVector();
      for (std::size_t k = 0; k < mat.num_blocks(); ++k) {
        const auto& info = mat.block_info(k);
        AccumulateSymmetry(tb.data() + info.offset, info.rows, sgn, vmax, dev);
      }
    }
    if (op.three_body()) {
      upper.Accumulate(*op.three_body(), three_body_values, vmax, dev);
    }
    if (dev > 1.0e-12 * std::max(vmax, 1.0e-300)) {
      FailAt(name, r.lineno(),
             "hermiticity check failed: max deviation " + Fmt17(dev) +
                 " for max|V| " + Fmt17(vmax));
    }
  }
  return std::move(op);
}

Operator ReadOperatorDump(const std::filesystem::path& path,
                          std::shared_ptr<const basis::ModelSpace> ms,
                          const OperatorDumpReadOptions& options) {
  std::ifstream in(path);
  const std::string name = path.string();
  if (!in) throw InputError(name + ":0: cannot open file");
  return ReadOperatorDump(in, name, std::move(ms), options);
}

void WriteOperatorDump(std::ostream& out, const Operator& op,
                       const OperatorDumpWriteOptions& options) {
  const basis::ModelSpace& ms = op.model_space();
  const auto& b = ms.basis();
  const std::string& ref_name =
      options.reference.empty() ? ms.reference().name() : options.reference;
  if (options.source.find('\n') != std::string::npos) {
    throw InputError(
        "OperatorDump writer: the source text must not contain a newline");
  }
  if (ref_name.empty() ||
      ref_name.find_first_of(" \t\n") != std::string::npos) {
    throw InputError(
        "OperatorDump writer: the reference name must be one non-empty word, "
        "got '" +
        ref_name + "'");
  }
  out << kVersionLine << "\n";
  out << "source " << options.source << "\n";
  out << "basis emax " << b.emax() << " lmax " << b.lmax() << " hw "
      << Fmt17(b.hbar_omega()) << "\n";
  out << "reference " << ref_name << "\n";
  const std::int32_t norb = ms.num_orbits();
  out << "norbits " << norb << "\n";
  for (std::int32_t i = 0; i < norb; ++i) {
    const basis::Orbit& o = b.orbit(OrbitIndex{i});
    out << "orbit " << i << " " << o.n << " " << o.l << " " << o.j2.twice << " "
        << o.tz2.twice << " " << Fmt17(ms.reference().occupation(OrbitIndex{i}))
        << "\n";
  }
  const OperatorSpec& s = op.spec();
  out << "spec particle_rank " << s.particle_rank << " rank_J " << s.rank_J
      << " parity " << static_cast<int>(s.parity) << " delta_Tz " << s.delta_Tz
      << " hermiticity " << HermiticityCode(s.hermiticity) << "\n";
  out << "zero_body " << Fmt17(op.zero_body()) << "\n";

  const auto ob = op.one_body().buffer().ToVector();
  std::int64_t n1 = 0;
  for (const double v : ob) n1 += (v != 0.0);
  out << "one_body " << n1 << "\n";
  for (std::int32_t a = 0; a < norb; ++a) {
    for (std::int32_t c = 0; c < norb; ++c) {
      const double v =
          ob[static_cast<std::size_t>(a) +
             static_cast<std::size_t>(c) * static_cast<std::size_t>(norb)];
      if (v != 0.0) out << a << " " << c << " " << Fmt17(v) << "\n";
    }
  }

  const auto& tbc = ms.two_body();
  std::vector<double> tb;
  const bool has2 = op.two_body().allocated();
  if (has2) tb = op.two_body().matrix().buffer().ToVector();
  std::int64_t n2 = 0;
  for (const double v : tb) n2 += (v != 0.0);
  out << "two_body " << n2 << "\n";
  if (has2) {
    for (std::int32_t ch = 0; ch < tbc.num_channels(); ++ch) {
      const auto k =
          op.two_body().block_index(ChannelIndex{ch}, ChannelIndex{ch});
      if (!k) continue;
      const auto& info = op.two_body().matrix().block_info(*k);
      const auto kets = tbc.channel_kets(ChannelIndex{ch});
      const std::int32_t J = tbc.channel(ChannelIndex{ch}).J();
      for (std::int64_t i = 0; i < info.rows; ++i) {
        const auto& bk = tbc.ket(kets[static_cast<std::size_t>(i)]);
        for (std::int64_t j = 0; j < info.cols; ++j) {
          const double v =
              tb[static_cast<std::size_t>(info.offset + i + j * info.rows)];
          if (v == 0.0) continue;
          const auto& kk = tbc.ket(kets[static_cast<std::size_t>(j)]);
          out << bk.p.value() << " " << bk.q.value() << " " << kk.p.value()
              << " " << kk.q.value() << " " << J << " " << J << " " << Fmt17(v)
              << "\n";
        }
      }
    }
  }
  if (op.three_body()) {
    WriteThreeBody(out, *op.three_body());
  } else {
    out << "three_body none\n";
  }
  out << "end\n";
}

void WriteOperatorDump(const std::filesystem::path& path, const Operator& op,
                       const OperatorDumpWriteOptions& options) {
  std::ofstream out(path);
  if (!out) throw InputError(path.string() + ":0: cannot write file");
  WriteOperatorDump(out, op, options);
  out.flush();
  if (!out) throw InputError(path.string() + ":0: write error");
}

}  // namespace imsrg::op
