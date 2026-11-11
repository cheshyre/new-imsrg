#include "imsrg/operator/operator_dump.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <istream>
#include <optional>
#include <ostream>
#include <utility>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/error.hpp"
#include "imsrg/basis/orbit.hpp"
#include "imsrg/basis/partition.hpp"
#include "imsrg/basis/reference.hpp"
#include "imsrg/basis/single_particle_basis.hpp"
#include "imsrg/basis/two_body_channels.hpp"

namespace imsrg::op {

namespace {

constexpr std::string_view kVersionLine = "# imsrg-operator-dump v1";

/// `%.17g` formatting of a double (round-trips exactly).
std::string fmt17(double x) {
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.17g", x);
  return buf;
}

[[noreturn]] void fail_at(std::string_view name, int line, const std::string& msg) {
  throw InputError(std::string(name) + ":" + std::to_string(line) + ": " + msg);
}

/// Whitespace-separated fields of one record line.
class Fields {
 public:
  Fields(std::string_view line, std::string_view name, int lineno) : name_(name), lineno_(lineno) {
    std::size_t i = 0;
    while (i < line.size()) {
      while (i < line.size() && (line[i] == ' ' || line[i] == '\t' || line[i] == '\r')) ++i;
      std::size_t j = i;
      while (j < line.size() && line[j] != ' ' && line[j] != '\t' && line[j] != '\r') ++j;
      if (j > i) tokens_.emplace_back(line.substr(i, j - i));
      i = j;
    }
  }
  [[noreturn]] void fail(const std::string& msg) const { fail_at(name_, lineno_, msg); }
  bool empty() const { return pos_ >= tokens_.size(); }
  std::string next_token(const char* what) {
    if (empty()) fail(std::string("missing field ") + what);
    return tokens_[pos_++];
  }
  std::int32_t next_int(const char* what) {
    const std::string tok = next_token(what);
    std::int32_t v = 0;
    const auto [ptr, ec] = std::from_chars(tok.data(), tok.data() + tok.size(), v);
    if (ec != std::errc() || ptr != tok.data() + tok.size()) {
      fail(std::string("cannot parse ") + what + " '" + tok + "'");
    }
    return v;
  }
  double next_double(const char* what) {
    const std::string tok = next_token(what);
    char* end = nullptr;
    const double v = std::strtod(tok.c_str(), &end);
    if (end == tok.c_str() || *end != '\0') {
      fail(std::string("cannot parse ") + what + " '" + tok + "'");
    }
    return v;
  }
  void expect_key(const std::string& key) {
    const std::string tok = next_token(key.c_str());
    if (tok != key) fail("expected key '" + key + "', got '" + tok + "'");
  }
  void expect_end() const {
    if (!empty()) fail("unexpected trailing field '" + tokens_[pos_] + "'");
  }
  int lineno() const { return lineno_; }

 private:
  std::string_view name_;
  int lineno_;
  std::vector<std::string> tokens_;
  std::size_t pos_ = 0;
};

/// Line reader that skips comments (except line 1) and blank lines and tracks line numbers.
class LineReader {
 public:
  LineReader(std::istream& in, std::string_view name) : in_(in), name_(name) {}

  [[noreturn]] void fail(const std::string& msg) const { fail_at(name_, lineno_ + 1, msg); }
  /// Next significant line; false at end of file.
  bool next(std::string& line) {
    while (std::getline(in_, line)) {
      ++lineno_;
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (lineno_ > 1 && (line.empty() || line[0] == '#')) continue;
      return true;
    }
    return false;
  }
  /// Next record, which must start with `keyword`; the keyword is consumed.
  Fields expect(const std::string& keyword) {
    std::string line;
    if (!next(line)) fail("unexpected end of file, expected '" + keyword + "'");
    Fields f(line, name_, lineno_);
    const std::string kw = f.empty() ? std::string() : f.next_token("keyword");
    if (kw != keyword) f.fail("expected '" + keyword + "', got '" + kw + "'");
    return f;
  }
  /// A data line inside `section`.
  Fields data(const std::string& section) {
    std::string line;
    if (!next(line)) fail("unexpected end of file in " + section);
    return Fields(line, name_, lineno_);
  }
  void check_version() {
    std::string line;
    if (!next(line)) fail("empty file, expected '" + std::string(kVersionLine) + "'");
    if (line != kVersionLine) {
      fail_at(name_, lineno_,
              "missing or unsupported version line (expected '" + std::string(kVersionLine) + "')");
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

std::int32_t read_count(Fields& f, const char* what) {
  const auto n = f.next_int(what);
  if (n < 0) f.fail(std::string("negative ") + what);
  return n;
}

ParsedHeader parse_header(LineReader& r) {
  ParsedHeader h;
  OperatorDumpHeader& hd = h.header;
  r.check_version();
  {
    std::string line;
    if (!r.next(line)) r.fail("unexpected end of file, expected 'source'");
    if (line != "source" && line.rfind("source ", 0) != 0) {
      r.fail("expected 'source', got '" + line.substr(0, line.find(' ')) + "'");
    }
    hd.source = line.size() > 7 ? line.substr(7) : "";
  }
  {
    auto f = r.expect("basis");
    h.basis_line = f.lineno();
    f.expect_key("emax");
    hd.emax = f.next_int("emax");
    f.expect_key("lmax");
    hd.lmax = f.next_int("lmax");
    f.expect_key("hw");
    hd.hw = f.next_double("hw");
    f.expect_end();
  }
  {
    auto f = r.expect("reference");
    hd.reference = f.next_token("reference");
    f.expect_end();
  }
  std::int32_t norb = 0;
  {
    auto f = r.expect("norbits");
    norb = read_count(f, "norbits");
    f.expect_end();
  }
  for (std::int32_t i = 0; i < norb; ++i) {
    auto f = r.expect("orbit");
    const auto idx = f.next_int("index");
    basis::Orbit o;
    o.n = f.next_int("n");
    o.l = f.next_int("l");
    o.j2 = TwoJ{f.next_int("2j")};
    o.tz2 = TwoTz{f.next_int("2tz")};
    const double occ = f.next_double("occupation");
    f.expect_end();
    if (idx != i) f.fail("orbit indices must be 0..N-1 in order");
    if (!basis::is_valid(o)) f.fail("invalid orbit quantum numbers");
    hd.orbits.push_back({o, occ});
    h.orbit_lines.push_back(f.lineno());
  }
  {
    auto f = r.expect("spec");
    h.spec_line = f.lineno();
    f.expect_key("particle_rank");
    hd.spec.particle_rank = f.next_int("particle_rank");
    f.expect_key("rank_J");
    hd.spec.rank_J = f.next_int("rank_J");
    f.expect_key("parity");
    const auto par = f.next_int("parity");
    if (par != 0 && par != 1) f.fail("parity must be 0 or 1");
    hd.spec.parity = par == 0 ? Parity::Even : Parity::Odd;
    f.expect_key("delta_Tz");
    hd.spec.delta_Tz = f.next_int("delta_Tz");
    f.expect_key("hermiticity");
    const std::string hc = f.next_token("hermiticity");
    f.expect_end();
    if (hc.size() != 1 || (hc[0] != 'H' && hc[0] != 'A' && hc[0] != 'N')) {
      f.fail("bad hermiticity '" + hc + "' (expected H, A or N)");
    }
    hd.spec.hermiticity = hermiticity_from_code(hc[0]);
  }
  return h;
}

/// A located ket side: channel label, channel index, local index and exchange phase.
struct Side {
  TwoJ two_j{0};
  Parity parity = Parity::Even;
  TwoTz two_tz{0};
  std::optional<ChannelIndex> ch;
  std::optional<LocalKetIndex> idx;
  int phase = 1;
};

Side locate_side(const basis::TwoBodyChannels& tb, OrbitIndex a, OrbitIndex b, TwoJ two_j) {
  const auto orbits = tb.orbits();
  const basis::Orbit& oa = orbits[a.to_size()];
  const basis::Orbit& ob = orbits[b.to_size()];
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
  if (swap) s.phase = basis::exchange_phase(oa.j2, ob.j2, two_j);
  return s;
}

/// Accumulates max|m| and max|m - sgn m^T| of a column-major n x n block at `data`.
void accumulate_symmetry(const double* data, std::int64_t n, double sgn, double& vmax,
                         double& dev) {
  for (std::int64_t j = 0; j < n; ++j) {
    for (std::int64_t i = 0; i < n; ++i) {
      const double v = data[i + j * n];
      vmax = std::max(vmax, std::abs(v));
      dev = std::max(dev, std::abs(v - sgn * data[j + i * n]));
    }
  }
}

}  // namespace

OperatorDumpHeader read_operator_dump_header(const std::filesystem::path& path) {
  std::ifstream in(path);
  const std::string name = path.string();
  if (!in) throw InputError(name + ":0: cannot open file");
  LineReader r(in, name);
  return parse_header(r).header;
}

std::shared_ptr<const basis::ModelSpace> model_space_for_dump(const OperatorDumpHeader& header) {
  auto b = basis::SingleParticleBasis::harmonic_oscillator(
      basis::BasisTruncation{.emax = header.emax, .lmax = header.lmax}, header.hw);
  std::vector<double> occ(static_cast<std::size_t>(b.num_orbits()), 0.0);
  for (const auto& d : header.orbits) {
    const auto k = b.find(d.orbit);
    if (!k) {
      throw InputError("dump orbit " + basis::orbit_label(d.orbit) +
                       " is not in the harmonic-oscillator basis of the dump header");
    }
    occ[k->to_size()] = d.occupation;
  }
  std::vector<basis::OrbitClass> classes(occ.size());
  for (std::size_t i = 0; i < occ.size(); ++i) {
    classes[i] =
        occ[i] >= basis::kOccupationCut ? basis::OrbitClass::Core : basis::OrbitClass::QSpace;
  }
  auto ref = basis::Reference::from_occupations(b, std::move(occ), header.reference);
  auto part = basis::Partition::from_classes(b, std::move(classes));
  return basis::ModelSpace::create(std::move(b), std::move(ref), std::move(part));
}

Operator read_operator_dump(std::istream& in, std::string_view name,
                            std::shared_ptr<const basis::ModelSpace> ms,
                            const OperatorDumpReadOptions& options) {
  IMSRG_CHECK(ms != nullptr);
  LineReader r(in, name);
  const ParsedHeader ph = parse_header(r);
  const OperatorDumpHeader& hd = ph.header;

  const double hw_ms = ms->basis().hbar_omega();
  if (std::abs(hd.hw - hw_ms) > 1e-12 * std::max(std::abs(hd.hw), std::abs(hw_ms))) {
    fail_at(name, ph.basis_line,
            "hw " + fmt17(hd.hw) + " differs from the model space's " + fmt17(hw_ms));
  }
  const std::int32_t norb = static_cast<std::int32_t>(hd.orbits.size());
  std::vector<OrbitIndex> map(hd.orbits.size());
  std::vector<bool> mapped(static_cast<std::size_t>(ms->num_orbits()), false);
  for (std::int32_t i = 0; i < norb; ++i) {
    const auto& d = hd.orbits[static_cast<std::size_t>(i)];
    const int line = ph.orbit_lines[static_cast<std::size_t>(i)];
    const auto k = ms->basis().find(d.orbit);
    if (!k) fail_at(name, line, "orbit " + basis::orbit_label(d.orbit) + " not in the model space");
    if (options.check_occupations &&
        std::abs(ms->reference().occupation(*k) - d.occupation) > options.occupation_tolerance) {
      fail_at(name, line,
              "occupation " + fmt17(d.occupation) + " of orbit " + basis::orbit_label(d.orbit) +
                  " differs from the model-space reference");
    }
    if (mapped[k->to_size()]) {
      fail_at(name, line, "orbit " + basis::orbit_label(d.orbit) + " is listed twice");
    }
    mapped[k->to_size()] = true;
    map[static_cast<std::size_t>(i)] = *k;
  }

  // NotImplementedError propagates; an InputError gets the spec line.
  std::optional<Operator> opt;
  try {
    opt.emplace(ms, hd.spec);
  } catch (const InputError& e) {
    fail_at(name, ph.spec_line, e.what());
  }
  Operator& op = *opt;

  {
    auto f = r.expect("zero_body");
    op.zero_body() = f.next_double("zero_body");
    f.expect_end();
  }
  std::int32_t n1 = 0;
  {
    auto f = r.expect("one_body");
    n1 = read_count(f, "count");
    f.expect_end();
  }
  for (std::int32_t i = 0; i < n1; ++i) {
    auto f = r.data("one_body");
    const auto a = f.next_int("a");
    const auto b = f.next_int("b");
    const double v = f.next_double("value");
    f.expect_end();
    if (a < 0 || b < 0 || a >= norb || b >= norb) f.fail("orbit index out of range");
    {
      // A scalar one-body operator only connects orbits with equal (l, 2j, tz2); the norms and
      // the normal ordering ignore other entries, so they must not be accepted silently.
      const auto& oa = hd.orbits[static_cast<std::size_t>(a)].orbit;
      const auto& ob = hd.orbits[static_cast<std::size_t>(b)].orbit;
      if (hd.spec.is_scalar() && (oa.l != ob.l || oa.j2 != ob.j2 || oa.tz2 != ob.tz2)) {
        f.fail("one-body entry between orbits of different (l, 2j, tz2) in a scalar operator");
      }
    }
    op.one_body().set_element(map[static_cast<std::size_t>(a)], map[static_cast<std::size_t>(b)],
                              v);
  }
  std::int32_t n2 = 0;
  {
    auto f = r.expect("two_body");
    n2 = read_count(f, "count");
    f.expect_end();
  }
  const auto& tbc = ms->two_body();
  for (std::int32_t i = 0; i < n2; ++i) {
    auto f = r.data("two_body");
    std::int32_t x[4];
    const char* const labels[4] = {"p", "q", "r", "s"};
    for (int k = 0; k < 4; ++k) x[k] = f.next_int(labels[k]);
    const auto jb = f.next_int("J_bra");
    const auto jk = f.next_int("J_ket");
    const double v = f.next_double("value");
    f.expect_end();
    for (int k = 0; k < 4; ++k) {
      if (x[k] < 0 || x[k] >= norb) f.fail("orbit index out of range");
    }
    if (jb < 0 || jk < 0) f.fail("negative J");
    if (x[0] > x[1] || x[2] > x[3]) f.fail("two-body entries need p <= q and r <= s");
    if (!op.two_body().allocated()) f.fail("two-body entry in an operator with particle_rank 1");
    const auto m = [&](int k) { return map[static_cast<std::size_t>(x[k])]; };
    const Side bra = locate_side(tbc, m(0), m(1), TwoJ::from_integer(jb));
    const Side ket = locate_side(tbc, m(2), m(3), TwoJ::from_integer(jk));
    if (!bra.idx) f.fail("bra ket |p q; J> is not present in its channel");
    if (!ket.idx) f.fail("ket |r s; J> is not present in its channel");
    if (bra.ch != ket.ch) {
      f.fail("bra and ket are in different channels (not allowed for a scalar operator)");
    }
    op.two_body().set(*bra.ch, *ket.ch, *bra.idx, *ket.idx, bra.phase * ket.phase * v);
  }
  {
    auto f = r.expect("three_body");
    const std::string w = f.next_token("count");
    f.expect_end();
    if (w != "none" && w != "0") {
      throw NotImplementedError(std::string(name) + ":" + std::to_string(f.lineno()) +
                                ": three_body entries are not supported yet");
    }
  }
  {
    auto f = r.expect("end");
    f.expect_end();
  }
  op.sync_to_device();

  // Hermiticity check, tolerance 1e-12 * max|V| (design/15).
  if (hd.spec.hermiticity != Hermiticity::NonHermitian) {
    const double sgn = hd.spec.hermiticity == Hermiticity::Hermitian ? 1.0 : -1.0;
    double vmax = 0.0;
    double dev = 0.0;
    const auto ob = std::as_const(op).one_body().buffer().to_vector();
    accumulate_symmetry(ob.data(), norb == 0 ? 0 : ms->num_orbits(), sgn, vmax, dev);
    if (op.two_body().allocated()) {
      const auto& mat = std::as_const(op).two_body().matrix();
      const auto tb = mat.buffer().to_vector();
      for (std::size_t k = 0; k < mat.num_blocks(); ++k) {
        const auto& info = mat.block_info(k);
        accumulate_symmetry(tb.data() + info.offset, info.rows, sgn, vmax, dev);
      }
    }
    if (dev > 1e-12 * std::max(vmax, 1e-300)) {
      fail_at(
          name, r.lineno(),
          "hermiticity check failed: max deviation " + fmt17(dev) + " for max|V| " + fmt17(vmax));
    }
  }
  return std::move(op);
}

Operator read_operator_dump(const std::filesystem::path& path,
                            std::shared_ptr<const basis::ModelSpace> ms,
                            const OperatorDumpReadOptions& options) {
  std::ifstream in(path);
  const std::string name = path.string();
  if (!in) throw InputError(name + ":0: cannot open file");
  return read_operator_dump(in, name, std::move(ms), options);
}

void write_operator_dump(std::ostream& out, const Operator& op,
                         const OperatorDumpWriteOptions& options) {
  const basis::ModelSpace& ms = op.model_space();
  const auto& b = ms.basis();
  const std::string& ref_name =
      options.reference.empty() ? ms.reference().name() : options.reference;
  if (options.source.find('\n') != std::string::npos) {
    throw InputError("OperatorDump writer: the source text must not contain a newline");
  }
  if (ref_name.empty() || ref_name.find_first_of(" \t\n") != std::string::npos) {
    throw InputError("OperatorDump writer: the reference name must be one non-empty word, got '" +
                     ref_name + "'");
  }
  out << kVersionLine << "\n";
  out << "source " << options.source << "\n";
  out << "basis emax " << b.emax() << " lmax " << b.lmax() << " hw " << fmt17(b.hbar_omega())
      << "\n";
  out << "reference " << ref_name << "\n";
  const std::int32_t norb = ms.num_orbits();
  out << "norbits " << norb << "\n";
  for (std::int32_t i = 0; i < norb; ++i) {
    const basis::Orbit& o = b.orbit(OrbitIndex{i});
    out << "orbit " << i << " " << o.n << " " << o.l << " " << o.j2.twice << " " << o.tz2.twice
        << " " << fmt17(ms.reference().occupation(OrbitIndex{i})) << "\n";
  }
  const OperatorSpec& s = op.spec();
  out << "spec particle_rank " << s.particle_rank << " rank_J " << s.rank_J << " parity "
      << static_cast<int>(s.parity) << " delta_Tz " << s.delta_Tz << " hermiticity "
      << hermiticity_code(s.hermiticity) << "\n";
  out << "zero_body " << fmt17(op.zero_body()) << "\n";

  const auto ob = op.one_body().buffer().to_vector();
  std::int64_t n1 = 0;
  for (const double v : ob) n1 += (v != 0.0);
  out << "one_body " << n1 << "\n";
  for (std::int32_t a = 0; a < norb; ++a) {
    for (std::int32_t c = 0; c < norb; ++c) {
      const double v = ob[static_cast<std::size_t>(a) +
                          static_cast<std::size_t>(c) * static_cast<std::size_t>(norb)];
      if (v != 0.0) out << a << " " << c << " " << fmt17(v) << "\n";
    }
  }

  const auto& tbc = ms.two_body();
  std::vector<double> tb;
  const bool has2 = op.two_body().allocated();
  if (has2) tb = op.two_body().matrix().buffer().to_vector();
  std::int64_t n2 = 0;
  for (const double v : tb) n2 += (v != 0.0);
  out << "two_body " << n2 << "\n";
  if (has2) {
    for (std::int32_t ch = 0; ch < tbc.num_channels(); ++ch) {
      const auto k = op.two_body().block_index(ChannelIndex{ch}, ChannelIndex{ch});
      if (!k) continue;
      const auto& info = op.two_body().matrix().block_info(*k);
      const auto kets = tbc.channel_kets(ChannelIndex{ch});
      const std::int32_t J = tbc.channel(ChannelIndex{ch}).J();
      for (std::int64_t i = 0; i < info.rows; ++i) {
        const auto& bk = tbc.ket(kets[static_cast<std::size_t>(i)]);
        for (std::int64_t j = 0; j < info.cols; ++j) {
          const double v = tb[static_cast<std::size_t>(info.offset + i + j * info.rows)];
          if (v == 0.0) continue;
          const auto& kk = tbc.ket(kets[static_cast<std::size_t>(j)]);
          out << bk.p.value() << " " << bk.q.value() << " " << kk.p.value() << " " << kk.q.value()
              << " " << J << " " << J << " " << fmt17(v) << "\n";
        }
      }
    }
  }
  out << "three_body none\n";
  out << "end\n";
}

void write_operator_dump(const std::filesystem::path& path, const Operator& op,
                         const OperatorDumpWriteOptions& options) {
  std::ofstream out(path);
  if (!out) throw InputError(path.string() + ":0: cannot write file");
  write_operator_dump(out, op, options);
  out.flush();
  if (!out) throw InputError(path.string() + ":0: write error");
}

}  // namespace imsrg::op
