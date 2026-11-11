// rh_opdump.cpp -- OperatorDump v1 writer/reader (design/15-operator-dump-format.md) for the
// old imsrg++ Operator, plus the `roundtrip` and `norms` subcommands.
//
//   refharness roundtrip IN.dump OUT.dump [valence_space=...]  read and rewrite (must be identical)
//   refharness norms IN.dump [out=norms.json]                   imsrg++ norms of a dump

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include "IMSRG.hh"
#include "rh_common.hh"

namespace rh {

namespace {

char herm_char(const Operator& op) {
  if (op.IsHermitian()) return 'H';
  if (op.IsAntiHermitian()) return 'A';
  return 'N';
}

struct TBEntry {
  int p, q, r, s, jb, jk;
  double v;
};

// All nonzero two-body entries <pq;Jb|V|rs;Jk> (normalized), p<=q, r<=s, both (pq,rs) and
// (rs,pq). Stored blocks are (ch_bra <= ch_ket); the transposed blocks of non-scalar operators
// are generated through GetTBME_norm, which applies the operator's hermiticity.
std::vector<TBEntry> two_body_entries(const Operator& op) {
  std::vector<TBEntry> out;
  ModelSpace* ms = op.GetModelSpace();
  for (auto& it : op.TwoBody.MatEl) {
    size_t chb = it.first[0], chk = it.first[1];
    const arma::mat& M = it.second;
    TwoBodyChannel& tb = ms->GetTwoBodyChannel(chb);
    TwoBodyChannel& tk = ms->GetTwoBodyChannel(chk);
    for (size_t i = 0; i < tb.GetNumberKets(); ++i) {
      Ket& bra = tb.GetKet(i);
      for (size_t j = 0; j < tk.GetNumberKets(); ++j) {
        Ket& ket = tk.GetKet(j);
        double v = M(i, j);
        if (v != 0.0)
          out.push_back({(int)bra.p, (int)bra.q, (int)ket.p, (int)ket.q, tb.J, tk.J, v});
      }
    }
    if (chb != chk) {  // transposed block via hermiticity
      for (size_t j = 0; j < tk.GetNumberKets(); ++j) {
        Ket& ket = tk.GetKet(j);
        for (size_t i = 0; i < tb.GetNumberKets(); ++i) {
          Ket& bra = tb.GetKet(i);
          double v = op.TwoBody.GetTBME_norm(chk, chb, ket.p, ket.q, bra.p, bra.q);
          if (v != 0.0)
            out.push_back({(int)ket.p, (int)ket.q, (int)bra.p, (int)bra.q, tk.J, tb.J, v});
        }
      }
    }
  }
  return out;
}

[[noreturn]] void fail(const std::string& path, int line, const std::string& msg) {
  std::ostringstream os;
  os << path << ":" << line << ": " << msg;
  throw std::runtime_error(os.str());
}

// Line reader that skips comments (except line 1) and blank lines, tracking line numbers.
struct LineReader {
  std::ifstream f;
  std::string path;
  int lineno = 0;
  explicit LineReader(const std::string& p) : f(p), path(p) {
    if (!f) throw std::runtime_error("cannot open " + p);
  }
  bool next(std::string& line) {
    while (std::getline(f, line)) {
      ++lineno;
      if (lineno > 1 && (line.empty() || line[0] == '#')) continue;
      return true;
    }
    return false;
  }
  std::istringstream expect(const std::string& keyword) {
    std::string line;
    if (!next(line)) fail(path, lineno, "unexpected end of file, expected '" + keyword + "'");
    std::istringstream is(line);
    std::string kw;
    is >> kw;
    if (kw != keyword) fail(path, lineno, "expected '" + keyword + "', got '" + kw + "'");
    return is;
  }
};

template <class T>
void get(std::istringstream& is, T& v, LineReader& r, const char* what) {
  if (!(is >> v)) fail(r.path, r.lineno, std::string("cannot parse ") + what);
}

void get_key(std::istringstream& is, const std::string& key, LineReader& r) {
  std::string k;
  if (!(is >> k) || k != key) fail(r.path, r.lineno, "expected key '" + key + "'");
}

}  // namespace

void write_opdump(const std::string& path, const Operator& op, const DumpMeta& meta, bool with_3b) {
  std::ofstream o(path);
  if (!o) throw std::runtime_error("cannot write " + path);
  ModelSpace* ms = op.GetModelSpace();
  const int lmax_eff = std::min(ms->GetLmax(), ms->GetEmax());
  o << "# imsrg-operator-dump v1\n";
  o << "source " << meta.source << "\n";
  o << "basis emax " << ms->GetEmax() << " lmax " << lmax_eff << " hw " << fmt17(ms->GetHbarOmega())
    << "\n";
  o << "reference " << meta.reference << "\n";
  o << "norbits " << ms->GetNumberOrbits() << "\n";
  for (size_t i = 0; i < ms->GetNumberOrbits(); ++i) {
    const Orbit& oi = ms->GetOrbit(i);
    o << "orbit " << i << " " << oi.n << " " << oi.l << " " << oi.j2 << " " << oi.tz2 << " "
      << fmt17(oi.occ) << "\n";
  }
  // A rank-3 operator whose 3B part is identically zero (e.g. the result of
  // Operator::DoNormalOrdering3, which allocates empty pn storage) is written as rank 2.
  int prank = op.GetParticleRank();
  if (prank >= 3 && !with_3b) {
    if (op.ThreeBodyNorm() != 0.0)
      throw std::runtime_error("operator has a nonzero 3B part; the 3B dump is not implemented");
    prank = 2;
  }
  o << "spec particle_rank " << prank << " rank_J " << op.GetJRank() << " parity "
    << op.GetParity() << " delta_Tz " << op.GetTRank() << " hermiticity " << herm_char(op) << "\n";
  o << "zero_body " << fmt17(op.ZeroBody) << "\n";

  std::vector<std::string> ob;
  for (arma::uword a = 0; a < op.OneBody.n_rows; ++a)
    for (arma::uword b = 0; b < op.OneBody.n_cols; ++b)
      if (op.OneBody(a, b) != 0.0)
        ob.push_back(std::to_string(a) + " " + std::to_string(b) + " " + fmt17(op.OneBody(a, b)));
  o << "one_body " << ob.size() << "\n";
  for (auto& s : ob) o << s << "\n";

  auto tb = two_body_entries(op);
  o << "two_body " << tb.size() << "\n";
  for (auto& e : tb)
    o << e.p << " " << e.q << " " << e.r << " " << e.s << " " << e.jb << " " << e.jk << " "
      << fmt17(e.v) << "\n";

  if (with_3b && op.GetParticleRank() >= 3)
    throw std::runtime_error("3B section of OperatorDump is not implemented yet (reserved for M10)");
  o << "three_body none\n";
  o << "end\n";
}

DumpHeader read_opdump_header(const std::string& path) {
  LineReader r(path);
  DumpHeader h;
  std::string line;
  if (!r.next(line) || line != "# imsrg-operator-dump v1")
    fail(path, r.lineno, "missing or unsupported version line");
  r.expect("source");
  {
    auto is = r.expect("basis");
    get_key(is, "emax", r); get(is, h.emax, r, "emax");
    get_key(is, "lmax", r); get(is, h.lmax, r, "lmax");
    get_key(is, "hw", r);   get(is, h.hw, r, "hw");
  }
  { auto is = r.expect("reference"); get(is, h.reference, r, "reference"); }
  int n = 0;
  { auto is = r.expect("norbits"); get(is, n, r, "norbits"); }
  for (int i = 0; i < n; ++i) r.expect("orbit");
  {
    auto is = r.expect("spec");
    std::string hc;
    get_key(is, "particle_rank", r); get(is, h.particle_rank, r, "particle_rank");
    get_key(is, "rank_J", r);        get(is, h.rank_J, r, "rank_J");
    get_key(is, "parity", r);        get(is, h.parity, r, "parity");
    get_key(is, "delta_Tz", r);      get(is, h.delta_Tz, r, "delta_Tz");
    get_key(is, "hermiticity", r);   get(is, hc, r, "hermiticity");
    if (hc != "H" && hc != "A" && hc != "N") fail(path, r.lineno, "bad hermiticity " + hc);
    h.hermiticity = hc[0];
  }
  return h;
}

Operator read_opdump(const std::string& path, ModelSpace& ms) {
  LineReader r(path);
  std::string line;
  if (!r.next(line) || line != "# imsrg-operator-dump v1")
    fail(path, r.lineno, "missing or unsupported version line");
  r.expect("source");
  r.expect("basis");
  r.expect("reference");
  int norb = 0;
  { auto is = r.expect("norbits"); get(is, norb, r, "norbits"); }
  std::vector<int> map(norb, -1);
  for (int i = 0; i < norb; ++i) {
    auto is = r.expect("orbit");
    int idx, n, l, j2, tz2;
    double occ;
    get(is, idx, r, "index"); get(is, n, r, "n"); get(is, l, r, "l"); get(is, j2, r, "2j");
    get(is, tz2, r, "2tz"); get(is, occ, r, "occupation");
    if (idx != i) fail(path, r.lineno, "orbit indices must be 0..N-1 in order");
    size_t k = ms.GetOrbitIndex(n, l, j2, tz2);
    if (k >= ms.GetNumberOrbits()) fail(path, r.lineno, "orbit not in the model space");
    if (std::abs(ms.GetOrbit(k).occ - occ) > 1e-12)
      fail(path, r.lineno, "occupation differs from the model-space reference");
    map[i] = (int)k;
  }
  int prank, rJ, par, dTz;
  std::string hc;
  {
    auto is = r.expect("spec");
    get_key(is, "particle_rank", r); get(is, prank, r, "particle_rank");
    get_key(is, "rank_J", r);        get(is, rJ, r, "rank_J");
    get_key(is, "parity", r);        get(is, par, r, "parity");
    get_key(is, "delta_Tz", r);      get(is, dTz, r, "delta_Tz");
    get_key(is, "hermiticity", r);   get(is, hc, r, "hermiticity");
  }
  if (prank > 2) fail(path, r.lineno, "particle_rank 3 is not supported by this reader yet");
  Operator op(ms, rJ, dTz, par, std::max(prank, 2));
  if (hc == "H") op.SetHermitian();
  else if (hc == "A") op.SetAntiHermitian();
  else op.SetNonHermitian();
  { auto is = r.expect("zero_body"); get(is, op.ZeroBody, r, "zero_body"); }

  int n1 = 0;
  { auto is = r.expect("one_body"); get(is, n1, r, "count"); }
  for (int i = 0; i < n1; ++i) {
    std::string l;
    if (!r.next(l)) fail(path, r.lineno, "unexpected end of one_body");
    std::istringstream is(l);
    int a, b;
    double v;
    get(is, a, r, "a"); get(is, b, r, "b"); get(is, v, r, "value");
    if (a < 0 || b < 0 || a >= norb || b >= norb) fail(path, r.lineno, "orbit index out of range");
    op.OneBody(map[a], map[b]) = v;
  }

  int n2 = 0;
  { auto is = r.expect("two_body"); get(is, n2, r, "count"); }
  for (int i = 0; i < n2; ++i) {
    std::string l;
    if (!r.next(l)) fail(path, r.lineno, "unexpected end of two_body");
    std::istringstream is(l);
    int p, q, rr, s, jb, jk;
    double v;
    get(is, p, r, "p"); get(is, q, r, "q"); get(is, rr, r, "r"); get(is, s, r, "s");
    get(is, jb, r, "J_bra"); get(is, jk, r, "J_ket"); get(is, v, r, "value");
    for (int x : {p, q, rr, s})
      if (x < 0 || x >= norb) fail(path, r.lineno, "orbit index out of range");
    if (p > q || rr > s) fail(path, r.lineno, "two-body entries need p <= q and r <= s");
    int a = map[p], b = map[q], c = map[rr], d = map[s];
    Orbit& oa = ms.GetOrbit(a); Orbit& ob = ms.GetOrbit(b);
    Orbit& oc = ms.GetOrbit(c); Orbit& od = ms.GetOrbit(d);
    size_t chb = ms.GetTwoBodyChannelIndex(jb, (oa.l + ob.l) % 2, (oa.tz2 + ob.tz2) / 2);
    size_t chk = ms.GetTwoBodyChannelIndex(jk, (oc.l + od.l) % 2, (oc.tz2 + od.tz2) / 2);
    if (chb > chk) continue;  // not stored: follows from the hermiticity of the stored block
    auto itm = op.TwoBody.MatEl.find({chb, chk});
    if (itm == op.TwoBody.MatEl.end()) fail(path, r.lineno, "channel pair not allowed by the spec");
    TwoBodyChannel& tb = ms.GetTwoBodyChannel(chb);
    TwoBodyChannel& tk = ms.GetTwoBodyChannel(chk);
    double phase = 1;
    if (a > b) { std::swap(a, b); phase *= ms.GetKet(a, b).Phase(jb); }
    if (c > d) { std::swap(c, d); phase *= ms.GetKet(c, d).Phase(jk); }
    int ib = tb.GetLocalIndex(a, b), ik = tk.GetLocalIndex(c, d);
    if (ib < 0 || ik < 0 || ib >= (int)tb.GetNumberKets() || ik >= (int)tk.GetNumberKets())
      fail(path, r.lineno, "ket not in channel");
    itm->second(ib, ik) = phase * v;
  }
  {
    auto is = r.expect("three_body");
    std::string w;
    get(is, w, r, "count");
    if (w != "none" && w != "0") fail(path, r.lineno, "three_body entries are not supported yet");
  }
  r.expect("end");

  // Hermiticity check (scalar diagonal blocks), tolerance 1e-12 * max|V| (spec 15)
  if (hc != "N") {
    double sgn = (hc == "H") ? 1.0 : -1.0;
    double vmax = 0, dev = 0;
    for (auto& it : op.TwoBody.MatEl) {
      if (it.first[0] != it.first[1] || it.second.n_elem == 0) continue;
      vmax = std::max(vmax, arma::abs(it.second).max());
      dev = std::max(dev, arma::abs(it.second - sgn * it.second.t()).max());
    }
    vmax = std::max(vmax, op.OneBody.n_elem ? arma::abs(op.OneBody).max() : 0.0);
    if (op.GetJRank() == 0)
      dev = std::max(dev, arma::abs(op.OneBody - sgn * op.OneBody.t()).max());
    if (dev > 1e-12 * std::max(vmax, 1e-300))
      fail(path, r.lineno, "hermiticity check failed: max deviation " + fmt17(dev));
  }
  return op;
}

std::unique_ptr<ModelSpace> modelspace_for_dump(const std::string& path, const Args& a) {
  DumpHeader h = read_opdump_header(path);
  SpaceParams p;
  p.emax = h.emax;
  p.hw = h.hw;
  p.reference = h.reference;
  p.valence_space = a.str("valence_space", h.reference);
  p.e3max = a.integer("e3max", std::min(12, 3 * h.emax));
  return make_modelspace(p);
}

namespace {

std::string dump_source_line(const std::string& path) {
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("source ", 0) == 0) return line.substr(7);
  return "";
}

std::string dump_reference(const std::string& path) { return read_opdump_header(path).reference; }

}  // namespace

int cmd_roundtrip(const Args& a) {
  if (a.positional.size() != 2) throw std::runtime_error("usage: refharness roundtrip IN OUT");
  const std::string& in = a.positional[0];
  auto ms = modelspace_for_dump(in, a);
  Operator op = read_opdump(in, *ms);
  write_opdump(a.positional[1], op, {dump_source_line(in), dump_reference(in)});
  return 0;
}

int cmd_norms(const Args& a) {
  if (a.positional.size() != 1) throw std::runtime_error("usage: refharness norms IN [out=FILE]");
  auto ms = modelspace_for_dump(a.positional[0], a);
  Operator op = read_opdump(a.positional[0], *ms);
  std::string j = norms_json(op);
  std::string out = a.str("out", "-");
  if (out == "-") std::cout << j << "\n";
  else { std::ofstream f(out); f << j << "\n"; }
  return 0;
}

}  // namespace rh
