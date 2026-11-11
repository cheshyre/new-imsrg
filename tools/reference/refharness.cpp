// refharness.cpp -- golden-data harness linked against the old imsrg++ library (imsrg-devel).
//
// Usage: refharness <subcommand> [key=value ...] [positional ...]
//
//   dump-basis        orbits, 2B channels and kets, ph (cross-coupled) channels, 3B channel
//                     counts of a model space                         (rh_basis.cpp)
//   dump-hamiltonian  OperatorDumps of the bare Hamiltonian pieces T, V2, (NO2B 3N
//                     contribution) and H, plus norms                 (rh_hamiltonian.cpp)
//   hf                HF: E_HF, SPEs, iterations, C, normal-ordered H, MP2/MP3
//                                                                     (rh_hamiltonian.cpp)
//   roundtrip IN OUT  read an OperatorDump and write it again         (rh_opdump.cpp)
//   norms IN          norms of an OperatorDump as imsrg++ computes them (rh_opdump.cpp)
//   normal-order IN OUT [reference=NAME] [undo=true]
//                     re-normal-order an OperatorDump w.r.t. another reference (Operator::
//                     DoNormalOrdering / UndoNormalOrdering)           (rh_operator.cpp)
//   basis-transform IN C.txt OUT   HartreeFock::TransformToHFBasis with C read from a text file
//                                                                     (rh_operator.cpp)
//   commutator TERM X Y Z   evaluate one named commutator term        (rh_commutator.cpp)
//   bench-commutator emax=N reps=R  time the imsrg++ commutator terms        (rh_bench.cpp)
//   generator TYPE H ETA   imsrg++ generator eta(H) of a given type    (rh_commutator.cpp)
//   flow [--factorized]     full IMSRG pipeline with %.17g trajectory  (rh_flow.cpp)
//
// Common keys (imsrg++ names): emax e3max lmax hw reference valence_space A
//   2bme file2e1max file2e2max file2lmax 3bme 3bme_type file3e1max file3e2max file3e3max
//   no2b_precision out (output directory or prefix, default ".").
// Never part of the new build; see design/30-testing-and-validation.md section 3.

#include <omp.h>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>

#include "IMSRG.hh"
#include "rh_common.hh"

namespace rh {

std::string Args::str(const std::string& k, const std::string& def) const {
  auto it = kv.find(k);
  return it == kv.end() ? def : it->second;
}

int Args::integer(const std::string& k, int def) const {
  auto it = kv.find(k);
  if (it == kv.end()) return def;
  size_t pos = 0;
  int v = std::stoi(it->second, &pos);
  if (pos != it->second.size()) throw std::runtime_error("bad integer for " + k + ": " + it->second);
  return v;
}

double Args::real(const std::string& k, double def) const {
  auto it = kv.find(k);
  if (it == kv.end()) return def;
  size_t pos = 0;
  double v = std::stod(it->second, &pos);
  if (pos != it->second.size()) throw std::runtime_error("bad number for " + k + ": " + it->second);
  return v;
}

bool Args::flag(const std::string& k) const {
  auto it = kv.find(k);
  return it != kv.end() && (it->second == "true" || it->second == "1" || it->second.empty());
}

Args parse_args(int argc, char** argv) {
  Args a;
  if (argc < 2) throw std::runtime_error("missing subcommand (see the header of refharness.cpp)");
  a.subcommand = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string s = argv[i];
    if (s.rfind("--", 0) == 0) {  // --flag or --key=value
      s = s.substr(2);
      auto eq = s.find('=');
      if (eq == std::string::npos) a.kv[s] = "";
      else a.kv[s.substr(0, eq)] = s.substr(eq + 1);
      continue;
    }
    auto eq = s.find('=');
    if (eq == std::string::npos || eq == 0) a.positional.push_back(s);
    else a.kv[s.substr(0, eq)] = s.substr(eq + 1);
  }
  return a;
}

std::string fmt17(double x) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17g", x);
  return buf;
}

std::string reference_version() {
  const char* root = std::getenv("IMSRG_REFERENCE_ROOT");
  std::string path = std::string(root ? root : "/Users/cheshyre/Downloads/new-imsrg-reference") +
                     "/REFERENCE_SRC_SHA256";
  std::ifstream f(path);
  std::string h;
  if (f >> h) return "imsrg-devel src-sha256:" + h;
  return "imsrg-devel src-sha256:unknown";
}

SpaceParams space_params(const Args& a) {
  SpaceParams p;
  p.emax = a.integer("emax", p.emax);
  p.e3max = a.integer("e3max", p.e3max);
  p.lmax = a.integer("lmax", p.lmax);
  p.hw = a.real("hw", p.hw);
  p.reference = a.str("reference", p.reference);
  p.valence_space = a.str("valence_space", p.valence_space);
  p.A = a.integer("A", p.A);
  return p;
}

std::unique_ptr<ModelSpace> make_modelspace(const SpaceParams& p) {
  // imsrg++.cc:331-363 (custom valence spaces and occ_file are not supported here)
  std::unique_ptr<ModelSpace> ms(p.reference == "default"
                                     ? new ModelSpace(p.emax, p.valence_space)
                                     : new ModelSpace(p.emax, p.reference, p.valence_space));
  ms->SetE3max(p.e3max);
  ms->SetLmax(p.lmax);
  ms->SetHbarOmega(p.hw);
  if (p.A > 0) ms->SetTargetMass(p.A);
  return ms;
}

std::string reference_name(const SpaceParams& p) {
  if (p.reference != "default") return p.reference;
  auto it = ModelSpace::ValenceSpaces.find(p.valence_space);
  if (it != ModelSpace::ValenceSpaces.end()) return it->second.at(0);
  return p.valence_space;  // a nucleus name
}

InteractionParams interaction_params(const Args& a) {
  InteractionParams ip;
  ip.file2 = a.str("2bme", ip.file2);
  ip.file2e1max = a.integer("file2e1max", ip.file2e1max);
  ip.file2e2max = a.integer("file2e2max", ip.file2e2max);
  ip.file2lmax = a.integer("file2lmax", ip.file2lmax);
  ip.file3 = a.str("3bme", ip.file3);
  ip.type3 = a.str("3bme_type", ip.type3);
  ip.file3e1max = a.integer("file3e1max", ip.file3e1max);
  ip.file3e2max = a.integer("file3e2max", ip.file3e2max);
  ip.file3e3max = a.integer("file3e3max", ip.file3e3max);
  ip.no2b_precision = a.str("no2b_precision", ip.no2b_precision);
  return ip;
}

BareH make_bare_hamiltonian(ModelSpace& ms, const InteractionParams& ip) {
  // imsrg++.cc:430-541 with fmt2=me2j, 3bme_type=no2b, no Lawson term, no hw_trap.
  ReadWrite rw;
  BareH b;
  b.has3n = (ip.file3 != "none");
  if (b.has3n && ip.type3 != "no2b")
    throw std::runtime_error("refharness supports only 3bme_type=no2b");

  b.V2 = Operator(ms, 0, 0, 0, 2);
  b.V2.SetHermitian();
  rw.ReadBareTBME_Darmstadt(ip.file2, b.V2, ip.file2e1max, ip.file2e2max, ip.file2lmax);

  b.T = imsrg_util::Trel_Op(ms);

  b.H = Operator(ms, 0, 0, 0, b.has3n ? 3 : 2);
  b.H.SetHermitian();
  rw.ReadBareTBME_Darmstadt(ip.file2, b.H, ip.file2e1max, ip.file2e2max, ip.file2lmax);
  if (b.has3n) {
    b.H.ThreeBody.SetMode(ip.no2b_precision == "half" ? "no2bhalf" : "no2b");
    b.H.ThreeBody.ReadFile({ip.file3}, {ip.file3e1max, ip.file3e2max, ip.file3e3max, ip.file3e1max});
  }
  b.H += b.T;
  if (b.H.OneBody.has_nan()) throw std::runtime_error("Trel has NaN");
  return b;
}

// ---------------------------------------------------------------- JSON
std::string json_escape(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (c == '\n') o += "\\n";
    else o += c;
  }
  return o + "\"";
}

static std::string json_number(double v) {
  if (std::isnan(v) || std::isinf(v)) return json_escape(fmt17(v));  // JSON has no NaN/Inf
  return fmt17(v);
}

Json& Json::num(const std::string& k, double v) { items_.push_back({k, json_number(v)}); return *this; }
Json& Json::integer(const std::string& k, long long v) { items_.push_back({k, std::to_string(v)}); return *this; }
Json& Json::str(const std::string& k, const std::string& v) { items_.push_back({k, json_escape(v)}); return *this; }
Json& Json::boolean(const std::string& k, bool v) { items_.push_back({k, v ? "true" : "false"}); return *this; }
Json& Json::raw(const std::string& k, const std::string& v) { items_.push_back({k, v}); return *this; }
Json& Json::nums(const std::string& k, const std::vector<double>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + json_number(v[i]);
  items_.push_back({k, s + "]"});
  return *this;
}
Json& Json::ints(const std::string& k, const std::vector<long long>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
  items_.push_back({k, s + "]"});
  return *this;
}

std::string Json::dump(int indent) const {
  std::string pad(indent, ' ');
  std::string s = "{\n";
  for (size_t i = 0; i < items_.size(); ++i) {
    // re-indent nested objects
    std::string v = items_[i].second, vv;
    for (char c : v) { vv += c; if (c == '\n') vv += pad; }
    s += pad + json_escape(items_[i].first) + ": " + vv + (i + 1 < items_.size() ? ",\n" : "\n");
  }
  return s + "}";
}

void Json::write(const std::string& path) const {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << dump() << "\n";
}

std::string norms_json(const Operator& op) {
  Json j;
  j.num("zero_body", op.ZeroBody);
  j.num("one_body_norm", op.OneBodyNorm());
  j.num("two_body_norm", op.TwoBodyNorm());
  j.num("three_body_norm", op.GetParticleRank() >= 3 ? op.ThreeBodyNorm() : 0.0);
  j.num("norm", op.Norm());
  return j.dump();
}

}  // namespace rh

int main(int argc, char** argv) {
  try {
    rh::Args a = rh::parse_args(argc, argv);
    const std::string& c = a.subcommand;
    if (c == "dump-basis") return rh::cmd_dump_basis(a);
    if (c == "dump-hamiltonian") return rh::cmd_dump_hamiltonian(a);
    if (c == "hf") return rh::cmd_hf(a);
    if (c == "roundtrip") return rh::cmd_roundtrip(a);
    if (c == "norms") return rh::cmd_norms(a);
    if (c == "normal-order") return rh::cmd_normal_order(a);
    if (c == "basis-transform") return rh::cmd_basis_transform(a);
    if (c == "commutator") return rh::cmd_commutator(a);
    if (c == "bench-commutator") return rh::cmd_bench_commutator(a);
    if (c == "generator") return rh::cmd_generator(a);
    if (c == "flow") return rh::cmd_flow(a);
    std::cerr << "refharness: unknown subcommand '" << c << "'\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "refharness: error: " << e.what() << "\n";
    return 1;
  }
}
