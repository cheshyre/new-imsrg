// rh_tensor.cpp -- tensor-operator reference data of the OLD imsrg++ (M15 reference prep, run 3).
//
//   refharness dump-tensor emax=2 reference=He4 basis=HO|HF ops=E2,M1,... out=DIR
//       Builds each operator with imsrg_util::OperatorFromString and writes it as a raw tensor
//       dump (`# imsrg-tensor-dump raw v0`, see design/15-operator-dump-format.md, section
//       "Tensor reference dumps"). basis=HO: the operator exactly as the builder returns it
//       (harmonic-oscillator basis, not normal ordered). basis=HF: the pipeline of imsrg++.cc
//       (lines 815-831): scalar ops that change Tz or parity are made not reduced, then
//       HartreeFock::TransformToHFBasis, DoNormalOrdering, and (J = 0 only) MakeReduced again so
//       that every dump holds reduced matrix elements. With write_h=true also DIR/H_HF.dump
//       (HF-basis normal-ordered H, OperatorDump v1) and DIR/random_A.dump (seed 7).
//   refharness commutator-tensor emax=2 reference=He4 X=H,A Y=E2,GamowTeller out=DIR [terms=...]
//       Every term of Commutator::CommutatorScalarTensor (Commutator.cc:491-611) for X = the
//       HF-basis normal-ordered H (X=H) or a seeded random anti-Hermitian scalar (X=A, seed 7,
//       as commutator/emax2_O16/random_A.dump) and Y = the HF-basis tensor operator, plus `all`
//       (Commutator::Commutator(X, Y)). Output DIR/Z_<term>__<X>__<Y>.tdump and DIR/norms.json.
//   refharness bch-tensor emax=2 reference=He4 Y=E2 omega_norm=0.3 out=DIR
//       BCH::BCH_Transform(Y, Omega) with Omega = omega_scale * random_A (seed 7; omega_norm=X
//       picks the scale so that Omega.Norm() = X, omega_scale=S sets it), Y in the HF
//       basis: the Magnus step of a tensor operator.

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include "IMSRG.hh"
#include "TensorCommutators.hh"
#include "rh_common.hh"

namespace rh {

namespace {

std::vector<std::string> split_commas(const std::string& s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ','))
    if (!t.empty()) v.push_back(t);
  return v;
}

std::string command_line(const Args& a) {
  std::string s = a.subcommand;
  for (const auto& kv : a.kv) {
    if (kv.first == "out") continue;  // run directory: not reproducible metadata
    std::string v = kv.second;
    if (kv.first == "2bme" && v.find('/') != std::string::npos) v = v.substr(v.find_last_of('/') + 1);
    s += " " + kv.first + "=" + v;
  }
  for (const auto& p : a.positional) s += " " + p;
  return s;
}

/// Everything the three subcommands share: model space, bare H, and (for basis=HF) the HF
/// solution exactly as `refharness hf` computes it (imsrg++.cc:560-567, 658-668).
struct Setup {
  std::unique_ptr<ModelSpace> ms;
  std::unique_ptr<BareH> bare;
  std::unique_ptr<HFMBPT> hf;
  std::string reference;
  std::string basis;
};

Setup make_setup(const Args& a) {
  Setup s;
  SpaceParams sp = space_params(a);
  s.ms = make_modelspace(sp);
  s.reference = reference_name(sp);
  s.basis = a.str("basis", "HF");
  if (s.basis != "HO" && s.basis != "HF")
    throw std::runtime_error("basis must be HO or HF");
  if (s.basis == "HF" || a.has("need_hf")) {
    InteractionParams ip = interaction_params(a);
    s.bare.reset(new BareH(make_bare_hamiltonian(*s.ms, ip)));
    s.hf.reset(new HFMBPT(s.bare->H));
    s.hf->UnFreezeOccupations();
    s.hf->Solve();
  }
  return s;
}

/// The operator pipeline of imsrg++.cc (lines 815-831 for basis=HF; HO = as built).
Operator make_tensor_op(Setup& s, const std::string& name) {
  if (name.rfind("rand_", 0) == 0) {  // rand_<J>_<dTz>_<parity>_<H|A>_<seed>: UnitTest::RandomOp, 1B+2B
    int J, dtz, par, seed;
    char h;
    if (std::sscanf(name.c_str(), "rand_%d_%d_%d_%c_%d", &J, &dtz, &par, &h, &seed) != 5)
      throw std::runtime_error("bad random operator name " + name);
    UnitTest ut(*s.ms);
    ut.SetRandomSeed(static_cast<uint64_t>(seed));
    return ut.RandomOp(*s.ms, J, dtz, par, 2, h == 'H' ? +1 : -1);
  }
  Operator op = imsrg_util::OperatorFromString(*s.ms, name);
  if (s.basis == "HF") {
    if (op.GetJRank() == 0 && (op.GetTRank() != 0 || op.GetParity() != 0) && op.IsReduced())
      op.MakeNotReduced();
    op = s.hf->TransformToHFBasis(op);
    op = op.DoNormalOrdering();
    if (op.GetJRank() == 0 && !op.IsReduced() && (op.GetTRank() != 0 || op.GetParity() != 0))
      op.MakeReduced();
  }
  return op;
}

Operator make_random_A(ModelSpace& ms, int seed) {
  UnitTest ut(ms);
  ut.SetRandomSeed(static_cast<uint64_t>(seed));
  return ut.RandomOp(ms, 0, 0, 0, 2, -1);
}

struct TensorMeta {
  std::string source;
  std::string command;
  std::string opname;
  std::string basis;
  std::string reference;
};

/// Raw tensor dump, format v0 (design/15-operator-dump-format.md, "Tensor reference dumps").
void write_tensor_dump(const std::string& path, const Operator& op, const TensorMeta& m) {
  ModelSpace& ms = *op.GetModelSpace();
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  std::vector<std::string> l1, l2;
  const size_t n = ms.GetNumberOrbits();
  for (size_t a = 0; a < n; ++a)
    for (size_t b = 0; b < n; ++b) {
      double v = op.OneBody(a, b);
      if (v != 0.0) l1.push_back("1b " + std::to_string(a) + " " + std::to_string(b) + " " + fmt17(v));
    }
  size_t nblocks = 0, n2 = 0;
  for (const auto& it : op.TwoBody.MatEl) {
    const size_t chb = it.first[0], chk = it.first[1];
    TwoBodyChannel& tb = ms.GetTwoBodyChannel(chb);
    TwoBodyChannel& tk = ms.GetTwoBodyChannel(chk);
    const arma::mat& M = it.second;
    std::vector<std::string> blk;
    for (arma::uword r = 0; r < M.n_rows; ++r)
      for (arma::uword c = 0; c < M.n_cols; ++c) {
        double v = M(r, c);
        if (v == 0.0) continue;
        Ket& kb = tb.GetKet(r);
        Ket& kk = tk.GetKet(c);
        blk.push_back("2b " + std::to_string(kb.p) + " " + std::to_string(kb.q) + " " + std::to_string(kk.p) + " " +
                      std::to_string(kk.q) + " " + fmt17(v));
      }
    if (blk.empty()) continue;
    ++nblocks;
    n2 += blk.size();
    l2.push_back("block " + std::to_string(chb) + " " + std::to_string(tb.J) + " " + std::to_string(tb.parity) + " " +
                 std::to_string(tb.Tz) + " " + std::to_string(chk) + " " + std::to_string(tk.J) + " " +
                 std::to_string(tk.parity) + " " + std::to_string(tk.Tz) + " " + std::to_string(blk.size()));
    l2.insert(l2.end(), blk.begin(), blk.end());
  }
  const char herm = op.IsHermitian() ? 'H' : (op.IsAntiHermitian() ? 'A' : 'N');
  f << "# imsrg-tensor-dump raw v0 (M15 reference prep, run 3; NOT the final OperatorDump format)\n";
  f << "source " << m.source << "\n";
  f << "command " << m.command << "\n";
  f << "operator " << m.opname << "\n";
  f << "basis " << m.basis << " emax " << ms.GetEmax() << " hw " << fmt17(ms.GetHbarOmega()) << " reference "
    << m.reference << "\n";
  f << "spec rank_J " << op.GetJRank() << " parity " << op.GetParity() << " delta_Tz " << op.GetTRank()
    << " hermiticity " << herm << " particle_rank " << op.GetParticleRank() << " reduced "
    << (op.IsReduced() ? 1 : 0) << "\n";
  f << "convention reduced matrix elements <bra||O||ket> as stored by imsrg++ (scalars: sqrt(2J_bra+1) times "
       "the J-coupled element, TwoBodyME::MakeReduced; sigma and E2 equal (-1)^(j_bra-j_ket) times the "
       "Edmonds element, see design/15-operator-dump-format.md)\n";
  f << "convention 2B pairs: p<=q, r<=s, normalized antisymmetrized (TwoBodyME::GetTBME_norm), "
       "bra pair in channel ch_bra, ket pair in ch_ket; only ch_bra<=ch_ket blocks are stored "
       "(TwoBodyME::Allocate, TwoBodyME.cc:94-113); full matrix of each block listed\n";
  f << "convention channel = (J, parity, Tz) with Tz=(tz2_p+tz2_q)/2 in {-1,0,1}, ch = 6J+2(Tz+1)+parity; "
       "delta_Tz = |Tz_bra-Tz_ket|; 2tz=-1 is proton\n";
  f << "columns 1b a b value (OneBody(a,b) = <a||O||b>)\n";
  f << "columns block ch_bra J_bra parity_bra Tz_bra ch_ket J_ket parity_ket Tz_ket count "
       "(header of the following `count` 2b lines; only blocks with a nonzero element)\n";
  f << "columns 2b p q r s value (<pq;J_bra||O||rs;J_ket>, channels of the preceding block line)\n";
  f << "norbits " << n << "\n";
  for (size_t i = 0; i < n; ++i) {
    const Orbit& o = ms.GetOrbit(i);
    f << "orbit " << i << " " << o.n << " " << o.l << " " << o.j2 << " " << o.tz2 << " " << fmt17(o.occ) << "\n";
  }
  f << "zero_body " << fmt17(op.ZeroBody) << "\n";
  f << "one_body " << l1.size() << "\n";
  for (const auto& s : l1) f << s << "\n";
  f << "two_body " << n2 << " blocks " << nblocks << "\n";
  for (const auto& s : l2) f << s << "\n";
  f << "end\n";
}

std::string version_with(const std::string& what) { return reference_version() + " refharness " + what; }

}  // namespace

int cmd_dump_tensor(const Args& a) {
  Setup s = make_setup(a);
  const std::string out = a.str("out", ".");
  const std::string cmd = command_line(a);
  Json norms;
  for (const auto& name : split_commas(a.str("ops", "E2"))) {
    double t0 = omp_get_wtime();
    Operator op = make_tensor_op(s, name);
    write_tensor_dump(out + "/" + name + ".tdump", op,
                      {version_with("dump-tensor"), cmd, name, s.basis, s.reference});
    Json j;
    j.integer("rank_J", op.GetJRank()).integer("parity", op.GetParity()).integer("delta_Tz", op.GetTRank());
    j.integer("reduced", op.IsReduced() ? 1 : 0);
    j.raw("norms", norms_json(op));
    j.num("wall_time_s", omp_get_wtime() - t0);
    norms.raw(name, j.dump());
  }
  if (a.flag("write_h") && s.hf) {
    Operator HNO = s.hf->GetNormalOrderedH(2);
    write_opdump(out + "/H_HF.dump", HNO, {version_with("dump-tensor H_HF = HartreeFock::GetNormalOrderedH(2)"), s.reference});
    norms.raw("H_HF", norms_json(HNO));
    Operator A = make_random_A(*s.ms, a.integer("random_seed", 7));
    write_opdump(out + "/random_A.dump", A, {version_with("dump-tensor random_A = UnitTest::RandomOp(ms,0,0,0,2,-1)"), s.reference});
    norms.raw("random_A", norms_json(A));
  }
  norms.write(out + "/norms.json");
  return 0;
}

namespace {

using TermFn = void (*)(const Operator&, const Operator&, Operator&);

const std::vector<std::pair<std::string, TermFn>>& tensor_terms() {
  static const std::vector<std::pair<std::string, TermFn>> t = {
      {"comm111st", Commutator::comm111st},
      {"comm121st", Commutator::comm121st},
      {"comm122st", Commutator::comm122st},
      {"comm221st", Commutator::comm221st},
      {"comm222_pp_hhst", Commutator::comm222_pp_hhst},
      {"comm222_pp_hh_221st", Commutator::comm222_pp_hh_221st},
      {"comm222_phst", Commutator::comm222_phst},
  };
  return t;
}

/// Z created as in Commutator::CommutatorScalarTensor (Commutator.cc:494-508).
Operator make_Z(const Operator& X, const Operator& Y) {
  Operator Z(*(Y.modelspace), Y.GetJRank(), Y.GetTRank(), (X.GetParity() + Y.GetParity()) % 2, Y.GetParticleRank());
  if ((X.IsHermitian() && Y.IsHermitian()) || (X.IsAntiHermitian() && Y.IsAntiHermitian())) Z.SetAntiHermitian();
  else if ((X.IsHermitian() && Y.IsAntiHermitian()) || (X.IsAntiHermitian() && Y.IsHermitian())) Z.SetHermitian();
  else Z.SetNonHermitian();
  return Z;
}

}  // namespace

int cmd_commutator_tensor(const Args& a) {
  if (a.has("list")) {
    for (auto& t : tensor_terms()) std::cout << t.first << "\n";
    std::cout << "all\n";
    return 0;
  }
  Args b = a;
  b.kv["basis"] = "HF";
  Setup s = make_setup(b);
  const std::string out = a.str("out", ".");
  const std::string cmd = command_line(a);
  Operator HNO = s.hf->GetNormalOrderedH(2);
  Operator RA = make_random_A(*s.ms, a.integer("random_seed", 7));
  std::vector<std::string> xs = split_commas(a.str("X", "H,A"));
  std::vector<std::string> terms = split_commas(a.str("terms", ""));
  if (terms.empty()) {
    for (auto& t : tensor_terms()) terms.push_back(t.first);
    terms.push_back("all");
  }
  s.ms->PreCalculateSixJ();
  Commutator::SetUseIMSRG3(false);
  Json norms;
  for (const auto& yname : split_commas(a.str("Y", "E2,GamowTeller"))) {
    Operator Y = make_tensor_op(s, yname);
    for (const auto& xname : xs) {
      const Operator& X = (xname == "H") ? HNO : RA;
      for (const auto& term : terms) {
        Operator Z;
        if (term == "all") {
          Z = Commutator::Commutator(X, Y);
        } else {
          TermFn fn = nullptr;
          for (auto& t : tensor_terms())
            if (t.first == term) fn = t.second;
          if (!fn) throw std::runtime_error("unknown tensor commutator term " + term);
          Z = make_Z(X, Y);
          // first call of a (lambda, parity) key: serial, as CommutatorScalarTensor does
          bool save = Commutator::single_thread;
          Commutator::single_thread = true;
          fn(X, Y, Z);
          Commutator::single_thread = save;
        }
        std::string tag = term + "__" + xname + "__" + yname;
        // An identically zero result (e.g. every 2B-2B term with a one-body Y) is recorded in
        // norms.json only; keep_zero=true writes its dump too.
        const bool zero = (std::abs(Z.ZeroBody) + Z.OneBodyNorm() + Z.TwoBodyNorm()) == 0.0;
        if (!zero || a.flag("keep_zero"))
          write_tensor_dump(out + "/Z_" + tag + ".tdump", Z,
                            {version_with("commutator-tensor " + term), cmd,
                             "[" + xname + "," + yname + "] term " + term, "HF", s.reference});
        Json zj;
        zj.boolean("zero", zero).raw("norms", norms_json(Z));
        norms.raw(tag, zj.dump());
      }
    }
  }
  norms.write(out + "/norms.json");
  return 0;
}

int cmd_bch_tensor(const Args& a) {
  Args b = a;
  b.kv["basis"] = "HF";
  Setup s = make_setup(b);
  const std::string out = a.str("out", ".");
  const std::string cmd = command_line(a);
  Operator A = make_random_A(*s.ms, a.integer("random_seed", 7));
  // omega_norm=X rescales so that Omega.Norm() = X; otherwise omega_scale multiplies random_A.
  const double scale = a.has("omega_norm") ? a.real("omega_norm", 0.3) / A.Norm() : a.real("omega_scale", 0.05);
  Operator Omega = A * scale;
  s.ms->PreCalculateSixJ();
  Commutator::SetUseIMSRG3(false);
  BCH::SetUseFactorizedCorrection(false);
  Json norms;
  norms.num("omega_scale", scale);
  norms.num("omega_norm", Omega.Norm());
  for (const auto& yname : split_commas(a.str("Y", "E2"))) {
    Operator Y = make_tensor_op(s, yname);
    Operator Out = BCH::BCH_Transform(Y, Omega);
    write_tensor_dump(out + "/Y_" + yname + "_transformed.tdump", Out,
                      {version_with("bch-tensor"), cmd, "BCH_Transform(" + yname + ", Omega)", "HF", s.reference});
    Json j;
    j.raw("input", norms_json(Y));
    j.raw("output", norms_json(Out));
    norms.raw(yname, j.dump());
  }
  norms.write(out + "/norms.json");
  return 0;
}

}  // namespace rh
