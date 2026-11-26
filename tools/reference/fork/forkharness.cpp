// forkharness -- golden-data harness linked against the owner's imsrg++ fork at 79e9330
// (design/50-al27-responses.md C1; built by tools/reference/build_fork.sh, driven by
// tools/reference/gen_golden_nreft.sh). Never part of the new build.
//
// Subcommands (all write plain text to stdout, numbers with 17 significant digits):
//   nreft-bare <emax> <hw> <all|shorthands> <version>
//       one-body tables of the bare DM_NREFT operators in the HO basis of a ModelSpace(emax,
//       "O16"): the shorthands FM/FPhipp (L = 0,2,4), FSigmap/FSigmapp/FDelta (L = 1,3,5) for p
//       and n through imsrg_util::OperatorFromString (fork prefactors included), and the raw
//       building blocks (p only) M, Phipp, Phip, Phitp, Sigma (= Ms(J,J)), Sigmap, Sigmapp, Delta,
//       Deltap for J = 0..5, Ms(J,L) and Mg(J,L) for |L-J| <= 1, at q = 0, 15, 150, 600, 1050 MeV.
//       The set "shorthands" writes only the shorthands (keeps the emax-4 golden small).
//       Format:
//         # version <version>          (the version string of the fork build)
//         emax <e> hw <hw>
//         orbit <i> <n> <l> <j2> <tz2>  (one line per orbit, imsrg++ order)
//         op <name> <J> <parity> <hermitian> <antihermitian>
//         <a> <b> <value>               (every nonzero <a||O_J||b>, J = 0 stored non-reduced)
//         end
//   nreft-radial <version>
//       the radial integrals jho, jdmho, jdpho(na,la,nb,lb,L,y) for 2n+l <= 3, L = 0..5, at y for
//       q = 15, 150, 1050 MeV and hw = 16. Lines: radial <f> <na> <la> <nb> <lb> <L> <y> <value>.
//   commutator-tensor <emax> <reference> <outdir> <version>
//       Every term of the fork's Commutator::CommutatorScalarTensor (Commutator.cc:449-511):
//       comm111st, comm121st, comm122st, comm222_pp_hh_221st, comm222_phst, and `all`
//       (Commutator::Commutator), for X = a random anti-Hermitian scalar (seed 7) and random
//       Hermitian tensors Y (1B + 2B; names rand_<J>_<dTz>_<parity>_H_<seed>) in the HO model
//       space ModelSpace(emax, reference). The fork's RandomOp ignores the Wigner-Eckart phase
//       of the one-body part of a reduced tensor; Y.Symmetrize() restores Hermiticity. Writes
//       <outdir>/inputs/random_A.dump (OperatorDump v1), <outdir>/inputs/<Y>.tdump and
//       <outdir>/commutator/Z_<term>__A__<Y>.tdump (raw tensor dump v0, as rh_tensor.cpp;
//       identically zero terms are not written). The AL-F1 investigation (odd-rank tensors).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Commutator.hh"
#include "DarkMatterNREFT.hh"
#include "UnitTest.hh"
#include "ModelSpace.hh"
#include "Operator.hh"
#include "PhysicalConstants.hh"
#include "imsrg_util.hh"

namespace {

const std::vector<double> kQ = {0.0, 15.0, 150.0, 600.0, 1050.0};

std::string Fmt(double q) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%g", q);
  return buf;
}

void DumpOp(const std::string& name, const Operator& op) {
  std::printf("op %s %d %d %d %d\n", name.c_str(), op.GetJRank(), op.GetParity(),
              op.IsHermitian() ? 1 : 0, op.IsAntiHermitian() ? 1 : 0);
  const arma::mat& m = op.OneBody;
  for (arma::uword a = 0; a < m.n_rows; ++a)
    for (arma::uword b = 0; b < m.n_cols; ++b)
      if (m(a, b) != 0.0) std::printf("%d %d %.16e\n", int(a), int(b), m(a, b));
  std::printf("end\n");
}

int NreftBare(int emax, double hw, bool raw, const std::string& version) {
  ModelSpace ms(emax, "O16");
  ms.SetHbarOmega(hw);
  std::printf("# version %s\n", version.c_str());
  std::printf("emax %d hw %.17g\n", emax, hw);
  for (int i = 0; i < ms.GetNumberOrbits(); ++i) {
    Orbit& o = ms.GetOrbit(i);
    std::printf("orbit %d %d %d %d %d\n", i, o.n, o.l, o.j2, o.tz2);
  }
  // Shorthands through the fork's parser (prefactors included).
  const std::vector<std::pair<std::string, std::vector<int>>> shorthands = {
      {"FM", {0, 2, 4}},      {"FPhipp", {0, 2, 4}},  {"FSigmap", {1, 3, 5}},
      {"FSigmapp", {1, 3, 5}}, {"FDelta", {1, 3, 5}}};
  for (const auto& s : shorthands)
    for (int L : s.second)
      for (const char* iso : {"p", "n"})
        for (double q : kQ) {
          const std::string name = s.first + std::to_string(L) + iso + "_" + Fmt(q);
          DumpOp(name, imsrg_util::OperatorFromString(ms, name));
        }
  if (!raw) return 0;
  // Raw building blocks, protons only (the neutron tables are the same numbers on the
  // neutron orbits; the p/n selection is tested through the shorthands above).
  using Fn = Operator (*)(ModelSpace&, std::string, int, double);
  const std::vector<std::pair<std::string, Fn>> blocks = {
      {"M", &DM_NREFT::M},           {"Phipp", &DM_NREFT::Phipp},
      {"Phip", &DM_NREFT::Phip},     {"Phitp", &DM_NREFT::Phitp},
      {"Sigma", &DM_NREFT::Sigma},   {"Sigmap", &DM_NREFT::Sigmap},
      {"Sigmapp", &DM_NREFT::Sigmapp}, {"Delta", &DM_NREFT::Delta},
      {"Deltap", &DM_NREFT::Deltap}};
  for (const auto& b : blocks)
    for (int J = 0; J <= 5; ++J)
      for (double q : kQ) {
        // DM_NREFT::Phip/Phipp/... with J = 0 of the abnormal families are legal calls
        // (zero tables); they are dumped as well, so the new code must give zeros there too.
        DumpOp("raw_" + b.first + "_" + std::to_string(J) + "_" + Fmt(q),
               b.second(ms, "p", J, q));
      }
  for (int J = 0; J <= 5; ++J)
    for (int L = std::max(0, J - 1); L <= J + 1; ++L)
      for (double q : kQ) {
        const std::string tag = std::to_string(J) + "_" + std::to_string(L) + "_" + Fmt(q);
        DumpOp("raw_Ms_" + tag, DM_NREFT::Ms(ms, "p", J, L, q));
        DumpOp("raw_Mg_" + tag, DM_NREFT::Mg(ms, "p", J, L, q));
      }
  return 0;
}

int NreftRadial(const std::string& version) {
  std::printf("# version %s\n", version.c_str());
  const double hw = 16.0;
  const double b2 = 1.0 / (hw * PhysConst::M_NUCLEON);
  for (double q : {15.0, 150.0, 1050.0}) {
    const double y = q * q * b2 / 4.0;
    for (int na = 0; na <= 1; ++na)
      for (int la = 0; 2 * na + la <= 3; ++la)
        for (int nb = 0; nb <= 1; ++nb)
          for (int lb = 0; 2 * nb + lb <= 3; ++lb)
            for (int L = 0; L <= 5; ++L) {
              std::printf("radial jho %d %d %d %d %d %.17e %.16e\n", na, la, nb, lb, L, y,
                          DM_NREFT::jho(na, la, nb, lb, L, y));
              std::printf("radial jdmho %d %d %d %d %d %.17e %.16e\n", na, la, nb, lb, L, y,
                          DM_NREFT::jdmho(na, la, nb, lb, L, y));
              std::printf("radial jdpho %d %d %d %d %d %.17e %.16e\n", na, la, nb, lb, L, y,
                          DM_NREFT::jdpho(na, la, nb, lb, L, y));
            }
  }
  return 0;
}


std::string F17(double x) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.17g", x);
  return buf;
}

char HermChar(const Operator& op) {
  return op.IsHermitian() ? 'H' : (op.IsAntiHermitian() ? 'A' : 'N');
}

void WriteOrbits(std::ofstream& f, ModelSpace& ms) {
  f << "norbits " << ms.GetNumberOrbits() << "\n";
  for (int i = 0; i < ms.GetNumberOrbits(); ++i) {
    const Orbit& o = ms.GetOrbit(i);
    f << "orbit " << i << " " << o.n << " " << o.l << " " << o.j2 << " " << o.tz2 << " "
      << F17(o.occ) << "\n";
  }
}

/// OperatorDump v1 (design/15) of a scalar 1B+2B operator: every nonzero normalized element of
/// the (diagonal) channel blocks, p<=q, r<=s.
void WriteScalarDump(const std::string& path, Operator& op, const std::string& source,
                     const std::string& reference) {
  ModelSpace& ms = *op.GetModelSpace();
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << "# imsrg-operator-dump v1\n";
  f << "source " << source << "\n";
  f << "basis emax " << ms.GetEmax() << " lmax " << std::min(ms.GetLmax(), ms.GetEmax())
    << " hw " << F17(ms.GetHbarOmega()) << "\n";
  f << "reference " << reference << "\n";
  WriteOrbits(f, ms);
  f << "spec particle_rank 2 rank_J 0 parity 0 delta_Tz 0 hermiticity " << HermChar(op) << "\n";
  f << "zero_body " << F17(op.ZeroBody) << "\n";
  std::vector<std::string> ob, tb;
  for (arma::uword a = 0; a < op.OneBody.n_rows; ++a)
    for (arma::uword b = 0; b < op.OneBody.n_cols; ++b)
      if (op.OneBody(a, b) != 0.0)
        ob.push_back(std::to_string(a) + " " + std::to_string(b) + " " + F17(op.OneBody(a, b)));
  for (auto& it : op.TwoBody.MatEl) {
    TwoBodyChannel& t = ms.GetTwoBodyChannel(it.first[0]);
    const arma::mat& M = it.second;
    for (arma::uword i = 0; i < M.n_rows; ++i)
      for (arma::uword j = 0; j < M.n_cols; ++j) {
        if (M(i, j) == 0.0) continue;
        Ket& bra = t.GetKet(i);
        Ket& ket = t.GetKet(j);
        tb.push_back(std::to_string(bra.p) + " " + std::to_string(bra.q) + " " +
                     std::to_string(ket.p) + " " + std::to_string(ket.q) + " " +
                     std::to_string(t.J) + " " + std::to_string(t.J) + " " + F17(M(i, j)));
      }
  }
  f << "one_body " << ob.size() << "\n";
  for (auto& l : ob) f << l << "\n";
  f << "two_body " << tb.size() << "\n";
  for (auto& l : tb) f << l << "\n";
  f << "three_body none\nend\n";
}

/// Raw tensor dump v0 (design/15, "Tensor reference dumps"), as tools/reference/rh_tensor.cpp.
void WriteTensorDump(const std::string& path, Operator& op, const std::string& source,
                     const std::string& command, const std::string& opname,
                     const std::string& reference) {
  ModelSpace& ms = *op.GetModelSpace();
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  std::vector<std::string> l1, l2;
  const int n = ms.GetNumberOrbits();
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b)
      if (op.OneBody(a, b) != 0.0)
        l1.push_back("1b " + std::to_string(a) + " " + std::to_string(b) + " " +
                     F17(op.OneBody(a, b)));
  size_t nblocks = 0, n2 = 0;
  for (auto& it : op.TwoBody.MatEl) {
    const size_t chb = it.first[0], chk = it.first[1];
    TwoBodyChannel& tb = ms.GetTwoBodyChannel(chb);
    TwoBodyChannel& tk = ms.GetTwoBodyChannel(chk);
    const arma::mat& M = it.second;
    std::vector<std::string> blk;
    for (arma::uword r = 0; r < M.n_rows; ++r)
      for (arma::uword c = 0; c < M.n_cols; ++c) {
        if (M(r, c) == 0.0) continue;
        Ket& kb = tb.GetKet(r);
        Ket& kk = tk.GetKet(c);
        blk.push_back("2b " + std::to_string(kb.p) + " " + std::to_string(kb.q) + " " +
                      std::to_string(kk.p) + " " + std::to_string(kk.q) + " " + F17(M(r, c)));
      }
    if (blk.empty()) continue;
    ++nblocks;
    n2 += blk.size();
    l2.push_back("block " + std::to_string(chb) + " " + std::to_string(tb.J) + " " +
                 std::to_string(tb.parity) + " " + std::to_string(tb.Tz) + " " +
                 std::to_string(chk) + " " + std::to_string(tk.J) + " " +
                 std::to_string(tk.parity) + " " + std::to_string(tk.Tz) + " " +
                 std::to_string(blk.size()));
    l2.insert(l2.end(), blk.begin(), blk.end());
  }
  f << "# imsrg-tensor-dump raw v0 (M15 reference prep, run 3; NOT the final OperatorDump "
       "format)\n";
  f << "source " << source << "\n";
  f << "command " << command << "\n";
  f << "operator " << opname << "\n";
  f << "basis HO emax " << ms.GetEmax() << " hw " << F17(ms.GetHbarOmega()) << " reference "
    << reference << "\n";
  f << "spec rank_J " << op.GetJRank() << " parity " << op.GetParity() << " delta_Tz "
    << op.GetTRank() << " hermiticity " << HermChar(op) << " particle_rank "
    << op.GetParticleRank() << " reduced " << (op.GetJRank() > 0 ? 1 : 0) << "\n";
  f << "convention as tools/reference/rh_tensor.cpp (reduced matrix elements as stored by "
       "imsrg++, normalized antisymmetrized 2B pairs p<=q, r<=s, ch_bra<=ch_ket blocks)\n";
  WriteOrbits(f, ms);
  f << "zero_body " << F17(op.ZeroBody) << "\n";
  f << "one_body " << l1.size() << "\n";
  for (const auto& s : l1) f << s << "\n";
  f << "two_body " << n2 << " blocks " << nblocks << "\n";
  for (const auto& s : l2) f << s << "\n";
  f << "end\n";
}

int CommutatorTensor(int emax, const std::string& reference, const std::string& out,
                     const std::string& version) {
  ModelSpace ms(emax, reference);
  ms.SetHbarOmega(16.0);
  ms.PreCalculateSixJ();
  Commutator::SetUseIMSRG3(false);
  const std::string src = "imsrg-fork-79e9330 " + version + " forkharness commutator-tensor";
  const std::string cmd = "commutator-tensor emax=" + std::to_string(emax) + " reference=" +
                          reference;
  UnitTest ut(ms);
  arma::arma_rng::set_seed(7);
  Operator A = ut.RandomOp(ms, 0, 0, 0, 2, -1);
  WriteScalarDump(out + "/inputs/random_A.dump", A,
                  src + " random_A = arma seed 7, UnitTest::RandomOp(ms,0,0,0,2,-1)", reference);
  using TermFn = void (*)(const Operator&, const Operator&, Operator&);
  const std::vector<std::pair<std::string, TermFn>> terms = {
      {"comm111st", Commutator::comm111st},
      {"comm121st", Commutator::comm121st},
      {"comm122st", Commutator::comm122st},
      {"comm222_pp_hh_221st", Commutator::comm222_pp_hh_221st},
      {"comm222_phst", Commutator::comm222_phst}};
  for (const std::string yname :
       {"rand_1_0_0_H_4", "rand_3_0_0_H_5", "rand_1_0_1_H_6", "rand_2_0_0_H_3"}) {
    int J, dtz, par, seed;
    char h;
    std::sscanf(yname.c_str(), "rand_%d_%d_%d_%c_%d", &J, &dtz, &par, &h, &seed);
    arma::arma_rng::set_seed(seed);
    Operator Y = ut.RandomOp(ms, J, dtz, par, 2, +1);
    Y.Symmetrize();
    WriteTensorDump(out + "/inputs/" + yname + ".tdump", Y, src, cmd, yname, reference);
    for (const auto& t : terms) {
      Operator Z = Y;  // as CommutatorScalarTensor
      Z.EraseZeroBody();
      Z.EraseOneBody();
      Z.EraseTwoBody();
      Z.SetHermitian();  // anti-Hermitian X, Hermitian Y
      t.second(A, Y, Z);
      if (Z.OneBodyNorm() + Z.TwoBodyNorm() == 0.0) continue;
      WriteTensorDump(out + "/commutator/Z_" + t.first + "__A__" + yname + ".tdump", Z, src,
                      cmd + " term " + t.first, "[A," + yname + "] term " + t.first, reference);
    }
    Operator Z = Commutator::Commutator(A, Y);
    WriteTensorDump(out + "/commutator/Z_all__A__" + yname + ".tdump", Z, src, cmd + " all",
                    "[A," + yname + "]", reference);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: forkharness nreft-bare <emax> <hw> <all|shorthands> "
                         "<version> | "
                         "nreft-radial <version>\n");
    return 2;
  }
  const std::string cmd = argv[1];
  // The fork prints diagnostics (isofactors, timings) on std::cout: send them to stderr so that
  // stdout carries only the dump (printf goes to the C stdout, which stays the dump).
  std::cout.rdbuf(std::cerr.rdbuf());
  if (cmd == "nreft-bare" && argc == 6)
    return NreftBare(std::atoi(argv[2]), std::atof(argv[3]), std::string(argv[4]) == "all",
                     argv[5]);
  if (cmd == "nreft-radial" && argc == 3) return NreftRadial(argv[2]);
  if (cmd == "commutator-tensor" && argc == 6)
    return CommutatorTensor(std::atoi(argv[2]), argv[3], argv[4], argv[5]);
  std::fprintf(stderr, "forkharness: bad arguments\n");
  return 2;
}
