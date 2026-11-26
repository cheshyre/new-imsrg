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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "DarkMatterNREFT.hh"
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
  std::fprintf(stderr, "forkharness: bad arguments\n");
  return 2;
}
