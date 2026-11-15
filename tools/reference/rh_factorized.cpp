// rh_factorized.cpp -- factorized double commutator (IMSRG(3f2)) and BCH with the OLD imsrg++.
//
//   refharness factorized PART ETA.dump GAMMA.dump Z.dump [valence_space=..] [e3max=..]
//       Z = the part PART of the factorized double commutator [Eta,[Eta,Gamma]_3b]_{1b,2b}
//       (comm223_231 + comm223_232 of Commutator::FactorizedDoubleCommutator). Z is a new
//       rank-2 scalar operator, not reduced, with Gamma's hermiticity (hZ = hGamma, as the
//       factorized code assumes). PART:
//         chi1b_231 chi1b_232 chi2b_231 chi2b_232   one routine each
//         d32      1b intermediates on, 2b off, comm223_231 + comm223_232 (the D32 correction)
//         all      both on, comm223_231 + comm223_232
//         direct   UnitTest::TestFactorizedDoubleCommutators route: comm223ss into a rank-3
//                  pn-mode intermediate, then comm231ss + comm232ss into Z
//   refharness bch transform|product A.dump B.dump OUT.dump [scale_a=S] [scale_b=S]
//       [--factorized] [--factorized-1b-intermediates=0|1] [--factorized-2b-intermediates=0|1]
//       A, B multiplied by the scales; transform: OUT = BCH_Transform(A, B) (B = Omega);
//       product: OUT = BCH_Product(A, B). Prints the norms with %.17g.

#include <iostream>

#include "IMSRG.hh"
#include "FactorizedDoubleCommutator.hh"
#include "rh_common.hh"

namespace rh {

namespace {
namespace FDC = Commutator::FactorizedDoubleCommutator;
}

int cmd_factorized(const Args& a) {
  if (a.positional.size() != 4)
    throw std::runtime_error(
        "usage: refharness factorized PART ETA.dump GAMMA.dump Z.dump "
        "(PART: chi1b_231 chi1b_232 chi2b_231 chi2b_232 d32 all direct)");
  const std::string& part = a.positional[0];
  const std::string& ep = a.positional[1];
  const std::string& gp = a.positional[2];
  const std::string& zp = a.positional[3];

  auto ms = modelspace_for_dump(ep, a);
  Operator Eta = read_opdump(ep, *ms);
  Operator Gamma = read_opdump(gp, *ms);
  if (Eta.GetJRank() != 0 || Gamma.GetJRank() != 0)
    throw std::runtime_error("only scalar ETA and GAMMA are supported");
  if (Eta.IsReduced()) Eta.MakeNotReduced();
  if (Gamma.IsReduced()) Gamma.MakeNotReduced();

  const int jrank = Eta.GetJRank() + Gamma.GetJRank();
  const int tz = Eta.GetTRank() + Gamma.GetTRank();
  const int parity = (Eta.GetParity() + Gamma.GetParity()) % 2;
  Operator Z(*ms, jrank, tz, parity, 2);
  if (Gamma.IsHermitian()) Z.SetHermitian();
  else Z.SetAntiHermitian();
  if (Z.IsReduced()) Z.MakeNotReduced();

  if (part == "chi1b_231") FDC::comm223_231_chi1b(Eta, Gamma, Z);
  else if (part == "chi1b_232") FDC::comm223_232_chi1b(Eta, Gamma, Z);
  else if (part == "chi2b_231") FDC::comm223_231_chi2b(Eta, Gamma, Z);
  else if (part == "chi2b_232") FDC::comm223_232_chi2b(Eta, Gamma, Z);
  else if (part == "d32" || part == "all") {
    FDC::SetUse_1b_Intermediates(true);
    FDC::SetUse_2b_Intermediates(part == "all");
    FDC::comm223_231(Eta, Gamma, Z);
    FDC::comm223_232(Eta, Gamma, Z);
  } else if (part == "direct") {
    // as UnitTest::TestFactorizedDoubleCommutators (UnitTest.cc:6020-6090)
    Operator Inter(*ms, jrank, tz, parity, 3);
    const int hEta = Eta.IsHermitian() ? 1 : -1;
    const int hH = Gamma.IsHermitian() ? 1 : -1;
    if (-1 * hEta * hH < 0) Inter.SetAntiHermitian();
    Inter.ThreeBody.SetMode("pn");
    if (Inter.IsReduced()) Inter.MakeNotReduced();
    Commutator::comm223ss(Eta, Gamma, Inter);
    Commutator::comm231ss(Eta, Inter, Z);
    Commutator::comm232ss(Eta, Inter, Z);
  } else {
    throw std::runtime_error("unknown PART '" + part + "'");
  }
  write_opdump(zp, Z,
               {reference_version() + " refharness factorized " + part,
                read_opdump_header(ep).reference});
  return 0;
}

int cmd_bch(const Args& a) {
  if (a.positional.size() != 4)
    throw std::runtime_error("usage: refharness bch transform|product A.dump B.dump OUT.dump "
                             "[scale_a=S] [scale_b=S] [--factorized] "
                             "[--factorized-1b-intermediates=0|1] [--factorized-2b-intermediates=0|1]");
  const std::string& mode = a.positional[0];
  if (mode != "transform" && mode != "product")
    throw std::runtime_error("bch mode must be transform or product");
  const std::string& ap = a.positional[1];
  auto ms = modelspace_for_dump(ap, a);
  Operator A = read_opdump(ap, *ms);
  Operator B = read_opdump(a.positional[2], *ms);
  A = A * a.real("scale_a", 1.0);
  B = B * a.real("scale_b", 1.0);

  BCH::SetUseFactorizedCorrection(a.flag("factorized"));
  if (a.has("factorized-1b-intermediates"))
    FDC::SetUse_1b_Intermediates(a.flag("factorized-1b-intermediates"));
  if (a.has("factorized-2b-intermediates"))
    FDC::SetUse_2b_Intermediates(a.flag("factorized-2b-intermediates"));

  Operator Out = (mode == "transform") ? BCH::BCH_Transform(A, B) : BCH::BCH_Product(A, B);
  std::cout << "norm_a " << fmt17(A.Norm()) << "\nnorm_b " << fmt17(B.Norm()) << "\nnorm_out "
            << fmt17(Out.Norm()) << std::endl;
  write_opdump(a.positional[3], Out,
               {reference_version() + " refharness bch " + mode, read_opdump_header(ap).reference});
  return 0;
}

}  // namespace rh
