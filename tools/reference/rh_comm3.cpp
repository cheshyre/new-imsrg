// rh_comm3.cpp -- IMSRG(3)-N7 commutator terms and the 3B generator of the OLD imsrg++
// (M10 step 3, design/14-imsrg3-plan.md section 4 item 2), for scripted goldens (D42).
//
//   refharness comm3 --list
//   refharness comm3 TERM X.dump Y.dump Z.dump [e3max=M] [valence_space=...]
//       X, Y: scalar OperatorDumps, particle rank 3 or 2 (3B section when rank 3).
//       TERM: comm330ss comm331ss comm231ss comm132ss comm232ss comm133ss comm223ss (one
//         Commutator:: function each, called on a fresh Z), or `n7` (the full
//         Commutator::Commutator(X, Y) after SetUseIMSRG3(true), SetUseIMSRG3N7(true): the
//         IMSRG(2) terms and the seven N7 terms in imsrg++'s call order).
//       Z is created exactly as in Commutator::CommutatorScalarScalar with IMSRG(3) on: particle
//         rank 3, hermiticity from X and Y, not reduced, ThreeBody.SetMode("pn"); it is written
//         with the three_body section. The norms of Z are printed to stdout as JSON.
//       e3max defaults to min(12, 3 emax), as for the other subcommands.
//   refharness generator3 white H.dump ETA.dump [e3max=M] [valence_space=...]
//       eta = imsrg++ Generator (type `white`, single reference, default settings) applied to a
//       particle-rank-3 H; Eta has rank 3 so ConstructGenerator_SingleRef_3body runs
//       (Generator.cc:328-427). Written with the three_body section; norms printed as JSON.

#include <fstream>
#include <functional>
#include <iostream>

#include "IMSRG.hh"
#include "rh_common.hh"

namespace rh {

namespace {

using TermFn = std::function<void(const Operator&, const Operator&, Operator&)>;

// Order is the order of calls in Commutator::CommutatorScalarScalar (Commutator.cc:396-460).
const std::vector<std::pair<std::string, TermFn>>& term_table() {
  static const std::vector<std::pair<std::string, TermFn>> t = {
      {"comm133ss", Commutator::comm133ss}, {"comm330ss", Commutator::comm330ss},
      {"comm223ss", Commutator::comm223ss}, {"comm232ss", Commutator::comm232ss},
      {"comm331ss", Commutator::comm331ss}, {"comm231ss", Commutator::comm231ss},
      {"comm132ss", Commutator::comm132ss},
  };
  return t;
}

std::string term_list() {
  std::string s;
  for (auto& t : term_table()) s += t.first + " ";
  return s + "n7";
}

std::unique_ptr<ModelSpace> modelspace_for(const std::string& path, const Args& a) {
  DumpHeader h = read_opdump_header(path);
  SpaceParams p;
  p.emax = h.emax;
  p.hw = h.hw;
  p.reference = h.reference;
  p.valence_space = a.str("valence_space", h.reference);
  p.e3max = a.integer("e3max", std::min(12, 3 * h.emax));
  return make_modelspace(p);
}

}  // namespace

int cmd_comm3(const Args& a) {
  if (a.has("list")) {
    for (auto& t : term_table()) std::cout << t.first << "\n";
    std::cout << "n7\n";
    return 0;
  }
  if (a.positional.size() != 4)
    throw std::runtime_error("usage: refharness comm3 TERM X.dump Y.dump Z.dump [e3max=M] | --list");
  const std::string& term = a.positional[0];
  const std::string& xp = a.positional[1];
  const std::string& yp = a.positional[2];
  const std::string& zp = a.positional[3];
  const TermFn* fn = nullptr;
  for (auto& t : term_table())
    if (t.first == term) fn = &t.second;
  if (!fn && term != "n7")
    throw std::runtime_error("unknown comm3 term '" + term + "'; valid: " + term_list());

  auto ms = modelspace_for(xp, a);
  // comm331ss and comm231ss read the 6j table inside parallel regions and die ("DANGER ...
  // Updating SixJList inside a parallel loop") unless it is filled first (comm223ss fills it itself).
  ms->PreCalculateSixJ();
  Operator X = read_opdump(xp, *ms);
  Operator Y = read_opdump(yp, *ms);
  if (X.GetJRank() != 0 || Y.GetJRank() != 0)
    throw std::runtime_error("only scalar X and Y are supported");
  if (X.IsReduced()) X.MakeNotReduced();
  if (Y.IsReduced()) Y.MakeNotReduced();

  Operator Z;
  if (term == "n7") {
    Commutator::SetUseIMSRG3(true);
    Commutator::SetUseIMSRG3N7(true);
    Z = Commutator::Commutator(X, Y);
  } else {
    // as in Commutator::CommutatorScalarScalar (Commutator.cc:322-355) with IMSRG(3) on
    Z = Operator(*ms, 0, X.GetTRank() + Y.GetTRank(), (X.GetParity() + Y.GetParity()) % 2, 3);
    if ((X.IsHermitian() && Y.IsHermitian()) || (X.IsAntiHermitian() && Y.IsAntiHermitian()))
      Z.SetAntiHermitian();
    else if ((X.IsHermitian() && Y.IsAntiHermitian()) || (X.IsAntiHermitian() && Y.IsHermitian()))
      Z.SetHermitian();
    else
      Z.SetNonHermitian();
    Z.ThreeBody.SetMode("pn");
    if (Z.IsReduced()) Z.MakeNotReduced();
    (*fn)(X, Y, Z);
  }
  write_opdump(zp, Z,
               {reference_version() + " refharness comm3 " + term, read_opdump_header(xp).reference},
               true);
  std::cout << norms_json(Z) << "\n";
  return 0;
}

int cmd_generator3(const Args& a) {
  if (a.positional.size() != 3)
    throw std::runtime_error("usage: refharness generator3 white H.dump ETA.dump [e3max=M]");
  const std::string& type = a.positional[0];
  const std::string& hp = a.positional[1];
  if (type != "white") throw std::runtime_error("generator3 supports only type 'white'");
  auto ms = modelspace_for(hp, a);
  Operator H = read_opdump(hp, *ms);
  if (H.GetParticleRank() < 3) throw std::runtime_error("generator3 needs a particle-rank-3 H");

  // IMSRGSolver ctor: Eta = H, Erase, SetAntiHermitian; defaults: Epstein_Nesbet, cutoff 1e-6.
  Operator Eta(H);
  Eta.Erase();
  Eta.SetAntiHermitian();
  Generator gen;
  gen.SetType(type);
  gen.SetDenominatorPartitioning("Epstein_Nesbet");
  gen.Update(H, Eta);
  write_opdump(a.positional[2], Eta,
               {reference_version() + " refharness generator3 " + type, read_opdump_header(hp).reference},
               true);
  std::cout << norms_json(Eta) << "\n";
  return 0;
}

}  // namespace rh
