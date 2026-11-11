// rh_commutator.cpp -- term-by-term commutator and generator evaluation with the OLD imsrg++.
//
//   refharness commutator --list
//   refharness commutator TERM X.dump Y.dump Z.dump [valence_space=...]
//       Z = [X, Y] restricted to one named IMSRG(2) term function (or `all` = the full
//       Commutator::Commutator with IMSRG(3) off). Z is created exactly as in
//       Commutator::CommutatorScalarScalar (particle rank 2, hermiticity from X and Y,
//       not reduced) and written as an OperatorDump.
//   refharness generator TYPE H.dump ETA.dump [valence_space=...]
//       eta = imsrg++ generator of TYPE applied to H (single reference, default settings).

#include <fstream>
#include <functional>
#include <iostream>

#include "IMSRG.hh"
#include "rh_common.hh"

namespace rh {

namespace {

using TermFn = std::function<void(const Operator&, const Operator&, Operator&)>;

// Order is the order of calls in Commutator::CommutatorScalarScalar.
const std::vector<std::pair<std::string, TermFn>>& term_table() {
  static const std::vector<std::pair<std::string, TermFn>> t = {
      {"comm110ss", Commutator::comm110ss},
      {"comm220ss", Commutator::comm220ss},
      {"comm111ss", Commutator::comm111ss},
      {"comm121ss", Commutator::comm121ss},
      {"comm221ss", Commutator::comm221ss},
      {"comm122ss", Commutator::comm122ss},
      {"comm222_pp_hhss", Commutator::comm222_pp_hhss},
      {"comm222_phss", Commutator::comm222_phss},
      {"comm222_pp_hh_221ss", Commutator::comm222_pp_hh_221ss},
  };
  return t;
}

std::string term_list() {
  std::string s;
  for (auto& t : term_table()) s += t.first + " ";
  return s + "all";
}

// Same construction as modelspace_for_dump in rh_opdump.cpp (file-local there).
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

int cmd_commutator(const Args& a) {
  if (a.has("list")) {
    for (auto& t : term_table()) std::cout << t.first << "\n";
    std::cout << "all\n";
    return 0;
  }
  if (a.positional.size() != 4)
    throw std::runtime_error("usage: refharness commutator TERM X.dump Y.dump Z.dump | --list");
  const std::string& term = a.positional[0];
  const std::string& xp = a.positional[1];
  const std::string& yp = a.positional[2];
  const std::string& zp = a.positional[3];
  const TermFn* fn = nullptr;
  for (auto& t : term_table())
    if (t.first == term) fn = &t.second;
  if (!fn && term != "all")
    throw std::runtime_error("unknown commutator term '" + term + "'; valid: " + term_list());

  auto ms = modelspace_for(xp, a);
  Operator X = read_opdump(xp, *ms);
  Operator Y = read_opdump(yp, *ms);
  if (X.GetJRank() != 0 || Y.GetJRank() != 0)
    throw std::runtime_error("only scalar X and Y are supported");

  Operator Z;
  if (term == "all") {
    Commutator::SetUseIMSRG3(false);
    Z = Commutator::Commutator(X, Y);
  } else {
    // as in Commutator::CommutatorScalarScalar (Commutator.cc:322-355), IMSRG(2)
    Z = Operator(*ms, 0, X.GetTRank() + Y.GetTRank(), (X.GetParity() + Y.GetParity()) % 2, 2);
    if ((X.IsHermitian() && Y.IsHermitian()) || (X.IsAntiHermitian() && Y.IsAntiHermitian()))
      Z.SetAntiHermitian();
    else if ((X.IsHermitian() && Y.IsAntiHermitian()) || (X.IsAntiHermitian() && Y.IsHermitian()))
      Z.SetHermitian();
    else
      Z.SetNonHermitian();
    if (Z.IsReduced()) Z.MakeNotReduced();
    (*fn)(X, Y, Z);
  }
  write_opdump(zp, Z,
               {reference_version() + " refharness commutator " + term, read_opdump_header(xp).reference});
  return 0;
}

int cmd_generator(const Args& a) {
  if (a.positional.size() != 3)
    throw std::runtime_error("usage: refharness generator TYPE H.dump ETA.dump");
  const std::string& type = a.positional[0];
  const std::string& hp = a.positional[1];
  auto ms = modelspace_for(hp, a);
  Operator H = read_opdump(hp, *ms);

  // IMSRGSolver ctor: Eta = H, Erase, SetAntiHermitian; defaults: Epstein_Nesbet, cutoff 1e-6.
  Operator Eta(H);
  Eta.Erase();
  Eta.SetAntiHermitian();
  Generator gen;
  gen.SetType(type);
  gen.SetDenominatorPartitioning(a.str("denominator_partitioning", "Epstein_Nesbet"));
  if (a.has("denominator_cutoff")) gen.SetDenominatorCutoff(a.real("denominator_cutoff", 1e-6));
  gen.Update(H, Eta);
  write_opdump(a.positional[2], Eta,
               {reference_version() + " refharness generator " + type, read_opdump_header(hp).reference});
  return 0;
}

}  // namespace rh
