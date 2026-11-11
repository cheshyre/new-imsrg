// rh_operator.cpp -- golden data for normal ordering and basis transformation (M4 operator).
//
//   refharness normal-order IN.dump OUT.dump [reference=NAME] [undo=true]
//   refharness basis-transform IN.dump C.txt OUT.dump
//
// Both read an OperatorDump v1 into the old imsrg++ Operator, apply the old code's
// Operator::DoNormalOrdering / UndoNormalOrdering (Operator.cc:380-398, 477-571) or
// HartreeFock::TransformToHFBasis (HartreeFock.cc:914-1000), and write an OperatorDump again.

#include <algorithm>
#include <fstream>
#include <iostream>

#include "HartreeFock.hh"
#include "IMSRG.hh"
#include "rh_common.hh"

namespace rh {

namespace {

std::string basename_of(const std::string& p) {
  auto k = p.find_last_of('/');
  return k == std::string::npos ? p : p.substr(k + 1);
}

}  // namespace

/// Re-normal-orders the operator of IN with respect to the reference NAME (default: the dump's
/// reference; same emax/hw/e3max) and writes it to OUT. The input is copied into an Operator of
/// the NAME model space (ZeroBody, OneBody, TwoBody matrices, hermiticity), because read_opdump
/// requires occupations equal to the model space. The two-body part is unchanged by
/// normal ordering at the 2B level; ZeroBody and OneBody change. undo=true calls
/// UndoNormalOrdering (sign -1).
int cmd_normal_order(const Args& a) {
  if (a.positional.size() != 2)
    throw std::runtime_error("usage: refharness normal-order IN OUT [reference=NAME] [undo=true]");
  const std::string& in = a.positional[0];
  auto ms_in = modelspace_for_dump(in, a);
  Operator op_in = read_opdump(in, *ms_in);
  DumpHeader h = read_opdump_header(in);
  const std::string name = a.str("reference", h.reference);

  SpaceParams p;
  p.emax = h.emax;
  p.hw = h.hw;
  p.e3max = a.integer("e3max", std::min(12, 3 * h.emax));
  p.reference = name;
  p.valence_space = name;
  auto ms = make_modelspace(p);
  if (ms->GetNumberOrbits() != ms_in->GetNumberOrbits())
    throw std::runtime_error("orbit count differs between the input and the target model space");

  Operator op(*ms, op_in.GetJRank(), op_in.GetTRank(), op_in.GetParity(), 2);
  if (op_in.IsHermitian()) op.SetHermitian();
  else if (op_in.IsAntiHermitian()) op.SetAntiHermitian();
  else op.SetNonHermitian();
  op.ZeroBody = op_in.ZeroBody;
  op.OneBody = op_in.OneBody;
  for (auto& it : op_in.TwoBody.MatEl) {
    auto jt = op.TwoBody.MatEl.find(it.first);
    if (jt == op.TwoBody.MatEl.end())
      throw std::runtime_error("two-body channel pair missing in the target operator");
    jt->second = it.second;
  }

  const bool undo = a.flag("undo");
  Operator out = undo ? op.UndoNormalOrdering() : op.DoNormalOrdering();
  std::string src = reference_version() + "; refharness normal-order " + basename_of(in) +
                    " reference=" + name + (undo ? " undo=true" : "");
  write_opdump(a.positional[1], out, {src, name});
  return 0;
}

/// Reads C (norb x norb, whitespace separated, row-major; row = old orbit, column = new orbit)
/// and writes HartreeFock::TransformToHFBasis(op). The HartreeFock constructor only needs the
/// operator's one-body part and the model space, so it is used as is (C is a public member).
int cmd_basis_transform(const Args& a) {
  if (a.positional.size() != 3)
    throw std::runtime_error("usage: refharness basis-transform IN C.txt OUT");
  const std::string& in = a.positional[0];
  auto ms = modelspace_for_dump(in, a);
  Operator op = read_opdump(in, *ms);
  const size_t n = ms->GetNumberOrbits();
  arma::mat C(n, n);
  {
    std::ifstream f(a.positional[1]);
    if (!f) throw std::runtime_error("cannot open " + a.positional[1]);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
        if (!(f >> C(i, j))) throw std::runtime_error("C.txt: too few numbers");
    double extra;
    if (f >> extra) throw std::runtime_error("C.txt: too many numbers");
  }
  HartreeFock hf(op);
  hf.C = C;
  Operator out = hf.TransformToHFBasis(op);
  std::string src = reference_version() + "; refharness basis-transform " + basename_of(in) +
                    " " + basename_of(a.positional[1]);
  write_opdump(a.positional[2], out, {src, read_opdump_header(in).reference});
  return 0;
}

}  // namespace rh
