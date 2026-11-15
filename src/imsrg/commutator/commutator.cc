// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/commutator/commutator.h"

#include <algorithm>
#include <string>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/commutator/detail/workspace_impl.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/operator/norms.h"

namespace imsrg::commutator {

namespace detail {

void CheckTermInputs(const op::Operator& X, const op::Operator& Y,
                     const op::Operator& Z, const Workspace& ws) {
  IMSRG_CHECK(ws.Compatible(X),
              "commutator: X is not on the workspace's model space");
  IMSRG_CHECK(ws.Compatible(Y),
              "commutator: Y is not on the workspace's model space");
  IMSRG_CHECK(ws.Compatible(Z),
              "commutator: Z is not on the workspace's model space");
  IMSRG_CHECK(&Z != &X && &Z != &Y, "commutator: Z must not alias X or Y");
  IMSRG_CHECK(
      Z.hermiticity() == CommutatorSpec(X.spec(), Y.spec()).hermiticity,
      "commutator: Z's hermiticity must be that of [X, Y] (Hermitian if X and "
      "Y differ in hermiticity, anti-Hermitian if they agree, general if "
      "either is general)");
}

bool IsGeneral(const op::Operator& X, const op::Operator& Y) {
  return X.hermiticity() == Hermiticity::kNonHermitian ||
         Y.hermiticity() == Hermiticity::kNonHermitian;
}

int CommutatorHermiticitySign(const op::Operator& X, const op::Operator& Y) {
  IMSRG_ASSERT(!IsGeneral(X, Y));
  return -op::HermiticitySign(X.hermiticity()) *
         op::HermiticitySign(Y.hermiticity());
}

}  // namespace detail

op::OperatorSpec CommutatorSpec(const op::OperatorSpec& x,
                                const op::OperatorSpec& y) {
  if (!x.is_scalar() || !y.is_scalar())
    throw NotImplementedError(
        "commutator: only scalar operators are supported");
  if (!x.is_number_conserving() || !y.is_number_conserving())
    throw NotImplementedError(
        "commutator: only number-conserving operators are supported");
  op::OperatorSpec z =
      op::OperatorSpec::Scalar(std::max(x.particle_rank, y.particle_rank));
  if (x.hermiticity == Hermiticity::kNonHermitian ||
      y.hermiticity == Hermiticity::kNonHermitian) {
    z.hermiticity = Hermiticity::kNonHermitian;
  } else {
    z.hermiticity = (x.hermiticity == y.hermiticity)
                        ? Hermiticity::kAntiHermitian
                        : Hermiticity::kHermitian;
  }
  return z;
}

void Commutator(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                const CommutatorOptions& opts, Workspace& ws) {
  IMSRG_CHECK(ws.Compatible(X),
              "commutator: X is not on the workspace's model space");
  IMSRG_CHECK(ws.Compatible(Y),
              "commutator: Y is not on the workspace's model space");
  IMSRG_CHECK(&Z != &X && &Z != &Y, "commutator: Z must not alias X or Y");

  const TermMask m = opts.terms;
  const TermMask m3 = m.ThreeBodyTerms();
  const bool imsrg3 = !m3.empty();
  if (imsrg3) {
    const TermMask n7_3b = TermMask::Imsrg3N7().ThreeBodyTerms();
    if ((m3 | n7_3b) != n7_3b)
      throw NotImplementedError(
          "commutator: only the IMSRG(2) and the N7 IMSRG(3) terms are "
          "implemented (the N8/N9 terms of the full IMSRG(3) are not)");
    if (ws.three_body_context() == nullptr)
      throw InputError(
          "commutator: the IMSRG(3) terms need a workspace on a model space "
          "with a three-body space (ModelSpace::WithThreeBody)");
  } else if (X.three_body().has_value() || Y.three_body().has_value()) {
    // Refuse three-body operands instead of silently dropping them.
    throw NotImplementedError(
        "commutator: operands with a three-body part need an IMSRG(3) term "
        "set (e.g. TermMask::Imsrg3N7()); the [2,2]->3 part alone is "
        "commutator::Comm223");
  }

  op::OperatorSpec spec = CommutatorSpec(X.spec(), Y.spec());
  spec.particle_rank = imsrg3 ? 3 : spec.particle_rank;
  if (!Z.empty() && Z.spec() == spec && ws.Compatible(Z)) {
    Z.set_zero();
  } else {
    Z = op::Operator(ws.model_space_ptr(), spec);
  }

  if (m.test(Term::kComm110ss)) terms::Comm110ss(X, Y, Z, ws);
  if (m.test(Term::kComm220ss)) terms::Comm220ss(X, Y, Z, ws);
  if (m.test(Term::kComm111ss)) terms::Comm111ss(X, Y, Z, ws);
  if (m.test(Term::kComm121ss)) terms::Comm121ss(X, Y, Z, ws);
  if (m.test(Term::kComm122ss)) terms::Comm122ss(X, Y, Z, ws);
  if (m.test(Term::kComm222PpHhss) && m.test(Term::kComm221ss)) {
    terms::Comm222PpHh221ss(X, Y, Z, ws);
  } else {
    if (m.test(Term::kComm221ss)) terms::Comm221ss(X, Y, Z, ws);
    if (m.test(Term::kComm222PpHhss)) terms::Comm222PpHhss(X, Y, Z, ws);
  }
  if (m.test(Term::kComm222Phss)) terms::Comm222Phss(X, Y, Z, ws);

  if (!imsrg3) return;
  // imsrg++ evaluates the IMSRG(3) block iff ||X|| > threshold and ||Y|| >
  // threshold (Commutator.cc:396); with the default threshold 0 that only
  // skips an exactly zero operand, whose terms vanish anyway.
  if (opts.threebody_threshold > 0.0 &&
      (op::Norm(X) <= opts.threebody_threshold ||
       op::Norm(Y) <= opts.threebody_threshold))
    return;
  const ThreeBodyContext& ctx = *ws.three_body_context();
  // imsrg++'s order (Commutator.cc:403-430).
  if (m.test(Term::kComm133ss)) terms::Comm133ss(X, Y, Z, ctx);
  if (m.test(Term::kComm330ss)) terms::Comm330ss(X, Y, Z, ctx);
  if (m.test(Term::kComm223ss)) terms::Comm223ss(X, Y, Z, ctx);
  if (m.test(Term::kComm232ss)) terms::Comm232ss(X, Y, Z, ctx);
  if (m.test(Term::kComm331ss)) terms::Comm331ss(X, Y, Z, ctx);
  if (m.test(Term::kComm231ss)) terms::Comm231ss(X, Y, Z, ctx);
  if (m.test(Term::kComm132ss)) terms::Comm132ss(X, Y, Z, ctx);
}

op::Operator Commutator(const op::Operator& X, const op::Operator& Y,
                        const CommutatorOptions& opts, Workspace& ws) {
  op::Operator Z;
  Commutator(X, Y, Z, opts, ws);
  return Z;
}

}  // namespace imsrg::commutator
