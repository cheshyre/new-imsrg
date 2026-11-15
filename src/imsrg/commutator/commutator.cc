// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/commutator/commutator.h"

#include <algorithm>
#include <string>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/commutator/detail/workspace_impl.h"

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
  if (X.hermiticity() == Hermiticity::kNonHermitian ||
      Y.hermiticity() == Hermiticity::kNonHermitian)
    throw NotImplementedError(
        "commutator: NonHermitian X or Y is not supported");
  IMSRG_CHECK(
      Z.hermiticity() == CommutatorSpec(X.spec(), Y.spec()).hermiticity,
      "commutator: Z's hermiticity must be that of [X, Y] (Hermitian if X and "
      "Y differ in "
      "hermiticity, anti-Hermitian otherwise)");
}

int CommutatorHermiticitySign(const op::Operator& X, const op::Operator& Y) {
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
  if (x.particle_rank > 2 || y.particle_rank > 2)
    throw NotImplementedError(
        "commutator: particle rank 3 is not supported (IMSRG(3), M10)");
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
  if (X.hermiticity() == Hermiticity::kNonHermitian ||
      Y.hermiticity() == Hermiticity::kNonHermitian)
    throw NotImplementedError(
        "commutator: NonHermitian X or Y is not supported");
  // The IMSRG(3) terms (N7) are not implemented yet: refuse three-body parts
  // instead of silently dropping them (the [2,2]->3 term alone is
  // commutator::Comm223, three_body.h).
  if (X.three_body().has_value() || Y.three_body().has_value() ||
      (!Z.empty() && Z.three_body().has_value()))
    throw NotImplementedError(
        "commutator: operands or outputs with a three-body part need the "
        "IMSRG(3) terms, which are not implemented yet (use "
        "commutator::Comm223 for the [2,2]->3 part)");

  const op::OperatorSpec spec = CommutatorSpec(X.spec(), Y.spec());
  if (!Z.empty() && Z.spec() == spec && ws.Compatible(Z)) {
    Z.set_zero();
  } else {
    Z = op::Operator(ws.model_space_ptr(), spec);
  }

  const TermMask m = opts.terms;
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
}

op::Operator Commutator(const op::Operator& X, const op::Operator& Y,
                        const CommutatorOptions& opts, Workspace& ws) {
  op::Operator Z;
  Commutator(X, Y, Z, opts, ws);
  return Z;
}

}  // namespace imsrg::commutator
