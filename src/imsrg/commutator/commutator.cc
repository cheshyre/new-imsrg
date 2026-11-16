// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/commutator/commutator.h"

#include <algorithm>
#include <string>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/comm/communicator.h"
#include "imsrg/commutator/detail/work_items.h"
#include "imsrg/commutator/detail/workspace_impl.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/exec/device_buffer.h"
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

namespace {

/// Throws unless the IMSRG(3) terms `m3` (empty for a pure IMSRG(2) set) and
/// the operands' three-body parts are supported by the workspace `ws`.
void CheckTermSet(const op::Operator& X, const op::Operator& Y,
                  const TermMask m3, const Workspace& ws) {
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
}

/// Throws unless the three-body parts involved are replicated: the phase-1
/// commutator runs the N7 terms on rank 0 with whole three-body operators and
/// broadcasts Z3 assuming equal buffer sizes, so a model space with a
/// three-body row distribution would hang or corrupt (D7; supported by the
/// distributed commutator of D10/D11). The condition depends only on the
/// replicated model space and the operands' structure, hence every rank
/// throws together.
void CheckThreeBodyReplicated(const op::Operator& X, const op::Operator& Y,
                              const op::Operator& Z, const TermMask m3,
                              const Workspace& ws) {
  if (ws.model_space().three_body_distribution() == nullptr) return;
  if (m3.empty() && !X.three_body().has_value() &&
      !Y.three_body().has_value() && !Z.three_body().has_value())
    return;
  throw NotImplementedError(
      "commutator: three-body parts on a model space with a distributed "
      "three-body row partition are not supported until the distributed "
      "commutator (tasks D10/D11); use a replicated model space");
}

/// Sets Z to zero with the spec of [X, Y] (rank 3 if `imsrg3`), reusing its
/// storage when it already has that spec on the workspace's model space.
void PrepareOutput(const op::Operator& X, const op::Operator& Y, bool imsrg3,
                   const Workspace& ws, op::Operator& Z) {
  op::OperatorSpec spec = CommutatorSpec(X.spec(), Y.spec());
  spec.particle_rank = imsrg3 ? 3 : spec.particle_rank;
  if (!Z.empty() && Z.spec() == spec && ws.Compatible(Z)) {
    Z.set_zero();
  } else {
    Z = op::Operator(ws.model_space_ptr(), spec);
  }
}

/// True on the part (rank) that evaluates the terms which are not split by
/// the work assignment: part 0, or the only part of a serial workspace.
bool IsFirstPart(const Workspace& ws) { return ws.impl().work.part() == 0; }

/// Adds the IMSRG(2) terms selected by `m` to Z (Z0, Z1 and Z2 only). On a
/// distributed workspace this is the part's share (D14 phase 1): the cheap
/// terms 110, 220, 111 and 121 run on part 0 only (every part would add them
/// once to the combined sum otherwise), the expensive ones over the part's
/// work items (detail/work_items.h).
void TwoBodyTerms(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                  const TermMask m, Workspace& ws) {
  if (IsFirstPart(ws)) {
    if (m.test(Term::kComm110ss)) terms::Comm110ss(X, Y, Z, ws);
    if (m.test(Term::kComm220ss)) terms::Comm220ss(X, Y, Z, ws);
    if (m.test(Term::kComm111ss)) terms::Comm111ss(X, Y, Z, ws);
    if (m.test(Term::kComm121ss)) terms::Comm121ss(X, Y, Z, ws);
  }
  if (m.test(Term::kComm122ss)) terms::Comm122ss(X, Y, Z, ws);
  if (m.test(Term::kComm222PpHhss) && m.test(Term::kComm221ss)) {
    terms::Comm222PpHh221ss(X, Y, Z, ws);
  } else {
    if (m.test(Term::kComm221ss)) terms::Comm221ss(X, Y, Z, ws);
    if (m.test(Term::kComm222PpHhss)) terms::Comm222PpHhss(X, Y, Z, ws);
  }
  if (m.test(Term::kComm222Phss)) terms::Comm222Phss(X, Y, Z, ws);
}

/// True if the IMSRG(3) block runs. imsrg++ evaluates it iff ||X|| >
/// threshold and ||Y|| > threshold (Commutator.cc:396); with the default
/// threshold 0 that only skips an exactly zero operand, whose terms vanish
/// anyway. The norms are computed only for a positive threshold.
bool ThreeBodyGateOpen(const op::Operator& X, const op::Operator& Y,
                       const double threshold) {
  return !(threshold > 0.0 &&
           (op::Norm(X) <= threshold || op::Norm(Y) <= threshold));
}

/// Adds the IMSRG(3) terms selected by `m` to Z, in imsrg++'s order
/// (Commutator.cc:403-430). Precondition: the workspace has a three-body
/// context (checked by CheckTermSet).
void ThreeBodyTerms(const op::Operator& X, const op::Operator& Y,
                    op::Operator& Z, const TermMask m, const Workspace& ws) {
  const ThreeBodyContext& ctx = *ws.three_body_context();
  if (m.test(Term::kComm133ss)) terms::Comm133ss(X, Y, Z, ctx);
  if (m.test(Term::kComm330ss)) terms::Comm330ss(X, Y, Z, ctx);
  if (m.test(Term::kComm223ss)) terms::Comm223ss(X, Y, Z, ctx);
  if (m.test(Term::kComm232ss)) terms::Comm232ss(X, Y, Z, ctx);
  if (m.test(Term::kComm331ss)) terms::Comm331ss(X, Y, Z, ctx);
  if (m.test(Term::kComm231ss)) terms::Comm231ss(X, Y, Z, ctx);
  if (m.test(Term::kComm132ss)) terms::Comm132ss(X, Y, Z, ctx);
}

}  // namespace

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
  CheckThreeBodyReplicated(X, Y, Z, m3, ws);
  CheckTermSet(X, Y, m3, ws);
  PrepareOutput(X, Y, imsrg3, ws, Z);

  TwoBodyTerms(X, Y, Z, m, ws);
  // Every part evaluates the gate (identical operands give identical norms);
  // the N7 terms run on part 0 only until the three-body storage is
  // distributed (M12b phase 1, M12b-Q5): CombinePartials sums their Z0-Z2
  // contributions with the others and broadcasts Z3 from rank 0.
  if (imsrg3 && ThreeBodyGateOpen(X, Y, opts.threebody_threshold) &&
      IsFirstPart(ws))
    ThreeBodyTerms(X, Y, Z, m, ws);
  // Single exit (M12b review note m2): every call, IMSRG(2)-only or gated,
  // reaches the combine.
  detail::CombinePartials(Z, ws);
}

namespace detail {

void CombinePartials(op::Operator& Z, const Workspace& ws) {
  const comm::Communicator& comm = ws.comm();
  if (comm.size() == 1) return;
  const WorkAssignment& work = ws.impl().work;
  IMSRG_CHECK(work.num_parts() == comm.size() && work.part() == comm.rank(),
              "commutator: the work assignment does not match the "
              "workspace's communicator");
  Z.SyncToHost();
  Z.zero_body() = comm.ReduceBcastSum(Z.zero_body());
  for (exec::DeviceBuffer<double>* b :
       {&Z.one_body().buffer(), &Z.two_body().matrix().buffer()}) {
    if (!b->empty()) comm.ReduceBcastSum(b->host_span());
  }
  if (Z.three_body().has_value() && !Z.three_body()->buffer().empty())
    comm.Broadcast(Z.three_body()->buffer().host_span(), 0);
  Z.SyncToDevice();
}

}  // namespace detail

op::Operator Commutator(const op::Operator& X, const op::Operator& Y,
                        const CommutatorOptions& opts, Workspace& ws) {
  op::Operator Z;
  Commutator(X, Y, Z, opts, ws);
  return Z;
}

}  // namespace imsrg::commutator
