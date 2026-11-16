// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/commutator/imsrg3.h"

#include <algorithm>
#include <string>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/detail/n7_common.h"
#include "imsrg/commutator/detail/three_body_impl.h"

namespace imsrg::commutator {

namespace detail {

namespace {

void CheckOne(const char* term, const op::Operator& op, const char* name,
              const ThreeBodyContext& ctx) {
  const std::string who = std::string(term) + ": " + name;
  if (op.empty()) throw InputError(who + " is an empty operator");
  if (!op.spec().is_scalar() || !op.spec().is_number_conserving()) {
    throw NotImplementedError(who +
                              " must be a scalar, number-conserving operator");
  }
  const basis::ModelSpace& ms = op.model_space();
  const basis::ModelSpace& ms3 = *ctx.model_space_ptr();
  IMSRG_CHECK(&ms == &ms3 || (ms.basis() == ms3.basis() &&
                              ms.reference() == ms3.reference() &&
                              ms.partition() == ms3.partition()),
              who +
                  " lives on a model space whose basis, reference or "
                  "partition differs from the context's");
  const ThreeBodyContext::Impl& impl = ctx.impl();
  if (op.two_body().allocated()) {
    const auto& offsets = impl.two_body_offset_host;
    const auto& m = op.two_body().matrix();
    IMSRG_CHECK(m.num_blocks() + 1 == offsets.size() &&
                    m.total_size() == offsets.back(),
                who + ": two-body layout differs from the context's");
  }
  if (op.three_body().has_value()) {
    const op::ThreeBody& t = *op.three_body();
    const std::vector<std::int64_t>& expected =
        t.packed() ? impl.packed_offset_host : impl.dense_offset_host;
    IMSRG_CHECK(std::equal(t.block_offsets().begin(), t.block_offsets().end(),
                           expected.begin(), expected.end()),
                who + ": three-body layout differs from the context's");
  }
}

}  // namespace

N7Views PrepareN7(const char* term, const op::Operator& X,
                  const op::Operator& Y, const op::Operator& Z,
                  const ThreeBodyContext& ctx) {
  CheckOne(term, X, "X", ctx);
  CheckOne(term, Y, "Y", ctx);
  CheckOne(term, Z, "Z", ctx);
  IMSRG_CHECK(&Z != &X && &Z != &Y,
              std::string(term) + ": Z must not alias X or Y");
  IMSRG_CHECK(
      Z.hermiticity() == CommutatorSpec(X.spec(), Y.spec()).hermiticity,
      std::string(term) +
          ": Z's hermiticity must be that of [X, Y] (Hermitian if X and Y "
          "differ in hermiticity, anti-Hermitian if they agree, general if "
          "either is general)");
  return ctx.impl().N7(&X, &Y);
}

ThreeBodyOutView ThreeBodyOut(const char* term, op::Operator& Z,
                              const ThreeBodyContext& ctx) {
  IMSRG_CHECK(Z.three_body().has_value(),
              std::string(term) + ": Z has no three-body part");
  op::ThreeBody& t = *Z.three_body();
  const ThreeBodyContext::Impl& impl = ctx.impl();
  const std::vector<std::int64_t>& expected =
      t.packed() ? impl.packed_offset_host : impl.dense_offset_host;
  IMSRG_CHECK(
      std::equal(t.block_offsets().begin(), t.block_offsets().end(),
                 expected.begin(), expected.end()),
      std::string(term) + ": Z's three-body layout differs from the context's");
  ThreeBodyOutView v;
  v.data = t.device_data();
  v.packed = t.packed() ? 1 : 0;
  v.zero_diagonal = t.hermiticity() == Hermiticity::kAntiHermitian ? 1 : 0;
  return v;
}

TwoBodyOutView TwoBodyOut(const char* term, op::Operator& Z,
                          const ThreeBodyContext& ctx) {
  IMSRG_CHECK(Z.two_body().allocated(),
              std::string(term) + ": Z has no two-body part");
  const auto& offsets = ctx.impl().two_body_offset_host;
  const auto& m = Z.two_body().matrix();
  IMSRG_CHECK(
      m.num_blocks() + 1 == offsets.size() && m.total_size() == offsets.back(),
      std::string(term) + ": Z's two-body layout differs from the context's");
  TwoBodyOutView v;
  v.data = Z.two_body().matrix().buffer().device_data();
  v.block_offset = ctx.impl().two_body_offset.device_data();
  return v;
}

double* OneBodyOut(op::Operator& Z) {
  return Z.one_body().buffer().device_data();
}

}  // namespace detail

namespace terms {

void Comm223ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx) {
  (void)detail::PrepareN7("Comm223ss", X, Y, Z, ctx);
  (void)detail::ThreeBodyOut("Comm223ss", Z, ctx);
  if (!X.two_body().allocated() || !Y.two_body().allocated()) return;
  // In place: no temporary three-body part (the largest object of an
  // IMSRG(3) flow at the production sizes).
  detail::AddComm223(X, Y, *Z.three_body(), ctx);
}

}  // namespace terms

}  // namespace imsrg::commutator
