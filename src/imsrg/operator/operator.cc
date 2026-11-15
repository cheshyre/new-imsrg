// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/operator/operator.h"

#include <utility>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"

namespace imsrg::op {

Operator::Operator(std::shared_ptr<const basis::ModelSpace> ms,
                   const OperatorSpec& spec)
    : ms_(std::move(ms)), spec_(spec) {
  Validate(spec_);
  IMSRG_CHECK(ms_ != nullptr);
  one_body_ = OneBody(ms_->num_orbits(), spec_.hermiticity);
  two_body_ = TwoBody(ms_, spec_);
  if (spec_.particle_rank == 3) {
    three_body_.emplace(ms_, spec_.hermiticity);
  }
}

void Operator::set_hermiticity(Hermiticity h) {
  spec_.hermiticity = h;
  one_body_.set_hermiticity(h);
  two_body_.set_hermiticity(h);
  if (three_body_) three_body_->set_hermiticity(h);
}

const basis::ModelSpace& Operator::model_space() const {
  IMSRG_CHECK(ms_ != nullptr);
  return *ms_;
}

void Operator::set_zero() {
  zero_body_ = 0.0;
  one_body_.set_zero();
  two_body_.set_zero();
  if (three_body_) three_body_->set_zero();
}

void Operator::SyncToHost() const {
  one_body_.SyncToHost();
  two_body_.SyncToHost();
  if (three_body_) three_body_->SyncToHost();
}

void Operator::SyncToDevice() {
  one_body_.SyncToDevice();
  two_body_.SyncToDevice();
  if (three_body_) three_body_->SyncToDevice();
}

bool SameModelSpace(const Operator& a, const Operator& b) noexcept {
  const auto& pa = a.model_space_ptr();
  const auto& pb = b.model_space_ptr();
  if (!pa || !pb) return false;
  return pa == pb || *pa == *pb;
}

}  // namespace imsrg::op
