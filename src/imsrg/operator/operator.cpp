#include "imsrg/operator/operator.hpp"

#include <utility>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/error.hpp"

namespace imsrg::op {

Operator::Operator(std::shared_ptr<const basis::ModelSpace> ms, const OperatorSpec& spec)
    : ms_(std::move(ms)), spec_(spec) {
  validate(spec_);
  IMSRG_CHECK(ms_ != nullptr);
  one_body_ = OneBody(ms_->num_orbits(), spec_.hermiticity);
  two_body_ = TwoBody(ms_, spec_);
}

void Operator::set_hermiticity(Hermiticity h) noexcept {
  spec_.hermiticity = h;
  one_body_.set_hermiticity(h);
  two_body_.set_hermiticity(h);
}

const basis::ModelSpace& Operator::model_space() const {
  IMSRG_CHECK(ms_ != nullptr);
  return *ms_;
}

void Operator::set_zero() {
  zero_body_ = 0.0;
  one_body_.set_zero();
  two_body_.set_zero();
}

void Operator::sync_to_host() {
  one_body_.sync_to_host();
  two_body_.sync_to_host();
}

void Operator::sync_to_device() {
  one_body_.sync_to_device();
  two_body_.sync_to_device();
}

bool same_model_space(const Operator& a, const Operator& b) noexcept {
  const auto& pa = a.model_space_ptr();
  const auto& pb = b.model_space_ptr();
  if (!pa || !pb) return false;
  return pa == pb || *pa == *pb;
}

}  // namespace imsrg::op
