#include "imsrg/operator/one_body.hpp"

#include <cstddef>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/error.hpp"
#include "imsrg/operator/operator_spec.hpp"

namespace imsrg::op {

namespace {
std::size_t flat(std::int32_t n, OrbitIndex a, OrbitIndex b) {
  IMSRG_ASSERT(a.value() >= 0 && a.value() < n && b.value() >= 0 && b.value() < n);
  return a.to_size() + b.to_size() * static_cast<std::size_t>(n);
}
}  // namespace

OneBody::OneBody(std::int32_t num_orbits, Hermiticity hermiticity)
    : n_(num_orbits), hermiticity_(hermiticity) {
  IMSRG_CHECK(num_orbits >= 0);
  data_ = exec::DeviceBuffer<double>(
      static_cast<std::size_t>(num_orbits) * static_cast<std::size_t>(num_orbits),
      "imsrg::op::OneBody");
}

double OneBody::get(OrbitIndex a, OrbitIndex b) const { return data_.host_span()[flat(n_, a, b)]; }

void OneBody::set_element(OrbitIndex a, OrbitIndex b, double v) {
  data_.host_span()[flat(n_, a, b)] = v;
}

void OneBody::set(OrbitIndex a, OrbitIndex b, double v) {
  auto h = data_.host_span();
  h[flat(n_, a, b)] = v;
  const std::int32_t s = hermiticity_sign(hermiticity_);
  if (a != b && s != 0) h[flat(n_, b, a)] = s * v;
}

void OneBody::add(OrbitIndex a, OrbitIndex b, double v) {
  auto h = data_.host_span();
  h[flat(n_, a, b)] += v;
  const std::int32_t s = hermiticity_sign(hermiticity_);
  if (a != b && s != 0) h[flat(n_, b, a)] += s * v;
}

void OneBody::set_zero() { data_.fill(0.0); }

linalg::MatrixView<double> OneBody::host_view() { return {data_.host_span().data(), n_, n_, n_}; }
linalg::MatrixView<const double> OneBody::host_view() const {
  return {data_.host_span().data(), n_, n_, n_};
}
linalg::MatrixView<double> OneBody::device_view() { return {data_.device_data(), n_, n_, n_}; }
linalg::MatrixView<const double> OneBody::device_view() const {
  return {data_.device_data(), n_, n_, n_};
}

}  // namespace imsrg::op
