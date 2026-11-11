#pragma once
/// @file
/// `OneBody`: the dense one-body part O_ab of an operator (design/10 section 3.3).
#include <cstdint>

#include "imsrg/base/types.hpp"
#include "imsrg/exec/device_buffer.hpp"
#include "imsrg/linalg/matrix_view.hpp"

namespace imsrg::op {

/// Dense `norb x norb` one-body matrix O_ab = <a|O|b> (orbit indices of the model space, imsrg++
/// order), column-major (element (a, b) at `a + b * norb`, like imsrg++'s Armadillo `OneBody`),
/// in one `exec::DeviceBuffer<double>`. norb is at most a few hundred, so the dense storage is
/// tiny. A scalar operator only connects orbits with equal (l, 2j, tz2); the storage does not
/// enforce this, but the norms and the normal ordering only read those entries (as imsrg++).
///
/// Memory model (design/12 section 2): the element accessors read and write the HOST MIRROR.
/// On backends with separate device memory, call `sync_to_device()` after host writes and
/// before kernels read the part, and `sync_to_host()` after kernels wrote it and before host
/// reads. On Serial/OpenMP both are fences only.
///
/// Value semantics: copies are deep, moves are cheap. Needs an initialized `exec::Runtime` unless
/// empty. The hermiticity flag is used by `set` and `add` only; it is kept equal to the owning
/// operator's by `Operator`.
class OneBody {
 public:
  /// Empty part (0 orbits).
  OneBody() = default;
  /// Zero matrix for `num_orbits` orbits with the given hermiticity flag.
  /// Precondition: num_orbits >= 0 (IMSRG_CHECK).
  OneBody(std::int32_t num_orbits, Hermiticity hermiticity);

  /// Number of orbits (the matrix dimension).
  std::int32_t num_orbits() const noexcept { return n_; }
  /// The hermiticity used by `set` and `add`.
  Hermiticity hermiticity() const noexcept { return hermiticity_; }
  /// Changes the hermiticity flag; the data are not touched.
  void set_hermiticity(Hermiticity h) noexcept { hermiticity_ = h; }

  /// O_ab from the host mirror. Precondition: valid orbit indices (IMSRG_ASSERT). O(1).
  double get(OrbitIndex a, OrbitIndex b) const;
  /// Sets O_ab = v and, for a != b, its partner O_ba = h v with h = hermiticity_sign (not for
  /// non-Hermitian operators). For a == b the element is written once (an anti-Hermitian part
  /// must have a zero diagonal; this is not checked). Host mirror. O(1).
  void set(OrbitIndex a, OrbitIndex b, double v);
  /// Adds v to O_ab and, for a != b, h v to O_ba (as `set`). Host mirror. O(1).
  void add(OrbitIndex a, OrbitIndex b, double v);
  /// Sets O_ab = v only (no partner). Host mirror. O(1).
  void set_element(OrbitIndex a, OrbitIndex b, double v);
  /// Sets every element to zero (device kernel).
  void set_zero();

  /// Column-major view of the host mirror (ld = norb). See the memory model above.
  linalg::MatrixView<double> host_view();
  /// Read-only view of the host mirror.
  linalg::MatrixView<const double> host_view() const;
  /// View of the device data, for kernels (and BLAS on host-accessible backends).
  linalg::MatrixView<double> device_view();
  /// Read-only view of the device data.
  linalg::MatrixView<const double> device_view() const;
  /// The underlying buffer (norb * norb elements, column-major).
  exec::DeviceBuffer<double>& buffer() noexcept { return data_; }
  /// The underlying buffer, read-only.
  const exec::DeviceBuffer<double>& buffer() const noexcept { return data_; }
  /// Copies the device data to the host mirror (see the memory model).
  void sync_to_host() { data_.sync_to_host(); }
  /// Copies the host mirror to the device data (see the memory model).
  void sync_to_device() { data_.sync_to_device(); }

 private:
  std::int32_t n_ = 0;
  Hermiticity hermiticity_ = Hermiticity::Hermitian;
  exec::DeviceBuffer<double> data_;
};

}  // namespace imsrg::op
