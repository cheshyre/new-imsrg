#pragma once
/// @file
/// Internal: the two- and three-body monopole interactions that turn a density matrix into the
/// mean-field one-body potentials of Hartree-Fock (imsrg++ `HartreeFock::BuildMonopoleV`,
/// `BuildMonopoleV3`, `UpdateF`). Not part of the public API of the hf module.
///
/// All matrices are column-major `linalg::Matrix<double>` of size num_orbits x num_orbits, indexed
/// by imsrg++ orbit index. Host only.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "imsrg/basis/single_particle_basis.hpp"
#include "imsrg/linalg/matrix_view.hpp"
#include "imsrg/operator/no2b.hpp"
#include "imsrg/operator/operator.hpp"

namespace imsrg::hf::detail {

/// The one-body channels (l, 2j, tz) of a basis: each entry lists the orbit indices of one
/// channel in ascending order; every orbit is in exactly one channel. Channels are ordered by
/// their first orbit.
std::vector<std::vector<std::int32_t>> one_body_channels(const basis::SingleParticleBasis& basis);

/// Two-body monopole: for every (Tz, parity) block of pairs p <= q the dense matrices
/// Vmon(ab, cd) = sum_J (2J+1) <ab|V|cd>_J and Vmon_exch(ab, cd) = sum_J (2J+1) <ab|V|dc>_J
/// (unnormalized matrix elements, J over the range allowed by the bra), imsrg++ ket order.
class Monopole2B {
 public:
  /// Builds the monopole blocks from the two-body part of `H`.
  /// Precondition: the host mirror of `H` is current (`H.sync_to_host()`).
  /// Complexity O(K^2 J) with K the number of pairs (memory O(K^2)).
  explicit Monopole2B(const op::Operator& H);

  /// Mean-field potential (imsrg++ `UpdateF`, 2-body part):
  ///   Vij(i,j) = 1/(2j_i+1) sum_{ab} rho(a,b) Vmon[(i,a),(j,b)]
  /// for i, j in the same (l, 2j, tz) channel, symmetrized from the upper triangle (j >= i);
  /// entries outside the channels are zero. Units: MeV. Complexity O(N c^2 N') with N orbits and
  /// c the channel size. `Vij` is resized to N x N and overwritten.
  void apply(const linalg::Matrix<double>& rho, linalg::Matrix<double>& Vij) const;

 private:
  std::int32_t norb_ = 0;
  std::vector<std::int32_t> blk_;    // pair (p <= q) -> block, p * norb + q
  std::vector<std::int32_t> local_;  // pair (p <= q) -> local ket index in its block
  std::array<linalg::Matrix<double>, 6> vmon_;
  std::array<linalg::Matrix<double>, 6> vmon_exch_;
  std::vector<std::int32_t> two_j_;
  std::vector<std::vector<std::int32_t>> channel_of_;  // per orbit: the orbits of its channel
};

/// Three-body monopole in the NO2B approximation: stored keys (i, j, a, b, c, d) with the values
/// 1/(2j_i+1) sum_J <(a c) J, i | V3 | (b d) J, j> (imsrg++ `SetUpMonopoleV3Keys` +
/// `BuildMonopoleV3`).
class Monopole3B {
 public:
  /// Enumerates the keys with the e3max cut and evaluates the values from `v3`; zero values are
  /// not stored. Precondition: `H` host mirror irrelevant (only its basis is used).
  /// Complexity O(N^2 c^4 J) with N orbits and c the channel size.
  Monopole3B(const op::Operator& H, const op::No2bThreeBody& v3, std::int32_t e3max);

  /// Three-body mean-field potential (imsrg++ `UpdateF`, 3-body part):
  ///   V3ij(i,j) = sum_keys rho(a,b) rho(c,d) value, symmetrized from the upper triangle.
  /// `V3ij` is resized to N x N and overwritten. Units: MeV. Complexity O(#keys).
  void apply(const linalg::Matrix<double>& rho, linalg::Matrix<double>& V3ij) const;

  /// Number of stored keys.
  std::size_t num_keys() const noexcept { return values_.size(); }

 private:
  struct Key {
    std::uint16_t i, j, a, b, c, d;
  };
  std::int32_t norb_ = 0;
  std::vector<Key> keys_;
  std::vector<double> values_;
};

}  // namespace imsrg::hf::detail
