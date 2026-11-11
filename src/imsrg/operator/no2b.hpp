#pragma once
/// @file
/// `No2bThreeBody`: the input container for normal-ordered-two-body (NO2B) three-nucleon matrix
/// elements in isospin coupling (design/00 D15, design/10 section 2: input 3N containers are data
/// types of `operator`; `io` fills them, normal ordering and HF consume them).
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "imsrg/base/types.hpp"
#include "imsrg/basis/orbit.hpp"

namespace imsrg::op {

/// Truncations of the NO2B storage, with the meaning of imsrg++ `ThreeBodySpaceNO2B(emax, E2max,
/// E3max, lmax)`: an isospin orbit is kept if e <= emax and l <= lmax; a state (a b c) needs
/// e_a + e_b, e_a + e_c and e_b + e_c <= e2max and e_a + e_b + e_c <= e3max. imsrg++ allocates
/// the storage with the MODEL-SPACE truncations (emax, E2max, E3max, min(emax, lmax)); the file
/// truncations only restrict what a reader fills in.
struct No2bTruncation {
  std::int32_t emax = 0;   ///< Largest single-particle shell e = 2n + l.
  std::int32_t e2max = 0;  ///< Largest pair energy of any two of the three orbits.
  std::int32_t e3max = 0;  ///< Largest three-body energy.
  std::int32_t lmax = 0;   ///< Largest orbital angular momentum.
  /// Field-wise equality.
  friend bool operator==(const No2bTruncation&, const No2bTruncation&) = default;
};

/// An isospin orbit |n l j> (proton and neutron together) of the NO2B storage.
struct IsospinOrbit {
  std::int32_t n = 0;  ///< Radial quantum number.
  std::int32_t l = 0;  ///< Orbital angular momentum.
  TwoJ j2{1};          ///< Twice the total angular momentum.
  /// Shell e = 2n + l.
  constexpr std::int32_t e() const noexcept { return 2 * n + l; }
  /// Field-wise equality.
  friend constexpr bool operator==(const IsospinOrbit&, const IsospinOrbit&) = default;
};
static_assert(std::is_trivially_copyable_v<IsospinOrbit>);

/// NO2B three-nucleon matrix elements in isospin coupling, stored as imsrg++
/// `ThreeBodyStorage_no2b<float>` stores them (ThreeBodyStorage_no2b.cc:19-160, 230-330), in
/// single precision (the precision of the input files; half-precision files convert exactly):
///
///   W^{J2 T3}_{(ab) Tab c, (de) Tde f} = sum_J (2J + 1) <(ab) J2 Tab, c; J T3 | V | (de) J2 Tde,
///   f; J T3>,
///
/// unnormalized, antisymmetrized, with the sum over the total three-body J already done by the
/// file producer; only j_c = j_f and l_c = l_f (mod 2) are stored, which is what normal ordering
/// and Hartree-Fock need.
///
/// Layout (identical to imsrg++):
/// * isospin orbits in the order e ascending, l ascending (l = e mod 2, ..., min(e, lmax), step
///   2), 2j ascending (2l - 1, then 2l + 1) — NOT the model-space order;
/// * channels (J2, P2, J1, P1, T3): J2 = 0..min(2 lmax + 1, e2max + 1), P2 in {0, 1}, J1 = 2j_c
///   = 1, 3, ..., 2 lmax + 1, P1 = l_c mod 2, T3 = 2T in {1, 3}, in this nesting order; channels
///   without states are not stored;
/// * the states of a channel: for a <= b (isospin orbit indices), c, then Tab in {0, 1}, in this
///   nesting order, with the truncations, triangle(j_a, j_b, 2 J2), (l_a + l_b) mod 2 = P2,
///   j_c = J1, l_c mod 2 = P1, triangle(2 Tab, 1, T3), and without a == b when J2 + Tab is even
///   (Pauli). The swapped state (b a c Tab) maps to the same index with the phase
///   (-1)^((j_a + j_b)/2 + J2 + Tab);
/// * per channel the symmetric matrix as a packed triangle: element (i, k) at
///   max(i,k) (max(i,k) + 1) / 2 + min(i,k). Hermitian storage is assumed (as imsrg++).
///
/// Host-only value type (used once, to build normal-ordered operators and the HF potential).
class No2bThreeBody {
 public:
  /// Empty container (no orbits, no channels).
  No2bThreeBody() = default;
  /// Zero-filled storage for the given truncations.
  /// @throws InputError if a truncation is negative.
  explicit No2bThreeBody(const No2bTruncation& truncation);

  /// The truncations given at construction.
  const No2bTruncation& truncation() const noexcept { return trunc_; }
  /// The isospin orbits in storage order.
  std::span<const IsospinOrbit> isospin_orbits() const noexcept { return orbits_; }
  /// Index of the isospin orbit (n, l, 2j), or nullopt if it is not stored.
  std::optional<std::int32_t> isospin_orbit_index(std::int32_t n, std::int32_t l,
                                                  TwoJ j2) const noexcept;
  /// Number of channels with at least one state.
  std::int32_t num_channels() const noexcept { return static_cast<std::int32_t>(channels_.size()); }
  /// Number of stored values (sum over channels of n (n + 1) / 2).
  std::size_t size() const noexcept { return values_.size(); }

  /// Sets W^{J2 T3}_{(ab) Tab c, (de) Tde f} = v (isospin orbit indices; twoT = 2T = 1 or 3),
  /// imsrg++ `SetME_iso_no2b`: the value is stored with the phases of both orderings, so the
  /// transposed element and the swapped orderings read back consistently. Readers must skip
  /// elements outside the truncations and Pauli-forbidden states first, as imsrg++ `ReadFile`
  /// does.
  /// @throws InvariantError if the element is not part of the storage (orbit index out of range,
  ///         j_c != j_f, parity mismatch, channel or state absent).
  void set_isospin(std::int32_t a, std::int32_t b, std::int32_t c, std::int32_t Tab, std::int32_t d,
                   std::int32_t e, std::int32_t f, std::int32_t Tde, std::int32_t J2,
                   std::int32_t twoT, double v);
  /// imsrg++ `GetME_iso_no2b`: the stored value times the ordering phases, or 0 if the element is
  /// not part of the storage (including out-of-range indices).
  double get_isospin(std::int32_t a, std::int32_t b, std::int32_t c, std::int32_t Tab,
                     std::int32_t d, std::int32_t e, std::int32_t f, std::int32_t Tde,
                     std::int32_t J2, std::int32_t twoT) const noexcept;
  /// imsrg++ `GetME_pn_no2b(a, b, c, d, e, f, J2)` (ThreeBodyStorage_no2b.cc:362-412): the NO2B
  /// element for proton/neutron orbits (|ab; J2> c and |de; J2> f, 2tz = -1 proton), from the
  /// isospin values with the isospin Clebsch-Gordan coefficients
  /// CG(1/2 tz_a 1/2 tz_b | Tab) CG(1/2 tz_d 1/2 tz_e | Tde) CG(Tab, 1/2 tz_c | T) CG(Tde, 1/2 tz_f
  /// | T) summed over Tab, Tde in {0, 1} and 2T = 1, 3 up to min(2 Tab + 1, 2 Tde + 1). Returns 0
  /// if l_c and l_f differ in parity, 2j_c != 2j_f, the pair parities differ, the total Tz differs,
  /// or an orbit is not stored.
  double get_pn(const basis::Orbit& a, const basis::Orbit& b, const basis::Orbit& c,
                const basis::Orbit& d, const basis::Orbit& e, const basis::Orbit& f,
                std::int32_t J2) const noexcept;
  /// Frobenius norm of the stored values (imsrg++ no2b `Norm`).
  double norm() const noexcept;

 private:
  struct Channel {
    std::int32_t J2 = 0, P2 = 0, J1 = 0, P1 = 0, T3 = 0;
    std::int32_t num_states = 0;
    std::size_t offset = 0;  // into values_
    // key(a, b, c, Tab) -> state index (both orderings of a, b) and ordering phase
    std::unordered_map<std::uint64_t, std::pair<std::int32_t, std::int32_t>> states;
  };
  std::optional<std::int32_t> channel_index(std::int32_t J2, std::int32_t P2, std::int32_t J1,
                                            std::int32_t P1, std::int32_t T3) const noexcept;

  No2bTruncation trunc_;
  std::vector<IsospinOrbit> orbits_;
  std::vector<Channel> channels_;
  std::vector<std::int32_t> channel_lookup_;  // dense (J2, P2, J1, P1, T3) -> channel or -1
  std::vector<float> values_;
};

}  // namespace imsrg::op
